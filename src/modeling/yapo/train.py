# 0. imports
import os
import random
import sys
from pathlib import Path

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, TrainingArguments
from transformers.modeling_outputs import ModelOutput

_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from helper.chat import ensure_tokenizer_has_chat_template
from sae import load_sae
from trl import BiPOTrainer, DPOConfig
from trl.trainer.callbacks import SaveVectorCallback

LLAMA_RIGHT_PAD_TOKEN = "<|finetune_right_pad_id|>"


def _requires_llama_right_pad(model_id: str) -> bool:
    name = (model_id or "").lower()
    return "llama" in name


def _supports_bf16() -> bool:
    try:
        return torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    except AttributeError:
        return False

# System prompt injected when formatting chat messages.
# Overridden at runtime via --system_prompt CLI argument.
# SYSTEM_PROMPT = ""


def _slugify(text: str) -> str:
    """Create a filesystem/log friendly slug from a string."""
    return (
        str(text)
        .strip()
        .replace("/", "-")
        .replace(" ", "-")
        .replace("::", "-")
        .lower()
    )


def build_wandb_run_name(args: "ScriptArguments") -> str:
    """Construct a descriptive W&B run name from model + hyperparameters.

    Example: bipo-mistral-7b-instruct-v0.2_beh-power-seeking_task-mcq_L13_beta0.1_lr5e-04_bs4_warm100_wd0.05
    """
    # prefer the leaf of the repo path if present
    model_leaf = args.model_name_or_path.split("/")[-1]
    model_slug = _slugify(model_leaf)

    # Infer task type (MCQ vs open-ended) from explicit flag or heuristics
    def _infer_is_mcq(a) -> bool:
        try:
            mcq_flag = getattr(a, "mcq_training", None)
            if mcq_flag is True:
                return True
        except Exception:
            pass
        s1 = str(getattr(a, "hub_dataset_path", "")).lower()
        s2 = str(getattr(a, "prompt_colname", "")).lower()
        s3 = str(getattr(a, "chosen_colname", "")).lower()
        return ("mcq" in s1) or ("mcq" in s2) or ("mcq" in s3)

    task_tag = "task-mcq" if _infer_is_mcq(args) else "task-oe"

    # Warmup tag: prefer steps if provided, otherwise ratio, otherwise 0
    if getattr(args, "warmup_steps", None) is not None:
        warm_tag = f"warmS{args.warmup_steps}"
    elif getattr(args, "warmup_ratio", None) is not None:
        warm_tag = f"warmR{args.warmup_ratio}"
    else:
        warm_tag = "warmR0.0"

    parts = [
        # "bipo",
        model_slug,
        f"beh-{_slugify(args.behavior)}",
        task_tag,
        f"loc-{_slugify(getattr(args, 'localization_status', 'both'))}",
        f"L{args.layer}",
        # f"beta{args.beta}",
        f"lr{args.learning_rate}",
        f"bs{args.per_device_train_batch_size}",
        # warm_tag,
        # f"wd{args.weight_decay}",
    ]
    return "_".join(parts)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def _resolve_model_output_hidden_key(output: ModelOutput) -> str:
    """
    Identify which field in a ModelOutput carries the hidden states so we can override it.
    """
    for key in ("last_hidden_state", "hidden_states", "hidden_state"):
        value = output.get(key, None)
        if value is not None:
            return key
    for key, value in output.items():
        if isinstance(value, torch.Tensor):
            return key
    raise ValueError("Unable to locate hidden states tensor in ModelOutput.")


def _get_hidden_states(block_output):
    """
    Extract the hidden-state tensor from either a tensor, tuple/list of tensors, or ModelOutput.
    """
    if isinstance(block_output, torch.Tensor):
        return block_output
    if isinstance(block_output, ModelOutput):
        key = _resolve_model_output_hidden_key(block_output)
        return block_output[key]
    if isinstance(block_output, (tuple, list)):
        return block_output[0]
    raise TypeError(f"Unsupported block output type: {type(block_output)}")


def _rebuild_block_output(original_output, new_hidden):
    """
    Reconstruct the module output after replacing the hidden states with `new_hidden`.
    """
    if isinstance(original_output, torch.Tensor):
        return new_hidden
    if isinstance(original_output, ModelOutput):
        key = _resolve_model_output_hidden_key(original_output)
        data = {k: v for k, v in original_output.items()}
        data[key] = new_hidden
        return original_output.__class__(**data)
    if isinstance(original_output, tuple):
        return (new_hidden,) + original_output[1:]
    if isinstance(original_output, list):
        new_sequence = list(original_output)
        new_sequence[0] = new_hidden
        return new_sequence
    raise TypeError(f"Unsupported block output type: {type(original_output)}")


class BlockWrapper(torch.nn.Module):
    def __init__(self, block, vec=None, vector_size: Optional[int] = None):
        super().__init__()
        self.multiplier = 1.0
        self.block = block
        if vec is not None:
            self.vec = torch.nn.Parameter(vec)
        else:
            # Zero Init using provided hidden size (fallback keeps previous default)
            hidden_size = vector_size if vector_size is not None else 4096
            # Match dtype/device to underlying block to avoid upcasting activations
            try:
                ref_param = next(p for p in self.block.parameters())
                param_dtype = ref_param.dtype
                param_device = ref_param.device
            except StopIteration:
                param_dtype = torch.get_default_dtype()
                param_device = torch.device("cpu")
            self.vec = torch.nn.Parameter(torch.zeros(hidden_size, dtype=param_dtype, device=param_device))

    def forward(self, *args, **kwargs):
        # Forward through the frozen block. We detach the block's hidden output
        # so gradients do not flow into lower layers, which dramatically reduces
        # activation memory when only training the steering vector.
        block_output = self.block(*args, **kwargs)
        hidden = _get_hidden_states(block_output).detach()
        steered = hidden + (self.multiplier * self.vec)
        return _rebuild_block_output(block_output, steered)

    def set_multiplier(self, multiplier):
        self.multiplier  = multiplier
        
    def __getattr__(self, name):
        # Try the standard torch.nn.Module lookup first to preserve parameters/submodules.
        try:
            return super().__getattr__(name)
        except AttributeError:
            # Delegate missing attributes (e.g., attention_type) to the wrapped block.
            return getattr(self.block, name)
        
class SparseBlockWrapper(torch.nn.Module):
    def __init__(self, block, sae, vec=None, vector_size=65536):
        super().__init__()
        self.multiplier = 1.0
        self.block = block
        self.sae = sae
        self.vector_size = vector_size
        
        # Ensure SAE is in eval mode and frozen
        self.sae.eval()
        for param in self.sae.parameters():
            param.requires_grad = False
            
        if vec is not None:
            self.vec = torch.nn.Parameter(vec)
        else:
            # Zero Init - steering vector in sparse space (match dtype/device to block)
            try:
                ref_param = next(p for p in self.block.parameters())
                param_dtype = ref_param.dtype
                param_device = ref_param.device
            except StopIteration:
                param_dtype = torch.get_default_dtype()
                param_device = torch.device("cpu")
            self.vec = torch.nn.Parameter(torch.zeros(self.vector_size, dtype=param_dtype, device=param_device))

    def _encode_sparse(self, hidden_flat: torch.Tensor) -> torch.Tensor:
        encoded = self.sae.encode(hidden_flat)
        if isinstance(encoded, tuple):
            encoded = encoded[0]
        return encoded

    def forward(self, *args, **kwargs):
        # Get the original hidden activations from the transformer layer
        h_l = self.block(*args, **kwargs)
        
        # Extract the hidden state (first element of the tuple) and detach to
        # stop gradients flowing into lower layers (they are frozen).
        hidden_states = _get_hidden_states(h_l).detach()  # Shape: [batch_size, seq_len, hidden_dim]
        
        # Encode hidden states to sparse space using SAE
        # Note: SAE expects 2D input [batch_size * seq_len, hidden_dim]
        original_shape = hidden_states.shape
        # print(f'Original hidden states shape: {original_shape}') # torch.Size([1, 1, 2304])
        hidden_flat = hidden_states.view(-1, hidden_states.size(-1))
        
        # Get sparse representation
        with torch.no_grad():
            sparse_activations = self._encode_sparse(hidden_flat)  # [batch_size * seq_len, sparse_dim]
        # print(f'Sparse activations shape: {sparse_activations.shape}') # torch.Size([1, 65536])
        
        # Add steering vector in sparse space (scaled by multiplier)
        steered_sparse = sparse_activations.detach() + (self.multiplier * self.vec.unsqueeze(0))
        
        # Apply activation function to maintain non-negativity in sparse space
        # The paper uses activation functions to ensure non-negative sparse activations
        steered_sparse = torch.relu(steered_sparse)
        
        # Decode back to dense space
        steered_hidden_flat = self.sae.decode(steered_sparse)
        # print(f'Steered hidden flat shape: {steered_hidden_flat.shape}') # torch.Size([1, 2304])
        
        # Compute error correction term (delta correction)
        # This compensates for SAE reconstruction loss as described in the paper
        with torch.no_grad():
            reconstructed_flat = self.sae.decode(sparse_activations)  # Original reconstruction
        delta = hidden_flat - reconstructed_flat  # Error correction term
        
        # Add error correction to maintain original information
        final_hidden_flat = steered_hidden_flat + delta
        # print(f'Final hidden flat shape: {final_hidden_flat.shape}') # torch.Size([1, 2304])
        
        # Reshape back to original dimensions
        final_hidden = final_hidden_flat.view(original_shape)
        # print(f'Final hidden shape: {final_hidden.shape}') # torch.Size([1, 1, 2304])
        
        # Return the modified output maintaining the original structure
        return _rebuild_block_output(h_l, final_hidden)

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.block, name)

# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    # data parameters
    beta: Optional[float] = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})
    use_hub_data: Optional[bool] = field(
        default=True,
        metadata={"help": "If true, load dataset from HF Hub. If no train/test split exists, create an 80/20 split deterministically."},
    )
    hub_dataset_path: Optional[str] = field(
        default="BounharAbdelaziz/Arabic_cultural_dataset_processed",
        metadata={"help": "HF Hub dataset path to load when use_hub_data=True."},
    )
    country_name: Optional[str] = field(
        default="egypt",
        metadata={"help": "Optional subset/config name for HF Hub dataset."},
    )
    localization_status: Optional[str] = field(
        default="both",
        metadata={
            "help": (
                "Filter by 'localization_status' column. Use 'localized' or 'nolocalized' to filter, "
                "or 'both' to disable filtering and train on all."
            )
        },
    )
    split_seed: Optional[int] = field(
        default=42,
        metadata={"help": "Random seed for fallback 80/20 split if hub dataset lacks splits."},
    )
    train_ratio: Optional[float] = field(
        default=0.8,
        metadata={"help": "Train fraction for fallback split when hub dataset lacks splits."},
    )

    # dataset schema and prompting controls
    system_prompt: Optional[str] = field(
        default="",
        metadata={
            "help": (
                "System prompt injected before the user message when building chat prompts. "
                "Pass from training bash scripts with --system_prompt."
            )
        },
    )
    prompt_colname: Optional[str] = field(
        default="prompt",
        metadata={
            "help": (
                "Column name containing the user prompt/question in the dataset. "
                "Defaults to 'prompt'; falls back to 'question' if missing."
            )
        },
    )
    chosen_colname: Optional[str] = field(
        default="chosen_open_ended",
        metadata={
            "help": (
                "Column name containing the chosen/ground-truth answer. "
                "Defaults to 'chosen_open_ended' then 'chosen' if not found."
            )
        },
    )
    rejected_colname: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional column name for the rejected answer. If not set, will try model-specific "
                "columns (e.g. rejected_gemma_2_9b_it_open_ended) then fall back to 'rejected'."
            )
        },
    )
    mcq_training: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "If True, mark task as MCQ for naming/logging. If unset, heuristic inference is used from dataset/columns."
            )
        },
    )

    # training parameters
    model_name_or_path: Optional[str] = field(
        default="meta-llama/Llama-2-7b-chat-hf",
        metadata={
            "help": (
                "Supported: meta-llama/Llama-2-7b-chat-hf, "
                "mistralai/Mistral-7B-Instruct-v0.2, google/gemma-2-2b"
            )
        },
    )
    learning_rate: Optional[float] = field(default=5e-4, metadata={"help": "optimizer learning rate"})
    lr_scheduler_type: Optional[str] = field(default="cosine", metadata={"help": "the lr scheduler type"})
    # Warmup: use either steps or ratio; leave both None for no warmup
    warmup_ratio: Optional[float] = field(default=None, metadata={"help": "warmup ratio of total steps (used when warmup_steps is None)"})
    warmup_steps: Optional[int] = field(default=None, metadata={"help": "number of warmup steps (overrides warmup_ratio when set)"})
    weight_decay: Optional[float] = field(default=0.05, metadata={"help": "the weight decay"})
    optimizer_type: Optional[str] = field(default="adamw_torch", metadata={"help": "the optimizer type"})

    per_device_train_batch_size: Optional[int] = field(default=4, metadata={"help": "train batch size per device"})
    per_device_eval_batch_size: Optional[int] = field(default=1, metadata={"help": "eval batch size per device"})
    # evaluation control
    do_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run evaluation. Defaults to False (no eval)."},
    )
    eval_steps: Optional[int] = field(
        default=0,
        metadata={
            "help": "(Ignored) Evaluation runs per epoch when --do_eval is set."
        },
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "whether to use gradient checkpointing (recommended to reduce activation memory)"}
    )

    max_prompt_length: Optional[int] = field(default=2048, metadata={"help": "the maximum prompt length"})
    max_length: Optional[int] = field(default=2048, metadata={"help": "the maximum sequence length"})
    num_train_epochs: Optional[int] = field(default=100, metadata={"help": "the number of training epochs"})
    logging_steps: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    log_freq: Optional[int] = field(default=1, metadata={"help": "the logging frequency"})

    behavior: Optional[str] = field(default="power-seeking", metadata={"help": "the behavior"})
    layer: Optional[int] = field(default=15, metadata={"help": "the layer the steering vector extracted from"})

    # instrumentation
    report_to: Optional[str] = field(
        default="none",
        metadata={
            "help": 'The list of integrations to report the results and logs to. Supported platforms are `"azure_ml"`,'
            '`"comet_ml"`, `"mlflow"`, `"neptune"`, `"tensorboard"`,`"clearml"` and `"wandb"`. '
            'Use `"all"` to report to all integrations installed, `"none"` for no integrations.'
        },
    )

    # wandb options
    wandb_project: Optional[str] = field(default="Alignement", metadata={"help": "Weights & Biases project name"})
    wandb_run_name: Optional[str] = field(default=None, metadata={"help": "Weights & Biases run name"})
    wandb_entity: Optional[str] = field(default=None, metadata={"help": "Weights & Biases entity (team/org)"})
    wandb_mode: Optional[str] = field(
        default="online",
        metadata={"help": "Set to 'offline' to disable network logging, or 'online' to enable."},
    )

    # eval generation preview (logs to W&B when enabled)
    generate_during_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Sample and log generations during evaluation (requires wandb)."},
    )
    
    # sparse steering
    sparse_steering: Optional[bool] = field(
        default=False,
        metadata={"help": "Use sparse steering with SAE."},
    )
    # SAE configuration (used when sparse_steering=True)
    sae_repo: Optional[str] = field(
        default="google/gemma-scope-2b-pt-res",
        metadata={
            "help": (
                "SAE model repo/path to load. Used directly for Gemma-Scope; for Llama-Scope this "
                "value can override the default repo resolved from site/expansion."
            )
        },
    )
    sae_vector_size: Optional[str] = field(
        default="65k",
        metadata={"help": "SAE vector size identifier used in filenames (e.g. 65k, 131k)."},
    )
    sae_avg_idx: Optional[str] = field(
        default="68",
        metadata={"help": "Average L0 index identifier used in filenames (e.g. 68)."},
    )
    sae_source: Optional[str] = field(
        default="gemma_scope",
        metadata={
            "help": "SAE family to use. Supported values: 'gemma_scope' (default) and 'llama_scope'. "
            "When set to 'llama_scope', see llama_sae_* arguments."
        },
    )
    llama_sae_site: Optional[str] = field(
        default="R",
        metadata={"help": "Llama-Scope site identifier (R, A, M, or TC)."},
    )
    llama_sae_expansion: Optional[int] = field(
        default=8,
        metadata={"help": "Llama-Scope expansion factor (8 or 32)."},
    )
    llama_sae_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Optional cache directory override for downloaded Llama-Scope SAEs."},
    )

    # debug argument for distributed training
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    
    # debugging helpers
    debug_subset_size: Optional[int] = field(
        default=0,
        metadata={
            "help": "If > 0, use only this many samples from train and eval datasets for quick debugging.",
        },
    )
    
def get_data(
    num_proc=1,
    behavior='power-seeking',
    train=True,
    model_name='meta-llama/Llama-2-7b-chat-hf',
    template_name='llama',
    subset_size=0,
    tokenizer=None,
    use_hub_data=True,
    dataset_path="Raniahossam33/Arabic_cultural_dataset",
    country_name=None,
    localization_status: Optional[str] = "both",
    split_seed: int = 42,
    train_ratio: float = 0.8,
    prompt_colname: Optional[str] = "prompt",
    chosen_colname: Optional[str] = "chosen_open_ended",
    rejected_colname: Optional[str] = None,
    SYSTEM_PROMPT: str = None,
):
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided.")

    # Load dataset either from local CSVs or from HF Hub
    if use_hub_data:
        # Try loading the dataset from hub (with optional subset)
        try:
            raw_ds = load_dataset(dataset_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load hub dataset '{dataset_path}': {e}")

        # Prefer using existing train/test (or validation) splits if available
        if isinstance(raw_ds, dict) or hasattr(raw_ds, "keys"):
            ds_keys = set(raw_ds.keys())
            if "train" in ds_keys and ("test" in ds_keys or "validation" in ds_keys):
                dataset = raw_ds["train"] if train else raw_ds.get("test", raw_ds["validation"])
            elif "train" in ds_keys:
                # Fallback: only train split exists; create a deterministic split
                base_ds = raw_ds["train"]
                test_size = 1.0 - float(train_ratio)
                split = base_ds.train_test_split(test_size=test_size, seed=int(split_seed))
                dataset = split["train"] if train else split["test"]
            else:
                # Fallback: take the first available split and split it
                first_split_name = next(iter(raw_ds.keys()))
                base_ds = raw_ds[first_split_name]
                test_size = 1.0 - float(train_ratio)
                split = base_ds.train_test_split(test_size=test_size, seed=int(split_seed))
                dataset = split["train"] if train else split["test"]
        else:
            # raw_ds is already a Dataset object; fall back to a split
            base_ds = raw_ds
            test_size = 1.0 - float(train_ratio)
            split = base_ds.train_test_split(test_size=test_size, seed=int(split_seed))
            dataset = split["train"] if train else split["test"]

        # Optional: filter by country if provided (skip if None)
        if country_name is not None:
            dataset = dataset.filter(
                lambda sample: sample.get("country") == country_name,
                desc=f"Filtering data to keep samples for country={country_name}"
            )

        # Optional: filter by localization_status when explicitly requested
        # Accept values: 'localized', 'nolocalized'; 'both' or None disables filtering
        _loc = (localization_status or "both").strip().lower()
        if _loc in {"localized", "nolocalized"}:
            if "localization_status" not in dataset.column_names:
                print("[WARN] 'localization_status' column not found; skipping localization filter.")
            else:
                dataset = dataset.filter(
                    lambda sample: sample.get("localization_status") == _loc,
                    desc=f"Filtering data to keep samples with localization_status={_loc}",
                )
    else:
        # Local CSVs fallback
        if train:
            dataset = load_dataset("csv", data_files=f"./data/{behavior}/train.csv", split='train')
        else:
            dataset = load_dataset("csv", data_files=f"./data/{behavior}/test.csv", split='train')

    original_columns = dataset.column_names
    
    def return_prompt_and_responses(samples) -> Dict[str, str]:
        """Format chat prompts and align chosen/rejected lists per batch.

        Note: When batched=True, each returned column must have length equal
        to the batch size. Previously we returned inside the per-example loop,
        which produced mismatched lengths like len(prompt)=1 vs len(chosen)=N.
        """
        prompt = []
        # Resolve prompt key per batch for robustness
        _prompt_key = None
        if prompt_colname and prompt_colname in samples:
            _prompt_key = prompt_colname
        elif "prompt" in samples:
            _prompt_key = "prompt"
        elif "question" in samples:
            _prompt_key = "question"
        else:
            raise ValueError(
                f"Prompt column not found. Tried '{prompt_colname}', 'prompt', and 'question'. Available: {list(samples.keys())}"
            )

        for question in samples[_prompt_key]:
            # Check if the tokenizer supports system messages
            try:
                # First try with system message
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question}
                ]
                formatted = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as e:
                if "system role not supported" in str(e).lower() or "system" in str(e).lower():
                    # Fallback: combine system prompt with user message
                    if SYSTEM_PROMPT == "":
                        combined_message = f"{question}"
                    else:
                        combined_message = f"{SYSTEM_PROMPT}\n\n{question}"
                    messages = [{"role": "user", "content": combined_message}]
                    formatted = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                else:
                    # Re-raise if it's a different error
                    raise e

            prompt.append(formatted)

        # Build outputs after processing the whole batch to keep lengths aligned
        if "matching" in samples and "not_matching" in samples:
            return {
                "prompt": prompt,
                "chosen": [' ' + s for s in samples["matching"]],
                "rejected": [' ' + s for s in samples["not_matching"]],
            }
        else:
            # Determine chosen/rejected keys using overrides or sensible fallbacks
            _chosen_key = None
            if chosen_colname and chosen_colname in samples:
                _chosen_key = chosen_colname
            elif "chosen_open_ended" in samples:
                _chosen_key = "chosen_open_ended"
            elif "chosen" in samples:
                _chosen_key = "chosen"
            
            _rejected_key = None
            # If user provided a rejected column name and it exists, use it
            if rejected_colname and rejected_colname in samples:
                _rejected_key = rejected_colname
            else:
                fallback_rejected_keys = [
                    "rejected_gemma_2_2b_it_open_ended",
                    "rejected_gemma_2_9b_it_open_ended",
                    "rejected_gemma_2_2b_it_mcq",
                    "rejected_gemma_2_9b_it_mcq",
                    "rejected",
                ]
                for key in fallback_rejected_keys:
                    if key in samples:
                        _rejected_key = key
                        break
                

            if _chosen_key is None or _rejected_key is None:
                raise ValueError(
                    f"Dataset must contain either matching/not_matching or chosen+rejected columns. "
                    f"Resolved chosen='{_chosen_key}', rejected='{_rejected_key}'. Available keys: {list(samples.keys())}"
                )
            print(f'_chosen_key: {_chosen_key}')
            print(f'_rejected_key: {_rejected_key}')
            
            sample = {
                "prompt": prompt,
                "chosen": [' ' + s for s in samples[_chosen_key]],
                "rejected": [' ' + s for s in samples[_rejected_key]],
            }
            
            # print(f'sample: {sample}')
            print('-'*50)
            
            return sample
    
    # Optionally reduce dataset size for debugging after split/loading
    if isinstance(subset_size, int) and subset_size > 0:
        dataset = dataset.select(range(min(subset_size, len(dataset))))

    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

# Alternative: Model-specific handling
def get_data_model_specific(num_proc=1, behavior='power-seeking', train=True, tokenizer=None, model_name=None):
    if tokenizer is None:
        raise ValueError("Tokenizer must be provided.")
        
    if train:
        dataset = load_dataset("csv", data_files=f"./data/{behavior}/train.csv", split='train')
    else:
        dataset = load_dataset("csv", data_files=f"./data/{behavior}/test.csv", split='train')
    
    original_columns = dataset.column_names
    
    def return_prompt_and_responses(samples) -> Dict[str, str]:
        prompt = []
        for question in samples["question"]:
            # Model-specific handling
            if "gemma" in model_name.lower():
                # Gemma models often don't support system role
                combined_message = f"{SYSTEM_PROMPT}\n\n{question}"
                messages = [{"role": "user", "content": combined_message}]
            elif "llama-2" in model_name.lower() or "llama" in model_name.lower():
                # Llama models typically support system role
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question}
                ]
            elif "mistral" in model_name.lower():
                # Mistral models typically support system role
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question}
                ]
            else:
                # Default fallback: try system first, then fallback to user-only
                try:
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": question}
                    ]
                    # Test if this works
                    tokenizer.apply_chat_template(messages[:1], tokenize=False, add_generation_prompt=False)
                except:
                    combined_message = f"{SYSTEM_PROMPT}\n\n{question}"
                    messages = [{"role": "user", "content": combined_message}]
            
            formatted = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            prompt.append(formatted)
            
        # Build chosen/rejected depending on dataset schema
        keys = ["A", "B", "C", "D", "E"]
        if "answer_choice" in samples and all(k in samples for k in keys):
            chosen, rejected = [], []
            n = len(samples["question"]) if "question" in samples else len(samples["answer_choice"]) 
            for i in range(n):
                correct_raw = samples["answer_choice"][i]
                # Normalize correct key: allow letter (A-E) or 0-4 index
                if isinstance(correct_raw, str):
                    correct_key = correct_raw.strip()
                    if correct_key.isdigit():
                        idx = int(correct_key)
                        correct_key = keys[idx] if 0 <= idx < len(keys) else keys[0]
                elif isinstance(correct_raw, (int, np.integer)):
                    idx = int(correct_raw)
                    correct_key = keys[idx] if 0 <= idx < len(keys) else keys[0]
                else:
                    correct_key = str(correct_raw).strip()
                # Fallback safety
                if correct_key not in keys:
                    correct_key = keys[0]

                chosen_val = samples[correct_key][i]
                # Deterministically pick first incorrect non-empty option as rejected
                rej_key = next((k for k in keys if k != correct_key and str(samples[k][i]).strip() != ""), keys[1 if correct_key == keys[0] else 0])
                rejected_val = samples[rej_key][i]
                chosen.append(' ' + str(chosen_val))
                rejected.append(' ' + str(rejected_val))
            return {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        else:
            # Backward compatibility with matching/not_matching schema
            return {
                "prompt": prompt,
                "chosen": [' ' + s for s in samples.get("matching", [])],
                "rejected": [' ' + s for s in samples.get("not_matching", [])],
            }
    
    return dataset.map(
        return_prompt_and_responses,
        batched=True,
        num_proc=num_proc,
        remove_columns=original_columns,
    )

# Alternative: Check tokenizer capabilities
def check_system_support(tokenizer):
    """Check if tokenizer supports system messages"""
    try:
        test_messages = [
            {"role": "system", "content": "Test"},
            {"role": "user", "content": "Hello"}
        ]
        tokenizer.apply_chat_template(test_messages, tokenize=False, add_generation_prompt=False)
        return True
    except Exception as e:
        if "system" in str(e).lower():
            return False
        raise e
    
if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # Configure Weights & Biases logging if requested
    use_wandb = bool(script_args.report_to and "wandb" in script_args.report_to)
    # Only initialize/log from the main process when running distributed
    rank_env = os.environ.get("RANK") or os.environ.get("ACCELERATE_PROCESS_INDEX") or "0"
    is_main_process = str(rank_env) == "0"
    # Build a stable run name early for both HF args and W&B
    if not script_args.wandb_run_name:
        try:
            script_args.wandb_run_name = build_wandb_run_name(script_args)
        except Exception:
            # Fallback to a simple name if anything goes wrong
            script_args.wandb_run_name = _slugify(script_args.model_name_or_path)

    if use_wandb and is_main_process:
        if script_args.wandb_project:
            os.environ["WANDB_PROJECT"] = script_args.wandb_project
        if script_args.wandb_entity:
            os.environ["WANDB_ENTITY"] = script_args.wandb_entity
        if script_args.wandb_run_name:
            os.environ["WANDB_NAME"] = script_args.wandb_run_name
        if script_args.wandb_mode in {"offline", "online"}:
            os.environ["WANDB_MODE"] = script_args.wandb_mode
        try:
            import wandb  # type: ignore
            wandb.init(
                project=os.environ.get("WANDB_PROJECT"),
                entity=os.environ.get("WANDB_ENTITY"),
                name=os.environ.get("WANDB_NAME"),
                config=asdict(script_args),
            )
        except Exception as e:
            print(f"[wandb] Failed to initialize W&B: {e}")

    # Set system prompt for this run (module-level variable)
    SYSTEM_PROMPT = script_args.system_prompt or ""

    set_seed(seed=11)
    _supported_models = {
        'meta-llama/Llama-2-7b-chat-hf': 'llama-2',
        'meta-llama/Llama-3.1-8B': 'llama-3',
        'meta-llama/Llama-3.1-8B-Instruct': 'llama-3',
        'mistralai/Mistral-7B-Instruct-v0.2': 'mistral',
        'google/gemma-2-2b': 'gemma',
        'google/gemma-2-2b-it': 'gemma',
        'google/gemma-2-9b': 'gemma',
        'google/gemma-2-9b-it': 'gemma',
    }
    if script_args.model_name_or_path not in _supported_models:
        _supported_list = ", ".join(_supported_models.keys())
        print(
            f"{script_args.model_name_or_path} is not in supported model list. "
            f"We support {_supported_list}"
        )
    template_name = _supported_models.get(script_args.model_name_or_path, 'llama-2')
    print('[Behavior:] ', script_args.behavior, '[Layer:] ', script_args.layer, '[Model:] ', script_args.model_name_or_path)

    # 1. load a pretrained model
    # Use eager attention for Gemma2 models as recommended
    _is_gemma2 = "gemma-2" in script_args.model_name_or_path.lower()
    _attn_kwargs = {"attn_implementation": "eager"} if _is_gemma2 else {}
    # Choose a lower-precision dtype to reduce GPU memory (bfloat16 when supported)
    _dtype = None
    if torch.cuda.is_available():
        try:
            _cap = torch.cuda.get_device_capability()
            if _cap and _cap[0] >= 8:
                _dtype = torch.bfloat16
            else:
                _dtype = torch.float16
        except Exception:
            _dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=True,
        torch_dtype=_dtype,
        **_attn_kwargs,
    )
    if script_args.sparse_steering:
        print('-----------------------------')
        print('Loading SAE...')
        sae = load_sae(
            model_path = script_args.sae_repo,
            layer_index = script_args.layer,
            sae_vector_size = script_args.sae_vector_size,
            avg_idx = script_args.sae_avg_idx,
            dtype=_dtype,
            sae_source=script_args.sae_source,
            llama_site=script_args.llama_sae_site,
            llama_expansion=script_args.llama_sae_expansion,
            llama_cache_dir=script_args.llama_sae_cache_dir,
        )
        # set to eval mode
        sae.eval()
        # freeze parameters
        for param in sae.parameters():
            param.requires_grad = False
        print('Injecting SAE into the model...')
        # Determine sparse dimension from SAE weights to size the trainable sparse vector
        sparse_dim = getattr(sae.W_enc, "shape", [None, None])[1]
        if sparse_dim is None:
            try:
                sparse_dim = sae.W_enc.size(1)  # type: ignore[attr-defined]
            except Exception:
                sparse_dim = 65536  # sensible default matching 65k
        model.model.layers[script_args.layer] = SparseBlockWrapper(
            model.model.layers[script_args.layer], 
            sae, 
            vector_size=int(sparse_dim),
        )
        print('-----------------------------')
    else:
        # Ensure dense steering vector matches model hidden size
        model.model.layers[script_args.layer] = BlockWrapper(
            model.model.layers[script_args.layer],
            vector_size=getattr(model.config, 'hidden_size', None),
        )
        
    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # Drop the separate reference model; we compute reference log-probs from the same
    # model with steering disabled via the trainer's null_ref_context.
    print('-----------------------------')
    print(script_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    ensure_tokenizer_has_chat_template(tokenizer, template_name)
    if _requires_llama_right_pad(script_args.model_name_or_path):
        tokenizer.pad_token = LLAMA_RIGHT_PAD_TOKEN
    elif tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is not None:
        if getattr(model.config, "pad_token_id", None) in (None, -1):
            model.config.pad_token_id = pad_token_id
        if getattr(model.generation_config, "pad_token_id", None) in (None, -1):
            model.generation_config.pad_token_id = pad_token_id

    # No separate ref model to freeze

    for name, param in model.named_parameters():
        if f'model.layers.{script_args.layer}.vec' not in name:
            param.requires_grad = False
            
    print('Finish loading pre-trained models...')

    # 2. Load training dataset
    train_dataset = get_data(
        behavior=script_args.behavior,
        train=True,
        template_name=template_name,
        subset_size=script_args.debug_subset_size or 0,
        tokenizer=tokenizer,
        model_name=script_args.model_name_or_path,
        use_hub_data=script_args.use_hub_data,
        dataset_path=script_args.hub_dataset_path,
        split_seed=script_args.split_seed,
        train_ratio=script_args.train_ratio,
        country_name=script_args.country_name,
        localization_status=script_args.localization_status,
        prompt_colname=script_args.prompt_colname,
        chosen_colname=script_args.chosen_colname,
        rejected_colname=script_args.rejected_colname,
    ) 

    # 3. Optionally load val dataset
    test_dataset = None
    if script_args.do_eval:
        test_dataset = get_data(
            behavior=script_args.behavior,
            train=False,
            template_name=template_name,
            subset_size=script_args.debug_subset_size or 0,
            tokenizer=tokenizer,
            model_name=script_args.model_name_or_path,
            use_hub_data=script_args.use_hub_data,
            dataset_path=script_args.hub_dataset_path,
            split_seed=script_args.split_seed,
            train_ratio=script_args.train_ratio,
            country_name=script_args.country_name,
            localization_status=script_args.localization_status,
            prompt_colname=script_args.prompt_colname,
            chosen_colname=script_args.chosen_colname,
            rejected_colname=script_args.rejected_colname,
        )
    if use_wandb and is_main_process:
        try:
            import wandb  # type: ignore
            payload = {"train_samples": len(train_dataset)}
            if test_dataset is not None:
                payload["eval_samples"] = len(test_dataset)
            wandb.summary.update(payload)
        except Exception as e:
            print(f"[wandb] Failed to update dataset sizes: {e}")
    
    # 4. initialize training arguments:
    # Ensure run_name is explicitly set and different from output_dir to avoid W&B warnings
    run_name = script_args.wandb_run_name
    output_dir = os.path.join("outputs", run_name)

    # Determine evaluation strategy: per-epoch only when enabled
    _eval_strategy = "epoch" if script_args.do_eval else "no"
    _eval_steps = None

    # Resolve warmup preference: steps overrides ratio; default to no warmup
    _warmup_kwargs = {}
    if script_args.warmup_steps is not None:
        _warmup_kwargs["warmup_steps"] = int(script_args.warmup_steps)
        _warmup_kwargs["warmup_ratio"] = 0.0
    elif script_args.warmup_ratio is not None:
        _warmup_kwargs["warmup_ratio"] = float(script_args.warmup_ratio)
    else:
        _warmup_kwargs["warmup_ratio"] = 0.0

    use_bf16 = _supports_bf16()
    if not use_bf16:
        print("[WARN] BF16 not supported on this setup; falling back to FP16.")

    training_args = DPOConfig(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        logging_steps=script_args.logging_steps,
        save_strategy="no",
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        dataloader_pin_memory=False,
        dataloader_num_workers=1,
        learning_rate=script_args.learning_rate,
        eval_strategy=_eval_strategy,
        eval_steps=_eval_steps,
        output_dir=output_dir,
        run_name=run_name,
        report_to=script_args.report_to,
        lr_scheduler_type=script_args.lr_scheduler_type,
        **_warmup_kwargs,
        optim=script_args.optimizer_type,
        bf16=use_bf16,
        fp16=not use_bf16,
        remove_unused_columns=False,
        max_prompt_length=script_args.max_prompt_length,
        max_length=script_args.max_length,
        generate_during_eval=script_args.generate_during_eval,
        # Use a true single-model BiPO setup, computing reference from the same model
        # via the trainer's null_ref_context (no precompute, no separate ref model).
        precompute_ref_log_probs=False,
        single_model_ref=True,
    )

    if use_wandb and is_main_process:
        try:
            # Attach training args to the run config for traceability
            wandb.config.update({"training_args": training_args.to_dict()}, allow_val_change=True)
        except Exception as e:
            print(f"[wandb] Failed to update config: {e}")

   # 5. initialize the DPO trainer
    # Derive task tag for vector directory naming (mcq vs open-ended)
    def _infer_is_mcq_from_args(a) -> bool:
        try:
            if getattr(a, "mcq_training", None) is True:
                return True
        except Exception:
            pass
        s1 = str(getattr(a, "hub_dataset_path", "")).lower()
        s2 = str(getattr(a, "prompt_colname", "")).lower()
        s3 = str(getattr(a, "chosen_colname", "")).lower()
        return ("mcq" in s1) or ("mcq" in s2) or ("mcq" in s3)

    _task_suffix = "mcq" if _infer_is_mcq_from_args(script_args) else "oe"
    _vec_name = f"{template_name}-{_task_suffix}"

    dpo_trainer = BiPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=(
            {'test_dataset_add': test_dataset, 'test_dataset_sub': test_dataset}
            if (script_args.do_eval and test_dataset is not None)
            else None
        ),
        tokenizer=tokenizer,
        behavior=script_args.behavior,
        layer=script_args.layer,
        name=_vec_name,
    )

    # Save steering vector after each training step (outside evaluation)
    try:
        dpo_trainer.add_callback(
            SaveVectorCallback(layer=script_args.layer, vec_dir=dpo_trainer.vec_dir, accelerator=dpo_trainer.accelerator)
        )
    except Exception as e:
        print(f"[WARN] Could not attach SaveVectorCallback: {e}")

    # 6. Start training
    print_trainable_parameters(model)
    if use_wandb and is_main_process:
        try:
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in model.parameters())
            wandb.summary.update(
                {
                    "trainable_params": trainable_params,
                    "all_params": all_params,
                    "trainable_percent": 100.0 * trainable_params / max(all_params, 1),
                }
            )
        except Exception as e:
            print(f"[wandb] Failed to update summary: {e}")
    dpo_trainer.train()
