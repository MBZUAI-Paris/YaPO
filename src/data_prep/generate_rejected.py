# -*- coding: utf-8 -*-
import os
import sys
from pathlib import Path

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Load libraries
import argparse
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from datasets import (
    load_dataset,
)
import torch
import re

# Allow importing shared helpers (chat templates, etc.) by mimicking how
# modeling/BiPO scripts locate them, even when SLURM copies this file to /var.
_repo_candidates = []
if os.getenv("SLURM_SUBMIT_DIR"):
    _repo_candidates.append(Path(os.environ["SLURM_SUBMIT_DIR"]).resolve())
_repo_candidates.append(Path(__file__).resolve().parents[1])
_modeling_dir = None
for _candidate in _repo_candidates:
    maybe_dir = _candidate / "modeling" / "BiPO"
    if maybe_dir.exists():
        _modeling_dir = maybe_dir
        break
if _modeling_dir and str(_modeling_dir) not in sys.path:
    sys.path.insert(0, str(_modeling_dir))

try:
    from chat_templates import ensure_tokenizer_has_chat_template
except Exception:  # pragma: no cover - fallback when helper unavailable
    def ensure_tokenizer_has_chat_template(tokenizer, template_hint=None):
        return getattr(tokenizer, "chat_template", None)

from utils import (
    pprint_json,
    generate_dpo_samples_batched,
    to_conversation_format,
    save_checkpoint_to_hub,
)


from accelerate import Accelerator

LLAMA_RIGHT_PAD_TOKEN = "<|finetune_right_pad_id|>"


def _infer_template_hint(model_id: str) -> str:
    name = (model_id or "").lower()
    template_map = [
        ("llama-3.1", "llama-3"),
        ("llama-3", "llama-3"),
        ("llama-2", "llama-2"),
        ("gemma-2", "gemma"),
        ("gemma", "gemma"),
        ("mistral", "mistral"),
    ]
    for pattern, template in template_map:
        if pattern in name:
            return template
    return "generic"


def _requires_llama_right_pad(model_id: str) -> bool:
    name = (model_id or "").lower()
    return "llama" in name


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Core generation params
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    # Data and model params
    parser.add_argument("--on_policy_model_name", type=str, default="google/gemma-2b-it")
    parser.add_argument("--data_path", type=str, default="Raniahossam33/Arabic_cultural_dataset")
    parser.add_argument("--save_data_path", type=str, default="BounharAbdelaziz/Arabic_cultural_dataset_processed")
    # Ranges and checkpointing
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=-1, help="-1 means use full dataset")
    parser.add_argument("--checkpoint_freq", type=int, default=50)
    # Optional augmentation
    parser.add_argument("--do_augment", action="store_true", help="Use varied sampling params per prompt to augment the dataset")
    parser.add_argument("--open_ended", action="store_true", help="Generate the rejected samples based on open-ended prompts.")
    # System prompt for MCQ formatting
    parser.add_argument(
        "--system_prompt",
        type=str,
        default=(
            "You will be given multiple-choice questions (MCQ)."
            " Think step by step if needed, but at the end output only the final answer index"
            " wrapped in \\boxed{index} with no extra text. For example: \\boxed{2}."
        ),
        help="System prompt injected before user message. Control from bash script.",
    )
    # Column names for prompt and chosen answer
    parser.add_argument(
        "--prompt_colname",
        type=str,
        default="prompt",
        help="Column name in dataset that contains the prompt/question.",
    )
    parser.add_argument(
        "--chosen_colname",
        type=str,
        default="chosen",
        help="Column name in dataset that contains the chosen/ground-truth answer.",
    )
    args = parser.parse_args()

    # Load env
    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")

    # Paths
    DATA_PATH = args.data_path
    SAVE_DATA_PATH = args.save_data_path

    # Compute device
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Distributed setup
    distributed_state = Accelerator()

    # Load dataset first to resolve end_index
    dataset = load_dataset(
        DATA_PATH,
        split="train",
        token=HF_TOKEN,
    )

    # Resolve index range
    start_index = max(0, args.start_index)
    resolved_end = (len(dataset) - 1) if args.end_index < 0 else args.end_index
    end_index = min(resolved_end, len(dataset) - 1)

    # Inference params
    checkpoint_freq = args.checkpoint_freq
    batch_size = args.batch_size
    max_new_tokens = args.max_new_tokens
    repetition_penalty = args.repetition_penalty

    augmentation_params = {
        "temperature": [0.7, 1.0],
        "top_p": [0.9, 1.0],
    } if args.do_augment else None

    if distributed_state.is_main_process:
        print(f"Total number of samples: {len(dataset)}")
        print("Sample example:")
        pprint_json(dataset[0])

    # Tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        args.on_policy_model_name,
        token=HF_TOKEN,
    )
    template_hint = _infer_template_hint(args.on_policy_model_name)
    ensure_tokenizer_has_chat_template(tokenizer, template_hint)
    if distributed_state.is_main_process:
        print(f"[INFO] Using chat template hint: {template_hint}")

    tokenizer.padding_side = "left"  # Essential for autoregressive models

    if _requires_llama_right_pad(args.on_policy_model_name):
        tokenizer.pad_token = LLAMA_RIGHT_PAD_TOKEN
    elif tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.on_policy_model_name,
        token=HF_TOKEN,
        torch_dtype=torch.bfloat16,
    )
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is not None:
        if getattr(model.config, "pad_token_id", None) in (None, -1):
            model.config.pad_token_id = pad_token_id
        if getattr(model.generation_config, "pad_token_id", None) in (None, -1):
            model.generation_config.pad_token_id = pad_token_id
    model = model.to(distributed_state.device)
    model.eval()

    # Debug augmentation combos
    if augmentation_params is not None and distributed_state.is_main_process:
        param_values = list(zip(*augmentation_params.values()))
        param_keys = list(augmentation_params.keys())
        for param_combo in param_values:
            print(f"param_combo: {param_combo}")
            print(f"dict(zip(param_keys, param_combo)): {dict(zip(param_keys, param_combo))}")

    # derive a stable, column-safe model name for rejected column
    def _derive_model_name(m):
        name = None
        # common attributes on HF models
        name = getattr(m, "name_or_path", None)
        if not name and hasattr(m, "config") and m.config is not None:
            name = getattr(m.config, "_name_or_path", None) or getattr(m.config, "name_or_path", None)
        if not name:
            name = m.__class__.__name__
        # take last component if repo path-like
        if "/" in name:
            name = name.split("/")[-1]
        # sanitize to be column-safe
        name = re.sub(r"[^A-Za-z0-9]+", "_", str(name)).strip("_")
        return name

    model_name_safe = _derive_model_name(model)
    if args.open_ended:
        model_name_safe = model_name_safe + "_open_ended"
    else:
        model_name_safe = model_name_safe + "_mcq"
        
    # Generation
    generated_data = generate_dpo_samples_batched(
        dataset,
        start_index,
        end_index,
        model,
        tokenizer,
        distributed_state,
        batch_size=batch_size,
        augmentation_params=augmentation_params,
        max_new_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
        checkpoint_freq=checkpoint_freq,
        hf_save_path=SAVE_DATA_PATH,
        HF_TOKEN=HF_TOKEN,
        model_name_safe=model_name_safe,
        in_sharegpt_format = False,
        conversation_colname = None,
        prompt_colname = args.prompt_colname,
        chosen_colname = args.chosen_colname,
        system_prompt = args.system_prompt,
    )

    # Save final dataset to hub
    if distributed_state.is_main_process:
        print("[INFO] Data generation finished! Now converting to HuggingFace format...")

        commit_message = "Final Batch"
        dpo_dataset = save_checkpoint_to_hub(
            samples=generated_data,
            batch_index=-1,
            hf_save_path=SAVE_DATA_PATH,
            commit_message=commit_message,
            distributed_state=distributed_state,
            token=HF_TOKEN,
            model_name_safe=model_name_safe,
            put_in_sharegpt_format=False,
            prompt_colname=args.prompt_colname,
            chosen_colname=args.chosen_colname,
        )

        print(f"[INFO] dataset: {dataset}")
        print(f"[INFO] dpo_dataset: {dpo_dataset}")
        if dpo_dataset is not None:
            print(f"[INFO] dpo_dataset[train][0]: {dpo_dataset['train'][0]}")
