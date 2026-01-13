import json
import re
from tqdm import tqdm
from datasets import (
    Dataset,
    DatasetDict,
)
import torch
import datetime
from accelerate.utils import gather_object

def pprint_json(obj):
    """ Pretty print of Json files/content"""
    print(json.dumps(obj, indent=3, ensure_ascii=False))

def clean_generated_text(text, prompt):
    """ Simple cleaning function that just splits on 'model' and 'model:' and takes the last part """
    # Remove the prompt if it's at the beginning
    if text.startswith(prompt):
        text = text[len(prompt):].strip()
    
    # First try to split by "model:"
    if "model:" in text:
        text = text.split("model:")[-1].strip()
    # If that didn't work, try "model"
    elif "model" in text:
        text = text.split("model")[-1].strip()
    
    # Remove a leading colon if one exists
    if text.startswith(":"):
        text = text[1:].strip()
        
    return text

def generate_dpo_samples_batched(
    sft_dataset,
    start_index,
    end_index,
    model,
    tokenizer,
    distributed_state,
    in_sharegpt_format: bool = False,
    conversation_colname: str = "conversations",
    prompt_colname: str = "prompt",
    chosen_colname: str = "chosen",
    batch_size: int = 8,
    augmentation_params: dict = None,
    max_new_tokens: int = 256,
    max_length: int = 256,
    repetition_penalty: float = 1.1,
    checkpoint_freq: int = 50,
    hf_save_path: str = None,
    HF_TOKEN: str = None,
    put_in_sharegpt_format: bool = False,
    model_name_safe: str = None,
    system_prompt: str = None,
):
    """
    Generate DPO samples using Accelerate's split_between_processes.
    - Builds a flat list of (prompt, chosen) pairs.
    - Splits these examples across processes with split_between_processes.
    - Performs generation in local batches.
    """
    # unwrap DataParallel
    on_policy_model = model.module if isinstance(model, torch.nn.DataParallel) else model
    device = distributed_state.device
    num_procs = distributed_state.num_processes
    rank = distributed_state.process_index

    rejected_colname = f"rejected_{model_name_safe}"

    # augmentation setup
    do_augment = augmentation_params is not None
    param_keys = list(augmentation_params.keys()) if do_augment else []
    param_values = list(zip(*augmentation_params.values())) if do_augment else []

    # build flat list of examples
    def build_flat_list_data(
        sft_dataset,
        start_index: int = 0,
        end_index: int = 0,
        in_sharegpt_format: bool = False,
        conversation_colname: str = "conversations",
        prompt_colname: str = "prompt",
        chosen_colname: str = "chosen",
    ):
        examples = []
        for i in range(start_index, end_index + 1):
            sample = sft_dataset[i]
            # Keep a copy of all original columns
            try:
                source_columns = {k: sample[k] for k in sample.keys()}
            except Exception:
                source_columns = dict(sample)

            # Exclude prompt/ chosen from the extension set
            filtered_source_columns = {
                k: v for k, v in source_columns.items()
                if k not in {prompt_colname, chosen_colname}
            }

            # if in sharegpt format, we need to extract data from the nested json
            if in_sharegpt_format:
                messages = sample[conversation_colname]
                users = [m["content"] for m in messages if m["role"].upper() == "USER"]
                ants  = [m["content"] for m in messages if m["role"].upper() == "ASSISTANT"]
                for u, a in zip(users, ants):
                    data_record = {
                        "prompt": u,
                        "chosen": a,
                        "meta": {
                            "dataset": sample.get("dataset", "unknown"),
                            "id": sample.get("id", "unknown"),
                        },
                    }
                    # Extend with all other columns except prompt/ chosen
                    data_record.update(filtered_source_columns)
                    examples.append(data_record)
            else:
                # we get data from the columns
                data_record = {
                    prompt_colname: sample.get(prompt_colname, "unknown"),
                    chosen_colname: sample.get(chosen_colname, "unknown"),
                    "meta": {
                        "dataset": sample.get("dataset", "unknown"),
                        "id": sample.get("id", "unknown"),
                    },
                }
                # Extend with all other columns except prompt/ chosen
                data_record.update(filtered_source_columns)
                examples.append(data_record)

        return examples
    
    examples = build_flat_list_data(
        sft_dataset,
        start_index,
        end_index,
        in_sharegpt_format,
        conversation_colname,
        prompt_colname,
        chosen_colname,
    )
    # split these examples across processes
    if num_procs > 1:
        with distributed_state.split_between_processes(examples, apply_padding=False) as local_examples:
            examples_to_process = local_examples
    else:
        examples_to_process = examples

    dpo_samples = []

    # process local examples in batches
    total_batches = len(range(0, len(examples_to_process), batch_size))
    for batch_idx, batch_start in enumerate(tqdm(
        range(0, len(examples_to_process), batch_size),
        disable=not distributed_state.is_main_process,
        total=total_batches,
        desc="Generating DPO samples..."
    )):
        batch = examples_to_process[batch_start : batch_start + batch_size]
        prompts = [ex[prompt_colname] for ex in batch]
        # Build conversations, optionally prepending a system prompt
        if system_prompt:
            convs = [[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": p},
            ] for p in prompts]
        else:
            convs = [[{"role": "user", "content": p}] for p in prompts]

        # tokenize & move to device + ensure aligned sequence lengths
        # Some models (e.g., Gemma) don't support a system role; fallback to user-only with prepended instructions
        try:
            inputs = tokenizer.apply_chat_template(
                convs,
                return_tensors="pt",
                padding="max_length",  # Use fixed-length padding
                max_length=max_length,
                truncation=True,
                return_dict=True,
                add_generation_prompt=True,
            )
        except Exception as e:
            if system_prompt and ("system" in str(e).lower() or "not support" in str(e).lower()):
                convs_fallback = [[{"role": "user", "content": f"{system_prompt}\n\n{p}"}] for p in prompts]
                inputs = tokenizer.apply_chat_template(
                    convs_fallback,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_dict=True,
                    add_generation_prompt=True,
                )
            else:
                raise

        # Ensure they are contiguous
        input_ids = inputs.input_ids.to(device).contiguous()  # Added .contiguous()
        attention_mask = (
            inputs.attention_mask.to(device).contiguous()  # Added .contiguous()
            if inputs.attention_mask is not None
            else None
        )

        src_lens = [len(ids) for ids in input_ids]

        # generate
        with torch.no_grad():
            outputs = on_policy_model.generate(
                input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # decode & collect
        for i, out_ids in enumerate(outputs):
            new_tokens = out_ids[src_lens[i] :].tolist()
            rejected = tokenizer.decode(new_tokens, skip_special_tokens=True)
            rejected = clean_generated_text(rejected, prompts[i])
            
            # Keep a copy of all original columns from the prepared batch item
            try:
                _all_cols = {k: batch[i][k] for k in batch[i].keys()}
            except Exception:
                _all_cols = dict(batch[i])
            # Exclude prompt/chosen/meta from the extension set
            _filtered_source_columns = {
                k: v for k, v in _all_cols.items()
                if k not in {prompt_colname, chosen_colname, "meta", "metadata"}
            }

            base = {
                prompt_colname: prompts[i],
                chosen_colname: batch[i][chosen_colname],
                # model-specific rejected column to allow multiple in the same dataset
                rejected_colname: rejected,
                "metadata": {
                    **batch[i]["meta"],
                    "generation_params": {
                        "do_sample": True,
                        "max_new_tokens": max_new_tokens,
                        # "repetition_penalty": repetition_penalty,
                    },
                    "is_augmented": False,
                },
            }
            # Extend with all other columns except prompt/ chosen
            base.update(_filtered_source_columns)
            dpo_samples.append(base)

            # augmentation
            if do_augment:
                for combo in param_values:
                    gen_args = dict(zip(param_keys, combo))
                    gen_args.update({
                        "do_sample": True,
                        "max_new_tokens": max_new_tokens,
                        # "repetition_penalty": repetition_penalty,
                        "pad_token_id": tokenizer.pad_token_id,
                        "eos_token_id": tokenizer.eos_token_id,
                    })
                    aug_out = on_policy_model.generate(
                        out_ids[: src_lens[i]].unsqueeze(0).to(device),
                        attention_mask=(
                            attention_mask[i : i + 1] if attention_mask is not None else None
                        ),
                        **gen_args,
                    )
                    aug_tokens = aug_out[0][src_lens[i] :].tolist()
                    aug_rej = tokenizer.decode(aug_tokens, skip_special_tokens=True)
                    aug_rej = clean_generated_text(aug_rej, prompts[i])

                    aug_record = {
                        prompt_colname: prompts[i],
                        chosen_colname: batch[i][chosen_colname],
                        # model-specific rejected column to allow multiple in the same dataset
                        rejected_colname: aug_rej,
                        "metadata": {
                            **batch[i]["meta"],
                            "generation_params": gen_args,
                            "is_augmented": True,
                        },
                    }
                    # Extend with all other columns except prompt/ chosen
                    aug_record.update(_filtered_source_columns)
                    dpo_samples.append(aug_record)

        # Checkpointing to not loose all the data if a crash happens
        if batch_idx % checkpoint_freq == 0:
            # First sync all processes
            if num_procs > 1:
                torch.distributed.barrier()
            
            # Gather ALL samples from ALL processes
            if num_procs > 1:
                gathered_for_checkpoint = [None] * num_procs
                torch.distributed.all_gather_object(
                    gathered_for_checkpoint, 
                    dpo_samples,
                )
                combined_samples = [item for sublist in gathered_for_checkpoint for item in sublist]
            else:
                combined_samples = dpo_samples
            
            # Only main process saves COMBINED data
            if distributed_state.is_main_process:
                commit_message = f"Batch {batch_idx}/{total_batches}"
                dpo_dataset = save_checkpoint_to_hub(
                    samples=combined_samples,  # Save COMBINED data
                    batch_index=batch_idx,
                    hf_save_path=hf_save_path,
                    commit_message=commit_message,
                    distributed_state=distributed_state,
                    token=HF_TOKEN,
                    model_name_safe=model_name_safe,
                    put_in_sharegpt_format=put_in_sharegpt_format,
                    prompt_colname=prompt_colname,
                    chosen_colname=chosen_colname,
                )
            
            # Sync after saving
            if num_procs > 1:
                torch.distributed.barrier()

        # Clear CUDA cache between generations
        #if torch.cuda.is_available():
        #    torch.cuda.empty_cache()
            
    # gather from all processes
    if num_procs > 1:
        gathered = [None] * num_procs

        try:
            torch.distributed.all_gather_object(
                gathered, 
                dpo_samples,
            )
        except RuntimeError as e:
            print(f"Rank {rank} failed to gather: {e}")
            return []
        if distributed_state.is_main_process:
            dpo_samples = [item for sub in gathered for item in sub]
        else:
            return []

    return dpo_samples

def to_conversation_format(generated_data, put_in_sharegpt_format, model_name_safe, prompt_colname, chosen_colname):
    """Create a HF Dataset preserving original columns and only adding rejected_*.

    - Preserves all original columns as found in the input samples (no renaming).
    - Adds the model-specific rejected column: `rejected_{model_name_safe}`.
    - Skips internal helper fields like `metadata` if present.
    - When put_in_sharegpt_format=True, transforms the values of chosen/rejected to
      ShareGPT-style turns while still keeping original column names.
    """

    rejected_colname = f"rejected_{model_name_safe}"

    # Determine all columns to include: union of keys across samples, excluding internal-only keys
    include_cols = set()
    for s in generated_data:
        try:
            keys = set(s.keys())
        except Exception:
            keys = set(dict(s).keys())
        include_cols.update(keys)
    # Remove internal helper columns we do not want to persist
    include_cols.discard("metadata")
    include_cols.discard("meta")

    # Ensure the rejected column is present in the schema
    include_cols.add(rejected_colname)

    # Initialize processed data containers
    processed_data = {col: [] for col in sorted(include_cols)}

    # Fill rows
    for sample in generated_data:
        # Handle optional ShareGPT conversion for chosen and rejected columns
        if put_in_sharegpt_format and prompt_colname in sample:
            prompt_val = sample.get(prompt_colname, None)
            if chosen_colname in sample:
                processed_data[chosen_colname].append([
                    {"content": prompt_val, "role": "user"},
                    {"content": sample.get(chosen_colname, None), "role": "assistant"},
                ])
            if rejected_colname in sample:
                processed_data[rejected_colname].append([
                    {"content": prompt_val, "role": "user"},
                    {"content": sample.get(rejected_colname, None), "role": "assistant"},
                ])

            # Fill the remaining columns for this row
            for col in processed_data.keys():
                if col in {chosen_colname, rejected_colname}:
                    continue  # already appended above
                processed_data[col].append(sample.get(col, None))
        else:
            # No ShareGPT conversion: copy values directly
            for col in processed_data.keys():
                processed_data[col].append(sample.get(col, None))

    dataset = DatasetDict({"train": Dataset.from_dict(processed_data)})
    return dataset


def save_checkpoint_to_hub(
    samples,
    batch_index,
    hf_save_path,
    commit_message,
    distributed_state,
    token,
    model_name_safe=None,
    put_in_sharegpt_format: bool = False,
    prompt_colname: str = "prompt",
    chosen_colname: str = "chosen",
):
    """ Helper function to save checkpoint to HF Hub """
    if distributed_state.is_main_process:
        try:
            # Infer model_name_safe if not provided: look for a single 'rejected_*' key in samples
            if model_name_safe is None:
                rejected_keys = set()
                if samples:
                    first = samples[0]
                    try:
                        keys = list(first.keys())
                    except Exception:
                        keys = list(dict(first).keys())
                    rejected_keys = [k for k in keys if k.startswith("rejected_")]
                if len(rejected_keys) == 1:
                    model_name_safe = rejected_keys[0].replace("rejected_", "", 1)
                elif len(rejected_keys) > 1:
                    raise ValueError(f"Multiple rejected_* columns found: {rejected_keys}. Please specify model_name_safe.")
                else:
                    raise ValueError("No rejected_* column found in samples; cannot infer model_name_safe.")

            # Convert to DPO format
            dpo_dataset = to_conversation_format(samples, put_in_sharegpt_format, model_name_safe, prompt_colname, chosen_colname)
            
            # Print some debug info
            print(f"[INFO] Checkpoint - Saving batch {batch_index} with {len(samples)} samples to HF Hub...")
            
            # Push to hub with version tag
            dpo_dataset.push_to_hub(
                hf_save_path,
                private=True,
                commit_message=commit_message,
                token=token,
            )
            print(f"[INFO] Successfully saved checkpoint to {hf_save_path}")
            return dpo_dataset
        except Exception as e:
            print(f"[ERROR] Failed to save checkpoint: {e}")
            return None
