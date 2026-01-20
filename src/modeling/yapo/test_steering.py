import os
import json
import argparse
import math
import torch as t
import pandas as pd
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

from datasets import load_dataset, Dataset, DatasetDict

from wrappers.model_wrapper import ModelWrapper
from wrappers.sparse_model_wrapper import SparseModelWrapper
from accelerate import Accelerator
from tqdm import tqdm


HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
SYSTEM_PROMPT = ""


def get_steering_vector(layer: int, ckp_epoch: int, vectors_path_or_file: str):
    """Load a steering vector.

    If `vectors_path_or_file` is a .pt file path, load it directly.
    Otherwise, treat it as a directory and build the filename from epoch/layer.
    """
    candidate = vectors_path_or_file
    if not (os.path.isfile(candidate) and candidate.endswith(".pt")):
        candidate = os.path.join(vectors_path_or_file, f"vec_ep{ckp_epoch}_layer{layer}.pt")
    print(f"Loading steering vector from: {candidate}")
    if not os.path.exists(candidate):
        print(f"Warning: Vector file {candidate} does not exist")
        return None
    return t.load(candidate, map_location='cpu')  # Load to CPU first to save GPU memory


def process_batch(
    prompts: List[str],
    model,
    max_new_tokens: int,
) -> List[Dict[str, str]]:
    """Process a batch of prompts and return results."""
    histories = [[(p, None)] for p in prompts]
    outputs = model.generate_text_with_conversation_history_batch(histories, max_new_tokens=max_new_tokens)
    results = []
    for p, out in zip(prompts, outputs):
        results.append({
            "question": p,
            "model_output": out.split("[/INST]")[-1].strip() if "[/INST]" in out else out.strip(),
            "raw_model_output": out,
        })
    return results


def _slugify_leaf(model_name_or_path: str) -> str:
    """Convert model name to safe filename."""
    leaf = model_name_or_path.split("/")[-1]
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in leaf)
    return safe


def load_or_generate_baseline(
    model: ModelWrapper,
    test_prompts: List[str],
    layer: int,
    country_tag: str,
    save_results_path_baseline: str,
    max_new_tokens: int,
    batch_size: int,
    verbose: bool = False
) -> List[str]:
    """Load existing baseline or generate it once."""
    loc_tag = (os.getenv("LOC_TAG") or "").strip()
    baseline_filename = os.path.join(
        save_results_path_baseline,
        f"baseline_layer{layer}_country{country_tag}{loc_tag}.json",
    )
    
    if os.path.exists(baseline_filename):
        print(f"Loading existing baseline from: {baseline_filename}")
        try:
            with open(baseline_filename, "r") as f:
                baseline_data = json.load(f)
                return [r.get("model_output", "") for r in baseline_data]
        except Exception as e:
            print(f"Error loading baseline: {e}. Regenerating...")
    
    print("Generating baseline outputs (no steering)...")
    baseline_results = []
    total_batches = math.ceil(len(test_prompts) / max(1, batch_size))
    
    for i in tqdm(range(0, len(test_prompts), batch_size), total=total_batches, desc="Baseline", leave=False):
        batch = test_prompts[i:i+batch_size]
        model.reset_all()  # Ensure no steering is active
        batch_results = process_batch(batch, model, max_new_tokens)
        baseline_results.extend(batch_results)
        
        if verbose:
            for r in batch_results:
                print(f"Q: {r['question']}")
                print(f"A: {r['model_output']}")
                print()
    
    # Save baseline
    os.makedirs(save_results_path_baseline, exist_ok=True)
    with open(baseline_filename, "w") as f:
        json.dump(baseline_results, f, indent=2)
    
    return [r["model_output"] for r in baseline_results]


def generate_steering_outputs(
    model,
    test_prompts: List[str],
    steering_vector: t.Tensor,
    layer: int,
    multipliers: List[float],
    max_new_tokens: int,
    batch_size: int,
    save_path: Optional[str],
    ckp_epoch: int,
    country_tag: str,
    steering_type: str,
    verbose: bool = False
) -> Dict[float, List[str]]:
    """Generate outputs for all multipliers for a given steering vector."""
    outputs_by_multiplier = {}
    
    for multiplier in multipliers:
        save_filename = None
        if save_path is not None:
            save_filename = os.path.join(
                save_path,
                f"result_ep{ckp_epoch}_layer{layer}_m{multiplier}_country{country_tag}.json",
            )
            if os.path.exists(save_filename):
                print(f"Loading existing {steering_type} results for m={multiplier}")
                try:
                    with open(save_filename, "r") as f:
                        prev_data = json.load(f)
                        outputs_by_multiplier[multiplier] = [r.get("model_output", "") for r in prev_data]
                    continue
                except Exception as e:
                    print(f"Error loading {save_filename}: {e}. Regenerating...")
        
        print(f"Generating {steering_type} outputs (multiplier={multiplier})...")
        results = []
        total_batches = math.ceil(len(test_prompts) / max(1, batch_size))
        
        for i in tqdm(range(0, len(test_prompts), batch_size), 
                     total=total_batches, desc=f"{steering_type} m={multiplier}", leave=False):
            batch = test_prompts[i:i+batch_size]
            model.reset_all()
            if multiplier != 0.0:  # Skip steering for multiplier 0
                model.set_add_activations(layer, multiplier * steering_vector.to(model.device))
            batch_results = process_batch(batch, model, max_new_tokens)
            results.extend(batch_results)
            
            if verbose:
                for r in batch_results:
                    print(f"Q: {r['question']}")
                    print(f"A: {r['model_output']}")
                    print()
        
        # Save results if a save path was provided
        if save_filename is not None:
            os.makedirs(save_path, exist_ok=True)
            with open(save_filename, "w") as f:
                json.dump(results, f, indent=2)
        
        outputs_by_multiplier[multiplier] = [r["model_output"] for r in results]
    
    return outputs_by_multiplier


def load_and_merge_existing_dataset(
    push_repo: str, 
    push_split_name: str, 
    base_df: pd.DataFrame,
    key_col: str
) -> Tuple[pd.DataFrame, Set[str]]:
    """Load existing dataset and return merged df + existing columns."""
    existing_columns = set()
    try:
        existing = load_dataset(push_repo, split=push_split_name, token=HUGGINGFACE_TOKEN)
        existing_df = existing.to_pandas()
        existing_columns = set(existing_df.columns)
        
        if key_col in existing_df.columns:
            print(f"Merging with existing dataset ({len(existing_df)} rows)")
            merged_df = existing_df.merge(base_df[[key_col]], on=key_col, how="outer")
            return merged_df, existing_columns
        else:
            print(f"Key column '{key_col}' not found in existing dataset")
            return base_df.copy(), existing_columns
            
    except Exception as e:
        print(f"No existing dataset to merge (or failed to load): {e}")
        return base_df.copy(), existing_columns


def run_test_steering_on_hf(
    model_name_or_path: str,
    behavior_dense: str,
    behavior_sparse: str,
    layer: int,
    epochs: List[int],  # Changed to handle multiple epochs efficiently
    multipliers: List[float],
    max_new_tokens: int,
    dataset_repo: str,
    country: str = "egypt",
    localization_status: str = "both",
    verbose: bool = False,
    dense_vector_template: Optional[str] = None,  # Template for dense vector paths
    sparse_vector_template: Optional[str] = None,  # Template for sparse vector paths
    sae_repo: str = "google/gemma-scope-2b-pt-res",
    sae_width: str = "65k",
    sae_avg_idx: str = "68",
    batch_size: int = 8,
    limit: int = 0,
    push_repo: Optional[str] = None,
    push_private: bool = True,
    push_split_name: str = "test",
    push_commit_message: str = "Add steering evaluation results",
    mcq_eval: bool = False,
):
    """
    Efficient batch processing of multiple epochs.
    """
    accelerator = Accelerator()
    is_main = accelerator.is_main_process
    world_size = accelerator.num_processes
    rank = accelerator.process_index
    if world_size > 1:
        print(f"Accelerate distributed inference: rank {rank}/{world_size}")
    else:
        print("Single-process inference")

    # Setup paths
    model_slug = _slugify_leaf(model_name_or_path)
    # save_results_path_baseline = f"result/baseline_{model_slug}"
    # save_results_path_dense = f"result/{behavior_dense}_{model_slug}"
    # save_results_path_sparse = f"result/{behavior_sparse}_{model_slug}"
    
    save_results_path_baseline = sparse_vector_template.split("vector/")[-1]
    save_results_path_dense = os.path.join("result", f"shards_dense_{save_results_path_baseline}")
    save_results_path_sparse = os.path.join("result", f"shards_sparse_{save_results_path_baseline}")
    # print(f"shard_path: ", shard_path)

    # Load and filter dataset once
    print(f"Loading dataset: {dataset_repo} (split='test')")
    ds = load_dataset(dataset_repo, split="test", token=HUGGINGFACE_TOKEN)
    
    # Validate columns
    cols = set(ds.column_names)
    if "country" not in cols:
        raise ValueError(f"Dataset missing required 'country' column. Available: {sorted(cols)}")
    
    prompt_field = "prompt-mcq" if mcq_eval else "prompt_open_ended"
    if prompt_field not in cols:
        raise ValueError(f"Dataset missing required '{prompt_field}' column. Available: {sorted(cols)}")

    # Filter by country
    country_norm = country.strip().lower()
    ds_country = ds.filter(lambda ex: str(ex["country"]).strip().lower() == country_norm)

    # Optional: filter by localization_status ('localized' | 'nolocalized'); 'both' disables filtering
    loc_norm = (localization_status or "both").strip().lower()
    if loc_norm in {"localized", "nolocalized"}:
        if "localization_status" not in cols:
            print("[WARN] 'localization_status' column not found; skipping localization filter.")
        else:
            ds_country = ds_country.filter(lambda ex: str(ex["localization_status"]).strip().lower() == loc_norm)
    if limit > 0:
        ds_country = ds_country.shuffle(seed=1998).select(range(min(limit, len(ds_country))))
    
    print(f"Filtered by country='{country}'. Rows: {len(ds_country)}")
    if len(ds_country) == 0:
        print("No rows found for requested country. Exiting.")
        return

    # Shard dataset across processes for data-parallel inference
    if world_size > 1:
        ds_shard = ds_country.shard(num_shards=world_size, index=rank)
    else:
        ds_shard = ds_country

    test_prompts = ds_shard[prompt_field]
    # Keep local key list for merging later
    local_keys = [str(x) for x in test_prompts]
    country_tag = "".join(c for c in country_norm if c.isalnum() or c in ("-", "_"))

    # Initialize models once
    print("Initializing models...")
    model_dense = ModelWrapper(HUGGINGFACE_TOKEN, SYSTEM_PROMPT, model_name_or_path)
    model_dense.set_save_internal_decodings(False)

    model_sparse = None
    try:
        model_sparse = SparseModelWrapper(
            HUGGINGFACE_TOKEN,
            SYSTEM_PROMPT,
            model_name_or_path,
            injection_layer=layer,
            sae_repo=sae_repo,
            sae_width=sae_width,
            sae_avg_idx=sae_avg_idx,
        )
        model_sparse.set_save_internal_decodings(False)
        print("Sparse model initialized successfully")
    except Exception as e:
        print(f"Failed to initialize sparse model: {e}")

    # Generate baseline once (shared across all epochs)
    # Baseline: if single process, reuse cache; else compute locally per shard without saving
    if world_size == 1:
        baseline_outputs = load_or_generate_baseline(
            model_dense, test_prompts, layer, country_tag,
            save_results_path_baseline, max_new_tokens, batch_size, verbose
        )
    else:
        print("Generating baseline on shard...")
        baseline_outputs = []
        total_batches = math.ceil(len(test_prompts) / max(1, batch_size))
        for i in tqdm(range(0, len(test_prompts), batch_size), total=total_batches, desc="Baseline(shard)", leave=False):
            batch = test_prompts[i:i+batch_size]
            model_dense.reset_all()
            batch_results = process_batch(batch, model_dense, max_new_tokens)
            baseline_outputs.extend([r["model_output"] for r in batch_results])

    # Prepare results accumulator
    results_data = defaultdict(dict)
    
    # Process each epoch
    for epoch in epochs:
        print(f"\n=== Processing epoch {epoch} ===")
        
        # Load vectors for this epoch
        dense_vector = None
        if dense_vector_template:
            dense_path = dense_vector_template.format(epoch=epoch, layer=layer)
            dense_vector = get_steering_vector(layer, epoch, dense_path)
            print(f'dense_path: {dense_path}')
            if dense_vector is not None:
                print(f"Loaded dense vector for epoch {epoch}")
        
        sparse_vector = None
        if sparse_vector_template and model_sparse:
            sparse_path = sparse_vector_template.format(epoch=epoch, layer=layer)
            sparse_vector = get_steering_vector(layer, epoch, sparse_path)
            print(f'sparse_path: {sparse_path}')
            if sparse_vector is not None:
                print(f"Loaded sparse vector for epoch {epoch}")

        # Generate dense steering outputs
        if dense_vector is not None:
            # In distributed mode, avoid clobbering cache files by disabling save_path
            dense_outputs = generate_steering_outputs(
                model_dense, test_prompts, dense_vector, layer, multipliers,
                max_new_tokens, batch_size, save_results_path_dense if world_size == 1 else None,
                epoch, country_tag, "dense", verbose
            )
            for m, outputs in dense_outputs.items():
                results_data[epoch][f"dense_m{m}"] = outputs
        
        # Generate sparse steering outputs
        if sparse_vector is not None and model_sparse:
            sparse_outputs = generate_steering_outputs(
                model_sparse, test_prompts, sparse_vector, layer, multipliers,
                max_new_tokens, batch_size, save_results_path_sparse if world_size == 1 else None,
                epoch, country_tag, "sparse", verbose
            )
            for m, outputs in sparse_outputs.items():
                results_data[epoch][f"sparse_m{m}"] = outputs

    # In distributed mode, write shard results and let rank 0 merge + push
    # shard_dir = os.path.join("result", f"shards_{_slugify_leaf(model_name_or_path)}_L{layer}_{country_tag}")
    shard_path = sparse_vector_template.split("vector/")[-1]
    shard_dir = os.path.join("result", f"shards_{shard_path}")
    # print(f"shard_path: ", shard_path)
    # print(f"shard_dir: ", shard_dir)
    # print(f"sparse_vector_template: ", sparse_vector_template)
    if world_size > 1:
        os.makedirs(shard_dir, exist_ok=True)
        shard_payload = {
            "rank": rank,
            "keys": local_keys,
            "baseline": baseline_outputs,
            "results_data": {epoch: {k: v for k, v in ep_dict.items()} for epoch, ep_dict in results_data.items()},
        }
        shard_file = os.path.join(shard_dir, f"shard_{rank}.json")
        with open(shard_file, "w") as f:
            json.dump(shard_payload, f)
        print(f"Wrote shard results to {shard_file}")

    accelerator.wait_for_everyone()

    # Push results if requested (rank 0 merges shards when distributed)
    if push_repo and is_main:
        print(f"\nPreparing results for push to {push_repo}...")
        base_df = ds_country.to_pandas()
        key_col = prompt_field
        
        # Load existing dataset and get existing columns
        final_df, existing_columns = load_and_merge_existing_dataset(
            push_repo, push_split_name, base_df, key_col
        )
        
        # Gather per-shard results from disk when distributed
        gathered = []
        if world_size > 1:
            shard_files = [os.path.join(shard_dir, f"shard_{i}.json") for i in range(world_size)]
            for fp in shard_files:
                if not os.path.exists(fp):
                    raise FileNotFoundError(f"Missing shard file: {fp}")
                with open(fp, "r") as f:
                    gathered.append(json.load(f))
        else:
            gathered.append({
                "rank": 0,
                "keys": local_keys,
                "baseline": baseline_outputs,
                # Ensure JSON-serializable keys for epochs
                "results_data": {str(ep): {rt: outs for rt, outs in ep_dict.items()} for ep, ep_dict in results_data.items()}
            })

        # Build a stable mapping from prompt key -> outputs to avoid order misalignment
        key_series = base_df[key_col].astype(str)
        key_set = set(key_series.tolist())

        # Merge baseline
        baseline_col = f"baseline_L{layer}"
        if baseline_col not in existing_columns:
            baseline_map_total = {}
            for shard in gathered:
                for k, v in zip(shard["keys"], shard["baseline"]):
                    baseline_map_total[str(k)] = v
            # Map and ensure keys exist
            final_df[baseline_col] = final_df[key_col].astype(str).map(lambda k: baseline_map_total.get(k, None))

        # Merge steering outputs per epoch/type
        for epoch in epochs:
            # Build merged maps for this epoch
            merged_maps: Dict[str, Dict[str, str]] = defaultdict(dict)
            for shard in gathered:
                ep_dict = shard["results_data"].get(str(epoch)) if isinstance(shard["results_data"], dict) else shard["results_data"].get(epoch)
                if ep_dict is None:
                    continue
                for result_type, outputs in ep_dict.items():
                    for k, v in zip(shard["keys"], outputs):
                        merged_maps[result_type][str(k)] = v

            for result_type, out_map in merged_maps.items():
                col_name = f"{result_type}_ep{epoch}_L{layer}"
                if col_name in existing_columns:
                    print(f"Column {col_name} already exists, skipping")
                    continue
                final_df[col_name] = final_df[key_col].astype(str).map(lambda k: out_map.get(k, None))
                print(f"Added column: {col_name}")
        
        # Push to hub
        result_ds = Dataset.from_pandas(final_df, preserve_index=False)
        out_dict = DatasetDict({push_split_name: result_ds})
        
        print(f"Pushing results to {push_repo}...")
        out_dict.push_to_hub(
            push_repo,
            private=push_private,
            token=HUGGINGFACE_TOKEN,
            commit_message=push_commit_message,
        )
        print("Push completed successfully!")

    # Cleanup
    try:
        accelerator.end_training()
    except Exception:
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Efficient steering evaluation across multiple epochs")
    parser.add_argument("--behavior_dense", type=str, default="egypt_dense")
    parser.add_argument("--behavior_sparse", type=str, default="egypt_sparse")
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--multipliers", nargs="+", type=float, required=True)
    parser.add_argument("--epochs", nargs="+", type=int, required=True, help="List of epochs to process")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--dataset_repo", type=str, default="MBZUAI-Paris/Deep-Culture-Lense")
    parser.add_argument("--country_name", type=str, default="egypt",
                        help="Country filter (e.g., egypt, morocco).")
    parser.add_argument("--localization_status", type=str, default="both",
                        help="Filter by localization_status: localized|nolocalized|both (default: both)")
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--sae_repo", type=str, default="google/gemma-scope-2b-pt-res")
    parser.add_argument("--sae_width", type=str, default="65k")
    parser.add_argument("--sae_avg_idx", type=str, default="68")
    parser.add_argument("--dense_vector_template", type=str, 
                       help="Template for dense vector paths with {epoch} and {layer} placeholders")
    parser.add_argument("--sparse_vector_template", type=str,
                       help="Template for sparse vector paths with {epoch} and {layer} placeholders")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--push_repo", type=str, default=None)
    parser.add_argument("--push_private", action="store_true", default=True)
    parser.add_argument("--push_split_name", type=str, default="test")
    parser.add_argument("--push_commit_message", type=str, default="Add steering evaluation results")
    parser.add_argument("--mcq_eval", action="store_true", default=False)

    args = parser.parse_args()

    run_test_steering_on_hf(
        model_name_or_path=args.model_name_or_path,
        behavior_dense=args.behavior_dense,
        behavior_sparse=args.behavior_sparse,
        layer=args.layer,
        epochs=args.epochs,
        multipliers=args.multipliers,
        max_new_tokens=args.max_new_tokens,
        dataset_repo=args.dataset_repo,
        country=args.country_name,
        localization_status=args.localization_status,
        verbose=args.verbose,
        dense_vector_template=args.dense_vector_template,
        sparse_vector_template=args.sparse_vector_template,
        sae_repo=args.sae_repo,
        sae_width=args.sae_width,
        sae_avg_idx=args.sae_avg_idx,
        batch_size=args.batch_size,
        limit=args.limit,
        push_repo=args.push_repo,
        push_private=args.push_private,
        push_split_name=args.push_split_name,
        push_commit_message=args.push_commit_message,
        mcq_eval=args.mcq_eval,
    )
