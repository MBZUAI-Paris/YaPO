# Data Preparation Pipeline

This folder contains the scripts that turn an aligned dataset of prompts and
preferred answers into a Direct Preference Optimization (DPO) style dataset.
We assume the current chat model is still flawed: we re-run the prompts through
that on-policy model, treat the generated answers as the **rejected** responses,
and pair them with the trusted answers from the original dataset. The resulting
dataset is then uploaded to the Hugging Face Hub for further training.

## Components

| File | Purpose |
| --- | --- |
| `generate_rejected.py` | Core generator that loads a dataset, samples model completions, and pushes checkpoints to the Hub. |
| `generate_rejected.sh` | SLURM-friendly wrapper around the Python script with sensible defaults, logging, and conda activation. |
| `utils.py` | Helper functions for batching, augmentation, ShareGPT conversion, and Hub uploads. |
| `inference_config.yaml` | Accelerate/DeepSpeed config used when launching distributed jobs. |

## Prerequisites

- Python environment with `transformers`, `datasets`, `accelerate`, `torch`,
  and `python-dotenv`. (Activate the `yapo` conda env or replicate it.)
- Hugging Face access token with read/write access to the source dataset and
  the destination repo. Export it as `HF_TOKEN` or place it in a local `.env`.
- Valid dataset on the Hub that contains at least a prompt column and a chosen
  (preferred) answer column.
- Optional: SLURM cluster if you want to use the batch script. Locally you can
  run the Python script directly.

## Running the generator

### Quick start (Python)

```bash
cd src/data_prep
export HF_TOKEN=hf_...
accelerate launch --config_file inference_config.yaml \
  generate_rejected.py \
  --on_policy_model_name meta-llama/Llama-3.1-8B-Instruct \
  --data_path MBZUAI-Paris/Deep-Culture-Lense \
  --save_data_path HF_USER/HF_REPO_WITH_REJECTED_SAMPLES \
  --prompt_colname prompt-mcq \
  --chosen_colname chosen-mcq \
  --checkpoint_freq 5
```

This streams the prompts through the selected model, treats the generated text as
`rejected_<model_name_safe>`, and periodically pushes private checkpoints to the
specified Hub dataset `--save_data_path` (Required).

### Running via SLURM

```bash
cd src/data_prep
sbatch generate_rejected.sh --mcq --model-size 2b --start-index 0 --end-index 999
```

Key features of the wrapper:

- Resolves script paths relative to the repo (no hard-coded personal paths).
- Logs to `logs/on_policy/<job>_<id>.{out,err}` inside the repository.
- Allows overriding model, dataset, token counts, prompt columns, etc.
- Automatically selects a Hugging Face repo name based on dataset type (Arabic
  vs. multilingual, MCQ vs. open-ended) unless explicitly provided.

Use `sbatch generate_rejected.sh --help` for the full list of flags. Each CLI
flag corresponds to the arguments in `generate_rejected.py`.

## Understanding the dataset schema

- `prompt_colname` and `chosen_colname` refer to the preferred data you already
  trust (e.g., MCQ solutions or high-quality open-ended answers).
- The script derives `rejected_<model_slug>` columns from the model outputs.
- Metadata (dataset/id/generation params) is preserved in the `metadata` column
  but stripped before uploading, so the Hub dataset mirrors the original schema
  plus the new rejected column.
- Set `--open_ended` if your dataset is conversational/open-ended; otherwise
  the defaults assume MCQ formatting.
- Use `--system_prompt` to inject formatting instructions (e.g., instructing a
  model to output `\boxed{index}` for MCQs).

## Notes

- Keep the target Hugging Face repo private unless you intend to publish the
  generated pairs; the script invokes `push_to_hub(..., private=True)` but you
  should double-check repo permissions.