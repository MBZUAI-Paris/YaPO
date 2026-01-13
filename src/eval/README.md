# Evaluation Pipeline

This directory vendors a lightly customized copy of
[`lm-evaluation-harness`](lm-evaluation-harness/README.md) plus a few SLURM
wrappers that reproduce the metrics we rely on before every release. The goal is
to measure how the current YaPO checkpoints behave on tasks such as MMLU,
HellaSwag, and our internal steering evaluations.

We focus on two scenarios:

1. **Baseline runs** – report few-shot accuracy of an unmodified model.
2. **Steered runs** – load steering vectors (BiPO/CAA/SAS/etc.) and verify they
   move the model in the desired safety/alignment direction without harming
   accuracy.

## Layout

| Path | Purpose |
| --- | --- |
| `lm-evaluation-harness/` | Upstream harness plus local config/scripts. |
| `run_baseline.slurm` | SLURM entry point for baseline evaluations (defaults to Gemma-2). |
| `run_mmlu_steering.slurm` | Production steering job that sweeps steering modes, countries, and layers. |
| `working_run_mmlu_steering.slurm` | Sandbox steering job with relaxed SLURM/QoS settings for experimentation. |
| `logs/`, `runs/` | Captured stdout/err plus JSON summaries for each experiment. |

## Prerequisites

- Python environment (conda `yapo` by default) with ROCm/CUDA-compatible
  `torch`, and all dependencies from `lm-evaluation-harness/pyproject.toml`.
  Inside the harness folder, run `pip install -e .` to install the CLI.
- Hugging Face token with access to the evaluated models. Export `HF_TOKEN`.
- Access to the steering vector checkpoints. By default the SLURM scripts look
  for them under `lm-evaluation-harness/vectors/…`; override with
  `VECTOR_ROOT_*` env vars if they live elsewhere.
- Optional ROCm-specific tweaks: set `HIP_VISIBLE_DEVICES`, `HSA_OVERRIDE_*`,
  and `HIP_PRELOAD_PATH` if your cluster requires a custom HIP runtime.

## Quick start (local CLI)

```bash
cd src/eval/lm-evaluation-harness
export HF_TOKEN=hf_...
python -m lm_eval run \
  --model hf \
  --model_args pretrained=google/gemma-2-2b-it,dtype=bfloat16 \
  --tasks hellaswag \
  --num_fewshot 10 \
  --batch_size 8 \
  --output_path runs/baseline/demo/results.json \
  --apply_chat_template
```

This uses the vendored harness exactly as upstream: point it at any Hugging Face
model, pick a task suite, and inspect the JSON output under `runs/`.

## Running on SLURM

Both wrapper scripts assume you submit from `src/eval/lm-evaluation-harness`
(so the relative log paths resolve). They create run artifacts under `runs/`
and log to `logs/<job_type>/`. Feel free to delete those folders if you do not
want to track past experiments—Slurm will recreate them automatically.

### Baseline job

```bash
cd src/eval/lm-evaluation-harness
sbatch run_baseline.slurm \
  --export=ALL,BASELINE_MODEL=meta-llama/Llama-3.1-8B-Instruct,TASKS=mmlu,NUM_FEWSHOT=5
```

Key toggles (`run_baseline.slurm`):

- `BASELINE_MODEL`, `BASELINE_REVISION`, `BASELINE_DTYPE`, `MODEL_BACKEND`
  configure which weights to load and how to talk to them.
- `TASKS`, `NUM_FEWSHOT`, `LIMIT`, `BATCH_SIZE`, `APPLY_CHAT_TEMPLATE` mirror
  harness CLI arguments.
- ROCm knobs (`HIP_VISIBLE_DEVICES`, `HIP_PRELOAD_PATH`) let you adapt the run
  to your cluster without editing the script.

### Steering job

```bash
cd src/eval/lm-evaluation-harness
sbatch run_mmlu_steering.slurm \
  --export=ALL,STEERING_MODE=bipo,STEERING_COUNTRY=egypt,STEERING_MODEL_SIZE=9b
```

Important parameters (`run_mmlu_steering.slurm`):

- `STEERING_MODE`: `yapo`, `sas`, `bipo`, or `caa` presets. Each toggles the
  steering vector roots plus activation settings.
- `STEERING_MODEL_SIZE`: `2b` or `9b`, auto-adjusts generation batch sizes and
  the default steering layer (15 vs. 28).
- `STEERING_COUNTRY`, `STEERING_LOCALIZATION`, `STEERING_MCQ`, `STEERING_TAU`,
  `STEERING_MULTIPLIER`, and `STEERING_LAYER` let you sweep safety vectors.
- `VECTOR_ROOT_*` env vars point to the folders containing `.pt` steering files.
- Logs end up in `logs/mmlu_steer/` and metrics in
  `runs/<task>_steering/<size>/…/results.json`.

Use `working_run_mmlu_steering.slurm` for ad-hoc tests—it drops QoS constraints
and defaults to smaller jobs while keeping the same interface.

## Safety & hygiene checklist

- Keep tokens, steering vectors, and any sensitive run metadata outside of the
  repository. The scripts never echo `HF_TOKEN`, but verify your `runs/`
  artifacts before pushing a release.
- Override `HIP_PRELOAD_PATH` rather than hard-coding ROCm library paths.
- Remove personal log directories or model checkpoints before tagging a public
  release; only the templates and harness code should remain.
- When sharing metrics externally, drop paths from the JSON (the harness can be
  configured to emit relative metadata only).

With these pieces documented, reviewers can understand how to reproduce every
evaluation result and adjust the configs without digging through the harness.
