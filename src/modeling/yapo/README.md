# YaPO Training Code

This folder contains implementation to train sparse steering vectors with **YaPO**, and dense ones with **BiPO**.

## 1. Environment setup

Make sure that your environment is ready

```
# AMD GPUs (MI210 were used in our experiments)
bash installation_amd.sh yapo
# NVIDIA GPUs
bash installation_nvidia.sh yapo
```

## 2. Training with Slurm

Main entry point: `src/modeling/yapo/train.sh`

```
cd ~/src/modeling/yapo
sbatch train.sh \
  --mode sparse \            # or --dense for BiPO
  --layer 15 \               # SAE insertion layer
  --country_name morocco \
  --per_device_train_batch_size 4 \
  --learning_rate 5e-4 \
  [extra args…]
```

Key notes:

* The script auto-detects the repo root, prepares `logs/modeling/train/*`, and
  loads env vars from `.env` if present (WANDB API key, etc.).
* By default it runs sparse steering with Gemma Scope SAEs. Use `--dense` or
  drop `--sparse` to train a dense steering vector.
* Adjust SAE parameters with `--sae_repo`, `--sae_vector_size`, `--sae_avg_idx`,
  or leave them blank to use the per-layer defaults.
* To disable Weights & Biases logging, unset `WANDB_API_KEY` or pass
  `--report_to none`.

Training logs live in `logs/modeling/train/`.

## 3. Generating with the learned

`run_all_tests.sh` orchestrates every steering run needed for evaluation. We first generate the outputs using the learned vectors, then run the evals in `src/evals/`

```
cd ~/yapo_clean/src/modeling/yapo
./run_all_tests.sh \
  --mode sparse \                 # matches train mode
  --layer 15 \
  --vector_dir ./vector/my_exp \
  --behavior my_behavior_tag
```

What it does:

1. Uses `test_steering.sh` to run each steering
   configuration (sparse or dense) and dump generations.
2. Writes outputs under `logs/modeling/test/*` plus `vector/<behavior>_*`.
3. Produces the datasets (`.jsonl` / `.parquet`) that `src/eval` consumes.

Typical workflow:

1. Train steering vectors (Section 2).
2. Run `./run_all_tests.sh` to generate responses for each steering setting.
3. Switch to `src/eval` and follow the local README to score the outputs.

## 4. Tips & troubleshooting

* `test_steering.sh` mirrors the training CLI. Pass `--help` to inspect flags.
* All Slurm scripts respect `--mode`, `--layer`, `--behavior`, dataset paths,
  and the localization filter (`--localization_status`).
* ROCm nodes require the environment variables already baked into `train.sh`
  (`HSA_OVERRIDE_GFX_VERSION`, `HIP_VISIBLE_DEVICES`, etc.); keep them when
  writing new scripts.
* Logs are verbose; tail the `.out` files in `logs/modeling/*` for issues.