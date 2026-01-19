# YaPO: Learnable Sparse Activation Steering Vectors

<p align="center">
  <a href="https://arxiv.org/abs/2601.08441">
    <img src="https://img.shields.io/badge/Paper-arXiv:2601.08441-B31B1B?logo=arxiv&logoColor=white" alt="Paper"/>
  </a>
  <a href="https://huggingface.co/datasets/MBZUAI-Paris/Deep-Culture-Lense">
    <img src="https://img.shields.io/badge/Dataset-HF%20Deep--Culture--Lense-ffad33?logo=huggingface&logoColor=white" alt="Dataset"/>
  </a>
  <a href="https://github.com/MBZUAI-Paris/YaPO/stargazers">
    <img src="https://img.shields.io/github/stars/MBZUAI-Paris/YaPO?label=Stars&logo=github" alt="GitHub stars"/>
  </a>
  <a href="https://mbzuai-paris.github.io/YaPO">
    <img src="https://img.shields.io/badge/Project%20Page-YaPO-green?logo=google-chrome&logoColor=white" alt="Project page"/>
  </a>
  <a href="https://wandb.ai/ahmad-chamma-mbzuai/Alignement">
    <img src="https://img.shields.io/badge/W%26B-Alignement-yellow?logo=weightsandbiases&logoColor=white" alt="Weights & Biases project"/>
  </a>
</p>

YaPO is a steering algorithm  instruction-tuned LLMs toward region- or domain-specific behaviors by learning **sparse activation vectors** on top of pretrained and frozen LLMs using Sparse AutoEncoders (SAEs). 

![Method diagram](assets/method.png)


This repository contains:

| Folder | Description |
| --- | --- |
| `src/data_prep/` | Scripts that regenerate “rejected” answers with the current model to build DPO-style ready datasets. |
| `src/modeling/yapo/` | Training, steering, and testing code for sparse/dense vectors (Slurm-friendly). |
| `src/eval/` | Evaluation code and adaptation of `lm-evaluation-harness` plus Slurm wrappers to benchmark baseline and steered models. |
| `assets/` | Figures (e.g., `method.png` above) and sample training logs for documentation. |
| `installation_*.sh` | Environment bootstrap scripts for AMD (ROCm) and NVIDIA clusters. |

The rest of the README walks through the complete workflow.


## 1. Environment setup

Pick the script that matches your hardware; each creates a Conda env with all
dependencies (`Torch`, `accelerate`, `trl`, etc.).

```bash
# AMD ROCm (MI210 were used in our experiments)
bash installation_amd.sh yapo

# NVIDIA CUDA
bash installation_nvidia.sh yapo

conda activate yapo  # or whichever env name you chose
```



## 2. End-to-end workflow

1. **Generate rejected answers (data prep).**  
   `src/data_prep/generate_rejected.py` streams your prompt dataset through the
   current policy model, treats its outputs as rejected responses, and pairs
   them with the trusted “chosen” answers. Use the shell wrapper for clusters:
   ```bash
   cd src/data_prep
   sbatch generate_rejected.sh \
     --data_path MBZUAI-Paris/Deep-Culture-Lense \
     --model-name google/gemma-2-9b-it \
     --mcq
   ```
   The script writes logs to `logs/on_policy/` and uploads the augmented dataset
   to your Hugging Face repo (see `src/data_prep/README.md` for details).

2. **Train steering vectors (modeling).**  
   Launch `sbatch src/modeling/yapo/train.sh` with the desired mode (`--sparse`
   for YaPO, `--dense` for BiPO), layer, dataset, and SAE parameters:
   ```bash
   cd src/modeling/yapo
   sbatch train.sh \
     --mode sparse \
     --layer 15 \
     --hub_dataset_path MBZUAI-Paris/Deep-Culture-Lense_processed_2b_mcq_mxlen_1024 \
     --country_name morocco \
     --per_device_train_batch_size 4 \
     --learning_rate 5e-4
   ```
   Logs land in `logs/modeling/train/` and steering vectors in `vector/<behavior>_*`.

3. **Generate evaluation traces.**  
   Run `./run_all_tests.sh` to sweep the trained steering vectors (or baselines)
   and produce `.jsonl`/`.parquet` files consumed by the eval stage:
   ```bash
   ./run_all_tests.sh \
     --mode sparse \
     --layer 15 \
     --behavior morocco_sparse_2b_15_0.0005_4_20_1_loc-localized \
     --vector_dir ./vector/morocco_sparse_2b_15_0.0005_4_20_1_loc-localized_gemma-mcq
   ```

4. **Evaluate accuracy & general knowledge.**  
   Move into `src/eval/lm-evaluation-harness` and submit either the baseline
   script or the steering sweep:
   ```bash
   cd src/eval/lm-evaluation-harness
   sbatch run_baseline.slurm \
     --export=ALL,BASELINE_MODEL=google/gemma-2-2b-it,TASKS=mmlu,NUM_FEWSHOT=5

   sbatch run_mmlu_steering.slurm \
     --export=ALL,STEERING_MODE=yapo,STEERING_COUNTRY=morocco,STEERING_MODEL_SIZE=2b
   ```
   Results appear under `runs/`, logs under `logs/`. The harness README explains
   all env vars and task settings.


## 3. Helpful references

* **Data prep README:** column schema, Hugging Face upload tips, and Slurm flags.
* **Modeling README:** clarifies ROCm env variables, steering CLI options, and
  troubleshooting notes for `train.sh`, `test_steering.sh`, and `run_all_tests.sh`.
* **Eval README:** explains how the vendored `lm-evaluation-harness` is wired
  into Slurm jobs for baseline vs. steering experiments.

All three documents live in their respective folders. Refer back to them for the
complete flag list and failure modes. Do not hesitate to open an issue if any.


## 4. Notes

* The `assets/` directory holds the method figure and sample training curves.
* All scripts look for secrets via a local `.env` (WANDB, HF tokens). Keep that
  file out of version control.


## 5. Troubleshooting & tips

* ROCm clusters require the system exports already present in `train.sh`
  (`HSA_OVERRIDE_GFX_VERSION`, `HIP_VISIBLE_DEVICES`, `LD_PRELOAD`). Reuse those
  lines when writing new job scripts.
* WandB is optional; set `WANDB_API_KEY` to enable logging or leave it empty to
  run offline.
* Steering configs are named
  `COUNTRY_MODELSIZE_LAYER_LR_BATCH_EPOCHS_MCQ_loc-STATUS`—use the same slug
  everywhere (training, testing, eval) to avoid mismatches, or make sure to take care of it.

With the environment ready and the steps above, you can reproduce every YaPO result: regenerate data, train the steering vectors, produce test generations, and benchmark them all while reusing the provided scripts.
