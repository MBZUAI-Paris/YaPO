#!/bin/bash
#SBATCH --job-name=baseline-mmlu
#SBATCH --partition=hermes-2
#SBATCH --qos=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --time=10:00:00
#SBATCH --output=logs/hellaswag_baseline/%x_%j.out
#SBATCH --error=logs/hellaswag_baseline/%x_%j.err

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="${SCRIPT_DIR}"
PROJECT_ROOT="$(cd "${HARNESS_DIR}/../../.." && pwd)"
LOG_DIR="${HARNESS_DIR}/logs/hellaswag_baseline"
RUN_ROOT="${HARNESS_DIR}/runs/hellaswag_baseline"
mkdir -p "${LOG_DIR}" "${RUN_ROOT}"

# ---------------------------- User configuration --------------------------- #
TASKS=${TASKS:-"hellaswag"}
NUM_FEWSHOT=${NUM_FEWSHOT:-10}
BATCH_SIZE=${BATCH_SIZE:-8}
DEVICE=${DEVICE:-"cuda"}
LIMIT=${LIMIT:-""}

BASELINE_MODEL=${BASELINE_MODEL:-"google/gemma-2-2b-it"}
STEERING_MODEL_SIZE="2b" # 2b, 9b ${STEERING_MODEL_SIZE:-"2b"}
BASELINE_REVISION=${BASELINE_REVISION:-""}
BASELINE_DTYPE=${BASELINE_DTYPE:-"bfloat16"}
BASELINE_MODEL_ARGS=${BASELINE_MODEL_ARGS:-""}
MODEL_BACKEND=${MODEL_BACKEND:-"hf"}
APPLY_CHAT_TEMPLATE=1 #${APPLY_CHAT_TEMPLATE:-1}

if [[ "$STEERING_MODEL_SIZE" == "2b" ]]; then
  BATCH_SIZE=8
  BASELINE_MODEL="google/gemma-2-2b-it"
else
  BATCH_SIZE=1
  BASELINE_MODEL="google/gemma-2-9b-it"
fi

JOB_NAME=${JOB_NAME:-"baseline-${TASKS}"}
OUTPUT_NAME=${OUTPUT_NAME:-"baseline_run"}

scontrol update JobId="${SLURM_JOB_ID}" JobName="${JOB_NAME}" >/dev/null 2>&1 || true

# FIX for "unbound variable" errors in conda/shell profiles
export QT_XCB_GL_INTEGRATION=${QT_XCB_GL_INTEGRATION:-}
export PS1=${PS1:-}

# --- ROCm / MI210 Specific Environment ---
export HSA_OVERRIDE_GFX_VERSION=9.0.10
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
if [[ -n "${HIP_PRELOAD_PATH:-}" ]]; then
  if [[ -n "${LD_PRELOAD:-}" ]]; then
    export LD_PRELOAD="${HIP_PRELOAD_PATH}:${LD_PRELOAD}"
  else
    export LD_PRELOAD="${HIP_PRELOAD_PATH}"
  fi
fi

echo "[INFO] Activating conda env 'yapo'"
eval "$(conda shell.bash hook)"
conda activate yapo

# Verify Torch sees the GPUs
python - <<'PY'
import torch
available = torch.cuda.is_available()
print(f"[INFO] Torch version: {torch.__version__}")
print(f"[INFO] ROCm/HIP visibility: {available}")
if available:
    print(f"[INFO] GPU Model: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] Device Count: {torch.cuda.device_count()}")
else:
    print("[ERROR] NO GPU DETECTED. Check PyTorch/ROCm compatibility.")
    exit(1)
PY

cd "${HARNESS_DIR}"

model_args="pretrained=${BASELINE_MODEL},dtype=${BASELINE_DTYPE}"
if [[ -n "${BASELINE_REVISION}" ]]; then
  model_args+=",revision=${BASELINE_REVISION}"
fi
if [[ -n "${BASELINE_MODEL_ARGS}" ]]; then
  model_args+=",${BASELINE_MODEL_ARGS}"
fi

CMD=(python -m lm_eval run
  --model "${MODEL_BACKEND}"
  --model_args "${MODEL_ARGS:-${model_args}}"
  --tasks "${TASKS}"
  --device "${DEVICE}"
  --batch_size "${BATCH_SIZE}"
  --output_path "${RUN_ROOT}/${OUTPUT_NAME}/results.json"
)

if [[ -n "${NUM_FEWSHOT}" ]]; then
  CMD+=(--num_fewshot "${NUM_FEWSHOT}")
fi
if [[ -n "${LIMIT}" ]]; then
  CMD+=(--limit "${LIMIT}")
fi
if [[ "${APPLY_CHAT_TEMPLATE}" == "1" || "${APPLY_CHAT_TEMPLATE,,}" == "true" ]]; then
  CMD+=(--apply_chat_template)
fi

mkdir -p "${RUN_ROOT}/${OUTPUT_NAME}"

set -x
"${CMD[@]}"
