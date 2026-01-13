#!/bin/bash
#SBATCH --job-name=b-mmlu
#SBATCH --partition=hermes-2
#SBATCH --qos=test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --time=04:00:00
#SBATCH --output=logs/mmlu_steer/%x_%j.out
#SBATCH --error=logs/mmlu_steer/%x_%j.err

# Use -e (exit on error) and -o pipefail, but be careful with -u (unbound)
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HARNESS_DIR="${SCRIPT_DIR}"
PROJECT_ROOT="$(cd "${HARNESS_DIR}/../../.." && pwd)"
VECTORS_DIR="${HARNESS_DIR}/vectors"

# ---------------------------- User configuration --------------------------- #
TASKS=${TASKS:-"mmlu"}
NUM_FEWSHOT=${NUM_FEWSHOT:-5}
BATCH_SIZE=8
DEVICE=${DEVICE:-"cuda"}
LIMIT=${LIMIT:-""}

STEERING_MODE=${STEERING_MODE:-"bipo"} # yapo|sas|bipo|caa
JOB_NAME=${JOB_NAME:-"${STEERING_MODE}-${TASKS}"}
scontrol update JobId="${SLURM_JOB_ID}" JobName="${JOB_NAME}" >/dev/null 2>&1 || true
echo "[INFO] Steering preset: ${STEERING_MODE}"

# Shared steering toggles so most runs only need STEERING_* overrides
STEERING_MODEL_SIZE=${STEERING_MODEL_SIZE:-"2b"}
# make sure it's in lowercase
STEERING_MODEL_SIZE=${STEERING_MODEL_SIZE,,}
STEERING_COUNTRY=${STEERING_COUNTRY:-"egypt"}
STEERING_LOCALIZATION=${STEERING_LOCALIZATION:-"localized"}
STEERING_MCQ=${STEERING_MCQ:-0}
STEERING_TAU=${STEERING_TAU:-0.7}
STEERING_MULTIPLIER=${STEERING_MULTIPLIER:-1.0}
STEERING_LAYER=${STEERING_LAYER:-""}
STEERING_BEHAVIOR=${STEERING_BEHAVIOR:-"refusal"}
STEERING_RELU=${STEERING_RELU:-True}
EPOCH=${EPOCH:-19} # last epoch by default

# Shared vector roots for Gemma steering
VECTOR_ROOT_2B=${VECTOR_ROOT_2B:-"${VECTORS_DIR}/2b"}
VECTOR_ROOT_9B=${VECTOR_ROOT_9B:-"${VECTORS_DIR}/9b"}

VECTOR_ROOT_2B_CAA=${VECTOR_ROOT_2B_CAA:-"${VECTORS_DIR}/2b_caa"}
VECTOR_ROOT_2B_SAS=${VECTOR_ROOT_2B_SAS:-"${VECTORS_DIR}/2b_sas"}

VECTOR_ROOT_9B_CAA=${VECTOR_ROOT_9B_CAA:-"${VECTORS_DIR}/9b_caa"}
VECTOR_ROOT_9B_SAS=${VECTOR_ROOT_9B_SAS:-"${VECTORS_DIR}/9b_sas"}

# FIX for "unbound variable" errors in conda/shell profiles
export QT_XCB_GL_INTEGRATION=${QT_XCB_GL_INTEGRATION:-}
export PS1=${PS1:-} # Often another culprit in SLURM

LOG_DIR="${HARNESS_DIR}/logs/mmlu_steer"
RUN_ROOT="${HARNESS_DIR}/runs/mmlu_steering"
mkdir -p "${LOG_DIR}" "${RUN_ROOT}"

# --- ROCm / MI210 Specific Environment ---
# MI210 is gfx90a. 10.3.0 is for consumer cards and will hide the GPU from Torch.
export HSA_OVERRIDE_GFX_VERSION=9.0.10 
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Preload HIP library if supplied
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

# CRITICAL CHECK: Verify Torch sees the MI210s
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


# Helper utilities ---------------------------------------------------------- #
normalize_loc() {
  local loc="${1:-both}"
  loc="${loc,,}"
  loc="${loc#loc-}"
  echo "${loc}" 
}

get_model_path() {
  local size="${1,,}"
  case "${size}" in
    2b) echo "google/gemma-2-2b-it" ;;
    9b) echo "google/gemma-2-9b-it" ;;
    *)
      echo "[ERROR] Unsupported model size '${1}'" >&2
      return 1
      ;;
  esac
}

get_layer_idx_for_size() {
  local size="${1,,}"
  case "${size}" in
    2b) echo 15 ;;
    9b) echo 28 ;;
    *)
      echo "[ERROR] Unsupported model size '${1}'" >&2
      return 1
      ;;
  esac
}

get_lr_for_size() {
  local size="${1,,}"
  case "${size}" in
    2b|9b) echo 0.0005 ;;
    *)
      echo "[ERROR] Unsupported model size '${1}'" >&2
      return 1
      ;;
  esac
}

get_batch_for_size() {
  local size="${1,,}"
  case "${size}" in
    2b) echo 2 ;;
    9b) echo 1 ;;
    *)
      echo "[ERROR] Unsupported model size '${1}'" >&2
      return 1
      ;;
  esac
}

get_epochs_for_size() {
  local size="${1,,}"
  case "${size}" in
    2b|9b) echo 20 ;;
    *)
      echo "[ERROR] Unsupported model size '${1}'" >&2
      return 1
      ;;
  esac
}

get_root_for_size() {
  local size="${1,,}"
  case "${size}" in
    2b) echo "${VECTOR_ROOT_2B}" ;;
    9b) echo "${VECTOR_ROOT_9B}" ;;
    *)
      echo "[ERROR] Unsupported model size '${1}'" >&2
      return 1
      ;;
  esac
}

get_sae_repo_for_size() {
  local size="${1,,}"
  case "${size}" in
    2b) echo "google/gemma-scope-2b-pt-res" ;;
    9b) echo "google/gemma-scope-9b-pt-res" ;;
    *)
      echo "[ERROR] Unsupported model size '${1}'" >&2
      return 1
      ;;
  esac
}

get_sae_width_for_size() {
  local size="${1,,}"
  case "${size}" in
    2b) echo "65k" ;;
    9b) echo "131k" ;;
    *)
      echo "[ERROR] Unsupported model size '${1}'" >&2
      return 1
      ;;
  esac
}

get_sae_avg_for_size() {
  local size="${1,,}"
  case "${size}" in
    2b) echo "68" ;;
    9b) echo "97" ;;
    *)
      echo "[ERROR] Unsupported model size '${1}'" >&2
      return 1
      ;;
  esac
}

resolve_vector_dir() {
  local variant="$1"
  local size="${2,,}"
  local country="$3"
  local loc="$4"
  local mcq="$5"

  local layer lr batch epochs root suffix loc_tag country_lc
  layer=$(get_layer_idx_for_size "${size}") || return 1
  lr=$(get_lr_for_size "${size}") || return 1
  batch=$(get_batch_for_size "${size}") || return 1
  epochs=$(get_epochs_for_size "${size}") || return 1
  root=$(get_root_for_size "${size}") || return 1
  loc_tag=$(normalize_loc "${loc}")
  country_lc=$(echo "${country}" | tr '[:upper:]' '[:lower:]')
  suffix=$([[ "${mcq}" == "1" ]] && echo "gemma-mcq" || echo "gemma-oe")

  echo "${root}/${country_lc}_${variant}_${size}_${layer}_${lr}_${batch}_${epochs}_${mcq}_loc-${loc_tag}_${suffix}"
}

# --------------------- Steering presets & defaults ------------------------- #
DEFAULT_MODEL_ARGS=""

LAYER=$(get_layer_idx_for_size "${STEERING_MODEL_SIZE}") || exit 1
SAE_AVG_IDX=$(get_sae_avg_for_size "${STEERING_MODEL_SIZE}") || exit 1
SAE_WIDTH=$(get_sae_width_for_size "${STEERING_MODEL_SIZE}") || exit 1
SAE_MODEL=$(get_sae_repo_for_size "${STEERING_MODEL_SIZE}") || exit 1
pretrained_model=$(get_model_path "${STEERING_MODEL_SIZE}") || exit 1


case "${STEERING_MODE}" in
  yapo)
    vec_dir=$(resolve_vector_dir "sparse" "${STEERING_MODEL_SIZE}" "${STEERING_COUNTRY}" "${STEERING_LOCALIZATION}" "${STEERING_MCQ}") || exit 1
    vec_path=${vec_dir}/vec_ep${EPOCH}_layer${LAYER}.pt
    [[ -f "${vec_path}" ]] || { echo "[ERROR] YaPO vector not found at ${vec_path}"; exit 1; }
    DEFAULT_MODEL_ARGS="pretrained=${pretrained_model},steer_path=${vec_path},steering_layer=${LAYER},steering_space=sparse,steering_multiplier=${STEERING_MULTIPLIER},sae_repo=${SAE_MODEL},sae_vector_size=${SAE_WIDTH},sae_avg_idx=${SAE_AVG_IDX},steering_relu=${STEERING_RELU}"
    ;;
  bipo)
    vec_dir=$(resolve_vector_dir "dense" "${STEERING_MODEL_SIZE}" "${STEERING_COUNTRY}" "${STEERING_LOCALIZATION}" "${STEERING_MCQ}") || exit 1
    vec_path=${vec_dir}/vec_ep${EPOCH}_layer${LAYER}.pt
    [[ -f "${vec_path}" ]] || { echo "[ERROR] BiPO vector not found at ${vec_path}"; exit 1; }
    DEFAULT_MODEL_ARGS="pretrained=${pretrained_model},steer_path=${vec_path},steering_layer=${LAYER},steering_space=dense,steering_multiplier=${STEERING_MULTIPLIER},sae_repo=${SAE_MODEL},sae_vector_size=${SAE_WIDTH},sae_avg_idx=${SAE_AVG_IDX},steering_relu=${STEERING_RELU}"
    ;;
  sas)
    case "${STEERING_MODEL_SIZE}" in
      2b) SAS_VECTOR_ROOT="${VECTOR_ROOT_2B_SAS}" ;;
      9b) SAS_VECTOR_ROOT="${VECTOR_ROOT_9B_SAS}" ;;
      *)
        echo "[ERROR] No default SAS vector root for model size '${STEERING_MODEL_SIZE}'. Set SAS_VECTOR_ROOT." >&2
        exit 1
        ;;
    esac
    SAS_VECTOR_PATH="${SAS_VECTOR_ROOT}/sas_${STEERING_COUNTRY}_${STEERING_LOCALIZATION}_mcq${STEERING_MCQ}_L${LAYER}_tau${STEERING_TAU}_mul_${STEERING_MULTIPLIER}_steer_mode_sas.pt"
    [[ -f "${SAS_VECTOR_PATH}" ]] || { echo "[ERROR] SAS vector not found at ${SAS_VECTOR_PATH}"; exit 1; }
    DEFAULT_MODEL_ARGS="pretrained=${pretrained_model},steer_path=${SAS_VECTOR_PATH},steering_layer=${LAYER},steering_space=sparse,steering_multiplier=${STEERING_MULTIPLIER},sae_repo=${SAE_MODEL},sae_vector_size=${SAE_WIDTH},sae_avg_idx=${SAE_AVG_IDX},steering_relu=${STEERING_RELU}"
    ;;
  caa)
    case "${STEERING_MODEL_SIZE}" in
      2b) CAA_VECTOR_ROOT="${VECTOR_ROOT_2B_CAA}" ;;
      9b) CAA_VECTOR_ROOT="${VECTOR_ROOT_9B_CAA}" ;;
      *)
        echo "[ERROR] No default CAA vector root for model size '${STEERING_MODEL_SIZE}'. Set CAA_VECTOR_ROOT." >&2
        exit 1
        ;;
    esac
    CAA_VECTOR_PATH="${CAA_VECTOR_ROOT}/vanilla_${STEERING_COUNTRY}_${STEERING_LOCALIZATION}_mcq${STEERING_MCQ}_L${LAYER}_tau${STEERING_TAU}_mul_${STEERING_MULTIPLIER}_steer_mode_vanilla.pt"
    [[ -f "${CAA_VECTOR_PATH}" ]] || { echo "[ERROR] CAA vector not found at ${CAA_VECTOR_PATH}"; exit 1; }
    DEFAULT_MODEL_ARGS="pretrained=${pretrained_model},steer_path=${CAA_VECTOR_PATH},steering_layer=${LAYER},steering_space=dense,steering_multiplier=${STEERING_MULTIPLIER},sae_repo=${SAE_MODEL},sae_vector_size=${SAE_WIDTH},sae_avg_idx=${SAE_AVG_IDX},steering_relu=${STEERING_RELU}"
    ;;
esac

MODEL_ARGS=${MODEL_ARGS:-${DEFAULT_MODEL_ARGS}}
ENABLE_PARALLELIZE=${ENABLE_PARALLELIZE:-""}

if [[ -n "${ENABLE_PARALLELIZE}" ]]; then
  MODEL_ARGS="${MODEL_ARGS},parallelize=${ENABLE_PARALLELIZE}"
fi

OUTPUT_NAME=${OUTPUT_NAME:-"manual_run"}

# --------------------------------------------------------------------------- #

echo "[INFO] Running lm-eval with output: ${OUTPUT_NAME}"

CMD=(python -m lm_eval run
  --model steered
  --model_args "${MODEL_ARGS}"
  --tasks "${TASKS}"
  --device "${DEVICE}"
  --batch_size "${BATCH_SIZE}"
  --output_path "${RUN_ROOT}/${OUTPUT_NAME}/results.json"
)

[[ -n "${NUM_FEWSHOT}" ]] && CMD+=(--num_fewshot "${NUM_FEWSHOT}")
[[ -n "${LIMIT}" ]] && CMD+=(--limit "${LIMIT}")

mkdir -p "${RUN_ROOT}/${OUTPUT_NAME}"

set -x
"${CMD[@]}"
