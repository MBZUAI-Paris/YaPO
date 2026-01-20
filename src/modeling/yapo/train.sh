#!/bin/bash
#SBATCH --job-name=yapo
#SBATCH --partition=hermes-1,hermes-2
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks-per-node=1      # One task per node
#SBATCH --cpus-per-task=32      # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --qos=ifm_others
#SBATCH --output=logs/modeling/train/%x_%j.out
#SBATCH --error=logs/modeling/train/%x_%j.err

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
find_repo_root() {
  local current="$1"
  while [[ "$current" != "/" ]]; do
    if [[ -d "${current}/.git" ]]; then
      echo "$current"
      return
    fi
    current="$(dirname "$current")"
  done
}
_repo_hint="${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}"
REPO_ROOT="$(find_repo_root "$_repo_hint")"
if [[ -z "${REPO_ROOT}" ]]; then
  REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
fi
TRAIN_DIR="${REPO_ROOT}/src/modeling/yapo"
ACCEL_CONFIG="${TRAIN_DIR}/training_config.yaml"
PY_SCRIPT="${TRAIN_DIR}/train.py"
mkdir -p "${REPO_ROOT}/logs/modeling/train"

# Load environment variables (including WANDB_API_KEY) from .env if present
if [[ -f ".env" ]]; then
  set -a; . ./.env; set +a
elif [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a; . "${REPO_ROOT}/.env"; set +a
fi

# Avoid CPU oversubscription by default; tune if needed.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

# --- ROCm / MI210 specific environment ---
# MI210 exposes gfx90a; forcing this avoids falling back to consumer HIP stacks.
export HSA_OVERRIDE_GFX_VERSION=9.0.10
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# LIBS_DIR="${LIBS_DIR:-${HOME}/verl_libs}"
# export LD_PRELOAD="${LIBS_DIR}/libamdhip64.so:${LD_PRELOAD:-}"
# -----------------------------------------


set -euo pipefail

cd "${TRAIN_DIR}"

# Defaults (can be overridden via flags below)
DATASET="MBZUAI-Paris/Deep-Culture-Lense"
MODE="sparse"                      # one of: dense|sparse
LOCALIZATION_STATUS="localized"   # one of: localized|nolocalized|both
MCQ_TRAINING=1
COUNTRY_NAME="morocco"
BEHAVIOR_BASE="${COUNTRY_NAME}"   # final behavior becomes "${BEHAVIOR_BASE}_${MODE}_${MODEL_SIZE}_${LAYER}_${LR}_${BATCH}_${EPOCHS}_${MCQ_TRAINING}"
MODEL_SIZE="2b"
LAYER=15
MODEL="google/gemma-2-${MODEL_SIZE}-it"
SAE_REPO="google/gemma-scope-${MODEL_SIZE}-pt-res"
SAE_SOURCE=""
LLAMA_SAE_SITE="R"
LLAMA_SAE_EXPANSION=8
LLAMA_SAE_CACHE_DIR=""
EPOCHS=20
BATCH=4
GRAD_ACC=1
GRAD_CKPT=0
LR=0.0005 #0.005 # 0.0005 # worked best so far: 0.0003
EXTRA_ARGS=()
DO_EVAL=0
WARMUP_STEPS="100"
WARMUP_RATIO=""
PROMPT_MAXLEN="512"

# Prompting and dataset schema defaults (override via flags)
if [[ "${MCQ_TRAINING}" == "1" ]]; then
  echo "Training with M.C.Q"
  # SYSTEM_PROMPT="You are an assistant solving MCQ. Respond in the language of the input. Output ONLY the final answer index inside \\boxed{index} followed by the correct answer from the MCQs and no extra text or explanation (e.g., \\boxed{2} : ...answer here...).}"
  SYSTEM_PROMPT=""
  PROMPT_COLNAME="prompt-mcq"
  CHOSEN_COLNAME="chosen-mcq"
  REJECTED_COLNAME="rejected_gemma_2_${MODEL_SIZE}_it_mcq"  # leave empty to auto-detect model-specific rejected columns
  MAXLEN=1024
  # Reserve some space for MCQ answers; override via --max_prompt_length if desired
  if [[ -z "${PROMPT_MAXLEN}" ]]; then PROMPT_MAXLEN=$(( MAXLEN - 32 )); fi
else
  echo "Training with O.E"
  SYSTEM_PROMPT=""
  PROMPT_COLNAME="prompt_open_ended"
  CHOSEN_COLNAME="chosen_open_ended"
  REJECTED_COLNAME="rejected_gemma_2_${MODEL_SIZE}_it_open_ended"  # leave empty to auto-detect model-specific rejected columns
  MAXLEN=512
  # Reserve space for longer open-ended answers
  if [[ -z "${PROMPT_MAXLEN}" ]]; then PROMPT_MAXLEN=$(( MAXLEN - 128 )); fi
fi


# Helper: set SAE params based on model size and layer
select_sae_params() {
  # for llama model
  if [[ "${SAE_SOURCE}" == "llama_scope" ]]; then
    SAE_VECTOR_SIZE=${SAE_VECTOR_SIZE:-"65k"}
    SAE_AVG_IDX=${SAE_AVG_IDX:-"68"}
    return
  fi
  # Normalize size inference from MODEL_SIZE, MODEL, or SAE_REPO
  local sz="${MODEL_SIZE}"
  local model_lc="${MODEL,,}"
  local sae_lc="${SAE_REPO,,}"
  local sz_lc="${sz,,}"
  if [[ -z "${sz_lc}" || ( "${sz_lc}" != "2b" && "${sz_lc}" != "9b" ) ]]; then
    if [[ "${model_lc}" == *"9b"* || "${sae_lc}" == *"9b"* ]]; then
      sz_lc="9b"
    else
      sz_lc="2b"
    fi
  fi

  if [[ "${sz_lc}" == "9b" ]]; then
    # 9B mapping
    case "${LAYER}" in
      25) SAE_VECTOR_SIZE="131k"; SAE_AVG_IDX="96" ;;
      26) SAE_VECTOR_SIZE="131k"; SAE_AVG_IDX="97" ;;
      27) SAE_VECTOR_SIZE="131k"; SAE_AVG_IDX="96" ;;
      28) SAE_VECTOR_SIZE="131k"; SAE_AVG_IDX="98" ;;
      29) SAE_VECTOR_SIZE="131k"; SAE_AVG_IDX="97" ;;
      30) SAE_VECTOR_SIZE="131k"; SAE_AVG_IDX="95" ;;
      *)
        SAE_VECTOR_SIZE=${SAE_VECTOR_SIZE:-"131k"}
        SAE_AVG_IDX=${SAE_AVG_IDX:-"98"}
        echo "[WARN] No 9B SAE params configured for layer ${LAYER}; using SAE_VECTOR_SIZE=${SAE_VECTOR_SIZE}, SAE_AVG_IDX=${SAE_AVG_IDX}" >&2
        ;;
    esac
  else
    # 2B mapping
    case "${LAYER}" in
      12) SAE_VECTOR_SIZE="65k"; SAE_AVG_IDX="72" ;;
      13) SAE_VECTOR_SIZE="65k"; SAE_AVG_IDX="75" ;;
      14) SAE_VECTOR_SIZE="65k"; SAE_AVG_IDX="73" ;;
      15) SAE_VECTOR_SIZE="65k"; SAE_AVG_IDX="68" ;;
      16) SAE_VECTOR_SIZE="65k"; SAE_AVG_IDX="69" ;;
      17) SAE_VECTOR_SIZE="65k"; SAE_AVG_IDX="68" ;;
      *)
        SAE_VECTOR_SIZE=${SAE_VECTOR_SIZE:-"65k"}
        SAE_AVG_IDX=${SAE_AVG_IDX:-"98"}
        echo "[WARN] No 2B SAE params configured for layer ${LAYER}; using SAE_VECTOR_SIZE=${SAE_VECTOR_SIZE}, SAE_AVG_IDX=${SAE_AVG_IDX}" >&2
        ;;
    esac
  fi
}

# Parse CLI flags
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sparse) MODE="sparse"; shift ;;
    --dense) MODE="dense"; shift ;;
    --mode) MODE="${2:-dense}"; shift 2 ;;
    --behavior-base) BEHAVIOR_BASE="${2}"; shift 2 ;;
    --layer) LAYER="${2}"; shift 2 ;;
    --dataset|--hub-dataset|--hub_dataset_path) DATASET="${2}"; shift 2 ;;
    --country-name|--country_name) COUNTRY_NAME="${2}"; BEHAVIOR_BASE="${COUNTRY_NAME}"; shift 2 ;;
    --model|--model-name|--model_name_or_path) MODEL="${2}"; shift 2 ;;
    --sae-repo|--sae_repo) SAE_REPO="${2}"; shift 2 ;;
    --sae-source|--sae_source) SAE_SOURCE="${2}"; shift 2 ;;
    --sae-vector-size|--sae_vector_size) SAE_VECTOR_SIZE="${2}"; shift 2 ;;
    --sae-avg-idx|--sae_avg_idx) SAE_AVG_IDX="${2}"; shift 2 ;;
    --llama-site|--llama_site) LLAMA_SAE_SITE="${2}"; shift 2 ;;
    --llama-expansion|--llama_expansion) LLAMA_SAE_EXPANSION="${2}"; shift 2 ;;
    --llama-cache-dir|--llama_cache_dir) LLAMA_SAE_CACHE_DIR="${2}"; shift 2 ;;
    --epochs|--num_train_epochs) EPOCHS="${2}"; shift 2 ;;
    --batch|--per_device_train_batch_size) BATCH="${2}"; shift 2 ;;
    --lr|--learning-rate|--learning_rate) LR="${2}"; shift 2 ;;
    --max-length|--max_length) MAXLEN="${2}"; shift 2 ;;
    --do-eval|--do_eval) DO_EVAL=1; shift ;;
    --localization-status|--localization_status) LOCALIZATION_STATUS="${2}"; shift 2 ;;
    --warmup-steps|--warmup_steps) WARMUP_STEPS="${2}"; shift 2 ;;
    --warmup-ratio|--warmup_ratio) WARMUP_RATIO="${2}"; shift 2 ;;
    # System prompt and dataset column names
    --system-prompt|--system_prompt) SYSTEM_PROMPT="${2}"; shift 2 ;;
    --prompt-colname|--prompt_colname) PROMPT_COLNAME="${2}"; shift 2 ;;
    --chosen-colname|--chosen_colname) CHOSEN_COLNAME="${2}"; shift 2 ;;
    --rejected-colname|--rejected_colname) REJECTED_COLNAME="${2}"; shift 2 ;;
    --) shift; EXTRA_ARGS=("$@"); break ;;
    *) echo "[ERR] Unknown option: $1" >&2; exit 1 ;;
  esac
done

if [[ -z "${SAE_SOURCE}" ]]; then
  model_lc="${MODEL,,}"
  repo_lc="${SAE_REPO,,}"
  if [[ "${model_lc}" == *"llama"* || "${repo_lc}" == *"llama"* ]]; then
    SAE_SOURCE="llama_scope"
  else
    SAE_SOURCE="gemma_scope"
  fi
fi

BEHAVIOR="${BEHAVIOR_BASE}_${MODE}_${MODEL_SIZE}_${LAYER}_${LR}_${BATCH}_${EPOCHS}_${MCQ_TRAINING}_loc-${LOCALIZATION_STATUS}"

# Choose SAE params after parsing flags
select_sae_params

echo "[INFO] SLURM job: ${SLURM_JOB_ID:-no-slurm} on $(hostname)"
echo "[INFO] Mode=${MODE} Behavior=${BEHAVIOR} Layer=${LAYER}"
echo "[INFO] Model=${MODEL} Dataset=${DATASET} SAE=${SAE_REPO} (${SAE_VECTOR_SIZE}, avg_idx=${SAE_AVG_IDX})"
echo "[INFO] SAE source=${SAE_SOURCE} (site=${LLAMA_SAE_SITE}, expansion=${LLAMA_SAE_EXPANSION}, cache='${LLAMA_SAE_CACHE_DIR}')"

echo "[INFO] Initializing conda"
if command -v conda >/dev/null 2>&1; then
  set +u
  eval "$(conda shell.bash hook)"
  conda deactivate || true
  echo "[INFO] Activating conda environment 'yapo'"
  conda activate yapo
  set -u
else
  echo "[WARN] 'conda' not found on PATH; assuming environment is already set"
fi

REPORT_BACKEND="wandb"
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "[WARN] WANDB_API_KEY not set; disabling Weights & Biases logging"
  REPORT_BACKEND="none"
  if [[ -z "${WANDB_MODE:-}" ]]; then
    export WANDB_MODE=disabled
  fi
else
  if [[ -z "${WANDB_MODE:-}" ]]; then
    export WANDB_MODE=online
  fi
fi

COMMON=(
  --report_to "${REPORT_BACKEND}"
  
  --layer "${LAYER}"
  --behavior "${BEHAVIOR}"
  --hub_dataset_path "${DATASET}"
  --country_name "${COUNTRY_NAME}"
  --model_name_or_path "${MODEL}"
  --sae_repo "${SAE_REPO}"
  --sae_vector_size "${SAE_VECTOR_SIZE}"
  --sae_avg_idx "${SAE_AVG_IDX}"
  --sae_source "${SAE_SOURCE}"
  --llama_sae_site "${LLAMA_SAE_SITE}"
  --llama_sae_expansion "${LLAMA_SAE_EXPANSION}"
  --num_train_epochs "${EPOCHS}"
  --per_device_train_batch_size "${BATCH}"
  --learning_rate "${LR}"
  --max_length "${MAXLEN}"
  --max_prompt_length "${PROMPT_MAXLEN}"
  --gradient_accumulation_steps "${GRAD_ACC}"
  # Prompting and schema
  --system_prompt "${SYSTEM_PROMPT}"
  --prompt_colname "${PROMPT_COLNAME}"
  --chosen_colname "${CHOSEN_COLNAME}"
)

if [[ -n "${LLAMA_SAE_CACHE_DIR}" ]]; then
  COMMON+=( --llama_sae_cache_dir "${LLAMA_SAE_CACHE_DIR}" )
fi

if [[ "$MODE" == "sparse" ]]; then
  COMMON+=( --sparse_steering )
fi

# Evaluation control (default: disabled; per-epoch when enabled)
if [[ "${DO_EVAL}" == "1" ]]; then
  COMMON+=( --do_eval --generate_during_eval )
fi

# Warmup preference: steps overrides ratio when both provided
if [[ -n "${WARMUP_STEPS}" ]]; then
  COMMON+=( --warmup_steps "${WARMUP_STEPS}" )
elif [[ -n "${WARMUP_RATIO}" ]]; then
  COMMON+=( --warmup_ratio "${WARMUP_RATIO}" )
fi

# Only pass rejected_colname if provided (non-empty) to allow auto-detect fallback
if [[ -n "${REJECTED_COLNAME}" ]]; then
  COMMON+=( --rejected_colname "${REJECTED_COLNAME}" )
fi

# Always pass localization filter (defaults to 'both' meaning no filter)
COMMON+=( --localization_status "${LOCALIZATION_STATUS}" )


if [[ "${GRAD_CKPT}" == "1" ]]; then
  COMMON+=( --gradient_checkpointing )
fi

set -x
accelerate launch --config_file "${ACCEL_CONFIG}" \
  "${PY_SCRIPT}" \
  "${COMMON[@]}" \
  "${EXTRA_ARGS[@]}"
set +x

echo "[INFO] Job complete"
