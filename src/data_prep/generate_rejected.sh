#!/usr/bin/env bash
#SBATCH --job-name=yapo-on-policy
#SBATCH --partition=hermes-1,hermes-2
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks-per-node=1      # One task per node
#SBATCH --cpus-per-task=128      # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --time=120:00:00
#SBATCH --output=logs/on_policy/%x_%j.out
#SBATCH --error=logs/on_policy/%x_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
mkdir -p "${REPO_ROOT}/logs/on_policy"

# Allow overriding threading via env; default to a reasonable value so torch
# doesn't clamp to 1 thread per process when launching distributed jobs.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

# Defaults (override via CLI flags below)
USE_ACCELERATE=1
ACCEL_CONFIG="${SCRIPT_DIR}/inference_config.yaml"
CONDA_ENV="yapo"
USE_CONDA=1

MODEL_FAMILY="gemma" # "gemma"
MULTILINGUAL=1 # 0 for arabic dataset
OPEN_ENDED=1  # 0=MCQ mode, 1=Open-ended mode
MODEL_SIZE="${MODEL_SIZE:-8b}"
ON_POLICY_MODEL="${ON_POLICY_MODEL:-}"
MODEL_SLUG=""

BATCH_SIZE=128
BATCH_SIZE_OVERRIDE=0

SYSTEM_PROMPT=""
ARABIC_DATA_PATH="Alignement/Arabic_Cultural_Dataset_MCQ"
MULTILINGUAL_DATA_PATH="Alignement/Multilingual_Cultural_Dataset_MCQ_flattened"
SAVE_PREFIX_ARABIC="Alignement/Arabic_cultural_dataset_processed"
SAVE_PREFIX_MULTI="Alignement/Multilingual_cultural_dataset_processed"

DATA_PATH="${ARABIC_DATA_PATH}"
DATA_PATH_OVERRIDE=0
SAVE_DATA_PATH=""
SAVE_PATH_OVERRIDE=0
PROMPT_COLNAME=""
CHOSEN_COLNAME=""
MAX_NEW_TOKENS=""


# Index range and checkpointing
START_INDEX=0
END_INDEX=-1
CHECKPOINT_FREQ=5

# Optional augmentation
DO_AUGMENT=0

EXTRA_ARGS=()

declare -A MODEL_PRESETS=(
  ["gemma:2b"]="google/gemma-2-2b-it"
  ["gemma:9b"]="google/gemma-2-9b-it"
  ["gemma:27b"]="google/gemma-2-27b-it"
  ["llama:1b"]="meta-llama/Llama-3.2-1B-Instruct"
  ["llama:3b"]="meta-llama/Llama-3.2-3B-Instruct"
  ["llama:8b"]="meta-llama/Llama-3.1-8B-Instruct"
  ["llama:70b"]="meta-llama/Llama-3.1-70B-Instruct"
  ["llama:405b"]="meta-llama/Llama-3.1-405B-Instruct"
  ["qwen:7b"]="Qwen/Qwen2.5-7B-Instruct"
  ["qwen:14b"]="Qwen/Qwen2.5-14B-Instruct"
  ["qwen:32b"]="Qwen/Qwen2.5-32B-Instruct"
)
DEFAULT_MODEL_FAMILY="llama"
DEFAULT_MODEL_SIZE="8b"
DEFAULT_MODEL_KEY="${DEFAULT_MODEL_FAMILY}:${DEFAULT_MODEL_SIZE}"

print_help() {
  cat <<EOF
Usage: $(basename "$0") [options] [-- extra-args]

Options:
  --no-accelerate                 Run with python instead of accelerate
  --accelerate-config <path>      Path to accelerate config (default: ${ACCEL_CONFIG})
  --no-conda                      Do not attempt to activate a conda env
  --conda-env <name>              Conda environment name (default: ${CONDA_ENV})

  --model <name>                  HF model for on-policy generation (default resolved via presets)
  --model-family <name>           Model family for preset lookup (default: ${MODEL_FAMILY})
  --model-size <tag>              Logical model size tag (default: ${MODEL_SIZE})
  --batch-size <int>              Batch size (default auto based on model size)
  --max-new-tokens <int>          Max new tokens (default auto per mode)

  --data-path <repo>              HF dataset repo to read (default: ${DATA_PATH})
  --save-data-path <repo>         HF dataset repo to push (default auto per mode)
  --multilingual                  Use the multilingual dataset defaults
  --arabic                        Use the Arabic dataset defaults (default)
  --start-index <int>             Start index (default: ${START_INDEX})
  --end-index <int>               End index, -1 for full (default: ${END_INDEX})
  --checkpoint-freq <int>         Checkpoint frequency in batches (default: ${CHECKPOINT_FREQ})

  --open-ended                    Use open-ended prompts/answers (sets column names accordingly)
  --mcq                           Use MCQ prompts/answers
  --prompt-colname <name>         Override prompt column name
  --chosen-colname <name>         Override chosen column name
  --system-prompt <text>          System prompt to inject (quote as needed)
  --do-augment                    Enable augmented generations

Examples:
  $0 --model-family gemma --model-size 9b --mcq \
     --data-path Alignement/Arabic_Cultural_Dataset_MCQ \
     --save-data-path Alignement/Arabic_cultural_dataset_processed_9b_mcq

  $0 --open-ended --model meta-llama/Llama-3.1-8B-Instruct \
     --data-path Alignement/Arabic_cultural_dataset_processed_2b_mxlen_512 \
     --prompt-colname prompt_open_ended --chosen-colname chosen_open_ended
EOF
}

normalize_model_family() {
  local fam="${1:-}"
  fam="${fam,,}"
  fam="${fam// /}"
  fam="${fam//-/_}"
  if [[ -z "${fam}" || "${fam}" == "auto" ]]; then
    echo ""
  else
    echo "${fam}"
  fi
}

normalize_model_size() {
  local size="${1:-}"
  size="${size,,}"
  size="${size// /}"
  size="${size//-/_}"
  if [[ -z "${size}" || "${size}" == "auto" ]]; then
    echo ""
    return
  fi
  if [[ "${size}" =~ ^([0-9]+)([bm])$ ]]; then
    echo "${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
    return
  fi
  if [[ "${size}" =~ ^([0-9]+)$ ]]; then
    echo "${BASH_REMATCH[1]}b"
    return
  fi
  echo "${size}"
}

infer_model_family_from_id() {
  local model_id="${1,,}"
  for candidate in llama gemma qwen mistral mixtral phi; do
    if [[ "${model_id}" == *"${candidate}"* ]]; then
      echo "${candidate}"
      return
    fi
  done
  echo ""
}

infer_model_size_from_id() {
  local model_id="${1,,}"
  if [[ "${model_id}" =~ ([0-9]{1,3})([bm]) ]]; then
    echo "${BASH_REMATCH[1]}${BASH_REMATCH[2]}"
    return
  fi
  echo ""
}

sanitize_token() {
  local raw="${1:-}"
  raw="${raw,,}"
  raw="${raw//\//_}"
  raw=$(printf '%s\n' "${raw}" | sed -E 's/[^a-z0-9]+/_/g')
  raw=$(printf '%s\n' "${raw}" | sed -E 's/_+/_/g; s/^_|_$//g')
  echo "${raw}"
}

derive_model_slug() {
  local primary="${1:-}"
  local fallback="${2:-}"
  local slug=""
  if [[ -n "${primary}" ]]; then
    slug="$(sanitize_token "${primary}")"
  fi
  if [[ -z "${slug}" && -n "${fallback}" ]]; then
    slug="$(sanitize_token "${fallback}")"
  fi
  echo "${slug}"
}

derive_default_batch_size() {
  local size="${1,,}"
  local digits="${size//[!0-9]/}"
  if [[ -z "${digits}" ]]; then
    echo "128"
    return
  fi
  local value=$((10#${digits}))
  if (( value <= 3 )); then
    echo "256"
  elif (( value <= 9 )); then
    echo "128"
  elif (( value <= 34 )); then
    echo "64"
  else
    echo "32"
  fi
}

resolve_model_defaults() {
  local normalized_family
  normalized_family="$(normalize_model_family "${MODEL_FAMILY}")"
  local normalized_size
  normalized_size="$(normalize_model_size "${MODEL_SIZE}")"

  if [[ -z "${ON_POLICY_MODEL}" ]]; then
    if [[ -n "${normalized_family}" && -n "${normalized_size}" ]]; then
      local key="${normalized_family}:${normalized_size}"
      if [[ -n "${MODEL_PRESETS[$key]+isset}" ]]; then
        ON_POLICY_MODEL="${MODEL_PRESETS[$key]}"
      fi
    fi

    if [[ -z "${ON_POLICY_MODEL}" ]]; then
      ON_POLICY_MODEL="${MODEL_PRESETS[$DEFAULT_MODEL_KEY]}"
      if [[ -n "${normalized_family}" || -n "${normalized_size}" ]]; then
        echo "[WARN] No preset found for family='${MODEL_FAMILY}' size='${MODEL_SIZE}'. Falling back to ${ON_POLICY_MODEL}" >&2
      fi
      normalized_family="${DEFAULT_MODEL_FAMILY}"
      normalized_size="${DEFAULT_MODEL_SIZE}"
    fi
  fi

  if [[ -z "${normalized_family}" ]]; then
    normalized_family="$(infer_model_family_from_id "${ON_POLICY_MODEL}")"
  fi
  if [[ -z "${normalized_size}" ]]; then
    normalized_size="$(normalize_model_size "$(infer_model_size_from_id "${ON_POLICY_MODEL}")")"
  fi

  MODEL_FAMILY="${normalized_family}"
  MODEL_SIZE="${normalized_size}"
  MODEL_SLUG="$(derive_model_slug "${MODEL_SIZE}" "${ON_POLICY_MODEL}")"
  if [[ -z "${MODEL_SLUG}" ]]; then
    MODEL_SLUG="model"
  fi
}

set_prompt_defaults() {
  local mode=""
  if [[ "${DATA_PATH_OVERRIDE}" == "0" ]]; then
    if [[ "${MULTILINGUAL}" == "1" ]]; then
      DATA_PATH="${MULTILINGUAL_DATA_PATH}"
    else
      DATA_PATH="${ARABIC_DATA_PATH}"
    fi
  fi
  if [[ "${OPEN_ENDED}" == "1" ]]; then
    PROMPT_COLNAME="${PROMPT_COLNAME:-prompt_open_ended}"
    CHOSEN_COLNAME="${CHOSEN_COLNAME:-chosen_open_ended}"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
    mode="oe"
  else
    PROMPT_COLNAME="${PROMPT_COLNAME:-prompt-mcq}"
    CHOSEN_COLNAME="${CHOSEN_COLNAME:-chosen-mcq}"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
    mode="mcq"
  fi

  if [[ -z "${SAVE_DATA_PATH}" ]]; then
    local prefix="${SAVE_PREFIX_ARABIC}"
    if [[ "${MULTILINGUAL}" == "1" ]]; then
      prefix="${SAVE_PREFIX_MULTI}"
    fi
    SAVE_DATA_PATH="${prefix}_${MODEL_FAMILY}_${MODEL_SIZE}_${mode}_mxlen_${MAX_NEW_TOKENS}"
  fi
}

#
# Parse CLI flags
#
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) print_help; exit 0 ;;
    --no-accelerate) USE_ACCELERATE=0; shift ;;
    --accelerate-config) ACCEL_CONFIG="${2}"; shift 2 ;;
    --no-conda) USE_CONDA=0; shift ;;
    --conda-env) CONDA_ENV="${2}"; shift 2 ;;

    --model|--on-policy-model|--on_policy_model_name) ON_POLICY_MODEL="${2}"; shift 2 ;;
    --model-family|--model_family) MODEL_FAMILY="${2}"; shift 2 ;;
    --model-size|--model_size) MODEL_SIZE="${2}"; shift 2 ;;
    --batch-size|--batch_size) BATCH_SIZE="${2}"; BATCH_SIZE_OVERRIDE=1; shift 2 ;;
    --max-new-tokens|--max_new_tokens) MAX_NEW_TOKENS="${2}"; shift 2 ;;

    --data-path|--data_path) DATA_PATH="${2}"; DATA_PATH_OVERRIDE=1; shift 2 ;;
    --save-data-path|--save_data_path) SAVE_DATA_PATH="${2}"; SAVE_PATH_OVERRIDE=1; shift 2 ;;
    --multilingual) MULTILINGUAL=1; shift ;;
    --arabic) MULTILINGUAL=0; shift ;;
    --start-index|--start_index) START_INDEX="${2}"; shift 2 ;;
    --end-index|--end_index) END_INDEX="${2}"; shift 2 ;;
    --checkpoint-freq|--checkpoint_freq) CHECKPOINT_FREQ="${2}"; shift 2 ;;

    --open-ended|--open_ended) OPEN_ENDED=1; shift ;;
    --mcq) OPEN_ENDED=0; shift ;;
    --prompt-colname|--prompt_colname) PROMPT_COLNAME="${2}"; shift 2 ;;
    --chosen-colname|--chosen_colname) CHOSEN_COLNAME="${2}"; shift 2 ;;
    --system-prompt|--system_prompt) SYSTEM_PROMPT="${2}"; shift 2 ;;
    --do-augment|--do_augment) DO_AUGMENT=1; shift ;;

    --) shift; EXTRA_ARGS=("$@" ); break ;;
    *) echo "[ERR] Unknown option: $1" >&2; exit 1 ;;
  esac
done

resolve_model_defaults

if [[ "${BATCH_SIZE_OVERRIDE}" == "0" ]]; then
  BATCH_SIZE="$(derive_default_batch_size "${MODEL_SIZE}")"
fi

set_prompt_defaults

#
# Environment activation (optional)
#
if [[ "${USE_CONDA}" == "1" ]]; then
  if command -v conda >/dev/null 2>&1; then
    echo "Initializing conda"
    set +u  # some activate.d scripts rely on unset vars
    eval "$(conda shell.bash hook)"
    conda deactivate || true
    echo "Activating conda environment '${CONDA_ENV}'"
    conda activate "${CONDA_ENV}"
    set -u
  else
    echo "[WARN] 'conda' not found on PATH; skipping activation" >&2
  fi
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "[INFO] SLURM job ${SLURM_JOB_ID} on $(hostname)"
else
  echo "[INFO] Running interactively on $(hostname)"
fi

echo "[INFO] On-policy model: ${ON_POLICY_MODEL} (family=${MODEL_FAMILY:-unknown}, size=${MODEL_SIZE:-unknown})"
echo "[INFO] Mode: $([[ "${OPEN_ENDED}" == "1" ]] && echo open-ended || echo mcq)"
echo "[INFO] Batch size: ${BATCH_SIZE}, max_new_tokens: ${MAX_NEW_TOKENS}"
echo "[INFO] Data path: ${DATA_PATH} -> ${SAVE_DATA_PATH}"
echo "[INFO] Prompt column: ${PROMPT_COLNAME}; chosen column: ${CHOSEN_COLNAME}"

#
# Build launcher
#
if [[ "${USE_ACCELERATE}" == "1" ]]; then
  LAUNCHER=(accelerate launch --config_file "${ACCEL_CONFIG}")
else
  LAUNCHER=(python)
fi

PY_SCRIPT="${SCRIPT_DIR}/generate_rejected.py"

CMD=(
  "${LAUNCHER[@]}"
  "${PY_SCRIPT}"
  --batch_size "${BATCH_SIZE}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --on_policy_model_name "${ON_POLICY_MODEL}"
  --data_path "${DATA_PATH}"
  --save_data_path "${SAVE_DATA_PATH}"
  --start_index "${START_INDEX}"
  --end_index "${END_INDEX}"
  --checkpoint_freq "${CHECKPOINT_FREQ}"
  --system_prompt "${SYSTEM_PROMPT}"
  --prompt_colname "${PROMPT_COLNAME}"
  --chosen_colname "${CHOSEN_COLNAME}"
)

if [[ "${OPEN_ENDED}" == "1" ]]; then
  CMD+=( --open_ended )
fi

if [[ "${DO_AUGMENT}" == "1" ]]; then
  CMD+=( --do_augment )
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=( "${EXTRA_ARGS[@]}" )
fi

echo "Executing command:"
echo "${CMD[@]}"
echo ""

"${CMD[@]}"

echo ""
echo "[INFO] Generation complete"
