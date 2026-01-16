#!/bin/bash
#SBATCH --job-name=yapo_test
#SBATCH --partition=hermes-2
#SBATCH --nodes=1                # Number of nodes
#SBATCH --ntasks-per-node=1      # One task per node
#SBATCH --cpus-per-task=128      # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --gres=gpu:8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=500:00:00
#SBATCH --output=logs/modeling/test/%x_%j.out
#SBATCH --error=logs/modeling/test/%x_%j.err

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
mkdir -p "${REPO_ROOT}/logs/modeling/test"
cd "${SCRIPT_DIR}"

# Activate the venv/conda env similar to other scripts
echo "Initializing conda"
eval "$(conda shell.bash hook)"
conda deactivate
echo "Activating conda environment 'yapo'"
conda activate yapo

# Defaults (override via env or CLI as needed)
USE_ACCELERATE=1

# important params
MODEL_SIZE="2b"
MODEL=google/gemma-2-${MODEL_SIZE}-it
MCQ_EVAL=0
MULTILINGUAL_EVAL=0
COUNTRY_NAME="egypt"
# COUNTRY_NAME="morocco"
LOCALIZATION_STATUS=both   # localized|nolocalized|both
MIN_EPOCH=1
MAX_EPOCH=20
MULTIPLIERS=${MULTIPLIERS:-"1.0"}
MAX_NEW_TOKENS=2048
LIMIT=8

if [[ "$MULTILINGUAL_EVAL" == "1" ]]; then
  DATASET_REPO=Alignement/Multilingual_Cultural_Dataset_MCQ_flattened
else
  DATASET_REPO=Alignement/Arabic_Cultural_Dataset_MCQ
fi

if [[ "$MCQ_EVAL" == "1" ]]; then
  IS_MCQ_STR="mcq"
else
  IS_MCQ_STR="oe"
fi

if [[ "$MODEL_SIZE" == "2b" ]]; then
  TRAIN_BATCH_SIZE=2
  LAYER=15
  EPOCHS=20
  LR=0.0005
  BATCH_SIZE=8
else
  TRAIN_BATCH_SIZE=1
  LAYER=28
  EPOCHS=20
  LR=0.0005
  BATCH_SIZE=1
fi

VECTORS_ROOT="${VECTORS_ROOT:-${REPO_ROOT}/vectors}"
VECTOR_TEMPLATE_DENSE="${VECTOR_TEMPLATE_DENSE:-${VECTORS_ROOT}/${COUNTRY_NAME}_dense_${MODEL_SIZE}_${LAYER}_${LR}_${TRAIN_BATCH_SIZE}_${EPOCHS}_${MCQ_EVAL}_loc-${LOCALIZATION_STATUS}_gemma-${IS_MCQ_STR}}"
VECTOR_TEMPLATE_SPARSE="${VECTOR_TEMPLATE_SPARSE:-${VECTORS_ROOT}/${COUNTRY_NAME}_sparse_${MODEL_SIZE}_${LAYER}_${LR}_${TRAIN_BATCH_SIZE}_${EPOCHS}_${MCQ_EVAL}_loc-${LOCALIZATION_STATUS}_gemma-${IS_MCQ_STR}}"

PUSH_PRIVATE=0
LAYER=15
# SAE config (for sparse)
SAE_REPO=google/gemma-scope-${MODEL_SIZE}-pt-res
# SAE_WIDTH=65k
# SAE_AVG_IDX=68

# Helper: set SAE params based on model size and layer
select_sae_params() {
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
    case "${LAYER}" in
      25) SAE_VECTOR_SIZE="131k"; SAE_WIDTH="131k"; SAE_AVG_IDX="96" ;;
      26) SAE_VECTOR_SIZE="131k"; SAE_WIDTH="131k"; SAE_AVG_IDX="97" ;;
      27) SAE_VECTOR_SIZE="131k"; SAE_WIDTH="131k"; SAE_AVG_IDX="96" ;;
      28) SAE_VECTOR_SIZE="131k"; SAE_WIDTH="131k"; SAE_AVG_IDX="98" ;;
      29) SAE_VECTOR_SIZE="131k"; SAE_WIDTH="131k"; SAE_AVG_IDX="97" ;;
      30) SAE_VECTOR_SIZE="131k"; SAE_WIDTH="131k"; SAE_AVG_IDX="95" ;;
      *)
        SAE_VECTOR_SIZE=${SAE_VECTOR_SIZE:-"131k"}
        SAE_WIDTH=${SAE_WIDTH:-"${SAE_VECTOR_SIZE}"}
        SAE_AVG_IDX=${SAE_AVG_IDX:-"98"}
        echo "[WARN] No 9B SAE params configured for layer ${LAYER}; using SAE_WIDTH=${SAE_WIDTH}, SAE_AVG_IDX=${SAE_AVG_IDX}" >&2
        ;;
    esac
  else
    case "${LAYER}" in
      12) SAE_VECTOR_SIZE="65k"; SAE_WIDTH="65k"; SAE_AVG_IDX="72" ;;
      13) SAE_VECTOR_SIZE="65k"; SAE_WIDTH="65k"; SAE_AVG_IDX="75" ;;
      14) SAE_VECTOR_SIZE="65k"; SAE_WIDTH="65k"; SAE_AVG_IDX="73" ;;
      15) SAE_VECTOR_SIZE="65k"; SAE_WIDTH="65k"; SAE_AVG_IDX="68" ;;
      16) SAE_VECTOR_SIZE="65k"; SAE_WIDTH="65k"; SAE_AVG_IDX="69" ;;
      17) SAE_VECTOR_SIZE="65k"; SAE_WIDTH="65k"; SAE_AVG_IDX="68" ;;
      *)
        SAE_VECTOR_SIZE=${SAE_VECTOR_SIZE:-"${SAE_WIDTH:-65k}"}
        SAE_WIDTH=${SAE_WIDTH:-"${SAE_VECTOR_SIZE}"}
        SAE_AVG_IDX=${SAE_AVG_IDX:-"98"}
        echo "[WARN] No 2B SAE params configured for layer ${LAYER}; using SAE_WIDTH=${SAE_WIDTH}, SAE_AVG_IDX=${SAE_AVG_IDX}" >&2
        ;;
    esac
  fi
}

# Parse CLI flags (localization + country)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --localization-status|--localization_status) LOCALIZATION_STATUS="${2}"; shift 2 ;;
    --country-name|--country_name) COUNTRY_NAME="${2}"; shift 2 ;;
    *) shift ;;
  esac
done

# Choose SAE params after defaults
select_sae_params

# Output labels and vector paths
BEHAVIOR_DENSE=${BEHAVIOR_DENSE:-egypt_dense}
BEHAVIOR_SPARSE=${BEHAVIOR_SPARSE:-egypt_sparse}

# Single efficient run processing all epochs at once
if [[ "$MULTILINGUAL_EVAL" == "1" ]]; then
  if [[ "$MCQ_EVAL" == "1" ]]; then
    PUSH_REPO=Alignement/Multilingual_cultural_dataset_eval_${MODEL_SIZE}_L${LAYER}_mcq_amd_${COUNTRY_NAME}_loc-${LOCALIZATION_STATUS}
  else
    PUSH_REPO=Alignement/Multilingual_cultural_dataset_eval_${MODEL_SIZE}_L${LAYER}_oe_amd_${COUNTRY_NAME}_loc-${LOCALIZATION_STATUS}
  fi
else
  if [[ "$MCQ_EVAL" == "1" ]]; then
    PUSH_REPO=Alignement/Arabic_cultural_dataset_eval_${MODEL_SIZE}_L${LAYER}_mcq_amd_${COUNTRY_NAME}_loc-${LOCALIZATION_STATUS}
  else
    PUSH_REPO=Alignement/Arabic_cultural_dataset_eval_${MODEL_SIZE}_L${LAYER}_oe_amd_${COUNTRY_NAME}_loc-${LOCALIZATION_STATUS}
  fi
fi

# Vector path templates (use {epoch} and {layer} as placeholders)
# Dense vectors live in the hidden-size space (e.g., 2304 for Gemma-2 2B)
# Sparse vectors live in SAE feature space (e.g., 65k) and are only valid with SparseModelWrapper
# Include localization suffix in vector directory name for consistency with training
if [[ "${LOCALIZATION_STATUS}" == "both" ]]; then
  LOC_SUFFIX=""
else
  LOC_SUFFIX="_loc-${LOCALIZATION_STATUS}"
fi
export LOC_TAG="${LOC_SUFFIX}"

echo "Running efficient batch steering evaluation with:"
echo "  model          = ${MODEL}"
echo "  layer          = ${LAYER}"
echo "  epochs         = ${MIN_EPOCH} to ${MAX_EPOCH}"
echo "  multipliers    = ${MULTIPLIERS}"
echo "  dataset_repo   = ${DATASET_REPO}"
echo "  country        = ${COUNTRY_NAME}"
echo "  batch_size     = ${BATCH_SIZE}"
echo "  limit          = ${LIMIT}"
echo "  behavior_dense = ${BEHAVIOR_DENSE}"
echo "  behavior_sparse= ${BEHAVIOR_SPARSE}"
echo "  push_repo      = ${PUSH_REPO}"

# Generate epoch sequence
EPOCH_LIST=$(seq -s ' ' ${MIN_EPOCH} ${MAX_EPOCH})

# Single efficient run processing all epochs at once
if [[ "$USE_ACCELERATE" == "1" ]]; then
  LAUNCHER=(accelerate launch --config_file inference_config.yaml)
else
  LAUNCHER=(python)
fi

CMD=(
  ${LAUNCHER[@]}
  test_steering.py
  --behavior_dense "${BEHAVIOR_DENSE}"
  --behavior_sparse "${BEHAVIOR_SPARSE}"
  --model_name_or_path "${MODEL}"
  --layer "${LAYER}"
  --epochs ${EPOCH_LIST}
  --multipliers ${MULTIPLIERS}
  --dataset_repo "${DATASET_REPO}"
  --country_name "${COUNTRY_NAME}"
  --localization_status "${LOCALIZATION_STATUS}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --batch_size "${BATCH_SIZE}"
  --verbose
  --sae_repo "${SAE_REPO}"
  --sae_width "${SAE_WIDTH}"
  --sae_avg_idx "${SAE_AVG_IDX}"
  --dense_vector_template "${VECTOR_TEMPLATE_DENSE}"
  --sparse_vector_template "${VECTOR_TEMPLATE_SPARSE}"
)

if [[ -n "${PUSH_REPO}" ]]; then
  CMD+=( --push_repo "${PUSH_REPO}" )
  if [[ -n "${PUSH_SPLIT_NAME}" ]]; then 
    CMD+=( --push_split_name "${PUSH_SPLIT_NAME}" )
  fi
  if [[ -n "${PUSH_COMMIT_MESSAGE}" ]]; then 
    CMD+=( --push_commit_message "${PUSH_COMMIT_MESSAGE}" )
  fi
  if [[ "${PUSH_PRIVATE}" == "1" ]]; then 
    CMD+=( --push_private )
  fi
fi

if [[ -n "${LIMIT}" ]]; then
  CMD+=( --limit "${LIMIT}" )
fi

# Enable MCQ evaluation only when explicitly requested
# Accepts MCQ_EVAL=1 (default off when 0 or empty)
if [[ "${MCQ_EVAL}" == "1" ]]; then
  CMD+=( --mcq_eval )
fi

echo "Executing command:"
echo "${CMD[@]}"
echo ""

"${CMD[@]}"

echo ""
echo "Batch evaluation completed!"
