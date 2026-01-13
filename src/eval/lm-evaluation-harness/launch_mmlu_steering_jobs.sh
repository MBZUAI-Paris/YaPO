#!/usr/bin/env bash
set -euo pipefail

# Small helper to submit many steering sweeps without typing long sbatch commands.
# Most values can still be overridden via CLI flags or env vars before calling.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_SLURM_SCRIPT="${SCRIPT_DIR}/run_mmlu_steering.slurm"

usage() {
  cat <<'EOF'
Usage: launch_mmlu_steering_jobs.sh [options]

Options:
  --mode <name>             Steering mode to export (default: $STEERING_MODE or bipo)
  --tasks <list>            lm-evaluation-harness tasks string (default: $TASKS or "mmlu")
  --countries a,b,c         Comma separated list of countries (default: egypt,morocco,ksa,levantine)
  --localizations x,y,z     Comma separated localization tags (default: localized,nolocalized,both)
  --mcq 0,1                 Comma separated MCQ toggle values (default: 0,1)
  --epochs e1,e2            Comma separated epochs to try (default: 19)
  --model-size <2b|9b>      Override STEERING_MODEL_SIZE (default: $STEERING_MODEL_SIZE or 2b)
  --tau <float>             Override STEERING_TAU (default: $STEERING_TAU or 0.7)
  --lambda <float>          Override STEERING_MULTIPLIER (default: $STEERING_MULTIPLIER or 1.0)
  --layer <int>             Force STEERING_LAYER (otherwise inferred by slurm script)
  --relu <True|False>       Override STEERING_RELU toggle
  --slurm-script <path>     Path to run_mmlu_steering.slurm (default: repo copy)
  --job-prefix <str>        Prefix for JOB_NAME (default: steering mode)
  --run-prefix <str>        Prefix for OUTPUT_NAME (default: "mmlu")
  --dry-run                 Only print sbatch commands without submitting
  -h, --help                Show this help and exit

Environment variables such as STEERING_TAU, STEERING_MULTIPLIER, etc. are
respected as fallbacks if the corresponding flag is omitted.
EOF
}

comma_split() {
  local input="${1:-}"
  local -n out_ref=$2
  IFS=',' read -r -a out_ref <<< "${input}"
}

STEERING_MODE=${STEERING_MODE:-"yapo"} # yapo, bipo, caa, sas
COUNTRIES=(uk usa australia spain mexico bolivia brazil mozambique portugal ksa egypt morocco levantine)
# COUNTRIES=(egypt)
LOCALIZATIONS=(localized nolocalized both)
# MCQ_VALUES=(0 1)
MCQ_VALUES=(1)
if [[ "$STEERING_MODE" == "caa" || "$STEERING_MODE" == "sas" ]]; then
    STEERING_MULTIPLIER=0.5 # for caa and sas mcq
else
    STEERING_MULTIPLIER=1.0
fi
TASKS=${TASKS:-"mmlu"} # hellaswag mmlu 
SLURM_SCRIPT="${DEFAULT_SLURM_SCRIPT}"
EPOCHS=(10)
STEERING_MODEL_SIZE="9b" # 2b, 9b ${STEERING_MODEL_SIZE:-"2b"}
STEERING_TAU=${STEERING_TAU:-"0.7"}
STEERING_LAYER=${STEERING_LAYER:-""}
STEERING_RELU=${STEERING_RELU:-"True"}
JOB_PREFIX=${JOB_PREFIX:-""}
RUN_PREFIX=${TASKS}
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      STEERING_MODE="$2"
      shift 2
      ;;
    --tasks)
      TASKS="$2"
      shift 2
      ;;
    --countries)
      comma_split "$2" COUNTRIES
      shift 2
      ;;
    --localizations)
      comma_split "$2" LOCALIZATIONS
      shift 2
      ;;
    --mcq)
      comma_split "$2" MCQ_VALUES
      shift 2
      ;;
    --epochs)
      comma_split "$2" EPOCHS
      shift 2
      ;;
    --model-size)
      STEERING_MODEL_SIZE="$2"
      shift 2
      ;;
    --tau)
      STEERING_TAU="$2"
      shift 2
      ;;
    --lambda)
      STEERING_MULTIPLIER="$2"
      shift 2
      ;;
    --layer)
      STEERING_LAYER="$2"
      shift 2
      ;;
    --relu)
      STEERING_RELU="$2"
      shift 2
      ;;
    --behavior)
      STEERING_BEHAVIOR="$2"
      shift 2
      ;;
    --slurm-script|--script)
      SLURM_SCRIPT="$2"
      shift 2
      ;;
    --job-prefix)
      JOB_PREFIX="$2"
      shift 2
      ;;
    --run-prefix)
      RUN_PREFIX="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option '$1'" >&2
      usage
      exit 1
      ;;
  esac
done

[[ -f "${SLURM_SCRIPT}" ]] || { echo "[ERROR] SLURM script not found at ${SLURM_SCRIPT}" >&2; exit 1; }
(( ${#COUNTRIES[@]} > 0 )) || { echo "[ERROR] No countries specified" >&2; exit 1; }
(( ${#LOCALIZATIONS[@]} > 0 )) || { echo "[ERROR] No localizations specified" >&2; exit 1; }
(( ${#MCQ_VALUES[@]} > 0 )) || { echo "[ERROR] No MCQ values specified" >&2; exit 1; }
(( ${#EPOCHS[@]} > 0 )) || { echo "[ERROR] No epochs specified" >&2; exit 1; }

printf "[INFO] Submitting %d combinations using %s\n" \
  $(( ${#COUNTRIES[@]} * ${#LOCALIZATIONS[@]} * ${#MCQ_VALUES[@]} * ${#EPOCHS[@]} )) \
  "${SLURM_SCRIPT}"

job_counter=0

for country in "${COUNTRIES[@]}"; do
  for loc in "${LOCALIZATIONS[@]}"; do
    for mcq in "${MCQ_VALUES[@]}"; do
      for epoch in "${EPOCHS[@]}"; do
        job_counter=$((job_counter + 1))
        slug="${STEERING_MODE}_${country}_${loc}_mcq${mcq}_ep${epoch}"
        output_name="${RUN_PREFIX}_${slug}"
        job_name="${JOB_PREFIX:-${STEERING_MODE}}-${country}-${loc}-mcq${mcq}-ep${epoch}"
        sbatch_cmd=(sbatch "--job-name=${job_name}" "${SLURM_SCRIPT}")
        echo "[INFO] (#${job_counter}) ${job_name} | country=${country}, loc=${loc}, mcq=${mcq}, epoch=${epoch}"
        if (( DRY_RUN )); then
          printf '       DRY RUN: env TASKS=%q STEERING_MODE=%q ... %s\n' \
            "${TASKS}" "${STEERING_MODE}" "${sbatch_cmd[*]}"
          continue
        fi

        env \
          TASKS="${TASKS}" \
          STEERING_MODE="${STEERING_MODE}" \
          STEERING_COUNTRY="${country}" \
          STEERING_LOCALIZATION="${loc}" \
          STEERING_MCQ="${mcq}" \
          EPOCH="${epoch}" \
          OUTPUT_NAME="${output_name}" \
          JOB_NAME="${job_name}" \
          STEERING_MODEL_SIZE="${STEERING_MODEL_SIZE}" \
          STEERING_TAU="${STEERING_TAU}" \
          STEERING_MULTIPLIER="${STEERING_MULTIPLIER}" \
          STEERING_RELU="${STEERING_RELU}" \
          ${STEERING_LAYER:+STEERING_LAYER="${STEERING_LAYER}"} \
          "${sbatch_cmd[@]}"
      done
    done
  done
done
