#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Path to test Slurm script (relative to repo)
BASE_SLURM_SCRIPT="${SCRIPT_DIR}/test_steering.sh"

# --- Customize the sweep here (comment out items to reduce fan-out) ---
MODEL_SIZE="2b"
MULTILINGUAL_EVAL=1
MCQ_LIST=(0 1)
COUNTRIES=(brazil mozambique portugal india nepal spain bolivia mexico usa australia u egypt morocco ksa levantine)
LOCS=(localized nolocalized both)


# Optional: limit total submissions (0 = no limit)
MAX_JOBS=0
LIMIT=8 # limit the number of samples in test for debug

# --- No changes needed below ---
submit_count=0
timestamp=$(date +%Y%m%d_%H%M%S)

echo "Starting submission matrix at ${timestamp}"
echo "Base script: ${BASE_SLURM_SCRIPT}"
echo "MCQ_EVAL: ${MCQ_LIST[*]}"
echo "Countries: ${COUNTRIES[*]}"
echo "Localization: ${LOCS[*]}"
echo

for mcq in "${MCQ_LIST[@]}"; do
  for country in "${COUNTRIES[@]}"; do
    for loc in "${LOCS[@]}"; do
      if [[ ${MAX_JOBS} -gt 0 && ${submit_count} -ge ${MAX_JOBS} ]]; then
        echo "Reached MAX_JOBS=${MAX_JOBS}, stopping."
        exit 0
      fi

      # Make a temp copy and patch the 3 lines safely
      tmpfile=$(mktemp --suffix ".slurm")
      cp "${BASE_SLURM_SCRIPT}" "${tmpfile}"

      # Replace the exact assignment lines; keep everything else intact.
      # Handles possible leading/trailing spaces.
      sed -i -E \
        -e "s|^([[:space:]]*MODEL_SIZE=).*|\1${MODEL_SIZE}|" \
        -e "s|^([[:space:]]*MULTILINGUAL_EVAL=).*|\1${MULTILINGUAL_EVAL}|" \
        -e "s|^([[:space:]]*MCQ_EVAL=).*|\1${mcq}|" \
        -e "s|^([[:space:]]*COUNTRY_NAME=).*|\1\"${country}\"|" \
        -e "s|^([[:space:]]*LOCALIZATION_STATUS=).*|\1${loc}|" \
        -e "s|^([[:space:]]*LIMIT=).*|\1${LIMIT}|" \
        "${tmpfile}"

      # Construct a readable job name (overrides the #SBATCH header)
      job_name="yapo_${MODEL_SIZE}_${country}_${loc}_mcq${mcq}"

      # Submit; also pass the CLI flags your script already parses (belt & suspenders)
      # Capture JobID from sbatch output
      sbatch_out=$(sbatch \
        --job-name "${job_name}" \
        "${tmpfile}" \
        --localization-status "${loc}" \
        --country-name "${country}")

      echo "${sbatch_out}  <-- ${job_name}"
      ((submit_count+=1))
    done
  done
done

echo
echo "Submitted ${submit_count} jobs total."
