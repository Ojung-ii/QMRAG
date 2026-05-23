#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "Usage: bash scripts/run_prompt_ablation.sh <dataset> <limit> [main.py args...]" >&2
  exit 2
fi

DATASET="$1"
LIMIT="$2"
shift 2

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

for PROFILE in common_qa qmrag_bundle_qa; do
  echo "==> Running ${DATASET} with prompt_profile=${PROFILE}"
  bash scripts/run_dataset.sh "$DATASET" \
    --timestamp "${RUN_ID}_${PROFILE}" \
    --limit "$LIMIT" \
    --prompt-profile "$PROFILE" \
    "$@"
done
