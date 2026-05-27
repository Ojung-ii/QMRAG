#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: bash scripts/run_prompt_ablation.sh <dataset> [limit=10] [main.py args...]" >&2
  exit 2
fi

DATASET="$1"
LIMIT="${2:-10}"
if [[ "$#" -ge 2 ]]; then
  shift 2
else
  shift 1
fi

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

for PROFILE in common_qa ace_rag_bundle_qa; do
  echo "==> Running ${DATASET} with prompt_profile=${PROFILE}"
  bash scripts/run_dataset.sh "$DATASET" \
    --timestamp "${RUN_ID}_${PROFILE}" \
    --limit "$LIMIT" \
    --prompt-profile "$PROFILE" \
    "$@"
done
