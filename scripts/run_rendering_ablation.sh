#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: bash scripts/run_rendering_ablation.sh <hotpotqa|2wiki|popqa> [limit]" >&2
  exit 2
fi

DATASET="$1"
LIMIT="${2:-100}"
PROMPT_PROFILE="${PROMPT_PROFILE:-common_qa}"
ANALYSIS_DIR="${ANALYSIS_DIR:-outputs/analysis/$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$ANALYSIS_DIR"
echo "dataset=$DATASET prompt_profile=$PROMPT_PROFILE limit=$LIMIT analysis_dir=$ANALYSIS_DIR"

for profile in plain_evidence chain_only_compact multi_anchor_table; do
  python scripts/replay_generation.py \
    --dataset "$DATASET" \
    --source-prompt "$PROMPT_PROFILE" \
    --target-prompt "$PROMPT_PROFILE" \
    --rendering-profile "$profile" \
    --latest \
    --limit "$LIMIT"

  python scripts/compare_rendering_runs.py \
    --dataset "$DATASET" \
    --prompt-profile "$PROMPT_PROFILE" \
    --left-rendering structured_chain \
    --right-rendering "$profile" \
    --latest \
    --analysis-dir "$ANALYSIS_DIR"
done

echo "wrote analysis to $ANALYSIS_DIR"
