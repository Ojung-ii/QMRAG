#!/usr/bin/env bash
set -euo pipefail

ANALYSIS_DIR="${ANALYSIS_DIR:-outputs/analysis/$(date +%Y%m%d_%H%M%S)}"
SAMPLE="${SAMPLE:-10}"

mkdir -p "$ANALYSIS_DIR"
echo "analysis_dir=$ANALYSIS_DIR"

for dataset in hotpotqa 2wiki popqa; do
  python scripts/analyze_failures.py \
    --dataset "$dataset" \
    --prompt-profile common_qa \
    --latest \
    --sample "$SAMPLE" \
    --analysis-dir "$ANALYSIS_DIR"
done

for dataset in hotpotqa 2wiki; do
  if python scripts/compare_prompt_runs.py \
    --dataset "$dataset" \
    --left-prompt common_qa \
    --right-prompt ace_rag_bundle_qa \
    --latest \
    --analysis-dir "$ANALYSIS_DIR"; then
    true
  else
    echo "skip compare for $dataset: missing common_qa or ace_rag_bundle_qa latest run" >&2
  fi
done

echo "wrote analysis to $ANALYSIS_DIR"
