#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: bash scripts/run_prompt_light_ablation.sh <hotpotqa|2wiki|popqa|musique|all> [limit]" >&2
  exit 2
fi

TARGET="$1"
LIMIT="${2:-1000}"
ANALYSIS_DIR="${ANALYSIS_DIR:-outputs/analysis/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$ANALYSIS_DIR"

if [[ "$TARGET" == "all" ]]; then
  DATASETS=(hotpotqa 2wiki popqa musique)
else
  DATASETS=("$TARGET")
fi

echo "analysis_dir=$ANALYSIS_DIR limit=$LIMIT datasets=${DATASETS[*]}"

for dataset in "${DATASETS[@]}"; do
  for prompt in qmrag_bundle_tiny qmrag_bundle_light qmrag_bundle_qa; do
    python scripts/replay_generation.py \
      --dataset "$dataset" \
      --source-prompt common_qa \
      --source-rendering-profile structured_chain \
      --target-prompt "$prompt" \
      --latest \
      --limit "$LIMIT"
  done

  python scripts/compare_prompt_efficiency.py \
    --dataset "$dataset" \
    --latest \
    --analysis-dir "$ANALYSIS_DIR"

  python scripts/evaluate_qmrag_retrieval_metrics.py \
    --dataset "$dataset" \
    --prompt-profile common_qa \
    --latest \
    --analysis-dir "$ANALYSIS_DIR"
done

echo "wrote analysis to $ANALYSIS_DIR"
