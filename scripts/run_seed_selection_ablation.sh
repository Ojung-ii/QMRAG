#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "Usage: bash scripts/run_seed_selection_ablation.sh <hotpotqa|2wiki|popqa|musique|all> <limit> [main.py args...]" >&2
  exit 2
fi

TARGET="$1"
LIMIT="$2"
shift 2

if [[ "$TARGET" == "all" ]]; then
  DATASETS=(hotpotqa 2wiki popqa musique)
else
  DATASETS=("$TARGET")
fi

export NVEMBED_MODEL_PATH="${NVEMBED_MODEL_PATH:-/home/dilab/.cache/huggingface/models--nvidia--NV-Embed-v2/snapshots/3fa59658547db50a1e8e3346cf057fd0c77ed6ef/}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8011/v1}"
export VLLM_MODEL="${VLLM_MODEL:-auto}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"

ANALYSIS_DIR="${ANALYSIS_DIR:-outputs/analysis/$(date +%Y%m%d_%H%M%S)_seed_selection}"
mkdir -p "$ANALYSIS_DIR"

VARIANTS=(diverse_seed_search global_seed_search anchor_first chain_potential)

echo "analysis_dir=$ANALYSIS_DIR limit=$LIMIT datasets=${DATASETS[*]} variants=${VARIANTS[*]}"

for dataset in "${DATASETS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    config="config/ablation/seed_${variant}.yaml"
    timestamp="$(date +%Y%m%d_%H%M%S)_${dataset}_seed_${variant}_n${LIMIT}"
    python main.py \
      --config "$config" \
      --datasets "$dataset" \
      --timestamp "$timestamp" \
      --limit "$LIMIT" \
      --prompt-profile common_qa \
      --rendering-profile structured_chain \
      --seed-selection-variant "$variant" \
      --enable-timing \
      "$@"
  done
  python scripts/compare_seed_selection_ablation.py \
    --dataset "$dataset" \
    --latest \
    --analysis-dir "$ANALYSIS_DIR"
done

echo "wrote analysis to $ANALYSIS_DIR"
