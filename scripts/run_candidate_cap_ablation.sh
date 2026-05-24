#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "Usage: bash scripts/run_candidate_cap_ablation.sh <hotpotqa|2wiki|popqa|musique|all> <limit> [main.py args...]" >&2
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

ANALYSIS_DIR="${ANALYSIS_DIR:-outputs/analysis/$(date +%Y%m%d_%H%M%S)_candidate_cap}"
mkdir -p "$ANALYSIS_DIR"

CAPS=(80 60 40 30)

echo "analysis_dir=$ANALYSIS_DIR limit=$LIMIT datasets=${DATASETS[*]} candidate_caps=${CAPS[*]}"

for dataset in "${DATASETS[@]}"; do
  for cap in "${CAPS[@]}"; do
    config="config/ablation/candidate_cap_${cap}.yaml"
    timestamp="$(date +%Y%m%d_%H%M%S)_${dataset}_candidate_cap_${cap}_n${LIMIT}"
    python main.py \
      --config "$config" \
      --datasets "$dataset" \
      --timestamp "$timestamp" \
      --limit "$LIMIT" \
      --prompt-profile common_qa \
      --rendering-profile structured_chain \
      --retrieval-variant full_hetero \
      --candidate-pool-size "$cap" \
      --enable-timing \
      "$@"
  done
  python scripts/compare_candidate_cap_ablation.py \
    --dataset "$dataset" \
    --latest \
    --analysis-dir "$ANALYSIS_DIR"
done

echo "wrote analysis to $ANALYSIS_DIR"
