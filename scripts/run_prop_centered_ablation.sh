#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "Usage: bash scripts/run_prop_centered_ablation.sh <hotpotqa|2wiki|popqa|musique|all> <limit> [main.py args...]" >&2
  exit 2
fi

DATASET_ARG="$1"
LIMIT="$2"
shift 2

export NVEMBED_MODEL_PATH="${NVEMBED_MODEL_PATH:-/home/dilab/.cache/huggingface/models--nvidia--NV-Embed-v2/snapshots/3fa59658547db50a1e8e3346cf057fd0c77ed6ef/}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8011/v1}"
export VLLM_MODEL="${VLLM_MODEL:-auto}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"

if [[ "${DATASET_ARG}" == "all" ]]; then
  DATASETS=(hotpotqa 2wiki popqa musique)
else
  DATASETS=("${DATASET_ARG}")
fi

VARIANTS=(
  full_hetero
  prop_text_only
  prop_parent_anchor
  prop_parent_mention_bidirectional
)

BASE_TS="$(date +%Y%m%d_%H%M%S)"
ANALYSIS_DIR="outputs/analysis/${BASE_TS}_prop_centered_ablation"
mkdir -p "${ANALYSIS_DIR}"

echo "analysis_dir=${ANALYSIS_DIR} datasets=${DATASETS[*]} limit=${LIMIT}"

for dataset in "${DATASETS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    ts="${BASE_TS}_${dataset}_${variant}_n${LIMIT}"
    cfg="config/ablation/retrieval_${variant}.yaml"
    echo "running dataset=${dataset} variant=${variant} limit=${LIMIT} timestamp=${ts}"
    python main.py \
      --config "${cfg}" \
      --datasets "${dataset}" \
      --limit "${LIMIT}" \
      --prompt-profile common_qa \
      --rendering-profile structured_chain \
      --retrieval-variant "${variant}" \
      --enable-timing \
      --timestamp "${ts}" \
      "$@"
  done
  python scripts/compare_prop_centered_ablation.py \
    --dataset "${dataset}" \
    --latest \
    --analysis-dir "${ANALYSIS_DIR}"
done

echo "wrote analysis to ${ANALYSIS_DIR}"
