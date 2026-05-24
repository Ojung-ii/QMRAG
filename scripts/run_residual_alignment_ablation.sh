#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "Usage: bash scripts/run_residual_alignment_ablation.sh <hotpotqa|2wiki|popqa|musique|all> <limit> [main.py args...]" >&2
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
PYTHON_BIN="${PYTHON_BIN:-python}"

ANALYSIS_DIR="${ANALYSIS_DIR:-outputs/analysis/$(date +%Y%m%d_%H%M%S)_residual_alignment}"
mkdir -p "$ANALYSIS_DIR"

VARIANTS=(
  residual_lexical
  bridge_fullquery
  residual_dense_only
  residual_hybrid_lex_first
  residual_dense_fallback
  residual_unified_alignment
)

echo "analysis_dir=$ANALYSIS_DIR limit=$LIMIT datasets=${DATASETS[*]} variants=${VARIANTS[*]}"

FAILURES=()
for dataset in "${DATASETS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    config="config/ablation/${variant}.yaml"
    if [[ "$variant" == "bridge_fullquery" ]]; then
      config="config/ablation/residual_bridge_fullquery.yaml"
    fi
    timestamp="$(date +%Y%m%d_%H%M%S)_${dataset}_${variant}_n${LIMIT}"
    echo "running dataset=$dataset residual_selection=$variant timestamp=$timestamp"
    if ! "$PYTHON_BIN" main.py \
      --config "$config" \
      --datasets "$dataset" \
      --timestamp "$timestamp" \
      --limit "$LIMIT" \
      --prompt-profile common_qa \
      --rendering-profile structured_chain \
      --retrieval-variant full_hetero \
      --seed-selection-variant top_relevance \
      --residual-selection "$variant" \
      --enable-timing \
      "$@"; then
      echo "FAILED dataset=$dataset residual_selection=$variant" >&2
      FAILURES+=("${dataset}:${variant}")
    fi
  done
  "$PYTHON_BIN" scripts/compare_residual_alignment_ablation.py \
    --dataset "$dataset" \
    --latest \
    --analysis-dir "$ANALYSIS_DIR"
done

if [[ "${#FAILURES[@]}" -gt 0 ]]; then
  printf '%s\n' "${FAILURES[@]}" > "$ANALYSIS_DIR/failed_variants.txt"
  echo "failures: ${FAILURES[*]}" >&2
fi

echo "wrote analysis to $ANALYSIS_DIR"
