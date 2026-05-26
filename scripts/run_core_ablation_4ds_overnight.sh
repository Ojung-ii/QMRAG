#!/usr/bin/env bash
set -euo pipefail

LIMIT=1000
DATASET_ARG="all"
INCLUDE_DIAGNOSTIC=0
CORE_ONLY=0
EXTRA_ARGS=()

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --limit)
      LIMIT="$2"; shift 2 ;;
    --datasets)
      DATASET_ARG="$2"; shift 2 ;;
    --include-diagnostic)
      INCLUDE_DIAGNOSTIC=1; shift ;;
    --core-only)
      CORE_ONLY=1; shift ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break ;;
    *)
      EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ "$DATASET_ARG" == "all" ]]; then
  DATASETS=(hotpotqa 2wiki popqa musique)
else
  IFS=',' read -r -a DATASETS <<< "$DATASET_ARG"
fi

CORE_VARIANTS=(
  core_qmrag_mainline
  core_no_bridge
  core_bridge_fullquery
  core_residual_unified_alignment
  core_no_anchor_ordering
  core_no_multi_anchor
)

DIAGNOSTIC_VARIANTS=(
  core_residual_dense_fallback
  core_residual_hybrid_lex_first
  core_residual_dense_only
)

VARIANTS=("${CORE_VARIANTS[@]}")
if [[ "$INCLUDE_DIAGNOSTIC" == "1" && "$CORE_ONLY" != "1" ]]; then
  VARIANTS+=("${DIAGNOSTIC_VARIANTS[@]}")
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/core_ablation_4ds_${RUN_TS}.log"
ANALYSIS_DIR="${ANALYSIS_DIR:-outputs/analysis/${RUN_TS}_core_ablation_4ds}"
mkdir -p "$ANALYSIS_DIR"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "started_at=$(date --iso-8601=seconds)"
echo "analysis_dir=$ANALYSIS_DIR"
echo "log_file=$LOG_FILE"
echo "limit=$LIMIT"
echo "datasets=${DATASETS[*]}"
echo "variants=${VARIANTS[*]}"
echo "python=$PYTHON_BIN"
echo "vllm_base_url=${VLLM_BASE_URL:-}"
echo "nvembed_model_path=${NVEMBED_MODEL_PATH:-}"

if [[ -z "${VLLM_BASE_URL:-}" ]]; then
  echo "warning: VLLM_BASE_URL is unset; config default will be used."
fi
if [[ -z "${NVEMBED_MODEL_PATH:-}" ]]; then
  echo "warning: NVEMBED_MODEL_PATH is unset; config default will be used."
fi

FAILURES=()
for dataset in "${DATASETS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    config="config/ablation/${variant}.yaml"
    timestamp="${RUN_TS}_${dataset}_${variant}_n${LIMIT}"
    echo
    echo "== running dataset=$dataset variant=$variant timestamp=$timestamp =="
    if ! "$PYTHON_BIN" main.py \
      --config "$config" \
      --datasets "$dataset" \
      --timestamp "$timestamp" \
      --limit "$LIMIT" \
      --prompt-profile common_qa \
      --rendering-profile structured_chain \
      --retrieval-variant full_hetero \
      --seed-selection-variant top_relevance \
      --ablation-variant "$variant" \
      --enable-timing \
      "${EXTRA_ARGS[@]}"; then
      echo "FAILED dataset=$dataset variant=$variant"
      FAILURES+=("${dataset}:${variant}")
      if [[ "$variant" == "core_qmrag_mainline" ]]; then
        printf '%s\n' "${FAILURES[@]}" > "$ANALYSIS_DIR/failed_variants.txt"
        echo "mainline failed; aborting core ablation run"
        exit 1
      fi
    fi
  done
done

if [[ "${#FAILURES[@]}" -gt 0 ]]; then
  printf '%s\n' "${FAILURES[@]}" > "$ANALYSIS_DIR/failed_variants.txt"
  echo "failures: ${FAILURES[*]}"
fi

"$PYTHON_BIN" scripts/compare_core_ablation_4ds.py \
  --latest \
  --analysis-dir "$ANALYSIS_DIR"

echo "finished_at=$(date --iso-8601=seconds)"
echo "wrote analysis to $ANALYSIS_DIR"

