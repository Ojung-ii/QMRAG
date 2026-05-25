#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-all}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/acerag_native_prompt_ablation}"
RUN_ID="${RUN_ID:-}"
N="${N:-100}"
DATASETS="${DATASETS:-hotpotqa 2wikimultihopqa}"
TOP_K="${TOP_K:-3}"
TOP_K_LIST="${TOP_K_LIST:-3 5 8 10}"
VARIANTS="${VARIANTS:-p0_current p1_supporting_fallback p2_relaxed_chain p3_minimal_extraction p4_fewshot_extraction}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8013/v1}"
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
VLLM_MODEL="${VLLM_MODEL:-auto}"

normalize_dataset() {
  case "$1" in
    2wikimultihopqa) echo "2wiki" ;;
    *) echo "$1" ;;
  esac
}

latest_run_root() {
  if [[ -f "$OUTPUT_ROOT/latest_run.txt" ]]; then
    cat "$OUTPUT_ROOT/latest_run.txt"
    return
  fi
  find "$OUTPUT_ROOT" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1
}

if [[ -n "$RUN_ID" ]]; then
  RUN_ROOT="$OUTPUT_ROOT/$RUN_ID"
elif [[ "$STAGE" == "stage2" ]]; then
  RUN_ROOT="$(latest_run_root || true)"
  if [[ -z "$RUN_ROOT" ]]; then
    RUN_ROOT="$OUTPUT_ROOT/$(date +%Y%m%d_%H%M%S)"
  fi
else
  RUN_ROOT="$OUTPUT_ROOT/$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "$RUN_ROOT"
mkdir -p "$OUTPUT_ROOT"
printf "%s\n" "$RUN_ROOT" > "$OUTPUT_ROOT/latest_run.txt"

run_replay() {
  local stage_dir="$1"
  local dataset="$2"
  local variant="$3"
  local top_k="$4"
  local limit="$5"
  local compaction="$6"
  local out_dir="$7"
  mkdir -p "$out_dir"
  local log="$out_dir/run.log"
  local cmd=(
    python scripts/replay_generation.py
    --dataset "$dataset"
    --source-prompt common_qa
    --source-rendering-profile structured_chain
    --latest
    --limit "$limit"
    --ace-native-prompt-variant "$variant"
    --compaction-profile "$compaction"
    --output-dir "$out_dir"
    --vllm-base-url "$VLLM_BASE_URL"
    --vllm-api-key "$VLLM_API_KEY"
    --vllm-model "$VLLM_MODEL"
  )
  if [[ "$compaction" == "chain_dedup" ]]; then
    cmd+=(--top-bundles "$top_k")
  fi
  {
    echo "stage=$stage_dir"
    echo "dataset=$dataset"
    echo "variant=$variant"
    echo "top_k=$top_k"
    echo "limit=$limit"
    echo "compaction=$compaction"
    printf 'COMMAND:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
  } | tee "$log"
  "${cmd[@]}" 2>&1 | tee -a "$log"
}

run_smoke() {
  for variant in p1_supporting_fallback p3_minimal_extraction; do
    run_replay "stage0_smoke" "hotpotqa" "$variant" "3" "5" "top3_chain_dedup" \
      "$RUN_ROOT/stage0_smoke/hotpotqa/$variant"
  done
  python scripts/compare_acerag_native_prompt_ablation.py --root "$RUN_ROOT"
}

run_stage1() {
  for raw_dataset in $DATASETS; do
    dataset="$(normalize_dataset "$raw_dataset")"
    for variant in $VARIANTS; do
      run_replay "stage1_prompt_top3" "$dataset" "$variant" "$TOP_K" "$N" "top3_chain_dedup" \
        "$RUN_ROOT/stage1_prompt_top3/$dataset/$variant"
    done
  done
  python scripts/compare_acerag_native_prompt_ablation.py --root "$RUN_ROOT"
}

run_stage2() {
  local best="${BEST_PROMPT_VARIANT:-}"
  if [[ -z "$best" ]]; then
    best="$(python scripts/compare_acerag_native_prompt_ablation.py --root "$RUN_ROOT" --print-best)"
  fi
  echo "BEST_PROMPT_VARIANT=$best"
  for raw_dataset in $DATASETS; do
    dataset="$(normalize_dataset "$raw_dataset")"
    for top_k in $TOP_K_LIST; do
      run_replay "stage2_context_scaling" "$dataset" "$best" "$top_k" "$N" "chain_dedup" \
        "$RUN_ROOT/stage2_context_scaling/$dataset/${best}_top${top_k}"
    done
  done
  python scripts/compare_acerag_native_prompt_ablation.py --root "$RUN_ROOT"
}

case "$STAGE" in
  smoke) run_smoke ;;
  stage1) run_stage1 ;;
  stage2) run_stage2 ;;
  all)
    run_smoke
    run_stage1
    run_stage2
    ;;
  *)
    echo "Usage: $0 {smoke|stage1|stage2|all}" >&2
    exit 2
    ;;
esac

echo "RUN_ROOT=$RUN_ROOT"
