#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-all}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/acerag_native_prompt_ablation}"
RUN_ID="${RUN_ID:-}"
N="${N:-100}"
DATASETS="${DATASETS:-hotpotqa 2wikimultihopqa}"
TOP_K="${TOP_K:-3}"
TOP_K_LIST="${TOP_K_LIST:-3 5 8 10}"
VARIANTS="${PROMPT_VARIANTS:-${VARIANTS:-p0_current p1_supporting_fallback p2_relaxed_chain p3_minimal_extraction p4_fewshot_extraction}}"
TOP_K_BY_VARIANT="${TOP_K_BY_VARIANT:-}"
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

top_values_for_variant() {
  local variant="$1"
  if [[ -n "$TOP_K_BY_VARIANT" ]]; then
    for spec in $TOP_K_BY_VARIANT; do
      local name="${spec%%:*}"
      local values="${spec#*:}"
      if [[ "$name" == "$variant" ]]; then
        echo "${values//,/ }"
        return
      fi
    done
    echo ""
    return
  fi
  echo "$TOP_K_LIST"
}

compaction_for_top_k() {
  local top_k="$1"
  if [[ "$top_k" == "3" ]]; then
    echo "top3_chain_dedup"
  else
    echo "chain_dedup"
  fi
}

setting_name_for() {
  local variant="$1"
  local top_k="$2"
  case "${variant}_${top_k}" in
    p0_current_3) echo "current_native_compact" ;;
    p2_relaxed_chain_3) echo "relaxed_native_compact" ;;
    p2_relaxed_chain_8) echo "relaxed_native_scaled" ;;
    p3_minimal_extraction_3) echo "minimal_native_compact" ;;
    *) echo "${variant}_top${top_k}" ;;
  esac
}

run_custom() {
  local stage_dir="${CUSTOM_STAGE_DIR:-custom}"
  for raw_dataset in $DATASETS; do
    dataset="$(normalize_dataset "$raw_dataset")"
    for variant in $VARIANTS; do
      for top_k in $(top_values_for_variant "$variant"); do
        [[ -z "$top_k" ]] && continue
        local compaction
        compaction="$(compaction_for_top_k "$top_k")"
        run_replay "$stage_dir" "$dataset" "$variant" "$top_k" "$N" "$compaction" \
          "$RUN_ROOT/$stage_dir/$dataset/${variant}_top${top_k}"
      done
    done
  done
  python scripts/compare_acerag_native_prompt_ablation.py --root "$RUN_ROOT"
}

run_final_core() {
  local stage_dir="core_n1000"
  local final_variants="${PROMPT_VARIANTS:-p0_current p2_relaxed_chain p3_minimal_extraction}"
  local final_map="${TOP_K_BY_VARIANT:-p0_current:3 p2_relaxed_chain:3,8 p3_minimal_extraction:3}"
  PROMPT_VARIANTS="$final_variants"
  VARIANTS="$final_variants"
  TOP_K_BY_VARIANT="$final_map"
  for raw_dataset in $DATASETS; do
    dataset="$(normalize_dataset "$raw_dataset")"
    for variant in $VARIANTS; do
      for top_k in $(top_values_for_variant "$variant"); do
        [[ -z "$top_k" ]] && continue
        local compaction setting
        compaction="$(compaction_for_top_k "$top_k")"
        setting="$(setting_name_for "$variant" "$top_k")"
        run_replay "$stage_dir" "$dataset" "$variant" "$top_k" "$N" "$compaction" \
          "$RUN_ROOT/$stage_dir/$dataset/$setting"
      done
    done
  done
  ln -sfn "$RUN_ROOT/$stage_dir" "$OUTPUT_ROOT/latest_core_n1000"
  printf "%s\n" "$RUN_ROOT/$stage_dir" > "$OUTPUT_ROOT/latest_core_n1000.txt"
  python scripts/compare_acerag_native_prompt_ablation.py --root "$RUN_ROOT"
}

run_appendix_4ds() {
  local stage_dir="appendix_4ds_p2_top8_n1000"
  for raw_dataset in $DATASETS; do
    dataset="$(normalize_dataset "$raw_dataset")"
    run_replay "$stage_dir" "$dataset" "p2_relaxed_chain" "8" "$N" "chain_dedup" \
      "$RUN_ROOT/$stage_dir/$dataset/p2_relaxed_chain_top8"
  done
  python scripts/compare_acerag_native_prompt_ablation.py --root "$RUN_ROOT"
}

case "$STAGE" in
  smoke) run_smoke ;;
  stage1) run_stage1 ;;
  stage2) run_stage2 ;;
  custom) run_custom ;;
  final_core) run_final_core ;;
  appendix_4ds) run_appendix_4ds ;;
  all)
    run_smoke
    run_stage1
    run_stage2
    ;;
  *)
    echo "Usage: $0 {smoke|stage1|stage2|custom|final_core|appendix_4ds|all}" >&2
    exit 2
    ;;
esac

echo "RUN_ROOT=$RUN_ROOT"
