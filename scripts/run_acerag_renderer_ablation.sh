#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-smoke}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/acerag_renderer_ablation}"
RUN_ID="${RUN_ID:-}"
N="${N:-}"
DATASETS="${DATASETS:-hotpotqa 2wikimultihopqa}"
PROMPT_VARIANT="${PROMPT_VARIANT:-p2_relaxed_chain}"
TOP_K_LIST="${TOP_K_LIST:-8}"
RENDERER_VARIANTS="${RENDERER_VARIANTS:-r0_current r1_clean_sentence r2_title_paragraph r3_chain_paragraph_hybrid}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8013/v1}"
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
VLLM_MODEL="${VLLM_MODEL:-auto}"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-1}"
PARALLEL="${PARALLEL:-0}"

normalize_dataset() {
  case "$1" in
    2wikimultihopqa) echo "2wiki" ;;
    *) echo "$1" ;;
  esac
}

if [[ -n "$RUN_ID" ]]; then
  RUN_ROOT="$OUTPUT_ROOT/$RUN_ID"
else
  RUN_ROOT="$OUTPUT_ROOT/$(date +%Y%m%d_%H%M%S)"
fi
mkdir -p "$RUN_ROOT" "$OUTPUT_ROOT"
printf "%s\n" "$RUN_ROOT" > "$OUTPUT_ROOT/latest_run.txt"

compaction_for_top_k() {
  if [[ "$1" == "3" ]]; then
    echo "top3_chain_dedup"
  else
    echo "chain_dedup"
  fi
}

run_replay() {
  local stage_dir="$1"
  local dataset="$2"
  local top_k="$3"
  local renderer="$4"
  local limit="$5"
  local gpu="${6:-}"
  local job_id="${7:-}"
  local compaction
  compaction="$(compaction_for_top_k "$top_k")"
  local out_dir="$RUN_ROOT/$stage_dir/$dataset/${PROMPT_VARIANT}_top${top_k}_${renderer}"
  mkdir -p "$out_dir"
  local log="$out_dir/run.log"
  local cmd=(
    python scripts/replay_generation.py
    --dataset "$dataset"
    --source-prompt common_qa
    --source-rendering-profile structured_chain
    --latest
    --limit "$limit"
    --ace-native-prompt-variant "$PROMPT_VARIANT"
    --ace-renderer-variant "$renderer"
    --compaction-profile "$compaction"
    --output-dir "$out_dir"
    --vllm-base-url "$VLLM_BASE_URL"
    --vllm-api-key "$VLLM_API_KEY"
    --vllm-model "$VLLM_MODEL"
    --save-rendered-context
    --save-final-prompt
  )
  if [[ "$compaction" == "chain_dedup" ]]; then
    cmd+=(--top-bundles "$top_k")
  fi
  {
    echo "stage=$stage_dir"
    echo "dataset=$dataset"
    echo "prompt_variant=$PROMPT_VARIANT"
    echo "top_k=$top_k"
    echo "renderer_variant=$renderer"
    echo "limit=$limit"
    echo "compaction=$compaction"
    echo "job_id=$job_id"
    echo "CUDA_VISIBLE_DEVICES=${gpu:-${CUDA_VISIBLE_DEVICES:-}}"
    printf 'COMMAND:'
    printf ' %q' "${cmd[@]}"
    printf '\n'
  } | tee "$log"
  "${cmd[@]}" 2>&1 | tee -a "$log"
}

run_parallel_matrix() {
  local stage_dir="$1"
  local limit="$2"
  local jobs_tsv="$RUN_ROOT/$stage_dir/jobs.tsv"
  local failed_tsv="$RUN_ROOT/$stage_dir/failed_jobs.tsv"
  mkdir -p "$RUN_ROOT/$stage_dir"
  : > "$jobs_tsv"
  : > "$failed_tsv"

  local -a pids=()
  local -a descs=()
  local active=0
  local job_id=0

  flush_batch() {
    local i pid desc status
    for i in "${!pids[@]}"; do
      pid="${pids[$i]}"
      desc="${descs[$i]}"
      if wait "$pid"; then
        :
      else
        status="$?"
        printf '%s\t%s\n' "$desc" "$status" >> "$failed_tsv"
      fi
    done
    pids=()
    descs=()
    active=0
  }

  for raw_dataset in $DATASETS; do
    local dataset
    dataset="$(normalize_dataset "$raw_dataset")"
    for top_k in $TOP_K_LIST; do
      for renderer in $RENDERER_VARIANTS; do
        local gpu
        if (( job_id % 2 == 0 )); then
          gpu="0"
        else
          gpu="1"
        fi
        local desc
        desc="${job_id}\t${gpu}\t${stage_dir}\t${dataset}\t${top_k}\t${renderer}\t${limit}"
        printf '%b\n' "$desc" >> "$jobs_tsv"
        (
          export CUDA_VISIBLE_DEVICES="$gpu"
          run_replay "$stage_dir" "$dataset" "$top_k" "$renderer" "$limit" "$gpu" "$job_id"
        ) &
        pids+=("$!")
        descs+=("$desc")
        active=$((active + 1))
        job_id=$((job_id + 1))
        if (( active >= MAX_PARALLEL_JOBS )); then
          flush_batch
        fi
      done
    done
  done
  if (( active > 0 )); then
    flush_batch
  fi

  if [[ -s "$failed_tsv" ]]; then
    echo "Some jobs failed; see $failed_tsv" >&2
  fi
}

run_matrix() {
  local stage_dir="$1"
  local limit="$2"
  if [[ "$PARALLEL" == "1" || "$MAX_PARALLEL_JOBS" -gt 1 ]]; then
    echo "parallel=1 max_parallel_jobs=$MAX_PARALLEL_JOBS"
    run_parallel_matrix "$stage_dir" "$limit"
  else
    for raw_dataset in $DATASETS; do
      local dataset
      dataset="$(normalize_dataset "$raw_dataset")"
      for top_k in $TOP_K_LIST; do
        for renderer in $RENDERER_VARIANTS; do
          run_replay "$stage_dir" "$dataset" "$top_k" "$renderer" "$limit"
        done
      done
    done
  fi
  python scripts/compare_acerag_renderer_ablation.py --root "$RUN_ROOT"
}

case "$STAGE" in
  smoke)
    N="${N:-5}"
    DATASETS="${DATASETS:-hotpotqa}"
    run_matrix "stage0_smoke" "$N"
    ;;
  stage1)
    N="${N:-200}"
    run_matrix "stage1_n200" "$N"
    ;;
  stage2)
    N="${N:-1000}"
    run_matrix "stage2_n1000" "$N"
    ;;
  stage3)
    N="${N:-1000}"
    run_matrix "stage3_compact_n1000" "$N"
    ;;
  stage4)
    N="${N:-1000}"
    run_matrix "stage4_appendix_4ds" "$N"
    ;;
  *)
    echo "Usage: $0 {smoke|stage1|stage2|stage3|stage4}" >&2
    exit 2
    ;;
esac

echo "RUN_ROOT=$RUN_ROOT"
