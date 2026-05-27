#!/usr/bin/env bash
set -uo pipefail

MODE="sequential"
PREFLIGHT_ONLY=0
LIMIT="${LIMIT:-1000}"

for arg in "$@"; do
  case "$arg" in
    --sequential)
      MODE="sequential"
      ;;
    --parallel)
      MODE="parallel"
      ;;
    --preflight)
      PREFLIGHT_ONLY=1
      ;;
    ''|*[!0-9]*)
      echo "Usage: bash scripts/run_full_structured_budget_scaling.sh [limit] [--preflight] [--sequential|--parallel]" >&2
      exit 2
      ;;
    *)
      LIMIT="$arg"
      ;;
  esac
done

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES_LIST="${CUDA_VISIBLE_DEVICES_LIST:-${CUDA_VISIBLE_DEVICES}}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8013/v1}"
export VLLM_BASE_URL_GPU0="${VLLM_BASE_URL_GPU0:-${VLLM_BASE_URL}}"
export VLLM_BASE_URL_GPU1="${VLLM_BASE_URL_GPU1:-${VLLM_BASE_URL}}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export VLLM_MODEL="${VLLM_MODEL:-auto}"
if [[ "$MODE" == "sequential" ]]; then
  export MAX_PARALLEL_JOBS=1
else
  export MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-5}"
fi

DATASETS=(hotpotqa 2wiki)
if [[ -n "${DATASETS_OVERRIDE:-}" ]]; then
  # shellcheck disable=SC2206
  DATASETS=(${DATASETS_OVERRIDE})
fi
BUDGETS=(500 1000 1500 2000)
if [[ -n "${BUDGETS_OVERRIDE:-}" ]]; then
  # shellcheck disable=SC2206
  BUDGETS=(${BUDGETS_OVERRIDE})
fi
# shellcheck disable=SC2206
GPU_LIST=(${CUDA_VISIBLE_DEVICES_LIST})

RUN_TS="$(date +%Y%m%d_%H%M%S)_full_structured_budget_scaling"
LOG_DIR="logs/full_structured_budget_scaling/${RUN_TS}"
JOBS_TSV="${LOG_DIR}/jobs.tsv"
FAILED_TSV="${LOG_DIR}/failed_jobs.tsv"
MASTER_LOG="${LOG_DIR}/master.log"
FAILED_PARTS="${LOG_DIR}/failed_parts"
mkdir -p "$LOG_DIR" "$FAILED_PARTS"
: > "$FAILED_TSV"

echo "[START] ${RUN_TS} mode=${MODE} limit=${LIMIT}" | tee -a "$MASTER_LOG"
echo "[ENV] CUDA_VISIBLE_DEVICES_LIST=${CUDA_VISIBLE_DEVICES_LIST} VLLM_BASE_URL_GPU0=${VLLM_BASE_URL_GPU0} VLLM_BASE_URL_GPU1=${VLLM_BASE_URL_GPU1} MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS}" | tee -a "$MASTER_LOG"
if [[ "$MODE" == "parallel" ]]; then
  echo "[NOTE] Parallel mode is for screening only. Use sequential mode for timing claims." | tee -a "$MASTER_LOG"
fi

python - <<'PY' | tee -a "$MASTER_LOG"
import os
from openai import OpenAI

key = os.environ.get("VLLM_API_KEY", "EMPTY")
endpoints = []
for gpu in ("0", "1"):
    base = os.environ.get(f"VLLM_BASE_URL_GPU{gpu}")
    if base and base not in endpoints:
        endpoints.append(base)
if not endpoints:
    endpoints.append(os.environ.get("VLLM_BASE_URL", "http://localhost:8013/v1"))
print("CUDA_VISIBLE_DEVICES_LIST:", os.environ.get("CUDA_VISIBLE_DEVICES_LIST"))
for base in endpoints:
    client = OpenAI(base_url=base, api_key=key, timeout=30)
    models = client.models.list()
    print("base:", base)
    print("models:", [m.id for m in models.data][:5])
PY
if [[ "${PIPESTATUS[0]}" -ne 0 ]]; then
  echo "[ABORT] vLLM endpoint preflight failed" | tee -a "$MASTER_LOG"
  exit 1
fi

python scripts/diagnose_predictions.py --latest | tee -a "$MASTER_LOG"

python - "$LIMIT" "${DATASETS[@]}" <<'PY' | tee -a "$MASTER_LOG"
import sys
from pathlib import Path

from scripts.replay_generation import find_latest_prediction_with_rendering
from utils.io_utils import read_jsonl

limit = int(sys.argv[1])
datasets = sys.argv[2:]
for dataset in datasets:
    path = find_latest_prediction_with_rendering(Path("outputs"), dataset, "common_qa", "structured_chain")
    rows = read_jsonl(path)
    if len(rows) < limit:
        raise SystemExit(f"{dataset}: source n={len(rows)} < limit={limit}: {path}")
    prefix = rows[:limit]
    checks = {
        "bad_prompt": sum(1 for row in prefix if str(row.get("prompt_profile") or "") != "common_qa"),
        "bad_rendering": sum(1 for row in prefix if str(row.get("rendering_profile") or "structured_chain") != "structured_chain"),
        "compact": sum(1 for row in prefix if row.get("context_compaction_enabled") or str(row.get("compaction_profile") or "none") != "none"),
        "truncated": sum(1 for row in prefix if row.get("context_truncation_enabled") or row.get("top_bundles") is not None or row.get("context_token_budget") is not None),
        "missing_bundles": sum(1 for row in prefix if not row.get("evidence_bundles")),
        "missing_context": sum(1 for row in prefix if not str(row.get("rendered_context") or "").strip()),
        "raw_none": sum(1 for row in prefix if row.get("raw_prediction") is None),
    }
    failed = {key: value for key, value in checks.items() if value}
    if failed:
        raise SystemExit(f"{dataset}: invalid common_qa full-context source {path}: {failed}")
    print(f"source_ok dataset={dataset} n={len(rows)} path={path}")
PY
if [[ "${PIPESTATUS[0]}" -ne 0 ]]; then
  echo "[ABORT] source validation failed" | tee -a "$MASTER_LOG"
  exit 1
fi

if [[ "$PREFLIGHT_ONLY" == "1" ]]; then
  echo "[DONE] preflight_only=1 failed_jobs=0" | tee -a "$MASTER_LOG"
  exit 0
fi

job_id=0
: > "$JOBS_TSV"
for dataset in "${DATASETS[@]}"; do
  gpu="${GPU_LIST[$((job_id % ${#GPU_LIST[@]}))]}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$job_id" "$gpu" "$dataset" "top3" "top3_chain_dedup" "common_qa" "$LIMIT" "" >> "$JOBS_TSV"
  job_id=$((job_id + 1))
  for budget in "${BUDGETS[@]}"; do
    gpu="${GPU_LIST[$((job_id % ${#GPU_LIST[@]}))]}"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$job_id" "$gpu" "$dataset" "full_structured_budget_${budget}" "full_structured_budget" "common_qa" "$LIMIT" "$budget" >> "$JOBS_TSV"
    job_id=$((job_id + 1))
  done
  gpu="${GPU_LIST[$((job_id % ${#GPU_LIST[@]}))]}"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$job_id" "$gpu" "$dataset" "full_common_replay" "none" "common_qa" "$LIMIT" "" >> "$JOBS_TSV"
  job_id=$((job_id + 1))
done

export LOG_DIR FAILED_PARTS MASTER_LOG FAILED_TSV LIMIT VLLM_BASE_URL VLLM_BASE_URL_GPU0 VLLM_BASE_URL_GPU1 VLLM_API_KEY VLLM_MODEL

run_one() {
  local job_id="$1"
  local gpu="$2"
  local dataset="$3"
  local setting="$4"
  local profile="$5"
  local prompt="$6"
  local limit="$7"
  local budget="$8"
  local log_file="${LOG_DIR}/${dataset}_${setting}.log"
  local base_url="$VLLM_BASE_URL"
  if [[ "$gpu" == "0" ]]; then
    base_url="${VLLM_BASE_URL_GPU0:-$VLLM_BASE_URL}"
  elif [[ "$gpu" == "1" ]]; then
    base_url="${VLLM_BASE_URL_GPU1:-$VLLM_BASE_URL}"
  fi
  echo "[JOB ${job_id}] START gpu=${gpu} endpoint=${base_url} dataset=${dataset} setting=${setting} profile=${profile} prompt=${prompt} limit=${limit} budget=${budget}" >> "$MASTER_LOG"
  local cmd=(
    python scripts/replay_generation.py
    --dataset "$dataset"
    --source-prompt common_qa
    --source-rendering-profile structured_chain
    --target-prompt "$prompt"
    --latest
    --limit "$limit"
    --compaction-profile "$profile"
    --vllm-base-url "$base_url"
    --vllm-api-key "$VLLM_API_KEY"
    --vllm-model "$VLLM_MODEL"
  )
  if [[ -n "$budget" ]]; then
    cmd+=(--context-token-budget "$budget")
  fi
  if CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$log_file" 2>&1; then
    local output_path
    output_path="$(grep '^output:' "$log_file" | tail -n 1 | sed 's/^output: //')"
    echo "[JOB ${job_id}] END output=${output_path}" >> "$MASTER_LOG"
  else
    local status="$?"
    echo -e "${job_id}\t${gpu}\t${dataset}\t${setting}\t${profile}\t${prompt}\t${limit}\t${budget}\tstatus=${status}\tlog=${log_file}" > "${FAILED_PARTS}/${job_id}.tsv"
    echo "[JOB ${job_id}] FAIL status=${status} log=${log_file}" >> "$MASTER_LOG"
  fi
}
export -f run_one

if [[ "$MODE" == "sequential" ]]; then
  while IFS=$'\t' read -r job_id gpu dataset setting profile prompt limit budget; do
    run_one "$job_id" "$gpu" "$dataset" "$setting" "$profile" "$prompt" "$limit" "$budget"
  done < "$JOBS_TSV"
else
  active_jobs=0
  while IFS=$'\t' read -r job_id gpu dataset setting profile prompt limit budget; do
    run_one "$job_id" "$gpu" "$dataset" "$setting" "$profile" "$prompt" "$limit" "$budget" &
    active_jobs=$((active_jobs + 1))
    if [[ "$active_jobs" -ge "$MAX_PARALLEL_JOBS" ]]; then
      wait -n
      active_jobs=$((active_jobs - 1))
    fi
  done < "$JOBS_TSV"
  wait
fi

cat "${FAILED_PARTS}"/*.tsv 2>/dev/null > "$FAILED_TSV" || : > "$FAILED_TSV"
failed_count="$(wc -l < "$FAILED_TSV" | tr -d ' ')"
echo "[DONE] failed_jobs=${failed_count}" | tee -a "$MASTER_LOG"
