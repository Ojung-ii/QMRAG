#!/usr/bin/env bash
set -uo pipefail

LIMIT="${LIMIT:-1000}"
RECHECK_BUDGETS_TEXT="${RECHECK_BUDGETS:-1000 1500}"
read -r -a RECHECK_BUDGETS <<< "$RECHECK_BUDGETS_TEXT"
DATASETS=(hotpotqa 2wiki)

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8013/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export VLLM_MODEL="${VLLM_MODEL:-auto}"
export MAX_PARALLEL_JOBS=1

RUN_TS="$(date +%Y%m%d_%H%M%S)_exp2_budget_timing_recheck"
LOG_DIR="logs/exp2_budget_timing_recheck/${RUN_TS}"
JOBS_TSV="${LOG_DIR}/jobs.tsv"
FAILED_TSV="${LOG_DIR}/failed_jobs.tsv"
MASTER_LOG="${LOG_DIR}/master.log"
mkdir -p "$LOG_DIR"
: > "$FAILED_TSV"

echo "[START] ${RUN_TS} limit=${LIMIT} recheck_budgets=${RECHECK_BUDGETS_TEXT}" | tee -a "$MASTER_LOG"
echo "[ENV] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} VLLM_BASE_URL=${VLLM_BASE_URL} MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS}" | tee -a "$MASTER_LOG"
echo "[NOTE] Sequential timing recheck: only this timing should be used if budget scaling timing is reported." | tee -a "$MASTER_LOG"

python - <<'PY' | tee -a "$MASTER_LOG"
import os
from openai import OpenAI

base = os.environ.get("VLLM_BASE_URL", "http://localhost:8013/v1")
key = os.environ.get("VLLM_API_KEY", "EMPTY")
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

job_id=0
: > "$JOBS_TSV"
for dataset in "${DATASETS[@]}"; do
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$job_id" "$CUDA_VISIBLE_DEVICES" "$dataset" "top3_chain_dedup" "top3_chain_dedup" "common_qa" "$LIMIT" "" >> "$JOBS_TSV"
  job_id=$((job_id + 1))
  for budget in "${RECHECK_BUDGETS[@]}"; do
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$job_id" "$CUDA_VISIBLE_DEVICES" "$dataset" "budget${budget}" "chain_dedup_budget" "common_qa" "$LIMIT" "$budget" >> "$JOBS_TSV"
    job_id=$((job_id + 1))
  done
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$job_id" "$CUDA_VISIBLE_DEVICES" "$dataset" "full_common_replay" "none" "common_qa" "$LIMIT" "" >> "$JOBS_TSV"
  job_id=$((job_id + 1))
done

while IFS=$'\t' read -r job_id gpu dataset budget_point profile prompt limit budget; do
  log_file="${LOG_DIR}/${dataset}_${budget_point}.log"
  echo "[JOB ${job_id}] START gpu=${gpu} dataset=${dataset} budget_point=${budget_point} profile=${profile} prompt=${prompt} budget=${budget}" | tee -a "$MASTER_LOG"
  cmd=(
    python scripts/replay_generation.py
    --dataset "$dataset"
    --source-prompt common_qa
    --source-rendering-profile structured_chain
    --target-prompt "$prompt"
    --latest
    --limit "$limit"
    --compaction-profile "$profile"
    --vllm-base-url "$VLLM_BASE_URL"
    --vllm-api-key "$VLLM_API_KEY"
    --vllm-model "$VLLM_MODEL"
  )
  if [[ -n "$budget" ]]; then
    cmd+=(--context-token-budget "$budget")
  fi
  CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$log_file" 2>&1
  status=$?
  if [[ "$status" -eq 0 ]]; then
    output_path="$(grep '^output:' "$log_file" | tail -n 1 | sed 's/^output: //')"
    echo "[JOB ${job_id}] END output=${output_path}" | tee -a "$MASTER_LOG"
  else
    echo -e "${job_id}\t${gpu}\t${dataset}\t${budget_point}\t${profile}\t${prompt}\t${limit}\t${budget}\tstatus=${status}\tlog=${log_file}" >> "$FAILED_TSV"
    echo "[JOB ${job_id}] FAIL status=${status} log=${log_file}" | tee -a "$MASTER_LOG"
  fi
done < "$JOBS_TSV"

failed_count="$(wc -l < "$FAILED_TSV" | tr -d ' ')"
echo "[DONE] failed_jobs=${failed_count}" | tee -a "$MASTER_LOG"
