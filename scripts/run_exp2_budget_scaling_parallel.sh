#!/usr/bin/env bash
set -uo pipefail

LIMIT="${LIMIT:-1000}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8013/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export VLLM_MODEL="${VLLM_MODEL:-auto}"
export MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-5}"

DATASETS=(hotpotqa 2wiki)
BUDGETS=(1000 1500 2000 2500)

RUN_TS="$(date +%Y%m%d_%H%M%S)_exp2_budget_parallel"
LOG_DIR="logs/exp2_budget_parallel/${RUN_TS}"
JOBS_TSV="${LOG_DIR}/jobs.tsv"
FAILED_TSV="${LOG_DIR}/failed_jobs.tsv"
MASTER_LOG="${LOG_DIR}/master.log"
FAILED_PARTS="${LOG_DIR}/failed_parts"
mkdir -p "$LOG_DIR" "$FAILED_PARTS"
: > "$FAILED_TSV"

echo "[START] ${RUN_TS} limit=${LIMIT}" | tee -a "$MASTER_LOG"
echo "[ENV] VLLM_BASE_URL=${VLLM_BASE_URL} MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS}" | tee -a "$MASTER_LOG"
echo "[NOTE] Parallel screening run: generation_ms/total_ms from this run must not be used as paper timing evidence." | tee -a "$MASTER_LOG"
echo "[NOTE] CUDA_VISIBLE_DEVICES is assigned to replay client workers; actual inference GPU follows the vLLM server bound to ${VLLM_BASE_URL}." | tee -a "$MASTER_LOG"

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
  for budget in "${BUDGETS[@]}"; do
    mod=$((job_id % 5))
    if [[ "$mod" -le 2 ]]; then gpu=0; else gpu=1; fi
    printf "%s\t%s\t%s\t%s\n" "$job_id" "$gpu" "$dataset" "$budget" >> "$JOBS_TSV"
    job_id=$((job_id + 1))
  done
done

export LOG_DIR FAILED_PARTS MASTER_LOG FAILED_TSV LIMIT VLLM_BASE_URL VLLM_API_KEY VLLM_MODEL

run_one() {
  local job_id="$1"
  local gpu="$2"
  local dataset="$3"
  local budget="$4"
  local log_file="${LOG_DIR}/${dataset}_budget${budget}.log"
  echo "[JOB ${job_id}] START gpu=${gpu} dataset=${dataset} budget=${budget}" >> "$MASTER_LOG"
  if CUDA_VISIBLE_DEVICES="$gpu" python scripts/replay_generation.py \
      --dataset "$dataset" \
      --source-prompt common_qa \
      --source-rendering-profile structured_chain \
      --target-prompt common_qa \
      --latest \
      --limit "$LIMIT" \
      --compaction-profile chain_dedup_budget \
      --context-token-budget "$budget" \
      --vllm-base-url "$VLLM_BASE_URL" \
      --vllm-api-key "$VLLM_API_KEY" \
      --vllm-model "$VLLM_MODEL" \
      > "$log_file" 2>&1; then
    local output_path
    output_path="$(grep '^output:' "$log_file" | tail -n 1 | sed 's/^output: //')"
    echo "[JOB ${job_id}] END output=${output_path}" >> "$MASTER_LOG"
  else
    local status="$?"
    echo -e "${job_id}\t${gpu}\t${dataset}\t${budget}\tstatus=${status}\tlog=${log_file}" > "${FAILED_PARTS}/${job_id}.tsv"
    echo "[JOB ${job_id}] FAIL status=${status} log=${log_file}" >> "$MASTER_LOG"
  fi
}
export -f run_one

xargs -P "$MAX_PARALLEL_JOBS" -n 4 bash -c 'run_one "$@"' _ < "$JOBS_TSV"

cat "${FAILED_PARTS}"/*.tsv 2>/dev/null > "$FAILED_TSV" || : > "$FAILED_TSV"
failed_count="$(wc -l < "$FAILED_TSV" | tr -d ' ')"
echo "[DONE] failed_jobs=${failed_count}" | tee -a "$MASTER_LOG"
