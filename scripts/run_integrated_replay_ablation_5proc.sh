#!/usr/bin/env bash
set -euo pipefail

LIMIT="${1:-100}"
MODE="${2:-}"
if [[ "${MODE:-}" != "" && "$MODE" != "--include-aggressive" && "$MODE" != "--recommended-only" ]]; then
  echo "Usage: bash scripts/run_integrated_replay_ablation_5proc.sh [limit] [--include-aggressive|--recommended-only]" >&2
  exit 1
fi

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8013/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export VLLM_MODEL="${VLLM_MODEL:-auto}"
export MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-5}"

RUN_TS="$(date +%Y%m%d_%H%M%S)_integrated_replay_ablation"
LOG_DIR="logs/integrated_replay_ablation/${RUN_TS}"
ANALYSIS_DIR="outputs/analysis/${RUN_TS}"
mkdir -p "$LOG_DIR" "$ANALYSIS_DIR"
MASTER_LOG="${LOG_DIR}/master.log"
JOBS_TSV="${LOG_DIR}/jobs.tsv"
FAILED_TSV="${LOG_DIR}/failed_jobs.tsv"
: > "$MASTER_LOG"
: > "$FAILED_TSV"

echo "[START] ${RUN_TS} limit=${LIMIT} mode=${MODE:-default}" | tee -a "$MASTER_LOG"
echo "[ENV] VLLM_BASE_URL=${VLLM_BASE_URL} MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS}" | tee -a "$MASTER_LOG"

python - <<'PY' | tee -a "$MASTER_LOG"
import os
from openai import OpenAI
base=os.environ.get("VLLM_BASE_URL","http://localhost:8013/v1")
key=os.environ.get("VLLM_API_KEY","EMPTY")
client=OpenAI(base_url=base, api_key=key, timeout=30)
models=client.models.list()
print("base:", base)
print("models:", [m.id for m in models.data][:5])
if not models.data:
    raise SystemExit("No models returned by vLLM endpoint")
PY

LIMIT_ENV="$LIMIT" python - <<'PY' | tee -a "$MASTER_LOG"
import os, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from scripts.replay_generation import find_latest_prediction_with_rendering
from utils.io_utils import read_jsonl

limit=int(os.environ["LIMIT_ENV"])
for dataset in ("hotpotqa","2wiki","popqa","musique"):
    path=find_latest_prediction_with_rendering(Path("outputs"), dataset, "common_qa", "structured_chain")
    rows=read_jsonl(path)
    if len(rows) < limit:
        raise SystemExit(f"{dataset}: source has n={len(rows)} < limit={limit}: {path}")
    first=rows[0]
    required=("evidence_bundles","rendered_context")
    missing=[k for k in required if not first.get(k)]
    if missing:
        raise SystemExit(f"{dataset}: source missing {missing}: {path}")
    raw_none=sum(1 for row in rows[:limit] if row.get("raw_prediction") is None)
    if raw_none:
        raise SystemExit(f"{dataset}: raw_prediction None count in prefix={raw_none}: {path}")
    print(f"source_ok dataset={dataset} n={len(rows)} path={path}")
PY

python - <<PY > "$JOBS_TSV"
datasets=["hotpotqa","2wiki","popqa","musique"]
mode="${MODE:-default}"
jobs=[]
base=[
    ("none","strict_short_qa","format"),
]
core=[
    ("chain_schema_k3","common_qa","compact_common"),
    ("chain_schema_k3","qmrag_compact_chain_qa","compact_prompt"),
    ("chain_schema_k3","qmrag_compact_chain_light","compact_light"),
    ("chain_schema_plus1_k3","qmrag_compact_chain_qa","compact_plus1"),
    ("chain_schema_k5","qmrag_compact_chain_qa","compact_k5"),
    ("top3_schema_dedup","qmrag_compact_chain_qa","top3_schema"),
]
aggressive=[
    ("chain_schema_k2","qmrag_compact_chain_qa","compact_k2"),
    ("chain_schema_plus1_k2","qmrag_compact_chain_qa","compact_plus1_k2"),
]
recommended=[
    ("none","strict_short_qa","format"),
    ("chain_schema_k3","qmrag_compact_chain_qa","compact_prompt"),
    ("chain_schema_plus1_k3","qmrag_compact_chain_qa","compact_plus1"),
    ("top3_schema_dedup","qmrag_compact_chain_qa","top3_schema"),
]
profiles = recommended if mode=="--recommended-only" else base+core+(aggressive if mode=="--include-aggressive" else [])
job_id=0
for dataset in datasets:
    for compaction,prompt,kind in profiles:
        gpu=0 if job_id % 5 == 0 else 1
        print(f"{job_id}\\t{gpu}\\t{dataset}\\t{compaction}\\t{prompt}\\t${LIMIT}\\t{kind}")
        job_id+=1
PY

echo "[JOBS] $(wc -l < "$JOBS_TSV") jobs written to $JOBS_TSV" | tee -a "$MASTER_LOG"

run_worker() {
  local worker_id="$1"
  local worker_log="${LOG_DIR}/worker${worker_id}.log"
  echo "[WORKER ${worker_id}] start" | tee -a "$MASTER_LOG" "$worker_log"
  while IFS=$'\t' read -r job_id gpu dataset compaction target_prompt limit kind; do
    if (( job_id % 5 != worker_id )); then
      continue
    fi
    local job_name="${job_id}_${dataset}_${compaction}_${target_prompt}"
    local job_log="${LOG_DIR}/${job_name}.log"
    echo "[JOB ${job_id}] START gpu=${gpu} dataset=${dataset} compaction=${compaction} prompt=${target_prompt} limit=${limit}" | tee -a "$worker_log" "$MASTER_LOG"
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" \
    VLLM_BASE_URL="$VLLM_BASE_URL" \
    VLLM_API_KEY="$VLLM_API_KEY" \
    VLLM_MODEL="$VLLM_MODEL" \
    python scripts/replay_generation.py \
      --dataset "$dataset" \
      --source-prompt common_qa \
      --source-rendering-profile structured_chain \
      --target-prompt "$target_prompt" \
      --latest \
      --limit "$limit" \
      --compaction-profile "$compaction" \
      --vllm-base-url "$VLLM_BASE_URL" \
      --vllm-api-key "$VLLM_API_KEY" \
      --vllm-model "$VLLM_MODEL" \
      > "$job_log" 2>&1
    local rc=$?
    set -e
    if [[ "$rc" -ne 0 ]]; then
      printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$job_id" "$gpu" "$dataset" "$compaction" "$target_prompt" "$rc" >> "$FAILED_TSV"
      echo "[JOB ${job_id}] FAILED rc=${rc} log=${job_log}" | tee -a "$worker_log" "$MASTER_LOG"
    else
      local out_path
      out_path="$(grep -E '^output: ' "$job_log" | tail -1 | sed 's/^output: //')"
      local eb_hash
      eb_hash="$(grep -E 'evidence_bundles_hash_match_rate' "$job_log" | tail -1 | sed 's/[ ,]//g' || true)"
      local ctx_hash
      ctx_hash="$(grep -E 'rendered_context_hash_match_rate' "$job_log" | tail -1 | sed 's/[ ,]//g' || true)"
      echo "[JOB ${job_id}] END output=${out_path} ${eb_hash} ${ctx_hash}" | tee -a "$worker_log" "$MASTER_LOG"
    fi
  done < "$JOBS_TSV"
  echo "[WORKER ${worker_id}] end" | tee -a "$MASTER_LOG" "$worker_log"
}

for wid in 0 1 2 3 4; do
  run_worker "$wid" &
done
wait

echo "[SUMMARY] generating comparisons" | tee -a "$MASTER_LOG"
python scripts/compare_compact_chain_prompt_runs.py --all-latest --analysis-dir "$ANALYSIS_DIR" | tee -a "$MASTER_LOG"
for dataset in hotpotqa 2wiki popqa musique; do
  python scripts/diagnose_output_format.py --dataset "$dataset" --prompt-profile common_qa --latest --analysis-dir "$ANALYSIS_DIR" | tee -a "$MASTER_LOG"
  python scripts/diagnose_output_format.py --dataset "$dataset" --prompt-profile strict_short_qa --latest --analysis-dir "$ANALYSIS_DIR" | tee -a "$MASTER_LOG"
  python scripts/diagnose_output_format.py --dataset "$dataset" --prompt-profile qmrag_compact_chain_qa --latest --analysis-dir "$ANALYSIS_DIR" | tee -a "$MASTER_LOG" || true
done

RUN_TS="$RUN_TS" LIMIT="$LIMIT" MODE="${MODE:-default}" LOG_DIR="$LOG_DIR" ANALYSIS_DIR="$ANALYSIS_DIR" FAILED_TSV="$FAILED_TSV" python - <<'PY'
import os
from pathlib import Path
analysis=Path(os.environ["ANALYSIS_DIR"])
parts=[
    "# Integrated Replay Ablation Summary",
    "",
    f"- run_ts: {os.environ['RUN_TS']}",
    f"- limit: {os.environ['LIMIT']}",
    f"- mode: {os.environ['MODE']}",
    f"- logs: {os.environ['LOG_DIR']}",
    "",
]
summary=analysis/"compact_chain_prompt_compare_summary.md"
if summary.exists():
    parts.append(summary.read_text(encoding="utf-8"))
failed=Path(os.environ["FAILED_TSV"])
parts.extend(["", "## Failed Jobs", ""])
if failed.exists() and failed.read_text(encoding="utf-8").strip():
    parts.append("```")
    parts.append(failed.read_text(encoding="utf-8").strip())
    parts.append("```")
else:
    parts.append("None")
(analysis/"integrated_replay_ablation_summary.md").write_text("\\n".join(parts)+"\\n", encoding="utf-8")
print("wrote:", analysis/"integrated_replay_ablation_summary.md")
PY

echo "[DONE] analysis=${ANALYSIS_DIR}" | tee -a "$MASTER_LOG"
if [[ -s "$FAILED_TSV" ]]; then
  echo "[WARN] failed jobs recorded in $FAILED_TSV" | tee -a "$MASTER_LOG"
  exit 2
fi
