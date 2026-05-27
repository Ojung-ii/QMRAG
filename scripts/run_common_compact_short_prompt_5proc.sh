#!/usr/bin/env bash
set -euo pipefail

LIMIT=100
MODE="--recommended-only"
INCLUDE_NATIVE_COMPACT=0
for arg in "$@"; do
  case "$arg" in
    --recommended-only|--include-all)
      MODE="$arg"
      ;;
    --include-native-compact)
      INCLUDE_NATIVE_COMPACT=1
      ;;
    ''|*[!0-9]*)
      echo "Usage: bash scripts/run_common_compact_short_prompt_5proc.sh [limit] [--recommended-only|--include-all] [--include-native-compact]" >&2
      exit 2
      ;;
    *)
      LIMIT="$arg"
      ;;
  esac
done

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8013/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export VLLM_MODEL="${VLLM_MODEL:-auto}"
export MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-5}"

RUN_TS="$(date +%Y%m%d_%H%M%S)_common_compact_short_prompt"
LOG_DIR="logs/common_compact_short_prompt/${RUN_TS}"
JOBS_TSV="${LOG_DIR}/jobs.tsv"
FAILED_TSV="${LOG_DIR}/failed_jobs.tsv"
PROFILE_STATUS_TSV="${LOG_DIR}/profile_status.tsv"
MASTER_LOG="${LOG_DIR}/master.log"
mkdir -p "$LOG_DIR"
: > "$FAILED_TSV"
: > "$PROFILE_STATUS_TSV"

echo "[START] ${RUN_TS} limit=${LIMIT} mode=${MODE} include_native_compact=${INCLUDE_NATIVE_COMPACT}" | tee -a "$MASTER_LOG"
echo "[ENV] VLLM_BASE_URL=${VLLM_BASE_URL} MAX_PARALLEL_JOBS=${MAX_PARALLEL_JOBS}" | tee -a "$MASTER_LOG"

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

python scripts/diagnose_predictions.py --latest | tee -a "$MASTER_LOG"

python - "$LIMIT" <<'PY' | tee -a "$MASTER_LOG"
import sys
from pathlib import Path

from scripts.replay_generation import find_latest_prediction_with_rendering
from utils.io_utils import read_jsonl

limit = int(sys.argv[1])
datasets = ["hotpotqa", "2wiki", "popqa", "musique"]
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

python - "$MODE" "$LIMIT" "$MAX_PARALLEL_JOBS" "$JOBS_TSV" "$PROFILE_STATUS_TSV" "$INCLUDE_NATIVE_COMPACT" <<'PY'
import sys
from pathlib import Path

from utils.generation import COMPACTION_PROFILES

mode, limit, slots, jobs_path, profile_status_path, include_native = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5]), bool(int(sys.argv[6]))
datasets = ["hotpotqa", "2wiki", "popqa", "musique"]
recommended_groups = [
    ("none", "strict_short_qa", "full_strict_short"),
    ("none", "ace_rag_bundle_short_qa", "full_bundle_short"),
    ("metadata_only_compact", "common_qa", "common_compact"),
    ("chain_dedup", "common_qa", "common_compact"),
    ("top3_chain_dedup", "common_qa", "common_compact"),
]
native_compact_groups = [
    ("chain_dedup", "ace_rag_bundle_short_qa", "native_compact"),
    ("top3_chain_dedup", "ace_rag_compact_chain_short_qa", "native_compact"),
]
all_groups = [
    ("none", "strict_short_qa", "full_strict_short"),
    ("none", "ace_rag_bundle_short_qa", "full_bundle_short"),
    ("metadata_only_compact", "common_qa", "common_compact"),
    ("chain_dedup", "common_qa", "common_compact"),
    ("top3_chain_dedup", "common_qa", "common_compact"),
    ("chain_schema_k3", "common_qa", "common_compact"),
    ("chain_schema_plus1_k3", "common_qa", "common_compact"),
    ("top3_schema_dedup", "common_qa", "common_compact"),
    ("metadata_only_compact", "ace_rag_bundle_short_qa", "native_compact"),
    ("chain_dedup", "ace_rag_bundle_short_qa", "native_compact"),
    ("top3_chain_dedup", "ace_rag_compact_chain_short_qa", "native_compact"),
    ("chain_schema_k3", "ace_rag_compact_chain_short_qa", "native_compact"),
    ("chain_schema_plus1_k3", "ace_rag_compact_chain_short_qa", "native_compact"),
    ("top3_schema_dedup", "ace_rag_compact_chain_short_qa", "native_compact"),
]
profile_aliases = {
    "metadata_only_compact": "metadata_only_compact",
    "chain_dedup": "chain_dedup",
    "top3_chain_dedup": "top3_chain_dedup",
    "chain_schema_k3": "chain_schema_k3",
    "chain_schema_plus1_k3": "chain_schema_plus1_k3",
    "top3_schema_dedup": "top3_schema_dedup",
    "chain_dedup_no_sources": "chain_dedup_no_sources",
    "source_sentence_light": "source_light_compact",
    "chain_support2_no_full_source": "chain_support2_no_full_source",
}
supported = set(COMPACTION_PROFILES)
with profile_status_path.open("w", encoding="utf-8") as handle:
    for requested, actual in profile_aliases.items():
        status = "implemented" if actual in supported else "not implemented"
        handle.write(f"{requested}\t{actual}\t{status}\n")

if mode == "--include-all":
    groups = all_groups
elif include_native:
    groups = recommended_groups + native_compact_groups
else:
    groups = recommended_groups
jobs = []
for profile, prompt, group_mode in groups:
    actual_profile = profile_aliases.get(profile, profile)
    if actual_profile not in supported:
        continue
    for dataset in datasets:
        job_id = len(jobs)
        gpu = 0 if job_id % 5 == 0 else 1
        jobs.append((job_id, gpu, dataset, actual_profile, prompt, limit, group_mode))
with jobs_path.open("w", encoding="utf-8") as handle:
    for row in jobs:
        handle.write("\t".join(str(x) for x in row) + "\n")
PY

echo "[PROFILE_STATUS]" | tee -a "$MASTER_LOG"
sed -n '1,120p' "$PROFILE_STATUS_TSV" | tee -a "$MASTER_LOG"
echo "[JOBS] $(wc -l < "$JOBS_TSV") jobs written to $JOBS_TSV" | tee -a "$MASTER_LOG"

run_worker() {
  local worker_id="$1"
  local worker_log="${LOG_DIR}/worker${worker_id}.log"
  echo "[WORKER ${worker_id}] start" | tee -a "$MASTER_LOG" "$worker_log"
  while IFS=$'\t' read -r job_id gpu dataset profile prompt limit job_mode; do
    if (( job_id % MAX_PARALLEL_JOBS != worker_id )); then
      continue
    fi
    local job_log="${LOG_DIR}/${dataset}_${profile}_${prompt}.log"
    echo "[JOB ${job_id}] START gpu=${gpu} dataset=${dataset} profile=${profile} prompt=${prompt} limit=${limit} mode=${job_mode}" | tee -a "$worker_log" "$MASTER_LOG"
    set +e
    CUDA_VISIBLE_DEVICES="$gpu" \
    VLLM_BASE_URL="$VLLM_BASE_URL" \
    VLLM_API_KEY="$VLLM_API_KEY" \
    VLLM_MODEL="$VLLM_MODEL" \
    python scripts/replay_generation.py \
      --dataset "$dataset" \
      --source-prompt common_qa \
      --source-rendering-profile structured_chain \
      --target-prompt "$prompt" \
      --latest \
      --limit "$limit" \
      --compaction-profile "$profile" \
      --vllm-base-url "$VLLM_BASE_URL" \
      --vllm-api-key "$VLLM_API_KEY" \
      --vllm-model "$VLLM_MODEL" \
      > "$job_log" 2>&1
    local rc=$?
    set -e
    if [[ "$rc" -ne 0 ]]; then
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$job_id" "$gpu" "$dataset" "$profile" "$prompt" "$job_mode" "$rc" >> "$FAILED_TSV"
      echo "[JOB ${job_id}] FAILED rc=${rc} log=${job_log}" | tee -a "$worker_log" "$MASTER_LOG"
      continue
    fi
    local out_path eb_hash ctx_hash prompt_seen
    out_path="$(grep -E '^output: ' "$job_log" | tail -1 | sed 's/^output: //')"
    eb_hash="$(grep -E '"evidence_bundles_hash_match_rate"' "$job_log" | tail -1 | sed -E 's/.*: ([0-9.]+),?/\1/')"
    ctx_hash="$(grep -E '"rendered_context_hash_match_rate"' "$job_log" | tail -1 | sed -E 's/.*: ([0-9.]+),?/\1/')"
    prompt_seen="$(grep -E '"prompt_profile"' "$job_log" | tail -1 | sed -E 's/.*: "([^"]+)".*/\1/')"
    if [[ "$eb_hash" != "1.0" || "$prompt_seen" != "$prompt" ]]; then
      printf "%s\t%s\t%s\t%s\t%s\t%s\tpostcheck_failed_eb=%s_prompt=%s\n" "$job_id" "$gpu" "$dataset" "$profile" "$prompt" "$job_mode" "$eb_hash" "$prompt_seen" >> "$FAILED_TSV"
      echo "[JOB ${job_id}] POSTCHECK_FAILED output=${out_path} eb=${eb_hash} prompt=${prompt_seen} ctx=${ctx_hash}" | tee -a "$worker_log" "$MASTER_LOG"
      continue
    fi
    if [[ "$profile" == "none" && "$ctx_hash" != "1.0" ]]; then
      printf "%s\t%s\t%s\t%s\t%s\t%s\tpostcheck_failed_ctx=%s\n" "$job_id" "$gpu" "$dataset" "$profile" "$prompt" "$job_mode" "$ctx_hash" >> "$FAILED_TSV"
      echo "[JOB ${job_id}] POSTCHECK_FAILED output=${out_path} eb=${eb_hash} prompt=${prompt_seen} ctx=${ctx_hash}" | tee -a "$worker_log" "$MASTER_LOG"
      continue
    fi
    echo "[JOB ${job_id}] END output=${out_path} evidence_bundles_hash_match_rate=${eb_hash} rendered_context_hash_match_rate=${ctx_hash}" | tee -a "$worker_log" "$MASTER_LOG"
  done < "$JOBS_TSV"
  echo "[WORKER ${worker_id}] end" | tee -a "$MASTER_LOG" "$worker_log"
}

for ((worker_id=0; worker_id<MAX_PARALLEL_JOBS; worker_id++)); do
  run_worker "$worker_id" &
done
wait

echo "[DONE] failed_jobs=$(wc -l < "$FAILED_TSV")" | tee -a "$MASTER_LOG"
if [[ -s "$FAILED_TSV" ]]; then
  echo "[FAILED_JOBS]" | tee -a "$MASTER_LOG"
  sed -n '1,160p' "$FAILED_TSV" | tee -a "$MASTER_LOG"
fi
