#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "Usage: bash scripts/run_context_budget_ablation.sh <hotpotqa|2wiki|popqa|musique|all> <common_qa|ace_rag_bundle_qa> [limit] [--limit N] [--include-raw-score]" >&2
  exit 2
fi

TARGET="$1"
PROMPT="$2"
shift 2

LIMIT="1000"
INCLUDE_RAW_SCORE="false"
if [[ "$#" -gt 0 && "$1" != --* ]]; then
  LIMIT="$1"
  shift
fi
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --include-raw-score)
      INCLUDE_RAW_SCORE="true"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$TARGET" == "all" ]]; then
  DATASETS=(hotpotqa 2wiki popqa musique)
else
  DATASETS=("$TARGET")
fi

ANALYSIS_DIR="${ANALYSIS_DIR:-outputs/analysis/$(date +%Y%m%d_%H%M%S)_context_budget}"
mkdir -p "$ANALYSIS_DIR"

ORDERINGS=(current)
if [[ "$INCLUDE_RAW_SCORE" == "true" ]]; then
  ORDERINGS+=(raw_score)
fi

echo "analysis_dir=$ANALYSIS_DIR prompt=$PROMPT limit=$LIMIT datasets=${DATASETS[*]} orderings=${ORDERINGS[*]}"

latest_replay_path() {
  local dataset="$1"
  local prompt="$2"
  local mode="$3"
  local ordering="$4"
  python - "$dataset" "$prompt" "$mode" "$ordering" <<'PY'
import json
import sys
from pathlib import Path

dataset, prompt, mode, ordering = sys.argv[1:5]
candidates = []
for path in Path("outputs/replay").rglob("predictions.jsonl"):
    try:
        first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    except Exception:
        continue
    if first.get("dataset") != dataset or first.get("prompt_profile") != prompt:
        continue
    if not first.get("context_truncation_enabled"):
        continue
    if str(first.get("ordering_source") or "current") != ordering:
        continue
    if mode.startswith("top:"):
        if str(first.get("top_bundles")) != mode.split(":", 1)[1]:
            continue
        if first.get("context_token_budget") is not None:
            continue
    elif mode.startswith("ctx:"):
        if str(first.get("context_token_budget")) != mode.split(":", 1)[1]:
            continue
    else:
        continue
    candidates.append((path.stat().st_mtime, path))
if not candidates:
    raise SystemExit(f"missing replay for {dataset} {prompt} {mode}")
print(max(candidates)[1])
PY
}

for dataset in "${DATASETS[@]}"; do
  for ordering in "${ORDERINGS[@]}"; do
    for top_k in 1 2 3 5; do
      python scripts/replay_generation.py \
        --dataset "$dataset" \
        --source-prompt "$PROMPT" \
        --source-rendering-profile structured_chain \
        --target-prompt "$PROMPT" \
        --latest \
        --limit "$LIMIT" \
        --top-bundles "$top_k" \
        --ordering-source "$ordering"
      right_path="$(latest_replay_path "$dataset" "$PROMPT" "top:$top_k" "$ordering")"
      python scripts/compare_context_budget_runs.py \
        --dataset "$dataset" \
        --prompt-profile "$PROMPT" \
        --left-full latest \
        --right-truncated "$right_path" \
        --analysis-dir "$ANALYSIS_DIR"
    done

    for budget in 256 384 512 768 1024; do
      python scripts/replay_generation.py \
        --dataset "$dataset" \
        --source-prompt "$PROMPT" \
        --source-rendering-profile structured_chain \
        --target-prompt "$PROMPT" \
        --latest \
        --limit "$LIMIT" \
        --context-token-budget "$budget" \
        --ordering-source "$ordering"
      right_path="$(latest_replay_path "$dataset" "$PROMPT" "ctx:$budget" "$ordering")"
      python scripts/compare_context_budget_runs.py \
        --dataset "$dataset" \
        --prompt-profile "$PROMPT" \
        --left-full latest \
        --right-truncated "$right_path" \
        --analysis-dir "$ANALYSIS_DIR"
    done
  done
done

python - "$ANALYSIS_DIR" <<'PY'
import json
import sys
from pathlib import Path

analysis_dir = Path(sys.argv[1])
rows = []
for path in sorted(analysis_dir.glob("context_budget_compare_*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data["summary"]
    rows.append(summary)

def fmt(value):
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)

headers = [
    "dataset",
    "prompt_profile",
    "truncation",
    "ordering_source",
    "EM",
    "F1",
    "answer_in_prediction",
    "answer_in_rendered_context",
    "insufficient_rate",
    "CtxTok",
    "InputTok",
    "TotalTok",
    "F1_per_1k_input_tokens",
    "token_reduction_rate",
    "fixed_by_right",
    "broken_by_right",
]
lines = [
    "# Context Budget Ablation Summary",
    "",
    "| " + " | ".join(headers) + " |",
    "| " + " | ".join(["---"] * len(headers)) + " |",
]
for row in rows:
    truncation = f"top_bundles={row.get('top_bundles')}" if row.get("top_bundles") is not None else f"context_token_budget={row.get('context_token_budget')}"
    values = [
        row.get("dataset"),
        row.get("prompt_profile"),
        truncation,
        row.get("ordering_source"),
        row.get("right_EM"),
        row.get("right_F1"),
        row.get("right_answer_in_prediction"),
        row.get("right_answer_in_rendered_context"),
        row.get("right_insufficient_rate"),
        row.get("right_CtxTok"),
        row.get("right_InputTok"),
        row.get("right_TotalTok"),
        row.get("right_F1_per_1k_input_prompt_tokens"),
        row.get("token_reduction_rate"),
        row.get("fixed_by_right"),
        row.get("broken_by_right"),
    ]
    lines.append("| " + " | ".join(fmt(v) for v in values) + " |")
lines.extend([
    "",
    "## Notes",
    "",
    "- Retrieval and ranking are not rerun; each replay reuses stored evidence_bundles.",
    "- `token_reduction_rate` is computed from full-context input prompt tokens.",
    "- `ordering_source=current` preserves ACE-RAG bundle ordering before top-k/budget selection.",
])
(analysis_dir / "context_budget_ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(analysis_dir / "context_budget_ablation_summary.md")
PY

echo "wrote analysis to $ANALYSIS_DIR"
