#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATASETS=(hotpotqa 2wiki popqa musique)
DRY_RUN=false

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
  shift
fi

if [[ "$#" -gt 0 ]]; then
  DATASETS=("$@")
fi

cd "${ROOT_DIR}"

run_cmd() {
  echo "[collect] $*"
  if [[ "${DRY_RUN}" == "false" ]]; then
    "$@"
  fi
}

run_optional() {
  echo "[collect optional] $*"
  if [[ "${DRY_RUN}" == "false" ]]; then
    "$@" || echo "[collect optional] command failed but collection continues: $*"
  fi
}

run_cmd python scripts/diagnose_predictions.py --latest

for dataset in "${DATASETS[@]}"; do
  run_cmd python scripts/evaluate_qmrag_retrieval_metrics.py \
    --dataset "${dataset}" \
    --prompt-profile common_qa \
    --latest
  run_optional python scripts/compare_prompt_runs.py \
    --dataset "${dataset}" \
    --left-prompt common_qa \
    --right-prompt qmrag_bundle_qa \
    --latest
done

for dataset in hotpotqa 2wiki; do
  run_optional python scripts/compare_seed_selection_ablation.py --dataset "${dataset}" --latest
  run_optional python scripts/compare_candidate_cap_ablation.py --dataset "${dataset}" --latest
done
