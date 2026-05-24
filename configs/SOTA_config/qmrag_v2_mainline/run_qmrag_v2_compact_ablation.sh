#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
DATASETS=(hotpotqa 2wiki popqa musique)
LIMIT="${LIMIT:-1000}"
DRY_RUN=false

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
  shift
fi

if [[ "$#" -gt 0 ]]; then
  DATASETS=("$@")
fi

cd "${ROOT_DIR}"

echo "[QMRAG v2 compact ablation] Uses replay_generation.py with --top-bundles 3."
echo "[QMRAG v2 compact ablation] Retrieval is not rerun; require evidence_bundles_hash_match_rate=1.0."

for dataset in "${DATASETS[@]}"; do
  for target_prompt in common_qa qmrag_bundle_qa; do
    cmd=(
      python scripts/replay_generation.py
      --dataset "${dataset}"
      --source-prompt common_qa
      --source-rendering-profile structured_chain
      --target-prompt "${target_prompt}"
      --latest
      --limit "${LIMIT}"
      --top-bundles 3
    )
    echo "[QMRAG v2 compact ${dataset} ${target_prompt}] ${cmd[*]}"
    if [[ "${DRY_RUN}" == "false" ]]; then
      "${cmd[@]}"
    fi
  done
done
