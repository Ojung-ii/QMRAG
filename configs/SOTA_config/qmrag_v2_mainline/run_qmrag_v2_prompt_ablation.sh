#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG="${SCRIPT_DIR}/qmrag_v2_bundle_full.yaml"
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

for dataset in "${DATASETS[@]}"; do
  cmd=(
    python main.py
    --config "${CONFIG}"
    --datasets "${dataset}"
    --limit "${LIMIT}"
    --prompt-profile qmrag_bundle_qa
    --rendering-profile structured_chain
    --retrieval-variant full_hetero
    --seed-selection-variant top_relevance
    --enable-timing
  )
  echo "[QMRAG v2 prompt ablation] ${cmd[*]}"
  if [[ "${DRY_RUN}" == "false" ]]; then
    "${cmd[@]}"
  fi
done
