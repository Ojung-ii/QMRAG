#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: bash scripts/run_chain_compaction_ablation.sh <hotpotqa|2wiki|popqa|musique|all> <common_qa|ace_rag_bundle_qa> [limit]" >&2
}

if [[ $# -lt 2 ]]; then
  usage
  exit 1
fi

DATASET_ARG="$1"
PROMPT="$2"
LIMIT="${3:-100}"

if [[ "$PROMPT" != "common_qa" && "$PROMPT" != "ace_rag_bundle_qa" ]]; then
  usage
  exit 1
fi

if [[ "$DATASET_ARG" == "all" ]]; then
  DATASETS=(hotpotqa 2wiki popqa musique)
else
  DATASETS=("$DATASET_ARG")
fi

PROFILES=(chain_dedup chain_skeleton chain_plus1 sentence_cap top3_chain_dedup)

for dataset in "${DATASETS[@]}"; do
  for profile in "${PROFILES[@]}"; do
    echo "[chain-compaction] dataset=${dataset} prompt=${PROMPT} profile=${profile} limit=${LIMIT}"
    cmd=(
      python scripts/replay_generation.py
      --dataset "${dataset}"
      --source-prompt common_qa
      --source-rendering-profile structured_chain
      --target-prompt "${PROMPT}"
      --latest
      --limit "${LIMIT}"
      --compaction-profile "${profile}"
    )
    if [[ "${profile}" == "sentence_cap" ]]; then
      cmd+=(--max-sentences-per-bundle 3)
    fi
    "${cmd[@]}"
  done
done
