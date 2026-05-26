#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: bash scripts/run_compact_chain_ablation.sh <hotpotqa|2wiki|popqa|musique|all> <common_qa|qmrag_bundle_qa> [limit]" >&2
  exit 1
fi

DATASET_ARG="$1"
PROMPT_PROFILE="$2"
LIMIT="${3:-100}"

if [[ "$PROMPT_PROFILE" != "common_qa" && "$PROMPT_PROFILE" != "qmrag_bundle_qa" ]]; then
  echo "Unsupported prompt_profile: $PROMPT_PROFILE" >&2
  exit 1
fi

if [[ "$DATASET_ARG" == "all" ]]; then
  DATASETS=(hotpotqa 2wiki popqa musique)
else
  DATASETS=("$DATASET_ARG")
fi

PROFILES=(
  chain_dedup_no_sources
  chain_dedup_plus1_no_sources
  sentence_cap_no_sources
  top3_chain_dedup_no_sources
  chain_skeleton_no_sources
)

for dataset in "${DATASETS[@]}"; do
  for profile in "${PROFILES[@]}"; do
    echo "== compact chain replay: dataset=${dataset} prompt=${PROMPT_PROFILE} profile=${profile} limit=${LIMIT} =="
    cmd=(
      python scripts/replay_generation.py
      --dataset "$dataset"
      --source-prompt common_qa
      --source-rendering-profile structured_chain
      --target-prompt "$PROMPT_PROFILE"
      --latest
      --limit "$LIMIT"
      --compaction-profile "$profile"
    )
    if [[ "$profile" == "sentence_cap_no_sources" ]]; then
      cmd+=(--max-sentences-per-bundle 3)
    fi
    "${cmd[@]}"
  done
done
