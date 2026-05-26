#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: bash scripts/run_safe_compact_ablation.sh <hotpotqa|2wiki|popqa|musique|all> <common_qa|qmrag_bundle_qa> [limit]" >&2
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
  metadata_only_compact
  chain_dedup_keep_sources
  source_light_compact
)

for dataset in "${DATASETS[@]}"; do
  for profile in "${PROFILES[@]}"; do
    echo "== safe compact replay: dataset=${dataset} prompt=${PROMPT_PROFILE} profile=${profile} limit=${LIMIT} =="
    python scripts/replay_generation.py \
      --dataset "$dataset" \
      --source-prompt common_qa \
      --source-rendering-profile structured_chain \
      --target-prompt "$PROMPT_PROFILE" \
      --latest \
      --limit "$LIMIT" \
      --compaction-profile "$profile"
  done
done
