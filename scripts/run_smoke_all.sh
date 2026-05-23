#!/usr/bin/env bash
set -euo pipefail
LIMIT="${1:-3}"
shift || true
python main.py --config config/smoke.yaml --datasets popqa hotpotqa 2wiki musique --limit "$LIMIT" --no-llm --no-embedding --reindex "$@"
