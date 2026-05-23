#!/usr/bin/env bash
set -euo pipefail
python main.py --config config/smoke.yaml --datasets popqa --limit 5 --no-llm --no-embedding --reindex "$@"
