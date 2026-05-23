#!/usr/bin/env bash
set -euo pipefail
export NVEMBED_MODEL_PATH="${NVEMBED_MODEL_PATH:-/home/dilab/.cache/huggingface/models--nvidia--NV-Embed-v2/snapshots/3fa59658547db50a1e8e3346cf057fd0c77ed6ef/}"
if [[ "$#" -eq 0 ]]; then set -- popqa; fi
python main.py --config config/default.yaml --mode index --datasets "$@"
