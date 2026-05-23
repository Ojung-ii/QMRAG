#!/usr/bin/env bash
set -euo pipefail
export NVEMBED_MODEL_PATH="${NVEMBED_MODEL_PATH:-/home/dilab/.cache/huggingface/models--nvidia--NV-Embed-v2/snapshots/3fa59658547db50a1e8e3346cf057fd0c77ed6ef/}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8011/v1}"
export VLLM_MODEL="${VLLM_MODEL:-auto}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
python main.py --config config/default.yaml --datasets popqa hotpotqa 2wiki musique --reindex "$@"
