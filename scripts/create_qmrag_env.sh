#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-QMRAG}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CUDA_VERSION="${CUDA_VERSION:-12.1}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
NVEMBED_MODEL_PATH="${NVEMBED_MODEL_PATH:-/home/dilab/.cache/huggingface/models--nvidia--NV-Embed-v2/snapshots/3fa59658547db50a1e8e3346cf057fd0c77ed6ef/}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda command not found" >&2
  exit 1
fi

eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Reusing conda env $ENV_NAME"
else
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION" pip
fi
conda activate "$ENV_NAME"
python -m pip install -U pip wheel setuptools
conda install -y -c pytorch -c nvidia "pytorch==2.2.0" "pytorch-cuda=$CUDA_VERSION"
python -m pip install -r requirements.txt
python -m pip uninstall -y transformer-engine || true

if [[ "$INSTALL_FLASH_ATTN" == "1" ]]; then
  echo "Installing optional flash-attn==2.2.0. This may compile locally."
  MAX_JOBS="${MAX_JOBS:-4}" python -m pip install -r requirements-flashattn-optional.txt --no-build-isolation
else
  echo "Skipping optional flash-attn. Set INSTALL_FLASH_ATTN=1 if your NV-Embed snapshot requires it."
fi

export NVEMBED_MODEL_PATH
python scripts/check_env.py
python scripts/check_nvembed.py --skip-load
