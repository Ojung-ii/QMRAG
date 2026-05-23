#!/usr/bin/env bash
set -euo pipefail
MAX_JOBS="${MAX_JOBS:-4}" python -m pip install -r requirements-flashattn-optional.txt --no-build-isolation
