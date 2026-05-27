#!/usr/bin/env bash
set -euo pipefail

# =========================
# ACE-RAG overnight full run
# =========================
# Purpose:
# 1. Rendering ablation for current ACE-RAG
# 2. Full runs over 4 datasets with common_qa
# 3. Full runs over 4 datasets with ace_rag_bundle_qa
# 4. Prompt comparison and failure analysis
#
# Important:
# - Do NOT force reindex by default.
# - Use existing bridge/dense index unless FORCE_REINDEX=1.
# - common_qa is the main fair-comparison prompt.
# - ace_rag_bundle_qa is ablation prompt.

export NVEMBED_MODEL_PATH="${NVEMBED_MODEL_PATH:-/home/dilab/.cache/huggingface/models--nvidia--NV-Embed-v2/snapshots/3fa59658547db50a1e8e3346cf057fd0c77ed6ef/}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8011/v1}"
export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
export VLLM_MODEL="${VLLM_MODEL:-auto}"

RENDERING_PROFILE="${RENDERING_PROFILE:-structured_chain}"
FORCE_REINDEX="${FORCE_REINDEX:-0}"

DATASETS=("hotpotqa" "2wiki" "popqa" "musique")
PROMPTS=("common_qa" "ace_rag_bundle_qa")

mkdir -p logs
LOG="logs/overnight_full_all_prompts_$(date +%Y%m%d_%H%M%S).log"

run_and_log() {
  echo "" | tee -a "$LOG"
  echo "[CMD] $*" | tee -a "$LOG"
  "$@" 2>&1 | tee -a "$LOG"
}

echo "[START] $(date)" | tee -a "$LOG"
echo "[INFO] branch/commit" | tee -a "$LOG"
git branch --show-current 2>&1 | tee -a "$LOG" || true
git log --oneline -1 2>&1 | tee -a "$LOG" || true
git status --short 2>&1 | tee -a "$LOG" || true

echo "[INFO] env" | tee -a "$LOG"
echo "NVEMBED_MODEL_PATH=${NVEMBED_MODEL_PATH}" | tee -a "$LOG"
echo "VLLM_BASE_URL=${VLLM_BASE_URL}" | tee -a "$LOG"
echo "VLLM_MODEL=${VLLM_MODEL}" | tee -a "$LOG"
echo "RENDERING_PROFILE=${RENDERING_PROFILE}" | tee -a "$LOG"
echo "FORCE_REINDEX=${FORCE_REINDEX}" | tee -a "$LOG"

echo "[CHECK] python/cuda" | tee -a "$LOG"
python - <<'PY' 2>&1 | tee -a "$LOG"
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY

echo "[CHECK] compile" | tee -a "$LOG"
run_and_log python -m compileall main.py utils scripts

echo "[CHECK] vLLM" | tee -a "$LOG"
run_and_log python scripts/check_vllm.py

echo "[CHECK] NV-Embed path" | tee -a "$LOG"
run_and_log python scripts/check_nvembed.py --skip-load

# =====================================
# 1. Original planned rendering ablation
# =====================================
echo "" | tee -a "$LOG"
echo "==============================" | tee -a "$LOG"
echo "[PHASE 1] Rendering ablation" | tee -a "$LOG"
echo "==============================" | tee -a "$LOG"

# These are replay experiments using existing retrieval outputs.
# They are useful for deciding whether structured_chain should remain the main rendering.
if [[ -x scripts/run_rendering_ablation.sh ]]; then
  run_and_log bash scripts/run_rendering_ablation.sh hotpotqa || true
  run_and_log bash scripts/run_rendering_ablation.sh 2wiki || true
else
  echo "[WARN] scripts/run_rendering_ablation.sh not found/executable. Skipping rendering ablation." | tee -a "$LOG"
fi

# =====================================
# 2. Full runs: common_qa and ace_rag
# =====================================
echo "" | tee -a "$LOG"
echo "==============================" | tee -a "$LOG"
echo "[PHASE 2] Full dataset runs" | tee -a "$LOG"
echo "==============================" | tee -a "$LOG"

for PROMPT in "${PROMPTS[@]}"; do
  echo "" | tee -a "$LOG"
  echo "[PROMPT] ${PROMPT}" | tee -a "$LOG"

  for DS in "${DATASETS[@]}"; do
    echo "" | tee -a "$LOG"
    echo "[RUN] dataset=${DS} prompt=${PROMPT} rendering=${RENDERING_PROFILE} $(date)" | tee -a "$LOG"

    CMD=(bash scripts/run_dataset.sh "$DS" --prompt-profile "$PROMPT" --rendering-profile "$RENDERING_PROFILE")

    if [[ "$FORCE_REINDEX" == "1" ]]; then
      CMD+=(--force-reindex)
    fi

    run_and_log "${CMD[@]}"

    echo "[DONE] dataset=${DS} prompt=${PROMPT} $(date)" | tee -a "$LOG"

    # Lightweight live diagnostics after each run
    run_and_log python scripts/diagnose_predictions.py --latest || true
  done
done

# =====================================
# 3. Prompt comparison
# =====================================
echo "" | tee -a "$LOG"
echo "==============================" | tee -a "$LOG"
echo "[PHASE 3] Prompt comparison" | tee -a "$LOG"
echo "==============================" | tee -a "$LOG"

for DS in "${DATASETS[@]}"; do
  echo "[COMPARE] ${DS}: common_qa vs ace_rag_bundle_qa" | tee -a "$LOG"
  run_and_log python scripts/compare_prompt_runs.py \
    --dataset "$DS" \
    --left-prompt common_qa \
    --right-prompt ace_rag_bundle_qa \
    --latest || true
done

# =====================================
# 4. Failure analysis
# =====================================
echo "" | tee -a "$LOG"
echo "==============================" | tee -a "$LOG"
echo "[PHASE 4] Failure analysis" | tee -a "$LOG"
echo "==============================" | tee -a "$LOG"

if [[ -x scripts/run_failure_analysis.sh ]]; then
  run_and_log bash scripts/run_failure_analysis.sh || true
else
  echo "[WARN] scripts/run_failure_analysis.sh not found/executable. Running manual failure analysis." | tee -a "$LOG"
  for DS in "${DATASETS[@]}"; do
    for PROMPT in "${PROMPTS[@]}"; do
      run_and_log python scripts/analyze_failures.py \
        --dataset "$DS" \
        --prompt-profile "$PROMPT" \
        --latest \
        --sample 10 || true
    done
  done
fi

# =====================================
# 5. Final diagnostics
# =====================================
echo "" | tee -a "$LOG"
echo "==============================" | tee -a "$LOG"
echo "[PHASE 5] Final diagnostics" | tee -a "$LOG"
echo "==============================" | tee -a "$LOG"

run_and_log python scripts/diagnose_predictions.py --latest

echo "[END] $(date)" | tee -a "$LOG"
echo "[LOG] ${LOG}" | tee -a "$LOG"
