# QMRAG v2 Mainline SOTA Config Pack

## Overview

This directory freezes the reproducible QMRAG v2 mainline and its paper-facing ablations. It does not change retrieval logic, ranking, prompts, or prediction post-processing. It only records the selected settings and provides scripts to rerun them.

## Mainline Setting

Main result uses `common_qa` + `structured_chain` + full context.

- `retrieval_variant`: `full_hetero`
- `seed_selection_variant`: `top_relevance`
- `candidate_pool_size`: default/full, no candidate cap
- `bridge.enabled`: `true`
- `bridge.selection`: `residual_lexical`
- `bridge.ordering`: `anchor_chain_aware`
- `rendering_profile`: `structured_chain`
- `prompt_profile`: `common_qa`
- `top_bundles`: unset
- `context_token_budget`: unset

This remains the primary fair-comparison mainline. The compact result is a
paper-facing efficiency setting, not a replacement for the main SOTA row.

Run:

```bash
source configs/SOTA_config/qmrag_v2_mainline/env.example
bash configs/SOTA_config/qmrag_v2_mainline/run_qmrag_v2_mainline.sh
```

## Prompt Ablation

QMRAG-bundle is an ablation/upper-bound for structured evidence utilization. It uses the same retrieval, context, and rendering as mainline, but swaps the prompt profile to `qmrag_bundle_qa`.

```bash
bash configs/SOTA_config/qmrag_v2_mainline/run_qmrag_v2_prompt_ablation.sh
```

## Compact Efficiency Ablation

The paper-facing compact setting is `top3_chain_dedup + common_qa` across all
datasets. This is the recommended QMRAG-Compact-common result because the n=1000
run beats the best common-prompt baseline on all four datasets while keeping
`InputTok <= 800`. It is not the primary fair-comparison mainline. The script
uses `replay_generation.py` so retrieval is not rerun and
`evidence_bundles_hash_match_rate` should remain `1.0`.

```bash
bash configs/SOTA_config/qmrag_v2_mainline/run_qmrag_v2_compact_ablation.sh
```

Optional compact prompt ablations can be replayed by overriding the prompt list:

```bash
TARGET_PROMPTS="common_qa qmrag_bundle_qa" bash configs/SOTA_config/qmrag_v2_mainline/run_qmrag_v2_compact_ablation.sh
```

## Runtime Ablation

Runtime ablation uses `candidate_pool_size=60` across all datasets. This is not the main default and should be reported separately as an efficiency/runtime ablation.

```bash
bash configs/SOTA_config/qmrag_v2_mainline/run_qmrag_v2_runtime_ablation.sh
```

## Reproduction Commands

Dry-run commands:

```bash
bash configs/SOTA_config/qmrag_v2_mainline/run_qmrag_v2_mainline.sh --dry-run
bash configs/SOTA_config/qmrag_v2_mainline/run_qmrag_v2_prompt_ablation.sh --dry-run
bash configs/SOTA_config/qmrag_v2_mainline/run_qmrag_v2_compact_ablation.sh --dry-run
bash configs/SOTA_config/qmrag_v2_mainline/run_qmrag_v2_runtime_ablation.sh --dry-run
```

Limit override:

```bash
LIMIT=100 bash configs/SOTA_config/qmrag_v2_mainline/run_qmrag_v2_mainline.sh hotpotqa 2wiki
```

Collect summaries:

```bash
bash configs/SOTA_config/qmrag_v2_mainline/collect_qmrag_v2_results.sh
```

## Expected Outputs

Main and runtime runs write under `outputs/{dataset}/eval/{timestamp}` and include:

- `predictions.jsonl`
- `eval.json`
- `eval_summary.md`
- `timing_events.jsonl`
- `timing_summary.json`
- `timing_summary.md`

Replay ablations write under `outputs/replay/{timestamp}/{dataset}/...`.

## What Not To Do

- Do not change `common_qa` for main fair-comparison runs.
- Do not use QMRAG-bundle as the main fair-comparison result.
- Do not replace the primary full-context mainline with compact results in the
  main SOTA table; report QMRAG-Compact-common as an efficiency result.
- Do not use dataset-specific caps or context settings.
- Do not commit `outputs/`, `data/`, `cache/`, `logs/`, zip files, pyc files, or `__pycache__/`.
- Do not add full KG/OpenIE/PPR/keyword embedding features to this config pack.

## Reported Result Sources

See `reported_results_n1000.md` for n=1000 tables and source output paths. See `manifest.json` for the frozen settings, source paths, and config hashes.
