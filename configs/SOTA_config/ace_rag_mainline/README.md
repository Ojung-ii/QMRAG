# ACE-RAG Mainline SOTA Config Pack

## Overview

This directory freezes the reproducible ACE-RAG mainline and its paper-facing ablations. It does not change retrieval logic, ranking, prompts, or prediction post-processing. It only records the selected settings and provides scripts to rerun them.

## Current Paper SOTA Snapshot

The current paper-facing SOTA snapshot is recorded in
`ace_rag_current_sota.yaml`.

- Main controlled comparison: `ACE-RAG-Compact = common_qa + r0_current structured_chain + top3_chain_dedup`.
- Native appendix result: `ACE-RAG native = p8_r0_section_aware + r0_current structured_chain + top8`.
- The earlier dawn prompt check `p2_relaxed_chain` is kept as an intermediate
  verification/source-context run, not as the final native SOTA setting.

## Full Retrieval Source Setting

The compact common-prompt result is replayed from the full common retrieval
source. That source uses `common_qa` + `structured_chain` + full context.

- `retrieval_variant`: `full_hetero`
- `seed_selection_variant`: `global_seed_search`
- `candidate_pool_size`: default/full, no candidate cap
- `bridge.enabled`: `true`
- `bridge.selection`: `residual_lexical`
- `bridge.ordering`: `anchor_chain_aware`
- `rendering_profile`: `structured_chain`
- `prompt_profile`: `common_qa`
- `top_bundles`: unset
- `context_token_budget`: unset

This full source remains the retrieval/evidence source for compact replay. The
paper main common-prompt row should report the compact top-3 rendering because
it beats the best common-prompt baseline on all four datasets with the shortest
context.

Run:

```bash
source configs/SOTA_config/ace_rag_mainline/env.example
bash configs/SOTA_config/ace_rag_mainline/run_ace_rag_mainline.sh
```

## Prompt Ablation

ACE-RAG-bundle is an ablation/upper-bound for structured evidence utilization. It uses the same retrieval, context, and rendering as mainline, but swaps the prompt profile to `ace_rag_bundle_qa`.

```bash
bash configs/SOTA_config/ace_rag_mainline/run_ace_rag_prompt_ablation.sh
```

## Main Common-Prompt Compact Setting

The main paper setting is `top3_chain_dedup + common_qa` across all datasets.
This is the recommended `ACE-RAG-Compact` common-prompt result because the
n=1000 run beats the best common-prompt baseline on all four datasets while
keeping the shortest context. The script uses `replay_generation.py` so
retrieval is not rerun and `evidence_bundles_hash_match_rate` should remain
`1.0`.

```bash
bash configs/SOTA_config/ace_rag_mainline/run_ace_rag_compact_ablation.sh
```

Optional compact prompt ablations can be replayed by overriding the prompt list:

```bash
TARGET_PROMPTS="common_qa ace_rag_bundle_qa" bash configs/SOTA_config/ace_rag_mainline/run_ace_rag_compact_ablation.sh
```

## Runtime Ablation

Runtime ablation uses `candidate_pool_size=60` across all datasets. This is not the main default and should be reported separately as an efficiency/runtime ablation.

```bash
bash configs/SOTA_config/ace_rag_mainline/run_ace_rag_runtime_ablation.sh
```

## Reproduction Commands

Dry-run commands:

```bash
bash configs/SOTA_config/ace_rag_mainline/run_ace_rag_mainline.sh --dry-run
bash configs/SOTA_config/ace_rag_mainline/run_ace_rag_prompt_ablation.sh --dry-run
bash configs/SOTA_config/ace_rag_mainline/run_ace_rag_compact_ablation.sh --dry-run
bash configs/SOTA_config/ace_rag_mainline/run_ace_rag_runtime_ablation.sh --dry-run
```

Final SOTA snapshot:

```bash
cat configs/SOTA_config/ace_rag_mainline/ace_rag_current_sota.yaml
```

Limit override:

```bash
LIMIT=100 bash configs/SOTA_config/ace_rag_mainline/run_ace_rag_mainline.sh hotpotqa 2wiki
```

Collect summaries:

```bash
bash configs/SOTA_config/ace_rag_mainline/collect_ace_rag_results.sh
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
- Do not use ACE-RAG-bundle as the main fair-comparison result.
- Do not report method-specific/native prompt rows as the main controlled
  comparison.
- Do not use `p2_relaxed_chain` as the final native SOTA; use
  `p8_r0_section_aware + r0_current + top8`.
- Do not use dataset-specific caps or context settings.
- Do not commit `outputs/`, `data/`, `cache/`, `logs/`, zip files, pyc files, or `__pycache__/`.
- Do not add full KG/OpenIE/PPR/keyword embedding features to this config pack.

## Reported Result Sources

See `ace_rag_current_sota.yaml` for the current frozen SOTA settings. See
`reported_results_n1000.md` for n=1000 tables and source output paths. See
`manifest.json` for config hashes and source paths.
