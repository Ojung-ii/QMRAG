# ACE-RAG Paper Results Reproduction Map

This file maps the ACE-RAG paper tables and appendix figures to the local
artifacts that produced them. Treat `LIMIT=1000` as the paper full-run setting.
Runs with a smaller `LIMIT` are smoke tests and should not replace the reported
paper numbers.

## Naming Conventions

- `dataset`: one of `hotpotqa`, `2wiki`, `musique`, or `popqa`.
- `prompt_profile`: the QA prompt profile, for example `common_qa` or
  `p8_r0_section_aware`.
- `rendering_profile`: the context rendering profile, usually
  `structured_chain`.
- `compaction_profile`: the replay compaction profile, usually
  `top3_chain_dedup` for the main common-prompt result.
- `source_output`: the original run directory or prediction/eval file used for
  the reported number.
- `aggregate_table`: the final Markdown/CSV table used to assemble the paper
  table.

## Environment

The current SOTA snapshot is:

- Config: `configs/SOTA_config/ace_rag_mainline/ace_rag_current_sota.yaml`
- Main common config: `configs/SOTA_config/ace_rag_mainline/ace_rag_common_top3.yaml`
- Environment template: `configs/SOTA_config/ace_rag_mainline/env.example`
- Generator: `Qwen/Qwen2.5-7B-Instruct`
- Embedding model: `nvidia/NV-Embed-v2`

Reference environment setup:

```bash
source configs/SOTA_config/ace_rag_mainline/env.example
```

Reference vLLM endpoint from the SOTA snapshot:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n vllm vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 127.0.0.1 --port 8013 --gpu-memory-utilization 0.55 \
  --max-model-len 16384 --api-key EMPTY
```

## Table 1: Main Common-Prompt Results

Paper setting: common QA prompt, compact ACE-RAG rendering, `LIMIT=1000`.

Aggregate tables:

- Paper-style table: `outputs/final_baseline_aware/20260526_053914/main_common_with_ace_rag_compact.md`
- Paper-style CSV: `outputs/final_baseline_aware/20260526_053914/main_common_with_ace_rag_compact.csv`
- Row-level source CSV: `outputs/final_baseline_aware/20260526_053914/final_common_prompt_table.csv`

Reported ACE-RAG-Compact rows:

| dataset | R@5 | EM | F1 | context_tokens | source_output |
| --- | ---: | ---: | ---: | ---: | --- |
| hotpotqa | 97.0 | 30.9 | 37.9 | 458 | `outputs/replay/20260526_000338/hotpotqa/common_qa_to_common_qa_top3_top3_chain_dedup/` |
| 2wiki | 88.1 | 11.5 | 14.3 | 462 | `outputs/replay/20260526_001204/2wiki/common_qa_to_common_qa_top3_top3_chain_dedup/` |
| musique | 75.7 | 5.1 | 7.0 | 592 | `outputs/replay/20260525_221850/musique/common_qa_to_common_qa_top3_top3_chain_dedup/` |
| popqa | 56.1 | 38.9 | 47.6 | 436 | `outputs/replay/20260525_220745/popqa/common_qa_to_common_qa_top3_top3_chain_dedup/` |

Each `source_output` directory contains `predictions.jsonl` and `eval.json`.
The baseline rows in `final_common_prompt_table.csv` are marked with
`source=user_supplied_baseline_table`; the final aggregate preserves those
baseline values, but the raw baseline prediction files are not part of the
ACE-RAG output tree.

Reproduce the ACE-RAG common-prompt run:

```bash
source configs/SOTA_config/ace_rag_mainline/env.example
LIMIT=1000 bash configs/SOTA_config/ace_rag_mainline/run_ace_rag_mainline.sh
LIMIT=1000 bash configs/SOTA_config/ace_rag_mainline/run_ace_rag_compact_ablation.sh
```

Use a smaller `LIMIT` only as a smoke test:

```bash
LIMIT=20 bash configs/SOTA_config/ace_rag_mainline/run_ace_rag_mainline.sh hotpotqa
LIMIT=20 bash configs/SOTA_config/ace_rag_mainline/run_ace_rag_compact_ablation.sh hotpotqa
```

## Table 2: Efficiency Analysis

Paper setting: average over the four common-prompt datasets.

Aggregate tables:

- Markdown: `outputs/final_baseline_aware/20260526_053914/final_efficiency_summary.md`
- CSV: `outputs/final_baseline_aware/20260526_053914/final_efficiency_summary.csv`

Reported ACE-RAG-Compact row:

| method | context_tokens | F1_per_1k_context | retrieval_ms | generation_ms |
| --- | ---: | ---: | ---: | ---: |
| ACE-RAG-Compact | 487.2 | 0.5865 | 277.6 | 83.3 |

This row is derived from the same ACE-RAG-Compact source outputs listed in
Table 1. Baseline timing rows are preserved in the aggregate table as
`source=user_supplied_baseline_table`.

## Table 3: Component Ablation

Paper setting: compact common-prompt ablation on HotpotQA and 2Wiki,
`LIMIT=1000`.

Aggregate tables:

- Markdown: `outputs/final_baseline_aware/20260526_053914/ablation_components_table.md`
- CSV: `outputs/final_baseline_aware/20260526_053914/ablation_components_table.csv`
- TeX: `outputs/final_baseline_aware/20260526_053914/ablation_components_table.tex`

Reported rows:

| component | HotpotQA F1 | 2Wiki F1 | local variant |
| --- | ---: | ---: | --- |
| ACE-RAG | 37.9 | 14.3 | `core_ace_rag_mainline` |
| w/o Mention Edge | 30.2 | 10.2 | `core_no_bridge` |
| w/o Residual Cues | 36.6 | 11.7 | `core_bridge_fullquery` |
| w/o Chain Order | 37.6 | 14.2 | `core_no_anchor_ordering` |
| w/o Anchor Bundle | 37.0 | 10.7 | `core_no_multi_anchor` |

The exact compact-ablation paper rows are preserved in the aggregate files
above. The related full common-prompt ablation outputs follow this pattern:

```text
outputs/{dataset}/eval/20260525_095722_core_ablation_5proc_{dataset}_{local_variant}_n1000/
```

For example:

```text
outputs/hotpotqa/eval/20260525_095722_core_ablation_5proc_hotpotqa_core_no_bridge_n1000/predictions.jsonl
outputs/2wiki/eval/20260525_095722_core_ablation_5proc_2wiki_core_no_multi_anchor_n1000/predictions.jsonl
```

## Appendix: Native Prompt Experiments

Paper setting: ACE-RAG section-aware native prompt, top-8 structured-chain
rendering, controlled sequential generation on GPU 0.

Aggregate tables:

- Markdown: `outputs/final_baseline_aware/20260526_053914/final_native_prompt_table.md`
- CSV: `outputs/final_baseline_aware/20260526_053914/final_native_prompt_table.csv`
- Expanded source table: `outputs/final_baseline_aware/20260526_053914/final_native_prompt_expanded_table.csv`

Controlled ACE-RAG native source outputs:

| dataset | R@5 | EM | F1 | input_tokens | source_output |
| --- | ---: | ---: | ---: | ---: | --- |
| hotpotqa | 97.0 | 50.4 | 63.4 | 985 | `outputs/native_timing_controlled/20260526_gpu0_seq055_native_top8/hotpotqa_native_p8_top8_gpu0_seq/p8_r0_section_aware/predictions.jsonl` |
| 2wiki | 88.1 | 36.0 | 43.4 | 993 | `outputs/native_timing_controlled/20260526_gpu0_seq055_native_top8/2wiki_native_p8_top8_gpu0_seq/p8_r0_section_aware/predictions.jsonl` |
| musique | 75.7 | 19.7 | 27.7 | 1199 | `outputs/native_timing_controlled/20260526_gpu0_seq055_native_top8/musique_native_p8_top8_gpu0_seq/p8_r0_section_aware/predictions.jsonl` |
| popqa | 56.1 | 46.8 | 59.8 | 1017 | `outputs/native_timing_controlled/20260526_gpu0_seq055_native_top8/popqa_native_p8_top8_gpu0_seq/p8_r0_section_aware/predictions.jsonl` |

The full native reproduction commands are recorded in:

```text
configs/SOTA_config/ace_rag_mainline/ace_rag_current_sota.yaml
```

Use the `native_prompt_appendix_sota.reproduction_commands` block with
`RUN_ID=20260526_gpu0_seq055_native_top8`.

## Appendix: Quality-Efficiency Budget Trade-Off

Paper figure: ACE-RAG budget trade-off on HotpotQA and 2Wiki.

Aggregate artifacts:

- Summary Markdown: `outputs/analysis/20260526_074635/full_structured_budget_scaling_summary.md`
- Summary JSON: `outputs/analysis/20260526_074635/full_structured_budget_scaling_summary.json`
- Run log root: `logs/full_structured_budget_scaling/20260526_073941_full_structured_budget_scaling`
- Figure PDF: `figures/ace_rag_budget_tradeoff.pdf`
- Figure PNG: `figures/ace_rag_budget_tradeoff.png`

Source outputs:

| dataset | budget_point | F1 | context_tokens | source_output |
| --- | --- | ---: | ---: | --- |
| hotpotqa | top3_chain_dedup | 37.8 | 458.2 | `outputs/replay/20260526_074257/hotpotqa/common_qa_to_common_qa_top3_top3_chain_dedup/predictions.jsonl` |
| hotpotqa | budget500 | 36.4 | 459.0 | `outputs/replay/20260526_074307/hotpotqa/common_qa_to_common_qa_ctx500_full_structured_budget/predictions.jsonl` |
| hotpotqa | budget1000 | 39.0 | 929.2 | `outputs/replay/20260526_074326/hotpotqa/common_qa_to_common_qa_ctx1000_full_structured_budget/predictions.jsonl` |
| hotpotqa | budget1500 | 42.1 | 1342.6 | `outputs/replay/20260526_074326/hotpotqa/common_qa_to_common_qa_ctx1500_full_structured_budget/predictions.jsonl` |
| hotpotqa | budget2000 | 43.3 | 1574.4 | `outputs/replay/20260526_074335/hotpotqa/common_qa_to_common_qa_ctx2000_full_structured_budget/predictions.jsonl` |
| hotpotqa | full_structured_full | 45.0 | 1700.4 | `outputs/replay/20260526_074314/hotpotqa/common_qa_to_common_qa/predictions.jsonl` |
| 2wiki | top3_chain_dedup | 14.0 | 462.1 | `outputs/replay/20260526_074514/2wiki/common_qa_to_common_qa_top3_top3_chain_dedup/predictions.jsonl` |
| 2wiki | budget500 | 15.2 | 461.4 | `outputs/replay/20260526_074535/2wiki/common_qa_to_common_qa_ctx500_full_structured_budget/predictions.jsonl` |
| 2wiki | budget1000 | 14.4 | 927.0 | `outputs/replay/20260526_074603/2wiki/common_qa_to_common_qa_ctx1000_full_structured_budget/predictions.jsonl` |
| 2wiki | budget1500 | 17.1 | 1323.6 | `outputs/replay/20260526_074622/2wiki/common_qa_to_common_qa_ctx1500_full_structured_budget/predictions.jsonl` |
| 2wiki | budget2000 | 16.7 | 1569.4 | `outputs/replay/20260526_074621/2wiki/common_qa_to_common_qa_ctx2000_full_structured_budget/predictions.jsonl` |
| 2wiki | full_structured_full | 17.7 | 1727.3 | `outputs/replay/20260526_074617/2wiki/common_qa_to_common_qa/predictions.jsonl` |

The budget-scaling workflow is:

```bash
LIMIT=1000 bash scripts/run_full_structured_budget_scaling.sh
```

Use a smaller `LIMIT` only to validate that the script and endpoint are working.

## Appendix: Coverage-Answerability Scatter

Paper figure: Recall@5 vs F1 and context tokens vs F1 under the common prompt.

Primary source tables:

- `outputs/final_baseline_aware/20260526_053914/final_common_prompt_table.csv`
- `outputs/final_baseline_aware/20260526_053914/main_common_with_ace_rag_compact.csv`

Generated figures:

- `figures/common_prompt_answerability_2panel.pdf`
- `figures/common_prompt_answerability_2panel.png`
- `figures/common_prompt_recall_f1_scatter.pdf`
- `figures/common_prompt_ctx_f1_scatter.pdf`

## Regenerating Aggregate Tables

After rerunning the paper outputs, regenerate the final baseline-aware tables:

```bash
python scripts/aggregate_final_baseline_aware_results.py \
  --root outputs/final_baseline_aware/20260526_053914 \
  --out-dir outputs/final_baseline_aware/20260526_053914
```

Sanity-check the current SOTA summary:

```bash
sed -n '1,220p' configs/SOTA_config/ace_rag_mainline/ace_rag_current_sota.yaml
sed -n '1,220p' configs/SOTA_config/ace_rag_mainline/reported_results_n1000.md
```

## Reproducibility Notes

- The paper-facing ACE-RAG rows are all `n=1000`.
- Smoke-test outputs are useful for CI or endpoint checks, but they should be
  labelled as smoke results in any table or output directory.
- On 2026-05-27, code/docs/configs and text output metadata were checked for
  legacy pre-ACE project strings and normalized to ACE-RAG naming. `HippoRAG2`
  remains only as a cited baseline method.
- The SOTA output directories listed above were preserved during output cleanup.
- Do not commit `outputs/`, `logs/`, model caches, datasets, or generated
  prediction files. Commit only configs, scripts, docs, and small source files
  needed to reproduce the runs.

## Smoke Reproduction Check

The SOTA compact replay path was revalidated after the rename cleanup with a
small generation smoke test using vLLM at `--gpu-memory-utilization 0.25`.

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n vllm vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 127.0.0.1 --port 8025 --gpu-memory-utilization 0.25 \
  --max-model-len 8192 --api-key EMPTY

python scripts/replay_generation.py \
  --predictions outputs/replay/20260526_074314/hotpotqa/common_qa_to_common_qa/predictions.jsonl \
  --dataset hotpotqa \
  --source-prompt common_qa \
  --source-rendering-profile structured_chain \
  --target-prompt common_qa \
  --limit 20 \
  --compaction-profile top3_chain_dedup \
  --vllm-base-url http://127.0.0.1:8025/v1 \
  --vllm-api-key EMPTY \
  --vllm-model auto \
  --temperature 0 \
  --max-tokens 64 \
  --output-root outputs/smoke_ace_rag_sota_repro_20260527
```

Smoke output:

```text
outputs/smoke_ace_rag_sota_repro_20260527/20260527_155006/hotpotqa/common_qa_to_common_qa_top3_top3_chain_dedup/predictions.jsonl
```

Compared with the preserved SOTA compact prefix:

```text
outputs/replay/20260526_074257/hotpotqa/common_qa_to_common_qa_top3_top3_chain_dedup/predictions.jsonl
```

Result: first 20 IDs matched, normalized predictions matched `20/20`, raw
predictions matched `20/20`, and both runs produced `EM=0.3500`,
`F1=0.4872`, and `context_tokens=479.4` on the smoke subset.
