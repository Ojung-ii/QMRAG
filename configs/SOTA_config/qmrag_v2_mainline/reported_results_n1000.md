# QMRAG v2 n=1000 Reported Results

This file records the n=1000 result sources used while freezing the reproducible QMRAG v2 mainline config pack. The mainline config in this directory fixes `seed_selection_variant=top_relevance`; HotpotQA and 2Wiki have matching top-relevance n=1000 runs. PopQA and MuSiQue reported sources are the existing full-context n=1000 common/bundle runs from the overnight pack and are listed explicitly for reproducibility.

## Main Common Prompt

`common_qa` is the main fair-comparison setting. It uses `structured_chain` rendering and full context.

| Dataset | n | EM | F1 | Ans in Context | Ans in Pred | Insufficient | CtxTok | InputTok | TotalTok | LatencyMs | Source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hotpotqa | 1000 | 0.371 | 0.4495 | 0.928 | 0.411 | 0.399 | 1700.4 | 1834.7 | 1839.3 | 610.1 | `outputs/hotpotqa/eval/20260524_204859_hotpotqa_seed_top_relevance_n1000/predictions.jsonl` |
| 2wiki | 1000 | 0.146 | 0.1818 | 0.856 | 0.197 | 0.664 | 1727.3 | 1857.3 | 1861.5 | 588.2 | `outputs/2wiki/eval/20260524_213408_2wiki_seed_top_relevance_n1000/predictions.jsonl` |
| popqa | 1000 | 0.397 | 0.4987 | 0.886 | 0.600 | 0.294 | 920.6 | 1854.8 | 1859.2 | 466.7 | `outputs/popqa/eval/20260524_022010/predictions.jsonl` |
| musique | 1000 | 0.065 | 0.0871 | 0.638 | 0.101 | 0.864 | 1466.9 | 2463.9 | 2468.1 | 773.3 | `outputs/musique/eval/20260524_022855/predictions.jsonl` |

## Bundle Prompt Ablation

`qmrag_bundle_qa` is a structured evidence utilization ablation/upper-bound, not the main fair-comparison result.

| Dataset | n | EM | F1 | Ans in Context | Ans in Pred | Insufficient | CtxTok | InputTok | TotalTok | LatencyMs | Source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hotpotqa | 1000 | 0.426 | 0.5234 | 0.928 | 0.482 | 0.283 | 1719.0 | 1906.2 | 1911.0 | 773.4 | `outputs/replay/20260524_123553/hotpotqa/common_qa_to_qmrag_bundle_qa/predictions.jsonl` |
| 2wiki | 1000 | 0.203 | 0.2536 | 0.856 | 0.270 | 0.583 | 1730.9 | 1914.0 | 1918.4 | 630.7 | `outputs/replay/20260524_125139/2wiki/common_qa_to_qmrag_bundle_qa/predictions.jsonl` |
| popqa | 1000 | 0.431 | 0.5366 | 0.886 | 0.626 | 0.215 | 1731.8 | 1907.8 | 1912.1 | 582.3 | `outputs/replay/20260524_130714/popqa/common_qa_to_qmrag_bundle_qa/predictions.jsonl` |
| musique | 1000 | 0.094 | 0.1295 | 0.638 | 0.136 | 0.772 | 2329.4 | 2516.9 | 2521.2 | 850.7 | `outputs/replay/20260524_132524/musique/common_qa_to_qmrag_bundle_qa/predictions.jsonl` |

## Prompt Improvement

The prompt comparison rows below are prompt-only comparisons with `rendered_context_hash_mismatch_count=0`.

| Dataset | Delta EM | Delta F1 | Delta AnsPred | Delta Insufficient | Fixed/Broken |
| --- | ---: | ---: | ---: | ---: | ---: |
| hotpotqa | +0.054 | +0.0739 | +0.072 | -0.118 | 108 / 36 |
| 2wiki | +0.061 | +0.0764 | +0.078 | -0.086 | 107 / 29 |
| popqa | +0.034 | +0.0379 | +0.026 | -0.079 | 54 / 28 |
| musique | +0.029 | +0.0424 | +0.035 | -0.092 | 48 / 13 |

## Retrieval Metrics

These rows follow the QMRAG retrieval metric evaluator and use title-compatible supporting fact matching. PopQA supporting facts are treated in the compatible diagnostic mode used by the evaluator.

| Dataset | Recall@5 | EM | F1 | CtxTok | InputTok | TotalTok | F1/1kInput | SF Prec | SF Recall | SF F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hotpotqa | 0.921 | 0.371 | 0.4495 | 1700.4 | 1834.7 | 1839.3 | 0.2450 | 0.3498 | 0.9630 | 0.5115 |
| 2wiki | 0.856 | 0.146 | 0.1818 | 1727.3 | 1857.3 | 1861.5 | 0.0979 | 0.3921 | 0.8809 | 0.5345 |
| popqa | 0.881 | 0.397 | 0.4987 | 920.6 | 1854.8 | 1859.2 | 0.2689 | 0.2053 | 0.5585 | 0.2994 |
| musique | 0.629 | 0.065 | 0.0871 | 1466.9 | 2463.9 | 2468.1 | 0.0353 | 0.2711 | 0.7302 | 0.3871 |

## Runtime Seed Selection Ablation

`top_relevance` replaces `medoid_current` in the mainline because it preserves F1 while removing almost all seed-selection latency on HotpotQA/2Wiki n=1000. Candidate cap remains an ablation, not the default.

| Dataset | Seed variant | F1 | Retrieval ms | Candidate ms | Seed ms | Total ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| hotpotqa | medoid_current | 0.4496 | 349.4 | 285.1 | 57.603 | 700.6 |
| hotpotqa | top_relevance | 0.4495 | 281.7 | 274.6 | 0.045 | 614.4 |
| hotpotqa | anchor_first | 0.4437 | 286.5 | 279.5 | 0.079 | 633.3 |
| hotpotqa | chain_potential | 0.4213 | 369.5 | 358.5 | 0.185 | 667.2 |
| 2wiki | medoid_current | 0.1771 | 271.2 | 205.7 | 57.444 | 629.9 |
| 2wiki | top_relevance | 0.1818 | 258.2 | 248.0 | 0.056 | 592.1 |
| 2wiki | anchor_first | 0.1791 | 202.3 | 193.7 | 0.103 | 519.6 |
| 2wiki | chain_potential | 0.1716 | 177.1 | 168.8 | 0.151 | 441.9 |

## Interpretation

- `common_qa` is the main fair-comparison setting.
- `qmrag_bundle_qa` is a prompt ablation/upper-bound for structured evidence utilization.
- `top_relevance` is the mainline seed selection because it matches `medoid_current` quality while sharply reducing seed-selection latency.
- Candidate cap is not a default; `candidate_pool_size=60` is retained only as runtime ablation.
- Compact top-3 is an efficiency ablation, not a main result.
- No dataset-specific settings are used in the mainline configs.
