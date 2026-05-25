# QMRAG v2 n=1000 Reported Results

This file records the n=1000 result sources used by the reproducible QMRAG v2
mainline config pack after the common-prompt compact replay and short-style
prompt ablation.

## Main Common Prompt

`common_qa` is the primary fair-comparison setting. It uses `structured_chain`
rendering and full context. This remains the main SOTA row.

| Dataset | n | EM | F1 | Ans in Context | Ans in Pred | Insufficient | CtxTok | InputTok | TotalTok | EffectiveMs | Source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hotpotqa | 1000 | 0.372 | 0.4510 | 0.928 | 0.412 | 0.396 | 1700.4 | 1834.7 | 1839.3 | 561.5 | `outputs/hotpotqa/eval/20260525_095722_core_ablation_5proc_hotpotqa_core_qmrag_mainline_n1000/predictions.jsonl` |
| 2wiki | 1000 | 0.144 | 0.1791 | 0.856 | 0.194 | 0.666 | 1727.3 | 1857.3 | 1861.5 | 414.4 | `outputs/2wiki/eval/20260525_095722_core_ablation_5proc_2wiki_core_qmrag_mainline_n1000/predictions.jsonl` |
| popqa | 1000 | 0.405 | 0.5051 | 0.886 | 0.603 | 0.289 | 1724.3 | 1847.2 | 1851.7 | 440.1 | `outputs/popqa/eval/20260525_095722_core_ablation_5proc_popqa_core_qmrag_mainline_n1000/predictions.jsonl` |
| musique | 1000 | 0.070 | 0.0917 | 0.636 | 0.106 | 0.860 | 2289.0 | 2423.5 | 2427.7 | 673.9 | `outputs/musique/eval/20260525_095722_core_ablation_5proc_musique_core_qmrag_mainline_n1000/predictions.jsonl` |

## QMRAG-Compact-common

The paper-facing compact efficiency candidate is `top3_chain_dedup + common_qa`.
It is a replay experiment: retrieval and `evidence_bundles` are unchanged, only
`rendered_context` is compacted. All rows have
`evidence_bundles_hash_match_rate=1.0`.

| Dataset | Best common baseline | Baseline F1 | Compact F1 | Margin | InputTok | F1/1kInput | Token reduction | Ans in Context | Insufficient | Source |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hotpotqa | LightRAG | 0.3229 | 0.3796 | +0.0567 | 593.4 | 0.6397 | 67.2% | 0.823 | 0.474 | `outputs/replay/20260525_144934/hotpotqa/common_qa_to_common_qa_top3_top3_chain_dedup/predictions.jsonl` |
| 2wiki | LightRAG | 0.0953 | 0.1412 | +0.0459 | 593.0 | 0.2381 | 67.2% | 0.702 | 0.710 | `outputs/replay/20260525_144906/2wiki/common_qa_to_common_qa_top3_top3_chain_dedup/predictions.jsonl` |
| popqa | Dense RAG | 0.4167 | 0.4708 | +0.0541 | 559.4 | 0.8417 | 68.8% | 0.840 | 0.333 | `outputs/replay/20260525_144908/popqa/common_qa_to_common_qa_top3_top3_chain_dedup/predictions.jsonl` |
| musique | HippoRAG2 | 0.0551 | 0.0676 | +0.0125 | 738.0 | 0.0915 | 68.7% | 0.412 | 0.889 | `outputs/replay/20260525_144927/musique/common_qa_to_common_qa_top3_top3_chain_dedup/predictions.jsonl` |

## Compact Profile Diagnostics

`metadata_only_compact` preserves the most quality but does not reduce tokens
enough for the compact main claim. `chain_dedup` is a safer compact backup, but
MuSiQue slightly exceeds the `InputTok <= 1000` target. `top3_chain_dedup` is the
recommended paper-facing compact profile.

| Profile | Avg F1 | Avg margin vs best baseline | Avg InputTok | Avg F1/1kInput | Avg token reduction | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| metadata_only_compact | 0.3213 | +0.0988 | 1761.5 | 0.1974 | 12.6% | diagnostic only |
| chain_dedup | 0.2758 | +0.0533 | 885.6 | 0.3246 | 53.7% | backup compact candidate |
| top3_chain_dedup | 0.2648 | +0.0423 | 620.9 | 0.4528 | 68.0% | QMRAG-Compact-common candidate |

## Prompt Ablation

Method/native prompt results are prompt ablations and should be reported in an
appendix or separate native-prompt table, not as the main fair-comparison row.

| Dataset | common_qa F1 | strict_short_qa F1 | Delta | qmrag_bundle_qa F1 | qmrag_bundle_short_qa F1 | Delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hotpotqa | 0.4510 | 0.5481 | +0.0972 | 0.5246 | 0.5690 | +0.0444 |
| 2wiki | 0.1791 | 0.3489 | +0.1697 | 0.2536 | 0.3819 | +0.1284 |
| popqa | 0.5051 | 0.5702 | +0.0650 | 0.5415 | 0.5510 | +0.0095 |
| musique | 0.0917 | 0.1117 | +0.0200 | 0.1309 | 0.1249 | -0.0061 |
| average | 0.3067 | 0.3947 | +0.0880 | 0.3626 | 0.4067 | +0.0441 |

## Interpretation

- Primary mainline remains `common_qa + structured_chain + full context`.
- `QMRAG-Compact-common` should use `top3_chain_dedup + common_qa`.
- `top3_chain_dedup + common_qa` beats the best common-prompt baseline on all
  four datasets while keeping `InputTok <= 800`.
- The compact quality drop is mostly evidence loss from context pruning, not a
  prompt-only issue; MuSiQue has the largest `Ans in Context` drop.
- `strict_short_qa` improves over `common_qa` on all four datasets.
- `qmrag_bundle_short_qa` improves the existing bundle prompt on three of four
  datasets and improves the average, but remains a method-prompt ablation.
- Compact latency should not be claimed from this replay run; recorded
  generation timing is affected by concurrent vLLM load. The robust claim is
  token efficiency plus baseline-beating F1.
- No dataset-specific settings are used in the mainline or compact candidate.
