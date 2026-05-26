# QMRAG v2 n=1000 Reported Results

This file records the n=1000 result sources used by the reproducible QMRAG v2
mainline config pack after the common-prompt compact replay and short-style
prompt ablation.

## Main Common Prompt

`common_qa` is the primary fair-comparison prompt. The paper main row reports
`ACE-RAG-Compact`, which replays the full common retrieval source with
`top3_chain_dedup` rendering.

| Dataset | n | EM | F1 | Ans in Context | Ans in Pred | Insufficient | CtxTok | InputTok | TotalTok | EffectiveMs | Source |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hotpotqa | 1000 | 0.372 | 0.4510 | 0.928 | 0.412 | 0.396 | 1700.4 | 1834.7 | 1839.3 | 561.5 | `outputs/hotpotqa/eval/20260525_095722_core_ablation_5proc_hotpotqa_core_qmrag_mainline_n1000/predictions.jsonl` |
| 2wiki | 1000 | 0.144 | 0.1791 | 0.856 | 0.194 | 0.666 | 1727.3 | 1857.3 | 1861.5 | 414.4 | `outputs/2wiki/eval/20260525_095722_core_ablation_5proc_2wiki_core_qmrag_mainline_n1000/predictions.jsonl` |
| popqa | 1000 | 0.405 | 0.5051 | 0.886 | 0.603 | 0.289 | 1724.3 | 1847.2 | 1851.7 | 440.1 | `outputs/popqa/eval/20260525_095722_core_ablation_5proc_popqa_core_qmrag_mainline_n1000/predictions.jsonl` |
| musique | 1000 | 0.070 | 0.0917 | 0.636 | 0.106 | 0.860 | 2289.0 | 2423.5 | 2427.7 | 673.9 | `outputs/musique/eval/20260525_095722_core_ablation_5proc_musique_core_qmrag_mainline_n1000/predictions.jsonl` |

## ACE-RAG-Compact Common SOTA

The paper-facing common-prompt SOTA is `top3_chain_dedup + common_qa`.
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

## Native Prompt / Rendering SOTA

The final native appendix setting uses the late-night prompt/rendering update:
`p8_r0_section_aware + r0_current + top8`. The earlier `p2_relaxed_chain` run is
kept as an intermediate verification, not the final native SOTA.

| Dataset | n | Prompt | TopK | EM | F1 | Recall@5 | CtxTok | InputTok | Ret. ms | Gen. ms | Source |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hotpotqa | 1000 | p8_r0_section_aware | 8 | 0.5040 | 0.6339 | 0.9700 | 687.2 | 985.4 | 447.2 | 88.1 | `outputs/native_timing_controlled/20260526_gpu0_seq055_native_top8/hotpotqa_native_p8_top8_gpu0_seq/p8_r0_section_aware/predictions.jsonl` |
| 2wikimultihopqa | 1000 | p8_r0_section_aware | 8 | 0.3600 | 0.4337 | 0.8812 | 699.2 | 993.2 | 254.8 | 85.9 | `outputs/native_timing_controlled/20260526_gpu0_seq055_native_top8/2wiki_native_p8_top8_gpu0_seq/p8_r0_section_aware/predictions.jsonl` |
| musique | 1000 | p8_r0_section_aware | 8 | 0.1970 | 0.2768 | 0.7572 | 900.6 | 1199.0 | 463.4 | 94.3 | `outputs/native_timing_controlled/20260526_gpu0_seq055_native_top8/musique_native_p8_top8_gpu0_seq/p8_r0_section_aware/predictions.jsonl` |
| popqa | 1000 | p8_r0_section_aware | 8 | 0.4680 | 0.5984 | 0.5615 | 729.9 | 1016.6 | 227.8 | 85.7 | `outputs/native_timing_controlled/20260526_gpu0_seq055_native_top8/popqa_native_p8_top8_gpu0_seq/p8_r0_section_aware/predictions.jsonl` |

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

- Primary controlled common-prompt row is `ACE-RAG-Compact = common_qa + top3_chain_dedup`.
- Native appendix row is `p8_r0_section_aware + r0_current + top8`.
- `top3_chain_dedup + common_qa` beats the best common-prompt baseline on all
  four datasets while keeping `InputTok <= 800`.
- The compact quality drop is mostly evidence loss from context pruning, not a
  prompt-only issue; MuSiQue has the largest `Ans in Context` drop.
- `strict_short_qa` improves over `common_qa` on all four datasets.
- `qmrag_bundle_short_qa` improves the existing bundle prompt on three of four
  datasets and improves the average, but remains a method-prompt ablation.
- Native generation timing should use the controlled GPU0 sequential rerun
  under `outputs/native_timing_controlled/20260526_gpu0_seq055_native_top8`.
- No dataset-specific settings are used in the mainline or compact candidate.
