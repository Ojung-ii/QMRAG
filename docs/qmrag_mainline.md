# QMRAG Mainline Record

This note records the current QMRAG mainline used for paper-facing runs.

## Mainline Configuration

- Retrieval variant: `full_hetero`
- Seed selection variant: `top_relevance`
- Residual bridge selection: `residual_lexical`
- Bridge ordering: `anchor_chain_aware`
- Multi-anchor bundles: enabled
- Prompt profile: `common_qa`
- Rendering profile: `structured_chain`
- Compaction profile: `top3_chain_dedup`

The reproducible config entry is:

```yaml
extends: ../default.yaml

run:
  ablation_variant: core_qmrag_mainline
  enable_timing: true

generation:
  prompt_profile: common_qa
  rendering_profile: structured_chain
  compaction_profile: top3_chain_dedup

retrieval:
  retrieval_variant: full_hetero
  seed_selection_variant: top_relevance
  residual_selection: residual_lexical
  candidate_cap:
    enabled: false
  bridge:
    enabled: true
    selection: residual_lexical
    ordering: anchor_chain_aware
    multi_anchor_bundle: true
```

## Context Rendering

`top3_chain_dedup` renders only the top 3 ordered evidence bundles and removes duplicate sentences across those bundles. It keeps evidence-chain sentences and compact source references, rather than printing full source chunks.

For an evidence-chain bundle:

```text
[Evidence Chain {i} | anchor_connected={true_or_false} | chain_complete_v2={true_or_false}]
Anchor: {anchor_title}
Bridge: {bridge_title_1, bridge_title_2}
Chain:
- {source_title}: {seed_proposition}
- {bridge_title}: {bridge_proposition}
Supporting Propositions:
- {title}: {supporting_proposition}
Sources: {source_title} | {chunk_id}; {source_title} | {chunk_id}
```

For a multi-anchor bundle:

```text
[Multi-Anchor Evidence {i} | anchors={anchor_title_1}; {anchor_title_2} | complete={true_or_false}]
- {title}: {proposition_or_source_sentence}
Sources: {source_title} | {chunk_id}; {source_title} | {chunk_id}
```

For a supporting-evidence bundle:

```text
[Supporting Evidence {i} | anchor={anchor_title} | relation_title={true_or_false}]
- {title}: {supporting_proposition_or_source_sentence}
Sources: {source_title} | {chunk_id}; {source_title} | {chunk_id}
```

## Mainline Prompt

The mainline uses `common_qa`.

```text
You are a QA assistant.
Answer the question using only the provided Context.
When possible, use a concise answer explicitly supported by the Context.
Return only the final short answer.
Do not output reasoning, explanations, citations, Markdown, or prefixes.
For yes/no questions, output exactly yes or no in lowercase.
If the answer cannot be found in the Context, output exactly: insufficient information

Question: {question}
Context:
{top3_chain_dedup_context}
Answer:
```

## Compact Chain Prompt Ablation

The chain-aware compact prompt is kept as an ablation, not as the fair-comparison mainline.

```text
---Role---

You are an expert QA assistant specializing in short answer extraction from compact evidence chains.

---Goal---

Answer the user query using ONLY the provided Context.
Return only the final short answer span.

---Instructions---

1. Grounding:
  - Use only facts explicitly present in the Context.
  - Do not use outside knowledge.
  - If the answer is not explicitly supported by the Context, output exactly: insufficient information

2. Compact chain use:
  - Each Evidence Chain represents: question anchor -> bridge entity -> answer evidence.
  - Use the first sentence to link the question anchor to the bridge entity.
  - Use the next sentence as the answer evidence about that bridge entity.
  - For Multi-Anchor Evidence, compare only the listed anchors.

3. Output format:
  - Output exactly one line with only the final answer text.
  - Do not output Markdown, bullets, headings, JSON, citations, references, or prefixes.
  - For yes/no questions, output exactly `yes` or `no` in lowercase.

Question: {question}
Context:
{top3_chain_dedup_context}
Answer:
```

## Example Commands

Run the current mainline:

```bash
bash scripts/run_dataset.sh hotpotqa --config config/ablation/core_qmrag_mainline.yaml
bash scripts/run_dataset.sh 2wiki --config config/ablation/core_qmrag_mainline.yaml
bash scripts/run_dataset.sh popqa --config config/ablation/core_qmrag_mainline.yaml
bash scripts/run_dataset.sh musique --config config/ablation/core_qmrag_mainline.yaml
```

Run the compact-chain prompt ablation from existing full-context outputs:

```bash
python scripts/replay_generation.py \
  --dataset hotpotqa \
  --source-prompt common_qa \
  --source-rendering-profile structured_chain \
  --target-prompt qmrag_compact_chain_short_qa \
  --latest \
  --compaction-profile top3_chain_dedup
```
