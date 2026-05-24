from __future__ import annotations
import os, re, time
from typing import Any, Dict, List, Mapping, Optional, Sequence
from .text import safe_truncate, sentence_split, token_count

DEFAULT_PROMPT_PROFILE = "common_qa"
DEFAULT_RENDERING_PROFILE = "structured_chain"
RENDERING_PROFILES = (
    "structured_chain",
    "plain_evidence",
    "chain_only_compact",
    "multi_anchor_table",
)

COMPACTION_PROFILES = (
    "none",
    "chain_dedup",
    "chain_skeleton",
    "chain_plus1",
    "sentence_cap",
    "top3_chain_dedup",
    "chain_dedup_no_sources",
    "chain_dedup_plus1_no_sources",
    "sentence_cap_no_sources",
    "top3_chain_dedup_no_sources",
    "chain_skeleton_no_sources",
    "metadata_only_compact",
    "chain_dedup_keep_sources",
    "source_light_compact",
)

PROMPT_TEMPLATES = {
    "common_qa": """You are a QA assistant.
Answer the question using only the provided Context.
When possible, use a concise answer explicitly supported by the Context.
Return only the final short answer.
Do not output reasoning, explanations, citations, Markdown, or prefixes.
For yes/no questions, output exactly yes or no in lowercase.
If the answer cannot be found in the Context, output exactly: insufficient information

Question: {question}
Context:
{context}
Answer:""",
    "qmrag_bundle_qa": """You are a QA assistant for evidence-bundle based retrieval.
Answer the question using only the provided Context.

The Context may contain Evidence Chain, Multi-Anchor Evidence, and Supporting Evidence sections.
For Evidence Chain, answer using the bridge/property sentence when it completes the question.
For Multi-Anchor Evidence, compare the listed anchors using only the provided evidence.

Use only information explicitly supported by the Context.
Return only the final short answer span.
Do not output reasoning, explanations, citations, Markdown, or prefixes.
For yes/no questions, output exactly yes or no in lowercase.
If the answer cannot be inferred from the Context, output exactly: insufficient information

Question: {question}
Context:
{context}
Answer:""",
    "qmrag_bundle_light": """You are a QA assistant.
Answer the question using only the provided Context.
The Context may contain Evidence Chains and Multi-Anchor Evidence.
For Evidence Chains, follow the Anchor → Bridge → answer evidence.
For Multi-Anchor Evidence, compare only the listed anchors.
Return only the final short answer.
Do not output reasoning, explanations, citations, Markdown, or prefixes.
For yes/no questions, output exactly yes or no in lowercase.
If the answer cannot be found in the Context, output exactly: insufficient information

Question: {question}
Context:
{context}
Answer:""",
    "qmrag_bundle_tiny": """You are a QA assistant.
Use only the Context.
If an Evidence Chain is given, follow Anchor → Bridge → answer evidence.
If Multi-Anchor Evidence is given, compare the listed anchors.
Return only the final short answer.
If unavailable, output exactly: insufficient information

Question: {question}
Context:
{context}
Answer:""",
}

INSUFFICIENT_PHRASES = (
    "insufficient information",
    "cannot be determined",
    "not provided in the context",
    "not found in the context",
)

IDK_PHRASES = (
    "i don't know",
    "i do not know",
)

_TOKENIZER_CACHE: dict[tuple[str, bool, bool], Any] = {}

def count_tokens(text: str, tokenizer: Any = None) -> int:
    if tokenizer is not None:
        try:
            return len(tokenizer.encode(str(text or ""), add_special_tokens=False))
        except TypeError:
            return len(tokenizer.encode(str(text or "")))
        except Exception:
            pass
    return token_count(str(text or ""))

def _token_counting_cfg(cfg: Mapping[str,Any] | None) -> Mapping[str,Any]:
    tc=(cfg or {}).get("token_counting",{}) if cfg else {}
    return tc if isinstance(tc,Mapping) else {}

def _tokenizer_model_from_cfg(cfg: Mapping[str,Any] | None, model: Any = None) -> str | None:
    tc=_token_counting_cfg(cfg)
    requested=str(tc.get("tokenizer_model","auto") or "auto")
    if requested.lower()!="auto":
        return requested
    for candidate in (model, (cfg or {}).get("model") if cfg else None):
        value=str(candidate or "").strip()
        if value and value.lower()!="auto":
            return value
    return None

def _load_counting_tokenizer(cfg: Mapping[str,Any] | None, model: Any = None) -> tuple[Any | None, str]:
    tc=_token_counting_cfg(cfg)
    if tc and not bool(tc.get("enabled",True)):
        return None,"approx"
    tokenizer_model=_tokenizer_model_from_cfg(cfg,model)
    if not tokenizer_model:
        return None,"approx"
    local_files_only=bool(tc.get("local_files_only",True))
    trust_remote_code=bool(tc.get("trust_remote_code",True))
    key=(tokenizer_model,local_files_only,trust_remote_code)
    if key in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[key],"tokenizer"
    try:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(
            tokenizer_model,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        _TOKENIZER_CACHE[key]=tokenizer
        return tokenizer,"tokenizer"
    except Exception:
        if tc and not bool(tc.get("fallback_to_estimate",True)):
            raise
        return None,"approx"

def prompt_template_token_count(prompt_profile: str, tokenizer: Any = None) -> int:
    profile=str(prompt_profile or DEFAULT_PROMPT_PROFILE)
    if profile not in PROMPT_TEMPLATES:
        profile=DEFAULT_PROMPT_PROFILE
    return count_tokens(PROMPT_TEMPLATES[profile].format(question="", context=""), tokenizer)

def extract_usage_token_counts(usage: Any) -> dict[str,int | None]:
    if usage is None:
        usage_dict={}
    elif isinstance(usage,Mapping):
        usage_dict=dict(usage)
    elif hasattr(usage,"model_dump"):
        usage_dict=usage.model_dump()
    else:
        usage_dict={k:getattr(usage,k) for k in ("prompt_tokens","completion_tokens","total_tokens") if hasattr(usage,k)}
    def get_int(*keys: str) -> int | None:
        for key in keys:
            value=usage_dict.get(key)
            if value is not None:
                try:
                    return int(value)
                except Exception:
                    return None
        return None
    return {
        "prompt_tokens":get_int("prompt_tokens","input_tokens"),
        "completion_tokens":get_int("completion_tokens","output_tokens"),
        "total_tokens":get_int("total_tokens"),
    }

def llm_input_text(prompt: str, cfg: Mapping[str,Any] | None = None) -> str:
    system_message=(cfg or {}).get("system_message") if cfg else None
    if system_message is None and cfg:
        system_message=cfg.get("system_prompt")
    if system_message:
        return f"{system_message}\n{prompt}"
    return str(prompt or "")

def add_token_accounting_fields(
    row: Dict[str,Any],
    prompt: str,
    rendered_context: str,
    raw_prediction: Any,
    prompt_profile: str,
    cfg: Mapping[str,Any] | None = None,
    usage: Any = None,
    model: Any = None,
) -> Dict[str,Any]:
    usage_counts=extract_usage_token_counts(usage)
    tokenizer,counter_source=_load_counting_tokenizer(cfg,model)
    usage_prompt=usage_counts.get("prompt_tokens")
    usage_completion=usage_counts.get("completion_tokens")
    usage_total=usage_counts.get("total_tokens")
    input_prompt_tokens=usage_prompt if usage_prompt is not None else count_tokens(llm_input_text(prompt,cfg),tokenizer)
    completion_tokens=usage_completion if usage_completion is not None else count_tokens(str(raw_prediction or ""),tokenizer)
    total_llm_tokens=usage_total if usage_total is not None else int(input_prompt_tokens)+int(completion_tokens)
    row.update({
        "prompt_template_tokens":prompt_template_token_count(prompt_profile,tokenizer),
        "rendered_context_tokens":count_tokens(rendered_context,tokenizer),
        "input_prompt_tokens":int(input_prompt_tokens),
        "completion_tokens":int(completion_tokens),
        "total_llm_tokens":int(total_llm_tokens),
        "token_count_source":"usage" if usage_prompt is not None or usage_completion is not None or usage_total is not None else counter_source,
        "llm_usage_prompt_tokens":usage_prompt,
        "llm_usage_completion_tokens":usage_completion,
        "llm_usage_total_tokens":usage_total,
    })
    return row

def _ordered_bundles(bundles: Sequence[Mapping[str,Any]]) -> list[Mapping[str,Any]]:
    order_rank={"anchor_connected_chain_complete":0,"multi_anchor":1,"anchor":2,"chain_complete_v2":3,"bridge_connected":4,"same_title":5,"generic_relation":6,"fallback":7,"complete_bridge_chain":3,"exact_query_anchor":2,"bridge_candidate":4}
    return [
        item[1]
        for item in sorted(
        list(enumerate(bundles)),
        key=lambda item:(
            order_rank.get(str(item[1].get("ordering_group","same_title")),8),
            not bool(item[1].get("chain_complete_v2",item[1].get("chain_complete"))),
            not bool(item[1].get("anchor_connected")),
            bool(item[1].get("is_generic_relation_title")),
            not bool(item[1].get("exact_anchor_match")),
            -float(item[1].get("residual_coverage_count",0.0) or 0.0),
            -float(item[1].get("score",0.0) or 0.0),
            item[0],
        ),
    )
    ]

def _bridge_paths(bundle: Mapping[str,Any]) -> list[Mapping[str,Any]]:
    return [x for x in bundle.get("evidence_path",[]) or [] if isinstance(x,Mapping) and x.get("path_type")=="mention_bridge"]

def _append_unique_sentence(lines: list[str], seen: set[str], title: str, text: Any, prefix: str="- ") -> None:
    sentence=str(text or "").strip()
    if not sentence:
        return
    key=" ".join(sentence.split())
    if key in seen:
        return
    seen.add(key)
    clean_title=str(title or "").strip()
    if clean_title:
        lines.append(f"{prefix}{clean_title}: {sentence}")
    else:
        lines.append(f"{prefix}{sentence}")

def _sentence_norm(text: Any) -> str:
    value=str(text or "").lower()
    value=re.sub(r"[^a-z0-9]+"," ",value)
    return " ".join(value.split())

def _record(title: Any, text: Any, role: str, priority: int, source_id: Any = None) -> dict[str,Any] | None:
    sentence=str(text or "").strip()
    if not sentence:
        return None
    return {
        "title":str(title or "").strip(),
        "text":sentence,
        "role":role,
        "priority":int(priority),
        "source_id":str(source_id or "").strip(),
        "norm":_sentence_norm(sentence),
    }

def _chain_records(bundle: Mapping[str,Any]) -> list[dict[str,Any]]:
    records=[]
    for path in _bridge_paths(bundle):
        seed=_record(path.get("source_title") or bundle.get("anchor_title"), path.get("seed_prop"), "chain_seed", 0, path.get("seed_prop_id"))
        bridge=_record(path.get("bridge_title"), path.get("bridge_prop"), "chain_bridge", 1, path.get("bridge_prop_id"))
        if seed: records.append(seed)
        if bridge: records.append(bridge)
    return records

def _prop_records(bundle: Mapping[str,Any], chain_norms: set[str] | None = None) -> list[dict[str,Any]]:
    chain_norms=chain_norms or set()
    records=[]
    for p in bundle.get("propositions",[]) or []:
        rec=_record(p.get("title") or bundle.get("anchor_title"), p.get("text"), "support", 3, p.get("prop_id"))
        if not rec or rec["norm"] in chain_norms:
            continue
        for score_key in ("residual_score","original_relevance_score","score","relevance_score"):
            if p.get(score_key) is not None:
                try:
                    rec["score"]=float(p.get(score_key) or 0.0)
                except Exception:
                    rec["score"]=0.0
                break
        if bundle.get("answer_slot_aligned") or int(bundle.get("residual_coverage_count",0) or 0)>0:
            rec["role"]="answer_slot_support"; rec["priority"]=2
        elif bundle.get("anchor_connected"):
            rec["role"]="anchor_support"; rec["priority"]=3
        records.append(rec)
    return records

def _source_records(bundle: Mapping[str,Any], chain_norms: set[str] | None = None, limit_per_chunk: int = 2) -> list[dict[str,Any]]:
    chain_norms=chain_norms or set()
    records=[]
    for c in bundle.get("source_chunks",[]) or []:
        for sent in sentence_split(str(c.get("text","")))[:limit_per_chunk]:
            rec=_record(c.get("title") or bundle.get("anchor_title"), sent, "source", 5, c.get("chunk_id"))
            if rec and rec["norm"] not in chain_norms:
                records.append(rec)
    return records

def _bundle_sentence_records(bundle: Mapping[str,Any], include_sources: bool = True) -> list[dict[str,Any]]:
    chain=_chain_records(bundle)
    chain_norms={r["norm"] for r in chain if r.get("norm")}
    props=_prop_records(bundle, chain_norms)
    records=chain+props
    if include_sources:
        records.extend(_source_records(bundle, {r["norm"] for r in records if r.get("norm")}))
    return records

def _append_record(lines: list[str], seen: set[str], rec: Mapping[str,Any], prefix: str = "- ") -> bool:
    norm=str(rec.get("norm") or _sentence_norm(rec.get("text")))
    if not norm or norm in seen:
        return False
    seen.add(norm)
    title=str(rec.get("title") or "").strip()
    text=str(rec.get("text") or "").strip()
    if title:
        lines.append(f"{prefix}{title}: {text}")
    else:
        lines.append(f"{prefix}{text}")
    return True

def _metadata_removed_count(bundles: Sequence[Mapping[str,Any]]) -> int:
    count=0
    bundle_keys=(
        "score","ordering_group","anchor_connected","chain_complete","chain_complete_v2",
        "anchor_connected_chain_complete","answer_slot_aligned","bridge_connected",
        "is_relation_title_bundle","is_generic_relation_title","exact_anchor_match",
        "residual_coverage_count","avg_residual_coverage_count",
    )
    for bundle in bundles or []:
        count+=sum(1 for key in bundle_keys if bundle.get(key) not in (None,"",False,[],{}))
        for path in bundle.get("evidence_path",[]) or []:
            if not isinstance(path,Mapping):
                continue
            count+=sum(1 for key in ("seed_prop_id","bridge_prop_id","path_type","score","residual_coverage_count") if path.get(key) not in (None,"",False,[],{}))
        for prop in bundle.get("propositions",[]) or []:
            if not isinstance(prop,Mapping):
                continue
            count+=sum(1 for key in ("prop_id","score","rank","chunk_id","residual_score","original_relevance_score") if prop.get(key) not in (None,"",False,[],{}))
        for chunk in bundle.get("source_chunks",[]) or []:
            if not isinstance(chunk,Mapping):
                continue
            count+=sum(1 for key in ("chunk_id","score","rank") if chunk.get(key) not in (None,"",False,[],{}))
    return count

def rendered_sentence_count(text: Any) -> int:
    count=0
    for line in str(text or "").splitlines():
        stripped=line.strip()
        if not stripped or stripped.endswith(":"):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            continue
        if stripped.startswith(("Anchor:","Bridge:","Sources:","Supporting Propositions:","Chain:","Title:","Evidence Chain:","Supporting Evidence:","Multi-Anchor Evidence:")):
            continue
        if stripped.startswith("- "):
            count+=1
        elif sentence_split(stripped):
            count+=len(sentence_split(stripped))
    return count

def bundle_sentence_statistics(evidence_bundles: Sequence[Mapping[str,Any]], rendered_context: Any = None) -> dict[str,Any]:
    bundles=list(evidence_bundles or [])
    per_bundle=[]
    all_norms=[]
    chain_count=0
    support_count=0
    source_count=0
    multi_anchor_count=0
    for bundle in bundles:
        records=_bundle_sentence_records(bundle, include_sources=True)
        per_bundle.append(len(records))
        for rec in records:
            norm=str(rec.get("norm") or "")
            if norm:
                all_norms.append(norm)
            role=str(rec.get("role") or "")
            if role.startswith("chain_"):
                chain_count+=1
            elif role=="source":
                source_count+=1
            else:
                support_count+=1
        if bundle.get("bundle_type")=="multi_anchor":
            multi_anchor_count+=len([r for r in records if r.get("role")!="source"])
    duplicate_count=max(0,len(all_norms)-len(set(all_norms)))
    rendered_count=rendered_sentence_count(rendered_context) if rendered_context is not None else sum(per_bundle)
    bundle_count=len(bundles)
    top1=sum(per_bundle[:1])
    top3=sum(per_bundle[:3])
    return {
        "bundle_count":bundle_count,
        "rendered_sentence_count":rendered_count,
        "avg_sentences_per_bundle":sum(per_bundle)/max(1,bundle_count),
        "max_sentences_per_bundle":max(per_bundle or [0]),
        "top1_bundle_sentence_count":top1,
        "top3_bundle_sentence_count":top3,
        "chain_sentence_count":chain_count,
        "support_sentence_count":support_count,
        "duplicate_sentence_count":duplicate_count,
        "duplicate_sentence_rate":duplicate_count/max(1,len(all_norms)),
        "source_sentence_count":source_count,
        "multi_anchor_sentence_count":multi_anchor_count,
    }

def _source_title_for_bundle(bundle: Mapping[str,Any]) -> str:
    return str(bundle.get("anchor_title") or bundle.get("title") or "").strip()

def _render_structured_chain(bundles: Sequence[Mapping[str,Any]], max_chars: int) -> str:
    parts=[]
    ordered=_ordered_bundles(bundles)
    for i,b in enumerate(ordered, start=1):
        bridge=", ".join(str(x) for x in b.get("bridge_titles",[]) or [])
        bundle_type=str(b.get("bundle_type") or "")
        bridge_paths=_bridge_paths(b)
        chain_texts=set()
        if bundle_type=="multi_anchor":
            anchors="; ".join(str(x) for x in b.get("anchor_titles",[]) or [])
            parts.append(f"[Multi-Anchor Evidence {i} | anchors={anchors} | complete={bool(b.get('multi_anchor_complete'))}]")
            seen=set()
            for p in b.get("propositions",[]) or []:
                title=str(p.get("title") or "")
                text=str(p.get("text") or "")
                key=(title,text)
                if title and text and key not in seen:
                    seen.add(key)
                    parts.append(f"- {title}: {text}")
            if not seen:
                for c in b.get("source_chunks",[]) or []:
                    title=str(c.get("title") or "")
                    text=str(c.get("text") or "")
                    if title and text:
                        parts.append(f"- {title}: {text}")
            parts.append("")
            continue
        if bridge_paths:
            parts.append(f"[Evidence Chain {i} | anchor_connected={bool(b.get('anchor_connected'))} | chain_complete_v2={bool(b.get('chain_complete_v2',b.get('chain_complete')))} | score={b.get('score')}]")
            parts.append(f"Anchor: {b.get('anchor_title')}")
            if bridge:
                parts.append(f"Bridge: {bridge}")
            parts.append("Chain:")
            for path in bridge_paths[:4]:
                for title_key,text_key in (("source_title","seed_prop"),("bridge_title","bridge_prop")):
                    text=str(path.get(text_key) or "")
                    if text and not any(text==seen or text in seen for seen in chain_texts):
                        chain_texts.add(text)
                        parts.append(f"- {path.get(title_key)}: {text}")
        else:
            parts.append(f"[Supporting Evidence {i} | anchor={b.get('anchor_title')} | relation_title={bool(b.get('is_relation_title_bundle'))} | score={b.get('score')}]")
            if b.get("anchor_title"):
                parts.append(f"Anchor: {b.get('anchor_title')}")
        if b.get("propositions"):
            prop_lines=[]
            for p in b.get("propositions",[]):
                text=str(p.get("text") or "")
                if text and any(text==seen or text in seen for seen in chain_texts):
                    continue
                prop_lines.append(f"- ({p.get('prop_id')} | {p.get('title')}) {text}")
            if prop_lines:
                parts.append("Supporting Propositions:")
                parts.extend(prop_lines)
        if b.get("source_chunks"):
            parts.append("Sources:")
            for c in b.get("source_chunks",[]): parts.append(f"- [{c.get('title')} | {c.get('chunk_id')}] {c.get('text')}")
        parts.append("")
    return safe_truncate("\n".join(parts).strip(), max_chars)

def _render_plain_evidence(bundles: Sequence[Mapping[str,Any]], max_chars: int) -> str:
    parts=[]
    seen_global=set()
    for b in _ordered_bundles(bundles):
        title=_source_title_for_bundle(b)
        lines=[]
        seen=set()
        for path in _bridge_paths(b):
            _append_unique_sentence(lines, seen, str(path.get("source_title") or title), path.get("seed_prop"), prefix="")
            _append_unique_sentence(lines, seen, str(path.get("bridge_title") or ""), path.get("bridge_prop"), prefix="")
        for p in b.get("propositions",[]) or []:
            _append_unique_sentence(lines, seen, str(p.get("title") or title), p.get("text"), prefix="")
        if not lines:
            for c in b.get("source_chunks",[]) or []:
                for sent in sentence_split(str(c.get("text","")))[:2]:
                    _append_unique_sentence(lines, seen, str(c.get("title") or title), sent, prefix="")
        kept=[]
        for line in lines:
            key=" ".join(line.split())
            if key and key not in seen_global:
                seen_global.add(key); kept.append(line)
        if not kept:
            continue
        if title:
            parts.append(f"Title: {title}")
        for line in kept:
            if ": " in line:
                maybe_title, sent=line.split(": ",1)
                if maybe_title==title:
                    parts.append(sent)
                else:
                    parts.append(f"{maybe_title}: {sent}")
            else:
                parts.append(line)
        parts.append("")
    return safe_truncate("\n".join(parts).strip(), max_chars)

def _is_complete_chain_bundle(bundle: Mapping[str,Any]) -> bool:
    return bool(bundle.get("anchor_connected_chain_complete") or bundle.get("chain_complete_v2") or bundle.get("chain_complete"))

def _render_chain_only_compact(bundles: Sequence[Mapping[str,Any]], max_chars: int) -> str:
    ordered=_ordered_bundles(bundles)
    chain_bundles=[b for b in ordered if _bridge_paths(b) and _is_complete_chain_bundle(b)]
    other_chain_bundles=[b for b in ordered if _bridge_paths(b) and b not in chain_bundles]
    fallback=[b for b in ordered if not _bridge_paths(b)]
    parts=[]
    seen=set()
    def append_bundle(bundle: Mapping[str,Any], label: str) -> None:
        title=_source_title_for_bundle(bundle)
        bridge=", ".join(str(x) for x in bundle.get("bridge_titles",[]) or [])
        if label=="chain":
            header=f"Evidence Chain: {title}"
            if bridge:
                header+=f" -> {bridge}"
            parts.append(header)
            for path in _bridge_paths(bundle):
                _append_unique_sentence(parts, seen, str(path.get("source_title") or title), path.get("seed_prop"))
                _append_unique_sentence(parts, seen, str(path.get("bridge_title") or ""), path.get("bridge_prop"))
        else:
            if title:
                parts.append(f"Supporting Evidence: {title}")
            for p in bundle.get("propositions",[]) or []:
                _append_unique_sentence(parts, seen, str(p.get("title") or title), p.get("text"))
        parts.append("")
    for bundle in chain_bundles:
        append_bundle(bundle,"chain")
    for bundle in other_chain_bundles:
        if len("\n".join(parts)) >= max_chars:
            break
        append_bundle(bundle,"chain")
    for bundle in fallback:
        if len("\n".join(parts)) >= max_chars:
            break
        append_bundle(bundle,"support")
    return safe_truncate("\n".join(parts).strip(), max_chars)

def _render_multi_anchor_table(bundles: Sequence[Mapping[str,Any]], max_chars: int) -> str:
    ordered=_ordered_bundles(bundles)
    parts=[]
    seen=set()
    for b in ordered:
        if b.get("bundle_type")!="multi_anchor":
            continue
        parts.append("Multi-Anchor Evidence:")
        for p in b.get("propositions",[]) or []:
            _append_unique_sentence(parts, seen, str(p.get("title") or ""), p.get("text"))
        if not b.get("propositions"):
            for c in b.get("source_chunks",[]) or []:
                for sent in sentence_split(str(c.get("text","")))[:2]:
                    _append_unique_sentence(parts, seen, str(c.get("title") or ""), sent)
        parts.append("")
    for b in ordered:
        if b.get("bundle_type")=="multi_anchor":
            continue
        if _bridge_paths(b):
            title=_source_title_for_bundle(b)
            bridge=", ".join(str(x) for x in b.get("bridge_titles",[]) or [])
            parts.append(f"Evidence Chain: {title}" + (f" -> {bridge}" if bridge else ""))
            for path in _bridge_paths(b):
                _append_unique_sentence(parts, seen, str(path.get("source_title") or title), path.get("seed_prop"))
                _append_unique_sentence(parts, seen, str(path.get("bridge_title") or ""), path.get("bridge_prop"))
            parts.append("")
        elif len("\n".join(parts)) < max_chars:
            title=_source_title_for_bundle(b)
            if title:
                parts.append(f"Supporting Evidence: {title}")
            for p in b.get("propositions",[]) or []:
                _append_unique_sentence(parts, seen, str(p.get("title") or title), p.get("text"))
            parts.append("")
    return safe_truncate("\n".join(parts).strip(), max_chars)

def _source_refs(bundle: Mapping[str,Any]) -> str:
    refs=[]
    for c in bundle.get("source_chunks",[]) or []:
        title=str(c.get("title") or "").strip()
        chunk_id=str(c.get("chunk_id") or "").strip()
        if title or chunk_id:
            refs.append(f"{title} | {chunk_id}".strip(" |"))
    return "; ".join(dict.fromkeys(refs))

def _append_source_chunks_no_metadata(
    parts: list[str],
    seen: set[str],
    bundle: Mapping[str,Any],
    sentence_limit_per_chunk: int | None = None,
) -> int:
    appended=0
    for chunk in bundle.get("source_chunks",[]) or []:
        title=str(chunk.get("title") or bundle.get("anchor_title") or "").strip()
        text=str(chunk.get("text") or "").strip()
        if not text:
            continue
        texts=[text]
        if sentence_limit_per_chunk is not None:
            texts=sentence_split(text)[:max(1,int(sentence_limit_per_chunk or 1))]
        for sent in texts:
            rec=_record(title,sent,"source",5,chunk.get("chunk_id"))
            if rec and _append_record(parts,seen,rec):
                appended+=1
    return appended

def _render_metadata_only_compact(bundles: Sequence[Mapping[str,Any]], max_chars: int) -> str:
    parts=[]
    for i,bundle in enumerate(_ordered_bundles(bundles), start=1):
        seen:set[str]=set()
        if bundle.get("bundle_type")=="multi_anchor":
            anchors="; ".join(str(x) for x in bundle.get("anchor_titles",[]) or [])
            parts.append(f"Multi-Anchor Evidence {i}: {anchors}" if anchors else f"Multi-Anchor Evidence {i}:")
            for rec in _prop_records(bundle):
                _append_record(parts,seen,rec)
            if bundle.get("source_chunks"):
                parts.append("Sources:")
                _append_source_chunks_no_metadata(parts,seen,bundle,sentence_limit_per_chunk=None)
            parts.append("")
            continue
        bridge=", ".join(str(x) for x in bundle.get("bridge_titles",[]) or [])
        chain=_chain_records(bundle)
        if chain:
            anchor=str(bundle.get("anchor_title") or "").strip()
            parts.append(f"Evidence Chain {i}: {anchor}" if anchor else f"Evidence Chain {i}:")
            if bridge:
                parts.append(f"Bridge: {bridge}")
            parts.append("Chain:")
            for rec in chain:
                _append_record(parts,seen,rec)
            support=_prop_records(bundle,{r.get("norm","") for r in chain})
            if support:
                parts.append("Supporting Evidence:")
                for rec in support:
                    _append_record(parts,seen,rec)
        else:
            anchor=str(bundle.get("anchor_title") or bundle.get("title") or "").strip()
            parts.append(f"Supporting Evidence {i}: {anchor}" if anchor else f"Supporting Evidence {i}:")
            for rec in _prop_records(bundle):
                _append_record(parts,seen,rec)
        if bundle.get("source_chunks"):
            parts.append("Sources:")
            _append_source_chunks_no_metadata(parts,seen,bundle,sentence_limit_per_chunk=None)
        parts.append("")
    return safe_truncate("\n".join(parts).strip(), max_chars)

def _render_chain_dedup(bundles: Sequence[Mapping[str,Any]], max_chars: int) -> str:
    parts=[]
    seen:set[str]=set()
    for i,bundle in enumerate(_ordered_bundles(bundles), start=1):
        if bundle.get("bundle_type")=="multi_anchor":
            anchors="; ".join(str(x) for x in bundle.get("anchor_titles",[]) or [])
            parts.append(f"[Multi-Anchor Evidence {i} | anchors={anchors} | complete={bool(bundle.get('multi_anchor_complete'))}]")
            appended=False
            for rec in _prop_records(bundle):
                appended=_append_record(parts,seen,rec) or appended
            if not appended:
                for rec in _source_records(bundle,limit_per_chunk=1):
                    _append_record(parts,seen,rec)
            parts.append("")
            continue
        bridge=", ".join(str(x) for x in bundle.get("bridge_titles",[]) or [])
        if _bridge_paths(bundle):
            parts.append(f"[Evidence Chain {i} | anchor_connected={bool(bundle.get('anchor_connected'))} | chain_complete_v2={bool(bundle.get('chain_complete_v2',bundle.get('chain_complete')))}]")
            parts.append(f"Anchor: {bundle.get('anchor_title')}")
            if bridge:
                parts.append(f"Bridge: {bridge}")
            parts.append("Chain:")
            for rec in _chain_records(bundle):
                _append_record(parts,seen,rec)
            support_records=_prop_records(bundle,{r.get("norm","") for r in _chain_records(bundle)})
            if support_records:
                parts.append("Supporting Propositions:")
                for rec in support_records:
                    _append_record(parts,seen,rec)
        else:
            parts.append(f"[Supporting Evidence {i} | anchor={bundle.get('anchor_title')} | relation_title={bool(bundle.get('is_relation_title_bundle'))}]")
            appended=False
            for rec in _prop_records(bundle):
                appended=_append_record(parts,seen,rec) or appended
            if not appended:
                for rec in _source_records(bundle,limit_per_chunk=1):
                    _append_record(parts,seen,rec)
        refs=_source_refs(bundle)
        if refs:
            parts.append(f"Sources: {refs}")
        parts.append("")
    return safe_truncate("\n".join(parts).strip(), max_chars)

def _render_chain_dedup_keep_sources(
    bundles: Sequence[Mapping[str,Any]],
    max_chars: int,
    source_sentence_limit_per_chunk: int | None = None,
) -> str:
    parts=[]
    seen:set[str]=set()
    for i,bundle in enumerate(_ordered_bundles(bundles), start=1):
        if bundle.get("bundle_type")=="multi_anchor":
            _render_multi_anchor_no_sources(parts,seen,bundle,i,per_anchor_limit=None)
        else:
            bridge=", ".join(str(x) for x in bundle.get("bridge_titles",[]) or [])
            chain=_chain_records(bundle)
            if chain:
                anchor=str(bundle.get("anchor_title") or "").strip()
                parts.append(f"Evidence Chain {i}: {anchor}" if anchor else f"Evidence Chain {i}:")
                if bridge:
                    parts.append(f"Bridge: {bridge}")
                parts.append("Chain:")
                for rec in chain:
                    _append_record(parts,seen,rec)
                support=_support_records_for_compact(bundle,chain,limit=None)
                if support:
                    parts.append("Supporting Evidence:")
                    for rec in support:
                        _append_record(parts,seen,rec)
            else:
                support=_support_records_for_compact(bundle,[],limit=None)
                if support:
                    anchor=str(bundle.get("anchor_title") or bundle.get("title") or "").strip()
                    parts.append(f"Supporting Evidence {i}: {anchor}" if anchor else f"Supporting Evidence {i}:")
                    for rec in support:
                        _append_record(parts,seen,rec)
        if bundle.get("source_chunks"):
            parts.append("Sources:")
            _append_source_chunks_no_metadata(parts,seen,bundle,sentence_limit_per_chunk=source_sentence_limit_per_chunk)
        parts.append("")
    return safe_truncate("\n".join(parts).strip(), max_chars)

def _support_records_for_compact(bundle: Mapping[str,Any], chain: Sequence[Mapping[str,Any]], limit: int | None = None) -> list[dict[str,Any]]:
    chain_norms={str(r.get("norm") or "") for r in chain if r.get("norm")}
    records=_prop_records(bundle,chain_norms)
    records=sorted(records,key=lambda rec:(int(rec.get("priority",9)), -float(rec.get("score",0.0) or 0.0), len(str(rec.get("text","")))))
    if limit is None:
        return records
    return records[:max(0,int(limit))]

def _render_multi_anchor_no_sources(parts: list[str], seen: set[str], bundle: Mapping[str,Any], i: int, per_anchor_limit: int | None = None) -> None:
    anchors=[str(x) for x in bundle.get("anchor_titles",[]) or []]
    if anchors:
        parts.append(f"Multi-Anchor Evidence {i}: " + "; ".join(anchors))
    else:
        parts.append(f"Multi-Anchor Evidence {i}:")
    by_title: dict[str,list[dict[str,Any]]]={}
    for rec in _prop_records(bundle):
        by_title.setdefault(str(rec.get("title") or ""),[]).append(rec)
    anchor_order=anchors or list(by_title.keys())
    appended=False
    for title in anchor_order:
        candidates=sorted(by_title.get(title,[]),key=lambda rec:(int(rec.get("priority",9)), -float(rec.get("score",0.0) or 0.0), len(str(rec.get("text","")))))
        if per_anchor_limit is not None:
            candidates=candidates[:max(0,int(per_anchor_limit))]
        for rec in candidates:
            appended=_append_record(parts,seen,rec) or appended
    if not appended:
        for records in by_title.values():
            for rec in records[:1]:
                _append_record(parts,seen,rec)
    parts.append("")

def _render_chain_dedup_no_sources(
    bundles: Sequence[Mapping[str,Any]],
    max_chars: int,
    support_limit: int | None = None,
) -> str:
    parts=[]
    seen:set[str]=set()
    for i,bundle in enumerate(_ordered_bundles(bundles), start=1):
        if bundle.get("bundle_type")=="multi_anchor":
            _render_multi_anchor_no_sources(parts,seen,bundle,i,per_anchor_limit=None if support_limit is None else 1)
            continue
        bridge=", ".join(str(x) for x in bundle.get("bridge_titles",[]) or [])
        chain=_chain_records(bundle)
        if chain:
            anchor=str(bundle.get("anchor_title") or "").strip()
            parts.append(f"Evidence Chain {i}: {anchor}" if anchor else f"Evidence Chain {i}:")
            if bridge:
                parts.append(f"Bridge: {bridge}")
            parts.append("Chain:")
            for rec in chain:
                _append_record(parts,seen,rec)
            support_records=_support_records_for_compact(bundle,chain,limit=support_limit)
            if support_records:
                parts.append("Supporting Evidence:")
                for rec in support_records:
                    _append_record(parts,seen,rec)
        else:
            support_records=_support_records_for_compact(bundle,[],limit=support_limit)
            if not support_records:
                continue
            anchor=str(bundle.get("anchor_title") or bundle.get("title") or "").strip()
            parts.append(f"Supporting Evidence {i}: {anchor}" if anchor else f"Supporting Evidence {i}:")
            for rec in support_records:
                _append_record(parts,seen,rec)
        parts.append("")
    return safe_truncate("\n".join(parts).strip(), max_chars)

def _append_skeleton_bundle(parts: list[str], seen: set[str], bundle: Mapping[str,Any], include_plus_one: bool = False) -> None:
    if bundle.get("bundle_type")=="multi_anchor":
        parts.append("[Multi-Anchor Evidence]")
        by_title: dict[str,dict[str,Any]]={}
        for rec in _prop_records(bundle):
            by_title.setdefault(str(rec.get("title") or ""), rec)
        if not by_title:
            for rec in _source_records(bundle,limit_per_chunk=1):
                by_title.setdefault(str(rec.get("title") or ""), rec)
        for rec in by_title.values():
            _append_record(parts,seen,rec)
        parts.append("")
        return
    chain=_chain_records(bundle)
    if chain:
        bridge=", ".join(str(x) for x in bundle.get("bridge_titles",[]) or [])
        header=f"Evidence Chain: {bundle.get('anchor_title')}"
        if bridge:
            header+=f" -> {bridge}"
        parts.append(header)
        seed_added=False
        bridge_added=False
        for rec in chain:
            role=str(rec.get("role") or "")
            if role=="chain_seed" and not seed_added:
                seed_added=_append_record(parts,seen,rec)
            elif role=="chain_bridge" and not bridge_added:
                bridge_added=_append_record(parts,seen,rec)
            if seed_added and bridge_added:
                break
        if include_plus_one:
            support=sorted(_prop_records(bundle,{r.get("norm","") for r in chain}), key=lambda r:(int(r.get("priority",9)), len(str(r.get("text","")))))
            for rec in support:
                if _append_record(parts,seen,rec):
                    break
        parts.append("")

def _render_chain_skeleton(bundles: Sequence[Mapping[str,Any]], max_chars: int, include_plus_one: bool = False) -> str:
    parts=[]
    seen:set[str]=set()
    for bundle in _ordered_bundles(bundles):
        if _bridge_paths(bundle) or bundle.get("bundle_type")=="multi_anchor":
            _append_skeleton_bundle(parts,seen,bundle,include_plus_one=include_plus_one)
    return safe_truncate("\n".join(parts).strip(), max_chars)

def _render_chain_skeleton_no_sources(bundles: Sequence[Mapping[str,Any]], max_chars: int) -> str:
    parts=[]
    seen:set[str]=set()
    for bundle in _ordered_bundles(bundles):
        if bundle.get("bundle_type")=="multi_anchor":
            _render_multi_anchor_no_sources(parts,seen,bundle,len(parts)+1,per_anchor_limit=1)
            continue
        chain=_chain_records(bundle)
        if not chain:
            continue
        bridge=", ".join(str(x) for x in bundle.get("bridge_titles",[]) or [])
        anchor=str(bundle.get("anchor_title") or "").strip()
        parts.append(f"Evidence Chain: {anchor}" if anchor else "Evidence Chain:")
        if bridge:
            parts.append(f"Bridge: {bridge}")
        seed_added=False
        bridge_added=False
        for rec in chain:
            role=str(rec.get("role") or "")
            if role=="chain_seed" and not seed_added:
                seed_added=_append_record(parts,seen,rec)
            elif role=="chain_bridge" and not bridge_added:
                bridge_added=_append_record(parts,seen,rec)
            if seed_added and bridge_added:
                break
        parts.append("")
    return safe_truncate("\n".join(parts).strip(), max_chars)

def _render_sentence_cap(bundles: Sequence[Mapping[str,Any]], max_chars: int, max_sentences_per_bundle: int = 3) -> str:
    parts=[]
    seen:set[str]=set()
    cap=max(1,int(max_sentences_per_bundle or 3))
    for i,bundle in enumerate(_ordered_bundles(bundles), start=1):
        bridge=", ".join(str(x) for x in bundle.get("bridge_titles",[]) or [])
        if bundle.get("bundle_type")=="multi_anchor":
            anchors="; ".join(str(x) for x in bundle.get("anchor_titles",[]) or [])
            parts.append(f"[Multi-Anchor Evidence {i} | anchors={anchors}]")
        elif _bridge_paths(bundle):
            parts.append(f"[Evidence Chain {i} | anchor_connected={bool(bundle.get('anchor_connected'))} | chain_complete_v2={bool(bundle.get('chain_complete_v2',bundle.get('chain_complete')))}]")
            parts.append(f"Anchor: {bundle.get('anchor_title')}")
            if bridge:
                parts.append(f"Bridge: {bridge}")
        else:
            parts.append(f"[Supporting Evidence {i} | anchor={bundle.get('anchor_title')}]")
        records=_bundle_sentence_records(bundle,include_sources=True)
        records=sorted(records,key=lambda rec:(int(rec.get("priority",9)), 0 if rec.get("role")!="source" else 1))
        kept=0
        for rec in records:
            if _append_record(parts,seen,rec):
                kept+=1
            if kept>=cap:
                break
        parts.append("")
    return safe_truncate("\n".join(parts).strip(), max_chars)

def _render_sentence_cap_no_sources(bundles: Sequence[Mapping[str,Any]], max_chars: int, max_sentences_per_bundle: int = 3) -> str:
    parts=[]
    seen:set[str]=set()
    cap=max(1,int(max_sentences_per_bundle or 3))
    for i,bundle in enumerate(_ordered_bundles(bundles), start=1):
        records=_bundle_sentence_records(bundle,include_sources=False)
        records=sorted(records,key=lambda rec:(int(rec.get("priority",9)), -float(rec.get("score",0.0) or 0.0), len(str(rec.get("text","")))))
        if not records:
            continue
        bridge=", ".join(str(x) for x in bundle.get("bridge_titles",[]) or [])
        if bundle.get("bundle_type")=="multi_anchor":
            anchors="; ".join(str(x) for x in bundle.get("anchor_titles",[]) or [])
            parts.append(f"Multi-Anchor Evidence {i}: {anchors}" if anchors else f"Multi-Anchor Evidence {i}:")
        elif _bridge_paths(bundle):
            anchor=str(bundle.get("anchor_title") or "").strip()
            parts.append(f"Evidence Chain {i}: {anchor}" if anchor else f"Evidence Chain {i}:")
            if bridge:
                parts.append(f"Bridge: {bridge}")
        else:
            anchor=str(bundle.get("anchor_title") or bundle.get("title") or "").strip()
            parts.append(f"Supporting Evidence {i}: {anchor}" if anchor else f"Supporting Evidence {i}:")
        kept=0
        for rec in records:
            if _append_record(parts,seen,rec):
                kept+=1
            if kept>=cap:
                break
        parts.append("")
    return safe_truncate("\n".join(parts).strip(), max_chars)

def _render_compacted_context(
    bundles: Sequence[Mapping[str,Any]],
    max_chars: int,
    compaction_profile: str,
    max_sentences_per_bundle: int = 3,
) -> str:
    profile=str(compaction_profile or "none")
    if profile=="metadata_only_compact":
        return _render_metadata_only_compact(bundles,max_chars)
    if profile=="chain_dedup":
        return _render_chain_dedup(bundles,max_chars)
    if profile=="chain_dedup_no_sources":
        return _render_chain_dedup_no_sources(bundles,max_chars,support_limit=None)
    if profile=="chain_dedup_keep_sources":
        return _render_chain_dedup_keep_sources(bundles,max_chars,source_sentence_limit_per_chunk=None)
    if profile=="source_light_compact":
        return _render_chain_dedup_keep_sources(bundles,max_chars,source_sentence_limit_per_chunk=1)
    if profile=="chain_skeleton":
        return _render_chain_skeleton(bundles,max_chars,include_plus_one=False)
    if profile=="chain_skeleton_no_sources":
        return _render_chain_skeleton_no_sources(bundles,max_chars)
    if profile=="chain_plus1":
        return _render_chain_skeleton(bundles,max_chars,include_plus_one=True)
    if profile=="chain_dedup_plus1_no_sources":
        return _render_chain_dedup_no_sources(bundles,max_chars,support_limit=1)
    if profile=="sentence_cap":
        return _render_sentence_cap(bundles,max_chars,max_sentences_per_bundle=max_sentences_per_bundle)
    if profile=="sentence_cap_no_sources":
        return _render_sentence_cap_no_sources(bundles,max_chars,max_sentences_per_bundle=max_sentences_per_bundle)
    if profile=="top3_chain_dedup":
        return _render_chain_dedup(_ordered_bundles(bundles)[:3],max_chars)
    if profile=="top3_chain_dedup_no_sources":
        return _render_chain_dedup_no_sources(_ordered_bundles(bundles)[:3],max_chars,support_limit=None)
    raise ValueError(f"Unsupported compaction_profile={profile!r}; choices={list(COMPACTION_PROFILES)}")

def render_context(
    evidence_bundles: Sequence[Mapping[str,Any]],
    rendering_profile: str = DEFAULT_RENDERING_PROFILE,
    max_chars: int = 24000,
    token_budget: int | None = None,
    compaction_profile: str = "none",
    max_sentences_per_bundle: int = 3,
    **_: Any,
) -> str:
    profile=str(rendering_profile or DEFAULT_RENDERING_PROFILE)
    if profile not in RENDERING_PROFILES:
        raise ValueError(f"Unsupported rendering_profile={profile!r}; choices={list(RENDERING_PROFILES)}")
    compact=str(compaction_profile or "none")
    if compact not in COMPACTION_PROFILES:
        raise ValueError(f"Unsupported compaction_profile={compact!r}; choices={list(COMPACTION_PROFILES)}")
    char_budget=int(max_chars or 24000)
    if token_budget:
        char_budget=min(char_budget, int(token_budget)*6)
    if compact!="none":
        return _render_compacted_context(evidence_bundles,char_budget,compact,max_sentences_per_bundle=max_sentences_per_bundle)
    if profile=="structured_chain":
        return _render_structured_chain(evidence_bundles,char_budget)
    if profile=="plain_evidence":
        return _render_plain_evidence(evidence_bundles,char_budget)
    if profile=="chain_only_compact":
        return _render_chain_only_compact(evidence_bundles,char_budget)
    if profile=="multi_anchor_table":
        return _render_multi_anchor_table(evidence_bundles,char_budget)
    raise AssertionError(profile)

def render_context_with_metadata(
    evidence_bundles: Sequence[Mapping[str,Any]],
    rendering_profile: str = DEFAULT_RENDERING_PROFILE,
    max_chars: int = 24000,
    token_budget: int | None = None,
    compaction_profile: str = "none",
    max_sentences_per_bundle: int = 3,
) -> tuple[str,dict[str,Any]]:
    original_stats=bundle_sentence_statistics(evidence_bundles)
    context=render_context(
        evidence_bundles,
        rendering_profile=rendering_profile,
        max_chars=max_chars,
        token_budget=token_budget,
        compaction_profile=compaction_profile,
        max_sentences_per_bundle=max_sentences_per_bundle,
    )
    rendered_stats=bundle_sentence_statistics(evidence_bundles,context)
    stats=dict(rendered_stats)
    compact=str(compaction_profile or "none")
    no_sources="no_sources" in compact
    stats.update({
        "original_sentence_count":original_stats.get("rendered_sentence_count",0),
        "dropped_sentence_count":max(0,int(original_stats.get("rendered_sentence_count",0) or 0)-int(rendered_stats.get("rendered_sentence_count",0) or 0)),
        "duplicate_removed_count":max(0,int(original_stats.get("duplicate_sentence_count",0) or 0)-int(rendered_stats.get("duplicate_sentence_count",0) or 0)),
        "source_removed_count":int(original_stats.get("source_sentence_count",0) or 0) if no_sources else 0,
        "metadata_removed_count":_metadata_removed_count(evidence_bundles) if compact!="none" else 0,
        "compaction_profile":compact,
        "max_sentences_per_bundle":int(max_sentences_per_bundle or 3),
    })
    return context,stats

def build_context_text(bundles: Sequence[Mapping[str,Any]], max_chars: int=24000, rendering_profile: str=DEFAULT_RENDERING_PROFILE) -> str:
    return render_context(bundles, rendering_profile=rendering_profile, max_chars=max_chars)

def build_prompt(question: str, context: str, prompt_profile: str = DEFAULT_PROMPT_PROFILE) -> str:
    profile = str(prompt_profile or DEFAULT_PROMPT_PROFILE)
    if profile not in PROMPT_TEMPLATES:
        raise ValueError(f"Unsupported prompt_profile={profile!r}; choices={sorted(PROMPT_TEMPLATES)}")
    return PROMPT_TEMPLATES[profile].format(question=question, context=context)

def is_insufficient_prediction(prediction: Any) -> bool:
    text=str(prediction or "").lower()
    return any(phrase in text for phrase in INSUFFICIENT_PHRASES)

def has_idk_phrase(prediction: Any) -> bool:
    text=str(prediction or "").lower()
    return any(phrase in text for phrase in IDK_PHRASES)

def normalize_prediction_for_eval(prediction: Any) -> str:
    if prediction is None:
        return ""
    return str(prediction).strip()

def build_generation_prompt(question: str, bundles: Sequence[Mapping[str,Any]], cfg: Optional[Mapping[str,Any]]=None) -> tuple[str, str, str]:
    cfg=cfg or {}
    prompt_profile=str(cfg.get("prompt_profile") or DEFAULT_PROMPT_PROFILE)
    rendering_profile=str(cfg.get("rendering_profile") or DEFAULT_RENDERING_PROFILE)
    context=render_context(
        bundles,
        rendering_profile,
        int(cfg.get("max_context_chars",24000)),
        cfg.get("context_token_budget"),
        compaction_profile=str(cfg.get("compaction_profile") or "none"),
        max_sentences_per_bundle=int(cfg.get("max_sentences_per_bundle",3) or 3),
    )
    return build_prompt(question, context, prompt_profile), prompt_profile, context

def extractive_fallback_answer(question: str, bundles: Sequence[Mapping[str,Any]]) -> str:
    cands=[]
    for b in bundles:
        for p in b.get("propositions",[]) or []:
            if p.get("text"): cands.append(str(p["text"]))
        for c in b.get("source_chunks",[]) or []: cands.extend(sentence_split(str(c.get("text","")))[:2])
    if not cands: return "I don't know"
    q=question.lower()
    if any(x in q for x in ["occupation","profession","job"]):
        for s in cands:
            m=re.search(r"\b(?:is|was|are|were)\s+(?:an?|the)?\s*([^.;]+?)(?:\s+who\b|\s+born\b|\.)", s, flags=re.I)
            if m: return m.group(1).strip(" ,")
    return min(cands, key=lambda x: abs(token_count(x)-18))

def _resolve_model(client: Any, configured: str) -> str:
    if configured and configured.lower() not in {"auto",""}: return configured
    models=client.models.list()
    if not models.data: raise RuntimeError("vLLM /v1/models returned no models; set generation.model explicitly.")
    return models.data[0].id

def _generate_vllm(prompt: str, cfg: Mapping[str,Any]) -> Dict[str,Any]:
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("openai package is required for generation.provider=vllm") from e
    base_url=str(cfg.get("base_url") or os.environ.get("VLLM_BASE_URL") or "http://127.0.0.1:8011/v1"); api_key=str(cfg.get("api_key") or os.environ.get("VLLM_API_KEY") or "EMPTY")
    client=OpenAI(base_url=base_url, api_key=api_key, timeout=float(cfg.get("timeout_s", cfg.get("timeout", 120))))
    model=_resolve_model(client, str(cfg.get("model","auto")))
    messages=[]
    system_message=cfg.get("system_message") if cfg.get("system_message") is not None else cfg.get("system_prompt")
    if system_message:
        messages.append({"role":"system","content":str(system_message)})
    messages.append({"role":"user","content":prompt})
    req={"model":model,"messages":messages,"temperature":float(cfg.get("temperature",0.0)),"max_tokens":int(cfg.get("max_new_tokens",64))}
    if cfg.get("top_p") is not None: req["top_p"]=float(cfg.get("top_p"))
    if cfg.get("stop"): req["stop"]=cfg.get("stop")
    if cfg.get("extra_body"): req["extra_body"]=dict(cfg.get("extra_body") or {})
    resp=client.chat.completions.create(**req); pred=resp.choices[0].message.content or ""; usage=getattr(resp,"usage",None)
    return {"raw_prediction":pred,"prediction":normalize_prediction_for_eval(pred),"llm_provider":"vllm","generation_provider":"vllm","model":model,"base_url":base_url,"usage":usage.model_dump() if hasattr(usage,"model_dump") else {}}

def generate_answer(question: str, bundles: Sequence[Mapping[str,Any]], cfg: Mapping[str,Any]) -> Dict[str,Any]:
    provider=str(cfg.get("provider","none")).lower()
    render_t0=time.perf_counter()
    prompt,prompt_profile,rendered_context=build_generation_prompt(question,bundles,cfg)
    context_rendering_s=round(time.perf_counter()-render_t0,6)
    rendering_profile=str(cfg.get("rendering_profile") or DEFAULT_RENDERING_PROFILE)
    t0=time.perf_counter()
    if provider in {"none","extractive","fallback","no-llm"}:
        raw_prediction=extractive_fallback_answer(question,bundles)
        generation_s=round(time.perf_counter()-t0,6)
        return {"raw_prediction":raw_prediction,"prediction":normalize_prediction_for_eval(raw_prediction),"rendered_context":rendered_context,"prompt":prompt,"prompt_profile":prompt_profile,"rendering_profile":rendering_profile,"generation_latency_s":generation_s,"generation_stage_timings_s":{"context_rendering":context_rendering_s,"generation":generation_s},"llm_provider":"extractive_fallback","generation_provider":"extractive_fallback"}
    if provider=="vllm":
        last=None
        for attempt in range(int(cfg.get("retries",1))+1):
            try:
                out=_generate_vllm(prompt,cfg)
                generation_s=round(time.perf_counter()-t0,6)
                out["prompt"]=prompt; out["rendered_context"]=rendered_context; out["prompt_profile"]=prompt_profile; out["rendering_profile"]=rendering_profile; out["generation_latency_s"]=generation_s; out["generation_stage_timings_s"]={"context_rendering":context_rendering_s,"generation":generation_s}; return out
            except Exception as e:
                last=e
                if attempt<int(cfg.get("retries",1)): time.sleep(float(cfg.get("retry_sleep_s",2.0)))
        generation_s=round(time.perf_counter()-t0,6)
        return {"raw_prediction":"","prediction":"","rendered_context":rendered_context,"prompt":prompt,"prompt_profile":prompt_profile,"rendering_profile":rendering_profile,"generation_latency_s":generation_s,"generation_stage_timings_s":{"context_rendering":context_rendering_s,"generation":generation_s},"llm_provider":"vllm","generation_provider":"vllm","generation_error":repr(last)}
    raise ValueError(f"Unsupported generation.provider={provider}")
