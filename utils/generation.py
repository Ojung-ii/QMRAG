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

def render_context(
    evidence_bundles: Sequence[Mapping[str,Any]],
    rendering_profile: str = DEFAULT_RENDERING_PROFILE,
    max_chars: int = 24000,
    token_budget: int | None = None,
    **_: Any,
) -> str:
    profile=str(rendering_profile or DEFAULT_RENDERING_PROFILE)
    if profile not in RENDERING_PROFILES:
        raise ValueError(f"Unsupported rendering_profile={profile!r}; choices={list(RENDERING_PROFILES)}")
    char_budget=int(max_chars or 24000)
    if token_budget:
        char_budget=min(char_budget, int(token_budget)*6)
    if profile=="structured_chain":
        return _render_structured_chain(evidence_bundles,char_budget)
    if profile=="plain_evidence":
        return _render_plain_evidence(evidence_bundles,char_budget)
    if profile=="chain_only_compact":
        return _render_chain_only_compact(evidence_bundles,char_budget)
    if profile=="multi_anchor_table":
        return _render_multi_anchor_table(evidence_bundles,char_budget)
    raise AssertionError(profile)

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
    context=render_context(bundles, rendering_profile, int(cfg.get("max_context_chars",24000)), cfg.get("context_token_budget"))
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
    provider=str(cfg.get("provider","none")).lower(); prompt,prompt_profile,rendered_context=build_generation_prompt(question,bundles,cfg); rendering_profile=str(cfg.get("rendering_profile") or DEFAULT_RENDERING_PROFILE); t0=time.perf_counter()
    if provider in {"none","extractive","fallback","no-llm"}:
        raw_prediction=extractive_fallback_answer(question,bundles)
        return {"raw_prediction":raw_prediction,"prediction":normalize_prediction_for_eval(raw_prediction),"rendered_context":rendered_context,"prompt":prompt,"prompt_profile":prompt_profile,"rendering_profile":rendering_profile,"generation_latency_s":round(time.perf_counter()-t0,6),"llm_provider":"extractive_fallback","generation_provider":"extractive_fallback"}
    if provider=="vllm":
        last=None
        for attempt in range(int(cfg.get("retries",1))+1):
            try:
                out=_generate_vllm(prompt,cfg); out["prompt"]=prompt; out["rendered_context"]=rendered_context; out["prompt_profile"]=prompt_profile; out["rendering_profile"]=rendering_profile; out["generation_latency_s"]=round(time.perf_counter()-t0,6); return out
            except Exception as e:
                last=e
                if attempt<int(cfg.get("retries",1)): time.sleep(float(cfg.get("retry_sleep_s",2.0)))
        return {"raw_prediction":"","prediction":"","rendered_context":rendered_context,"prompt":prompt,"prompt_profile":prompt_profile,"rendering_profile":rendering_profile,"generation_latency_s":round(time.perf_counter()-t0,6),"llm_provider":"vllm","generation_provider":"vllm","generation_error":repr(last)}
    raise ValueError(f"Unsupported generation.provider={provider}")
