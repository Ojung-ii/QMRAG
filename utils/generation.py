from __future__ import annotations
import os, re, time
from typing import Any, Dict, List, Mapping, Optional, Sequence
from .text import safe_truncate, sentence_split, token_count

DEFAULT_PROMPT_PROFILE = "common_qa"

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

def build_context_text(bundles: Sequence[Mapping[str,Any]], max_chars: int=24000) -> str:
    parts=[]
    order_rank={"anchor_connected_chain_complete":0,"multi_anchor":1,"anchor":2,"chain_complete_v2":3,"bridge_connected":4,"same_title":5,"generic_relation":6,"fallback":7,"complete_bridge_chain":3,"exact_query_anchor":2,"bridge_candidate":4}
    ordered=sorted(
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
    for i,(_,b) in enumerate(ordered, start=1):
        bridge=", ".join(str(x) for x in b.get("bridge_titles",[]) or [])
        bundle_type=str(b.get("bundle_type") or "")
        bridge_paths=[x for x in b.get("evidence_path",[]) or [] if isinstance(x,Mapping) and x.get("path_type")=="mention_bridge"]
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
    context=build_context_text(bundles, int(cfg.get("max_context_chars",24000)))
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
    provider=str(cfg.get("provider","none")).lower(); prompt,prompt_profile,rendered_context=build_generation_prompt(question,bundles,cfg); t0=time.perf_counter()
    if provider in {"none","extractive","fallback","no-llm"}:
        raw_prediction=extractive_fallback_answer(question,bundles)
        return {"raw_prediction":raw_prediction,"prediction":normalize_prediction_for_eval(raw_prediction),"rendered_context":rendered_context,"prompt":prompt,"prompt_profile":prompt_profile,"generation_latency_s":round(time.perf_counter()-t0,6),"llm_provider":"extractive_fallback","generation_provider":"extractive_fallback"}
    if provider=="vllm":
        last=None
        for attempt in range(int(cfg.get("retries",1))+1):
            try:
                out=_generate_vllm(prompt,cfg); out["prompt"]=prompt; out["rendered_context"]=rendered_context; out["prompt_profile"]=prompt_profile; out["generation_latency_s"]=round(time.perf_counter()-t0,6); return out
            except Exception as e:
                last=e
                if attempt<int(cfg.get("retries",1)): time.sleep(float(cfg.get("retry_sleep_s",2.0)))
        return {"raw_prediction":"","prediction":"","rendered_context":rendered_context,"prompt":prompt,"prompt_profile":prompt_profile,"generation_latency_s":round(time.perf_counter()-t0,6),"llm_provider":"vllm","generation_provider":"vllm","generation_error":repr(last)}
    raise ValueError(f"Unsupported generation.provider={provider}")
