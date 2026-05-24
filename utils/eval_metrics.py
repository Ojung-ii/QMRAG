from __future__ import annotations
from collections import Counter
from datetime import datetime
import json
from typing import Any, Mapping, Sequence
from .text import normalize_answer, token_count
from .generation import build_prompt, count_tokens, extract_usage_token_counts, has_idk_phrase, is_insufficient_prediction, prompt_template_token_count

def exact_match(pred: str, golds: Sequence[str]) -> float:
    p=normalize_answer(pred); return float(any(p==normalize_answer(g) for g in golds if str(g).strip()))
def f1_score(pred: str, gold: str) -> float:
    p=normalize_answer(pred).split(); g=normalize_answer(gold).split()
    if not p and not g: return 1.0
    if not p or not g: return 0.0
    same=sum((Counter(p)&Counter(g)).values())
    if same==0: return 0.0
    pr=same/len(p); rc=same/len(g); return 2*pr*rc/(pr+rc)
def answer_f1(pred: str, golds: Sequence[str]) -> float:
    return max([f1_score(pred,g) for g in golds if str(g).strip()] or [0.0])
def answer_contains(pred: str, golds: Sequence[str]) -> float:
    p=normalize_answer(pred)
    return float(any(normalize_answer(g) and normalize_answer(g) in p for g in golds))

def text_has_gold(text: Any, golds: Sequence[str]) -> float:
    ctx=str(text or "")
    ctx_l=ctx.lower(); ctx_norm=normalize_answer(ctx)
    for gold in golds:
        g=str(gold).strip()
        if not g:
            continue
        if g.lower() in ctx_l or normalize_answer(g) in ctx_norm:
            return 1.0
    return 0.0

def answer_in_evidence_bundles(row: Mapping[str,Any], golds: Sequence[str]) -> float:
    try:
        ctx=json.dumps(row.get("evidence_bundles",[]) or [], ensure_ascii=False)
    except Exception:
        ctx=str(row.get("evidence_bundles",[]) or "")
    return text_has_gold(ctx,golds)

def answer_in_rendered_context(row: Mapping[str,Any], golds: Sequence[str]) -> float:
    ctx=row.get("rendered_context")
    if ctx is None:
        ctx=row.get("rendered_context_preview","")
    return text_has_gold(ctx,golds)
def support_title_recall(row: Mapping[str,Any]) -> float:
    gold={str(x).strip().lower() for x in row.get("support_titles",[]) if str(x).strip()}
    if not gold: return 0.0
    pred=set()
    for b in row.get("evidence_bundles",[]) or []:
        if b.get("anchor_title"): pred.add(str(b.get("anchor_title")).strip().lower())
        for key in ("propositions","source_chunks"):
            for x in b.get(key,[]) or []:
                if x.get("title"): pred.add(str(x.get("title")).strip().lower())
    return len(gold&pred)/max(1,len(gold))
def context_tokens(row: Mapping[str,Any]) -> int:
    if row.get("rendered_context_tokens") is not None: return int(row.get("rendered_context_tokens") or 0)
    d=row.get("retrieval_diagnostics",{}) or {}
    if d.get("context_tokens") is not None: return int(d.get("context_tokens") or 0)
    return 0

def _context_text(row: Mapping[str,Any]) -> str:
    ctx=row.get("rendered_context")
    if ctx is None:
        ctx=row.get("rendered_context_preview","")
    return str(ctx or "")

def _first_int(*values: Any) -> int | None:
    for value in values:
        if value is None:
            continue
        try:
            return int(value)
        except Exception:
            continue
    return None

def row_token_metrics(row: Mapping[str,Any]) -> dict[str,Any]:
    prompt_profile=str(row.get("prompt_profile") or "common_qa")
    context=_context_text(row)
    raw_pred=str(row.get("raw_prediction", row.get("prediction", "")) or "")
    usage_counts=extract_usage_token_counts(row.get("llm_usage"))
    usage_prompt=_first_int(row.get("llm_usage_prompt_tokens"), usage_counts.get("prompt_tokens"))
    usage_completion=_first_int(row.get("llm_usage_completion_tokens"), usage_counts.get("completion_tokens"))
    usage_total=_first_int(row.get("llm_usage_total_tokens"), usage_counts.get("total_tokens"))
    rendered=_first_int(row.get("rendered_context_tokens"), context_tokens(row))
    template=_first_int(row.get("prompt_template_tokens"), prompt_template_token_count(prompt_profile))
    input_prompt=_first_int(row.get("input_prompt_tokens"), usage_prompt)
    if input_prompt is None:
        if row.get("prompt"):
            prompt=str(row.get("prompt") or "")
        else:
            try:
                prompt=build_prompt(str(row.get("question","")), context, prompt_profile)
            except Exception:
                prompt=f"Question: {row.get('question','')}\nContext:\n{context}\nAnswer:"
        input_prompt=count_tokens(prompt)
    completion=_first_int(row.get("completion_tokens"), usage_completion)
    if completion is None:
        completion=count_tokens(raw_pred)
    total=_first_int(row.get("total_llm_tokens"), usage_total)
    if total is None:
        total=int(input_prompt)+int(completion)
    source=str(row.get("token_count_source") or ("usage" if usage_prompt is not None or usage_completion is not None or usage_total is not None else "approx"))
    return {
        "prompt_template_tokens":int(template or 0),
        "rendered_context_tokens":int(rendered or 0),
        "input_prompt_tokens":int(input_prompt or 0),
        "completion_tokens":int(completion or 0),
        "total_llm_tokens":int(total or 0),
        "token_count_source":source,
        "llm_usage_prompt_tokens":usage_prompt,
        "llm_usage_completion_tokens":usage_completion,
        "llm_usage_total_tokens":usage_total,
    }

def bridge_stats(row: Mapping[str,Any]) -> dict[str,float]:
    bundles=row.get("evidence_bundles",[]) or []
    titles=set()
    bridge_bundles=0
    chain_complete=0
    bridge_connected=0
    answer_slot_aligned=0
    chain_complete_v2=0
    anchor_connected_chain_complete=0
    anchor_mismatch_chain=0
    multi_anchor_bundle=0
    residual_counts=[]
    for bundle in bundles:
        for title in bundle.get("bridge_titles",[]) or []:
            titles.add(str(title))
        if bundle.get("has_bridge"):
            bridge_bundles+=1
        if bundle.get("chain_complete"):
            chain_complete+=1
        if bundle.get("bridge_connected"):
            bridge_connected+=1
        if bundle.get("answer_slot_aligned"):
            answer_slot_aligned+=1
        if bundle.get("chain_complete_v2"):
            chain_complete_v2+=1
        if bundle.get("anchor_connected_chain_complete"):
            anchor_connected_chain_complete+=1
        if bundle.get("anchor_mismatch_chain"):
            anchor_mismatch_chain+=1
        if bundle.get("bundle_type")=="multi_anchor":
            multi_anchor_bundle+=1
        residual_counts.append(float(bundle.get("residual_coverage_count",0.0) or 0.0))
    rd=row.get("retrieval_diagnostics",{}) or {}
    bridge_connected_count=float(rd.get("bridge_connected_count",bridge_connected) or 0)
    answer_slot_aligned_count=float(rd.get("answer_slot_aligned_count",answer_slot_aligned) or 0)
    chain_complete_v2_count=float(rd.get("chain_complete_v2_count",chain_complete_v2) or 0)
    anchor_connected_chain_complete_count=float(rd.get("anchor_connected_chain_complete_count",anchor_connected_chain_complete) or 0)
    anchor_mismatch_chain_count=float(rd.get("anchor_mismatch_chain_count",anchor_mismatch_chain) or 0)
    multi_anchor_bundle_count=float(rd.get("multi_anchor_bundle_count",multi_anchor_bundle) or 0)
    avg_residual=float(rd.get("avg_residual_coverage_count",sum(residual_counts)/max(1,len(residual_counts))) or 0)
    return {
        "bridge_title_count":float(rd.get("bridge_title_count",len(titles)) or 0),
        "bridge_bundle_count":float(rd.get("bridge_bundle_count",bridge_bundles) or 0),
        "chain_complete_count":float(rd.get("chain_complete_count",chain_complete) or 0),
        "has_chain_complete":1.0 if bool(rd.get("has_chain_complete",chain_complete>0)) else 0.0,
        "bridge_connected_count":bridge_connected_count,
        "answer_slot_aligned_count":answer_slot_aligned_count,
        "chain_complete_v2_count":chain_complete_v2_count,
        "has_bridge_connected":1.0 if bool(rd.get("has_bridge_connected",bridge_connected_count>0)) else 0.0,
        "has_answer_slot_aligned":1.0 if bool(rd.get("has_answer_slot_aligned",answer_slot_aligned_count>0)) else 0.0,
        "has_chain_complete_v2":1.0 if bool(rd.get("has_chain_complete_v2",chain_complete_v2_count>0)) else 0.0,
        "avg_residual_coverage_count":avg_residual,
        "anchor_connected_chain_complete_count":anchor_connected_chain_complete_count,
        "anchor_mismatch_chain_count":anchor_mismatch_chain_count,
        "multi_anchor_bundle_count":multi_anchor_bundle_count,
        "has_anchor_connected_chain_complete":1.0 if bool(rd.get("has_anchor_connected_chain_complete",anchor_connected_chain_complete_count>0)) else 0.0,
        "has_anchor_mismatch_chain":1.0 if bool(rd.get("has_anchor_mismatch_chain",anchor_mismatch_chain_count>0)) else 0.0,
        "has_multi_anchor_bundle":1.0 if bool(rd.get("has_multi_anchor_bundle",multi_anchor_bundle_count>0)) else 0.0,
        "generic_relation_top1":1.0 if bool(rd.get("generic_relation_top1",bool(bundles and bundles[0].get("is_relation_title_bundle")))) else 0.0,
        "query_anchor_coverage":float(rd.get("query_anchor_coverage",0.0) or 0.0),
    }
def first_present(rows: Sequence[Mapping[str,Any]], key: str, fallback: Any=None) -> Any:
    for row in rows:
        value=row.get(key)
        if value not in {None, ""}:
            return value
    return fallback

def evaluate_predictions(rows: Sequence[Mapping[str,Any]], dataset: str | None=None, prompt_profile: str | None=None) -> dict[str,Any]:
    per=[]
    seed_totals: Counter[str]=Counter()
    selected_totals: Counter[str]=Counter()
    chain_totals: Counter[str]=Counter()
    anchor_chain_totals: Counter[str]=Counter()
    aligned_totals: Counter[str]=Counter()
    for r in rows:
        d=r.get("retrieval_diagnostics",{}) or {}; timings=d.get("timings",{}) or {}
        seed_totals.update({str(k):int(v or 0) for k,v in dict(d.get("seed_unit_type_distribution",{}) or {}).items()})
        selected_totals.update({str(k):int(v or 0) for k,v in dict(d.get("selected_bundle_source_type_distribution",{}) or {}).items()})
        chain_totals.update({str(k):int(v or 0) for k,v in dict(d.get("chain_success_by_seed_type",{}) or {}).items()})
        anchor_chain_totals.update({str(k):int(v or 0) for k,v in dict(d.get("anchor_connected_chain_by_seed_type",{}) or {}).items()})
        aligned_totals.update({str(k):int(v or 0) for k,v in dict(d.get("answer_slot_aligned_by_seed_type",{}) or {}).items()})
        raw_pred=str(r.get("raw_prediction", r.get("prediction", "")) or "")
        pred=raw_pred
        golds=[str(x) for x in r.get("answers",[])]
        ret_ms=1000*float(timings.get("total_retrieval_s",0.0)); gen_ms=1000*float(r.get("generation_latency_s",0.0))
        in_bundles=answer_in_evidence_bundles(r,golds); in_rendered=answer_in_rendered_context(r,golds); in_prediction=answer_contains(pred,golds)
        bs=bridge_stats(r); tm=row_token_metrics(r)
        per.append({"id":r.get("id"),"em":exact_match(pred,golds),"f1":answer_f1(pred,golds),"answer_contains":in_prediction,"answer_in_context":in_bundles,"answer_in_evidence_bundles":in_bundles,"answer_in_rendered_context":in_rendered,"answer_in_prediction":in_prediction,"idk":has_idk_phrase(raw_pred),"insufficient":is_insufficient_prediction(raw_pred),"support_title_recall":support_title_recall(r),"context_tokens":tm["rendered_context_tokens"],"latency_ms":ret_ms+gen_ms,"retrieval_latency_ms":ret_ms,"generation_latency_ms":gen_ms,"candidate_count":int(d.get("candidate_count") or 0),"seed_count":int(d.get("seed_count") or 0),"bundle_count":int(d.get("bundle_count") or 0),"rendered_bundle_count":float(r.get("rendered_bundle_count",len(r.get("evidence_bundles",[]) or [])) or 0.0),"dropped_bundle_count":float(r.get("dropped_bundle_count",0.0) or 0.0),"dense_enabled":bool(d.get("dense_enabled",False)),**bs,**tm})
    avg=lambda k: sum(float(x[k]) for x in per)/max(1,len(per))
    resolved_prompt=prompt_profile or first_present(rows,"prompt_profile","UNKNOWN")
    rendering_profile=first_present(rows,"rendering_profile","structured_chain")
    prompt_experiment_type=first_present(rows,"prompt_experiment_type", "main_comparison" if resolved_prompt=="common_qa" else "ablation" if resolved_prompt in {"qmrag_bundle_qa","qmrag_bundle_light","qmrag_bundle_tiny"} else "unknown")
    res={"dataset":dataset or first_present(rows,"dataset","UNKNOWN"),"prompt_profile":resolved_prompt,"rendering_profile":rendering_profile,"retrieval_variant":first_present(rows,"retrieval_variant","full_hetero"),"prompt_experiment_type":prompt_experiment_type,"generation_provider":first_present(rows,"generation_provider",first_present(rows,"llm_provider","UNKNOWN")),"created_at":datetime.now().isoformat(timespec="seconds"),"n":len(rows),"em":avg("em"),"f1":avg("f1"),"answer_contains":avg("answer_contains"),"support_title_recall":avg("support_title_recall"),"context_tokens":avg("context_tokens"),"latency_ms":avg("latency_ms"),"retrieval_latency_ms":avg("retrieval_latency_ms"),"generation_latency_ms":avg("generation_latency_ms"),"candidate_count":avg("candidate_count"),"seed_count":avg("seed_count"),"bundle_count":avg("bundle_count"),"dense_enabled_rate":sum(1.0 if x["dense_enabled"] else 0.0 for x in per)/max(1,len(per)),"answer_in_context":avg("answer_in_context"),"answer_in_evidence_bundles":avg("answer_in_evidence_bundles"),"answer_in_rendered_context":avg("answer_in_rendered_context"),"answer_in_prediction":avg("answer_in_prediction"),"idk_rate":sum(1.0 if x["idk"] else 0.0 for x in per)/max(1,len(per)),"insufficient_rate":sum(1.0 if x["insufficient"] else 0.0 for x in per)/max(1,len(per)),"avg_bridge_title_count":avg("bridge_title_count"),"avg_bridge_bundle_count":avg("bridge_bundle_count"),"chain_complete_rate":sum(1.0 if x["has_chain_complete"] else 0.0 for x in per)/max(1,len(per)),"bridge_connected_rate":sum(1.0 if x["has_bridge_connected"] else 0.0 for x in per)/max(1,len(per)),"answer_slot_aligned_rate":sum(1.0 if x["has_answer_slot_aligned"] else 0.0 for x in per)/max(1,len(per)),"chain_complete_v2_rate":sum(1.0 if x["has_chain_complete_v2"] else 0.0 for x in per)/max(1,len(per)),"anchor_connected_chain_complete_rate":sum(1.0 if x["has_anchor_connected_chain_complete"] else 0.0 for x in per)/max(1,len(per)),"anchor_mismatch_chain_rate":sum(1.0 if x["has_anchor_mismatch_chain"] else 0.0 for x in per)/max(1,len(per)),"multi_anchor_bundle_rate":sum(1.0 if x["has_multi_anchor_bundle"] else 0.0 for x in per)/max(1,len(per)),"generic_relation_top1_rate":avg("generic_relation_top1"),"query_anchor_coverage_rate":avg("query_anchor_coverage"),"avg_residual_coverage_count":avg("avg_residual_coverage_count"),"per_example":per}
    seed_total=sum(seed_totals.values()) or 1
    selected_total=sum(selected_totals.values()) or 1
    chain_total=sum(chain_totals.values()) or 1
    res.update({
        "seed_unit_type_distribution":dict(seed_totals),
        "selected_bundle_source_type_distribution":dict(selected_totals),
        "chain_success_by_seed_type":dict(chain_totals),
        "anchor_connected_chain_by_seed_type":dict(anchor_chain_totals),
        "answer_slot_aligned_by_seed_type":dict(aligned_totals),
        "seed_title_rate":seed_totals.get("title",0)/seed_total,
        "seed_chunk_rate":seed_totals.get("chunk",0)/seed_total,
        "seed_proposition_rate":seed_totals.get("proposition",0)/seed_total,
        "selected_title_rate":selected_totals.get("title",0)/selected_total,
        "selected_chunk_rate":selected_totals.get("chunk",0)/selected_total,
        "selected_proposition_rate":selected_totals.get("proposition",0)/selected_total,
        "chain_from_title_rate":chain_totals.get("title",0)/chain_total,
        "chain_from_chunk_rate":chain_totals.get("chunk",0)/chain_total,
        "chain_from_proposition_rate":chain_totals.get("proposition",0)/chain_total,
    })
    res.update({"avg_prompt_template_tokens":avg("prompt_template_tokens"),"avg_rendered_context_tokens":avg("rendered_context_tokens"),"avg_context_tokens":avg("rendered_context_tokens"),"avg_input_prompt_tokens":avg("input_prompt_tokens"),"avg_completion_tokens":avg("completion_tokens"),"avg_total_llm_tokens":avg("total_llm_tokens"),"avg_rendered_bundle_count":avg("rendered_bundle_count"),"avg_dropped_bundle_count":avg("dropped_bundle_count"),"token_count_source_counts":dict(Counter(str(x.get("token_count_source","unknown")) for x in per))})
    res["support_recall_per_1k_tokens"]=res["support_title_recall"]/max(1e-9,res["context_tokens"]/1000.0)
    res["F1_per_1k_context_tokens"]=res["f1"]/max(1e-9,res["avg_rendered_context_tokens"]/1000.0)
    res["F1_per_1k_input_prompt_tokens"]=res["f1"]/max(1e-9,res["avg_input_prompt_tokens"]/1000.0)
    res["F1_per_1k_total_llm_tokens"]=res["f1"]/max(1e-9,res["avg_total_llm_tokens"]/1000.0)
    res.update({"EM":res["em"],"F1":res["f1"],"AnsContains":res["answer_contains"],"SupportRecall":res["support_title_recall"],"SR/1kTok":res["support_recall_per_1k_tokens"],"CtxTok":res["context_tokens"],"LatencyMs":res["latency_ms"],"DenseRate":res["dense_enabled_rate"]})
    return res
def summary_markdown(dataset: str, result: Mapping[str,Any]) -> str:
    ds=result.get("dataset") or dataset or "UNKNOWN"
    prompt=result.get("prompt_profile") or "UNKNOWN"
    header=["# QMRAG Evaluation Summary","",f"- dataset: {ds}",f"- prompt_profile: {prompt}",f"- rendering_profile: {result.get('rendering_profile','structured_chain')}",f"- retrieval_variant: {result.get('retrieval_variant','full_hetero')}",f"- n: {result.get('n',0)}",f"- generation_provider: {result.get('generation_provider','UNKNOWN')}",f"- created_at: {result.get('created_at','UNKNOWN')}",""]
    header.insert(4, f"- prompt_experiment_type: {result.get('prompt_experiment_type','unknown')}")
    meta=[]
    if result.get("index_source") is not None:
        meta.append(("index_source",result.get("index_source")))
    if result.get("index_dir") is not None:
        meta.append(("index_dir",result.get("index_dir")))
    rows=meta+[("EM",f"{result.get('em',0):.4f}"),("F1",f"{result.get('f1',0):.4f}"),("AnsContains",f"{result.get('answer_contains',0):.4f}"),("SupportRecall",f"{result.get('support_title_recall',0):.4f}"),("SR/1kTok",f"{result.get('support_recall_per_1k_tokens',0):.4f}"),("CtxTok",f"{result.get('context_tokens',0):.1f}"),("RenderedBundles",f"{result.get('avg_rendered_bundle_count',0):.2f}"),("DroppedBundles",f"{result.get('avg_dropped_bundle_count',0):.2f}"),("PromptTemplateTok",f"{result.get('avg_prompt_template_tokens',0):.1f}"),("InputPromptTok",f"{result.get('avg_input_prompt_tokens',0):.1f}"),("CompletionTok",f"{result.get('avg_completion_tokens',0):.1f}"),("TotalLLMTok",f"{result.get('avg_total_llm_tokens',0):.1f}"),("F1/1kContextTok",f"{result.get('F1_per_1k_context_tokens',0):.4f}"),("F1/1kInputTok",f"{result.get('F1_per_1k_input_prompt_tokens',0):.4f}"),("F1/1kTotalLLMTok",f"{result.get('F1_per_1k_total_llm_tokens',0):.4f}"),("TokenCountSources",json.dumps(result.get('token_count_source_counts',{}),ensure_ascii=False)),("LatencyMs",f"{result.get('latency_ms',0):.1f}"),("DenseRate",f"{result.get('dense_enabled_rate',0):.2f}"),("SeedTitleRate",f"{result.get('seed_title_rate',0):.4f}"),("SeedChunkRate",f"{result.get('seed_chunk_rate',0):.4f}"),("SeedPropositionRate",f"{result.get('seed_proposition_rate',0):.4f}"),("ChainFromTitleRate",f"{result.get('chain_from_title_rate',0):.4f}"),("ChainFromChunkRate",f"{result.get('chain_from_chunk_rate',0):.4f}"),("ChainFromPropositionRate",f"{result.get('chain_from_proposition_rate',0):.4f}"),("AvgBridgeTitleCount",f"{result.get('avg_bridge_title_count',0):.2f}"),("AvgBridgeBundleCount",f"{result.get('avg_bridge_bundle_count',0):.2f}"),("BridgeConnectedRate",f"{result.get('bridge_connected_rate',0):.4f}"),("AnswerSlotAlignedRate",f"{result.get('answer_slot_aligned_rate',0):.4f}"),("ChainCompleteV2Rate",f"{result.get('chain_complete_v2_rate',0):.4f}"),("AnchorConnectedChainCompleteRate",f"{result.get('anchor_connected_chain_complete_rate',0):.4f}"),("AnchorMismatchChainRate",f"{result.get('anchor_mismatch_chain_rate',0):.4f}"),("MultiAnchorBundleRate",f"{result.get('multi_anchor_bundle_rate',0):.4f}"),("GenericRelationTop1Rate",f"{result.get('generic_relation_top1_rate',0):.4f}"),("QueryAnchorCoverageRate",f"{result.get('query_anchor_coverage_rate',0):.4f}"),("AvgResidualCoverage",f"{result.get('avg_residual_coverage_count',0):.2f}"),("ChainCompleteRate",f"{result.get('chain_complete_rate',0):.4f}"),("AnswerInEvidenceBundles",f"{result.get('answer_in_evidence_bundles',result.get('answer_in_context',0)):.4f}"),("AnswerInRenderedContext",f"{result.get('answer_in_rendered_context',0):.4f}"),("AnswerInPrediction",f"{result.get('answer_in_prediction',0):.4f}"),("IDKRate",f"{result.get('idk_rate',0):.4f}"),("InsufficientRate",f"{result.get('insufficient_rate',0):.4f}")]
    return "\n".join(header+["| metric | value |","|---|---:|"]+[f"| {k} | {v} |" for k,v in rows])+"\n"
