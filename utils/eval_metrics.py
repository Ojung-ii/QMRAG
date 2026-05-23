from __future__ import annotations
from collections import Counter
from typing import Any, Mapping, Sequence
from .text import normalize_answer, token_count

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
    d=row.get("retrieval_diagnostics",{}) or {}
    if d.get("context_tokens") is not None: return int(d.get("context_tokens") or 0)
    return 0
def evaluate_predictions(rows: Sequence[Mapping[str,Any]]) -> dict[str,Any]:
    per=[]
    for r in rows:
        d=r.get("retrieval_diagnostics",{}) or {}; timings=d.get("timings",{}) or {}; pred=str(r.get("prediction",'')); golds=[str(x) for x in r.get("answers",[])]
        ret_ms=1000*float(timings.get("total_retrieval_s",0.0)); gen_ms=1000*float(r.get("generation_latency_s",0.0))
        per.append({"id":r.get("id"),"em":exact_match(pred,golds),"f1":answer_f1(pred,golds),"answer_contains":answer_contains(pred,golds),"support_title_recall":support_title_recall(r),"context_tokens":context_tokens(r),"latency_ms":ret_ms+gen_ms,"retrieval_latency_ms":ret_ms,"generation_latency_ms":gen_ms,"candidate_count":int(d.get("candidate_count") or 0),"seed_count":int(d.get("seed_count") or 0),"bundle_count":int(d.get("bundle_count") or 0),"dense_enabled":bool(d.get("dense_enabled",False))})
    avg=lambda k: sum(float(x[k]) for x in per)/max(1,len(per))
    res={"n":len(rows),"em":avg("em"),"f1":avg("f1"),"answer_contains":avg("answer_contains"),"support_title_recall":avg("support_title_recall"),"context_tokens":avg("context_tokens"),"latency_ms":avg("latency_ms"),"retrieval_latency_ms":avg("retrieval_latency_ms"),"generation_latency_ms":avg("generation_latency_ms"),"candidate_count":avg("candidate_count"),"seed_count":avg("seed_count"),"bundle_count":avg("bundle_count"),"dense_enabled_rate":sum(1.0 if x["dense_enabled"] else 0.0 for x in per)/max(1,len(per)),"per_example":per}
    res["support_recall_per_1k_tokens"]=res["support_title_recall"]/max(1e-9,res["context_tokens"]/1000.0); return res
def summary_markdown(dataset: str, result: Mapping[str,Any]) -> str:
    rows=[("dataset",dataset)]
    if result.get("prompt_profile") is not None:
        rows.append(("prompt_profile",result.get("prompt_profile")))
    if result.get("index_source") is not None:
        rows.append(("index_source",result.get("index_source")))
    if result.get("index_dir") is not None:
        rows.append(("index_dir",result.get("index_dir")))
    rows.extend([("n",result.get("n",0)),("EM",f"{result.get('em',0):.4f}"),("F1",f"{result.get('f1',0):.4f}"),("AnswerContains",f"{result.get('answer_contains',0):.4f}"),("SupportTitleRecall",f"{result.get('support_title_recall',0):.4f}"),("SupportRecallPer1kTokens",f"{result.get('support_recall_per_1k_tokens',0):.4f}"),("AvgContextTokens",f"{result.get('context_tokens',0):.1f}"),("AvgLatencyMs",f"{result.get('latency_ms',0):.1f}"),("AvgRetrievalLatencyMs",f"{result.get('retrieval_latency_ms',0):.1f}"),("AvgGenerationLatencyMs",f"{result.get('generation_latency_ms',0):.1f}"),("DenseEnabledRate",f"{result.get('dense_enabled_rate',0):.2f}")])
    return "\n".join([f"# Evaluation Summary: {dataset}","","| metric | value |","|---|---:|"]+[f"| {k} | {v} |" for k,v in rows])+"\n"
