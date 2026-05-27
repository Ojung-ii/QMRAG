#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_failures import infer_dataset, infer_prompt, infer_rendering
from utils.eval_metrics import answer_contains, evaluate_predictions, exact_match
from utils.generation import is_insufficient_prediction
from utils.io_utils import dump_json, ensure_dir, read_jsonl


TARGET_PROMPTS = {"common_qa", "strict_short_qa", "ace_rag_compact_chain_qa", "ace_rag_compact_chain_light"}
TARGET_COMPACTIONS = {
    "none",
    "chain_schema_k2",
    "chain_schema_k3",
    "chain_schema_k5",
    "chain_schema_plus1_k2",
    "chain_schema_plus1_k3",
    "top3_schema_dedup",
}


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def iter_prediction_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.rglob("predictions.jsonl"))

def first_jsonl_row(path: Path) -> dict[str,Any] | None:
    try:
        with open(path,"r",encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    return json.loads(line)
    except Exception:
        return None
    return None

def source_preference(first: Mapping[str,Any]) -> int:
    ablation=str(first.get("ablation_variant") or "")
    residual=str((first.get("retrieval_diagnostics",{}) or {}).get("residual_selection_variant") or first.get("residual_selection_variant") or "")
    if ablation in {"", "core_ace_rag_mainline"} and residual in {"", "residual_lexical"}:
        return 2
    if ablation in {"", "core_ace_rag_mainline"}:
        return 1
    return 0


def row_compaction(row: Mapping[str,Any]) -> str:
    return str(row.get("compaction_profile") or "none")


def has_truncation(row: Mapping[str,Any]) -> bool:
    return bool(row.get("context_truncation_enabled") or row.get("top_bundles") is not None or row.get("context_token_budget") is not None)


def summarize_file(path: Path) -> dict[str,Any] | None:
    first=first_jsonl_row(path)
    if not first:
        return None
    return {
        "path":path,
        "dataset":str(first.get("dataset") or infer_dataset(path,[first])),
        "prompt_profile":str(first.get("prompt_profile") or "UNKNOWN"),
        "rendering_profile":str(first.get("rendering_profile") or "structured_chain"),
        "compaction_profile":row_compaction(first),
        "n":None,
        "mtime":path.stat().st_mtime,
        "source_preference":source_preference(first),
        "context_truncation":has_truncation(first),
        "context_compaction":bool(first.get("context_compaction_enabled") or row_compaction(first)!="none"),
    }


def find_latest_full(root: Path, dataset: str, prompt_profile: str) -> Path | None:
    candidates=[]
    for path in iter_prediction_files(root):
        try:
            rel=path.relative_to(root)
            if rel.parts and rel.parts[0]=="replay":
                continue
        except Exception:
            pass
        info=summarize_file(path)
        if not info:
            continue
        if info["dataset"]!=dataset or info["prompt_profile"]!=prompt_profile:
            continue
        if info["rendering_profile"]!="structured_chain" or info["compaction_profile"]!="none" or info["context_truncation"]:
            continue
        candidates.append(info)
    return max(candidates,key=lambda x:(int(x.get("source_preference",0)),float(x["mtime"]),str(x["path"])))["path"] if candidates else None


def find_latest_runs(root: Path, dataset: str) -> list[Path]:
    latest: dict[tuple[str,str],dict[str,Any]]={}
    for path in iter_prediction_files(root):
        info=summarize_file(path)
        if not info or info["dataset"]!=dataset:
            continue
        prompt=str(info["prompt_profile"])
        comp=str(info["compaction_profile"])
        if comp not in TARGET_COMPACTIONS:
            continue
        if comp=="none" and prompt!="strict_short_qa":
            continue
        if prompt not in TARGET_PROMPTS:
            continue
        key=(prompt,comp)
        if key not in latest or float(info["mtime"])>float(latest[key]["mtime"]):
            latest[key]=info
    return [x["path"] for x in sorted(latest.values(), key=lambda y:(str(y["prompt_profile"]),str(y["compaction_profile"])))]


def correct(row: Mapping[str,Any]) -> bool:
    answers=[str(x) for x in row.get("answers",[]) if str(x).strip()]
    raw=str(row.get("raw_prediction",row.get("prediction","")) or "")
    return bool(exact_match(raw,answers) or answer_contains(raw,answers))


def overlap_pairs(left_rows: Sequence[Mapping[str,Any]], right_rows: Sequence[Mapping[str,Any]]) -> list[tuple[Mapping[str,Any],Mapping[str,Any]]]:
    left={str(row.get("id")):row for row in left_rows}
    right={str(row.get("id")):row for row in right_rows}
    return [(left[qid],right[qid]) for qid in left if qid in right]


def compare_counts(left_rows: Sequence[Mapping[str,Any]], right_rows: Sequence[Mapping[str,Any]]) -> dict[str,int]:
    pairs=overlap_pairs(left_rows,right_rows)
    fixed=broken=both_correct=both_wrong=0
    for left,right in pairs:
        lc=correct(left); rc=correct(right)
        fixed+=int((not lc) and rc)
        broken+=int(lc and (not rc))
        both_correct+=int(lc and rc)
        both_wrong+=int((not lc) and (not rc))
    return {
        "fixed_by_right":fixed,
        "broken_by_right":broken,
        "both_correct":both_correct,
        "both_wrong":both_wrong,
    }


def eval_for_path(path: Path, dataset: str) -> tuple[list[dict[str,Any]],dict[str,Any]]:
    rows=read_jsonl(path)
    return rows,evaluate_predictions(rows,dataset=dataset,prompt_profile=infer_prompt(rows))


def avg_bool(rows: Sequence[Mapping[str,Any]], key: str, default: bool = False) -> float:
    return sum(1.0 if row.get(key,default) else 0.0 for row in rows)/max(1,len(rows))


def build_summary(dataset: str, paths: Sequence[Path], full_common: Path | None, full_bundle: Path | None) -> dict[str,Any]:
    common_rows, common_eval = eval_for_path(full_common,dataset) if full_common else ([],{})
    bundle_rows, bundle_eval = eval_for_path(full_bundle,dataset) if full_bundle else ([],{})
    runs=[]
    for path in paths:
        rows,ev=eval_for_path(path,dataset)
        first=rows[0] if rows else {}
        counts=compare_counts(common_rows,rows) if common_rows else {}
        f1=float(ev.get("f1",0.0) or 0.0)
        common_f1=float(common_eval.get("f1",0.0) or 0.0)
        bundle_f1=float(bundle_eval.get("f1",0.0) or 0.0)
        runs.append({
            "dataset":dataset,
            "path":str(path),
            "prompt_profile":infer_prompt(rows),
            "compaction_profile":row_compaction(first),
            "prompt_experiment_type":first.get("prompt_experiment_type",""),
            "n":len(rows),
            "EM":ev.get("em",0.0),
            "F1":f1,
            "delta_F1_vs_full_common":f1-common_f1 if common_eval else None,
            "delta_F1_vs_full_bundle":f1-bundle_f1 if bundle_eval else None,
            "answer_in_prediction":ev.get("answer_in_prediction",0.0),
            "answer_in_rendered_context":ev.get("answer_in_rendered_context",0.0),
            "insufficient_rate":ev.get("insufficient_rate",0.0),
            "CtxTok":ev.get("avg_rendered_context_tokens",ev.get("context_tokens",0.0)),
            "InputTok":ev.get("avg_input_prompt_tokens",0.0),
            "TotalTok":ev.get("avg_total_llm_tokens",0.0),
            "token_reduction_rate":ev.get("token_reduction_rate",0.0),
            "F1_per_1k_input_tokens":ev.get("F1_per_1k_input_prompt_tokens",0.0),
            "rendered_chain_count":ev.get("avg_rendered_chain_count",0.0),
            "rendered_sentence_count":ev.get("avg_rendered_sentence_count",0.0),
            "support_sentence_count":ev.get("avg_support_sentence_count",0.0),
            "fallback_rate":ev.get("fallback_rate",0.0),
            "fixed_by_right":counts.get("fixed_by_right",0),
            "broken_by_right":counts.get("broken_by_right",0),
            "both_correct":counts.get("both_correct",0),
            "both_wrong":counts.get("both_wrong",0),
            "evidence_bundles_hash_match_rate":avg_bool(rows,"evidence_bundles_hash_match",True),
            "rendered_context_hash_match_rate":avg_bool(rows,"rendered_context_hash_match",False),
            "insufficient_count":sum(1 for row in rows if is_insufficient_prediction(row.get("raw_prediction",row.get("prediction","")))),
        })
    return {
        "dataset":dataset,
        "full_common_path":str(full_common) if full_common else None,
        "full_bundle_path":str(full_bundle) if full_bundle else None,
        "full_common":common_eval,
        "full_bundle":bundle_eval,
        "runs":runs,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value,float):
        return f"{value:.4f}"
    return str(value)


def markdown(summary: Mapping[str,Any]) -> str:
    lines=[
        "# Compact Chain Prompt Comparison",
        "",
        f"- dataset: {summary.get('dataset')}",
        f"- full_common: {summary.get('full_common_path')}",
        f"- full_bundle: {summary.get('full_bundle_path')}",
        "",
        "| dataset | mode | compaction_profile | prompt | n | EM | F1 | dF1_common | dF1_bundle | AnsPred | AnsCtx | Insuff | InputTok | TokenDown | F1/1kInput | ChainCnt | SentCnt | SupportCnt | Fallback | Fixed | Broken | EBHash | CtxHash |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("runs",[]):
        lines.append(
            "| {dataset} | {mode} | {comp} | {prompt} | {n} | {em} | {f1} | {dc} | {db} | {ap} | {ac} | {ins} | {inp} | {tokdn} | {f1tok} | {chain} | {sent} | {supp} | {fallback} | {fixed} | {broken} | {eb} | {ctx} |".format(
                dataset=row["dataset"],
                mode=row.get("prompt_experiment_type") or "-",
                comp=row["compaction_profile"],
                prompt=row["prompt_profile"],
                n=row["n"],
                em=fmt(row["EM"]),
                f1=fmt(row["F1"]),
                dc=fmt(row["delta_F1_vs_full_common"]),
                db=fmt(row["delta_F1_vs_full_bundle"]),
                ap=fmt(row["answer_in_prediction"]),
                ac=fmt(row["answer_in_rendered_context"]),
                ins=fmt(row["insufficient_rate"]),
                inp=fmt(row["InputTok"]),
                tokdn=fmt(row["token_reduction_rate"]),
                f1tok=fmt(row["F1_per_1k_input_tokens"]),
                chain=fmt(row["rendered_chain_count"]),
                sent=fmt(row["rendered_sentence_count"]),
                supp=fmt(row["support_sentence_count"]),
                fallback=fmt(row["fallback_rate"]),
                fixed=row["fixed_by_right"],
                broken=row["broken_by_right"],
                eb=fmt(row["evidence_bundles_hash_match_rate"]),
                ctx=fmt(row["rendered_context_hash_match_rate"]),
            )
        )
    return "\n".join(lines)+"\n"


def main() -> None:
    parser=argparse.ArgumentParser(description="Compare full, strict-short, and compact chain schema replay runs.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--all-latest", action="store_true")
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--analysis-root", default="outputs/analysis")
    parser.add_argument("--analysis-dir", default=None)
    args=parser.parse_args()
    datasets=["hotpotqa","2wiki","popqa","musique"] if args.all_latest else [args.dataset]
    if not datasets or not datasets[0]:
        raise SystemExit("--dataset or --all-latest is required")
    analysis_dir=Path(args.analysis_dir) if args.analysis_dir else Path(args.analysis_root)/now_timestamp()
    ensure_dir(analysis_dir)
    all_summaries=[]
    for dataset in datasets:
        root=Path(args.output_root)
        full_common=find_latest_full(root,dataset,"common_qa")
        full_bundle=find_latest_full(root,dataset,"ace_rag_bundle_qa")
        paths=find_latest_runs(root,dataset)
        summary=build_summary(dataset,paths,full_common,full_bundle)
        all_summaries.append(summary)
        stem=f"compact_chain_prompt_compare_{dataset}"
        dump_json(summary,analysis_dir/f"{stem}.json")
        (analysis_dir/f"{stem}.md").write_text(markdown(summary),encoding="utf-8")
        print(markdown(summary))
    dump_json({"summaries":all_summaries},analysis_dir/"compact_chain_prompt_compare_summary.json")
    (analysis_dir/"compact_chain_prompt_compare_summary.md").write_text("\n\n".join(markdown(x) for x in all_summaries),encoding="utf-8")
    print(f"wrote: {analysis_dir}")


if __name__=="__main__":
    main()
