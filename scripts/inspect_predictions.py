#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows=[]
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def infer_dataset(path: Path, rows: Iterable[Dict[str, Any]]) -> str:
    for row in rows:
        if row.get("dataset"):
            return str(row["dataset"])
    parts=path.parts
    if "outputs" in parts:
        rest=parts[parts.index("outputs")+1:]
        if len(rest)>=4 and rest[1]=="eval":
            return rest[0]
        if len(rest)>=3:
            return rest[1]
    return "UNKNOWN"


def infer_prompt(rows: Iterable[Dict[str, Any]]) -> str:
    for row in rows:
        if row.get("prompt_profile"):
            return str(row["prompt_profile"])
    return "UNKNOWN"


def evidence_titles(row: Dict[str, Any]) -> List[str]:
    titles=[]
    for bundle in row.get("evidence_bundles",[]) or []:
        if bundle.get("anchor_title"):
            titles.append(str(bundle["anchor_title"]))
        for key in ("propositions","source_chunks"):
            for item in bundle.get(key,[]) or []:
                title=item.get("title")
                if title:
                    titles.append(str(title))
    seen=set(); out=[]
    for title in titles:
        key=title.lower()
        if key not in seen:
            seen.add(key); out.append(title)
    return out


def find_latest(output_root: Path, dataset: str | None, prompt_profile: str | None) -> tuple[Path, List[Dict[str, Any]]]:
    best=None
    for path in output_root.rglob("predictions.jsonl"):
        rows=read_jsonl(path)
        if not rows:
            continue
        ds=infer_dataset(path, rows)
        prompt=infer_prompt(rows)
        if dataset and ds != dataset:
            continue
        if prompt_profile and prompt != prompt_profile:
            continue
        key=path.stat().st_mtime
        if best is None or key > best[0]:
            best=(key,path,rows)
    if best is None:
        raise SystemExit("No matching predictions.jsonl found")
    return best[1], best[2]


def main() -> None:
    parser=argparse.ArgumentParser(description="Inspect latest ACE-RAG prediction rows.")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--prompt-profile", default=None)
    parser.add_argument("--limit", type=int, default=3)
    args=parser.parse_args()

    path, rows=find_latest(Path(args.output_root), args.dataset, args.prompt_profile)
    print(f"path: {path}")
    print(f"dataset: {infer_dataset(path, rows)}")
    print(f"prompt_profile: {infer_prompt(rows)}")
    print(f"prompt_experiment_type: {rows[0].get('prompt_experiment_type','unknown')}")
    print()
    for i,row in enumerate(rows[:max(0,args.limit)]):
        print(f"## Row {i+1}: {row.get('id')}")
        print(f"question: {row.get('question','')}")
        print(f"gold answers: {row.get('answers',[])}")
        print(f"raw_prediction: {row.get('raw_prediction','')}")
        print(f"prediction: {row.get('prediction','')}")
        rd=row.get("retrieval_diagnostics",{}) or {}
        if rd.get("query_anchor_titles") is not None:
            print(f"query_anchor_titles: {rd.get('query_anchor_titles',[])}")
            print(f"query_relation_titles: {rd.get('query_relation_titles',[])}")
        print(f"evidence titles: {', '.join(evidence_titles(row)[:20])}")
        for j,bundle in enumerate((row.get("evidence_bundles",[]) or [])[:3], start=1):
            print(
                f"bundle {j}: anchor={bundle.get('anchor_title')} "
                f"bundle_type={bundle.get('bundle_type')} "
                f"bridge_titles={bundle.get('bridge_titles',[])} "
                f"query_anchor_titles={bundle.get('query_anchor_titles',[])} "
                f"query_relation_titles={bundle.get('query_relation_titles',[])} "
                f"is_query_anchor_bundle={bundle.get('is_query_anchor_bundle')} "
                f"is_relation_title_bundle={bundle.get('is_relation_title_bundle')} "
                f"anchor_connected={bundle.get('anchor_connected')} "
                f"anchor_connected_chain_complete={bundle.get('anchor_connected_chain_complete')} "
                f"anchor_mismatch_chain={bundle.get('anchor_mismatch_chain')} "
                f"bridge_connected={bundle.get('bridge_connected')} "
                f"answer_slot_aligned={bundle.get('answer_slot_aligned')} "
                f"chain_complete_v2={bundle.get('chain_complete_v2')} "
                f"multi_anchor_complete={bundle.get('multi_anchor_complete')} "
                f"residual_coverage_count={bundle.get('residual_coverage_count')} "
                f"is_generic_relation_title={bundle.get('is_generic_relation_title')} "
                f"ordering_group={bundle.get('ordering_group')}"
            )
            if bundle.get("residual_query"):
                print(f"residual_query: {bundle.get('residual_query')}")
            if bundle.get("evidence_path"):
                print(f"evidence_path: {bundle.get('evidence_path')}")
        preview=row.get("rendered_context_preview")
        if preview is None:
            preview=str(row.get("rendered_context",""))[:2000]
        print("rendered_context_preview:")
        print(preview)
        print()


if __name__=="__main__":
    main()
