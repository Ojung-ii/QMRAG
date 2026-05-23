#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.eval_metrics import evaluate_predictions


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
        value=row.get("dataset")
        if value:
            return str(value)
    parts=path.parts
    if "outputs" in parts:
        i=parts.index("outputs")
        rest=parts[i+1:]
        if len(rest)>=4 and rest[1]=="eval":
            return rest[0]
        if len(rest)>=3:
            return rest[1]
    return "UNKNOWN"


def infer_prompt(rows: Iterable[Dict[str, Any]]) -> str:
    for row in rows:
        value=row.get("prompt_profile")
        if value:
            return str(value)
    return "UNKNOWN"


def iter_prediction_files(output_root: Path) -> Iterable[Path]:
    yield from sorted(output_root.rglob("predictions.jsonl"))


def summarize(path: Path) -> Dict[str, Any]:
    rows=read_jsonl(path)
    dataset=infer_dataset(path, rows)
    prompt_profile=infer_prompt(rows)
    result=evaluate_predictions(rows, dataset=dataset, prompt_profile=prompt_profile)
    raw_none=sum(1 for row in rows if row.get("raw_prediction") is None)
    denom=max(1,len(rows))
    return {
        "dataset": dataset,
        "prompt_profile": prompt_profile,
        "prompt_experiment_type": result.get("prompt_experiment_type", "unknown"),
        "n": result.get("n", 0),
        "answer_in_evidence_bundles": result.get("answer_in_evidence_bundles", result.get("answer_in_context", 0.0)),
        "answer_in_rendered_context": result.get("answer_in_rendered_context", 0.0),
        "answer_in_prediction": result.get("answer_in_prediction", 0.0),
        "chain_complete_rate": result.get("chain_complete_rate", 0.0),
        "bridge_connected_rate": result.get("bridge_connected_rate", 0.0),
        "answer_slot_aligned_rate": result.get("answer_slot_aligned_rate", 0.0),
        "chain_complete_v2_rate": result.get("chain_complete_v2_rate", 0.0),
        "avg_residual_coverage_count": result.get("avg_residual_coverage_count", 0.0),
        "avg_bridge_title_count": result.get("avg_bridge_title_count", 0.0),
        "avg_bridge_bundle_count": result.get("avg_bridge_bundle_count", 0.0),
        "idk_rate": result.get("idk_rate", 0.0),
        "insufficient_rate": result.get("insufficient_rate", 0.0),
        "raw_none_rate": raw_none/denom,
        "mtime": path.stat().st_mtime,
        "path": str(path),
    }


def latest_by_dataset_prompt(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    latest: Dict[Tuple[str, str], Dict[str, Any]]={}
    for row in rows:
        key=(str(row["dataset"]), str(row["prompt_profile"]))
        if key not in latest or float(row["mtime"]) > float(latest[key]["mtime"]):
            latest[key]=row
    return sorted(latest.values(), key=lambda r: (str(r["dataset"]), str(r["prompt_profile"])))


def markdown_table(rows: List[Dict[str, Any]]) -> str:
    headers=["dataset","prompt_profile","prompt_experiment_type","n","bridge_connected_rate","answer_slot_aligned_rate","chain_complete_v2_rate","avg_residual_coverage_count","chain_complete_rate","avg_bridge_title_count","avg_bridge_bundle_count","answer_in_evidence_bundles","answer_in_rendered_context","answer_in_prediction","idk_rate","insufficient_rate","raw_none_rate","path"]
    lines=["| "+" | ".join(headers)+" |","| "+" | ".join(["---"]*len(headers))+" |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join([
                str(row["dataset"]),
                str(row["prompt_profile"]),
                str(row["prompt_experiment_type"]),
                str(row["n"]),
                f"{float(row['bridge_connected_rate']):.4f}",
                f"{float(row['answer_slot_aligned_rate']):.4f}",
                f"{float(row['chain_complete_v2_rate']):.4f}",
                f"{float(row['avg_residual_coverage_count']):.2f}",
                f"{float(row['chain_complete_rate']):.4f}",
                f"{float(row['avg_bridge_title_count']):.2f}",
                f"{float(row['avg_bridge_bundle_count']):.2f}",
                f"{float(row['answer_in_evidence_bundles']):.4f}",
                f"{float(row['answer_in_rendered_context']):.4f}",
                f"{float(row['answer_in_prediction']):.4f}",
                f"{float(row['idk_rate']):.4f}",
                f"{float(row['insufficient_rate']):.4f}",
                f"{float(row['raw_none_rate']):.4f}",
                str(row["path"]),
            ])
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser=argparse.ArgumentParser(description="Diagnose QMRAG prediction files.")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--include-empty", action="store_true")
    args=parser.parse_args()

    output_root=Path(args.output_root)
    rows=[]
    for path in iter_prediction_files(output_root):
        summary=summarize(path)
        if args.dataset and summary["dataset"] != args.dataset:
            continue
        if not args.include_empty and int(summary.get("n") or 0)==0:
            continue
        rows.append(summary)
    if args.latest:
        rows=latest_by_dataset_prompt(rows)
    else:
        rows=sorted(rows, key=lambda r: (str(r["dataset"]), str(r["prompt_profile"]), str(r["path"])))
    print(markdown_table(rows))


if __name__=="__main__":
    main()
