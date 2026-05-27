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


def infer_rendering(rows: Iterable[Dict[str, Any]]) -> str:
    for row in rows:
        value=row.get("rendering_profile")
        if value:
            return str(value)
    return "structured_chain"


def infer_retrieval_variant(rows: Iterable[Dict[str, Any]]) -> str:
    for row in rows:
        value=row.get("retrieval_variant") or (row.get("retrieval_diagnostics",{}) or {}).get("retrieval_variant")
        if value:
            return str(value)
    return "full_hetero"


def infer_seed_selection_variant(rows: Iterable[Dict[str, Any]]) -> str:
    for row in rows:
        value=row.get("seed_selection_variant") or (row.get("retrieval_diagnostics",{}) or {}).get("seed_selection_variant")
        if value:
            return str(value)
    return "global_seed_search"


def iter_prediction_files(output_root: Path) -> Iterable[Path]:
    yield from sorted(output_root.rglob("predictions.jsonl"))


def first_jsonl_row(path: Path) -> Dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if line:
                    return json.loads(line)
    except Exception:
        return None
    return None


def summarize_for_latest(path: Path) -> Dict[str, Any] | None:
    first=first_jsonl_row(path)
    if not first:
        return None
    return {
        "dataset": infer_dataset(path, [first]),
        "prompt_profile": infer_prompt([first]),
        "rendering_profile": infer_rendering([first]),
        "retrieval_variant": infer_retrieval_variant([first]),
        "seed_selection_variant": infer_seed_selection_variant([first]),
        "mtime": path.stat().st_mtime,
        "path": str(path),
    }


def latest_paths_by_dataset_prompt(rows: List[Dict[str, Any]]) -> List[Path]:
    latest: Dict[Tuple[str, str, str, str, str], Dict[str, Any]]={}
    for row in rows:
        key=(str(row["dataset"]), str(row["prompt_profile"]), str(row["rendering_profile"]), str(row.get("retrieval_variant","full_hetero")), str(row.get("seed_selection_variant","global_seed_search")))
        if key not in latest or float(row["mtime"]) > float(latest[key]["mtime"]):
            latest[key]=row
    return [Path(x["path"]) for x in sorted(latest.values(), key=lambda r: (str(r["dataset"]), str(r["prompt_profile"]), str(r["rendering_profile"]), str(r.get("retrieval_variant","full_hetero")), str(r.get("seed_selection_variant","global_seed_search"))))]


def summarize(path: Path) -> Dict[str, Any]:
    rows=read_jsonl(path)
    dataset=infer_dataset(path, rows)
    prompt_profile=infer_prompt(rows)
    rendering_profile=infer_rendering(rows)
    retrieval_variant=infer_retrieval_variant(rows)
    seed_selection_variant=infer_seed_selection_variant(rows)
    eval_path=path.parent/"eval.json"
    if eval_path.exists():
        with eval_path.open("r", encoding="utf-8") as f:
            result=json.load(f)
    else:
        result=evaluate_predictions(rows, dataset=dataset, prompt_profile=prompt_profile)
    raw_none=sum(1 for row in rows if row.get("raw_prediction") is None)
    denom=max(1,len(rows))
    return {
        "dataset": dataset,
        "prompt_profile": prompt_profile,
        "rendering_profile": rendering_profile,
        "retrieval_variant": retrieval_variant,
        "seed_selection_variant": seed_selection_variant,
        "prompt_experiment_type": result.get("prompt_experiment_type", "unknown"),
        "n": result.get("n", 0),
        "answer_in_evidence_bundles": result.get("answer_in_evidence_bundles", result.get("answer_in_context", 0.0)),
        "answer_in_rendered_context": result.get("answer_in_rendered_context", 0.0),
        "answer_in_prediction": result.get("answer_in_prediction", 0.0),
        "chain_complete_rate": result.get("chain_complete_rate", 0.0),
        "bridge_connected_rate": result.get("bridge_connected_rate", 0.0),
        "answer_slot_aligned_rate": result.get("answer_slot_aligned_rate", 0.0),
        "chain_complete_v2_rate": result.get("chain_complete_v2_rate", 0.0),
        "anchor_connected_chain_complete_rate": result.get("anchor_connected_chain_complete_rate", 0.0),
        "anchor_mismatch_chain_rate": result.get("anchor_mismatch_chain_rate", 0.0),
        "multi_anchor_bundle_rate": result.get("multi_anchor_bundle_rate", 0.0),
        "generic_relation_top1_rate": result.get("generic_relation_top1_rate", 0.0),
        "query_anchor_coverage_rate": result.get("query_anchor_coverage_rate", 0.0),
        "avg_residual_coverage_count": result.get("avg_residual_coverage_count", 0.0),
        "avg_bridge_title_count": result.get("avg_bridge_title_count", 0.0),
        "avg_bridge_bundle_count": result.get("avg_bridge_bundle_count", 0.0),
        "CtxTok": result.get("context_tokens", 0.0),
        "LatencyMs": result.get("latency_ms", 0.0),
        "RetrievalMs": result.get("retrieval_latency_ms", 0.0),
        "SeedSelectionMs": result.get("seed_selection_ms", 0.0),
        "QueryEmbCalls": result.get("num_query_embedding_calls", 0.0),
        "DenseSearchCalls": result.get("num_dense_search_calls", 0.0),
        "BM25SearchCalls": result.get("num_bm25_search_calls", 0.0),
        "TitleSearchCalls": result.get("num_title_search_calls", 0.0),
        "ChunkSearchCalls": result.get("num_chunk_search_calls", 0.0),
        "PropSearchCalls": result.get("num_proposition_search_calls", 0.0),
        "RawCandidates": result.get("raw_candidate_count", 0.0),
        "UniqueCandidates": result.get("unique_candidate_count", 0.0),
        "DuplicateCandidates": result.get("duplicate_candidate_count", 0.0),
        "CandScoreComp": result.get("num_candidate_score_computations", 0.0),
        "CandScoreCacheHits": result.get("num_candidate_score_cache_hits", 0.0),
        "BridgeLookups": result.get("num_bridge_title_lookups", 0.0),
        "BridgeCacheHits": result.get("num_bridge_title_cache_hits", 0.0),
        "BridgePropScoreComp": result.get("num_bridge_prop_score_computations", 0.0),
        "BridgePropScoreCacheHits": result.get("num_bridge_prop_score_cache_hits", 0.0),
        "PairwiseComp": result.get("num_pairwise_similarity_computations", 0.0),
        "PairwiseCacheHits": result.get("num_pairwise_similarity_cache_hits", 0.0),
        "PairwiseMatrixSize": result.get("pairwise_matrix_size", 0.0),
        "CandidateMergeReduction": result.get("candidate_merge_reduction_rate", 0.0),
        "idk_rate": result.get("idk_rate", 0.0),
        "insufficient_rate": result.get("insufficient_rate", 0.0),
        "raw_none_rate": raw_none/denom,
        "seed_title_rate": result.get("seed_title_rate", 0.0),
        "seed_chunk_rate": result.get("seed_chunk_rate", 0.0),
        "seed_proposition_rate": result.get("seed_proposition_rate", 0.0),
        "chain_from_title_rate": result.get("chain_from_title_rate", 0.0),
        "chain_from_chunk_rate": result.get("chain_from_chunk_rate", 0.0),
        "chain_from_proposition_rate": result.get("chain_from_proposition_rate", 0.0),
        "mtime": path.stat().st_mtime,
        "path": str(path),
    }


def latest_by_dataset_prompt(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    latest: Dict[Tuple[str, str, str, str, str], Dict[str, Any]]={}
    for row in rows:
        key=(str(row["dataset"]), str(row["prompt_profile"]), str(row["rendering_profile"]), str(row.get("retrieval_variant","full_hetero")), str(row.get("seed_selection_variant","global_seed_search")))
        if key not in latest or float(row["mtime"]) > float(latest[key]["mtime"]):
            latest[key]=row
    return sorted(latest.values(), key=lambda r: (str(r["dataset"]), str(r["prompt_profile"]), str(r["rendering_profile"]), str(r.get("retrieval_variant","full_hetero")), str(r.get("seed_selection_variant","global_seed_search"))))


def markdown_table(rows: List[Dict[str, Any]]) -> str:
    headers=["dataset","prompt_profile","rendering_profile","retrieval_variant","seed_selection_variant","prompt_experiment_type","n","seed_title_rate","seed_chunk_rate","seed_proposition_rate","chain_from_title_rate","chain_from_chunk_rate","chain_from_proposition_rate","bridge_connected_rate","answer_slot_aligned_rate","chain_complete_v2_rate","anchor_connected_chain_complete_rate","anchor_mismatch_chain_rate","multi_anchor_bundle_rate","generic_relation_top1_rate","query_anchor_coverage_rate","avg_residual_coverage_count","chain_complete_rate","avg_bridge_title_count","avg_bridge_bundle_count","answer_in_evidence_bundles","answer_in_rendered_context","answer_in_prediction","idk_rate","insufficient_rate","CtxTok","LatencyMs","RetrievalMs","SeedSelectionMs","QueryEmbCalls","DenseSearchCalls","BM25SearchCalls","TitleSearchCalls","ChunkSearchCalls","PropSearchCalls","RawCandidates","UniqueCandidates","DuplicateCandidates","CandScoreComp","CandScoreCacheHits","BridgeLookups","BridgeCacheHits","BridgePropScoreComp","BridgePropScoreCacheHits","PairwiseComp","PairwiseCacheHits","PairwiseMatrixSize","CandidateMergeReduction","raw_none_rate","path"]
    lines=["| "+" | ".join(headers)+" |","| "+" | ".join(["---"]*len(headers))+" |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join([
                str(row["dataset"]),
                str(row["prompt_profile"]),
                str(row["rendering_profile"]),
                str(row.get("retrieval_variant","full_hetero")),
                str(row.get("seed_selection_variant","global_seed_search")),
                str(row["prompt_experiment_type"]),
                str(row["n"]),
                f"{float(row['seed_title_rate']):.4f}",
                f"{float(row['seed_chunk_rate']):.4f}",
                f"{float(row['seed_proposition_rate']):.4f}",
                f"{float(row['chain_from_title_rate']):.4f}",
                f"{float(row['chain_from_chunk_rate']):.4f}",
                f"{float(row['chain_from_proposition_rate']):.4f}",
                f"{float(row['bridge_connected_rate']):.4f}",
                f"{float(row['answer_slot_aligned_rate']):.4f}",
                f"{float(row['chain_complete_v2_rate']):.4f}",
                f"{float(row['anchor_connected_chain_complete_rate']):.4f}",
                f"{float(row['anchor_mismatch_chain_rate']):.4f}",
                f"{float(row['multi_anchor_bundle_rate']):.4f}",
                f"{float(row['generic_relation_top1_rate']):.4f}",
                f"{float(row['query_anchor_coverage_rate']):.4f}",
                f"{float(row['avg_residual_coverage_count']):.2f}",
                f"{float(row['chain_complete_rate']):.4f}",
                f"{float(row['avg_bridge_title_count']):.2f}",
                f"{float(row['avg_bridge_bundle_count']):.2f}",
                f"{float(row['answer_in_evidence_bundles']):.4f}",
                f"{float(row['answer_in_rendered_context']):.4f}",
                f"{float(row['answer_in_prediction']):.4f}",
                f"{float(row['idk_rate']):.4f}",
                f"{float(row['insufficient_rate']):.4f}",
                f"{float(row['CtxTok']):.1f}",
                f"{float(row['LatencyMs']):.1f}",
                f"{float(row['RetrievalMs']):.1f}",
                f"{float(row['SeedSelectionMs']):.1f}",
                f"{float(row['QueryEmbCalls']):.2f}",
                f"{float(row['DenseSearchCalls']):.2f}",
                f"{float(row['BM25SearchCalls']):.2f}",
                f"{float(row['TitleSearchCalls']):.2f}",
                f"{float(row['ChunkSearchCalls']):.2f}",
                f"{float(row['PropSearchCalls']):.2f}",
                f"{float(row['RawCandidates']):.2f}",
                f"{float(row['UniqueCandidates']):.2f}",
                f"{float(row['DuplicateCandidates']):.2f}",
                f"{float(row['CandScoreComp']):.2f}",
                f"{float(row['CandScoreCacheHits']):.2f}",
                f"{float(row['BridgeLookups']):.2f}",
                f"{float(row['BridgeCacheHits']):.2f}",
                f"{float(row['BridgePropScoreComp']):.2f}",
                f"{float(row['BridgePropScoreCacheHits']):.2f}",
                f"{float(row['PairwiseComp']):.2f}",
                f"{float(row['PairwiseCacheHits']):.2f}",
                f"{float(row['PairwiseMatrixSize']):.2f}",
                f"{float(row['CandidateMergeReduction']):.4f}",
                f"{float(row['raw_none_rate']):.4f}",
                str(row["path"]),
            ])
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser=argparse.ArgumentParser(description="Diagnose ACE-RAG prediction files.")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--include-empty", action="store_true")
    args=parser.parse_args()

    output_root=Path(args.output_root)
    if args.latest:
        latest_infos=[]
        for path in iter_prediction_files(output_root):
            info=summarize_for_latest(path)
            if not info:
                continue
            if args.dataset and info["dataset"] != args.dataset:
                continue
            latest_infos.append(info)
        rows=[summarize(path) for path in latest_paths_by_dataset_prompt(latest_infos)]
        if not args.include_empty:
            rows=[row for row in rows if int(row.get("n") or 0)!=0]
    else:
        rows=[]
        for path in iter_prediction_files(output_root):
            summary=summarize(path)
            if args.dataset and summary["dataset"] != args.dataset:
                continue
            if not args.include_empty and int(summary.get("n") or 0)==0:
                continue
            rows.append(summary)
        rows=sorted(rows, key=lambda r: (str(r["dataset"]), str(r["prompt_profile"]), str(r["rendering_profile"]), str(r["path"])))
    print(markdown_table(rows))


if __name__=="__main__":
    main()
