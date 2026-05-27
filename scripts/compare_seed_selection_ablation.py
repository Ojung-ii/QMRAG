#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_prop_centered_ablation import infer_dataset, iter_prediction_files, load_timing_summary, top_bottlenecks
from utils.eval_metrics import evaluate_predictions
from utils.io_utils import dump_json, ensure_dir, read_jsonl


VARIANTS = (
    "diverse_seed_search",
    "global_seed_search",
    "anchor_first",
    "chain_potential",
)


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def infer_seed_variant(rows: Iterable[Mapping[str, Any]]) -> str:
    for row in rows:
        value = row.get("seed_selection_variant") or (row.get("retrieval_diagnostics", {}) or {}).get("seed_selection_variant")
        if value:
            return str(value)
    return "global_seed_search"


def summarize_path(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            first_line = next((line for line in f if line.strip()), "")
        if not first_line:
            return None
        first = json.loads(first_line)
    except Exception:
        return None
    return {
        "path": path,
        "dataset": str(first.get("dataset") or infer_dataset(path, [first])),
        "seed_selection_variant": infer_seed_variant([first]),
        "candidate_cap_enabled": bool((first.get("retrieval_diagnostics", {}) or {}).get("candidate_cap_enabled", False)),
        "retrieval_variant": str(first.get("retrieval_variant") or (first.get("retrieval_diagnostics", {}) or {}).get("retrieval_variant") or "full_hetero"),
        "prompt_profile": str(first.get("prompt_profile", "common_qa")),
        "rendering_profile": str(first.get("rendering_profile", "structured_chain")),
        "mtime": path.stat().st_mtime,
    }


def find_latest_variant_run(output_root: Path, dataset: str, variant: str) -> Path | None:
    candidates = []
    for path in iter_prediction_files(output_root):
        summary = summarize_path(path)
        if not summary:
            continue
        if summary["dataset"] != dataset:
            continue
        if summary["seed_selection_variant"] != variant:
            continue
        if summary["retrieval_variant"] != "full_hetero":
            continue
        if summary.get("candidate_cap_enabled"):
            continue
        if summary["prompt_profile"] != "common_qa" or summary["rendering_profile"] != "structured_chain":
            continue
        candidates.append(summary)
    if not candidates:
        return None
    return max(candidates, key=lambda x: (float(x["mtime"]), str(x["path"])))["path"]


def summarize_run(path: Path, dataset: str, variant: str) -> dict[str, Any]:
    rows = read_jsonl(path)
    result = evaluate_predictions(rows, dataset=dataset, prompt_profile="common_qa")
    timing = load_timing_summary(path)
    return {
        "dataset": dataset,
        "seed_selection_variant": variant,
        "path": str(path),
        "n": result.get("n", 0),
        "EM": result.get("em", 0.0),
        "F1": result.get("f1", 0.0),
        "answer_in_rendered_context": result.get("answer_in_rendered_context", 0.0),
        "answer_in_prediction": result.get("answer_in_prediction", 0.0),
        "insufficient_rate": result.get("insufficient_rate", 0.0),
        "chain_complete_v2_rate": result.get("chain_complete_v2_rate", 0.0),
        "anchor_connected_chain_complete_rate": result.get("anchor_connected_chain_complete_rate", 0.0),
        "anchor_mismatch_chain_rate": result.get("anchor_mismatch_chain_rate", 0.0),
        "retrieval_ms": result.get("retrieval_latency_ms", 0.0),
        "candidate_retrieval_ms": _stage_mean(timing, "candidate_retrieval"),
        "seed_selection_ms": result.get("seed_selection_ms", 0.0),
        "generation_ms": result.get("generation_latency_ms", 0.0),
        "total_ms": result.get("latency_ms", 0.0),
        "CtxTok": result.get("avg_context_tokens", result.get("context_tokens", 0.0)),
        "InputTok": result.get("avg_input_prompt_tokens", 0.0),
        "selected_seed_count": result.get("selected_seed_count", 0.0),
        "seed_unit_type_distribution": result.get("seed_unit_type_distribution", {}),
        "chain_success_by_seed_type": result.get("chain_success_by_seed_type", {}),
        "num_query_embedding_calls": result.get("num_query_embedding_calls", 0.0),
        "num_dense_search_calls": result.get("num_dense_search_calls", 0.0),
        "num_bm25_search_calls": result.get("num_bm25_search_calls", 0.0),
        "num_title_search_calls": result.get("num_title_search_calls", 0.0),
        "num_chunk_search_calls": result.get("num_chunk_search_calls", 0.0),
        "num_proposition_search_calls": result.get("num_proposition_search_calls", 0.0),
        "raw_candidate_count": result.get("raw_candidate_count", 0.0),
        "unique_candidate_count": result.get("unique_candidate_count", 0.0),
        "num_candidate_score_computations": result.get("num_candidate_score_computations", 0.0),
        "num_bridge_title_lookups": result.get("num_bridge_title_lookups", 0.0),
        "num_bridge_prop_score_computations": result.get("num_bridge_prop_score_computations", 0.0),
        "num_pairwise_similarity_computations": result.get("num_pairwise_similarity_computations", 0.0),
        "num_pairwise_similarity_cache_hits": result.get("num_pairwise_similarity_cache_hits", 0.0),
        "pairwise_matrix_size": result.get("pairwise_matrix_size", 0.0),
        "candidate_merge_reduction_rate": result.get("candidate_merge_reduction_rate", 0.0),
        "timing_top_bottlenecks": top_bottlenecks(timing),
    }


def _stage_mean(timing: Mapping[str, Any], stage: str) -> float:
    row = (timing.get("stages", {}) or {}).get(stage, {}) or {}
    return float(row.get("mean_ms", 0.0) or 0.0)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def markdown(dataset: str, rows: list[Mapping[str, Any]], missing: list[str]) -> str:
    headers = [
        "dataset",
        "seed_selection_variant",
        "n",
        "EM",
        "F1",
        "answer_in_rendered_context",
        "answer_in_prediction",
        "insufficient_rate",
        "chain_complete_v2_rate",
        "anchor_connected_chain_complete_rate",
        "anchor_mismatch_chain_rate",
        "retrieval_ms",
        "candidate_retrieval_ms",
        "seed_selection_ms",
        "total_ms",
        "CtxTok",
        "InputTok",
        "selected_seed_count",
        "num_query_embedding_calls",
        "num_dense_search_calls",
        "num_bm25_search_calls",
        "num_title_search_calls",
        "num_chunk_search_calls",
        "num_proposition_search_calls",
        "raw_candidate_count",
        "unique_candidate_count",
        "num_candidate_score_computations",
        "num_bridge_title_lookups",
        "num_bridge_prop_score_computations",
        "num_pairwise_similarity_computations",
        "num_pairwise_similarity_cache_hits",
        "pairwise_matrix_size",
        "candidate_merge_reduction_rate",
    ]
    lines = ["# Seed Selection Ablation", "", f"- dataset: {dataset}", ""]
    if missing:
        lines.append(f"- missing_variants: {', '.join(missing)}")
        lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(h)) for h in headers) + " |")
    lines.extend(["", "## Timing Bottlenecks", ""])
    for row in rows:
        lines.append(f"### {row.get('seed_selection_variant')}")
        for item in row.get("timing_top_bottlenecks", []) or []:
            lines.append(f"- {item.get('stage')}: total_ms={item.get('total_ms'):.3f}, mean_ms={item.get('mean_ms'):.3f}")
        lines.append("")
    lines.extend(["## Seed Type Distributions", ""])
    for row in rows:
        lines.append(f"- {row.get('seed_selection_variant')}: seeds={json.dumps(row.get('seed_unit_type_distribution',{}),ensure_ascii=False)}, chain_success={json.dumps(row.get('chain_success_by_seed_type',{}),ensure_ascii=False)}")
    return "\n".join(lines) + "\n"


def compare_dataset(dataset: str, output_root: Path, analysis_dir: Path) -> dict[str, Any]:
    rows = []
    missing = []
    for variant in VARIANTS:
        path = find_latest_variant_run(output_root, dataset, variant)
        if path is None:
            missing.append(variant)
            continue
        rows.append(summarize_run(path, dataset, variant))
    result = {"dataset": dataset, "rows": rows, "missing_variants": missing}
    dump_json(result, analysis_dir / f"seed_selection_ablation_{dataset}.json")
    text = markdown(dataset, rows, missing)
    (analysis_dir / f"seed_selection_ablation_{dataset}.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote: {analysis_dir}")
    return result


def latest_datasets(output_root: Path) -> list[str]:
    datasets = set()
    for path in iter_prediction_files(output_root):
        summary = summarize_path(path)
        if summary:
            datasets.add(str(summary["dataset"]))
    return sorted(datasets)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ACE-RAG seed selection ablation variants")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--all-latest", action="store_true")
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--analysis-dir", default=None)
    args = parser.parse_args()
    output_root = Path(args.output_root)
    analysis_dir = ensure_dir(args.analysis_dir or output_root / "analysis" / now_timestamp())
    datasets = latest_datasets(output_root) if args.all_latest else [args.dataset]
    if not datasets or not datasets[0]:
        raise SystemExit("--dataset or --all-latest is required")
    for dataset in datasets:
        compare_dataset(str(dataset), output_root, analysis_dir)


if __name__ == "__main__":
    main()
