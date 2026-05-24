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
from scripts.compare_seed_selection_ablation import infer_seed_variant
from utils.eval_metrics import evaluate_predictions
from utils.io_utils import ensure_dir, read_jsonl


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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
        "prompt_profile": str(first.get("prompt_profile", "common_qa")),
        "rendering_profile": str(first.get("rendering_profile", "structured_chain")),
        "prompt_experiment_type": str(first.get("prompt_experiment_type", "")),
        "retrieval_variant": str(first.get("retrieval_variant") or (first.get("retrieval_diagnostics", {}) or {}).get("retrieval_variant") or "full_hetero"),
        "seed_selection_variant": infer_seed_variant([first]),
        "candidate_cap_enabled": bool((first.get("retrieval_diagnostics", {}) or {}).get("candidate_cap_enabled", False)),
        "mtime": path.stat().st_mtime,
    }


def latest_reference_run(output_root: Path, dataset: str) -> Path | None:
    candidates = []
    for path in iter_prediction_files(output_root):
        summary = summarize_path(path)
        if not summary:
            continue
        if summary["dataset"] != dataset:
            continue
        if summary["prompt_profile"] != "common_qa" or summary["rendering_profile"] != "structured_chain":
            continue
        if summary.get("prompt_experiment_type") not in {"", "main_comparison"}:
            continue
        if summary["retrieval_variant"] != "full_hetero" or summary["seed_selection_variant"] != "medoid_current":
            continue
        if summary.get("candidate_cap_enabled"):
            continue
        candidates.append(summary)
    if not candidates:
        return None
    return max(candidates, key=lambda x: (float(x["mtime"]), str(x["path"])))["path"]


def latest_file(output_root: Path, pattern: str) -> Path | None:
    files = list(output_root.glob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: (p.stat().st_mtime, str(p)))


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def reference_section(dataset: str, path: Path) -> str:
    rows = read_jsonl(path)
    result = evaluate_predictions(rows, dataset=dataset, prompt_profile=str(rows[0].get("prompt_profile", "common_qa")))
    timing = load_timing_summary(path)
    headers = [
        "n",
        "F1",
        "answer_in_rendered_context",
        "answer_in_prediction",
        "insufficient_rate",
        "retrieval_ms",
        "candidate_retrieval_ms",
        "seed_selection_ms",
        "num_query_embedding_calls",
        "num_dense_search_calls",
        "num_bm25_search_calls",
        "raw_candidate_count",
        "unique_candidate_count",
        "duplicate_candidate_count",
        "candidate_merge_reduction_rate",
        "num_pairwise_similarity_computations",
        "num_pairwise_similarity_cache_hits",
    ]
    stage = (timing.get("stages", {}) or {}).get("candidate_retrieval", {}) or {}
    values = {
        "n": result.get("n", 0),
        "F1": result.get("f1", 0.0),
        "answer_in_rendered_context": result.get("answer_in_rendered_context", 0.0),
        "answer_in_prediction": result.get("answer_in_prediction", 0.0),
        "insufficient_rate": result.get("insufficient_rate", 0.0),
        "retrieval_ms": result.get("retrieval_latency_ms", 0.0),
        "candidate_retrieval_ms": float(stage.get("mean_ms", 0.0) or 0.0),
        "seed_selection_ms": result.get("seed_selection_ms", 0.0),
        "num_query_embedding_calls": result.get("num_query_embedding_calls", 0.0),
        "num_dense_search_calls": result.get("num_dense_search_calls", 0.0),
        "num_bm25_search_calls": result.get("num_bm25_search_calls", 0.0),
        "raw_candidate_count": result.get("raw_candidate_count", 0.0),
        "unique_candidate_count": result.get("unique_candidate_count", 0.0),
        "duplicate_candidate_count": result.get("duplicate_candidate_count", 0.0),
        "candidate_merge_reduction_rate": result.get("candidate_merge_reduction_rate", 0.0),
        "num_pairwise_similarity_computations": result.get("num_pairwise_similarity_computations", 0.0),
        "num_pairwise_similarity_cache_hits": result.get("num_pairwise_similarity_cache_hits", 0.0),
    }
    lines = [
        f"## {dataset} Reference Runtime",
        "",
        f"- path: {path}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        "| " + " | ".join(fmt(values.get(h)) for h in headers) + " |",
        "",
        "### Top Timing Bottlenecks",
    ]
    for item in top_bottlenecks(timing, k=6):
        lines.append(f"- {item.get('stage')}: total_ms={item.get('total_ms'):.3f}, mean_ms={item.get('mean_ms'):.3f}")
    return "\n".join(lines)


def collect_report_links(output_root: Path, dataset: str) -> list[tuple[str, Path]]:
    patterns = [
        (f"candidate_cap_ablation_{dataset}", f"analysis/**/candidate_cap_ablation_{dataset}.md"),
        (f"seed_selection_ablation_{dataset}", f"analysis/**/seed_selection_ablation_{dataset}.md"),
        ("context_budget_ablation_summary", "analysis/**/context_budget_ablation_summary.md"),
    ]
    out = []
    for label, pattern in patterns:
        path = latest_file(output_root, pattern)
        if path is not None:
            out.append((label, path))
    return out


def build_report(datasets: Iterable[str], output_root: Path) -> str:
    lines = ["# Runtime Optimization Summary", "", f"- generated_at: {datetime.now().isoformat(timespec='seconds')}", ""]
    for dataset in datasets:
        path = latest_reference_run(output_root, dataset)
        if path is None:
            lines.extend([f"## {dataset}", "", "- no latest full_hetero/common_qa/structured_chain reference run found.", ""])
            continue
        lines.append(reference_section(dataset, path))
        lines.extend(["", "### Related Ablation Reports", ""])
        for label, report_path in collect_report_links(output_root, dataset):
            lines.append(f"- {label}: {report_path}")
        lines.append("")
    lines.extend([
        "## Recommendation Notes",
        "",
        "- Keep default accuracy setting at full_hetero + medoid_current until candidate cap and seed simplification preserve F1 at n=100+.",
        "- Treat candidate caps and simplified seed selection as ablation flags, not default changes.",
        "- Use context-budget/top-bundle replay results as separate context cost ablations because they do not change retrieval runtime.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a compact runtime optimization summary from latest QMRAG outputs")
    parser.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wiki"])
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--analysis-dir", default=None)
    args = parser.parse_args()
    output_root = Path(args.output_root)
    analysis_dir = ensure_dir(args.analysis_dir or output_root / "analysis" / now_timestamp())
    text = build_report(args.datasets, output_root)
    out = analysis_dir / "runtime_optimization_summary.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
