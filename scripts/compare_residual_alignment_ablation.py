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
    "residual_lexical",
    "bridge_fullquery",
    "residual_dense_only",
    "residual_hybrid_lex_first",
    "residual_dense_fallback",
    "residual_unified_alignment",
)


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def residual_variant_from_rows(rows: Iterable[Mapping[str, Any]]) -> str:
    for row in rows:
        diag = row.get("retrieval_diagnostics", {}) or {}
        value = diag.get("residual_selection_variant") or row.get("residual_selection_variant")
        if value:
            return str(value)
    return "residual_lexical"


def summarize_path(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            first_line = next((line for line in f if line.strip()), "")
        if not first_line:
            return None
        first = json.loads(first_line)
    except Exception:
        return None
    diag = first.get("retrieval_diagnostics", {}) or {}
    return {
        "path": path,
        "dataset": str(first.get("dataset") or infer_dataset(path, [first])),
        "residual_selection_variant": residual_variant_from_rows([first]),
        "retrieval_variant": str(first.get("retrieval_variant") or diag.get("retrieval_variant") or "full_hetero"),
        "seed_selection_variant": str(first.get("seed_selection_variant") or diag.get("seed_selection_variant") or "global_seed_search"),
        "candidate_cap_enabled": bool(diag.get("candidate_cap_enabled", False)),
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
        if summary["residual_selection_variant"] != variant:
            continue
        if summary["retrieval_variant"] != "full_hetero":
            continue
        if summary["seed_selection_variant"] != "global_seed_search":
            continue
        if summary.get("candidate_cap_enabled"):
            continue
        if summary["prompt_profile"] != "common_qa" or summary["rendering_profile"] != "structured_chain":
            continue
        candidates.append(summary)
    if not candidates:
        return None
    return max(candidates, key=lambda x: (float(x["mtime"]), str(x["path"])))["path"]


def _stage_mean(timing: Mapping[str, Any], stage: str) -> float:
    row = (timing.get("stages", {}) or {}).get(stage, {}) or {}
    return float(row.get("mean_ms", 0.0) or 0.0)


def selected_bridge_map(rows: list[Mapping[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in rows:
        diag = row.get("retrieval_diagnostics", {}) or {}
        value = diag.get("selected_bridge_prop_id")
        if not value:
            for bundle in row.get("evidence_bundles", []) or []:
                for path in bundle.get("evidence_path", []) or []:
                    if isinstance(path, Mapping) and path.get("bridge_prop_id"):
                        value = path.get("bridge_prop_id")
                        break
                if value:
                    break
        out[str(row.get("id"))] = str(value or "")
    return out


def changed_rate(rows: list[Mapping[str, Any]], baseline_rows: list[Mapping[str, Any]] | None) -> float | None:
    if not baseline_rows:
        return None
    base = selected_bridge_map(baseline_rows)
    cur = selected_bridge_map(rows)
    ids = [i for i in cur if i in base]
    if not ids:
        return None
    return sum(1.0 for i in ids if cur.get(i) != base.get(i)) / len(ids)


def summarize_run(path: Path, dataset: str, variant: str, baseline_rows: list[Mapping[str, Any]] | None) -> dict[str, Any]:
    rows = read_jsonl(path)
    result = evaluate_predictions(rows, dataset=dataset, prompt_profile="common_qa")
    timing = load_timing_summary(path)
    return {
        "dataset": dataset,
        "residual_selection_variant": variant,
        "path": str(path),
        "n": result.get("n", 0),
        "EM": result.get("em", 0.0),
        "F1": result.get("f1", 0.0),
        "answer_in_rendered_context": result.get("answer_in_rendered_context", 0.0),
        "answer_in_prediction": result.get("answer_in_prediction", 0.0),
        "insufficient_rate": result.get("insufficient_rate", 0.0),
        "supporting_fact_precision": result.get("supporting_fact_precision", None),
        "supporting_fact_recall": result.get("supporting_fact_recall", result.get("support_title_recall", 0.0)),
        "supporting_fact_f1": result.get("supporting_fact_f1", None),
        "bridge_connected_rate": result.get("bridge_connected_rate", 0.0),
        "answer_slot_aligned_rate": result.get("answer_slot_aligned_rate", 0.0),
        "chain_complete_v2_rate": result.get("chain_complete_v2_rate", 0.0),
        "anchor_connected_chain_complete_rate": result.get("anchor_connected_chain_complete_rate", 0.0),
        "anchor_mismatch_chain_rate": result.get("anchor_mismatch_chain_rate", 0.0),
        "avg_context_tokens": result.get("avg_context_tokens", result.get("context_tokens", 0.0)),
        "avg_input_prompt_tokens": result.get("avg_input_prompt_tokens", 0.0),
        "F1_per_1k_input_prompt_tokens": result.get("F1_per_1k_input_prompt_tokens", 0.0),
        "retrieval_ms": result.get("retrieval_latency_ms", 0.0),
        "generation_ms": result.get("generation_latency_ms", 0.0),
        "total_ms": result.get("latency_ms", 0.0),
        "bridge_prop_selection_ms": result.get("avg_bridge_prop_selection_ms", 0.0),
        "candidate_retrieval_ms": _stage_mean(timing, "candidate_retrieval"),
        "residual_bridge_selection_ms": _stage_mean(timing, "residual_bridge_selection"),
        "residual_dense_used_rate": result.get("residual_dense_used_rate", 0.0),
        "residual_dense_embedding_calls": result.get("avg_residual_dense_embedding_calls", 0.0),
        "residual_dense_similarity_computations": result.get("avg_residual_dense_similarity_computations", 0.0),
        "bridge_prop_candidate_count": result.get("avg_bridge_prop_candidate_count", 0.0),
        "avg_residual_lexical_coverage_count": result.get("avg_residual_coverage_count", 0.0),
        "selected_bridge_prop_changed_rate": changed_rate(rows, baseline_rows),
        "timing_top_bottlenecks": top_bottlenecks(timing),
    }


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def markdown(dataset: str, rows: list[Mapping[str, Any]], missing: list[str]) -> str:
    headers = [
        "dataset",
        "residual_selection_variant",
        "n",
        "EM",
        "F1",
        "delta_EM",
        "delta_F1",
        "delta_answer_in_prediction",
        "delta_insufficient_rate",
        "delta_chain_complete_v2_rate",
        "delta_anchor_mismatch_chain_rate",
        "answer_in_rendered_context",
        "answer_in_prediction",
        "insufficient_rate",
        "bridge_connected_rate",
        "answer_slot_aligned_rate",
        "chain_complete_v2_rate",
        "anchor_connected_chain_complete_rate",
        "anchor_mismatch_chain_rate",
        "avg_context_tokens",
        "avg_input_prompt_tokens",
        "F1_per_1k_input_prompt_tokens",
        "retrieval_ms",
        "generation_ms",
        "total_ms",
        "bridge_prop_selection_ms",
        "residual_dense_used_rate",
        "residual_dense_similarity_computations",
        "selected_bridge_prop_changed_rate",
    ]
    lines = ["# Residual Alignment Ablation", "", f"- dataset: {dataset}", ""]
    if missing:
        lines.extend([f"- missing_variants: {', '.join(missing)}", ""])
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(h)) for h in headers) + " |")
    lines.extend(["", "## Timing Bottlenecks", ""])
    for row in rows:
        lines.append(f"### {row.get('residual_selection_variant')}")
        for item in row.get("timing_top_bottlenecks", []) or []:
            lines.append(f"- {item.get('stage')}: total_ms={item.get('total_ms'):.3f}, mean_ms={item.get('mean_ms'):.3f}")
        lines.append("")
    return "\n".join(lines) + "\n"


def add_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = next((r for r in rows if r.get("residual_selection_variant") == "residual_lexical"), None)
    if not baseline:
        return rows
    for row in rows:
        row["delta_EM"] = float(row.get("EM", 0.0) or 0.0) - float(baseline.get("EM", 0.0) or 0.0)
        row["delta_F1"] = float(row.get("F1", 0.0) or 0.0) - float(baseline.get("F1", 0.0) or 0.0)
        row["delta_answer_in_prediction"] = float(row.get("answer_in_prediction", 0.0) or 0.0) - float(baseline.get("answer_in_prediction", 0.0) or 0.0)
        row["delta_insufficient_rate"] = float(row.get("insufficient_rate", 0.0) or 0.0) - float(baseline.get("insufficient_rate", 0.0) or 0.0)
        row["delta_chain_complete_v2_rate"] = float(row.get("chain_complete_v2_rate", 0.0) or 0.0) - float(baseline.get("chain_complete_v2_rate", 0.0) or 0.0)
        row["delta_anchor_mismatch_chain_rate"] = float(row.get("anchor_mismatch_chain_rate", 0.0) or 0.0) - float(baseline.get("anchor_mismatch_chain_rate", 0.0) or 0.0)
    return rows


def compare_dataset(dataset: str, output_root: Path, analysis_dir: Path) -> dict[str, Any]:
    baseline_path = find_latest_variant_run(output_root, dataset, "residual_lexical")
    baseline_rows = read_jsonl(baseline_path) if baseline_path else None
    rows = []
    missing = []
    for variant in VARIANTS:
        path = find_latest_variant_run(output_root, dataset, variant)
        if path is None:
            missing.append(variant)
            continue
        rows.append(summarize_run(path, dataset, variant, baseline_rows))
    rows = add_deltas(rows)
    result = {"dataset": dataset, "rows": rows, "missing_variants": missing}
    dump_json(result, analysis_dir / f"residual_alignment_ablation_{dataset}.json")
    text = markdown(dataset, rows, missing)
    (analysis_dir / f"residual_alignment_ablation_{dataset}.md").write_text(text, encoding="utf-8")
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
    parser = argparse.ArgumentParser(description="Compare ACE-RAG residual bridge proposition selection variants")
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

