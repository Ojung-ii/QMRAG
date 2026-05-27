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


DATASETS = ("hotpotqa", "2wiki", "popqa", "musique")
CORE_VARIANTS = (
    "core_ace_rag_mainline",
    "core_no_bridge",
    "core_bridge_fullquery",
    "core_residual_unified_alignment",
    "core_no_anchor_ordering",
    "core_no_multi_anchor",
)
DIAGNOSTIC_VARIANTS = (
    "core_residual_dense_fallback",
    "core_residual_hybrid_lex_first",
    "core_residual_dense_only",
)
ALL_VARIANTS = CORE_VARIANTS + DIAGNOSTIC_VARIANTS


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def first_row(path: Path) -> Mapping[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    return json.loads(line)
    except Exception:
        return None
    return None


def ablation_variant(row: Mapping[str, Any], path: Path) -> str:
    diag = row.get("retrieval_diagnostics", {}) or {}
    value = row.get("ablation_variant") or diag.get("ablation_variant")
    if value:
        return str(value)
    text = str(path)
    for variant in ALL_VARIANTS:
        if variant in text:
            return variant
    return ""


def summarize_path(path: Path) -> dict[str, Any] | None:
    row = first_row(path)
    if not row:
        return None
    diag = row.get("retrieval_diagnostics", {}) or {}
    return {
        "path": path,
        "dataset": str(row.get("dataset") or infer_dataset(path, [row])),
        "ablation_variant": ablation_variant(row, path),
        "retrieval_variant": str(row.get("retrieval_variant") or diag.get("retrieval_variant") or "full_hetero"),
        "seed_selection_variant": str(row.get("seed_selection_variant") or diag.get("seed_selection_variant") or "global_seed_search"),
        "prompt_profile": str(row.get("prompt_profile", "common_qa")),
        "rendering_profile": str(row.get("rendering_profile", "structured_chain")),
        "candidate_cap_enabled": bool(diag.get("candidate_cap_enabled", False)),
        "top_bundles": row.get("top_bundles"),
        "context_token_budget": row.get("context_token_budget"),
        "compaction_profile": row.get("compaction_profile", "none"),
        "mtime": path.stat().st_mtime,
    }


def find_latest_variant_run(output_root: Path, dataset: str, variant: str) -> Path | None:
    candidates = []
    for path in iter_prediction_files(output_root):
        info = summarize_path(path)
        if not info:
            continue
        if info["dataset"] != dataset or info["ablation_variant"] != variant:
            continue
        if info["retrieval_variant"] != "full_hetero":
            continue
        if info["seed_selection_variant"] != "global_seed_search":
            continue
        if info["prompt_profile"] != "common_qa" or info["rendering_profile"] != "structured_chain":
            continue
        if info.get("candidate_cap_enabled"):
            continue
        if info.get("top_bundles") not in (None, "", 0):
            continue
        if info.get("context_token_budget") not in (None, "", 0):
            continue
        if str(info.get("compaction_profile") or "none") != "none":
            continue
        candidates.append(info)
    if not candidates:
        return None
    return max(candidates, key=lambda x: (float(x["mtime"]), str(x["path"])))["path"]


def _stage_mean(timing: Mapping[str, Any], stage: str) -> float:
    row = (timing.get("stages", {}) or {}).get(stage, {}) or {}
    return float(row.get("mean_ms", 0.0) or 0.0)


def summarize_run(path: Path, dataset: str, variant: str) -> dict[str, Any]:
    rows = read_jsonl(path)
    result = evaluate_predictions(rows, dataset=dataset, prompt_profile="common_qa")
    timing = load_timing_summary(path)
    return {
        "dataset": dataset,
        "ablation_variant": variant,
        "path": str(path),
        "n": result.get("n", 0),
        "EM": result.get("em", 0.0),
        "F1": result.get("f1", 0.0),
        "Recall@5": result.get("Recall@5", result.get("support_title_recall", 0.0)),
        "answer_in_rendered_context": result.get("answer_in_rendered_context", 0.0),
        "answer_in_prediction": result.get("answer_in_prediction", 0.0),
        "insufficient_rate": result.get("insufficient_rate", 0.0),
        "supporting_fact_precision": result.get("supporting_fact_precision"),
        "supporting_fact_recall": result.get("supporting_fact_recall", result.get("support_title_recall", 0.0)),
        "supporting_fact_f1": result.get("supporting_fact_f1"),
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
        "bridge_enabled": result.get("bridge_enabled"),
        "anchor_ordering_enabled": result.get("anchor_ordering_enabled"),
        "multi_anchor_enabled": result.get("multi_anchor_enabled"),
        "residual_selection_variant": result.get("residual_selection_variant", "residual_lexical"),
        "residual_dense_used_rate": result.get("residual_dense_used_rate", 0.0),
        "residual_dense_similarity_computations": result.get("avg_residual_dense_similarity_computations", 0.0),
        "bridge_prop_selection_ms": result.get("avg_bridge_prop_selection_ms", 0.0),
        "candidate_retrieval_ms": _stage_mean(timing, "candidate_retrieval"),
        "residual_bridge_selection_ms": _stage_mean(timing, "residual_bridge_selection"),
        "timing_top_bottlenecks": top_bottlenecks(timing),
    }


def add_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_dataset: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row["ablation_variant"] == "core_ace_rag_mainline":
            by_dataset[row["dataset"]] = row
    for row in rows:
        base = by_dataset.get(row["dataset"])
        if not base:
            continue
        for src, dst in (
            ("EM", "delta_EM"),
            ("F1", "delta_F1"),
            ("answer_in_prediction", "delta_answer_in_prediction"),
            ("insufficient_rate", "delta_insufficient_rate"),
            ("answer_in_rendered_context", "delta_answer_in_rendered_context"),
            ("chain_complete_v2_rate", "delta_chain_complete_v2_rate"),
            ("anchor_connected_chain_complete_rate", "delta_anchor_connected_chain_complete_rate"),
            ("anchor_mismatch_chain_rate", "delta_anchor_mismatch_chain_rate"),
            ("retrieval_ms", "delta_retrieval_ms"),
            ("total_ms", "delta_total_ms"),
        ):
            row[dst] = float(row.get(src, 0.0) or 0.0) - float(base.get(src, 0.0) or 0.0)
    return rows


def average_deltas(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for variant in ALL_VARIANTS:
        vals = [r for r in rows if r.get("ablation_variant") == variant and "delta_F1" in r]
        if not vals:
            continue
        avg = lambda key: sum(float(r.get(key, 0.0) or 0.0) for r in vals) / max(1, len(vals))
        out.append({
            "ablation_variant": variant,
            "datasets": len(vals),
            "avg_delta_F1": avg("delta_F1"),
            "avg_delta_answer_in_prediction": avg("delta_answer_in_prediction"),
            "avg_delta_insufficient_rate": avg("delta_insufficient_rate"),
            "avg_delta_chain_complete_v2_rate": avg("delta_chain_complete_v2_rate"),
            "avg_delta_anchor_mismatch_chain_rate": avg("delta_anchor_mismatch_chain_rate"),
            "avg_delta_retrieval_ms": avg("delta_retrieval_ms"),
            "avg_delta_total_ms": avg("delta_total_ms"),
        })
    return out


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def table(headers: list[str], rows: list[Mapping[str, Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(h)) for h in headers) + " |")
    return lines


def interpretation(rows: list[Mapping[str, Any]]) -> list[str]:
    avg = {r["ablation_variant"]: r for r in average_deltas(rows)}
    labels = {
        "core_no_bridge": "Mention bridge",
        "core_bridge_fullquery": "Residual answer-slot query",
        "core_residual_unified_alignment": "Residual unified alignment",
        "core_no_anchor_ordering": "Anchor-aware ordering",
        "core_no_multi_anchor": "Multi-anchor grouping",
    }
    lines = ["## Core Element Interpretation", ""]
    for variant, label in labels.items():
        row = avg.get(variant)
        if not row:
            lines.append(f"- {label}: missing result for `{variant}`.")
            continue
        df1 = float(row.get("avg_delta_F1", 0.0) or 0.0)
        dins = float(row.get("avg_delta_insufficient_rate", 0.0) or 0.0)
        if variant.startswith("core_no_") or variant == "core_bridge_fullquery":
            if df1 < -0.01:
                verdict = "supports keeping this component"
            elif df1 > 0.005:
                verdict = "suggests the removed/replaced component may be unnecessary"
            else:
                verdict = "shows a small or neutral effect"
        else:
            if df1 > 0.005:
                verdict = "is a candidate mainline replacement"
            elif df1 < -0.005:
                verdict = "does not beat the lexical baseline"
            else:
                verdict = "is roughly tied with the lexical baseline"
        lines.append(f"- {label}: avg ΔF1={df1:.4f}, avg ΔInsufficient={dins:.4f}; {verdict}.")
    return lines


def markdown(rows: list[Mapping[str, Any]], missing: dict[str, list[str]]) -> str:
    lines = ["# ACE-RAG Four-Dataset Core Ablation", ""]
    if missing:
        lines.append("## Missing Variants")
        lines.append("")
        for ds, variants in missing.items():
            if variants:
                lines.append(f"- {ds}: {', '.join(variants)}")
        lines.append("")
    dataset_headers = [
        "dataset",
        "ablation_variant",
        "n",
        "EM",
        "F1",
        "delta_F1",
        "answer_in_rendered_context",
        "answer_in_prediction",
        "delta_answer_in_prediction",
        "insufficient_rate",
        "delta_insufficient_rate",
        "Recall@5",
        "supporting_fact_precision",
        "supporting_fact_recall",
        "supporting_fact_f1",
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
        "residual_dense_used_rate",
        "residual_dense_similarity_computations",
    ]
    for dataset in DATASETS:
        ds_rows = [r for r in rows if r.get("dataset") == dataset]
        if not ds_rows:
            continue
        lines.extend([f"## {dataset}", ""])
        lines.extend(table(dataset_headers, ds_rows))
        lines.append("")
    avg_rows = average_deltas(rows)
    lines.extend(["## Average Delta Vs Mainline", ""])
    lines.extend(table([
        "ablation_variant",
        "datasets",
        "avg_delta_F1",
        "avg_delta_answer_in_prediction",
        "avg_delta_insufficient_rate",
        "avg_delta_chain_complete_v2_rate",
        "avg_delta_anchor_mismatch_chain_rate",
        "avg_delta_retrieval_ms",
        "avg_delta_total_ms",
    ], avg_rows))
    lines.append("")
    lines.extend(interpretation(rows))
    lines.append("")
    lines.extend(["## Source Paths", ""])
    for row in rows:
        lines.append(f"- {row.get('dataset')} / {row.get('ablation_variant')}: `{row.get('path')}`")
    return "\n".join(lines) + "\n"


def collect(output_root: Path, include_diagnostic: bool) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    variants = list(CORE_VARIANTS)
    if include_diagnostic:
        variants.extend(DIAGNOSTIC_VARIANTS)
    rows = []
    missing: dict[str, list[str]] = {}
    for dataset in DATASETS:
        ds_missing = []
        for variant in variants:
            path = find_latest_variant_run(output_root, dataset, variant)
            if path is None:
                ds_missing.append(variant)
                continue
            rows.append(summarize_run(path, dataset, variant))
        if ds_missing:
            missing[dataset] = ds_missing
    return add_deltas(rows), missing


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare four-dataset ACE-RAG core leave-one-out ablations.")
    ap.add_argument("--output-root", default="outputs")
    ap.add_argument("--analysis-dir", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--latest", action="store_true", help="Use latest matching run for each dataset/variant.")
    ap.add_argument("--include-diagnostic", action="store_true")
    args = ap.parse_args()

    analysis_dir = ensure_dir(Path(args.analysis_dir) if args.analysis_dir else Path("outputs") / "analysis" / f"{now_timestamp()}_core_ablation_4ds")
    rows, missing = collect(Path(args.output_root), include_diagnostic=args.include_diagnostic)
    result = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "rows": rows,
        "missing_variants": missing,
        "average_deltas": average_deltas(rows),
    }
    dump_json(result, analysis_dir / "core_ablation_4ds_summary.json")
    text = markdown(rows, missing)
    summary_path = Path(args.output) if args.output else analysis_dir / "core_ablation_4ds_summary.md"
    ensure_dir(summary_path.parent)
    summary_path.write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote: {summary_path}")


if __name__ == "__main__":
    main()

