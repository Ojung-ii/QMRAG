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

from utils.eval_metrics import evaluate_predictions
from utils.io_utils import dump_json, ensure_dir, read_jsonl


VARIANTS = (
    "full_hetero",
    "prop_text_only",
    "prop_parent_anchor",
    "prop_parent_mention_bidirectional",
)


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def iter_prediction_files(output_root: Path) -> Iterable[Path]:
    yield from sorted(output_root.rglob("predictions.jsonl"))


def infer_dataset(path: Path, rows: list[Mapping[str, Any]]) -> str:
    for row in rows:
        if row.get("dataset"):
            return str(row["dataset"])
    parts = path.parts
    if "outputs" in parts:
        rest = parts[parts.index("outputs") + 1 :]
        if len(rest) >= 3 and rest[1] == "eval":
            return rest[0]
        if len(rest) >= 2:
            return rest[1]
    return "UNKNOWN"


def infer_variant(rows: list[Mapping[str, Any]]) -> str:
    for row in rows:
        value = row.get("retrieval_variant") or (row.get("retrieval_diagnostics", {}) or {}).get("retrieval_variant")
        if value:
            return str(value)
    return "full_hetero"


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
        "variant": infer_variant([first]),
        "candidate_cap_enabled": bool((first.get("retrieval_diagnostics", {}) or {}).get("candidate_cap_enabled", False)),
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
        if summary["variant"] != variant:
            continue
        if summary.get("candidate_cap_enabled"):
            continue
        if summary["prompt_profile"] != "common_qa" or summary["rendering_profile"] != "structured_chain":
            continue
        candidates.append(summary)
    if not candidates:
        return None
    return max(candidates, key=lambda x: (float(x["mtime"]), str(x["path"])))["path"]


def load_timing_summary(pred_path: Path) -> dict[str, Any]:
    path = pred_path.parent / "timing_summary.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def top_bottlenecks(timing: Mapping[str, Any], k: int = 4) -> list[dict[str, Any]]:
    stages = dict(timing.get("stages", {}) or {})
    rows = []
    for stage, row in stages.items():
        rows.append({"stage": stage, "total_ms": float(row.get("total_ms", 0.0) or 0.0), "mean_ms": float(row.get("mean_ms", 0.0) or 0.0)})
    rows.sort(key=lambda x: x["total_ms"], reverse=True)
    return rows[:k]


def summarize_run(path: Path, dataset: str, variant: str) -> dict[str, Any]:
    rows = read_jsonl(path)
    result = evaluate_predictions(rows, dataset=dataset, prompt_profile="common_qa")
    timing = load_timing_summary(path)
    return {
        "dataset": dataset,
        "retrieval_variant": variant,
        "path": str(path),
        "n": result.get("n", 0),
        "EM": result.get("em", 0.0),
        "F1": result.get("f1", 0.0),
        "Recall@5": result.get("support_title_recall", 0.0),
        "answer_in_rendered_context": result.get("answer_in_rendered_context", 0.0),
        "answer_in_prediction": result.get("answer_in_prediction", 0.0),
        "insufficient_rate": result.get("insufficient_rate", 0.0),
        "bridge_connected_rate": result.get("bridge_connected_rate", 0.0),
        "chain_complete_v2_rate": result.get("chain_complete_v2_rate", 0.0),
        "anchor_connected_chain_complete_rate": result.get("anchor_connected_chain_complete_rate", 0.0),
        "anchor_mismatch_chain_rate": result.get("anchor_mismatch_chain_rate", 0.0),
        "avg_context_tokens": result.get("avg_context_tokens", result.get("context_tokens", 0.0)),
        "avg_input_prompt_tokens": result.get("avg_input_prompt_tokens", 0.0),
        "F1_per_1k_input_prompt_tokens": result.get("F1_per_1k_input_prompt_tokens", 0.0),
        "retrieval_ms": result.get("retrieval_latency_ms", 0.0),
        "generation_ms": result.get("generation_latency_ms", 0.0),
        "total_ms": result.get("latency_ms", 0.0),
        "seed_unit_type_distribution": result.get("seed_unit_type_distribution", {}),
        "selected_bundle_source_type_distribution": result.get("selected_bundle_source_type_distribution", {}),
        "chain_success_by_seed_type": result.get("chain_success_by_seed_type", {}),
        "seed_title_rate": result.get("seed_title_rate", 0.0),
        "seed_chunk_rate": result.get("seed_chunk_rate", 0.0),
        "seed_proposition_rate": result.get("seed_proposition_rate", 0.0),
        "chain_from_title_rate": result.get("chain_from_title_rate", 0.0),
        "chain_from_chunk_rate": result.get("chain_from_chunk_rate", 0.0),
        "chain_from_proposition_rate": result.get("chain_from_proposition_rate", 0.0),
        "timing_top_bottlenecks": top_bottlenecks(timing),
    }


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def markdown(dataset: str, rows: list[Mapping[str, Any]], missing: list[str]) -> str:
    headers = [
        "dataset",
        "retrieval_variant",
        "n",
        "EM",
        "F1",
        "Recall@5",
        "answer_in_rendered_context",
        "answer_in_prediction",
        "insufficient_rate",
        "bridge_connected_rate",
        "chain_complete_v2_rate",
        "anchor_connected_chain_complete_rate",
        "anchor_mismatch_chain_rate",
        "avg_context_tokens",
        "avg_input_prompt_tokens",
        "F1_per_1k_input_prompt_tokens",
        "retrieval_ms",
        "generation_ms",
        "total_ms",
        "seed_proposition_rate",
        "chain_from_proposition_rate",
    ]
    lines = ["# Proposition-Centered Retrieval Ablation", "", f"- dataset: {dataset}", ""]
    if missing:
        lines.append(f"- missing_variants: {', '.join(missing)}")
        lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(h)) for h in headers) + " |")
    lines.extend(["", "## Timing Bottlenecks", ""])
    for row in rows:
        lines.append(f"### {row.get('retrieval_variant')}")
        for item in row.get("timing_top_bottlenecks", []) or []:
            lines.append(f"- {item.get('stage')}: total_ms={item.get('total_ms'):.3f}, mean_ms={item.get('mean_ms'):.3f}")
        lines.append("")
    lines.extend(["## Seed Type Distributions", ""])
    for row in rows:
        lines.append(f"- {row.get('retrieval_variant')}: seeds={json.dumps(row.get('seed_unit_type_distribution',{}),ensure_ascii=False)}, chain_success={json.dumps(row.get('chain_success_by_seed_type',{}),ensure_ascii=False)}")
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
    dump_json(result, analysis_dir / f"prop_centered_ablation_{dataset}.json")
    text = markdown(dataset, rows, missing)
    (analysis_dir / f"prop_centered_ablation_{dataset}.md").write_text(text, encoding="utf-8")
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
    parser = argparse.ArgumentParser(description="Compare ACE-RAG proposition-centered retrieval ablation variants")
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
