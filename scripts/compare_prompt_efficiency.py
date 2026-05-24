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

from scripts.analyze_failures import infer_dataset, infer_prompt, infer_rendering
from utils.eval_metrics import evaluate_predictions
from utils.generation import PROMPT_TEMPLATES, build_prompt
from utils.io_utils import dump_json, ensure_dir, read_jsonl
from utils.text import token_count


PROMPT_ORDER = ("common_qa", "qmrag_bundle_tiny", "qmrag_bundle_light", "qmrag_bundle_qa")


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def iter_prediction_files(output_root: Path) -> Iterable[Path]:
    yield from sorted(output_root.rglob("predictions.jsonl"))


def summarize_path(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    return {
        "path": path,
        "dataset": infer_dataset(path, rows),
        "prompt_profile": infer_prompt(rows),
        "rendering_profile": infer_rendering(rows),
        "n": len(rows),
        "mtime": path.stat().st_mtime,
    }


def find_latest_prompt_run(output_root: Path, dataset: str, prompt_profile: str) -> Path | None:
    candidates = []
    for path in iter_prediction_files(output_root):
        try:
            summary = summarize_path(path)
        except Exception:
            continue
        if summary["dataset"] != dataset:
            continue
        if summary["prompt_profile"] != prompt_profile:
            continue
        if summary["rendering_profile"] != "structured_chain":
            continue
        if int(summary["n"]) <= 0:
            continue
        candidates.append(summary)
    if not candidates:
        return None
    return max(candidates, key=lambda x: (float(x["mtime"]), str(x["path"])))["path"]


def context_from_row(row: Mapping[str, Any]) -> str:
    value = row.get("rendered_context")
    if value is None:
        value = row.get("rendered_context_preview", "")
    return str(value or "")


def avg_prompt_total_tokens(rows: list[dict[str, Any]], prompt_profile: str) -> float | None:
    if prompt_profile not in PROMPT_TEMPLATES:
        return None
    values = [token_count(build_prompt(str(row.get("question", "")), context_from_row(row), prompt_profile)) for row in rows]
    if not values:
        return None
    return sum(values) / len(values)


def prompt_template_tokens(prompt_profile: str) -> int | None:
    if prompt_profile not in PROMPT_TEMPLATES:
        return None
    return token_count(PROMPT_TEMPLATES[prompt_profile].format(question="", context=""))


def summarize_run(path: Path, dataset: str, prompt_profile: str) -> dict[str, Any]:
    rows = read_jsonl(path)
    result = evaluate_predictions(rows, dataset=dataset, prompt_profile=prompt_profile)
    return {
        "dataset": dataset,
        "prompt_profile": prompt_profile,
        "path": str(path),
        "n": result.get("n", 0),
        "EM": result.get("em", 0.0),
        "F1": result.get("f1", 0.0),
        "answer_in_prediction": result.get("answer_in_prediction", 0.0),
        "insufficient_rate": result.get("insufficient_rate", 0.0),
        "avg_context_tokens": result.get("context_tokens", 0.0),
        "avg_generation_ms": result.get("generation_latency_ms", 0.0),
        "avg_total_ms": result.get("latency_ms", 0.0),
        "avg_prompt_template_tokens": prompt_template_tokens(prompt_profile),
        "avg_prompt_total_tokens": avg_prompt_total_tokens(rows, prompt_profile),
    }


def add_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_prompt = {row["prompt_profile"]: row for row in rows}
    common = by_prompt.get("common_qa")
    full = by_prompt.get("qmrag_bundle_qa")
    common_f1 = float(common.get("F1", 0.0)) if common else 0.0
    common_ins = float(common.get("insufficient_rate", 0.0)) if common else 0.0
    full_gain = (float(full.get("F1", 0.0)) - common_f1) if full else 0.0
    for row in rows:
        row["delta_F1_vs_common"] = float(row.get("F1", 0.0)) - common_f1
        row["delta_insufficient_vs_common"] = float(row.get("insufficient_rate", 0.0)) - common_ins
        if row["prompt_profile"] == "common_qa":
            row["retained_gain_vs_full"] = None
        elif abs(full_gain) < 1e-12:
            row["retained_gain_vs_full"] = None
        else:
            row["retained_gain_vs_full"] = row["delta_F1_vs_common"] / full_gain
    return rows


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(dataset: str, rows: list[Mapping[str, Any]]) -> str:
    headers = [
        "dataset",
        "prompt_profile",
        "n",
        "EM",
        "F1",
        "answer_in_prediction",
        "insufficient_rate",
        "avg_context_tokens",
        "avg_generation_ms",
        "avg_total_ms",
        "avg_prompt_template_tokens",
        "avg_prompt_total_tokens",
        "delta_F1_vs_common",
        "delta_insufficient_vs_common",
        "retained_gain_vs_full",
    ]
    lines = ["# Prompt Efficiency", "", f"- dataset: {dataset}", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(h)) for h in headers) + " |")
    return "\n".join(lines) + "\n"


def compare_dataset(dataset: str, output_root: Path, analysis_dir: Path) -> dict[str, Any]:
    rows = []
    missing = []
    for prompt in PROMPT_ORDER:
        path = find_latest_prompt_run(output_root, dataset, prompt)
        if path is None:
            missing.append(prompt)
            continue
        rows.append(summarize_run(path, dataset, prompt))
    rows = add_deltas(rows)
    result = {"dataset": dataset, "rows": rows, "missing_prompt_profiles": missing}
    dump_json(result, analysis_dir / f"prompt_efficiency_{dataset}.json")
    (analysis_dir / f"prompt_efficiency_{dataset}.md").write_text(markdown_table(dataset, rows), encoding="utf-8")
    print(markdown_table(dataset, rows))
    if missing:
        print(f"missing prompt profiles for {dataset}: {', '.join(missing)}")
    print(f"wrote: {analysis_dir}")
    return result


def latest_datasets(output_root: Path) -> list[str]:
    datasets = set()
    for path in iter_prediction_files(output_root):
        try:
            summary = summarize_path(path)
        except Exception:
            continue
        if int(summary["n"]) > 0:
            datasets.add(str(summary["dataset"]))
    return sorted(datasets)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare prompt efficiency across QMRAG prompt profiles.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--all-latest", action="store_true")
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--analysis-root", default="outputs/analysis")
    parser.add_argument("--analysis-dir", default=None)
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else Path(args.analysis_root) / now_timestamp()
    ensure_dir(analysis_dir)
    datasets = latest_datasets(Path(args.output_root)) if args.all_latest else [args.dataset]
    datasets = [d for d in datasets if d]
    if not datasets:
        raise SystemExit("Provide --dataset or --all-latest")
    results = [compare_dataset(dataset, Path(args.output_root), analysis_dir) for dataset in datasets]
    if len(results) > 1:
        dump_json({"runs": results}, analysis_dir / "prompt_efficiency_index.json")


if __name__ == "__main__":
    main()
