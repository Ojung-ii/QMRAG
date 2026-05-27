#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_failures import find_all_latest, find_latest_prediction, infer_dataset, infer_prompt
from utils.eval_metrics import row_token_metrics
from utils.generation import bundle_sentence_statistics
from utils.io_utils import dump_json, ensure_dir, read_jsonl
from utils.text import token_count


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def avg(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))


def pct(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(statistics.quantiles(list(values), n=100, method="inclusive")[max(0, min(99, int(q) - 1))])


def context_text(row: Mapping[str, Any]) -> str:
    return str(row.get("rendered_context") if row.get("rendered_context") is not None else row.get("rendered_context_preview", "") or "")


def row_stats(row: Mapping[str, Any]) -> dict[str, Any]:
    context = context_text(row)
    stats = bundle_sentence_statistics(row.get("evidence_bundles", []) or [], context)
    context_tokens = row.get("rendered_context_tokens")
    if context_tokens is None:
        context_tokens = token_count(context)
    stats.update(
        {
            "id": row.get("id"),
            "dataset": row.get("dataset"),
            "prompt_profile": row.get("prompt_profile"),
            "rendering_profile": row.get("rendering_profile"),
            "context_tokens": int(context_tokens or 0),
            "input_prompt_tokens": int(row_token_metrics(row).get("input_prompt_tokens", 0) or 0),
            "tokens_per_sentence": float(context_tokens or 0) / max(1, int(stats.get("rendered_sentence_count", 0) or 0)),
        }
    )
    return stats


def summarize(rows: Sequence[Mapping[str, Any]], path: Path) -> dict[str, Any]:
    per = [row_stats(row) for row in rows]
    dataset = infer_dataset(path, rows)
    prompt = infer_prompt(rows)
    sent_per_bundle = [float(x.get("avg_sentences_per_bundle", 0.0) or 0.0) for x in per]
    summary = {
        "dataset": dataset,
        "prompt_profile": prompt,
        "source_predictions": str(path),
        "n": len(rows),
        "avg_bundle_count": avg([float(x.get("bundle_count", 0.0) or 0.0) for x in per]),
        "avg_sentences_per_bundle": avg(sent_per_bundle),
        "p50_sentences_per_bundle": statistics.median(sent_per_bundle) if sent_per_bundle else 0.0,
        "p95_sentences_per_bundle": pct(sent_per_bundle, 95),
        "avg_top1_bundle_sentence_count": avg([float(x.get("top1_bundle_sentence_count", 0.0) or 0.0) for x in per]),
        "avg_top3_bundle_sentence_count": avg([float(x.get("top3_bundle_sentence_count", 0.0) or 0.0) for x in per]),
        "avg_chain_sentence_count": avg([float(x.get("chain_sentence_count", 0.0) or 0.0) for x in per]),
        "avg_support_sentence_count": avg([float(x.get("support_sentence_count", 0.0) or 0.0) for x in per]),
        "avg_source_sentence_count": avg([float(x.get("source_sentence_count", 0.0) or 0.0) for x in per]),
        "duplicate_sentence_rate": avg([float(x.get("duplicate_sentence_rate", 0.0) or 0.0) for x in per]),
        "avg_context_tokens": avg([float(x.get("context_tokens", 0.0) or 0.0) for x in per]),
        "avg_input_prompt_tokens": avg([float(x.get("input_prompt_tokens", 0.0) or 0.0) for x in per]),
        "tokens_per_sentence": avg([float(x.get("tokens_per_sentence", 0.0) or 0.0) for x in per]),
        "per_example": per,
    }
    return summary


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown(summary: Mapping[str, Any]) -> str:
    keys = [
        "n",
        "avg_bundle_count",
        "avg_sentences_per_bundle",
        "p50_sentences_per_bundle",
        "p95_sentences_per_bundle",
        "avg_top1_bundle_sentence_count",
        "avg_top3_bundle_sentence_count",
        "avg_chain_sentence_count",
        "avg_support_sentence_count",
        "avg_source_sentence_count",
        "duplicate_sentence_rate",
        "avg_context_tokens",
        "avg_input_prompt_tokens",
        "tokens_per_sentence",
    ]
    lines = [
        "# Bundle Sentence Statistics",
        "",
        f"- dataset: {summary.get('dataset')}",
        f"- prompt_profile: {summary.get('prompt_profile')}",
        f"- source: {summary.get('source_predictions')}",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in keys:
        lines.append(f"| {key} | {fmt(summary.get(key, 0))} |")
    return "\n".join(lines) + "\n"


def evaluate_path(path: Path, analysis_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    summary = summarize(rows, path)
    stem = f"bundle_sentence_stats_{summary['dataset']}_{summary['prompt_profile']}"
    dump_json(summary, analysis_dir / f"{stem}.json")
    (analysis_dir / f"{stem}.md").write_text(markdown(summary), encoding="utf-8")
    print(markdown(summary))
    print(f"wrote: {analysis_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze sentence count and duplicate structure inside ACE-RAG evidence bundles.")
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--prompt-profile", default=None)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--all-latest", action="store_true")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--analysis-root", default="outputs/analysis")
    parser.add_argument("--analysis-dir", default=None)
    args = parser.parse_args()

    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else Path(args.analysis_root) / now_timestamp()
    ensure_dir(analysis_dir)
    if args.all_latest:
        paths = find_all_latest(Path(args.output_root))
    elif args.predictions:
        paths = [Path(args.predictions)]
    else:
        paths = [find_latest_prediction(Path(args.output_root), args.dataset, args.prompt_profile)]
    for path in paths:
        evaluate_path(path, analysis_dir)


if __name__ == "__main__":
    main()
