#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.eval_metrics import (
    answer_contains,
    answer_f1,
    answer_in_rendered_context,
    bridge_stats,
    context_tokens,
    evaluate_predictions,
    exact_match,
)
from utils.generation import has_idk_phrase, is_insufficient_prediction
from utils.io_utils import dump_json, ensure_dir, read_jsonl, to_jsonable, write_jsonl
from utils.text import safe_truncate


FAILURE_CATEGORIES = (
    "OK",
    "RETRIEVAL_MISS",
    "REFUSAL_FAIL",
    "MULTI_ANCHOR_FAIL",
    "ANCHOR_MISMATCH_FAIL",
    "GENERATION_FAIL",
    "OTHER_FAIL",
)


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def infer_dataset(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    for row in rows:
        if row.get("dataset"):
            return str(row["dataset"])
    parts = path.parts
    if "outputs" in parts:
        rest = parts[parts.index("outputs") + 1 :]
        if len(rest) >= 3 and rest[1] == "eval":
            return rest[0]
        if len(rest) >= 3:
            return rest[1]
    return "UNKNOWN"


def infer_prompt(rows: Sequence[Mapping[str, Any]], fallback: str | None = None) -> str:
    for row in rows:
        if row.get("prompt_profile"):
            return str(row["prompt_profile"])
    return str(fallback or "UNKNOWN")


def infer_rendering(rows: Sequence[Mapping[str, Any]], fallback: str | None = None) -> str:
    for row in rows:
        if row.get("rendering_profile"):
            return str(row["rendering_profile"])
    return str(fallback or "structured_chain")


def iter_prediction_files(output_root: Path) -> Iterable[Path]:
    for path in sorted(output_root.rglob("predictions.jsonl")):
        try:
            rel = path.relative_to(output_root)
        except ValueError:
            rel = path
        if output_root.name != "replay" and rel.parts and rel.parts[0] == "replay":
            continue
        yield path


def summarize_prediction_file(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    dataset = infer_dataset(path, rows)
    prompt = infer_prompt(rows)
    rendering = infer_rendering(rows)
    return {
        "path": path,
        "dataset": dataset,
        "prompt_profile": prompt,
        "rendering_profile": rendering,
        "n": len(rows),
        "mtime": path.stat().st_mtime,
    }


def find_latest_prediction(
    output_root: Path,
    dataset: str | None = None,
    prompt_profile: str | None = None,
) -> Path:
    candidates = []
    for path in iter_prediction_files(output_root):
        try:
            summary = summarize_prediction_file(path)
        except Exception:
            continue
        if dataset and summary["dataset"] != dataset:
            continue
        if prompt_profile and summary["prompt_profile"] != prompt_profile:
            continue
        if int(summary["n"]) <= 0:
            continue
        candidates.append(summary)
    if not candidates:
        filters = f"dataset={dataset!r} prompt_profile={prompt_profile!r}"
        raise SystemExit(f"No matching predictions.jsonl found for {filters}")
    return max(candidates, key=lambda x: (float(x["mtime"]), str(x["path"])))["path"]


def find_all_latest(output_root: Path) -> list[Path]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for path in iter_prediction_files(output_root):
        try:
            summary = summarize_prediction_file(path)
        except Exception:
            continue
        if int(summary["n"]) <= 0:
            continue
        key = (str(summary["dataset"]), str(summary["prompt_profile"]))
        if key not in latest or float(summary["mtime"]) > float(latest[key]["mtime"]):
            latest[key] = summary
    return [x["path"] for x in sorted(latest.values(), key=lambda y: (str(y["dataset"]), str(y["prompt_profile"])))]


def retrieval_flag(row: Mapping[str, Any], key: str, fallback: Any = False) -> bool:
    rd = row.get("retrieval_diagnostics", {}) or {}
    if key in rd:
        return bool(rd.get(key))
    return bool(fallback)


def query_titles(row: Mapping[str, Any], key: str) -> list[str]:
    rd = row.get("retrieval_diagnostics", {}) or {}
    value = rd.get(key)
    if value is None:
        for bundle in row.get("evidence_bundles", []) or []:
            value = bundle.get(key)
            if value:
                break
    if isinstance(value, list):
        return [str(x) for x in value]
    if value:
        return [str(value)]
    return []


def row_latency_ms(row: Mapping[str, Any]) -> float:
    rd = row.get("retrieval_diagnostics", {}) or {}
    timings = rd.get("timings", {}) or {}
    retrieval_ms = 1000.0 * float(timings.get("total_retrieval_s", 0.0) or 0.0)
    generation_ms = 1000.0 * float(row.get("generation_latency_s", 0.0) or 0.0)
    return retrieval_ms + generation_ms


def classify_example(row: Mapping[str, Any]) -> dict[str, Any]:
    answers = [str(x) for x in row.get("answers", [])]
    raw_prediction = str(row.get("raw_prediction", row.get("prediction", "")) or "")
    prediction = str(row.get("prediction", raw_prediction) or "")
    em = exact_match(raw_prediction, answers)
    f1 = answer_f1(raw_prediction, answers)
    in_context = answer_in_rendered_context(row, answers)
    in_prediction = answer_contains(raw_prediction, answers)
    insufficient = bool(is_insufficient_prediction(raw_prediction))
    idk = bool(has_idk_phrase(raw_prediction))
    bs = bridge_stats(row)
    bridge_connected = bool(bs.get("has_bridge_connected", 0.0))
    answer_slot_aligned = bool(bs.get("has_answer_slot_aligned", 0.0))
    chain_complete_v2 = bool(bs.get("has_chain_complete_v2", 0.0))
    anchor_connected_chain_complete = bool(bs.get("has_anchor_connected_chain_complete", 0.0))
    anchor_mismatch_chain = bool(bs.get("has_anchor_mismatch_chain", 0.0))
    multi_anchor_bundle = bool(bs.get("has_multi_anchor_bundle", 0.0))
    generic_relation_top1 = bool(bs.get("generic_relation_top1", 0.0))

    if em or in_prediction:
        category = "OK"
    elif not in_context:
        category = "RETRIEVAL_MISS"
    elif insufficient or idk:
        category = "REFUSAL_FAIL"
    elif multi_anchor_bundle and not in_prediction:
        category = "MULTI_ANCHOR_FAIL"
    elif anchor_mismatch_chain and not in_prediction:
        category = "ANCHOR_MISMATCH_FAIL"
    elif in_context and not in_prediction:
        category = "GENERATION_FAIL"
    else:
        category = "OTHER_FAIL"

    return {
        "id": row.get("id"),
        "dataset": row.get("dataset"),
        "prompt_profile": row.get("prompt_profile"),
        "rendering_profile": row.get("rendering_profile", "structured_chain"),
        "question": row.get("question", ""),
        "answers": answers,
        "prediction": prediction,
        "raw_prediction": raw_prediction,
        "em": float(em),
        "f1": float(f1),
        "answer_in_rendered_context": float(in_context),
        "answer_in_prediction": float(in_prediction),
        "insufficient": insufficient,
        "idk": idk,
        "bridge_connected": bridge_connected,
        "answer_slot_aligned": answer_slot_aligned,
        "chain_complete_v2": chain_complete_v2,
        "anchor_connected_chain_complete": anchor_connected_chain_complete,
        "anchor_mismatch_chain": anchor_mismatch_chain,
        "multi_anchor_bundle": multi_anchor_bundle,
        "generic_relation_top1": generic_relation_top1,
        "query_anchor_titles": query_titles(row, "query_anchor_titles"),
        "query_relation_titles": query_titles(row, "query_relation_titles"),
        "failure_category": category,
        "context_tokens": context_tokens(row),
        "latency_ms": row_latency_ms(row),
    }


def build_summary(examples: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]], dataset: str, prompt: str) -> dict[str, Any]:
    result = evaluate_predictions(rows, dataset=dataset, prompt_profile=prompt)
    rendering = infer_rendering(rows)
    n = max(1, len(examples))
    counts = Counter(str(x["failure_category"]) for x in examples)
    summary = {
        "dataset": dataset,
        "prompt_profile": prompt,
        "rendering_profile": rendering,
        "prompt_experiment_type": result.get("prompt_experiment_type", "unknown"),
        "n": len(examples),
        "EM": result.get("em", 0.0),
        "F1": result.get("f1", 0.0),
        "answer_in_rendered_context": result.get("answer_in_rendered_context", 0.0),
        "answer_in_prediction": result.get("answer_in_prediction", 0.0),
        "insufficient_rate": result.get("insufficient_rate", 0.0),
        "idk_rate": result.get("idk_rate", 0.0),
        "retrieval_miss_rate": counts["RETRIEVAL_MISS"] / n,
        "refusal_fail_rate": counts["REFUSAL_FAIL"] / n,
        "multi_anchor_fail_rate": counts["MULTI_ANCHOR_FAIL"] / n,
        "anchor_mismatch_fail_rate": counts["ANCHOR_MISMATCH_FAIL"] / n,
        "generation_fail_rate": counts["GENERATION_FAIL"] / n,
        "ok_rate": counts["OK"] / n,
        "avg_ctx_tokens": result.get("context_tokens", 0.0),
        "avg_latency_ms": result.get("latency_ms", 0.0),
        "failure_counts": {cat: counts.get(cat, 0) for cat in FAILURE_CATEGORIES},
    }
    return summary


def markdown_summary(summary: Mapping[str, Any]) -> str:
    rows = [
        ("n", summary.get("n", 0)),
        ("EM", f"{float(summary.get('EM', 0.0)):.4f}"),
        ("F1", f"{float(summary.get('F1', 0.0)):.4f}"),
        ("answer_in_rendered_context", f"{float(summary.get('answer_in_rendered_context', 0.0)):.4f}"),
        ("answer_in_prediction", f"{float(summary.get('answer_in_prediction', 0.0)):.4f}"),
        ("insufficient_rate", f"{float(summary.get('insufficient_rate', 0.0)):.4f}"),
        ("idk_rate", f"{float(summary.get('idk_rate', 0.0)):.4f}"),
        ("retrieval_miss_rate", f"{float(summary.get('retrieval_miss_rate', 0.0)):.4f}"),
        ("refusal_fail_rate", f"{float(summary.get('refusal_fail_rate', 0.0)):.4f}"),
        ("multi_anchor_fail_rate", f"{float(summary.get('multi_anchor_fail_rate', 0.0)):.4f}"),
        ("anchor_mismatch_fail_rate", f"{float(summary.get('anchor_mismatch_fail_rate', 0.0)):.4f}"),
        ("generation_fail_rate", f"{float(summary.get('generation_fail_rate', 0.0)):.4f}"),
        ("ok_rate", f"{float(summary.get('ok_rate', 0.0)):.4f}"),
        ("avg_ctx_tokens", f"{float(summary.get('avg_ctx_tokens', 0.0)):.1f}"),
        ("avg_latency_ms", f"{float(summary.get('avg_latency_ms', 0.0)):.1f}"),
    ]
    lines = [
        "# Failure Analysis Summary",
        "",
        f"- dataset: {summary.get('dataset', 'UNKNOWN')}",
        f"- prompt_profile: {summary.get('prompt_profile', 'UNKNOWN')}",
        f"- rendering_profile: {summary.get('rendering_profile', 'structured_chain')}",
        f"- prompt_experiment_type: {summary.get('prompt_experiment_type', 'unknown')}",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in rows)
    lines.extend(["", "## Failure Counts", "", "| category | count |", "|---|---:|"])
    counts = summary.get("failure_counts", {}) or {}
    lines.extend(f"| {cat} | {int(counts.get(cat, 0))} |" for cat in FAILURE_CATEGORIES)
    return "\n".join(lines) + "\n"


def top_chain_lines(row: Mapping[str, Any], max_items: int = 4) -> list[str]:
    lines = []
    for bundle in row.get("evidence_bundles", []) or []:
        paths = [
            x
            for x in bundle.get("evidence_path", []) or []
            if isinstance(x, Mapping) and x.get("path_type") == "mention_bridge"
        ]
        if not paths:
            continue
        lines.append(
            f"- chain anchor={bundle.get('anchor_title')} bridge={bundle.get('bridge_titles')} "
            f"anchor_connected={bundle.get('anchor_connected')} chain_complete_v2={bundle.get('chain_complete_v2')}"
        )
        for path in paths[:max_items]:
            seed = safe_truncate(str(path.get("seed_prop", "")), 240)
            bridge = safe_truncate(str(path.get("bridge_prop", "")), 240)
            lines.append(f"  - {path.get('source_title')}: {seed}")
            lines.append(f"  - {path.get('bridge_title')}: {bridge}")
        break
    return lines


def multi_anchor_lines(row: Mapping[str, Any], max_items: int = 6) -> list[str]:
    lines = []
    for bundle in row.get("evidence_bundles", []) or []:
        if bundle.get("bundle_type") != "multi_anchor":
            continue
        lines.append(f"- multi_anchor anchors={bundle.get('anchor_titles')} complete={bundle.get('multi_anchor_complete')}")
        for prop in (bundle.get("propositions", []) or [])[:max_items]:
            lines.append(f"  - {prop.get('title')}: {safe_truncate(str(prop.get('text', '')), 240)}")
        break
    return lines


def markdown_samples(rows: Sequence[Mapping[str, Any]], examples: Sequence[Mapping[str, Any]], sample: int) -> str:
    by_id = {str(row.get("id")): row for row in rows}
    by_cat: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for example in examples:
        if example["failure_category"] != "OK":
            by_cat[str(example["failure_category"])].append(example)
    lines = ["# Failure Samples", ""]
    for cat in FAILURE_CATEGORIES:
        if cat == "OK":
            continue
        selected = by_cat.get(cat, [])[:sample]
        if not selected:
            continue
        lines.extend([f"## {cat}", ""])
        for ex in selected:
            row = by_id.get(str(ex.get("id")), {})
            context = row.get("rendered_context")
            if context is None:
                context = row.get("rendered_context_preview", "")
            lines.extend(
                [
                    f"### {ex.get('id')}",
                    "",
                    f"- question: {ex.get('question')}",
                    f"- answers: {ex.get('answers')}",
                    f"- prediction: {safe_truncate(str(ex.get('raw_prediction', '')), 300)}",
                    f"- query_anchor_titles: {ex.get('query_anchor_titles')}",
                    f"- query_relation_titles: {ex.get('query_relation_titles')}",
                    "",
                    "Top evidence chain:",
                ]
            )
            chain = top_chain_lines(row)
            lines.extend(chain if chain else ["- none"])
            lines.extend(["", "Multi-anchor evidence:"])
            multi = multi_anchor_lines(row)
            lines.extend(multi if multi else ["- none"])
            lines.extend(["", "Rendered context preview:", "```text", safe_truncate(str(context or ""), 1600), "```", ""])
    return "\n".join(lines).strip() + "\n"


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    ensure_dir(path.parent)
    fields = [
        "id",
        "dataset",
        "prompt_profile",
        "rendering_profile",
        "failure_category",
        "em",
        "f1",
        "answer_in_rendered_context",
        "answer_in_prediction",
        "insufficient",
        "idk",
        "bridge_connected",
        "answer_slot_aligned",
        "chain_complete_v2",
        "anchor_connected_chain_complete",
        "anchor_mismatch_chain",
        "multi_anchor_bundle",
        "generic_relation_top1",
        "query_anchor_titles",
        "query_relation_titles",
        "context_tokens",
        "latency_ms",
        "question",
        "prediction",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            flat["query_anchor_titles"] = "; ".join(str(x) for x in row.get("query_anchor_titles", []) or [])
            flat["query_relation_titles"] = "; ".join(str(x) for x in row.get("query_relation_titles", []) or [])
            writer.writerow({field: flat.get(field, "") for field in fields})


def analyze_predictions(path: Path, analysis_dir: Path, sample: int) -> dict[str, Any]:
    rows = read_jsonl(path)
    dataset = infer_dataset(path, rows)
    prompt = infer_prompt(rows)
    rendering = infer_rendering(rows)
    for row in rows:
        row.setdefault("dataset", dataset)
        row.setdefault("prompt_profile", prompt)
        row.setdefault("rendering_profile", rendering)
    examples = [classify_example(row) for row in rows]
    summary = build_summary(examples, rows, dataset, prompt)
    stem = f"{dataset}_{prompt}_{rendering}"
    write_jsonl(examples, analysis_dir / f"{stem}_failure_examples.jsonl")
    write_csv(examples, analysis_dir / f"{stem}_failure_examples.csv")
    dump_json(summary, analysis_dir / f"{stem}_failure_summary.json")
    (analysis_dir / f"{stem}_failure_summary.md").write_text(markdown_summary(summary), encoding="utf-8")
    (analysis_dir / f"{stem}_failure_samples.md").write_text(markdown_samples(rows, examples, sample), encoding="utf-8")
    print(markdown_summary(summary))
    print(f"wrote: {analysis_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Categorize QMRAG prediction failures.")
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--prompt-profile", default=None)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--all-latest", action="store_true")
    parser.add_argument("--sample", type=int, default=10)
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
    elif args.latest or args.dataset or args.prompt_profile:
        paths = [find_latest_prediction(Path(args.output_root), args.dataset, args.prompt_profile)]
    else:
        raise SystemExit("Provide --predictions, --latest, --dataset/--prompt-profile, or --all-latest")

    summaries = [analyze_predictions(path, analysis_dir, args.sample) for path in paths]
    if len(summaries) > 1:
        dump_json({"runs": summaries}, analysis_dir / "failure_analysis_index.json")


if __name__ == "__main__":
    main()
