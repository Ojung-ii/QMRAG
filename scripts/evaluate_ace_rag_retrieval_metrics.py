#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_failures import infer_dataset, infer_prompt, infer_rendering
from utils.eval_metrics import answer_contains, answer_f1, answer_in_rendered_context, exact_match, row_token_metrics, text_has_gold
from utils.io_utils import dump_json, ensure_dir, read_jsonl
from utils.text import normalize_text


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def iter_prediction_files(output_root: Path) -> Iterable[Path]:
    for path in sorted(output_root.rglob("predictions.jsonl")):
        try:
            rel = path.relative_to(output_root)
        except ValueError:
            rel = path
        if output_root.name != "replay" and rel.parts and rel.parts[0] == "replay":
            continue
        yield path


def summarize_file(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    return {
        "path": path,
        "dataset": infer_dataset(path, rows),
        "prompt_profile": infer_prompt(rows),
        "rendering_profile": infer_rendering(rows),
        "n": len(rows),
        "mtime": path.stat().st_mtime,
    }


def find_latest_prediction(output_root: Path, dataset: str | None, prompt_profile: str | None) -> Path:
    candidates = []
    for path in iter_prediction_files(output_root):
        try:
            summary = summarize_file(path)
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
        raise SystemExit(f"No matching predictions found for dataset={dataset!r} prompt_profile={prompt_profile!r}")
    return max(candidates, key=lambda x: (float(x["mtime"]), str(x["path"])))["path"]


def find_all_latest(output_root: Path) -> list[Path]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for path in iter_prediction_files(output_root):
        try:
            summary = summarize_file(path)
        except Exception:
            continue
        if int(summary["n"]) <= 0:
            continue
        key = (str(summary["dataset"]), str(summary["prompt_profile"]))
        if key not in latest or float(summary["mtime"]) > float(latest[key]["mtime"]):
            latest[key] = summary
    return [x["path"] for x in sorted(latest.values(), key=lambda y: (str(y["dataset"]), str(y["prompt_profile"])))]


def norm_title(title: Any) -> str:
    return normalize_text(str(title or "")).strip().lower()


def bundle_titles(bundle: Mapping[str, Any]) -> list[str]:
    titles = []
    for key in ("anchor_title", "title"):
        if bundle.get(key):
            titles.append(str(bundle[key]))
    for title in bundle.get("anchor_titles", []) or []:
        titles.append(str(title))
    for title in bundle.get("bridge_titles", []) or []:
        titles.append(str(title))
    for key in ("propositions", "source_chunks"):
        for item in bundle.get(key, []) or []:
            if isinstance(item, Mapping) and item.get("title"):
                titles.append(str(item["title"]))
    for path in bundle.get("evidence_path", []) or []:
        if not isinstance(path, Mapping):
            continue
        for key in ("source_title", "bridge_title"):
            if path.get(key):
                titles.append(str(path[key]))
    dedup = []
    seen = set()
    for title in titles:
        nt = norm_title(title)
        if nt and nt not in seen:
            seen.add(nt)
            dedup.append(title)
    return dedup


def top_k_bundles(row: Mapping[str, Any], k: int = 5) -> list[Mapping[str, Any]]:
    bundles = row.get("evidence_bundles", []) or []
    return [b for b in bundles[:k] if isinstance(b, Mapping)]


def answer_recall_at_k(row: Mapping[str, Any], k: int = 5) -> float:
    answers = [str(x) for x in row.get("answers", [])]
    try:
        text = json.dumps(top_k_bundles(row, k), ensure_ascii=False)
    except Exception:
        text = str(top_k_bundles(row, k))
    return float(text_has_gold(text, answers))


def support_metrics(row: Mapping[str, Any], k: int = 5) -> dict[str, Any]:
    gold = {norm_title(x) for x in row.get("support_titles", []) or [] if norm_title(x)}
    pred = set()
    for bundle in top_k_bundles(row, k):
        pred.update(norm_title(x) for x in bundle_titles(bundle) if norm_title(x))
    if not gold:
        return {
            "has_support_labels": False,
            "supporting_fact_precision": None,
            "supporting_fact_recall": None,
            "supporting_fact_f1": None,
            "title_recall_at_5": None,
            "pred_support_count": len(pred),
            "gold_support_count": 0,
        }
    hit = len(gold & pred)
    precision = hit / max(1, len(pred))
    recall = hit / max(1, len(gold))
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "has_support_labels": True,
        "supporting_fact_precision": precision,
        "supporting_fact_recall": recall,
        "supporting_fact_f1": f1,
        "title_recall_at_5": recall,
        "pred_support_count": len(pred),
        "gold_support_count": len(gold),
    }


def timing_ms(row: Mapping[str, Any]) -> tuple[float, float, float]:
    rd = row.get("retrieval_diagnostics", {}) or {}
    timings = rd.get("timings", {}) or {}
    retrieval_ms = 1000.0 * float(timings.get("total_retrieval_s", 0.0) or 0.0)
    generation_ms = 1000.0 * float(row.get("generation_latency_s", 0.0) or 0.0)
    return retrieval_ms, generation_ms, retrieval_ms + generation_ms


def method_name(prompt_profile: str) -> str:
    if prompt_profile == "common_qa":
        return "ace_rag_common"
    if prompt_profile == "ace_rag_bundle_qa":
        return "ace_rag_bundle"
    if prompt_profile.startswith("ace_rag_bundle_"):
        return prompt_profile.replace("ace_rag_", "ace_rag_")
    return f"ace_rag_{prompt_profile}"


def avg(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))


def avg_optional(values: Sequence[float | None]) -> float | None:
    clean = [float(x) for x in values if x is not None]
    if not clean:
        return None
    return avg(clean)


def evaluate_file(path: Path, analysis_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    dataset = infer_dataset(path, rows)
    prompt = infer_prompt(rows)
    rendering = infer_rendering(rows)
    per = []
    for row in rows:
        raw = str(row.get("raw_prediction", row.get("prediction", "")) or "")
        answers = [str(x) for x in row.get("answers", [])]
        retrieval_ms, generation_ms, total_ms = timing_ms(row)
        sm = support_metrics(row, 5)
        tm = row_token_metrics(row)
        per.append(
            {
                "id": row.get("id"),
                "recall_at_5": answer_recall_at_k(row, 5),
                "em": exact_match(raw, answers),
                "f1": answer_f1(raw, answers),
                "answer_in_rendered_context": answer_in_rendered_context(row, answers),
                "answer_in_prediction": answer_contains(raw, answers),
                "retrieval_ms": retrieval_ms,
                "generation_ms": generation_ms,
                "total_ms": total_ms,
                **sm,
                **tm,
            }
        )
    summary = {
        "dataset": dataset,
        "method": method_name(prompt),
        "prompt_profile": prompt,
        "rendering_profile": rendering,
        "n": len(rows),
        "Recall@5": avg([x["recall_at_5"] for x in per]),
        "EM": avg([x["em"] for x in per]),
        "F1": avg([x["f1"] for x in per]),
        "avg_context_tokens": avg([x["rendered_context_tokens"] for x in per]),
        "avg_input_prompt_tokens": avg([x["input_prompt_tokens"] for x in per]),
        "avg_completion_tokens": avg([x["completion_tokens"] for x in per]),
        "avg_total_llm_tokens": avg([x["total_llm_tokens"] for x in per]),
        "F1_per_1k_context_tokens": avg([x["f1"] for x in per]) / max(1e-9, avg([x["rendered_context_tokens"] for x in per]) / 1000.0),
        "F1_per_1k_input_prompt_tokens": avg([x["f1"] for x in per]) / max(1e-9, avg([x["input_prompt_tokens"] for x in per]) / 1000.0),
        "F1_per_1k_total_llm_tokens": avg([x["f1"] for x in per]) / max(1e-9, avg([x["total_llm_tokens"] for x in per]) / 1000.0),
        "supporting_fact_precision": avg_optional([x["supporting_fact_precision"] for x in per]),
        "supporting_fact_recall": avg_optional([x["supporting_fact_recall"] for x in per]),
        "supporting_fact_f1": avg_optional([x["supporting_fact_f1"] for x in per]),
        "answer_in_rendered_context": avg([x["answer_in_rendered_context"] for x in per]),
        "answer_in_prediction": avg([x["answer_in_prediction"] for x in per]),
        "retrieval_ms": avg([x["retrieval_ms"] for x in per]),
        "generation_ms": avg([x["generation_ms"] for x in per]),
        "total_ms": avg([x["total_ms"] for x in per]),
        "token_count_source_counts": {k: sum(1 for x in per if x.get("token_count_source") == k) for k in sorted({str(x.get("token_count_source","unknown")) for x in per})},
        "supporting_fact_mode": "title_compatible",
        "support_labeled_rate": avg([1.0 if x["has_support_labels"] else 0.0 for x in per]),
        "source_predictions": str(path),
        "per_example": per,
    }
    stem = f"ace_rag_retrieval_metrics_{dataset}_{prompt}"
    dump_json(summary, analysis_dir / f"{stem}.json")
    (analysis_dir / f"{stem}.md").write_text(markdown_summary(summary), encoding="utf-8")
    print(markdown_summary(summary))
    print(f"wrote: {analysis_dir}")
    return summary


def fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_summary(summary: Mapping[str, Any]) -> str:
    headers = [
        "dataset",
        "method",
        "Recall@5",
        "EM",
        "F1",
        "avg_context_tokens",
        "F1_per_1k_context_tokens",
        "supporting_fact_precision",
        "supporting_fact_recall",
        "supporting_fact_f1",
        "retrieval_ms",
        "generation_ms",
        "total_ms",
    ]
    row = [
        summary.get("dataset"),
        summary.get("method"),
        fmt(summary.get("Recall@5")),
        fmt(summary.get("EM")),
        fmt(summary.get("F1")),
        fmt(summary.get("avg_context_tokens")),
        fmt(summary.get("F1_per_1k_context_tokens")),
        fmt(summary.get("supporting_fact_precision")),
        fmt(summary.get("supporting_fact_recall")),
        fmt(summary.get("supporting_fact_f1")),
        fmt(summary.get("retrieval_ms")),
        fmt(summary.get("generation_ms")),
        fmt(summary.get("total_ms")),
    ]
    lines = [
        "# ACE-RAG Retrieval Metrics",
        "",
        f"- prompt_profile: {summary.get('prompt_profile')}",
        f"- rendering_profile: {summary.get('rendering_profile')}",
        f"- supporting_fact_mode: {summary.get('supporting_fact_mode')}",
        f"- support_labeled_rate: {fmt(summary.get('support_labeled_rate'))}",
        f"- token_count_source_counts: {json.dumps(summary.get('token_count_source_counts',{}), ensure_ascii=False)}",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        "| " + " | ".join(str(x) for x in row) + " |",
        "",
        "## ACE-RAG Token Efficiency",
        "",
        "| avg_input_prompt_tokens | avg_completion_tokens | avg_total_llm_tokens | F1_per_1k_input_prompt_tokens | F1_per_1k_total_llm_tokens |",
        "| ---: | ---: | ---: | ---: | ---: |",
        "| "
        + " | ".join(
            [
                fmt(summary.get("avg_input_prompt_tokens")),
                fmt(summary.get("avg_completion_tokens")),
                fmt(summary.get("avg_total_llm_tokens")),
                fmt(summary.get("F1_per_1k_input_prompt_tokens")),
                fmt(summary.get("F1_per_1k_total_llm_tokens")),
            ]
        )
        + " |",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ACE-RAG retrieval/supporting-fact metrics from predictions.")
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
    summaries = [evaluate_file(path, analysis_dir) for path in paths]
    if len(summaries) > 1:
        dump_json({"runs": summaries}, analysis_dir / "ace_rag_retrieval_metrics_index.json")


if __name__ == "__main__":
    main()
