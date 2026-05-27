#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_failures import infer_dataset, infer_prompt, infer_rendering
from utils.eval_metrics import answer_contains, answer_f1, answer_in_rendered_context, context_tokens, exact_match
from utils.generation import is_insufficient_prediction
from utils.io_utils import dump_json, ensure_dir, read_jsonl
from utils.text import safe_truncate


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def iter_prediction_files(output_root: Path) -> Iterable[Path]:
    yield from sorted(output_root.rglob("predictions.jsonl"))


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


def find_latest_prediction(output_root: Path, dataset: str, prompt_profile: str, rendering_profile: str) -> Path:
    candidates = []
    for path in iter_prediction_files(output_root):
        try:
            summary = summarize_file(path)
        except Exception:
            continue
        if summary["dataset"] != dataset:
            continue
        if summary["prompt_profile"] != prompt_profile:
            continue
        if summary["rendering_profile"] != rendering_profile:
            continue
        if int(summary["n"]) <= 0:
            continue
        candidates.append(summary)
    if not candidates:
        raise SystemExit(
            f"No predictions found for dataset={dataset!r} prompt_profile={prompt_profile!r} rendering_profile={rendering_profile!r}"
        )
    return max(candidates, key=lambda x: (float(x["mtime"]), str(x["path"])))["path"]


def row_metrics(row: Mapping[str, Any]) -> dict[str, float | bool]:
    answers = [str(x) for x in row.get("answers", [])]
    raw = str(row.get("raw_prediction", row.get("prediction", "")) or "")
    return {
        "em": exact_match(raw, answers),
        "f1": answer_f1(raw, answers),
        "answer_in_prediction": answer_contains(raw, answers),
        "answer_in_rendered_context": answer_in_rendered_context(row, answers),
        "insufficient": bool(is_insufficient_prediction(raw)),
        "ctx_tokens": float(context_tokens(row)),
        "latency_ms": 1000.0 * float(((row.get("retrieval_diagnostics", {}) or {}).get("timings", {}) or {}).get("total_retrieval_s", 0.0) or 0.0)
        + 1000.0 * float(row.get("generation_latency_s", 0.0) or 0.0),
    }


def avg(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))


def compare(left_path: Path, right_path: Path, analysis_dir: Path) -> dict[str, Any]:
    left_rows = read_jsonl(left_path)
    right_rows = read_jsonl(right_path)
    left_by_id = {str(row.get("id")): row for row in left_rows}
    right_by_id = {str(row.get("id")): row for row in right_rows}
    ids = [qid for qid in left_by_id if qid in right_by_id]
    pair_rows = []
    for qid in ids:
        left = left_by_id[qid]
        right = right_by_id[qid]
        lm = row_metrics(left)
        rm = row_metrics(right)
        left_correct = bool(lm["em"] or lm["answer_in_prediction"])
        right_correct = bool(rm["em"] or rm["answer_in_prediction"])
        pair_rows.append(
            {
                "id": qid,
                "question": left.get("question", right.get("question", "")),
                "answers": left.get("answers", right.get("answers", [])),
                "left_prediction": left.get("raw_prediction", left.get("prediction", "")),
                "right_prediction": right.get("raw_prediction", right.get("prediction", "")),
                "left_correct": left_correct,
                "right_correct": right_correct,
                "left_metrics": lm,
                "right_metrics": rm,
                "evidence_bundles_hash_match": json_hash(left.get("evidence_bundles", []) or [])
                == json_hash(right.get("evidence_bundles", []) or []),
            }
        )
    n = max(1, len(pair_rows))
    left_metrics = [row["left_metrics"] for row in pair_rows]
    right_metrics = [row["right_metrics"] for row in pair_rows]
    fixed = [row for row in pair_rows if not row["left_correct"] and row["right_correct"]]
    broken = [row for row in pair_rows if row["left_correct"] and not row["right_correct"]]
    both_correct = [row for row in pair_rows if row["left_correct"] and row["right_correct"]]
    both_wrong = [row for row in pair_rows if not row["left_correct"] and not row["right_correct"]]
    dataset = infer_dataset(left_path, left_rows)
    prompt = infer_prompt(left_rows)
    left_rendering = infer_rendering(left_rows)
    right_rendering = infer_rendering(right_rows)
    summary = {
        "dataset": dataset,
        "prompt_profile": prompt,
        "left_path": str(left_path),
        "right_path": str(right_path),
        "left_rendering_profile": left_rendering,
        "right_rendering_profile": right_rendering,
        "n": len(pair_rows),
        "left_EM": avg([float(x["em"]) for x in left_metrics]),
        "left_F1": avg([float(x["f1"]) for x in left_metrics]),
        "left_answer_in_prediction": avg([float(x["answer_in_prediction"]) for x in left_metrics]),
        "left_insufficient_rate": avg([1.0 if x["insufficient"] else 0.0 for x in left_metrics]),
        "right_EM": avg([float(x["em"]) for x in right_metrics]),
        "right_F1": avg([float(x["f1"]) for x in right_metrics]),
        "right_answer_in_prediction": avg([float(x["answer_in_prediction"]) for x in right_metrics]),
        "right_insufficient_rate": avg([1.0 if x["insufficient"] else 0.0 for x in right_metrics]),
        "fixed_by_right": len(fixed),
        "broken_by_right": len(broken),
        "both_correct": len(both_correct),
        "both_wrong": len(both_wrong),
        "left_avg_CtxTok": avg([float(x["ctx_tokens"]) for x in left_metrics]),
        "right_avg_CtxTok": avg([float(x["ctx_tokens"]) for x in right_metrics]),
        "left_avg_LatencyMs": avg([float(x["latency_ms"]) for x in left_metrics]),
        "right_avg_LatencyMs": avg([float(x["latency_ms"]) for x in right_metrics]),
        "evidence_bundles_hash_match_rate": sum(1.0 if row["evidence_bundles_hash_match"] else 0.0 for row in pair_rows) / n,
    }
    stem = f"rendering_compare_{dataset}_{left_rendering}_{right_rendering}"
    dump_json({"summary": summary, "examples": pair_rows}, analysis_dir / f"{stem}.json")
    (analysis_dir / f"{stem}.md").write_text(markdown_compare(summary, pair_rows), encoding="utf-8")
    print(markdown_compare(summary, pair_rows[:20]))
    print(f"wrote: {analysis_dir}")
    return summary


def markdown_compare(summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Rendering Run Comparison",
        "",
        f"- dataset: {summary.get('dataset')}",
        f"- prompt_profile: {summary.get('prompt_profile')}",
        f"- left_rendering: {summary.get('left_rendering_profile')} ({summary.get('left_path')})",
        f"- right_rendering: {summary.get('right_rendering_profile')} ({summary.get('right_path')})",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in [
        "n",
        "left_EM",
        "left_F1",
        "left_answer_in_prediction",
        "left_insufficient_rate",
        "right_EM",
        "right_F1",
        "right_answer_in_prediction",
        "right_insufficient_rate",
        "fixed_by_right",
        "broken_by_right",
        "both_correct",
        "both_wrong",
        "left_avg_CtxTok",
        "right_avg_CtxTok",
        "left_avg_LatencyMs",
        "right_avg_LatencyMs",
        "evidence_bundles_hash_match_rate",
    ]:
        value = summary.get(key, 0)
        if isinstance(value, float):
            value = f"{value:.4f}"
        lines.append(f"| {key} | {value} |")
    changed = [row for row in rows if row.get("left_correct") != row.get("right_correct")]
    if changed:
        lines.extend(["", "## Changed Examples", ""])
        for row in changed[:20]:
            lines.extend(
                [
                    f"### {row.get('id')}",
                    "",
                    f"- question: {row.get('question')}",
                    f"- answers: {row.get('answers')}",
                    f"- left_correct: {row.get('left_correct')}",
                    f"- right_correct: {row.get('right_correct')}",
                    f"- left_prediction: {safe_truncate(str(row.get('left_prediction', '')), 240)}",
                    f"- right_prediction: {safe_truncate(str(row.get('right_prediction', '')), 240)}",
                    "",
                ]
            )
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ACE-RAG rendering-profile replay runs by id.")
    parser.add_argument("--left", default=None)
    parser.add_argument("--right", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--prompt-profile", default="common_qa")
    parser.add_argument("--left-rendering", default=None)
    parser.add_argument("--right-rendering", default=None)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--analysis-root", default="outputs/analysis")
    parser.add_argument("--analysis-dir", default=None)
    args = parser.parse_args()

    if args.left:
        left_path = Path(args.left)
    else:
        if not args.dataset or not args.left_rendering:
            raise SystemExit("--dataset and --left-rendering are required when --left is omitted")
        left_path = find_latest_prediction(Path(args.output_root), args.dataset, args.prompt_profile, args.left_rendering)
    if args.right:
        right_path = Path(args.right)
    else:
        if not args.dataset or not args.right_rendering:
            raise SystemExit("--dataset and --right-rendering are required when --right is omitted")
        right_path = find_latest_prediction(Path(args.output_root), args.dataset, args.prompt_profile, args.right_rendering)
    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else Path(args.analysis_root) / now_timestamp()
    ensure_dir(analysis_dir)
    compare(left_path, right_path, analysis_dir)


if __name__ == "__main__":
    main()
