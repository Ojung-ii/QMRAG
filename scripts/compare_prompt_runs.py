#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_failures import classify_example, find_latest_prediction, infer_dataset, infer_prompt
from utils.eval_metrics import answer_contains, answer_f1, answer_in_rendered_context, exact_match
from utils.generation import is_insufficient_prediction
from utils.io_utils import dump_json, ensure_dir, read_jsonl
from utils.text import safe_truncate


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def context_hash(row: Mapping[str, Any]) -> str:
    value = row.get("rendered_context_hash")
    if value:
        return str(value)
    context = row.get("rendered_context")
    if context is None:
        context = row.get("rendered_context_preview", "")
    return sha256_text(str(context or ""))


def row_metrics(row: Mapping[str, Any]) -> dict[str, float | bool]:
    answers = [str(x) for x in row.get("answers", [])]
    raw = str(row.get("raw_prediction", row.get("prediction", "")) or "")
    return {
        "em": exact_match(raw, answers),
        "f1": answer_f1(raw, answers),
        "answer_in_prediction": answer_contains(raw, answers),
        "answer_in_rendered_context": answer_in_rendered_context(row, answers),
        "insufficient": bool(is_insufficient_prediction(raw)),
    }


def avg(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / max(1, len(rows))


def compare(left_path: Path, right_path: Path, analysis_dir: Path) -> dict[str, Any]:
    left_rows = read_jsonl(left_path)
    right_rows = read_jsonl(right_path)
    left_by_id = {str(row.get("id")): row for row in left_rows}
    right_by_id = {str(row.get("id")): row for row in right_rows}
    ids = [qid for qid in left_by_id if qid in right_by_id]
    dataset = infer_dataset(left_path, left_rows)
    left_prompt = infer_prompt(left_rows)
    right_prompt = infer_prompt(right_rows)
    pair_rows = []
    hash_mismatches = 0
    for qid in ids:
        left = left_by_id[qid]
        right = right_by_id[qid]
        lm = row_metrics(left)
        rm = row_metrics(right)
        lcat = classify_example(left)["failure_category"]
        rcat = classify_example(right)["failure_category"]
        same_hash = context_hash(left) == context_hash(right)
        if not same_hash:
            hash_mismatches += 1
        pair_rows.append(
            {
                "id": qid,
                "question": left.get("question", right.get("question", "")),
                "answers": left.get("answers", right.get("answers", [])),
                "left_prediction": left.get("raw_prediction", left.get("prediction", "")),
                "right_prediction": right.get("raw_prediction", right.get("prediction", "")),
                "left_correct": bool(lm["em"] or lm["answer_in_prediction"]),
                "right_correct": bool(rm["em"] or rm["answer_in_prediction"]),
                "left_failure_category": lcat,
                "right_failure_category": rcat,
                "same_rendered_context_hash": same_hash,
                "left_answer_in_rendered_context": lm["answer_in_rendered_context"],
                "right_answer_in_rendered_context": rm["answer_in_rendered_context"],
                "left_insufficient": lm["insufficient"],
                "right_insufficient": rm["insufficient"],
                "left_em": lm["em"],
                "right_em": rm["em"],
                "left_f1": lm["f1"],
                "right_f1": rm["f1"],
            }
        )

    n = max(1, len(pair_rows))
    fixed = [x for x in pair_rows if not x["left_correct"] and x["right_correct"]]
    broken = [x for x in pair_rows if x["left_correct"] and not x["right_correct"]]
    both_correct = [x for x in pair_rows if x["left_correct"] and x["right_correct"]]
    both_wrong = [x for x in pair_rows if not x["left_correct"] and not x["right_correct"]]
    summary = {
        "dataset": dataset,
        "left_path": str(left_path),
        "right_path": str(right_path),
        "left_prompt_profile": left_prompt,
        "right_prompt_profile": right_prompt,
        "n": len(pair_rows),
        "left_EM": avg([row_metrics(left_by_id[x["id"]]) for x in pair_rows], "em") if pair_rows else 0.0,
        "left_F1": avg([row_metrics(left_by_id[x["id"]]) for x in pair_rows], "f1") if pair_rows else 0.0,
        "left_answer_in_prediction": sum(1.0 if row_metrics(left_by_id[x["id"]])["answer_in_prediction"] else 0.0 for x in pair_rows) / n,
        "left_insufficient": sum(1.0 if x["left_insufficient"] else 0.0 for x in pair_rows) / n,
        "right_EM": avg([row_metrics(right_by_id[x["id"]]) for x in pair_rows], "em") if pair_rows else 0.0,
        "right_F1": avg([row_metrics(right_by_id[x["id"]]) for x in pair_rows], "f1") if pair_rows else 0.0,
        "right_answer_in_prediction": sum(1.0 if row_metrics(right_by_id[x["id"]])["answer_in_prediction"] else 0.0 for x in pair_rows) / n,
        "right_insufficient": sum(1.0 if x["right_insufficient"] else 0.0 for x in pair_rows) / n,
        "fixed_by_right": len(fixed),
        "broken_by_right": len(broken),
        "both_correct": len(both_correct),
        "both_wrong": len(both_wrong),
        "retrieval_miss_both": sum(
            1
            for x in pair_rows
            if not x["left_answer_in_rendered_context"] and not x["right_answer_in_rendered_context"]
        ),
        "generation_fail_left_fixed_right": sum(
            1 for x in fixed if x["left_failure_category"] == "GENERATION_FAIL"
        ),
        "multi_anchor_fixed": sum(1 for x in fixed if x["left_failure_category"] == "MULTI_ANCHOR_FAIL"),
        "anchor_mismatch_fixed": sum(1 for x in fixed if x["left_failure_category"] == "ANCHOR_MISMATCH_FAIL"),
        "rendered_context_hash_mismatch_count": hash_mismatches,
        "prompt_only_comparison": hash_mismatches == 0,
    }
    stem = f"prompt_compare_{dataset}_{left_prompt}_{right_prompt}"
    dump_json({"summary": summary, "examples": pair_rows}, analysis_dir / f"{stem}.json")
    (analysis_dir / f"{stem}.md").write_text(markdown_compare(summary, pair_rows), encoding="utf-8")
    print(markdown_compare(summary, pair_rows[:20]))
    print(f"wrote: {analysis_dir}")
    return summary


def markdown_compare(summary: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Prompt Run Comparison",
        "",
        f"- dataset: {summary.get('dataset')}",
        f"- left: {summary.get('left_prompt_profile')} ({summary.get('left_path')})",
        f"- right: {summary.get('right_prompt_profile')} ({summary.get('right_path')})",
        f"- prompt_only_comparison: {summary.get('prompt_only_comparison')}",
    ]
    if not summary.get("prompt_only_comparison"):
        lines.append("- warning: rendered_context_hash differs for at least one shared id; this is not a prompt-only comparison.")
    metrics = [
        "n",
        "left_EM",
        "left_F1",
        "left_answer_in_prediction",
        "left_insufficient",
        "right_EM",
        "right_F1",
        "right_answer_in_prediction",
        "right_insufficient",
        "fixed_by_right",
        "broken_by_right",
        "both_correct",
        "both_wrong",
        "retrieval_miss_both",
        "generation_fail_left_fixed_right",
        "multi_anchor_fixed",
        "anchor_mismatch_fixed",
        "rendered_context_hash_mismatch_count",
    ]
    lines.extend(["", "| metric | value |", "|---|---:|"])
    for key in metrics:
        value = summary.get(key, 0)
        if isinstance(value, float):
            value = f"{value:.4f}"
        lines.append(f"| {key} | {value} |")
    changed = [x for x in rows if x.get("left_correct") != x.get("right_correct")]
    if changed:
        lines.extend(["", "## Changed Examples", ""])
        for row in changed[:20]:
            lines.extend(
                [
                    f"### {row.get('id')}",
                    "",
                    f"- question: {row.get('question')}",
                    f"- answers: {row.get('answers')}",
                    f"- left_correct: {row.get('left_correct')} ({row.get('left_failure_category')})",
                    f"- right_correct: {row.get('right_correct')} ({row.get('right_failure_category')})",
                    f"- left_prediction: {safe_truncate(str(row.get('left_prediction', '')), 240)}",
                    f"- right_prediction: {safe_truncate(str(row.get('right_prediction', '')), 240)}",
                    "",
                ]
            )
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two QMRAG prompt runs by id.")
    parser.add_argument("--left", default=None)
    parser.add_argument("--right", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--left-prompt", default=None)
    parser.add_argument("--right-prompt", default=None)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--analysis-root", default="outputs/analysis")
    parser.add_argument("--analysis-dir", default=None)
    args = parser.parse_args()

    if args.left:
        left_path = Path(args.left)
    else:
        left_path = find_latest_prediction(Path(args.output_root), args.dataset, args.left_prompt)
    if args.right:
        right_path = Path(args.right)
    else:
        right_path = find_latest_prediction(Path(args.output_root), args.dataset, args.right_prompt)
    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else Path(args.analysis_root) / now_timestamp()
    ensure_dir(analysis_dir)
    compare(left_path, right_path, analysis_dir)


if __name__ == "__main__":
    main()
