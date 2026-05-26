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
from utils.eval_metrics import answer_contains, evaluate_predictions, exact_match
from utils.generation import is_insufficient_prediction
from utils.io_utils import dump_json, ensure_dir, read_jsonl


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def iter_prediction_files(output_root: Path) -> Iterable[Path]:
    yield from sorted(output_root.rglob("predictions.jsonl"))


def find_latest_full_context(output_root: Path, dataset: str, prompt_profile: str) -> Path:
    candidates = []
    for path in iter_prediction_files(output_root):
        try:
            rows = read_jsonl(path)
        except Exception:
            continue
        if not rows:
            continue
        first = rows[0]
        if infer_dataset(path, rows) != dataset:
            continue
        if infer_prompt(rows) != prompt_profile:
            continue
        if infer_rendering(rows) != "structured_chain":
            continue
        if first.get("context_truncation_enabled") or first.get("top_bundles") is not None or first.get("context_token_budget") is not None:
            continue
        if str(first.get("compaction_profile") or "none") != "none":
            continue
        candidates.append((path.stat().st_mtime, str(path), path))
    if not candidates:
        raise SystemExit(f"No full structured_chain source found for dataset={dataset!r} prompt_profile={prompt_profile!r}")
    return max(candidates)[2]


def correct(row: Mapping[str, Any]) -> bool:
    answers = [str(x) for x in row.get("answers", [])]
    raw = str(row.get("raw_prediction", row.get("prediction", "")) or "")
    return bool(exact_match(raw, answers) or answer_contains(raw, answers))


def avg(values: list[float]) -> float:
    return sum(values) / max(1, len(values))


def compare(left_path: Path, right_path: Path, analysis_dir: Path, dataset: str, prompt_profile: str) -> dict[str, Any]:
    left_all = read_jsonl(left_path)
    right_all = read_jsonl(right_path)
    left_by_id = {str(row.get("id")): row for row in left_all}
    right_by_id = {str(row.get("id")): row for row in right_all}
    ids = [qid for qid in left_by_id if qid in right_by_id]
    left_rows = [left_by_id[qid] for qid in ids]
    right_rows = [right_by_id[qid] for qid in ids]
    left_eval = evaluate_predictions(left_rows, dataset=dataset, prompt_profile=prompt_profile)
    right_eval = evaluate_predictions(right_rows, dataset=dataset, prompt_profile=prompt_profile)
    pairs = []
    for qid in ids:
        left = left_by_id[qid]
        right = right_by_id[qid]
        pairs.append(
            {
                "id": qid,
                "left_correct": correct(left),
                "right_correct": correct(right),
                "left_prediction": left.get("raw_prediction", left.get("prediction", "")),
                "right_prediction": right.get("raw_prediction", right.get("prediction", "")),
            }
        )
    fixed = [row for row in pairs if not row["left_correct"] and row["right_correct"]]
    broken = [row for row in pairs if row["left_correct"] and not row["right_correct"]]
    both_correct = [row for row in pairs if row["left_correct"] and row["right_correct"]]
    both_wrong = [row for row in pairs if not row["left_correct"] and not row["right_correct"]]
    left_input = float(left_eval.get("avg_input_prompt_tokens", 0.0) or 0.0)
    right_input = float(right_eval.get("avg_input_prompt_tokens", 0.0) or 0.0)
    left_context = float(left_eval.get("avg_rendered_context_tokens", left_eval.get("context_tokens", 0.0)) or 0.0)
    right_context = float(right_eval.get("avg_rendered_context_tokens", right_eval.get("context_tokens", 0.0)) or 0.0)
    right_first = right_rows[0] if right_rows else {}
    summary = {
        "dataset": dataset,
        "prompt_profile": prompt_profile,
        "left_path": str(left_path),
        "right_path": str(right_path),
        "compaction_profile": right_first.get("compaction_profile", "none"),
        "max_sentences_per_bundle": right_first.get("max_sentences_per_bundle"),
        "top_bundles": right_first.get("top_bundles"),
        "n": len(pairs),
        "left_EM": left_eval.get("em", 0.0),
        "left_F1": left_eval.get("f1", 0.0),
        "left_answer_in_prediction": left_eval.get("answer_in_prediction", 0.0),
        "left_answer_in_rendered_context": left_eval.get("answer_in_rendered_context", 0.0),
        "left_insufficient_rate": left_eval.get("insufficient_rate", 0.0),
        "left_CtxTok": left_context,
        "left_InputTok": left_input,
        "left_TotalTok": left_eval.get("avg_total_llm_tokens", 0.0),
        "right_EM": right_eval.get("em", 0.0),
        "right_F1": right_eval.get("f1", 0.0),
        "right_answer_in_prediction": right_eval.get("answer_in_prediction", 0.0),
        "right_answer_in_rendered_context": right_eval.get("answer_in_rendered_context", 0.0),
        "right_insufficient_rate": right_eval.get("insufficient_rate", 0.0),
        "right_CtxTok": right_context,
        "right_InputTok": right_input,
        "right_TotalTok": right_eval.get("avg_total_llm_tokens", 0.0),
        "right_F1_per_1k_input_prompt_tokens": right_eval.get("F1_per_1k_input_prompt_tokens", 0.0),
        "context_token_reduction_rate": 1.0 - right_context / max(1e-9, left_context),
        "token_reduction_rate": 1.0 - right_input / max(1e-9, left_input),
        "avg_rendered_sentence_count": right_eval.get("avg_rendered_sentence_count", 0.0),
        "avg_sentences_per_bundle": right_eval.get("avg_sentences_per_bundle", 0.0),
        "avg_dropped_sentence_count": right_eval.get("avg_dropped_sentence_count", 0.0),
        "avg_duplicate_removed_count": right_eval.get("avg_duplicate_removed_count", 0.0),
        "avg_source_removed_count": right_eval.get("avg_source_removed_count", 0.0),
        "avg_metadata_removed_count": right_eval.get("avg_metadata_removed_count", 0.0),
        "fixed_by_right": len(fixed),
        "broken_by_right": len(broken),
        "both_correct": len(both_correct),
        "both_wrong": len(both_wrong),
        "right_insufficient_count": sum(1 for row in right_rows if is_insufficient_prediction(row.get("raw_prediction", row.get("prediction", "")))),
        "evidence_bundles_hash_match_rate": avg([1.0 if row.get("evidence_bundles_hash_match", True) else 0.0 for row in right_rows]),
    }
    stem = f"compaction_compare_{dataset}_{prompt_profile}_{summary['compaction_profile']}"
    dump_json({"summary": summary, "examples": pairs}, analysis_dir / f"{stem}.json")
    (analysis_dir / f"{stem}.md").write_text(markdown(summary), encoding="utf-8")
    print(markdown(summary))
    print(f"wrote: {analysis_dir}")
    return summary


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown(summary: Mapping[str, Any]) -> str:
    keys = [
        "n",
        "left_EM",
        "left_F1",
        "left_answer_in_prediction",
        "left_answer_in_rendered_context",
        "left_insufficient_rate",
        "left_CtxTok",
        "left_InputTok",
        "right_EM",
        "right_F1",
        "right_answer_in_prediction",
        "right_answer_in_rendered_context",
        "right_insufficient_rate",
        "right_CtxTok",
        "right_InputTok",
        "right_TotalTok",
        "right_F1_per_1k_input_prompt_tokens",
        "context_token_reduction_rate",
        "token_reduction_rate",
        "avg_rendered_sentence_count",
        "avg_sentences_per_bundle",
        "avg_dropped_sentence_count",
        "avg_duplicate_removed_count",
        "avg_source_removed_count",
        "avg_metadata_removed_count",
        "fixed_by_right",
        "broken_by_right",
        "both_correct",
        "both_wrong",
        "evidence_bundles_hash_match_rate",
    ]
    lines = [
        "# Chain-Aware Context Compaction Comparison",
        "",
        f"- dataset: {summary.get('dataset')}",
        f"- prompt_profile: {summary.get('prompt_profile')}",
        f"- compaction_profile: {summary.get('compaction_profile')}",
        f"- max_sentences_per_bundle: {summary.get('max_sentences_per_bundle')}",
        f"- top_bundles: {summary.get('top_bundles')}",
        f"- left: {summary.get('left_path')}",
        f"- right: {summary.get('right_path')}",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in keys:
        lines.append(f"| {key} | {fmt(summary.get(key, 0))} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare full structured_chain and compacted context replay runs.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--prompt-profile", default="common_qa")
    parser.add_argument("--left-full", default="latest")
    parser.add_argument("--right-compact", required=True)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--analysis-root", default="outputs/analysis")
    parser.add_argument("--analysis-dir", default=None)
    args = parser.parse_args()

    left_path = find_latest_full_context(Path(args.output_root), args.dataset, args.prompt_profile) if args.left_full == "latest" else Path(args.left_full)
    right_path = Path(args.right_compact)
    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else Path(args.analysis_root) / now_timestamp()
    ensure_dir(analysis_dir)
    compare(left_path, right_path, analysis_dir, args.dataset, args.prompt_profile)


if __name__ == "__main__":
    main()
