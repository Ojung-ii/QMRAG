#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_acerag_context_rendering import norm_contains
from utils.eval_metrics import answer_f1, exact_match, is_insufficient_prediction
from utils.io_utils import dump_json, read_jsonl


SUMMARY_COLUMNS = [
    "dataset",
    "prompt_variant",
    "renderer_variant",
    "top_k",
    "n",
    "EM",
    "F1",
    "Delta F1 vs p2",
    "avg_prompt_tokens",
    "F1_per_1k_prompt_tokens",
    "insufficient_information_rate",
    "answer_present_but_wrong_rate",
    "answer_present_but_insufficient_rate",
    "empty_answer_rate",
    "generation_ms",
    "total_ms",
    "predictions_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare HotpotQA F1 sprint outputs.")
    parser.add_argument("--root", default=None)
    parser.add_argument("--output-root", default="outputs/hotpotqa_f1_sprint")
    return parser.parse_args()


def fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def latest_root(output_root: str) -> Path:
    marker = Path(output_root) / "latest_run.txt"
    if marker.exists():
        return Path(marker.read_text(encoding="utf-8").strip())
    roots = sorted([p for p in Path(output_root).glob("*") if p.is_dir()])
    if not roots:
        raise SystemExit(f"No run roots found under {output_root}")
    return roots[-1]


def answers(row: Mapping[str, Any]) -> list[str]:
    vals = row.get("answers")
    if isinstance(vals, list) and vals:
        return [str(x) for x in vals if str(x).strip()]
    if row.get("gold_answer") is not None:
        return [str(row.get("gold_answer"))]
    return []


def diagnostic_rates(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    present_wrong = 0
    present_insuff = 0
    empty = 0
    for row in rows:
        golds = answers(row)
        pred = str(row.get("raw_prediction", row.get("prediction", "")) or "")
        context = str(row.get("rendered_context") or "")
        has_answer = norm_contains(context, golds)
        insuff = is_insufficient_prediction(pred)
        present_wrong += int(has_answer and exact_match(pred, golds) == 0.0 and not insuff)
        present_insuff += int(has_answer and insuff)
        empty += int(not pred.strip())
    n = max(1, len(rows))
    return {
        "answer_present_but_wrong_rate": present_wrong / n,
        "answer_present_but_insufficient_rate": present_insuff / n,
        "empty_answer_rate": empty / n,
    }


def load_summary(path: Path) -> dict[str, Any]:
    if any(part.startswith("context_") for part in path.parts):
        return {}
    rows = read_jsonl(path)
    if not rows:
        return {}
    if rows[0].get("no_llm"):
        return {}
    diag = diagnostic_rates(rows)
    dataset = str(rows[0].get("dataset") or "")
    variant = str(rows[0].get("ace_native_prompt_variant") or path.parent.name)
    renderer = str(rows[0].get("ace_renderer_variant") or "r0_current")
    top_k = rows[0].get("top_k") or rows[0].get("top_bundles") or 8
    ems = [exact_match(str(row.get("raw_prediction", row.get("prediction", "")) or ""), answers(row)) for row in rows]
    f1s = [answer_f1(str(row.get("raw_prediction", row.get("prediction", "")) or ""), answers(row)) for row in rows]
    prompt_tokens = [float(row.get("input_prompt_tokens") or row.get("prompt_tokens") or 0.0) for row in rows]
    generation_ms = [1000.0 * float(row.get("generation_latency_s") or 0.0) for row in rows]
    retrieval_ms = [1000.0 * float(((row.get("retrieval_diagnostics") or {}).get("timings") or {}).get("total_retrieval_s") or 0.0) for row in rows]
    f1 = mean(f1s)
    avg_prompt = mean(prompt_tokens)
    return {
        "dataset": dataset,
        "prompt_variant": variant,
        "renderer_variant": renderer,
        "top_k": top_k,
        "n": len(rows),
        "EM": mean(ems),
        "F1": f1,
        "avg_prompt_tokens": avg_prompt,
        "F1_per_1k_prompt_tokens": f1 / max(1e-9, avg_prompt / 1000.0),
        "insufficient_information_rate": sum(1 for row in rows if is_insufficient_prediction(str(row.get("raw_prediction", row.get("prediction", "")) or ""))) / len(rows),
        "generation_ms": mean(generation_ms),
        "total_ms": mean([a + b for a, b in zip(retrieval_ms, generation_ms)]),
        "predictions_path": str(path),
        **diag,
    }


def markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        vals = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                val = fmt(val)
            vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def case_key(row: Mapping[str, Any]) -> str:
    return str(row.get("id") or row.get("question") or "")


def write_cases(root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    pred_paths = [
        Path(str(row["predictions_path"]))
        for row in rows
        if row.get("predictions_path") and str(row.get("dataset") or "") == "hotpotqa"
    ]
    by_variant = {path.parent.name: {case_key(row): row for row in read_jsonl(path)} for path in pred_paths}
    if "p2_relaxed_chain" not in by_variant:
        return
    p2 = by_variant["p2_relaxed_chain"]
    lines = ["# HotpotQA F1 Sprint Cases", ""]

    def row_f1(row: Mapping[str, Any]) -> float:
        return answer_f1(str(row.get("raw_prediction", row.get("prediction", "")) or ""), answers(row))

    def row_insuff(row: Mapping[str, Any]) -> bool:
        return is_insufficient_prediction(str(row.get("raw_prediction", row.get("prediction", "")) or ""))

    categories = [
        ("p2 wrong but new prompt correct", lambda b, n: row_f1(b) < 0.5 and exact_match(str(n.get("raw_prediction", n.get("prediction", "")) or ""), answers(n)) > 0.0),
        ("p2 insufficient but new prompt correct", lambda b, n: row_insuff(b) and exact_match(str(n.get("raw_prediction", n.get("prediction", "")) or ""), answers(n)) > 0.0),
        ("p2 correct but new prompt wrong", lambda b, n: exact_match(str(b.get("raw_prediction", b.get("prediction", "")) or ""), answers(b)) > 0.0 and row_f1(n) < 0.5),
    ]
    for title, pred_fn in categories:
        lines.extend([f"## {title}", ""])
        count = 0
        for variant, mapping in by_variant.items():
            if variant == "p2_relaxed_chain":
                continue
            for key, base in p2.items():
                new = mapping.get(key)
                if not new or not pred_fn(base, new):
                    continue
                lines.extend(
                    [
                        f"### {variant} / {key}",
                        "",
                        f"Question: {new.get('question')}",
                        f"Gold: {answers(new)}",
                        f"p2 prediction: {base.get('prediction')}",
                        f"{variant} prediction: {new.get('prediction')}",
                        f"p2 F1: {fmt(row_f1(base))}",
                        f"{variant} F1: {fmt(row_f1(new))}",
                        "",
                        "Rendered context:",
                        "```text",
                        str(new.get("rendered_context") or "")[:6000],
                        "```",
                        "",
                    ]
                )
                count += 1
                if count >= 3:
                    break
            if count >= 3:
                break
        if count == 0:
            lines.append("_No examples found._\n")

    lines.extend(["## all prompts wrong despite answer present", ""])
    all_keys = set.intersection(*(set(mapping) for mapping in by_variant.values())) if by_variant else set()
    count = 0
    for key in sorted(all_keys):
        variants = [mapping[key] for mapping in by_variant.values()]
        if all(row_f1(row) < 0.5 for row in variants) and norm_contains(str(variants[0].get("rendered_context") or ""), answers(variants[0])):
            lines.extend(
                [
                    f"### {key}",
                    "",
                    f"Question: {variants[0].get('question')}",
                    f"Gold: {answers(variants[0])}",
                    "Predictions:",
                ]
            )
            for variant, mapping in by_variant.items():
                lines.append(f"- {variant}: {mapping[key].get('prediction')}")
            lines.extend(["", "```text", str(variants[0].get("rendered_context") or "")[:6000], "```", ""])
            count += 1
            if count >= 5:
                break
    if count == 0:
        lines.append("_No examples found._\n")
    (root / "hotpotqa_f1_sprint_cases.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.root) if args.root else latest_root(args.output_root)
    paths = sorted(root.glob("**/predictions.jsonl"))
    summaries = [load_summary(path) for path in paths]
    summaries = [row for row in summaries if row]
    p2_by_dataset = {
        str(row.get("dataset") or ""): float(row.get("F1") or 0.0)
        for row in summaries
        if row.get("prompt_variant") == "p2_relaxed_chain"
    }
    for row in summaries:
        dataset = str(row.get("dataset") or "")
        row["Delta F1 vs p2"] = float(row.get("F1") or 0.0) - p2_by_dataset.get(dataset, 0.0) if dataset in p2_by_dataset else ""
    summaries.sort(key=lambda row: (str(row.get("dataset") or ""), -float(row.get("F1") or 0.0)))
    with (root / "hotpotqa_f1_sprint_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for row in summaries:
            writer.writerow({col: row.get(col, "") for col in SUMMARY_COLUMNS})
    dump_json(summaries, root / "hotpotqa_f1_sprint_summary.json")
    hotpot_rows = [row for row in summaries if str(row.get("dataset") or "") == "hotpotqa"]
    best = max(hotpot_rows, key=lambda row: float(row.get("F1") or 0.0), default={})
    md = [
        "# HotpotQA F1 Sprint Summary",
        "",
        f"- root: `{root}`",
        f"- best_prompt: `{best.get('prompt_variant', '')}`",
        f"- best_F1: {fmt(best.get('F1', 0.0))}",
        f"- gap_to_0.61: {fmt(0.61 - float(best.get('F1') or 0.0))}",
        "",
        markdown_table(summaries, SUMMARY_COLUMNS),
    ]
    (root / "hotpotqa_f1_sprint_summary.md").write_text("\n".join(md), encoding="utf-8")
    write_cases(root, summaries)
    print(f"wrote {root / 'hotpotqa_f1_sprint_summary.md'}")


if __name__ == "__main__":
    main()
