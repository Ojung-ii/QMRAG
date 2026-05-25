#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.io_utils import dump_json, ensure_dir


DISPLAY_DATASETS = {"2wiki": "2wikimultihopqa"}

CSV_COLUMNS = [
    "stage",
    "dataset",
    "prompt_variant",
    "prompt_setting",
    "top_k",
    "EM",
    "F1",
    "Recall@5",
    "avg_context_tokens",
    "avg_prompt_tokens",
    "F1_per_1k_prompt_tokens",
    "insufficient_information_rate",
    "empty_answer_rate",
    "retrieval_ms",
    "generation_ms",
    "total_ms",
    "output_path",
]


def fnum(value: Any) -> float | None:
    if value in {None, "", "NA"}:
        return None
    try:
        return float(value)
    except Exception:
        return None


def mean(values: Sequence[Any]) -> float | None:
    nums = [fnum(v) for v in values]
    nums = [x for x in nums if x is not None]
    return statistics.fmean(nums) if nums else None


def fmt_metric(value: Any, digits: int = 3) -> str:
    number = fnum(value)
    if number is None:
        return "NA"
    return f"{number:.{digits}f}"


def fmt_int(value: Any) -> str:
    number = fnum(value)
    if number is None:
        return "NA"
    return str(int(round(number)))


def display_dataset(dataset: Any) -> str:
    return DISPLAY_DATASETS.get(str(dataset), str(dataset))


def resolve_root(root: Path) -> Path:
    if root.exists() and root.is_file():
        text = root.read_text(encoding="utf-8").strip()
        return Path(text)
    if root.exists():
        if (root / "latest_run.txt").exists():
            latest = Path((root / "latest_run.txt").read_text(encoding="utf-8").strip())
            if latest.exists():
                return latest
        if any(root.glob("rag_summary.json")) or any(root.rglob("rag_summary.json")):
            return root
        dirs = [p for p in root.iterdir() if p.is_dir()]
        if dirs:
            return max(dirs, key=lambda p: p.stat().st_mtime)
    if root.name == "latest_or_timestamp" and root.parent.exists():
        dirs = [p for p in root.parent.iterdir() if p.is_dir()]
        if dirs:
            return max(dirs, key=lambda p: p.stat().st_mtime)
    raise SystemExit(f"No ACE-RAG native prompt ablation root found: {root}")


def stage_from_path(path: Path) -> str:
    parts = set(path.parts)
    if "stage1_prompt_top3" in parts:
        return "stage1_prompt_top3"
    if "stage2_context_scaling" in parts:
        return "stage2_context_scaling"
    if "stage0_smoke" in parts:
        return "stage0_smoke"
    return "unknown"


def row_from_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    f1 = fnum(data.get("F1"))
    prompt_tokens = fnum(data.get("avg_prompt_tokens", data.get("avg_prompt_input_tokens")))
    f1_per_prompt = data.get("F1_per_1k_prompt_tokens")
    if fnum(f1_per_prompt) is None and f1 is not None and prompt_tokens and prompt_tokens > 0:
        f1_per_prompt = f1 / (prompt_tokens / 1000.0)
    return {
        "stage": stage_from_path(path),
        "dataset": display_dataset(data.get("dataset", "UNKNOWN")),
        "prompt_variant": data.get("ace_native_prompt_variant") or "UNKNOWN",
        "prompt_setting": data.get("prompt_setting") or "UNKNOWN",
        "top_k": data.get("top_bundles") or data.get("qa_top_k") or "NA",
        "EM": data.get("EM"),
        "F1": data.get("F1"),
        "Recall@5": data.get("Recall@5"),
        "avg_context_tokens": data.get("avg_context_tokens"),
        "avg_prompt_tokens": data.get("avg_prompt_tokens", data.get("avg_prompt_input_tokens")),
        "F1_per_1k_prompt_tokens": f1_per_prompt,
        "insufficient_information_rate": data.get("insufficient_information_rate"),
        "empty_answer_rate": data.get("empty_answer_rate"),
        "retrieval_ms": data.get("retrieval_ms"),
        "generation_ms": data.get("generation_ms"),
        "total_ms": data.get("total_ms"),
        "output_path": data.get("output_path") or str(path),
    }


def load_rows(root: Path) -> list[dict[str, Any]]:
    return sorted([row_from_summary(path) for path in root.rglob("rag_summary.json")], key=row_sort_key)


def top_k_int(row: Mapping[str, Any]) -> int:
    try:
        return int(row.get("top_k") or 999)
    except Exception:
        return 999


def row_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    stage_order = {"stage0_smoke": 0, "stage1_prompt_top3": 1, "stage2_context_scaling": 2}
    variant_order = {
        "p0_current": 0,
        "p1_supporting_fallback": 1,
        "p2_relaxed_chain": 2,
        "p3_minimal_extraction": 3,
        "p4_fewshot_extraction": 4,
    }
    return (
        stage_order.get(str(row.get("stage")), 99),
        str(row.get("dataset")),
        variant_order.get(str(row.get("prompt_variant")), 99),
        top_k_int(row),
    )


def average_row(rows: Sequence[Mapping[str, Any]], *, stage: str, label_key: str, label_value: Any) -> dict[str, Any]:
    top_values = sorted({str(row.get("top_k")) for row in rows})
    variant_values = sorted({str(row.get("prompt_variant")) for row in rows})
    prompt_values = sorted({str(row.get("prompt_setting")) for row in rows})
    return {
        "stage": stage,
        "dataset": "average",
        "prompt_variant": label_value if label_key == "prompt_variant" else ";".join(variant_values),
        "prompt_setting": ";".join(prompt_values),
        "top_k": label_value if label_key == "top_k" else ";".join(top_values),
        "EM": mean([row.get("EM") for row in rows]),
        "F1": mean([row.get("F1") for row in rows]),
        "Recall@5": mean([row.get("Recall@5") for row in rows]),
        "avg_context_tokens": mean([row.get("avg_context_tokens") for row in rows]),
        "avg_prompt_tokens": mean([row.get("avg_prompt_tokens") for row in rows]),
        "F1_per_1k_prompt_tokens": mean([row.get("F1_per_1k_prompt_tokens") for row in rows]),
        "insufficient_information_rate": mean([row.get("insufficient_information_rate") for row in rows]),
        "empty_answer_rate": mean([row.get("empty_answer_rate") for row in rows]),
        "retrieval_ms": mean([row.get("retrieval_ms") for row in rows]),
        "generation_ms": mean([row.get("generation_ms") for row in rows]),
        "total_ms": mean([row.get("total_ms") for row in rows]),
        "output_path": "",
    }


def add_stage1_averages(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    stage_rows = sorted([row for row in rows if row.get("stage") == "stage1_prompt_top3"], key=row_sort_key)
    out = list(stage_rows)
    for variant in sorted({str(row.get("prompt_variant")) for row in stage_rows}):
        group = [row for row in stage_rows if str(row.get("prompt_variant")) == variant]
        out.append(average_row(group, stage="stage1_prompt_top3", label_key="prompt_variant", label_value=variant))
    return out


def add_stage2_averages(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    stage_rows = sorted([row for row in rows if row.get("stage") == "stage2_context_scaling"], key=row_sort_key)
    out = list(stage_rows)
    for top_k in sorted({str(row.get("top_k")) for row in stage_rows}, key=lambda x: int(x) if x.isdigit() else 999):
        group = [row for row in stage_rows if str(row.get("top_k")) == top_k]
        out.append(average_row(group, stage="stage2_context_scaling", label_key="top_k", label_value=top_k))
    return out


def best_prompt_variant(rows: Sequence[dict[str, Any]]) -> str | None:
    stage_rows = [row for row in rows if row.get("stage") == "stage1_prompt_top3" and row.get("dataset") != "average"]
    scores: list[tuple[float, str]] = []
    for variant in sorted({str(row.get("prompt_variant")) for row in stage_rows}):
        avg = mean([row.get("F1") for row in stage_rows if str(row.get("prompt_variant")) == variant])
        if avg is not None:
            scores.append((avg, variant))
    if not scores:
        return None
    scores.sort(reverse=True)
    return scores[0][1]


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "NA") for col in CSV_COLUMNS})


def markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:
    cols = [
        "dataset",
        "prompt_variant",
        "top_k",
        "EM",
        "F1",
        "Recall@5",
        "avg_context_tokens",
        "avg_prompt_tokens",
        "F1_per_1k_prompt_tokens",
        "insufficient_information_rate",
        "retrieval_ms",
        "generation_ms",
        "total_ms",
    ]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        values = []
        for col in cols:
            value = row.get(col, "NA")
            if col in {"avg_context_tokens", "avg_prompt_tokens", "retrieval_ms", "generation_ms", "total_ms"}:
                value = fmt_int(value)
            elif col in {"EM", "F1", "Recall@5", "F1_per_1k_prompt_tokens", "insufficient_information_rate"}:
                value = fmt_metric(value)
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def write_markdown(path: Path, root: Path, rows: Sequence[dict[str, Any]]) -> None:
    stage1 = add_stage1_averages(rows)
    stage2 = add_stage2_averages(rows)
    best = best_prompt_variant(rows)
    lines = [
        "# ACE-RAG Native Prompt Ablation",
        "",
        f"- root: `{root}`",
        f"- created_at: {datetime.now().isoformat(timespec='seconds')}",
        f"- best_prompt_variant_by_stage1_avg_F1: `{best or 'NA'}`",
        "",
        "## Table 1. Prompt Ablation, Top-3 Fixed",
        "",
        markdown_table(stage1),
        "",
        "## Table 2. Context Scaling For Best Prompt",
        "",
        markdown_table(stage2) if stage2 else "_No stage2 rows found._\n",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate ACE-RAG native prompt ablation runs.")
    parser.add_argument("--root", default="outputs/acerag_native_prompt_ablation")
    parser.add_argument("--print-best", action="store_true")
    args = parser.parse_args()

    root = resolve_root(Path(args.root))
    rows = load_rows(root)
    best = best_prompt_variant(rows)
    if args.print_best:
        if not best:
            raise SystemExit("No stage1 rows available to select best prompt.")
        print(best)
        return
    all_rows = rows + [row for row in add_stage1_averages(rows) if row.get("dataset") == "average"] + [row for row in add_stage2_averages(rows) if row.get("dataset") == "average"]
    write_csv(root / "prompt_ablation_summary.csv", all_rows)
    dump_json({"root": str(root), "best_prompt_variant": best, "rows": all_rows}, root / "prompt_ablation_summary.json")
    write_markdown(root / "prompt_ablation_summary.md", root, rows)
    print(f"root: {root}")
    print(f"best_prompt_variant: {best or 'NA'}")
    print(f"wrote: {root / 'prompt_ablation_summary.md'}")


if __name__ == "__main__":
    main()
