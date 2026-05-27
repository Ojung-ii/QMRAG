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
    "setting_name",
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
    "evidence_bundles_hash_match_rate",
    "output_path",
]

SETTING_LABELS = {
    "current_native_compact": "Current chain-aware",
    "relaxed_native_compact": "Relaxed chain",
    "relaxed_native_scaled": "Relaxed chain + scaled context",
    "minimal_native_compact": "Minimal extraction",
}

SETTING_ORDER = {
    "current_native_compact": 0,
    "relaxed_native_compact": 1,
    "relaxed_native_scaled": 2,
    "minimal_native_compact": 3,
}


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
    if "core_n1000" in parts:
        return "core_n1000"
    if "appendix_4ds_p2_top8_n1000" in parts:
        return "appendix_4ds_p2_top8_n1000"
    if "custom" in parts:
        return "custom"
    if "stage1_prompt_top3" in parts:
        return "stage1_prompt_top3"
    if "stage2_context_scaling" in parts:
        return "stage2_context_scaling"
    if "stage0_smoke" in parts:
        return "stage0_smoke"
    return "unknown"


def setting_name_from(path: Path, data: Mapping[str, Any]) -> str:
    for part in reversed(path.parts):
        if part in SETTING_ORDER:
            return part
    variant = str(data.get("ace_native_prompt_variant") or "UNKNOWN")
    top_k = str(data.get("top_bundles") or data.get("qa_top_k") or "NA")
    if variant == "p0_current" and top_k == "3":
        return "current_native_compact"
    if variant == "p2_relaxed_chain" and top_k == "3":
        return "relaxed_native_compact"
    if variant == "p2_relaxed_chain" and top_k == "8":
        return "relaxed_native_scaled"
    if variant == "p3_minimal_extraction" and top_k == "3":
        return "minimal_native_compact"
    return f"{variant}_top{top_k}"


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
        "setting_name": setting_name_from(path, data),
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
        "evidence_bundles_hash_match_rate": data.get("evidence_bundles_hash_match_rate"),
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
    stage_order = {
        "stage0_smoke": 0,
        "custom": 1,
        "stage1_prompt_top3": 2,
        "stage2_context_scaling": 3,
        "core_n1000": 4,
        "appendix_4ds_p2_top8_n1000": 5,
    }
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
        SETTING_ORDER.get(str(row.get("setting_name")), 99),
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
        "setting_name": str(rows[0].get("setting_name")) if rows else "average",
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
        "evidence_bundles_hash_match_rate": mean([row.get("evidence_bundles_hash_match_rate") for row in rows]),
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


def final_core_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted([row for row in rows if row.get("stage") == "core_n1000"], key=row_sort_key)


def appendix_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted([row for row in rows if row.get("stage") == "appendix_4ds_p2_top8_n1000"], key=row_sort_key)


def average_by_setting(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for setting in sorted({str(row.get("setting_name")) for row in rows}, key=lambda x: SETTING_ORDER.get(x, 99)):
        group = [row for row in rows if str(row.get("setting_name")) == setting]
        if not group:
            continue
        avg = average_row(group, stage=str(group[0].get("stage", "average")), label_key="setting_name", label_value=setting)
        avg["setting_name"] = setting
        avg["prompt_variant"] = str(group[0].get("prompt_variant"))
        avg["top_k"] = str(group[0].get("top_k"))
        out.append(avg)
    return out


def core_table_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    cols = [
        "dataset",
        "setting_name",
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
        "evidence_bundles_hash_match_rate",
    ]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        vals = []
        for col in cols:
            value = row.get(col, "NA")
            if col in {"avg_context_tokens", "avg_prompt_tokens", "retrieval_ms", "generation_ms", "total_ms"}:
                value = fmt_int(value)
            elif col in {"EM", "F1", "Recall@5", "F1_per_1k_prompt_tokens", "insufficient_information_rate", "evidence_bundles_hash_match_rate"}:
                value = fmt_metric(value)
            vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def prompt_improvement_rows(core_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    avg_rows = average_by_setting(core_rows)
    baseline = next((row for row in avg_rows if row.get("setting_name") == "current_native_compact"), None)
    baseline_f1 = fnum(baseline.get("F1")) if baseline else None
    out = []
    for row in avg_rows:
        f1 = fnum(row.get("F1"))
        delta = None if baseline_f1 is None or f1 is None else f1 - baseline_f1
        out.append(
            {
                "setting": f"{row.get('prompt_variant')}_top{row.get('top_k')}",
                "setting_name": row.get("setting_name"),
                "prompt_variant": row.get("prompt_variant"),
                "top_k": row.get("top_k"),
                "Avg EM": row.get("EM"),
                "Avg F1": row.get("F1"),
                "Delta F1 vs p0_top3": delta,
                "Avg prompt tokens": row.get("avg_prompt_tokens"),
                "F1/1K prompt": row.get("F1_per_1k_prompt_tokens"),
                "insuff": row.get("insufficient_information_rate"),
            }
        )
    return out


def prompt_improvement_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    cols = ["setting", "prompt_variant", "top_k", "Avg EM", "Avg F1", "Delta F1 vs p0_top3", "Avg prompt tokens", "F1/1K prompt", "insuff"]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        vals = []
        for col in cols:
            value = row.get(col, "NA")
            if col == "Avg prompt tokens":
                value = fmt_int(value)
            elif col in {"Avg EM", "Avg F1", "Delta F1 vs p0_top3", "F1/1K prompt", "insuff"}:
                value = fmt_metric(value)
            vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def appendix_table_markdown(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "_No four-dataset appendix rows found._\n"
    by_dataset = {str(row.get("dataset")): row for row in rows}
    order = ["hotpotqa", "2wikimultihopqa", "musique", "popqa"]
    labels = {
        "hotpotqa": "HotpotQA EM/F1 Tok.",
        "2wikimultihopqa": "2Wiki EM/F1 Tok.",
        "musique": "MuSiQue EM/F1 Tok.",
        "popqa": "PopQA EM/F1 Tok.",
    }
    lines = ["| Method | " + " | ".join(labels[x] for x in order) + " |", "| " + " | ".join(["---"] * (len(order) + 1)) + " |"]
    vals = []
    for dataset in order:
        row = by_dataset.get(dataset)
        if not row:
            vals.append("NA")
        else:
            vals.append(f"{fmt_metric(row.get('EM'))}/{fmt_metric(row.get('F1'))} {fmt_int(row.get('avg_prompt_tokens'))}")
    lines.append("| ACE-RAG native (p2 top-8) | " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def latex_sensitivity_table(rows: Sequence[Mapping[str, Any]]) -> str:
    avg_rows = average_by_setting(rows)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Setting & Top-$k$ & EM & F1 & Prompt Tok. & F1/1K & Insuff. \\\\",
        "\\midrule",
    ]
    for row in avg_rows:
        label = SETTING_LABELS.get(str(row.get("setting_name")), str(row.get("setting_name")))
        lines.append(
            f"{label} & {row.get('top_k')} & {fmt_metric(row.get('EM'))} & {fmt_metric(row.get('F1'))} & "
            f"{fmt_int(row.get('avg_prompt_tokens'))} & {fmt_metric(row.get('F1_per_1k_prompt_tokens'))} & "
            f"{fmt_metric(row.get('insufficient_information_rate'))} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{ACE-RAG native-prompt sensitivity on HotpotQA and 2Wiki. Relaxing chain-position assumptions and allowing Supporting Evidence as direct answer evidence improve native QA performance. Increasing the rendered evidence bundles further improves absolute F1, while compact prompts remain more token-efficient.}",
            "\\label{tab:ace_rag_native_prompt_sensitivity}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def latex_appendix_table(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return ""
    by_dataset = {str(row.get("dataset")): row for row in rows}
    order = [("hotpotqa", "HotpotQA"), ("2wikimultihopqa", "2Wiki"), ("musique", "MuSiQue"), ("popqa", "PopQA")]
    cells = []
    for dataset, _label in order:
        row = by_dataset.get(dataset)
        cells.append("NA" if not row else f"{fmt_metric(row.get('EM'))}/{fmt_metric(row.get('F1'))} {fmt_int(row.get('avg_prompt_tokens'))}")
    return "\n".join(
        [
            "\\begin{table*}[t]",
            "\\centering",
            "\\small",
            "\\begin{tabular}{lcccc}",
            "\\toprule",
            "Method & HotpotQA EM/F1 Tok. & 2Wiki EM/F1 Tok. & MuSiQue EM/F1 Tok. & PopQA EM/F1 Tok. \\\\",
            "\\midrule",
            "ACE-RAG native (p2 top-8) & " + " & ".join(cells) + " \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Native-prompt results with method-specific QA prompts. ACE-RAG uses the relaxed chain prompt with top-8 rendered evidence bundles. Tok. denotes average prompt tokens, including retrieved context and prompt text.}",
            "\\label{tab:ace_rag_native_appendix_updated}",
            "\\end{table*}",
            "",
        ]
    )


def interpretation_markdown(core_rows: Sequence[Mapping[str, Any]]) -> str:
    avg = {str(row.get("setting_name")): row for row in average_by_setting(core_rows)}
    p0 = avg.get("current_native_compact", {})
    p2_top3 = avg.get("relaxed_native_compact", {})
    p2_top8 = avg.get("relaxed_native_scaled", {})
    p3 = avg.get("minimal_native_compact", {})
    p0_f1, p2_f1, p28_f1, p3_eff, p2_eff = (
        fnum(p0.get("F1")),
        fnum(p2_top3.get("F1")),
        fnum(p2_top8.get("F1")),
        fnum(p3.get("F1_per_1k_prompt_tokens")),
        fnum(p2_top3.get("F1_per_1k_prompt_tokens")),
    )
    p0_ins, p2_ins = fnum(p0.get("insufficient_information_rate")), fnum(p2_top3.get("insufficient_information_rate"))
    supports = {
        "p2_over_p0": p2_f1 is not None and p0_f1 is not None and p2_f1 > p0_f1,
        "top8_over_top3": p28_f1 is not None and p2_f1 is not None and p28_f1 > p2_f1,
        "p3_more_efficient": p3_eff is not None and p2_eff is not None and p3_eff > p2_eff,
        "insuff_decreased": p2_ins is not None and p0_ins is not None and p2_ins < p0_ins,
    }
    if all(supports.values()):
        conclusion = (
            "The n=1000 verification confirms that the original ACE-RAG native prompt underestimated ACE-RAG because it was too restrictive about Supporting Evidence and relied on rigid chain-position assumptions. "
            "The relaxed chain prompt improves native F1, and increasing rendered evidence bundles to top-8 further improves absolute native performance. "
            "However, the common-prompt setting should remain the main comparison because it isolates retrieved-context quality across methods. "
            "The native results should be reported as prompt/context sensitivity analysis in the appendix."
        )
    else:
        conclusion = (
            "The n=1000 verification gives a mixed result. The native appendix setting should be chosen from the measured table rather than assumed from n=100."
        )
    lines = [
        "# Final ACE-RAG Native Verification Interpretation",
        "",
        f"- Did p2 top-3 improve over p0 top-3 at n=1000? {supports['p2_over_p0']} ({fmt_metric(p0_f1)} -> {fmt_metric(p2_f1)})",
        f"- Did p2 top-8 improve over p2 top-3 at n=1000? {supports['top8_over_top3']} ({fmt_metric(p2_f1)} -> {fmt_metric(p28_f1)})",
        f"- Did p3 top-3 remain more token-efficient than p2? {supports['p3_more_efficient']} ({fmt_metric(p2_eff)} -> {fmt_metric(p3_eff)})",
        f"- Did insufficient-information rate decrease after prompt relaxation? {supports['insuff_decreased']} ({fmt_metric(p0_ins)} -> {fmt_metric(p2_ins)})",
        f"- Is top-8 still better than top-3 for native absolute F1? {supports['top8_over_top3']}",
        "- Should the paper keep common-prompt main results? Yes.",
        "- Should native results be placed in appendix as prompt/context sensitivity? Yes.",
        "",
        "## Conclusion",
        "",
        conclusion,
        "",
    ]
    return "\n".join(lines)


def write_final_reports(root: Path, rows: Sequence[dict[str, Any]]) -> None:
    core = final_core_rows(rows)
    appendix = appendix_rows(rows)
    if not core and not appendix:
        return
    final_rows = list(core) + average_by_setting(core) + list(appendix)
    write_csv(root / "final_native_verification_summary.csv", final_rows)
    dump_json({"root": str(root), "core_rows": core, "appendix_rows": appendix, "average_core_rows": average_by_setting(core)}, root / "final_native_verification_summary.json")
    summary_lines = [
        "# Final ACE-RAG Native Verification Summary",
        "",
        f"- root: `{root}`",
        f"- created_at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Table A. Core n=1000 Prompt/Context Verification",
        "",
        core_table_markdown(list(core) + average_by_setting(core)) if core else "_No core rows found._\n",
        "",
        "## Table B. Prompt Improvement Summary",
        "",
        prompt_improvement_markdown(prompt_improvement_rows(core)) if core else "_No core rows found._\n",
        "",
        "## Table C. Optional Four-Dataset Native Appendix Table",
        "",
        appendix_table_markdown(appendix),
        "",
    ]
    (root / "final_native_verification_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")
    (root / "final_native_core_table.md").write_text(core_table_markdown(list(core) + average_by_setting(core)), encoding="utf-8")
    (root / "final_native_prompt_improvement_table.md").write_text(prompt_improvement_markdown(prompt_improvement_rows(core)), encoding="utf-8")
    (root / "final_native_appendix_table.md").write_text(appendix_table_markdown(appendix), encoding="utf-8")
    (root / "final_native_prompt_context_sensitivity.tex").write_text(latex_sensitivity_table(core), encoding="utf-8")
    if appendix:
        (root / "final_native_appendix_table.tex").write_text(latex_appendix_table(appendix), encoding="utf-8")
    (root / "final_native_verification_interpretation.md").write_text(interpretation_markdown(core), encoding="utf-8")


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
    parser.add_argument("--root", default="outputs/ace_rag_native_prompt_ablation")
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
    write_final_reports(root, rows)
    print(f"root: {root}")
    print(f"best_prompt_variant: {best or 'NA'}")
    print(f"wrote: {root / 'prompt_ablation_summary.md'}")


if __name__ == "__main__":
    main()
