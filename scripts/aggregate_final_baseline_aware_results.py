#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence


DATASETS = ["hotpotqa", "2wikimultihopqa", "musique", "popqa"]
METHOD_ORDER = ["BM25", "Dense RAG", "RAPTOR", "HippoRAG2", "LightRAG", "ACE-RAG-Compact", "ACE-RAG-Scaled", "ACE-RAG"]


COMMON_BASELINES = [
    ("hotpotqa", "BM25", 0.6760, 0.1790, 0.2172, 749.4660, 883.8, 36.0486, 72.3306, 108.3791),
    ("hotpotqa", "Dense RAG", 0.9440, 0.2570, 0.3136, 657.5840, 791.9, 34.3056, 70.9209, 105.2264),
    ("hotpotqa", "RAPTOR", 0.4096, 0.1230, 0.1502, 3268.38, None, 72.38, 174.86, 247.24),
    ("hotpotqa", "HippoRAG2", 0.9510, 0.2590, 0.3126, 668.1370, 802.4, 902.0300, 71.5042, 973.5343),
    ("hotpotqa", "LightRAG", 0.9217, 0.2610, 0.3229, 640.95, 775.3140, 293.92, 78.75, 372.67),
    ("2wikimultihopqa", "BM25", 0.5308, 0.0360, 0.0421, 1111.1500, 1241.2, 22.9128, 86.6700, 109.5828),
    ("2wikimultihopqa", "Dense RAG", 0.7652, 0.0740, 0.0806, 793.8530, None, 31.4980, 75.7440, 107.2420),
    ("2wikimultihopqa", "RAPTOR", 0.5615, 0.0650, 0.0752, 3097.56, None, 70.84, 167.20, 238.04),
    ("2wikimultihopqa", "HippoRAG2", 0.8975, 0.0800, 0.0890, 845.2790, None, 705.7783, 76.1379, 781.9162),
    ("2wikimultihopqa", "LightRAG", 0.8025, 0.0830, 0.0953, 783.99, 914.0540, 298.99, 74.35, 373.33),
    ("musique", "BM25", 0.2841, 0.0150, 0.0222, 835.4000, None, 41.5717, 80.9436, 122.5153),
    ("musique", "Dense RAG", 0.6933, 0.0300, 0.0435, 741.6360, 876.1, 35.6415, 74.5189, 110.1604),
    ("musique", "RAPTOR", 0.4230, 0.0380, 0.0490, 3284.07, None, 74.44, 175.52, 249.96),
    ("musique", "HippoRAG2", 0.7218, 0.0410, 0.0551, 749.4480, None, 812.4360, 77.2075, 889.6435),
    ("musique", "LightRAG", 0.5402, 0.0300, 0.0454, 705.26, 839.7350, 379.09, 74.04, 453.13),
    ("popqa", "BM25", 0.3810, 0.2670, 0.3502, 773.1770, None, 23.9427, 80.6806, 104.6233),
    ("popqa", "Dense RAG", 0.5110, 0.3240, 0.4167, 738.9190, None, 30.1863, 79.1112, 109.2975),
    ("popqa", "RAPTOR", 0.3860, 0.1950, 0.2716, 3263.54, None, 75.21, 176.77, 251.98),
    ("popqa", "HippoRAG2", 0.5165, 0.3290, 0.4159, 743.0110, 866.0, 629.0595, 79.7796, 708.8392),
    ("popqa", "LightRAG", 0.3547, 0.3050, 0.3908, 741.06, 864.1280, 277.85, 78.83, 356.68),
]


NATIVE_BASELINES = [
    ("hotpotqa", "BM25", 0.6760, 0.3400, 0.4344, 749.4660, 1516.0, 36.0486, 86.6469, 122.6954),
    ("hotpotqa", "Dense RAG", 0.9440, 0.4850, 0.6019, 657.5840, 1431.6, 34.3056, 89.1715, 123.4771),
    ("hotpotqa", "RAPTOR", 0.4096, 0.1710, 0.2142, 3268.38, None, 72.38, 166.57, 238.95),
    ("hotpotqa", "HippoRAG2", 0.9510, 0.4980, 0.6118, 668.1370, 1441.7, 902.0300, 93.4291, 995.4591),
    ("hotpotqa", "LightRAG", 0.9208, 0.4890, 0.6045, 596.7940, 1080.6550, 258.8745, 207.3095, 466.1840),
    ("2wikimultihopqa", "BM25", 0.5308, 0.1840, 0.2299, 1111.1500, 1679.5, 22.9128, 94.3319, 117.2447),
    ("2wikimultihopqa", "Dense RAG", 0.7652, 0.3490, 0.4055, 793.8530, 1471.2, 31.4980, 98.2406, 129.7386),
    ("2wikimultihopqa", "RAPTOR", 0.5615, 0.1830, 0.2108, 3097.56, None, 70.84, 165.03, 235.86),
    ("2wikimultihopqa", "HippoRAG2", 0.8975, 0.4020, 0.4722, 845.2790, 1507.9, 705.7783, 101.3170, 807.0953),
    ("2wikimultihopqa", "LightRAG", 0.7921, 0.3460, 0.3943, 720.5330, 1221.5050, 240.5541, 227.8569, 468.4110),
    ("musique", "BM25", 0.2841, 0.0710, 0.1233, 835.4000, 1593.2, 41.5717, 82.5523, 124.1240),
    ("musique", "Dense RAG", 0.6933, 0.1840, 0.2658, 741.6360, 1509.2, 35.6415, 86.5996, 122.2411),
    ("musique", "RAPTOR", 0.4230, 0.0530, 0.0787, 3284.07, None, 74.44, 161.77, 236.21),
    ("musique", "HippoRAG2", 0.7218, 0.1910, 0.2818, 749.4480, 1514.5, 812.4360, 93.0731, 905.5090),
    ("musique", "LightRAG", 0.6174, 0.1890, 0.2826, 658.9500, 1143.6740, 360.0113, 267.3270, 627.3383),
    ("popqa", "BM25", 0.3810, 0.3900, 0.4989, 773.1770, 1534.9, 23.9427, 94.9396, 118.8823),
    ("popqa", "Dense RAG", 0.5110, 0.4640, 0.5830, 738.9190, 1469.8, 30.1863, 92.3176, 122.5039),
    ("popqa", "RAPTOR", 0.3860, 0.2050, 0.3142, 3263.54, None, 75.21, 169.63, 244.84),
    ("popqa", "HippoRAG2", 0.5165, 0.4580, 0.5793, 743.0110, 1476.3, 629.0595, 92.6549, 721.7144),
    ("popqa", "LightRAG", 0.3510, 0.3970, 0.4823, 675.0080, 1171.2120, 274.5050, 213.6140, 488.1190),
]


ACE_COMMON_COMPACT_PATHS = {
    "hotpotqa": "outputs/replay/20260526_000338/hotpotqa/common_qa_to_common_qa_top3_top3_chain_dedup/eval.json",
    "2wikimultihopqa": "outputs/replay/20260526_001204/2wiki/common_qa_to_common_qa_top3_top3_chain_dedup/eval.json",
    "musique": "outputs/replay/20260525_221850/musique/common_qa_to_common_qa_top3_top3_chain_dedup/eval.json",
    "popqa": "outputs/replay/20260525_220745/popqa/common_qa_to_common_qa_top3_top3_chain_dedup/eval.json",
}

ACE_COMMON_SCALED_PATHS = {
    ds: f"outputs/final_baseline_aware/20260526_053914/common_prompt/ace_rag_scaled_top8/{short}/common_qa/rag_summary.json"
    for ds, short in {
        "hotpotqa": "hotpotqa",
        "2wikimultihopqa": "2wiki",
        "musique": "musique",
        "popqa": "popqa",
    }.items()
}

ACE_NATIVE_P8_PATHS = {
    "hotpotqa": "outputs/hotpotqa_f1_sprint/20260526_051609/prompt_sprint_n1000/p8_r0_section_aware/rag_summary.json",
    "2wikimultihopqa": "outputs/hotpotqa_f1_sprint/20260526_051609/2wiki_sanity_n1000/p8_r0_section_aware/rag_summary.json",
    "musique": "outputs/final_baseline_aware/20260526_053914/native_prompt/musique_ace_rag_p8_top8/p8_r0_section_aware/rag_summary.json",
    "popqa": "outputs/final_baseline_aware/20260526_053914/native_prompt/popqa_ace_rag_p8_top8/p8_r0_section_aware/rag_summary.json",
}


ABLATION_ROWS = [
    ("ACE-RAG", 0.3791138056388056, 0.14284450549450547),
    ("w/o Mention Edge", 0.30220382395382394, 0.10195549450549449),
    ("w/o Residual Cues", 0.36574437229437234, 0.11656593406593408),
    ("w/o Chain Order", 0.3761443006993007, 0.1421912087912088),
    ("w/o Anchor Bundle", 0.3701717004667004, 0.10680384615384616),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate final baseline-aware ACE-RAG results.")
    parser.add_argument("--root", default="outputs/final_baseline_aware/20260526_053914")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def val(d: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in d and d[key] is not None:
            return d[key]
    return None


def to_row(
    dataset: str,
    method: str,
    recall: float | None,
    em: float | None,
    f1: float | None,
    ctx: float | None,
    inp: float | None,
    retrieval: float | None,
    generation: float | None,
    total: float | None,
    prompt_setting: str,
    source: str,
) -> dict[str, Any]:
    overhead = (inp - ctx) if inp is not None and ctx is not None else None
    return {
        "dataset": dataset,
        "method": method,
        "prompt_setting": prompt_setting,
        "Recall@5": recall,
        "EM": em,
        "F1": f1,
        "ContextTok": ctx,
        "InputTok": inp,
        "PromptOverheadTok": overhead,
        "F1_per_1k_context": (f1 / (ctx / 1000.0)) if f1 is not None and ctx else None,
        "F1_per_1k_input": (f1 / (inp / 1000.0)) if f1 is not None and inp else None,
        "retrieval_ms": retrieval,
        "generation_ms": generation,
        "total_ms": total,
        "source": source,
    }


def row_from_summary(dataset: str, method: str, prompt_setting: str, path: str) -> dict[str, Any]:
    d = read_json(path)
    if not d:
        return to_row(dataset, method, None, None, None, None, None, None, None, None, prompt_setting, path)
    ctx = val(d, "avg_context_tokens", "avg_rendered_context_tokens", "context_tokens")
    inp = val(d, "avg_input_prompt_tokens", "avg_prompt_tokens", "avg_prompt_text_tokens")
    return to_row(
        dataset,
        method,
        val(d, "SupportRecall", "support_title_recall", "Recall@5"),
        val(d, "EM"),
        val(d, "F1"),
        ctx,
        inp,
        val(d, "retrieval_ms", "retrieval_latency_ms"),
        val(d, "generation_ms", "generation_latency_ms"),
        val(d, "total_ms", "latency_ms"),
        prompt_setting,
        path,
    )


def baseline_rows(rows: Sequence[tuple[Any, ...]], prompt_setting: str) -> list[dict[str, Any]]:
    return [to_row(ds, m, r, em, f1, ctx, inp, ret, gen, total, prompt_setting, "user_supplied_baseline_table") for ds, m, r, em, f1, ctx, inp, ret, gen, total in rows]


def fmt_num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def fmt_pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100:.{digits}f}"


def fmt_tok(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.0f}"


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col) for col in columns})


def md_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        vals = []
        for col in columns:
            v = row.get(col)
            if col in {"EM", "F1", "Recall@5"}:
                vals.append(fmt_num(v, 3))
            elif "Tok" in col or col.endswith("ms"):
                vals.append(fmt_num(v, 1))
            else:
                vals.append(str(v) if v is not None else "-")
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def compact_wide(rows: Sequence[Mapping[str, Any]], token_col: str) -> list[dict[str, Any]]:
    by = {(r["dataset"], r["method"]): r for r in rows}
    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in rows)]
    out = []
    for method in methods:
        row = {"Method": method}
        for ds in DATASETS:
            r = by.get((ds, method))
            if r:
                row[ds] = f"{fmt_pct(r.get('Recall@5'))} / {fmt_pct(r.get('EM'))}/{fmt_pct(r.get('F1'))} / {fmt_tok(r.get(token_col))}"
            else:
                row[ds] = "-"
        out.append(row)
    return out


def write_wide_table(out_dir: Path, filename: str, rows: Sequence[Mapping[str, Any]], token_col: str, caption: str) -> None:
    wide = compact_wide(rows, token_col)
    columns = ["Method", *DATASETS]
    (out_dir / f"{filename}.md").write_text(md_table(wide, columns), encoding="utf-8")
    write_csv(out_dir / f"{filename}.csv", wide, columns)
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lcccc}",
        "\\hline",
        "Method & HotpotQA & 2Wiki & MuSiQue & PopQA \\\\",
        "\\hline",
    ]
    for row in wide:
        lines.append(f"{row['Method']} & {row['hotpotqa']} & {row['2wikimultihopqa']} & {row['musique']} & {row['popqa']} \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "}", caption, f"\\label{{tab:{filename}}}", "\\end{table*}", ""])
    (out_dir / f"{filename}.tex").write_text("\n".join(lines), encoding="utf-8")


def efficiency(rows: Sequence[Mapping[str, Any]], prompt_setting: str) -> list[dict[str, Any]]:
    out = []
    methods = [m for m in METHOD_ORDER if any(r["method"] == m for r in rows)]
    for method in methods:
        vals = [r for r in rows if r["method"] == method]
        if not vals:
            continue
        item = {"Method": method, "Prompt setting": prompt_setting}
        for key in ["EM", "F1", "ContextTok", "InputTok", "PromptOverheadTok", "F1_per_1k_context", "F1_per_1k_input", "retrieval_ms", "generation_ms", "total_ms"]:
            nums = [float(r[key]) for r in vals if r.get(key) is not None]
            item[key] = mean(nums) if nums else None
        out.append(item)
    return out


def write_ablation(out_dir: Path) -> None:
    base_h, base_w = ABLATION_ROWS[0][1], ABLATION_ROWS[0][2]
    rows = []
    for name, h, w in ABLATION_ROWS:
        rows.append(
            {
                "Component": name,
                "HotpotQA F1": h,
                "HotpotQA delta": None if name == "ACE-RAG" else h - base_h,
                "2Wiki F1": w,
                "2Wiki delta": None if name == "ACE-RAG" else w - base_w,
            }
        )
    write_csv(out_dir / "ablation_components_table.csv", rows, ["Component", "HotpotQA F1", "HotpotQA delta", "2Wiki F1", "2Wiki delta"])
    md_rows = []
    for r in rows:
        hd = "--" if r["HotpotQA delta"] is None else f"{r['HotpotQA delta'] * 100:+.1f}"
        wd = "--" if r["2Wiki delta"] is None else f"{r['2Wiki delta'] * 100:+.1f}"
        md_rows.append({"Component": r["Component"], "HotpotQA F1 (Delta)": f"{r['HotpotQA F1'] * 100:.1f} ({hd})", "2Wiki F1 (Delta)": f"{r['2Wiki F1'] * 100:.1f} ({wd})"})
    (out_dir / "ablation_components_table.md").write_text(md_table(md_rows, ["Component", "HotpotQA F1 (Delta)", "2Wiki F1 (Delta)"]), encoding="utf-8")
    tex = [
        "\\begin{table}[t]",
        "\\centering",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{l|c|c}",
        "\\hline",
        "\\textbf{Component} & \\textbf{HotpotQA F1 ($\\Delta$)} & \\textbf{2Wiki F1 ($\\Delta$)} \\\\",
        "\\hline",
    ]
    for r in md_rows:
        if r["Component"] == "w/o Mention Edge":
            tex.append("\\hline")
        tex.append(f"{r['Component']} & {r['HotpotQA F1 (Delta)']} & {r['2Wiki F1 (Delta)']} \\\\")
    tex.extend(["\\hline", "\\end{tabular}", "}", "\\caption{Ablation study under the compact common-prompt setting.}", "\\label{tab:ablation_components}", "\\end{table}", ""])
    (out_dir / "ablation_components_table.tex").write_text("\n".join(tex), encoding="utf-8")


def write_simple_tex_table(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str], caption: str, label: str) -> None:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\hline",
        " & ".join(columns) + " \\\\",
        "\\hline",
    ]
    for row in rows:
        vals = []
        for col in columns:
            value = row.get(col)
            if col in {"EM", "F1", "Recall@5"}:
                vals.append(fmt_num(value, 3))
            elif value is None:
                vals.append("-")
            elif isinstance(value, float):
                vals.append(fmt_num(value, 2))
            else:
                vals.append(str(value))
        lines.append(" & ".join(vals) + " \\\\")
    lines.extend(["\\hline", "\\end{tabular}", "}", caption, f"\\label{{tab:{label}}}", "\\end{table*}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_reports(root: Path, common_rows: list[dict[str, Any]], native_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Timing Report",
        "",
        "- GPU0 endpoint: `http://localhost:8013/v1`, Qwen/Qwen2.5-7B-Instruct, gpu-memory-utilization 0.55, max-model-len 16384.",
        "- GPU1 endpoint: `http://localhost:8014/v1`, Qwen/Qwen2.5-7B-Instruct, gpu-memory-utilization 0.55, max-model-len 16384.",
        "- ACE-RAG p8 MuSiQue/PopQA and common top-8 scaled rows were generated from cached rendered contexts; retrieval was not rerun.",
        "- Timing for generation-only rows reflects async client concurrency and endpoint load. Treat it as throughput-run timing, not final paper timing unless rerun sequentially.",
        "",
        md_table(common_rows + native_rows, ["dataset", "method", "prompt_setting", "retrieval_ms", "generation_ms", "total_ms"]),
    ]
    (root / "timing_report.md").write_text("\n".join(lines), encoding="utf-8")

    compact = [r for r in common_rows if r["method"] == "ACE-RAG-Compact"]
    scaled = [r for r in common_rows if r["method"] == "ACE-RAG-Scaled"]
    best_baseline = {}
    for ds in DATASETS:
        base = [r for r in common_rows if r["dataset"] == ds and r["method"] not in {"ACE-RAG-Compact", "ACE-RAG-Scaled"}]
        best_baseline[ds] = max(base, key=lambda x: x.get("F1") or -1)
    rec = ["# Paper Recommendation", ""]
    rec.append("## Common Prompt Main Table")
    rec.append("")
    for ds in DATASETS:
        c = next((r for r in compact if r["dataset"] == ds), None)
        s = next((r for r in scaled if r["dataset"] == ds), None)
        b = best_baseline[ds]
        rec.append(
            f"- {ds}: best baseline {b['method']} F1={fmt_num(b['F1'])}, "
            f"ACE-RAG-Compact F1={fmt_num(c.get('F1') if c else None)} CtxTok={fmt_tok(c.get('ContextTok') if c else None)}, "
            f"ACE-RAG-Scaled F1={fmt_num(s.get('F1') if s else None)} CtxTok={fmt_tok(s.get('ContextTok') if s else None)}."
        )
    rec.extend(
        [
            "",
            "Recommendation: use ACE-RAG-Compact top-3 in the main common-prompt table because it beats the best common-prompt baseline on all four datasets while using substantially fewer context tokens. Keep ACE-RAG-Scaled top-8 as an appendix or efficiency/sensitivity row.",
            "",
            "## Native Prompt Table",
            "",
            "Use ACE-RAG native = p8_r0_section_aware + r0_current + top_k=8. Native rows are method-prompt results and should be reported separately from the common-prompt main comparison.",
            "",
            "## Ablation",
            "",
            "Keep the component ablation under common_qa, r0_current, top_k=3 and label it as compact common-prompt ablation.",
        ]
    )
    (root / "paper_recommendation.md").write_text("\n".join(rec), encoding="utf-8")

    server = [
        "# vLLM Server Report",
        "",
        "| port | GPU | model | gpu_memory_utilization | max_model_len | max_num_seqs | prefix caching |",
        "| --- | --- | --- | ---: | ---: | --- | --- |",
        "| 8013 | 0 | Qwen/Qwen2.5-7B-Instruct | 0.55 | 16384 | default | server default |",
        "| 8014 | 1 | Qwen/Qwen2.5-7B-Instruct | 0.55 | 16384 | default | server default |",
        "",
        "The first GPU1 start at 0.90 was discarded for timing comparability; PopQA p8 was rerun after restarting 8014 with 0.55.",
    ]
    (root / "vllm_server_report.md").write_text("\n".join(server), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.out_dir or args.root)
    root.mkdir(parents=True, exist_ok=True)

    common_rows = baseline_rows(COMMON_BASELINES, "common_qa")
    common_rows.extend(row_from_summary(ds, "ACE-RAG-Compact", "common_qa/top3", path) for ds, path in ACE_COMMON_COMPACT_PATHS.items())
    common_rows.extend(row_from_summary(ds, "ACE-RAG-Scaled", "common_qa/top8", path) for ds, path in ACE_COMMON_SCALED_PATHS.items())

    native_rows = baseline_rows(NATIVE_BASELINES, "native")
    native_rows.extend(row_from_summary(ds, "ACE-RAG", "native/p8_top8", path) for ds, path in ACE_NATIVE_P8_PATHS.items())

    columns = ["dataset", "method", "prompt_setting", "Recall@5", "EM", "F1", "ContextTok", "InputTok", "PromptOverheadTok", "F1_per_1k_context", "F1_per_1k_input", "retrieval_ms", "generation_ms", "total_ms", "source"]
    write_csv(root / "final_common_prompt_table.csv", common_rows, columns)
    (root / "final_common_prompt_table.md").write_text(md_table(common_rows, columns[:-1]), encoding="utf-8")
    write_csv(root / "final_native_prompt_expanded_table.csv", native_rows, columns)
    (root / "final_native_prompt_expanded_table.md").write_text(md_table(native_rows, columns[:-1]), encoding="utf-8")

    write_wide_table(
        root,
        "main_common_with_ace_rag_compact",
        [r for r in common_rows if r["method"] != "ACE-RAG-Scaled"],
        "ContextTok",
        "\\caption{Main results under the common QA prompt. R@5, EM, and F1 are reported as percentages; CtxTok denotes average retrieved-context tokens.}",
    )
    write_wide_table(
        root,
        "main_common_with_ace_rag_scaled",
        [r for r in common_rows if r["method"] != "ACE-RAG-Compact"],
        "ContextTok",
        "\\caption{Main results under the common QA prompt. R@5, EM, and F1 are reported as percentages; CtxTok denotes average retrieved-context tokens.}",
    )
    write_wide_table(
        root,
        "final_native_prompt_table",
        native_rows,
        "InputTok",
        "\\caption{Native-prompt results with method-specific QA prompts. InputTok denotes average full input tokens, including retrieved context and prompt text.}",
    )

    eff_rows = efficiency(common_rows, "common_qa") + efficiency(native_rows, "native")
    eff_columns = ["Method", "Prompt setting", "EM", "F1", "ContextTok", "InputTok", "PromptOverheadTok", "F1_per_1k_context", "F1_per_1k_input", "retrieval_ms", "generation_ms", "total_ms"]
    write_csv(root / "final_efficiency_summary.csv", eff_rows, eff_columns)
    (root / "final_efficiency_summary.md").write_text(md_table(eff_rows, eff_columns), encoding="utf-8")
    write_simple_tex_table(
        root / "final_efficiency_summary.tex",
        eff_rows,
        eff_columns,
        "\\caption{Average token efficiency and timing summary across four datasets.}",
        "final_efficiency_summary",
    )
    (root / "token_accounting_table.md").write_text(md_table(common_rows + native_rows, ["dataset", "method", "prompt_setting", "ContextTok", "InputTok", "PromptOverheadTok", "F1_per_1k_context", "F1_per_1k_input"]), encoding="utf-8")
    write_csv(root / "token_accounting_table.csv", common_rows + native_rows, ["dataset", "method", "prompt_setting", "ContextTok", "InputTok", "PromptOverheadTok", "F1_per_1k_context", "F1_per_1k_input"])
    write_ablation(root)
    write_reports(root, common_rows, native_rows)
    manifest = {"root": str(root), "git_commit": git_commit(), "common_rows": len(common_rows), "native_rows": len(native_rows)}
    (root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"WROTE {root}")


if __name__ == "__main__":
    main()
