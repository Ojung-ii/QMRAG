#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.io_utils import read_jsonl


DATASETS = ("hotpotqa", "2wiki", "popqa", "musique")
BEST_BASELINES = {
    "hotpotqa": {"method": "LightRAG", "f1": 0.3229},
    "2wiki": {"method": "LightRAG", "f1": 0.0953},
    "popqa": {"method": "Dense RAG", "f1": 0.4167},
    "musique": {"method": "HippoRAG2", "f1": 0.0551},
}


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def latest_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda p: (p.stat().st_mtime, p.name))


def parse_master_outputs(log_path: Path) -> dict[int, str]:
    outputs: dict[int, str] = {}
    if not log_path.exists():
        return outputs
    pat = re.compile(r"\[JOB\s+(\d+)\]\s+END\s+output=(\S+)")
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pat.search(line)
        if m:
            outputs[int(m.group(1))] = m.group(2)
    return outputs


def read_tsv(path: Path) -> list[list[str]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(line.rstrip("\n").split("\t"))
    return rows


def load_eval(pred_path: str | None) -> dict[str, Any]:
    if not pred_path:
        return {}
    path = Path(pred_path)
    eval_path = path.parent / "eval.json"
    if not eval_path.exists():
        return {}
    with open(eval_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    data["_predictions_path"] = str(path)
    data["_eval_path"] = str(eval_path)
    return data


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        x = float(value)
        if math.isnan(x):
            return default
        return x
    except Exception:
        return default


def metric(eval_data: Mapping[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in eval_data:
            return fnum(eval_data.get(key), default)
    return default


def saturated_rate(pred_path: str | None, budget: int | None) -> float | None:
    if not pred_path or not budget:
        return None
    try:
        rows = read_jsonl(Path(pred_path))
    except Exception:
        return None
    if not rows:
        return None
    threshold = 0.95 * float(budget)
    hits = 0
    for row in rows:
        tokens = fnum(row.get("rendered_context_tokens") or row.get("avg_context_tokens"))
        if tokens >= threshold:
            hits += 1
    return hits / max(1, len(rows))


def base_row(eval_data: Mapping[str, Any], dataset: str, setting: str, prompt: str, compaction: str) -> dict[str, Any]:
    baseline = BEST_BASELINES.get(dataset, {"method": "", "f1": 0.0})
    generation_ms = metric(eval_data, "generation_latency_ms")
    retrieval_ms = metric(eval_data, "retrieval_latency_ms")
    effective_total_ms = retrieval_ms + generation_ms
    f1 = metric(eval_data, "F1", "f1")
    input_tok = metric(eval_data, "avg_input_prompt_tokens", "InputTok")
    ctx_tok = metric(eval_data, "avg_context_tokens", "avg_rendered_context_tokens", "CtxTok")
    return {
        "dataset": dataset,
        "setting": setting,
        "prompt": prompt,
        "compaction_profile": compaction,
        "path": eval_data.get("_predictions_path"),
        "n": int(metric(eval_data, "n", default=0.0)),
        "F1": f1,
        "EM": metric(eval_data, "EM", "em"),
        "AnsPred": metric(eval_data, "answer_in_prediction", "answer_contains"),
        "AnsCtx": metric(eval_data, "answer_in_rendered_context", "answer_in_context"),
        "Insuff": metric(eval_data, "insufficient_rate"),
        "InputTok": input_tok,
        "CtxTok": ctx_tok,
        "TotalTok": metric(eval_data, "avg_total_llm_tokens", "TotalTok"),
        "F1_per_1k_input": f1 / max(1e-9, input_tok) * 1000.0 if input_tok else 0.0,
        "generation_ms": generation_ms,
        "source_retrieval_ms": retrieval_ms,
        "effective_total_ms": effective_total_ms,
        "best_baseline_method": baseline["method"],
        "best_baseline_f1": baseline["f1"],
        "margin_vs_best_baseline": f1 - float(baseline["f1"]),
        "evidence_bundles_hash_match_rate": metric(eval_data, "evidence_bundles_hash_match_rate", default=0.0),
        "rendered_context_hash_match_rate": metric(eval_data, "rendered_context_hash_match_rate", default=0.0),
        "rendered_bundle_count": metric(eval_data, "avg_rendered_bundle_count"),
        "rendered_sentence_count": metric(eval_data, "avg_rendered_sentence_count"),
        "context_token_budget": eval_data.get("context_token_budget"),
        "token_reduction_rate": metric(eval_data, "token_reduction_rate"),
    }


def collect_exp1(log_root: Path) -> tuple[Path | None, list[dict[str, Any]], list[list[str]]]:
    run_dir = latest_dir(log_root)
    if not run_dir:
        return None, [], []
    outputs = parse_master_outputs(run_dir / "master.log")
    failed = read_tsv(run_dir / "failed_jobs.tsv")
    rows: list[dict[str, Any]] = []
    for cols in read_tsv(run_dir / "jobs.tsv"):
        if len(cols) < 7:
            continue
        job_id, _gpu, dataset, setting, profile, prompt, _limit = cols[:7]
        eval_data = load_eval(outputs.get(int(job_id)))
        if not eval_data:
            continue
        rows.append(base_row(eval_data, dataset, setting, prompt, profile))
    full_by_dataset = {r["dataset"]: r for r in rows if r["setting"] == "full_common_replay"}
    for row in rows:
        full = full_by_dataset.get(row["dataset"])
        if full:
            row["TokenDown_vs_full"] = 1.0 - row["InputTok"] / max(1e-9, full["InputTok"])
            row["generation_ms_reduction_vs_full"] = full["generation_ms"] - row["generation_ms"]
            row["effective_total_ms_reduction_vs_full"] = full["effective_total_ms"] - row["effective_total_ms"]
        else:
            row["TokenDown_vs_full"] = None
            row["generation_ms_reduction_vs_full"] = None
            row["effective_total_ms_reduction_vs_full"] = None
    return run_dir, rows, failed


def collect_exp2_parallel(log_root: Path, exp1_rows: list[dict[str, Any]]) -> tuple[Path | None, list[dict[str, Any]], list[list[str]]]:
    run_dir = latest_dir(log_root)
    rows: list[dict[str, Any]] = []
    failed: list[list[str]] = []
    if run_dir:
        outputs = parse_master_outputs(run_dir / "master.log")
        failed = read_tsv(run_dir / "failed_jobs.tsv")
        for cols in read_tsv(run_dir / "jobs.tsv"):
            if len(cols) < 4:
                continue
            job_id, _gpu, dataset, budget = cols[:4]
            eval_data = load_eval(outputs.get(int(job_id)))
            if not eval_data:
                continue
            row = base_row(eval_data, dataset, f"budget{budget}", "common_qa", "chain_dedup_budget")
            row["budget_point"] = f"budget{budget}"
            row["budget"] = int(budget)
            row["budget_saturated_rate"] = saturated_rate(row["path"], int(budget))
            row["generation_ms_parallel_diagnostic"] = row["generation_ms"]
            rows.append(row)
    for row in exp1_rows:
        if row["dataset"] not in {"hotpotqa", "2wiki"}:
            continue
        if row["setting"] == "compact_common":
            merged = dict(row)
            merged["budget_point"] = "top3_chain_dedup"
            merged["budget"] = None
            merged["budget_saturated_rate"] = None
            merged["generation_ms_parallel_diagnostic"] = None
            rows.append(merged)
        elif row["setting"] == "full_common_replay":
            merged = dict(row)
            merged["budget_point"] = "full_common"
            merged["budget"] = None
            merged["budget_saturated_rate"] = None
            merged["generation_ms_parallel_diagnostic"] = None
            rows.append(merged)
    by_dataset = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)
    for dataset, dataset_rows in by_dataset.items():
        top3 = next((r for r in dataset_rows if r.get("budget_point") == "top3_chain_dedup"), None)
        full = next((r for r in dataset_rows if r.get("budget_point") == "full_common"), None)
        for row in dataset_rows:
            row["F1_delta_vs_top3"] = row["F1"] - top3["F1"] if top3 else None
            row["F1_delta_vs_full"] = row["F1"] - full["F1"] if full else None
    return run_dir, sorted(rows, key=budget_sort_key), failed


def collect_recheck(log_root: Path) -> tuple[Path | None, list[dict[str, Any]], list[list[str]]]:
    run_dir = latest_dir(log_root)
    if not run_dir:
        return None, [], []
    outputs = parse_master_outputs(run_dir / "master.log")
    failed = read_tsv(run_dir / "failed_jobs.tsv")
    rows: list[dict[str, Any]] = []
    for cols in read_tsv(run_dir / "jobs.tsv"):
        if len(cols) < 8:
            continue
        job_id, _gpu, dataset, budget_point, profile, prompt, _limit, budget = cols[:8]
        eval_data = load_eval(outputs.get(int(job_id)))
        if not eval_data:
            continue
        row = base_row(eval_data, dataset, budget_point, prompt, profile)
        row["budget_point"] = budget_point
        row["budget"] = int(budget) if str(budget).strip() else None
        rows.append(row)
    full_by_dataset = {r["dataset"]: r for r in rows if r["budget_point"] == "full_common_replay"}
    for row in rows:
        full = full_by_dataset.get(row["dataset"])
        if full:
            row["GenDown_vs_full"] = full["generation_ms"] - row["generation_ms"]
            row["EffDown_vs_full"] = full["effective_total_ms"] - row["effective_total_ms"]
        else:
            row["GenDown_vs_full"] = None
            row["EffDown_vs_full"] = None
    return run_dir, sorted(rows, key=budget_sort_key), failed


def budget_sort_key(row: Mapping[str, Any]) -> tuple[str, int, str]:
    point = str(row.get("budget_point") or row.get("setting") or "")
    if point == "top3_chain_dedup":
        order = 0
    elif point.startswith("budget"):
        order = 1 + int(re.sub(r"\D", "", point) or "0")
    elif point in {"full_common", "full_common_replay"}:
        order = 999999
    else:
        order = 500000
    return str(row.get("dataset") or ""), order, point


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def md_table(headers: list[str], rows: list[Mapping[str, Any]], key_map: list[str], digits: Mapping[str, int] | None = None) -> list[str]:
    digits = digits or {}
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:" for _ in headers[1:]]) + " |",
    ]
    for row in rows:
        cells = []
        for key in key_map:
            value = row.get(key)
            cells.append(fmt(value, digits.get(key, 4)))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def interpret(exp1_rows: list[dict[str, Any]], exp2_rows: list[dict[str, Any]], recheck_rows: list[dict[str, Any]]) -> dict[str, Any]:
    top3_common = [r for r in exp1_rows if r["setting"] == "compact_common" and r["prompt"] == "common_qa"]
    compact_main_success = bool(top3_common) and all(r["F1"] > r["best_baseline_f1"] for r in top3_common)
    native_rows = [r for r in exp1_rows if r["setting"].startswith("compact_native") or r["setting"] == "compact_strict_short"]
    native_appendix_candidate = bool(native_rows and top3_common) and mean(r["F1"] for r in native_rows) > mean(r["F1"] for r in top3_common)

    budget_positive_by_dataset: dict[str, bool] = {}
    for dataset in sorted({r["dataset"] for r in exp2_rows}):
        rows = [r for r in exp2_rows if r["dataset"] == dataset]
        top3 = next((r for r in rows if r.get("budget_point") == "top3_chain_dedup"), None)
        full = next((r for r in rows if r.get("budget_point") == "full_common"), None)
        budget_rows = [r for r in rows if str(r.get("budget_point", "")).startswith("budget")]
        if not top3 or not budget_rows:
            budget_positive_by_dataset[dataset] = False
            continue
        best_budget = max(r["F1"] for r in budget_rows)
        budget_positive_by_dataset[dataset] = best_budget >= top3["F1"] and (not full or full["F1"] >= top3["F1"])
    budget_scaling_positive = bool(budget_positive_by_dataset) and all(budget_positive_by_dataset.values())

    runtime_positive_by_dataset: dict[str, bool] = {}
    for dataset in sorted({r["dataset"] for r in recheck_rows}):
        rows = [r for r in recheck_rows if r["dataset"] == dataset]
        top3 = next((r for r in rows if r.get("budget_point") == "top3_chain_dedup"), None)
        full = next((r for r in rows if r.get("budget_point") == "full_common_replay"), None)
        runtime_positive_by_dataset[dataset] = bool(top3 and full and top3["generation_ms"] < full["generation_ms"] and top3["effective_total_ms"] < full["effective_total_ms"])
    runtime_positive = bool(runtime_positive_by_dataset) and all(runtime_positive_by_dataset.values())

    return {
        "compact_main_success": compact_main_success,
        "native_appendix_candidate": native_appendix_candidate,
        "budget_scaling_positive": budget_scaling_positive,
        "budget_scaling_positive_by_dataset": budget_positive_by_dataset,
        "runtime_positive": runtime_positive,
        "runtime_positive_by_dataset": runtime_positive_by_dataset,
    }


def write_markdown(
    out_path: Path,
    exp1_dir: Path | None,
    exp1_rows: list[dict[str, Any]],
    exp1_failed: list[list[str]],
    exp2_dir: Path | None,
    exp2_rows: list[dict[str, Any]],
    exp2_failed: list[list[str]],
    recheck_dir: Path | None,
    recheck_rows: list[dict[str, Any]],
    recheck_failed: list[list[str]],
    interp: Mapping[str, Any],
) -> None:
    lines: list[str] = ["# Exp1/Exp2 Compact Budget Summary", ""]
    lines += [
        f"- exp1_log_dir: `{exp1_dir}`",
        f"- exp2_parallel_log_dir: `{exp2_dir}`",
        f"- exp2_recheck_log_dir: `{recheck_dir}`",
        f"- compact_main_success: {fmt(interp.get('compact_main_success'))}",
        f"- native_appendix_candidate: {fmt(interp.get('native_appendix_candidate'))}",
        f"- budget_scaling_positive: {fmt(interp.get('budget_scaling_positive'))}",
        f"- runtime_positive: {fmt(interp.get('runtime_positive'))}",
        "",
    ]
    lines += ["## Section 1. Exp1 Controlled Compact/Native", ""]
    if exp1_rows:
        lines += md_table(
            ["dataset", "setting", "prompt", "F1", "EM", "AnsPred", "AnsCtx", "Insuff", "InputTok", "CtxTok", "TokenDown", "F1/1kInput", "gen_ms", "retr_ms", "eff_ms", "GenDown", "EffDown", "base_f1", "margin", "EBHash", "CtxHash"],
            sorted(exp1_rows, key=lambda r: (r["dataset"], r["setting"])),
            ["dataset", "setting", "prompt", "F1", "EM", "AnsPred", "AnsCtx", "Insuff", "InputTok", "CtxTok", "TokenDown_vs_full", "F1_per_1k_input", "generation_ms", "source_retrieval_ms", "effective_total_ms", "generation_ms_reduction_vs_full", "effective_total_ms_reduction_vs_full", "best_baseline_f1", "margin_vs_best_baseline", "evidence_bundles_hash_match_rate", "rendered_context_hash_match_rate"],
            {"InputTok": 1, "CtxTok": 1, "generation_ms": 1, "source_retrieval_ms": 1, "effective_total_ms": 1, "generation_ms_reduction_vs_full": 1, "effective_total_ms_reduction_vs_full": 1},
        )
    else:
        lines.append("- No Exp1 controlled sequential run found.")
    lines += ["", "## Section 2. Exp2 Budget Scaling", ""]
    if exp2_rows:
        lines += md_table(
            ["dataset", "budget_point", "F1", "AnsPred", "AnsCtx", "Insuff", "CtxTok", "InputTok", "bundles", "sentences", "saturated", "F1/1kInput", "dF1_top3", "dF1_full", "parallel_gen_ms_diag"],
            exp2_rows,
            ["dataset", "budget_point", "F1", "AnsPred", "AnsCtx", "Insuff", "CtxTok", "InputTok", "rendered_bundle_count", "rendered_sentence_count", "budget_saturated_rate", "F1_per_1k_input", "F1_delta_vs_top3", "F1_delta_vs_full", "generation_ms_parallel_diagnostic"],
            {"CtxTok": 1, "InputTok": 1, "generation_ms_parallel_diagnostic": 1},
        )
        lines += ["", "Note: `parallel_gen_ms_diag` is logged for diagnostics only and must not be used as paper timing evidence."]
    else:
        lines.append("- No Exp2 parallel budget run found.")
    lines += ["", "## Section 3. Exp2 Sequential Timing Recheck", ""]
    if recheck_rows:
        lines += md_table(
            ["dataset", "budget_point", "F1", "InputTok", "gen_ms", "eff_ms", "GenDown", "EffDown"],
            recheck_rows,
            ["dataset", "budget_point", "F1", "InputTok", "generation_ms", "effective_total_ms", "GenDown_vs_full", "EffDown_vs_full"],
            {"InputTok": 1, "generation_ms": 1, "effective_total_ms": 1, "GenDown_vs_full": 1, "EffDown_vs_full": 1},
        )
    else:
        lines.append("- No Exp2 sequential timing recheck run found.")
    lines += [
        "",
        "## Section 4. Interpretation",
        "",
        f"- QMRAG-Compact-common main result usable: {fmt(interp.get('compact_main_success'))}.",
        f"- Native prompt appendix candidate: {fmt(interp.get('native_appendix_candidate'))}.",
        f"- Budget scaling positive: {fmt(interp.get('budget_scaling_positive'))} ({interp.get('budget_scaling_positive_by_dataset')}).",
        f"- Sequential timing positive: {fmt(interp.get('runtime_positive'))} ({interp.get('runtime_positive_by_dataset')}).",
        "- Use Exp1 or Exp2 sequential recheck for timing claims; do not use Exp2 parallel timing.",
        "",
        "## Failed Jobs",
        "",
        f"- Exp1 failed jobs: {len(exp1_failed)}",
        f"- Exp2 parallel failed jobs: {len(exp2_failed)}",
        f"- Exp2 recheck failed jobs: {len(recheck_failed)}",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare controlled compact timing and budget scaling runs.")
    parser.add_argument("--latest", action="store_true", help="Use the latest log directories for each experiment.")
    parser.add_argument("--analysis-root", default="outputs/analysis")
    args = parser.parse_args()
    if not args.latest:
        raise SystemExit("Only --latest is currently supported.")

    exp1_dir, exp1_rows, exp1_failed = collect_exp1(Path("logs/exp1_controlled_sequential"))
    exp2_dir, exp2_rows, exp2_failed = collect_exp2_parallel(Path("logs/exp2_budget_parallel"), exp1_rows)
    recheck_dir, recheck_rows, recheck_failed = collect_recheck(Path("logs/exp2_budget_timing_recheck"))
    interp = interpret(exp1_rows, exp2_rows, recheck_rows)
    payload = {
        "exp1_log_dir": str(exp1_dir) if exp1_dir else None,
        "exp2_parallel_log_dir": str(exp2_dir) if exp2_dir else None,
        "exp2_recheck_log_dir": str(recheck_dir) if recheck_dir else None,
        "exp1_rows": exp1_rows,
        "exp2_budget_rows": exp2_rows,
        "exp2_recheck_rows": recheck_rows,
        "failed_jobs": {
            "exp1": exp1_failed,
            "exp2_parallel": exp2_failed,
            "exp2_recheck": recheck_failed,
        },
        "interpretation": interp,
    }
    out_dir = Path(args.analysis_root) / now_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "exp1_exp2_compact_budget_summary.json"
    md_path = out_dir / "exp1_exp2_compact_budget_summary.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, exp1_dir, exp1_rows, exp1_failed, exp2_dir, exp2_rows, exp2_failed, recheck_dir, recheck_rows, recheck_failed, interp)
    print(f"wrote: {json_path}")
    print(f"wrote: {md_path}")


if __name__ == "__main__":
    main()
