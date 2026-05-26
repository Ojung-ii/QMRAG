#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def latest_dir(root: Path) -> Path | None:
    if not root.exists():
        return None
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda p: (p.stat().st_mtime, p.name))


def read_tsv(path: Path) -> list[list[str]]:
    if not path.exists():
        return []
    return [line.rstrip("\n").split("\t") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_outputs(master_log: Path) -> dict[int, str]:
    outputs: dict[int, str] = {}
    if not master_log.exists():
        return outputs
    pat = re.compile(r"\[JOB\s+(\d+)\]\s+END\s+output=(\S+)")
    for line in master_log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pat.search(line)
        if match:
            outputs[int(match.group(1))] = match.group(2)
    return outputs


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        result = float(value)
        return default if math.isnan(result) else result
    except Exception:
        return default


def metric(data: Mapping[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in data:
            return fnum(data.get(key), default)
    return default


def load_eval(predictions_path: str | None) -> dict[str, Any]:
    if not predictions_path:
        return {}
    pred = Path(predictions_path)
    eval_path = pred.parent / "eval.json"
    if not eval_path.exists():
        return {}
    with open(eval_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    data["_predictions_path"] = str(pred)
    data["_eval_path"] = str(eval_path)
    return data


def budget_point(setting: str, budget: str) -> str:
    if setting == "top3":
        return "top3_chain_dedup"
    if setting == "full_common_replay":
        return "full_structured_full"
    if budget:
        return f"budget{budget}"
    return setting


def sort_key(row: Mapping[str, Any]) -> tuple[str, int]:
    point = str(row.get("budget_point") or "")
    if point == "top3_chain_dedup":
        rank = 0
    elif point.startswith("budget"):
        rank = int(re.sub(r"\D", "", point) or "0")
    elif point == "full_structured_full":
        rank = 999999
    else:
        rank = 500000
    return str(row.get("dataset") or ""), rank


def row_from_eval(eval_data: Mapping[str, Any], cols: list[str]) -> dict[str, Any]:
    job_id, gpu, dataset, setting, profile, prompt, limit, budget = (cols + [""] * 8)[:8]
    f1 = metric(eval_data, "F1", "f1")
    input_tok = metric(eval_data, "avg_input_prompt_tokens")
    generation_ms = metric(eval_data, "generation_latency_ms")
    retrieval_ms = metric(eval_data, "retrieval_latency_ms")
    actual_ctx = metric(eval_data, "avg_actual_context_tokens", "avg_context_tokens", "avg_rendered_context_tokens")
    return {
        "job_id": int(job_id),
        "gpu": gpu,
        "dataset": dataset,
        "setting": setting,
        "budget_point": budget_point(setting, budget),
        "budget": int(budget) if str(budget).strip() else None,
        "prompt_profile": prompt,
        "compaction_profile": profile,
        "limit": int(limit),
        "path": eval_data.get("_predictions_path"),
        "n": int(metric(eval_data, "n", default=0.0)),
        "F1": f1,
        "EM": metric(eval_data, "EM", "em"),
        "AnsPred": metric(eval_data, "answer_in_prediction", "answer_contains"),
        "AnsCtx": metric(eval_data, "answer_in_rendered_context"),
        "Insuff": metric(eval_data, "insufficient_rate"),
        "actual_CtxTok": actual_ctx,
        "InputTok": input_tok,
        "TotalTok": metric(eval_data, "avg_total_llm_tokens"),
        "rendered_bundle_count": metric(eval_data, "avg_rendered_bundle_count"),
        "rendered_chain_count": metric(eval_data, "avg_rendered_chain_count"),
        "rendered_support_count": metric(eval_data, "avg_rendered_support_count", "avg_support_sentence_count"),
        "rendered_source_count": metric(eval_data, "avg_rendered_source_count"),
        "rendered_sentence_count": metric(eval_data, "avg_rendered_sentence_count"),
        "budget_saturated_rate": metric(eval_data, "budget_saturated_rate"),
        "F1_per_1k_input": metric(eval_data, "F1_per_1k_input_prompt_tokens", default=(f1 / max(1e-9, input_tok) * 1000.0 if input_tok else 0.0)),
        "generation_ms": generation_ms,
        "source_retrieval_ms": retrieval_ms,
        "effective_total_ms": generation_ms + retrieval_ms,
        "evidence_bundles_hash_match_rate": metric(eval_data, "evidence_bundles_hash_match_rate"),
        "rendered_context_hash_match_rate": metric(eval_data, "rendered_context_hash_match_rate"),
    }


def collect(run_dir: Path) -> tuple[list[dict[str, Any]], list[list[str]]]:
    outputs = parse_outputs(run_dir / "master.log")
    failed = read_tsv(run_dir / "failed_jobs.tsv")
    rows: list[dict[str, Any]] = []
    for cols in read_tsv(run_dir / "jobs.tsv"):
        if len(cols) < 8:
            continue
        eval_data = load_eval(outputs.get(int(cols[0])))
        if not eval_data:
            continue
        rows.append(row_from_eval(eval_data, cols))
    return sorted(rows, key=sort_key), failed


def validate_growth(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        ds_rows = [row for row in rows if row["dataset"] == dataset]
        budget_rows = [row for row in ds_rows if str(row.get("budget_point")).startswith("budget")]
        budget_rows = sorted(budget_rows, key=lambda row: int(row.get("budget") or 0))
        values = [float(row.get("actual_CtxTok", 0.0) or 0.0) for row in budget_rows]
        nondecreasing = all(values[i] <= values[i + 1] + 5.0 for i in range(len(values) - 1))
        all_same = len({round(v, 1) for v in values}) <= 1
        top3 = next((row for row in ds_rows if row["budget_point"] == "top3_chain_dedup"), None)
        full = next((row for row in ds_rows if row["budget_point"] == "full_structured_full"), None)
        starts_after_top3 = True if not top3 or not values else values[0] + 5.0 >= float(top3.get("actual_CtxTok", 0.0) or 0.0)
        ends_before_full = True if not full or not values else values[-1] <= float(full.get("actual_CtxTok", 0.0) or 0.0) + 5.0
        valid = bool(values) and nondecreasing and not all_same
        reason = "ok"
        if not values:
            reason = "missing_budget_rows"
        elif all_same:
            reason = "budget_points_have_identical_actual_context_tokens"
        elif not nondecreasing:
            reason = "actual_context_tokens_decreased_with_larger_budget"
        elif not starts_after_top3:
            reason = "ok_budget500_near_or_below_top3"
        elif not ends_before_full:
            reason = "ok_budget2000_near_or_above_full"
        result[dataset] = {
            "budget_context_growth_valid": valid,
            "failure_reason": reason,
            "budget_actual_context_tokens": values,
        }
        for row in ds_rows:
            row["budget_valid"] = valid
            row["budget_failure_reason"] = reason
    return result


def add_deltas(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    interp: dict[str, dict[str, Any]] = {}
    for dataset in sorted({row["dataset"] for row in rows}):
        ds_rows = [row for row in rows if row["dataset"] == dataset]
        top3 = next((row for row in ds_rows if row["budget_point"] == "top3_chain_dedup"), None)
        full = next((row for row in ds_rows if row["budget_point"] == "full_structured_full"), None)
        if not top3 or not full:
            continue
        for row in ds_rows:
            row["TokenDown_vs_full"] = 1.0 - float(row["InputTok"]) / max(1e-9, float(full["InputTok"]))
            row["F1_delta_vs_top3"] = float(row["F1"]) - float(top3["F1"])
            row["F1_delta_vs_full"] = float(row["F1"]) - float(full["F1"])
        token_peak = max(ds_rows, key=lambda row: float(row.get("F1_per_1k_input", 0.0) or 0.0))
        best_quality = max(ds_rows, key=lambda row: float(row.get("F1", 0.0) or 0.0))
        full_gain = float(full["F1"]) - float(top3["F1"])
        best_gain = float(best_quality["F1"]) - float(top3["F1"])
        interp[dataset] = {
            "token_efficiency_peak": token_peak["budget_point"],
            "quality_peak": best_quality["budget_point"],
            "full_quality_gain": full_gain,
            "best_quality_gain_vs_top3": best_gain,
            "budget_scaling_positive": best_gain > 0.0,
            "full_greater_than_top3": full_gain > 0.0,
        }
    return interp


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def md_table(headers: list[str], rows: list[Mapping[str, Any]], keys: list[str], digits: Mapping[str, int] | None = None) -> list[str]:
    digits = digits or {}
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:" for _ in headers[1:]]) + " |",
    ]
    for row in rows:
        cells = [fmt(row.get(key), digits.get(key, 4)) for key in keys]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_markdown(path: Path, run_dir: Path, rows: list[dict[str, Any]], failed: list[list[str]], growth: Mapping[str, Any], interp: Mapping[str, Any]) -> None:
    lines: list[str] = [
        "# Full Structured Budget Scaling Summary",
        "",
        f"- run_dir: `{run_dir}`",
        f"- failed_jobs: {len(failed)}",
        "",
        "## Section 1. Budget Scaling Table",
        "",
    ]
    lines += md_table(
        ["dataset","budget_point","setting","F1","EM","AnsPred","AnsCtx","Insuff","actual_CtxTok","InputTok","TotalTok","bundles","chains","support","sources","sentences","saturated","F1/1kInput","gen_ms","retr_ms","eff_ms","TokenDown","dF1_top3","dF1_full","EBHash","CtxHash","budget_valid"],
        rows,
        ["dataset","budget_point","setting","F1","EM","AnsPred","AnsCtx","Insuff","actual_CtxTok","InputTok","TotalTok","rendered_bundle_count","rendered_chain_count","rendered_support_count","rendered_source_count","rendered_sentence_count","budget_saturated_rate","F1_per_1k_input","generation_ms","source_retrieval_ms","effective_total_ms","TokenDown_vs_full","F1_delta_vs_top3","F1_delta_vs_full","evidence_bundles_hash_match_rate","rendered_context_hash_match_rate","budget_valid"],
        {"actual_CtxTok":1,"InputTok":1,"TotalTok":1,"generation_ms":1,"source_retrieval_ms":1,"effective_total_ms":1},
    )
    lines += ["", "## Section 2. Context Growth Validation", ""]
    validation_rows = []
    for dataset, data in growth.items():
        validation_rows.append({
            "dataset": dataset,
            "budget_context_growth_valid": data.get("budget_context_growth_valid"),
            "actual_tokens": ", ".join(fmt(x, 1) for x in data.get("budget_actual_context_tokens", [])),
            "failure_reason": data.get("failure_reason"),
        })
    lines += md_table(
        ["dataset","budget_context_growth_valid","budget actual tokens 500/1000/1500/2000","failure_reason"],
        validation_rows,
        ["dataset","budget_context_growth_valid","actual_tokens","failure_reason"],
    )
    lines += ["", "## Section 3. Interpretation", ""]
    for dataset, data in interp.items():
        growth_ok = bool(growth.get(dataset, {}).get("budget_context_growth_valid"))
        scaling = bool(data.get("budget_scaling_positive")) and growth_ok
        lines += [
            f"- {dataset}: budget_scaling_positive={fmt(scaling)}, token_efficiency_peak={data.get('token_efficiency_peak')}, quality_peak={data.get('quality_peak')}, full_quality_gain={fmt(data.get('full_quality_gain'))}.",
        ]
    if any(bool(data.get("budget_context_growth_valid")) and bool(interp.get(dataset, {}).get("budget_scaling_positive")) for dataset, data in growth.items()):
        lines += [
            "",
            "Paper draft: As the rendering budget increases, BRACE-RAG recovers additional answer accuracy, showing that the same retrieved evidence chains can trade token efficiency for higher answer quality.",
        ]
    else:
        lines += [
            "",
            "Paper draft: Performance does not increase monotonically with raw context budget. This suggests that answerability depends not only on the amount of context but also on whether the rendered context preserves the right chain-supporting evidence.",
        ]
    lines += ["", "## Failed Jobs", ""]
    if failed:
        for item in failed:
            lines.append("- " + " | ".join(item))
    else:
        lines.append("- none")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare full structured context budget scaling replay runs.")
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--dataset", choices=["hotpotqa","2wiki"], default=None)
    parser.add_argument("--analysis-root", default="outputs/analysis")
    args = parser.parse_args()
    if not args.latest:
        raise SystemExit("Only --latest is supported.")
    run_dir = latest_dir(Path("logs/full_structured_budget_scaling"))
    if not run_dir:
        raise SystemExit("No logs/full_structured_budget_scaling run found")
    rows, failed = collect(run_dir)
    if args.dataset:
        rows = [row for row in rows if row["dataset"] == args.dataset]
    growth = validate_growth(rows)
    interp = add_deltas(rows)
    payload = {
        "run_dir": str(run_dir),
        "rows": rows,
        "context_growth_validation": growth,
        "interpretation": interp,
        "failed_jobs": failed,
    }
    out_dir = Path(args.analysis_root) / now_timestamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "full_structured_budget_scaling_summary.json"
    md_path = out_dir / "full_structured_budget_scaling_summary.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, run_dir, rows, failed, growth, interp)
    print(f"wrote: {json_path}")
    print(f"wrote: {md_path}")


if __name__ == "__main__":
    main()
