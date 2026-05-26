#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.eval_metrics import answer_contains, evaluate_predictions, exact_match
from utils.io_utils import dump_json, ensure_dir, read_jsonl
from utils.generation import has_idk_phrase, is_insufficient_prediction
from utils.text import normalize_answer


BEST_BASELINES = {
    "hotpotqa": {"method": "LightRAG", "f1": 0.3229},
    "2wiki": {"method": "LightRAG", "f1": 0.0953},
    "musique": {"method": "HippoRAG2", "f1": 0.0551},
    "popqa": {"method": "Dense RAG", "f1": 0.4167},
}

TARGET_PROMPTS = {
    "common_qa",
    "strict_short_qa",
    "qmrag_bundle_short_qa",
    "qmrag_compact_chain_short_qa",
}
TARGET_COMPACTIONS = {
    "none",
    "metadata_only_compact",
    "chain_dedup",
    "top3_chain_dedup",
    "chain_schema_k3",
    "chain_schema_plus1_k3",
    "top3_schema_dedup",
}

PREFIX_RE = re.compile(r"^\s*(answer|final answer|the answer is|it is|based on)\s*[:,-]?\s+", re.I)
MARKDOWN_RE = re.compile(r"(^|\n)\s*(#{1,6}\s+|[-*]\s+|\d+\.\s+|```|\*\*)")
CITATION_RE = re.compile(r"\[[^\]]+\]|\(\s*(?:source|citation|context|evidence)[^)]+\)", re.I)
EXPLANATION_RE = re.compile(r"\b(because|based on|according to|context|provided|evidence|therefore|so the answer)\b", re.I)


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def iter_prediction_files(root: Path) -> Iterable[Path]:
    yield from sorted(root.rglob("predictions.jsonl"))


def iter_full_eval_files(root: Path, dataset: str) -> Iterable[Path]:
    base = root / dataset / "eval"
    if base.exists():
        yield from sorted(base.rglob("predictions.jsonl"))


def iter_replay_files(root: Path, dataset: str) -> Iterable[Path]:
    base = root / "replay"
    if not base.exists():
        return
    for run_dir in sorted(base.iterdir()):
        ds_dir = run_dir / dataset
        if ds_dir.exists():
            yield from sorted(ds_dir.rglob("predictions.jsonl"))


def first_jsonl_row(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    return json.loads(line)
    except Exception:
        return None
    return None


def row_dataset(path: Path, row: Mapping[str, Any]) -> str:
    if row.get("dataset"):
        return str(row["dataset"])
    parts = path.parts
    if "outputs" in parts:
        rest = parts[parts.index("outputs") + 1 :]
        if len(rest) >= 2 and rest[1] == "eval":
            return rest[0]
        if len(rest) >= 3 and rest[0] == "replay":
            return rest[2]
    return "UNKNOWN"


def row_compaction(row: Mapping[str, Any]) -> str:
    return str(row.get("compaction_profile") or "none")


def has_truncation(row: Mapping[str, Any]) -> bool:
    return bool(row.get("context_truncation_enabled") or row.get("top_bundles") is not None or row.get("context_token_budget") is not None)


def source_preference(row: Mapping[str, Any]) -> int:
    ablation = str(row.get("ablation_variant") or "")
    residual = str((row.get("retrieval_diagnostics", {}) or {}).get("residual_selection_variant") or row.get("residual_selection_variant") or "")
    if ablation in {"", "core_qmrag_mainline"} and residual in {"", "residual_lexical"}:
        return 2
    if ablation in {"", "core_qmrag_mainline"}:
        return 1
    return 0


def is_under_replay(root: Path, path: Path) -> bool:
    try:
        rel = path.relative_to(root)
        return bool(rel.parts and rel.parts[0] == "replay")
    except ValueError:
        return False


def find_full_run(root: Path, dataset: str, prompt_profile: str) -> Path | None:
    candidates: list[dict[str, Any]] = []
    for path in iter_full_eval_files(root, dataset):
        row = first_jsonl_row(path)
        if not row:
            continue
        if row_dataset(path, row) != dataset:
            continue
        if str(row.get("prompt_profile") or "") != prompt_profile:
            continue
        if str(row.get("rendering_profile") or "structured_chain") != "structured_chain":
            continue
        if row_compaction(row) != "none" or row.get("context_compaction_enabled") or has_truncation(row):
            continue
        candidates.append({"path": path, "mtime": path.stat().st_mtime, "source_preference": source_preference(row)})
    if not candidates:
        return None
    return max(candidates, key=lambda x: (int(x["source_preference"]), float(x["mtime"]), str(x["path"])))["path"]


def find_latest_replays(root: Path, dataset: str) -> list[Path]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for path in iter_replay_files(root, dataset):
        row = first_jsonl_row(path)
        if not row:
            continue
        if row_dataset(path, row) != dataset:
            continue
        prompt = str(row.get("prompt_profile") or "")
        compaction = row_compaction(row)
        if prompt not in TARGET_PROMPTS or compaction not in TARGET_COMPACTIONS:
            continue
        if str(row.get("source_prompt_profile") or "common_qa") != "common_qa":
            continue
        if str(row.get("rendering_profile") or "structured_chain") != "structured_chain":
            continue
        if prompt == "common_qa" and compaction == "none":
            continue
        if prompt == "strict_short_qa" and compaction != "none":
            continue
        key = (prompt, compaction)
        info = {"path": path, "mtime": path.stat().st_mtime}
        if key not in latest or float(info["mtime"]) > float(latest[key]["mtime"]):
            latest[key] = info
    return [info["path"] for _, info in sorted(latest.items())]


def correct(row: Mapping[str, Any]) -> bool:
    answers = [str(x) for x in row.get("answers", []) if str(x).strip()]
    raw = str(row.get("raw_prediction", row.get("prediction", "")) or "")
    return bool(exact_match(raw, answers) or answer_contains(raw, answers))


def overlap_rows(left_rows: Sequence[Mapping[str, Any]], right_rows: Sequence[Mapping[str, Any]]) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    left = {str(row.get("id")): row for row in left_rows}
    ids = [str(row.get("id")) for row in right_rows if str(row.get("id")) in left]
    return [left[qid] for qid in ids], [row for row in right_rows if str(row.get("id")) in left]


def compare_counts(left_rows: Sequence[Mapping[str, Any]], right_rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    left_overlap, right_overlap = overlap_rows(left_rows, right_rows)
    fixed = broken = both_correct = both_wrong = 0
    for left, right in zip(left_overlap, right_overlap):
        lc = correct(left)
        rc = correct(right)
        fixed += int((not lc) and rc)
        broken += int(lc and (not rc))
        both_correct += int(lc and rc)
        both_wrong += int((not lc) and (not rc))
    return {
        "fixed_by_right": fixed,
        "broken_by_right": broken,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
    }


def avg_bool(rows: Sequence[Mapping[str, Any]], key: str, default: bool = False) -> float:
    return sum(1.0 if row.get(key, default) else 0.0 for row in rows) / max(1, len(rows))


def eval_rows(rows: Sequence[Mapping[str, Any]], dataset: str, prompt: str) -> dict[str, Any]:
    return evaluate_predictions(list(rows), dataset=dataset, prompt_profile=prompt)


def row_mode(prompt: str, compaction: str) -> str:
    if prompt == "common_qa" and compaction == "none":
        return "full_common"
    if prompt == "qmrag_bundle_qa" and compaction == "none":
        return "full_bundle"
    if prompt == "strict_short_qa":
        return "full_strict_short"
    if prompt == "qmrag_bundle_short_qa" and compaction == "none":
        return "full_bundle_short"
    if prompt == "common_qa":
        return "common_compact"
    return "native_compact"


def completion_tokens(row: Mapping[str, Any]) -> int:
    for key in ("completion_tokens", "llm_usage_completion_tokens"):
        value = row.get(key)
        if value is not None:
            try:
                return int(value)
            except Exception:
                pass
    usage = row.get("llm_usage")
    if isinstance(usage, Mapping):
        value = usage.get("completion_tokens")
        if value is not None:
            try:
                return int(value)
            except Exception:
                pass
    return len(str(row.get("raw_prediction", row.get("prediction", "")) or "").split())


def is_yes_no_question(answers: Sequence[Any]) -> bool:
    norms = {normalize_answer(str(x)) for x in answers if str(x).strip()}
    return bool(norms) and norms.issubset({"yes", "no"})


def format_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pred_tokens: list[float] = []
    flags = {
        "long_answer": 0,
        "prefix": 0,
        "markdown": 0,
        "citation": 0,
        "explanation": 0,
        "yes_no_format_error": 0,
        "insufficient": 0,
        "answer_in_prediction": 0,
        "formatting_loss": 0,
        "idk": 0,
    }
    for row in rows:
        raw = str(row.get("raw_prediction", row.get("prediction", "")) or "")
        answers = [str(x) for x in row.get("answers", []) if str(x).strip()]
        tok = completion_tokens(row)
        words = len(raw.split())
        answer_hit = bool(answer_contains(raw, answers))
        em = bool(exact_match(raw, answers))
        pred_tokens.append(float(tok))
        flags["long_answer"] += int(tok > 16 or words > 12)
        flags["prefix"] += int(bool(PREFIX_RE.search(raw)))
        flags["markdown"] += int(bool(MARKDOWN_RE.search(raw)))
        flags["citation"] += int(bool(CITATION_RE.search(raw)))
        flags["explanation"] += int(bool(EXPLANATION_RE.search(raw)))
        flags["yes_no_format_error"] += int(is_yes_no_question(answers) and normalize_answer(raw) not in {"yes", "no"})
        flags["insufficient"] += int(is_insufficient_prediction(raw))
        flags["answer_in_prediction"] += int(answer_hit)
        flags["formatting_loss"] += int(answer_hit and not em)
        flags["idk"] += int(has_idk_phrase(raw))
    n = max(1, len(rows))
    return {
        "avg_prediction_tokens": sum(pred_tokens) / n,
        "median_prediction_tokens": float(statistics.median(pred_tokens)) if pred_tokens else 0.0,
        "long_answer_rate": flags["long_answer"] / n,
        "prefix_rate": flags["prefix"] / n,
        "markdown_rate": flags["markdown"] / n,
        "citation_rate": flags["citation"] / n,
        "explanation_rate": flags["explanation"] / n,
        "yes_no_format_error_rate": flags["yes_no_format_error"] / n,
        "insufficient_rate": flags["insufficient"] / n,
        "answer_in_prediction": flags["answer_in_prediction"] / n,
        "formatting_loss_rate": flags["formatting_loss"] / n,
        "idk_rate": flags["idk"] / n,
    }


def metric_row(
    dataset: str,
    path: Path | None,
    rows: list[dict[str, Any]],
    full_common_rows: list[dict[str, Any]],
    full_bundle_rows: list[dict[str, Any]],
    prompt: str,
    compaction: str,
    reference_kind: str | None = None,
) -> dict[str, Any]:
    ev = eval_rows(rows, dataset, prompt)
    fmt = format_metrics(rows)
    common_overlap, _ = overlap_rows(full_common_rows, rows)
    bundle_overlap, _ = overlap_rows(full_bundle_rows, rows) if full_bundle_rows else ([], [])
    common_eval = eval_rows(common_overlap, dataset, "common_qa") if common_overlap else {}
    bundle_eval = eval_rows(bundle_overlap, dataset, "qmrag_bundle_qa") if bundle_overlap else {}
    counts = compare_counts(full_common_rows, rows)
    baseline = BEST_BASELINES[dataset]
    f1 = float(ev.get("f1", 0.0) or 0.0)
    input_tok = float(ev.get("avg_input_prompt_tokens", 0.0) or 0.0)
    delta_common = f1 - float(common_eval.get("f1", 0.0) or 0.0) if common_eval else None
    delta_bundle = f1 - float(bundle_eval.get("f1", 0.0) or 0.0) if bundle_eval else None
    source_retrieval_ms = float(ev.get("retrieval_latency_ms", 0.0) or 0.0)
    generation_ms = float(ev.get("generation_latency_ms", 0.0) or 0.0)
    full_common_generation_ms = float(common_eval.get("generation_latency_ms", 0.0) or 0.0) if common_eval else generation_ms
    effective_total_ms = source_retrieval_ms + generation_ms
    full_common_effective_total_ms = source_retrieval_ms + full_common_generation_ms
    native_prompt_success = (
        (prompt == "strict_short_qa" and compaction == "none" and delta_common is not None and delta_common >= 0.0)
        or (prompt == "qmrag_bundle_short_qa" and compaction == "none" and delta_bundle is not None and delta_bundle >= 0.0)
    )
    common_success = prompt == "common_qa" and compaction != "none" and f1 > float(baseline["f1"]) and input_tok <= 1000.0
    aggressive_success = prompt == "common_qa" and compaction != "none" and f1 > float(baseline["f1"]) and input_tok <= 800.0
    row = {
        "dataset": dataset,
        "mode": reference_kind or row_mode(prompt, compaction),
        "compaction_profile": compaction,
        "prompt_profile": prompt,
        "n": len(rows),
        "EM": ev.get("em", 0.0),
        "F1": f1,
        "delta_F1_vs_full_common": delta_common,
        "delta_F1_vs_full_bundle": delta_bundle,
        "best_baseline_f1": baseline["f1"],
        "best_baseline_method": baseline["method"],
        "margin_vs_best_baseline": f1 - float(baseline["f1"]),
        "beats_best_baseline": f1 > float(baseline["f1"]),
        "answer_in_prediction": ev.get("answer_in_prediction", 0.0),
        "answer_in_rendered_context": ev.get("answer_in_rendered_context", 0.0),
        "insufficient_rate": ev.get("insufficient_rate", 0.0),
        "CtxTok": ev.get("avg_rendered_context_tokens", ev.get("context_tokens", 0.0)),
        "InputTok": input_tok,
        "TotalTok": ev.get("avg_total_llm_tokens", 0.0),
        "token_reduction_rate": ev.get("token_reduction_rate", 0.0) if compaction != "none" else 0.0,
        "F1_per_1k_context_tokens": ev.get("F1_per_1k_context_tokens", 0.0),
        "F1_per_1k_input_tokens": ev.get("F1_per_1k_input_prompt_tokens", 0.0),
        "source_retrieval_ms": source_retrieval_ms,
        "retrieval_ms": ev.get("retrieval_latency_ms", 0.0),
        "generation_ms": generation_ms,
        "total_ms": ev.get("latency_ms", 0.0),
        "effective_total_ms": effective_total_ms,
        "generation_ms_reduction_vs_full_common": full_common_generation_ms - generation_ms,
        "total_ms_reduction_vs_full_common_est": full_common_effective_total_ms - effective_total_ms,
        "avg_prediction_tokens": fmt.get("avg_prediction_tokens", 0.0),
        "long_answer_rate": fmt.get("long_answer_rate", 0.0),
        "prefix_rate": fmt.get("prefix_rate", 0.0),
        "markdown_rate": fmt.get("markdown_rate", 0.0),
        "citation_rate": fmt.get("citation_rate", 0.0),
        "explanation_rate": fmt.get("explanation_rate", 0.0),
        "formatting_loss_rate": fmt.get("formatting_loss_rate", 0.0),
        "evidence_bundles_hash_match_rate": avg_bool(rows, "evidence_bundles_hash_match", True if reference_kind else False),
        "rendered_context_hash_match_rate": avg_bool(rows, "rendered_context_hash_match", True if reference_kind else False),
        "fixed_by_right": counts["fixed_by_right"],
        "broken_by_right": counts["broken_by_right"],
        "both_correct": counts["both_correct"],
        "both_wrong": counts["both_wrong"],
        "common_compact_success": common_success,
        "aggressive_common_compact_success": aggressive_success,
        "native_compact_success": prompt != "common_qa" and compaction != "none" and input_tok <= 1000.0 and delta_bundle is not None and delta_bundle >= -0.06,
        "native_prompt_success": native_prompt_success,
        "qmrag_compact_paper_candidate": common_success or aggressive_success,
        "qmrang_compact_paper_candidate": common_success or aggressive_success,
        "strict_prompt_helpful": False,
        "path": str(path) if path else None,
    }
    return row


def baseline_row(dataset: str) -> dict[str, Any]:
    baseline = BEST_BASELINES[dataset]
    return {
        "dataset": dataset,
        "mode": "best_baseline",
        "compaction_profile": "best_baseline",
        "prompt_profile": "common_qa",
        "n": None,
        "EM": None,
        "F1": baseline["f1"],
        "delta_F1_vs_full_common": None,
        "delta_F1_vs_full_bundle": None,
        "best_baseline_f1": baseline["f1"],
        "best_baseline_method": baseline["method"],
        "margin_vs_best_baseline": 0.0,
        "beats_best_baseline": False,
        "answer_in_prediction": None,
        "answer_in_rendered_context": None,
        "insufficient_rate": None,
        "CtxTok": None,
        "InputTok": None,
        "TotalTok": None,
        "token_reduction_rate": None,
        "F1_per_1k_context_tokens": None,
        "F1_per_1k_input_tokens": None,
        "retrieval_ms": None,
        "source_retrieval_ms": None,
        "generation_ms": None,
        "total_ms": None,
        "effective_total_ms": None,
        "generation_ms_reduction_vs_full_common": None,
        "total_ms_reduction_vs_full_common_est": None,
        "avg_prediction_tokens": None,
        "long_answer_rate": None,
        "prefix_rate": None,
        "markdown_rate": None,
        "citation_rate": None,
        "explanation_rate": None,
        "formatting_loss_rate": None,
        "evidence_bundles_hash_match_rate": None,
        "rendered_context_hash_match_rate": None,
        "fixed_by_right": None,
        "broken_by_right": None,
        "common_compact_success": False,
        "aggressive_common_compact_success": False,
        "native_compact_success": False,
        "native_prompt_success": False,
        "qmrag_compact_paper_candidate": False,
        "qmrang_compact_paper_candidate": False,
        "strict_prompt_helpful": False,
        "path": None,
    }


def build_dataset_summary(root: Path, dataset: str) -> dict[str, Any]:
    full_common_path = find_full_run(root, dataset, "common_qa")
    if full_common_path is None:
        raise SystemExit(f"No full common_qa run found for dataset={dataset}")
    full_bundle_path = find_full_run(root, dataset, "qmrag_bundle_qa")
    full_common_rows = read_jsonl(full_common_path)
    full_bundle_rows = read_jsonl(full_bundle_path) if full_bundle_path else []
    rows = [
        metric_row(dataset, full_common_path, full_common_rows, full_common_rows, full_bundle_rows, "common_qa", "none", "full_common"),
        baseline_row(dataset),
    ]
    if full_bundle_rows:
        rows.append(metric_row(dataset, full_bundle_path, full_bundle_rows, full_common_rows, full_bundle_rows, "qmrag_bundle_qa", "none", "full_bundle"))
    replay_payloads = [(path, read_jsonl(path)) for path in find_latest_replays(root, dataset)]
    max_replay_n = max((len(replay_rows) for _, replay_rows in replay_payloads), default=0)
    if max_replay_n >= 1000:
        replay_payloads = [(path, replay_rows) for path, replay_rows in replay_payloads if len(replay_rows) >= 1000]
    for path, replay_rows in replay_payloads:
        first = replay_rows[0] if replay_rows else {}
        prompt = str(first.get("prompt_profile") or "UNKNOWN")
        compaction = row_compaction(first)
        rows.append(metric_row(dataset, path, replay_rows, full_common_rows, full_bundle_rows, prompt, compaction))

    full_common_f1 = float(rows[0].get("F1", 0.0) or 0.0)
    full_strict = next((row for row in rows if row["mode"] == "full_strict_short"), None)
    if full_strict:
        full_strict["strict_prompt_helpful"] = float(full_strict.get("F1", 0.0) or 0.0) > full_common_f1
        full_strict["native_prompt_success"] = bool(full_strict.get("strict_prompt_helpful"))
    full_bundle = next((row for row in rows if row["mode"] == "full_bundle"), None)
    full_bundle_short = next((row for row in rows if row["mode"] == "full_bundle_short"), None)
    common_success = [row for row in rows if row.get("common_compact_success")]
    aggressive_success = [row for row in rows if row.get("aggressive_common_compact_success")]
    native_success = [row for row in rows if row.get("native_compact_success")]
    native_prompt_success = [row for row in rows if row.get("native_prompt_success")]
    return {
        "dataset": dataset,
        "full_common_path": str(full_common_path),
        "full_bundle_path": str(full_bundle_path) if full_bundle_path else None,
        "best_baseline": BEST_BASELINES[dataset],
        "rows": rows,
        "interpretation": {
            "strict_short_qa_beats_common_qa": bool(full_strict and full_strict.get("strict_prompt_helpful")),
            "qmrag_bundle_short_qa_beats_qmrag_bundle_qa": bool(full_bundle and full_bundle_short and float(full_bundle_short["F1"]) > float(full_bundle["F1"])),
            "common_prompt_compact_beats_best_baseline": bool(common_success),
            "aggressive_common_prompt_compact_beats_best_baseline": bool(aggressive_success),
            "native_compact_recovers_performance": bool(native_success),
            "paper_candidate_profiles": [row["compaction_profile"] for row in common_success],
            "aggressive_paper_candidate_profiles": [row["compaction_profile"] for row in aggressive_success],
            "native_candidate_profiles": [f"{row['compaction_profile']}:{row['prompt_profile']}" for row in native_success],
            "native_prompt_success_profiles": [row["prompt_profile"] for row in native_prompt_success],
        },
    }


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown(summaries: Sequence[Mapping[str, Any]]) -> str:
    headers = [
        "dataset",
        "mode",
        "compaction_profile",
        "prompt_profile",
        "n",
        "EM",
        "F1",
        "dF1_common",
        "dF1_bundle",
        "best_base",
        "margin_base",
        "beats_base",
        "AnsPred",
        "AnsCtx",
        "Insuff",
        "CtxTok",
        "InputTok",
        "TotalTok",
        "TokenDown",
        "F1/1kCtx",
        "F1/1kInput",
        "src_retr_ms",
        "gen_ms",
        "eff_total_ms",
        "gen_down",
        "eff_total_down",
        "avg_pred_tok",
        "long_ans",
        "prefix",
        "markdown",
        "citation",
        "explain",
        "fmt_loss",
        "EBHash",
        "CtxHash",
        "fixed",
        "broken",
        "common_success",
        "aggr_success",
        "native_prompt_success",
        "native_success",
        "strict_helpful",
        "paper_candidate",
    ]
    lines = ["# Common Compact Short Prompt Summary", ""]
    for summary in summaries:
        interp = summary.get("interpretation", {})
        common_candidates = [row for row in summary.get("rows", []) if row.get("common_compact_success")]
        strict_row = next((row for row in summary.get("rows", []) if row.get("mode") == "full_strict_short"), None)
        bundle_short_row = next((row for row in summary.get("rows", []) if row.get("mode") == "full_bundle_short"), None)
        candidate_names = ", ".join(row["compaction_profile"] for row in common_candidates) or "none"
        candidate_margins = ", ".join(
            f"{row['compaction_profile']}={float(row['margin_vs_best_baseline']):.4f}" for row in common_candidates
        ) or "none"
        candidate_token_down = ", ".join(
            f"{row['compaction_profile']}={float(row['token_reduction_rate']):.4f}" for row in common_candidates
        ) or "none"
        candidate_f1_per_input = ", ".join(
            f"{row['compaction_profile']}={float(row['F1_per_1k_input_tokens']):.4f}" for row in common_candidates
        ) or "none"
        lines.extend(
            [
                f"## {summary['dataset']}",
                "",
                f"- full_common: {summary.get('full_common_path')}",
                f"- full_bundle: {summary.get('full_bundle_path')}",
                f"- best_baseline: {summary['best_baseline']['method']} F1={summary['best_baseline']['f1']:.4f}",
                "",
                "### Interpretation",
                "",
                f"1. strict_short_qa better than common_qa: {fmt(interp.get('strict_short_qa_beats_common_qa'))}",
                f"2. qmrag_bundle_short_qa better than qmrag_bundle_qa: {fmt(interp.get('qmrag_bundle_short_qa_beats_qmrag_bundle_qa'))}",
                f"3. common prompt compact beats best baseline: {fmt(interp.get('common_prompt_compact_beats_best_baseline'))}",
                f"4. aggressive compact beats best baseline: {fmt(interp.get('aggressive_common_prompt_compact_beats_best_baseline'))}",
                f"5. native compact prompt recovers compact performance: {fmt(interp.get('native_compact_recovers_performance'))}",
                f"6. QMRAG-Compact paper candidates: {', '.join(interp.get('paper_candidate_profiles') or []) or 'none'}",
                f"7. Aggressive candidates: {', '.join(interp.get('aggressive_paper_candidate_profiles') or []) or 'none'}",
                f"8. Native prompt successes: {', '.join(interp.get('native_prompt_success_profiles') or []) or 'none'}",
                "",
                "### N1000 Decision Notes",
                "",
                f"- QMRAG-Compact-common candidates: {candidate_names}",
                f"- Best-baseline margins: {candidate_margins}",
                f"- Token reduction: {candidate_token_down}",
                f"- F1/1K input: {candidate_f1_per_input}",
                f"- strict_short_qa effect: {fmt(strict_row.get('delta_F1_vs_full_common') if strict_row else None)}",
                f"- qmrag_bundle_short_qa effect: {fmt(bundle_short_row.get('delta_F1_vs_full_bundle') if bundle_short_row else None)}",
                "- Method/native prompt results should remain appendix unless the effect is consistently positive across datasets.",
                "- Failed profiles are those with negative margin, InputTok > 1000 for common compact, or increased insufficient/evidence loss.",
                "",
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join(["---"] * len(headers)) + " |",
            ]
        )
        for row in summary.get("rows", []):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("dataset")),
                        str(row.get("mode")),
                        str(row.get("compaction_profile")),
                        str(row.get("prompt_profile")),
                        fmt(row.get("n")),
                        fmt(row.get("EM")),
                        fmt(row.get("F1")),
                        fmt(row.get("delta_F1_vs_full_common")),
                        fmt(row.get("delta_F1_vs_full_bundle")),
                        fmt(row.get("best_baseline_f1")),
                        fmt(row.get("margin_vs_best_baseline")),
                        fmt(row.get("beats_best_baseline")),
                        fmt(row.get("answer_in_prediction")),
                        fmt(row.get("answer_in_rendered_context")),
                        fmt(row.get("insufficient_rate")),
                        fmt(row.get("CtxTok")),
                        fmt(row.get("InputTok")),
                        fmt(row.get("TotalTok")),
                        fmt(row.get("token_reduction_rate")),
                        fmt(row.get("F1_per_1k_context_tokens")),
                        fmt(row.get("F1_per_1k_input_tokens")),
                        fmt(row.get("source_retrieval_ms")),
                        fmt(row.get("generation_ms")),
                        fmt(row.get("effective_total_ms")),
                        fmt(row.get("generation_ms_reduction_vs_full_common")),
                        fmt(row.get("total_ms_reduction_vs_full_common_est")),
                        fmt(row.get("avg_prediction_tokens")),
                        fmt(row.get("long_answer_rate")),
                        fmt(row.get("prefix_rate")),
                        fmt(row.get("markdown_rate")),
                        fmt(row.get("citation_rate")),
                        fmt(row.get("explanation_rate")),
                        fmt(row.get("formatting_loss_rate")),
                        fmt(row.get("evidence_bundles_hash_match_rate")),
                        fmt(row.get("rendered_context_hash_match_rate")),
                        fmt(row.get("fixed_by_right")),
                        fmt(row.get("broken_by_right")),
                        fmt(row.get("common_compact_success")),
                        fmt(row.get("aggressive_common_compact_success")),
                        fmt(row.get("native_prompt_success")),
                        fmt(row.get("native_compact_success")),
                        fmt(row.get("strict_prompt_helpful")),
                        fmt(row.get("qmrag_compact_paper_candidate")),
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare common compact and short-style prompt replay runs.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--all-latest", action="store_true")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--analysis-root", default="outputs/analysis")
    parser.add_argument("--analysis-dir", default=None)
    args = parser.parse_args()
    datasets = ["hotpotqa", "2wiki", "popqa", "musique"] if args.all_latest or not args.dataset else [args.dataset]
    analysis_dir = Path(args.analysis_dir) if args.analysis_dir else Path(args.analysis_root) / now_timestamp()
    ensure_dir(analysis_dir)
    summaries = [build_dataset_summary(Path(args.output_root), dataset) for dataset in datasets]
    result = {"summaries": summaries}
    dump_json(result, analysis_dir / "common_compact_short_prompt_summary.json")
    dump_json(result, analysis_dir / "common_compact_short_prompt_n1000_summary.json")
    md = markdown(summaries)
    (analysis_dir / "common_compact_short_prompt_summary.md").write_text(md + "\n", encoding="utf-8")
    (analysis_dir / "common_compact_short_prompt_n1000_summary.md").write_text(md + "\n", encoding="utf-8")
    print(md)
    print(f"wrote: {analysis_dir}")


if __name__ == "__main__":
    main()
