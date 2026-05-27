#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.eval_metrics import answer_f1, exact_match, is_insufficient_prediction
from utils.generation import build_prompt
from utils.io_utils import dump_json, ensure_dir, read_jsonl, write_jsonl
from utils.text import normalize_answer, token_count


SETTING_NAMES = {
    ("p0_current", 3): "current_native_compact",
    ("p2_relaxed_chain", 3): "relaxed_native_compact",
    ("p2_relaxed_chain", 8): "relaxed_native_scaled",
    ("p3_minimal_extraction", 3): "minimal_native_compact",
}

SUMMARY_COLUMNS = [
    "dataset",
    "setting_name",
    "prompt_variant",
    "top_k",
    "n",
    "EM",
    "F1",
    "insufficient_information_rate",
    "avg_context_tokens",
    "avg_prompt_tokens",
    "avg_num_lines",
    "avg_num_evidence_sentences",
    "avg_empty_sections",
    "empty_section_rate",
    "avg_source_lines",
    "source_line_rate",
    "avg_metadata_lines",
    "metadata_line_rate",
    "avg_unique_anchors",
    "avg_duplicate_anchors",
    "duplicate_anchor_rate",
    "avg_supporting_evidence_sections",
    "avg_evidence_chain_sections",
    "avg_multi_anchor_sections",
    "answer_string_present_rate",
    "answer_present_but_wrong_rate",
    "answer_present_but_insufficient_rate",
    "render_omission_rate",
    "retrieval_miss_rate",
    "distractor_anchor_error_rate",
]

DELTA_COLUMNS = [
    "query_id",
    "dataset",
    "top3_prediction",
    "top8_prediction",
    "top3_F1",
    "top8_F1",
    "delta_F1",
    "top3_context_tokens",
    "top8_context_tokens",
    "delta_context_tokens",
    "top3_prompt_tokens",
    "top8_prompt_tokens",
    "delta_prompt_tokens",
    "top3_has_answer",
    "top8_has_answer",
    "new_anchors_added",
    "new_evidence_sentences_added",
    "top8_added_gold_answer_string",
    "top3_failure_type",
    "top8_failure_type",
]


def sha256_text(text: Any) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def parse_setting(value: str) -> tuple[str, int]:
    if ":" not in value:
        raise argparse.ArgumentTypeError(f"setting must be prompt_variant:top_k, got {value!r}")
    variant, top_k = value.split(":", 1)
    return variant.strip(), int(top_k)


def canonical_dataset(value: Any) -> str:
    text = str(value or "").strip()
    if text in {"2wiki", "2wikimultihopqa"}:
        return "2wikimultihopqa"
    return text


def setting_name(variant: str, top_k: int) -> str:
    return SETTING_NAMES.get((variant, top_k), f"{variant}_top{top_k}")


def as_answers(row: Mapping[str, Any]) -> list[str]:
    value = row.get("answers")
    if isinstance(value, list):
        return [str(x) for x in value if str(x).strip()]
    if value is not None and str(value).strip():
        return [str(value)]
    for key in ("answer", "gold_answer"):
        if row.get(key) is not None and str(row.get(key)).strip():
            return [str(row.get(key))]
    return []


def norm_contains(text: Any, answers: Sequence[str]) -> bool:
    norm_text = normalize_answer(str(text or ""))
    if not norm_text:
        return False
    for answer in answers:
        norm_answer = normalize_answer(answer)
        if norm_answer and norm_answer in norm_text:
            return True
    return False


def context_lines(context: str) -> list[str]:
    return [line.rstrip() for line in str(context or "").splitlines()]


def is_section_header(line: str) -> bool:
    stripped = line.strip()
    return (
        stripped.startswith("[Evidence Chain")
        or stripped.startswith("[Supporting Evidence")
        or stripped.startswith("[Multi-Anchor Evidence")
        or stripped.startswith("Evidence Chain")
        or stripped.startswith("Supporting Evidence")
        or stripped.startswith("Multi-Anchor Evidence")
    )


def is_metadata_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if is_section_header(stripped):
        return True
    if stripped.startswith(("Anchor:", "Bridge:", "Chain:", "Sources:", "Supporting Propositions:", "Title:")):
        return True
    if "relation_title=" in stripped or "score=" in stripped or "anchor_connected=" in stripped or "chain_complete_v2=" in stripped:
        return True
    return False


def is_source_line(line: str, in_sources: bool) -> bool:
    stripped = line.strip()
    return stripped == "Sources:" or stripped.startswith("Sources:") or (in_sources and stripped.startswith("- "))


def is_evidence_line(line: str) -> bool:
    return line.strip().startswith("- ")


def section_empty_count(lines: Sequence[str]) -> int:
    count = 0
    idx = 0
    while idx < len(lines):
        if not is_section_header(lines[idx]):
            idx += 1
            continue
        j = idx + 1
        has_evidence = False
        while j < len(lines) and not is_section_header(lines[j]):
            if is_evidence_line(lines[j]):
                has_evidence = True
            j += 1
        if not has_evidence:
            count += 1
        idx = j
    return count


def selected_bundles(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    bundles = list(row.get("evidence_bundles", []) or [])
    top_k = row.get("top_bundles")
    if top_k is None:
        return bundles
    try:
        return bundles[: max(0, int(top_k))]
    except Exception:
        return bundles


def bundle_id(bundle: Mapping[str, Any], index: int) -> str:
    return str(bundle.get("bundle_id") or bundle.get("id") or f"b{index}")


def bundle_anchors(bundle: Mapping[str, Any]) -> list[str]:
    anchors: list[str] = []
    if bundle.get("anchor_title"):
        anchors.append(str(bundle.get("anchor_title")))
    for anchor in bundle.get("anchor_titles", []) or []:
        if str(anchor).strip():
            anchors.append(str(anchor))
    return anchors


def selected_anchor_list(row: Mapping[str, Any]) -> list[str]:
    anchors: list[str] = []
    for bundle in selected_bundles(row):
        anchors.extend(bundle_anchors(bundle))
    return anchors


def evidence_sentences_from_context(context: str) -> list[str]:
    return [line.strip() for line in context_lines(context) if is_evidence_line(line)]


def first_answer_sentence_position(context: str, answers: Sequence[str]) -> int | None:
    position = 0
    for line in context_lines(context):
        if not is_evidence_line(line):
            continue
        position += 1
        if norm_contains(line, answers):
            return position
    return None


def context_stats(context: str, anchors: Sequence[str]) -> dict[str, Any]:
    lines = context_lines(context)
    non_empty_lines = [line for line in lines if line.strip()]
    in_sources = False
    source_lines = 0
    metadata_lines = 0
    evidence_lines = 0
    support_sections = 0
    chain_sections = 0
    multi_sections = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[Evidence Chain") or stripped.startswith("Evidence Chain"):
            chain_sections += 1
        if stripped.startswith("[Supporting Evidence") or stripped.startswith("Supporting Evidence"):
            support_sections += 1
        if stripped.startswith("[Multi-Anchor Evidence") or stripped.startswith("Multi-Anchor Evidence"):
            multi_sections += 1
        if stripped.startswith("Sources:") or stripped == "Sources:":
            in_sources = True
        elif is_section_header(stripped):
            in_sources = False
        if is_source_line(stripped, in_sources):
            source_lines += 1
        if is_metadata_line(stripped):
            metadata_lines += 1
        if is_evidence_line(stripped):
            evidence_lines += 1
    norm_anchors = [normalize_answer(anchor) for anchor in anchors if normalize_answer(anchor)]
    unique_anchors = sorted(set(norm_anchors))
    return {
        "num_lines": len(non_empty_lines),
        "num_empty_sections": section_empty_count(lines),
        "num_source_lines": source_lines,
        "num_metadata_lines": metadata_lines,
        "num_evidence_sentences": evidence_lines,
        "num_unique_anchors": len(unique_anchors),
        "num_duplicate_anchors": max(0, len(norm_anchors) - len(unique_anchors)),
        "num_supporting_evidence_sections": support_sections,
        "num_evidence_chain_sections": chain_sections,
        "num_multi_anchor_sections": multi_sections,
    }


def prediction_from_distractor(row: Mapping[str, Any], context: str, anchors: Sequence[str], answers: Sequence[str]) -> bool:
    pred = str(row.get("prediction") or row.get("raw_prediction") or "").strip()
    if not pred or is_insufficient_prediction(pred):
        return False
    if norm_contains(pred, answers):
        return False
    norm_pred = normalize_answer(pred)
    if not norm_pred:
        return False
    if norm_pred in normalize_answer(context):
        return True
    return any(normalize_answer(anchor) and normalize_answer(anchor) in norm_pred for anchor in anchors)


def classify_failure(
    em: float,
    insufficient: bool,
    answer_in_raw: bool,
    answer_in_context: bool,
    stats: Mapping[str, Any],
    distractor: bool,
) -> str:
    if em > 0:
        return "correct"
    if not answer_in_raw and not answer_in_context:
        return "retrieval_miss"
    if answer_in_raw and not answer_in_context:
        return "render_omission"
    if answer_in_context and insufficient:
        return "answer_present_but_insufficient"
    if answer_in_context:
        return "answer_present_but_wrong"
    num_lines = max(1, int(stats.get("num_lines", 0) or 0))
    noisy = (
        int(stats.get("num_empty_sections", 0) or 0) > 0
        or float(stats.get("num_metadata_lines", 0) or 0) / num_lines >= 0.4
        or float(stats.get("num_source_lines", 0) or 0) / num_lines >= 0.4
    )
    if noisy:
        return "empty_or_metadata_noise"
    if distractor:
        return "distractor_anchor_error"
    return "unknown"


def final_prompt_for_row(row: Mapping[str, Any], context: str) -> str:
    prompt_profile = str(row.get("prompt_profile") or "common_qa")
    try:
        return build_prompt(str(row.get("question") or ""), context, prompt_profile)
    except Exception:
        return f"Question: {row.get('question') or ''}\nContext:\n{context}\nAnswer:"


def audit_row(row: Mapping[str, Any], setting: tuple[str, int]) -> dict[str, Any]:
    variant, top_k = setting
    answers = as_answers(row)
    context = str(row.get("rendered_context") or row.get("rendered_context_preview") or "")
    prediction = str(row.get("prediction") or row.get("raw_prediction") or "")
    em = exact_match(prediction, answers)
    f1 = answer_f1(prediction, answers)
    insufficient = is_insufficient_prediction(prediction)
    anchors = selected_anchor_list(row)
    stats = context_stats(context, anchors)
    raw_bundles = row.get("evidence_bundles", []) or []
    raw_json = json.dumps(raw_bundles, ensure_ascii=False)
    answer_in_raw = norm_contains(raw_json, answers)
    answer_in_context = norm_contains(context, answers)
    distractor = prediction_from_distractor(row, context, anchors, answers)
    prompt = final_prompt_for_row(row, context)
    selected = selected_bundles(row)
    record = {
        "dataset": canonical_dataset(row.get("dataset")),
        "query_id": str(row.get("id") or row.get("query_id") or ""),
        "question": str(row.get("question") or ""),
        "gold_answer": answers[0] if answers else "",
        "gold_answers": answers,
        "prediction": prediction,
        "normalized_prediction": normalize_answer(prediction),
        "EM": em,
        "F1": f1,
        "is_correct": bool(em > 0),
        "is_insufficient": bool(insufficient),
        "prompt_variant": variant,
        "top_k": top_k,
        "setting_name": setting_name(variant, top_k),
        "context_tokens": int(row.get("rendered_context_tokens") or row.get("actual_context_tokens") or token_count(context)),
        "prompt_tokens": int(row.get("input_prompt_tokens") or token_count(prompt)),
        "evidence_bundles_hash": str(row.get("evidence_bundles_hash") or sha256_text(raw_json)),
        "rendered_context_hash": str(row.get("rendered_context_hash") or sha256_text(context)),
        "final_prompt_hash": sha256_text(prompt),
        "stored_prompt_hash": str(row.get("prompt_hash") or ""),
        "final_prompt_hash_matches_stored": bool(str(row.get("prompt_hash") or "") == sha256_text(prompt)) if row.get("prompt_hash") else None,
        "raw_evidence_bundle_json": raw_bundles,
        "rendered_context": context,
        "final_prompt_text": prompt,
        "selected_bundle_ids": [bundle_id(bundle, i) for i, bundle in enumerate(selected)],
        "selected_anchors": anchors,
        **stats,
        "has_gold_answer_string_in_raw_evidence": bool(answer_in_raw),
        "has_gold_answer_string_in_context": bool(answer_in_context),
        "answer_sentence_position": first_answer_sentence_position(context, answers),
        "failure_type": classify_failure(em, insufficient, answer_in_raw, answer_in_context, stats, distractor),
    }
    return record


def discover_prediction_files(input_root: Path, datasets: Sequence[str], settings: Sequence[tuple[str, int]]) -> dict[tuple[str, str, int], Path]:
    wanted_datasets = {canonical_dataset(x) for x in datasets}
    wanted_settings = set(settings)
    candidates: dict[tuple[str, str, int], list[Path]] = defaultdict(list)
    for path in input_root.rglob("predictions.jsonl"):
        try:
            with path.open("r", encoding="utf-8") as handle:
                first_line = next((line for line in handle if line.strip()), "")
            if not first_line:
                continue
            first = json.loads(first_line)
        except Exception:
            continue
        dataset = canonical_dataset(first.get("dataset") or path.parent.parent.name)
        variant = str(first.get("ace_native_prompt_variant") or "")
        try:
            top_k = int(first.get("top_bundles") or 0)
        except Exception:
            top_k = 0
        if dataset in wanted_datasets and (variant, top_k) in wanted_settings:
            candidates[(dataset, variant, top_k)].append(path)

    selected: dict[tuple[str, str, int], Path] = {}
    for key, paths in candidates.items():
        def score(path: Path) -> tuple[int, float, str]:
            parts = set(path.parts)
            stage_score = 100 if "core_n1000" in parts else 80 if "appendix_4ds_p2_top8_n1000" in parts else 20
            setting = setting_name(key[1], key[2])
            if setting in parts:
                stage_score += 20
            return (stage_score, path.stat().st_mtime, str(path))
        selected[key] = sorted(paths, key=score, reverse=True)[0]
    return selected


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def avg(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    vals = []
    for row in rows:
        value = row.get(key)
        if value is None or value == "":
            continue
        try:
            vals.append(float(value))
        except Exception:
            continue
    return mean(vals) if vals else 0.0


def aggregate(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(str(record["dataset"]), str(record["setting_name"]), str(record["prompt_variant"]), int(record["top_k"]))].append(record)
    rows: list[dict[str, Any]] = []
    for (dataset, setting, variant, top_k), group in sorted(grouped.items()):
        n = len(group)
        lines = [max(1.0, float(row.get("num_lines", 0) or 0)) for row in group]
        rows.append(
            {
                "dataset": dataset,
                "setting_name": setting,
                "prompt_variant": variant,
                "top_k": top_k,
                "n": n,
                "EM": avg(group, "EM"),
                "F1": avg(group, "F1"),
                "insufficient_information_rate": sum(1 for row in group if row.get("is_insufficient")) / max(1, n),
                "avg_context_tokens": avg(group, "context_tokens"),
                "avg_prompt_tokens": avg(group, "prompt_tokens"),
                "avg_num_lines": avg(group, "num_lines"),
                "avg_num_evidence_sentences": avg(group, "num_evidence_sentences"),
                "avg_empty_sections": avg(group, "num_empty_sections"),
                "empty_section_rate": sum(1 for row in group if float(row.get("num_empty_sections", 0) or 0) > 0) / max(1, n),
                "avg_source_lines": avg(group, "num_source_lines"),
                "source_line_rate": mean([float(row.get("num_source_lines", 0) or 0) / line for row, line in zip(group, lines)]) if group else 0.0,
                "avg_metadata_lines": avg(group, "num_metadata_lines"),
                "metadata_line_rate": mean([float(row.get("num_metadata_lines", 0) or 0) / line for row, line in zip(group, lines)]) if group else 0.0,
                "avg_unique_anchors": avg(group, "num_unique_anchors"),
                "avg_duplicate_anchors": avg(group, "num_duplicate_anchors"),
                "duplicate_anchor_rate": sum(1 for row in group if float(row.get("num_duplicate_anchors", 0) or 0) > 0) / max(1, n),
                "avg_supporting_evidence_sections": avg(group, "num_supporting_evidence_sections"),
                "avg_evidence_chain_sections": avg(group, "num_evidence_chain_sections"),
                "avg_multi_anchor_sections": avg(group, "num_multi_anchor_sections"),
                "answer_string_present_rate": sum(1 for row in group if row.get("has_gold_answer_string_in_context")) / max(1, n),
                "answer_present_but_wrong_rate": sum(1 for row in group if row.get("failure_type") == "answer_present_but_wrong") / max(1, n),
                "answer_present_but_insufficient_rate": sum(1 for row in group if row.get("failure_type") == "answer_present_but_insufficient") / max(1, n),
                "render_omission_rate": sum(1 for row in group if row.get("failure_type") == "render_omission") / max(1, n),
                "retrieval_miss_rate": sum(1 for row in group if row.get("failure_type") == "retrieval_miss") / max(1, n),
                "distractor_anchor_error_rate": sum(1 for row in group if row.get("failure_type") == "distractor_anchor_error") / max(1, n),
            }
        )
    return rows


def fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        vals = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                value = fmt(value)
            vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def raw_bundle_summary(record: Mapping[str, Any], max_bundles: int = 8) -> str:
    lines = []
    bundles = record.get("raw_evidence_bundle_json", []) or []
    selected_ids = set(record.get("selected_bundle_ids", []) or [])
    for i, bundle in enumerate(bundles[:max_bundles]):
        bid = bundle_id(bundle, i)
        marker = "*" if bid in selected_ids else "-"
        props = len(bundle.get("propositions", []) or [])
        chunks = len(bundle.get("source_chunks", []) or [])
        bridges = ", ".join(str(x) for x in bundle.get("bridge_titles", []) or [])
        anchors = "; ".join(bundle_anchors(bundle))
        lines.append(f"{marker} {bid}: type={bundle.get('bundle_type')} anchor={anchors} bridge={bridges} props={props} chunks={chunks} score={bundle.get('score')}")
    return "\n".join(lines) if lines else "NA"


def prompt_excerpt(prompt: str, max_head: int = 1800, max_tail: int = 900) -> str:
    text = str(prompt or "")
    if len(text) <= max_head + max_tail + 80:
        return text
    return text[:max_head].rstrip() + "\n\n...[truncated prompt middle]...\n\n" + text[-max_tail:].lstrip()


def context_excerpt(context: str, max_chars: int = 18000) -> str:
    text = str(context or "")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n...[truncated rendered context]..."


def sample_block(title: str, records: Sequence[Mapping[str, Any]], limit: int = 3) -> str:
    lines = [f"## {title}", ""]
    if not records:
        lines.extend(["No examples found.", ""])
        return "\n".join(lines)
    for record in records[:limit]:
        lines.extend(
            [
                f"### {record.get('dataset')} / {record.get('query_id')}",
                "",
                f"Dataset: {record.get('dataset')}",
                f"Query ID: {record.get('query_id')}",
                f"Question: {record.get('question')}",
                f"Gold answer: {record.get('gold_answer')}",
                f"Prediction: {record.get('prediction')}",
                f"EM / F1: {fmt(record.get('EM'))} / {fmt(record.get('F1'))}",
                f"Prompt variant: {record.get('prompt_variant')}",
                f"Top-k: {record.get('top_k')}",
                f"Context tokens: {record.get('context_tokens')}",
                f"Prompt tokens: {record.get('prompt_tokens')}",
                f"Selected anchors: {', '.join(record.get('selected_anchors') or [])}",
                f"Failure type: {record.get('failure_type')}",
                f"Diagnostic notes: answer_in_context={record.get('has_gold_answer_string_in_context')}, answer_position={record.get('answer_sentence_position')}, empty_sections={record.get('num_empty_sections')}, metadata_lines={record.get('num_metadata_lines')}, source_lines={record.get('num_source_lines')}",
                "",
                "Rendered Context:",
                "```text",
                context_excerpt(str(record.get("rendered_context") or "")),
                "```",
                "",
                "Final Prompt Excerpt:",
                "```text",
                prompt_excerpt(str(record.get("final_prompt_text") or "")),
                "```",
                "",
                "Raw Evidence Bundle Summary:",
                "```text",
                raw_bundle_summary(record),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def write_samples(path: Path, records: Sequence[Mapping[str, Any]], delta_rows: Sequence[Mapping[str, Any]]) -> None:
    by_key = {(str(r["dataset"]), str(r["query_id"]), str(r["prompt_variant"]), int(r["top_k"])): r for r in records}
    p2_top8 = [r for r in records if r.get("prompt_variant") == "p2_relaxed_chain" and int(r.get("top_k", 0)) == 8]
    blocks = [
        sample_block("p2 top8 correct cases", [r for r in p2_top8 if r.get("is_correct")]),
        sample_block("p2 top8 wrong cases", [r for r in p2_top8 if not r.get("is_correct") and not r.get("is_insufficient")]),
        sample_block("p2 top8 insufficient cases", [r for r in p2_top8 if r.get("is_insufficient")]),
    ]
    top3_to_top8 = []
    p0_to_top8 = []
    for delta in delta_rows:
        if float(delta.get("top3_F1", 0.0) or 0.0) == 0.0 and float(delta.get("top8_F1", 0.0) or 0.0) > 0.0:
            key = (str(delta["dataset"]), str(delta["query_id"]), "p2_relaxed_chain", 8)
            if key in by_key:
                top3_to_top8.append(by_key[key])
    p0_by_id = {(str(r["dataset"]), str(r["query_id"])): r for r in records if r.get("prompt_variant") == "p0_current" and int(r.get("top_k", 0)) == 3}
    for r in p2_top8:
        p0 = p0_by_id.get((str(r["dataset"]), str(r["query_id"])))
        if p0 and float(p0.get("F1", 0.0) or 0.0) == 0.0 and float(r.get("F1", 0.0) or 0.0) > 0.0:
            p0_to_top8.append(r)
    blocks.extend(
        [
            sample_block("p2 top3 wrong but p2 top8 correct cases", top3_to_top8),
            sample_block("p0 top3 wrong but p2 top8 correct cases", p0_to_top8),
            sample_block("p2 top8 answer-present-but-wrong cases", [r for r in p2_top8 if r.get("failure_type") == "answer_present_but_wrong"]),
            sample_block("p2 top8 answer-present-but-insufficient cases", [r for r in p2_top8 if r.get("failure_type") == "answer_present_but_insufficient"]),
            sample_block("top8 and top10 rendered-context-identical cases if available", []),
        ]
    )
    path.write_text("# ACE-RAG Context Audit Samples\n\n" + "\n".join(blocks), encoding="utf-8")


def records_by_dataset_id(records: Sequence[Mapping[str, Any]], variant: str, top_k: int) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {
        (str(r["dataset"]), str(r["query_id"])): r
        for r in records
        if r.get("prompt_variant") == variant and int(r.get("top_k", 0)) == top_k
    }


def topk_delta(records: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    top3 = records_by_dataset_id(records, "p2_relaxed_chain", 3)
    top8 = records_by_dataset_id(records, "p2_relaxed_chain", 8)
    rows: list[dict[str, Any]] = []
    for key in sorted(set(top3) & set(top8)):
        r3 = top3[key]
        r8 = top8[key]
        anchors3 = {normalize_answer(x) for x in r3.get("selected_anchors", []) or [] if normalize_answer(x)}
        anchors8 = {normalize_answer(x) for x in r8.get("selected_anchors", []) or [] if normalize_answer(x)}
        sents3 = set(evidence_sentences_from_context(str(r3.get("rendered_context") or "")))
        sents8 = set(evidence_sentences_from_context(str(r8.get("rendered_context") or "")))
        new_sentences = sorted(sents8 - sents3)
        top8_added_answer = bool((not r3.get("has_gold_answer_string_in_context")) and r8.get("has_gold_answer_string_in_context"))
        rows.append(
            {
                "query_id": key[1],
                "dataset": key[0],
                "top3_prediction": r3.get("prediction"),
                "top8_prediction": r8.get("prediction"),
                "top3_F1": r3.get("F1"),
                "top8_F1": r8.get("F1"),
                "delta_F1": float(r8.get("F1", 0.0) or 0.0) - float(r3.get("F1", 0.0) or 0.0),
                "top3_context_tokens": r3.get("context_tokens"),
                "top8_context_tokens": r8.get("context_tokens"),
                "delta_context_tokens": int(r8.get("context_tokens", 0) or 0) - int(r3.get("context_tokens", 0) or 0),
                "top3_prompt_tokens": r3.get("prompt_tokens"),
                "top8_prompt_tokens": r8.get("prompt_tokens"),
                "delta_prompt_tokens": int(r8.get("prompt_tokens", 0) or 0) - int(r3.get("prompt_tokens", 0) or 0),
                "top3_has_answer": bool(r3.get("has_gold_answer_string_in_context")),
                "top8_has_answer": bool(r8.get("has_gold_answer_string_in_context")),
                "new_anchors_added": "; ".join(sorted(anchors8 - anchors3)),
                "new_evidence_sentences_added": len(new_sentences),
                "top8_added_gold_answer_string": top8_added_answer,
                "top3_failure_type": r3.get("failure_type"),
                "top8_failure_type": r8.get("failure_type"),
            }
        )
    summary = {
        "top3_wrong_top8_correct_count": sum(1 for r in rows if r["top3_failure_type"] != "correct" and r["top8_failure_type"] == "correct"),
        "top3_insufficient_top8_correct_count": sum(1 for r in rows if r["top3_failure_type"] == "answer_present_but_insufficient" and r["top8_failure_type"] == "correct"),
        "top3_no_answer_top8_answer_present_count": sum(1 for r in rows if not r["top3_has_answer"] and r["top8_has_answer"]),
        "top8_no_gain_despite_answer_present_count": sum(1 for r in rows if r["top8_has_answer"] and float(r["delta_F1"] or 0.0) <= 0.0),
        "top8_added_answer_evidence_count": sum(1 for r in rows if r["top8_added_gold_answer_string"]),
        "top8_only_added_noise_count": sum(
            1
            for r in rows
            if r["top3_failure_type"] != "correct"
            and r["top8_failure_type"] != "correct"
            and float(r["delta_context_tokens"] or 0.0) > 0.0
            and not r["top8_added_gold_answer_string"]
            and float(r["delta_F1"] or 0.0) <= 0.0
        ),
    }
    return rows, summary


def write_topk_delta(root: Path, rows: Sequence[Mapping[str, Any]], summary: Mapping[str, Any]) -> None:
    write_csv(root / "topk_context_delta.csv", rows, DELTA_COLUMNS)
    lines = [
        "# Top-k Context Delta",
        "",
        "Comparison: `p2_relaxed_chain top3` vs `p2_relaxed_chain top8`.",
        "",
        "## Summary Counts",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Per-query Delta", "", markdown_table(rows[:200], DELTA_COLUMNS)])
    (root / "topk_context_delta.md").write_text("\n".join(lines), encoding="utf-8")


def def_ref(path: str, name: str) -> str:
    file_path = ROOT / path
    pattern = re.compile(rf"^def\s+{re.escape(name)}\s*\(")
    try:
        for idx, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
            if pattern.match(line):
                return f"{path}:{idx}"
    except Exception:
        return f"{path}:NA"
    return f"{path}:NA"


def write_codepath_doc(root: Path) -> None:
    refs = {
        "build_or_load_index": def_ref("main.py", "build_or_load_index"),
        "add_generation_logging_fields": def_ref("main.py", "add_generation_logging_fields"),
        "ordered_bundles_for_context": def_ref("scripts/replay_generation.py", "ordered_bundles_for_context"),
        "render_context_with_truncation": def_ref("scripts/replay_generation.py", "render_context_with_truncation"),
        "replay_rows": def_ref("scripts/replay_generation.py", "replay_rows"),
        "_ordered_bundles": def_ref("utils/generation.py", "_ordered_bundles"),
        "_render_structured_chain": def_ref("utils/generation.py", "_render_structured_chain"),
        "_render_chain_dedup": def_ref("utils/generation.py", "_render_chain_dedup"),
        "_render_compacted_context": def_ref("utils/generation.py", "_render_compacted_context"),
        "render_context": def_ref("utils/generation.py", "render_context"),
        "render_context_with_metadata": def_ref("utils/generation.py", "render_context_with_metadata"),
        "build_prompt": def_ref("utils/generation.py", "build_prompt"),
        "build_generation_prompt": def_ref("utils/generation.py", "build_generation_prompt"),
        "add_token_accounting_fields": def_ref("utils/generation.py", "add_token_accounting_fields"),
    }
    lines = [
        "# ACE-RAG Context Rendering Code Path",
        "",
        "This audit documents the existing code path only. No retrieval, graph construction, scoring, evidence bundle construction, prompt templates, or generation behavior was modified.",
        "",
        "## Files And Functions",
        "",
    ]
    for name, ref in refs.items():
        lines.append(f"- `{name}`: `{ref}`")
    lines.extend(
        [
            "",
            "## Retrieval And Evidence Bundles",
            "",
            f"- Index loading/reuse happens in `main.build_or_load_index` at `{refs['build_or_load_index']}`.",
            "- The final verification outputs used here are replay outputs. They keep `evidence_bundles` from the source common-prompt structured-chain run.",
            f"- `scripts.replay_generation.replay_rows` at `{refs['replay_rows']}` copies each source row, reads `row['evidence_bundles']`, and writes `evidence_bundles_hash`, `source_evidence_bundles_hash`, and `evidence_bundles_hash_match=True`.",
            "",
            "## Where top_k / top_bundles Is Applied",
            "",
            f"- `render_context_with_truncation` at `{refs['render_context_with_truncation']}` receives `top_bundles`.",
            "- It calls `ordered_bundles_for_context`, then selects `candidates = ordered[:top_bundles]` when `top_bundles` is not `None`.",
            "- For current ordering, `ordered_bundles_for_context` returns `list(bundles)` without re-ranking.",
            "- For `top3_chain_dedup`, replay also sets an effective top-bundle cap of 3 before rendering.",
            "",
            "## Ordering And Deduplication",
            "",
            f"- `_ordered_bundles` at `{refs['_ordered_bundles']}` sorts bundles by ordering group, chain completeness, anchor connectivity, residual coverage, score, and original index.",
            f"- `_render_chain_dedup` at `{refs['_render_chain_dedup']}` serializes selected bundles while keeping a `seen` sentence set.",
            "- Sentence deduplication happens during rendering, so top8 and top10 can become identical when extra bundles add only duplicate sentences, metadata-only material, or no additional selected bundles exist.",
            "- Bundle saturation can also make top8/top10 identical because many source rows contain only six evidence bundles.",
            "",
            "## Context Serialization",
            "",
            f"- Full structured-chain rendering is in `_render_structured_chain` at `{refs['_render_structured_chain']}`.",
            "- It emits `[Evidence Chain ...]`, `Anchor:`, `Bridge:`, `Chain:`, `Supporting Propositions:`, `Sources:`, and `[Supporting Evidence ... relation_title=...]` lines.",
            f"- Compact rendering is dispatched by `_render_compacted_context` at `{refs['_render_compacted_context']}`.",
            "- `chain_dedup` and `top3_chain_dedup` both use `_render_chain_dedup`; `top3_chain_dedup` first slices `_ordered_bundles(bundles)[:3]`.",
            "- `Sources:` in `chain_dedup` are compact source references, not full source chunks.",
            "- `relation_title=False` is emitted in supporting-evidence section headers by both structured and chain-dedup renderers.",
            "",
            "## Final Prompt Assembly",
            "",
            f"- `build_prompt` at `{refs['build_prompt']}` formats the selected prompt template with `{{question}}` and `{{context}}`.",
            f"- `build_generation_prompt` at `{refs['build_generation_prompt']}` calls `render_context_with_metadata`, then `build_prompt`.",
            f"- Replay final prompt assembly happens in `replay_rows` at `{refs['replay_rows']}` via `build_prompt(question, context, target_prompt)`.",
            "",
            "## Token Accounting",
            "",
            f"- `add_token_accounting_fields` at `{refs['add_token_accounting_fields']}` records rendered context tokens, input prompt tokens, completion tokens, total LLM tokens, and token count source.",
            f"- `main.add_generation_logging_fields` at `{refs['add_generation_logging_fields']}` stores `rendered_context_hash`, `prompt_hash`, and optional full rendered context/prompt depending on logging config.",
            "- Replay outputs store full `rendered_context` and `prompt_hash`; the full final prompt text was not stored, so this audit reconstructs it with the same `build_prompt` call.",
            "",
        ]
    )
    (root / "context_rendering_codepath.md").write_text("\n".join(lines), encoding="utf-8")


def write_summary(root: Path, rows: Sequence[Mapping[str, Any]], files: Mapping[tuple[str, str, int], Path]) -> None:
    write_csv(root / "context_audit_summary.csv", rows, SUMMARY_COLUMNS)
    lines = [
        "# ACE-RAG Context Rendering Audit Summary",
        "",
        f"- created_at: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Resolved Prediction Files",
        "",
    ]
    for key, path in sorted(files.items()):
        lines.append(f"- {key[0]} / {key[1]} top{key[2]}: `{path}`")
    lines.extend(["", "## Aggregate Diagnostics", "", markdown_table(rows, SUMMARY_COLUMNS)])
    (root / "context_audit_summary.md").write_text("\n".join(lines), encoding="utf-8")


def metric_lookup(summary_rows: Sequence[Mapping[str, Any]], variant: str, top_k: int, metric: str) -> float:
    vals = [
        float(row.get(metric, 0.0) or 0.0)
        for row in summary_rows
        if row.get("prompt_variant") == variant and int(row.get("top_k", 0)) == top_k
    ]
    return mean(vals) if vals else 0.0


def write_risk_report(root: Path, summary_rows: Sequence[Mapping[str, Any]], delta_summary: Mapping[str, int]) -> None:
    empty_rate = metric_lookup(summary_rows, "p2_relaxed_chain", 8, "empty_section_rate")
    metadata_rate = metric_lookup(summary_rows, "p2_relaxed_chain", 8, "metadata_line_rate")
    source_rate = metric_lookup(summary_rows, "p2_relaxed_chain", 8, "source_line_rate")
    duplicate_rate = metric_lookup(summary_rows, "p2_relaxed_chain", 8, "duplicate_anchor_rate")
    answer_wrong = metric_lookup(summary_rows, "p2_relaxed_chain", 8, "answer_present_but_wrong_rate")
    answer_insuff = metric_lookup(summary_rows, "p2_relaxed_chain", 8, "answer_present_but_insufficient_rate")
    render_omission = metric_lookup(summary_rows, "p2_relaxed_chain", 8, "render_omission_rate")
    retrieval_miss = metric_lookup(summary_rows, "p2_relaxed_chain", 8, "retrieval_miss_rate")
    present_rate = metric_lookup(summary_rows, "p2_relaxed_chain", 8, "answer_string_present_rate")
    added_answer = int(delta_summary.get("top8_added_answer_evidence_count", 0) or 0)
    only_noise = int(delta_summary.get("top8_only_added_noise_count", 0) or 0)
    wrong_to_correct = int(delta_summary.get("top3_wrong_top8_correct_count", 0) or 0)

    recommendations: list[str] = []
    if render_omission >= 0.05:
        recommendations.append("Fix renderer omission before broader renderer variants.")
    if retrieval_miss >= 0.4:
        recommendations.append("Renderer-only ablation will not solve all failures; retrieval or chain expansion still limits many cases.")
    if empty_rate >= 0.2 or metadata_rate >= 0.35:
        recommendations.append("Test R1 clean sentence renderer first, because metadata/section overhead is high.")
    if answer_wrong >= 0.15:
        recommendations.append("Test R2 title-paragraph or R3 chain-paragraph hybrid, because answers are often present but not extracted correctly.")
    if answer_insuff >= 0.03:
        recommendations.append("Keep p2, but expose answer evidence more directly in rendering to reduce insufficient outputs.")
    if not recommendations:
        recommendations.append("Current format risks are moderate; start with a conservative R1 cleanup before more semantic renderers.")
    if added_answer > 0 and added_answer >= max(1, only_noise // 2):
        topk_conclusion = "top8 is justified as native-performance setting because it adds missing answer evidence in a non-trivial number of cases."
    elif only_noise > added_answer:
        topk_conclusion = "top8 often adds context without answer-evidence gain; reconsider top8 or add pruning/reranking diagnostics."
    else:
        topk_conclusion = "top8 benefit appears mixed; inspect samples before deciding between R1/R2/R3."

    lines = [
        "# Context Format Risk Report",
        "",
        "## Key Rates For p2_relaxed_chain top8",
        "",
        f"- empty_section_rate: {fmt(empty_rate)}",
        f"- metadata_line_rate: {fmt(metadata_rate)}",
        f"- source_line_rate: {fmt(source_rate)}",
        f"- duplicate_anchor_rate: {fmt(duplicate_rate)}",
        f"- answer_string_present_rate: {fmt(present_rate)}",
        f"- answer_present_but_wrong_rate: {fmt(answer_wrong)}",
        f"- answer_present_but_insufficient_rate: {fmt(answer_insuff)}",
        f"- render_omission_rate: {fmt(render_omission)}",
        f"- retrieval_miss_rate: {fmt(retrieval_miss)}",
        "",
        "## Top-k Interpretation",
        "",
        f"- top3_wrong_top8_correct_count: {wrong_to_correct}",
        f"- top8_added_answer_evidence_count: {added_answer}",
        f"- top8_only_added_noise_count: {only_noise}",
        f"- Conclusion: {topk_conclusion}",
        "",
        "## Questions",
        "",
        f"- Are empty sections common? {'Yes' if empty_rate >= 0.2 else 'No'}",
        f"- Are Sources lines common and useful? Common={source_rate >= 0.1}; usefulness requires sample inspection.",
        f"- Are metadata fields such as relation_title=False frequently included? {'Yes' if metadata_rate >= 0.35 else 'Moderate/No'}",
        f"- Are unrelated anchors common? Duplicate-anchor proxy={fmt(duplicate_rate)}; inspect samples for true unrelated anchors.",
        f"- Does the gold answer often appear in rendered context but still yield wrong or insufficient output? {'Yes' if answer_wrong + answer_insuff >= 0.2 else 'No'}",
        f"- Does top8 improve because it adds answer evidence or because of prompt/context effects? {topk_conclusion}",
        f"- Is current ACE-RAG context closer to a debug format than a natural reading-comprehension format? {'Yes' if metadata_rate >= 0.35 else 'Partly'}",
        "",
        "## Renderer Recommendation",
        "",
    ]
    lines.extend([f"- {x}" for x in recommendations])
    lines.extend(
        [
            "",
            "## Final Recommendation",
            "",
            "A render-only ablation is justified with fixed retrieval bundles and fixed p2/top8 generation. Choose the first renderer according to the dominant risk above; in the current report, prioritize the first bullet under Renderer Recommendation.",
            "",
        ]
    )
    (root / "context_format_risk_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_reuse(args: argparse.Namespace) -> Path:
    output_root = ensure_dir(args.output_root)
    settings = [parse_setting(x) for x in args.settings]
    files = discover_prediction_files(Path(args.input_root), args.datasets, settings)
    missing = []
    for dataset in [canonical_dataset(x) for x in args.datasets]:
        for variant, top_k in settings:
            if (dataset, variant, top_k) not in files:
                missing.append((dataset, variant, top_k))
    if missing:
        raise SystemExit(f"Missing predictions for settings: {missing}")

    write_codepath_doc(output_root)
    records: list[dict[str, Any]] = []
    for dataset in [canonical_dataset(x) for x in args.datasets]:
        for setting in settings:
            path = files[(dataset, setting[0], setting[1])]
            rows = read_jsonl(path)
            if args.limit:
                rows = rows[: int(args.limit)]
            for row in rows:
                records.append(audit_row(row, setting))
    write_jsonl(records, output_root / "query_level_context_audit.jsonl")
    summary_rows = aggregate(records)
    write_summary(output_root, summary_rows, files)
    delta_rows, delta_summary = topk_delta(records)
    write_topk_delta(output_root, delta_rows, delta_summary)
    write_samples(output_root / "context_audit_samples.md", records, delta_rows)
    write_risk_report(output_root, summary_rows, delta_summary)
    dump_json(
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "mode": "reuse",
            "input_root": str(args.input_root),
            "output_root": str(output_root),
            "datasets": [canonical_dataset(x) for x in args.datasets],
            "settings": [f"{variant}:{top_k}" for variant, top_k in settings],
            "resolved_prediction_files": {f"{k[0]}::{k[1]}::{k[2]}": str(v) for k, v in files.items()},
            "n_records": len(records),
            "topk_delta_summary": delta_summary,
        },
        output_root / "run_manifest.json",
    )
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc ACE-RAG context rendering audit over existing predictions.jsonl files.")
    parser.add_argument("--input-root", default="outputs/ace_rag_native_final_verification/20260526_034219")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wikimultihopqa"])
    parser.add_argument("--settings", nargs="+", default=["p2_relaxed_chain:8"])
    parser.add_argument("--mode", choices=["reuse", "export"], default="reuse")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "export":
        raise SystemExit(
            "export mode is intentionally not implemented here because existing final verification outputs contain rendered_context, "
            "raw evidence_bundles, hashes, predictions, answers, and token counts. Use --mode reuse."
        )
    out = run_reuse(args)
    print(f"wrote audit outputs: {out}")


if __name__ == "__main__":
    main()
