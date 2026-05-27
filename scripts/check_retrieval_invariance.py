#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.eval_metrics import answer_in_rendered_context
from utils.io_utils import dump_json, read_jsonl


KEY_METRICS = (
    "candidate_count",
    "seed_count",
    "bundle_count",
    "context_tokens",
    "bridge_title_count",
    "bridge_bundle_count",
    "chain_complete_v2_count",
    "anchor_connected_chain_complete_count",
    "anchor_mismatch_chain_count",
    "selected_seed_hash",
)


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def strip_volatile(value: Any) -> Any:
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            if key in {"timing", "generation_latency_s"}:
                continue
            out[str(key)] = strip_volatile(item)
        return out
    if isinstance(value, list):
        return [strip_volatile(x) for x in value]
    return value


def selected_seed_hash(row: Mapping[str, Any]) -> str:
    diag = row.get("retrieval_diagnostics", {}) or {}
    if diag.get("selected_seed_hash"):
        return str(diag["selected_seed_hash"])
    if diag.get("selected_seed_ids"):
        return stable_hash(list(diag.get("selected_seed_ids") or []))
    seeds = row.get("seeds", []) or []
    seed_ids = []
    for seed in seeds:
        if not isinstance(seed, Mapping):
            continue
        seed_ids.append(f"{seed.get('seed_unit_type') or seed.get('unit') or ''}:{seed.get('id') or seed.get('source_candidate_id') or seed.get('title') or ''}")
    return stable_hash(seed_ids)


def evidence_bundles_hash(row: Mapping[str, Any]) -> str:
    return str(row.get("evidence_bundles_hash") or stable_hash(strip_volatile(row.get("evidence_bundles", []) or [])))


def rendered_context_hash(row: Mapping[str, Any]) -> str:
    if row.get("rendered_context_hash"):
        return str(row["rendered_context_hash"])
    return stable_hash(row.get("rendered_context") or row.get("rendered_context_preview") or "")


def metric_value(row: Mapping[str, Any], key: str) -> Any:
    if key == "selected_seed_hash":
        return selected_seed_hash(row)
    return (row.get("retrieval_diagnostics", {}) or {}).get(key)


def compare(before: Path, after: Path) -> dict[str, Any]:
    left = read_jsonl(before)
    right = read_jsonl(after)
    n = min(len(left), len(right))
    id_matches = 0
    bundle_hash_matches = 0
    context_hash_matches = 0
    seed_hash_matches = 0
    answer_in_context_delta = 0.0
    mismatches: dict[str, list[str]] = {
        "id": [],
        "evidence_bundles": [],
        "rendered_context": [],
        "selected_seed": [],
        "answer_in_rendered_context": [],
    }
    metric_diffs = {key: 0 for key in KEY_METRICS}
    for idx in range(n):
        a = left[idx]
        b = right[idx]
        qa_id = str(a.get("id", idx))
        if a.get("id") == b.get("id"):
            id_matches += 1
        elif len(mismatches["id"]) < 20:
            mismatches["id"].append(f"{qa_id} != {b.get('id')}")
        if evidence_bundles_hash(a) == evidence_bundles_hash(b):
            bundle_hash_matches += 1
        elif len(mismatches["evidence_bundles"]) < 20:
            mismatches["evidence_bundles"].append(qa_id)
        if rendered_context_hash(a) == rendered_context_hash(b):
            context_hash_matches += 1
        elif len(mismatches["rendered_context"]) < 20:
            mismatches["rendered_context"].append(qa_id)
        if selected_seed_hash(a) == selected_seed_hash(b):
            seed_hash_matches += 1
        elif len(mismatches["selected_seed"]) < 20:
            mismatches["selected_seed"].append(qa_id)
        golds = [str(x) for x in a.get("answers", [])]
        ain = answer_in_rendered_context(a, golds)
        bin_ = answer_in_rendered_context(b, golds)
        answer_in_context_delta += float(bin_) - float(ain)
        if ain != bin_ and len(mismatches["answer_in_rendered_context"]) < 20:
            mismatches["answer_in_rendered_context"].append(qa_id)
        for key in KEY_METRICS:
            if metric_value(a, key) != metric_value(b, key):
                metric_diffs[key] += 1
    denom = max(1, n)
    result = {
        "before": str(before),
        "after": str(after),
        "n_before": len(left),
        "n_after": len(right),
        "n_compared": n,
        "id_sequence_match_rate": id_matches / denom,
        "evidence_bundles_hash_match_rate": bundle_hash_matches / denom,
        "rendered_context_hash_match_rate": context_hash_matches / denom,
        "selected_seed_hash_match_rate": seed_hash_matches / denom,
        "answer_in_rendered_context_delta": answer_in_context_delta / denom,
        "retrieval_diagnostic_metric_mismatch_counts": metric_diffs,
        "mismatched_example_ids": mismatches,
    }
    return result


def markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Retrieval Invariance Check",
        "",
        f"- before: {result.get('before')}",
        f"- after: {result.get('after')}",
        f"- n_compared: {result.get('n_compared')}",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in (
        "id_sequence_match_rate",
        "evidence_bundles_hash_match_rate",
        "rendered_context_hash_match_rate",
        "selected_seed_hash_match_rate",
        "answer_in_rendered_context_delta",
    ):
        value = result.get(key)
        if isinstance(value, float):
            value = f"{value:.6f}"
        lines.append(f"| {key} | {value} |")
    lines.extend(["", "## Retrieval Diagnostic Mismatches", ""])
    for key, value in (result.get("retrieval_diagnostic_metric_mismatch_counts", {}) or {}).items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Mismatched Examples", ""])
    for key, ids in (result.get("mismatched_example_ids", {}) or {}).items():
        lines.append(f"- {key}: {json.dumps(ids, ensure_ascii=False)}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Check retrieval invariance between two ACE-RAG prediction files")
    parser.add_argument("--before", required=True)
    parser.add_argument("--after", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = compare(Path(args.before), Path(args.after))
    text = markdown(result)
    print(text)
    if args.output:
        out = Path(args.output)
        if out.suffix == ".json":
            dump_json(result, out)
            out.with_suffix(".md").write_text(text, encoding="utf-8")
        else:
            out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
