#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_acerag_context_rendering import as_answers, norm_contains, selected_anchor_list
from utils.eval_metrics import answer_f1, exact_match, is_insufficient_prediction
from utils.io_utils import dump_json, read_jsonl


SUMMARY_COLUMNS = [
    "stage",
    "dataset",
    "renderer_variant",
    "prompt_variant",
    "top_k",
    "n",
    "EM",
    "F1",
    "Delta F1 vs r0",
    "Recall@5",
    "avg_context_tokens",
    "avg_prompt_tokens",
    "F1/1K prompt",
    "insufficient_information_rate",
    "answer_string_present_rate",
    "answer_present_but_wrong_rate",
    "answer_present_but_insufficient_rate",
    "empty_section_rate",
    "metadata_line_rate",
    "source_line_rate",
    "evidence_bundles_hash_match_rate",
    "retrieval_ms",
    "generation_ms",
    "total_ms",
    "output_path",
]

DIAG_COLUMNS = [
    "stage",
    "renderer_variant",
    "explicit_chain_blocks_rate",
    "anchor_visible_rate",
    "bridge_entity_visible_rate",
    "supporting_evidence_visible_rate",
    "multi_anchor_visible_rate",
    "avg_evidence_sentences",
    "avg_context_tokens",
    "metadata_line_rate",
    "empty_section_rate",
]


def fnum(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def canonical_dataset(value: Any) -> str:
    text = str(value or "")
    return "2wikimultihopqa" if text in {"2wiki", "2wikimultihopqa"} else text


def stage_from_path(path: Path) -> str:
    parts = set(path.parts)
    for stage in ("stage0_smoke", "stage1_n200", "stage2_n1000", "stage3_compact_n1000", "stage4_appendix_4ds"):
        if stage in parts:
            return stage
    return "unknown"


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


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


def renderer_context_lines(context: str) -> list[str]:
    return [line.rstrip() for line in str(context or "").splitlines()]


def renderer_is_section_header(line: str) -> bool:
    stripped = line.strip()
    return (
        stripped.startswith("[Evidence Chain")
        or stripped.startswith("[Supporting Evidence")
        or stripped.startswith("[Multi-Anchor Evidence")
        or stripped.startswith("Evidence Chain")
        or stripped.startswith("Supporting Evidence")
        or stripped.startswith("Multi-Anchor Evidence")
    )


def renderer_is_source_line(line: str, in_sources: bool) -> bool:
    stripped = line.strip()
    return stripped == "Sources:" or stripped.startswith("Sources:") or (in_sources and stripped.startswith("- "))


def renderer_is_debug_metadata_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("[Evidence Chain") or stripped.startswith("[Supporting Evidence") or stripped.startswith("[Multi-Anchor Evidence"):
        return True
    if stripped.startswith(("Sources:", "Supporting Propositions:")):
        return True
    debug_tokens = ("relation_title=", "score=", "anchor_connected=", "chain_complete_v2=", "source_id=", "chunk_id=")
    return any(token in stripped for token in debug_tokens)


def renderer_is_evidence_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or renderer_is_section_header(stripped) or renderer_is_debug_metadata_line(stripped):
        return False
    if stripped.startswith("- "):
        return True
    structural_prefixes = (
        "Question anchor:",
        "Bridge entity:",
        "Connection evidence:",
        "Answer evidence:",
        "Evidence:",
        "Anchor:",
        "Anchors:",
        "Title:",
        "Wikipedia Title:",
        "Sources:",
    )
    if stripped.startswith(structural_prefixes):
        return False
    return any(ch.isalpha() for ch in stripped) and (len(stripped.split()) >= 4 or stripped.endswith((".", "!", "?")))


def renderer_section_empty_count(lines: Sequence[str]) -> int:
    count = 0
    idx = 0
    while idx < len(lines):
        if not renderer_is_section_header(lines[idx]):
            idx += 1
            continue
        j = idx + 1
        has_evidence = False
        while j < len(lines) and not renderer_is_section_header(lines[j]):
            if renderer_is_evidence_line(lines[j]):
                has_evidence = True
            j += 1
        if not has_evidence:
            count += 1
        idx = j
    return count


def renderer_context_stats(context: str, anchors: Sequence[str]) -> dict[str, Any]:
    del anchors
    lines = renderer_context_lines(context)
    non_empty_lines = [line for line in lines if line.strip()]
    in_sources = False
    source_lines = 0
    metadata_lines = 0
    evidence_lines = 0
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Sources:") or stripped == "Sources:":
            in_sources = True
        elif renderer_is_section_header(stripped):
            in_sources = False
        if renderer_is_source_line(stripped, in_sources):
            source_lines += 1
        if renderer_is_debug_metadata_line(stripped):
            metadata_lines += 1
        if renderer_is_evidence_line(stripped):
            evidence_lines += 1
    return {
        "num_lines": len(non_empty_lines),
        "num_empty_sections": renderer_section_empty_count(lines),
        "num_source_lines": source_lines,
        "num_metadata_lines": metadata_lines,
        "num_evidence_sentences": evidence_lines,
    }


def prediction_diagnostics(path: Path) -> dict[str, float]:
    rows = read_jsonl(path)
    if not rows:
        return {}
    answer_present = 0
    present_wrong = 0
    present_insuff = 0
    empty_sections = 0
    metadata_rates = []
    source_rates = []
    evidence_counts = []
    explicit_chain = 0
    anchor_visible = 0
    bridge_available = 0
    bridge_visible = 0
    support_visible = 0
    multi_available = 0
    multi_visible = 0
    for row in rows:
        context = str(row.get("rendered_context") or row.get("rendered_context_preview") or "")
        answers = as_answers(row)
        pred = str(row.get("prediction") or row.get("raw_prediction") or "")
        em = exact_match(pred, answers)
        insuff = is_insufficient_prediction(pred)
        anchors = selected_anchor_list(row)
        stats = renderer_context_stats(context, anchors)
        n_lines = max(1.0, float(stats.get("num_lines", 0) or 0))
        has_answer = norm_contains(context, answers)
        answer_present += int(has_answer)
        present_wrong += int(has_answer and em == 0.0 and not insuff)
        present_insuff += int(has_answer and insuff)
        empty_sections += int(float(stats.get("num_empty_sections", 0) or 0) > 0)
        metadata_rates.append(float(stats.get("num_metadata_lines", 0) or 0) / n_lines)
        source_rates.append(float(stats.get("num_source_lines", 0) or 0) / n_lines)
        evidence_counts.append(float(stats.get("num_evidence_sentences", 0) or 0))
        explicit_chain += int("Evidence Chain" in context)
        anchor_visible += int("Anchor:" in context or "Title:" in context or "Wikipedia Title:" in context)
        bundles = row.get("evidence_bundles", []) or []
        has_bridge = any((b.get("bridge_titles") or b.get("evidence_path")) for b in bundles)
        has_multi = any(b.get("bundle_type") == "multi_anchor" or b.get("anchor_titles") for b in bundles)
        bridge_available += int(has_bridge)
        multi_available += int(has_multi)
        bridge_visible += int(has_bridge and ("Bridge entity:" in context or "Bridge:" in context))
        support_visible += int("Supporting Evidence" in context or "Title:" in context or "Wikipedia Title:" in context or "Evidence:" in context)
        multi_visible += int(has_multi and "Multi-Anchor Evidence" in context)
    n = len(rows)
    return {
        "answer_string_present_rate": answer_present / n,
        "answer_present_but_wrong_rate": present_wrong / n,
        "answer_present_but_insufficient_rate": present_insuff / n,
        "empty_section_rate": empty_sections / n,
        "metadata_line_rate": mean(metadata_rates),
        "source_line_rate": mean(source_rates),
        "explicit_chain_blocks_rate": explicit_chain / n,
        "anchor_visible_rate": anchor_visible / n,
        "bridge_entity_visible_rate": bridge_visible / max(1, bridge_available),
        "supporting_evidence_visible_rate": support_visible / n,
        "multi_anchor_visible_rate": multi_visible / max(1, multi_available),
        "avg_evidence_sentences": mean(evidence_counts),
    }


def rows_from_root(root: Path) -> list[dict[str, Any]]:
    out = []
    for summary_path in sorted(root.rglob("rag_summary.json")):
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        pred_path = Path(str(data.get("output_path") or summary_path.parent / "predictions.jsonl"))
        diag = prediction_diagnostics(pred_path) if pred_path.exists() else {}
        row = {
            "stage": stage_from_path(summary_path),
            "dataset": canonical_dataset(data.get("dataset")),
            "renderer_variant": data.get("ace_renderer_variant") or "r0_current",
            "prompt_variant": data.get("ace_native_prompt_variant") or "UNKNOWN",
            "top_k": data.get("top_bundles") or data.get("qa_top_k") or "NA",
            "n": data.get("n"),
            "EM": data.get("EM"),
            "F1": data.get("F1"),
            "Delta F1 vs r0": 0.0,
            "Recall@5": data.get("Recall@5"),
            "avg_context_tokens": data.get("avg_context_tokens"),
            "avg_prompt_tokens": data.get("avg_prompt_tokens"),
            "F1/1K prompt": data.get("F1_per_1k_prompt_tokens"),
            "insufficient_information_rate": data.get("insufficient_information_rate"),
            "answer_string_present_rate": diag.get("answer_string_present_rate"),
            "answer_present_but_wrong_rate": diag.get("answer_present_but_wrong_rate"),
            "answer_present_but_insufficient_rate": diag.get("answer_present_but_insufficient_rate"),
            "empty_section_rate": diag.get("empty_section_rate"),
            "metadata_line_rate": diag.get("metadata_line_rate"),
            "source_line_rate": diag.get("source_line_rate"),
            "explicit_chain_blocks_rate": diag.get("explicit_chain_blocks_rate"),
            "anchor_visible_rate": diag.get("anchor_visible_rate"),
            "bridge_entity_visible_rate": diag.get("bridge_entity_visible_rate"),
            "supporting_evidence_visible_rate": diag.get("supporting_evidence_visible_rate"),
            "multi_anchor_visible_rate": diag.get("multi_anchor_visible_rate"),
            "avg_evidence_sentences": diag.get("avg_evidence_sentences"),
            "evidence_bundles_hash_match_rate": data.get("evidence_bundles_hash_match_rate"),
            "retrieval_ms": data.get("retrieval_ms"),
            "generation_ms": data.get("generation_ms"),
            "total_ms": data.get("total_ms"),
            "output_path": str(pred_path),
        }
        out.append(row)
    baseline = {}
    for row in out:
        if row["renderer_variant"] == "r0_current":
            baseline[(row["stage"], row["dataset"], str(row["top_k"]))] = fnum(row["F1"])
    for row in out:
        row["Delta F1 vs r0"] = fnum(row["F1"]) - baseline.get((row["stage"], row["dataset"], str(row["top_k"])), fnum(row["F1"]))
    return out


def average_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("dataset") not in {"hotpotqa", "2wikimultihopqa"}:
            continue
        grouped[(str(row["stage"]), str(row["renderer_variant"]), str(row["top_k"]))].append(row)
    out = []
    numeric = [c for c in SUMMARY_COLUMNS if c not in {"stage", "dataset", "renderer_variant", "prompt_variant", "top_k", "output_path"}]
    for (stage, renderer, top_k), group in sorted(grouped.items()):
        if len(group) < 2:
            continue
        row = {
            "stage": stage,
            "dataset": "average",
            "renderer_variant": renderer,
            "prompt_variant": str(group[0].get("prompt_variant")),
            "top_k": top_k,
            "output_path": "",
        }
        for col in numeric:
            row[col] = mean([fnum(x.get(col)) for x in group])
        out.append(row)
    return out


def diagnostics_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["stage"]), str(row["renderer_variant"]))].append(row)
    out = []
    for (stage, renderer), group in sorted(grouped.items()):
        out.append(
            {
                "stage": stage,
                "renderer_variant": renderer,
                "explicit_chain_blocks_rate": mean([fnum(x.get("explicit_chain_blocks_rate")) for x in group]),
                "anchor_visible_rate": mean([fnum(x.get("anchor_visible_rate")) for x in group]),
                "bridge_entity_visible_rate": mean([fnum(x.get("bridge_entity_visible_rate")) for x in group]),
                "supporting_evidence_visible_rate": mean([fnum(x.get("supporting_evidence_visible_rate")) for x in group]),
                "multi_anchor_visible_rate": mean([fnum(x.get("multi_anchor_visible_rate")) for x in group]),
                "avg_evidence_sentences": mean([fnum(x.get("avg_evidence_sentences")) for x in group]),
                "avg_context_tokens": mean([fnum(x.get("avg_context_tokens")) for x in group]),
                "metadata_line_rate": mean([fnum(x.get("metadata_line_rate")) for x in group]),
                "empty_section_rate": mean([fnum(x.get("empty_section_rate")) for x in group]),
            }
        )
    return out


def write_latex(root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    avg_rows = [r for r in rows if r.get("dataset") == "average" and r.get("stage") in {"stage1_n200", "stage2_n1000"}]
    stage = "stage2_n1000" if any(r.get("stage") == "stage2_n1000" for r in avg_rows) else "stage1_n200"
    avg_rows = [r for r in avg_rows if r.get("stage") == stage and str(r.get("top_k")) == "8"]
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Renderer & EM & F1 & Prompt Tok. & F1/1K & Insuff. \\\\",
        "\\midrule",
    ]
    for row in avg_rows:
        renderer = str(row.get("renderer_variant"))
        lines.append(
            f"{renderer} & {fmt(row.get('EM'))} & {fmt(row.get('F1'))} & {fmt(row.get('avg_prompt_tokens'),0)} & {fmt(row.get('F1/1K prompt'))} & {fmt(row.get('insufficient_information_rate'))} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{ACE-RAG renderer sensitivity under the native prompt setting. All variants use the same retrieved evidence bundles with the relaxed chain prompt and top-8 rendered bundles. R1 removes metadata and empty sections, R2 renders evidence as titled paragraphs, and R3 exposes chain structure in a readable paragraph format.}",
            "\\label{tab:acerag_renderer_sensitivity}",
            "\\end{table}",
            "",
        ]
    )
    (root / "renderer_ablation_latex_table.tex").write_text("\n".join(lines), encoding="utf-8")


def load_predictions(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = Path(str(row.get("output_path") or ""))
    return read_jsonl(path) if path.exists() else []


def write_case_samples(root: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    candidates = [r for r in rows if r.get("stage") in {"stage1_n200", "stage2_n1000"} and str(r.get("top_k")) == "8" and r.get("dataset") != "average"]
    grouped: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for row in candidates:
        grouped[(str(row["stage"]), str(row["dataset"]))][str(row["renderer_variant"])] = load_predictions(row)
    blocks = ["# Renderer Case Samples", ""]
    desired = [
        ("r0 is wrong but r1 is correct", "r1_clean_sentence"),
        ("r0 is wrong but r2 is correct", "r2_title_paragraph"),
        ("r0 is wrong but r3 is correct", "r3_chain_paragraph_hybrid"),
        ("r2 is correct but r3 is wrong", "r2_vs_r3"),
        ("r3 is correct but r2 is wrong", "r3_vs_r2"),
        ("all renderers are wrong even though gold answer appears in context", "all_wrong_answer_present"),
        ("all renderers are wrong and gold answer does not appear in context", "all_wrong_no_answer"),
    ]
    for title, kind in desired:
        blocks.extend([f"## {title}", ""])
        found = 0
        for (_stage, dataset), preds in grouped.items():
            if found >= 3:
                break
            if "r0_current" not in preds:
                continue
            by_renderer = {r: {str(x.get("id")): x for x in xs} for r, xs in preds.items()}
            ids = set(by_renderer.get("r0_current", {}))
            for mapping in by_renderer.values():
                ids &= set(mapping)
            for qid in sorted(ids):
                rows_by_r = {r: mapping[qid] for r, mapping in by_renderer.items()}
                answers = as_answers(rows_by_r["r0_current"])
                f1s = {r: answer_f1(str(x.get("prediction") or ""), answers) for r, x in rows_by_r.items()}
                contexts_have = {r: norm_contains(x.get("rendered_context") or "", answers) for r, x in rows_by_r.items()}
                ok = False
                if kind.startswith("r") and "_vs_" not in kind and kind in f1s:
                    ok = f1s.get("r0_current", 0.0) == 0.0 and f1s.get(kind, 0.0) > 0.0
                elif kind == "r2_vs_r3":
                    ok = f1s.get("r2_title_paragraph", 0.0) > 0.0 and f1s.get("r3_chain_paragraph_hybrid", 0.0) == 0.0
                elif kind == "r3_vs_r2":
                    ok = f1s.get("r3_chain_paragraph_hybrid", 0.0) > 0.0 and f1s.get("r2_title_paragraph", 0.0) == 0.0
                elif kind == "all_wrong_answer_present":
                    ok = all(v == 0.0 for v in f1s.values()) and any(contexts_have.values())
                elif kind == "all_wrong_no_answer":
                    ok = all(v == 0.0 for v in f1s.values()) and not any(contexts_have.values())
                if not ok:
                    continue
                base = rows_by_r["r0_current"]
                blocks.extend(
                    [
                        f"### {dataset} / {qid}",
                        "",
                        f"Question: {base.get('question')}",
                        f"Gold answer: {answers[0] if answers else ''}",
                        "Renderer predictions:",
                    ]
                )
                for renderer, row in sorted(rows_by_r.items()):
                    blocks.append(f"- {renderer}: {row.get('prediction')} (F1={fmt(f1s[renderer])})")
                blocks.append(f"Selected anchors: {', '.join(selected_anchor_list(base))}")
                blocks.append(f"Evidence bundle hash: {base.get('evidence_bundles_hash')}")
                blocks.append("")
                for renderer, row in sorted(rows_by_r.items()):
                    blocks.extend([f"Rendered context for {renderer}:", "```text", str(row.get("rendered_context") or "")[:8000], "```", ""])
                found += 1
                break
        if found == 0:
            blocks.extend(["No examples found.", ""])
    (root / "renderer_case_samples.md").write_text("\n".join(blocks), encoding="utf-8")


def best_renderer(rows: Sequence[Mapping[str, Any]], stage: str = "stage1_n200") -> str | None:
    avg = [r for r in rows if r.get("dataset") == "average" and r.get("stage") == stage and str(r.get("top_k")) == "8"]
    if not avg:
        return None
    return max(avg, key=lambda r: (fnum(r.get("F1")), fnum(r.get("F1/1K prompt")))).get("renderer_variant")  # type: ignore[return-value]


def write_reports(root: Path, rows: Sequence[dict[str, Any]]) -> None:
    all_rows = sorted(rows + average_rows(rows), key=lambda r: (str(r.get("stage")), str(r.get("dataset")), str(r.get("top_k")), str(r.get("renderer_variant"))))
    diag = diagnostics_rows(rows)
    write_csv(root / "renderer_ablation_summary.csv", all_rows, SUMMARY_COLUMNS)
    write_csv(root / "renderer_context_diagnostics.csv", diag, DIAG_COLUMNS)
    dump_json({"created_at": datetime.now().isoformat(timespec="seconds"), "rows": all_rows, "diagnostics": diag}, root / "renderer_ablation_summary.json")
    lines = [
        "# ACE-RAG Renderer Ablation Summary",
        "",
        f"- root: `{root}`",
        f"- best_stage1_renderer: `{best_renderer(all_rows) or 'NA'}`",
        "",
        "## Table 1. Renderer Ablation Summary",
        "",
        markdown_table(all_rows, SUMMARY_COLUMNS),
        "",
        "## Table 2. Chain-Preservation Diagnostic",
        "",
        markdown_table(diag, DIAG_COLUMNS),
    ]
    (root / "renderer_ablation_summary.md").write_text("\n".join(lines), encoding="utf-8")
    (root / "renderer_context_diagnostics.md").write_text("# Renderer Context Diagnostics\n\n" + markdown_table(diag, DIAG_COLUMNS), encoding="utf-8")
    write_latex(root, all_rows)
    write_case_samples(root, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare ACE-RAG renderer ablation runs.")
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root)
    rows = rows_from_root(root)
    write_reports(root, rows)
    bad = [r for r in rows if fnum(r.get("evidence_bundles_hash_match_rate")) != 1.0]
    print(f"root: {root}")
    print(f"runs: {len(rows)}")
    print(f"bad_hash_count: {len(bad)}")
    print(f"wrote: {root / 'renderer_ablation_summary.md'}")


if __name__ == "__main__":
    main()
