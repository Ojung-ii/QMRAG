#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.eval_metrics import evaluate_predictions
from utils.io_utils import dump_json, ensure_dir, read_jsonl


DATASET_ALIASES = {
    "2wikimultihopqa": "2wiki",
    "2wiki": "2wiki",
    "hotpotqa": "hotpotqa",
    "musique": "musique",
    "popqa": "popqa",
}

DISPLAY_NAMES = {
    "2wiki": "2wikimultihopqa",
    "hotpotqa": "hotpotqa",
    "musique": "musique",
    "popqa": "popqa",
}

MAINLINE_VARIANTS = {"", "ace-rag", "acerag", "ace_rag", "core_qmrag_mainline", "mainline", "qmrag"}

ABLATION_VARIANT_PREFIXES = ("core_no_",)
ABLATION_VARIANTS = {"core_bridge_fullquery", "core_no_anchor_ordering", "core_no_multi_anchor"}

DIAGNOSTIC_COLUMNS = [
    "dataset",
    "n",
    "EM",
    "F1",
    "Recall@5",
    "avg_context_tokens",
    "F1_per_1k_context_tokens",
    "retrieval_ms",
    "generation_ms",
    "bridge_connected_rate",
    "answer_slot_aligned_rate",
    "chain_complete_v2_rate",
    "anchor_connected_chain_complete_rate",
    "multi_anchor_bundle_rate",
    "avg_residual_coverage_count",
    "query_anchor_coverage_rate",
    "answer_in_evidence_bundles",
    "answer_in_rendered_context",
    "avg_rendered_bundle_count",
    "avg_rendered_sentence_count",
    "predictions_path",
]

CONDITIONS = [
    "has_bridge_connected",
    "has_answer_slot_aligned",
    "has_chain_complete_v2",
    "has_multi_anchor_bundle",
    "answer_in_rendered_context",
]


def canonical_dataset(name: str) -> str:
    return DATASET_ALIASES.get(str(name).strip().lower(), str(name).strip().lower())


def display_dataset(name: str) -> str:
    return DISPLAY_NAMES.get(canonical_dataset(name), str(name))


def load_first_row(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    return json.loads(line)
    except Exception:
        return None
    return None


def nested_get(row: Mapping[str, Any], keys: Sequence[str]) -> Any:
    cur: Any = row
    for key in keys:
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(key)
    return cur


def first_metadata(row: Mapping[str, Any], candidates: Sequence[str | Sequence[str]]) -> Any:
    for candidate in candidates:
        if isinstance(candidate, str):
            value = row.get(candidate)
        else:
            value = nested_get(row, candidate)
        if value not in {None, ""}:
            return value
    return None


def path_parts_lower(path: Path) -> str:
    return str(path).lower()


def dataset_match(row: Mapping[str, Any], path: Path, dataset: str, warnings: list[str]) -> bool:
    canonical = canonical_dataset(dataset)
    row_dataset = canonical_dataset(str(row.get("dataset") or ""))
    if row_dataset == canonical:
        return True

    path_text = path_parts_lower(path)
    aliases = {canonical, display_dataset(canonical)}
    if any(alias and alias in path_text for alias in aliases):
        if not row.get("dataset"):
            warnings.append(f"{path}: dataset metadata missing; matched by path for {dataset}.")
        return True
    return False


def prompt_match(row: Mapping[str, Any], path: Path, prompt_profile: str, warnings: list[str]) -> bool:
    value = first_metadata(
        row,
        [
            "prompt_profile",
            "prompt_name",
            "target_prompt",
            ("retrieval_diagnostics", "prompt_profile"),
        ],
    )
    if value is not None:
        return str(value) == prompt_profile
    if prompt_profile.lower() in path_parts_lower(path):
        warnings.append(f"{path}: prompt metadata missing; matched by path for {prompt_profile}.")
        return True
    return False


def rendering_match(row: Mapping[str, Any], path: Path, rendering_profile: str, warnings: list[str]) -> bool:
    values = [
        row.get("rendering_profile"),
        row.get("render_profile"),
        row.get("compact_rendering_profile"),
        row.get("compaction_profile"),
        nested_get(row, ("retrieval_diagnostics", "rendering_profile")),
        nested_get(row, ("retrieval_diagnostics", "compaction_profile")),
    ]
    values = [str(v) for v in values if v not in {None, ""}]
    if rendering_profile in values:
        return True
    if rendering_profile.lower() in path_parts_lower(path):
        if not values:
            warnings.append(f"{path}: rendering metadata missing; matched by path for {rendering_profile}.")
        return True
    return False


def ablation_variant(row: Mapping[str, Any]) -> str:
    value = row.get("ablation_variant")
    if value in {None, ""}:
        value = nested_get(row, ("retrieval_diagnostics", "ablation_variant"))
    return str(value or "").strip()


def discovery_score(row: Mapping[str, Any], path: Path, prompt_profile: str, rendering_profile: str) -> int:
    score = 0
    variant = ablation_variant(row).lower()
    if variant in MAINLINE_VARIANTS:
        score += 1000
    if variant.startswith(ABLATION_VARIANT_PREFIXES) or variant in ABLATION_VARIANTS:
        score -= 1000
    if str(row.get("prompt_profile") or row.get("target_prompt") or "") == prompt_profile:
        score += 50
    if str(row.get("compaction_profile") or row.get("compact_rendering_profile") or "") == rendering_profile:
        score += 50
    if rendering_profile.lower() in path_parts_lower(path):
        score += 25
    if "common_qa_to_common_qa" in path_parts_lower(path):
        score += 20
    if "top3" in rendering_profile and "top3" in path_parts_lower(path):
        score += 10
    return score


def find_predictions(
    output_root: Path,
    datasets: Sequence[str],
    prompt_profile: str,
    rendering_profile: str,
    prefer_latest: bool,
    strict: bool,
) -> tuple[dict[str, Path], list[str]]:
    warnings: list[str] = []
    files = list(output_root.rglob("predictions.jsonl")) if output_root.exists() else []
    resolved: dict[str, Path] = {}

    for dataset in datasets:
        canonical = canonical_dataset(dataset)
        matches: list[tuple[int, float, Path]] = []
        for path in files:
            row = load_first_row(path)
            if not row:
                continue
            local_warnings: list[str] = []
            if not dataset_match(row, path, canonical, local_warnings):
                continue
            if not prompt_match(row, path, prompt_profile, local_warnings):
                continue
            if not rendering_match(row, path, rendering_profile, local_warnings):
                continue
            score = discovery_score(row, path, prompt_profile, rendering_profile)
            mtime = path.stat().st_mtime
            matches.append((score, mtime, path))
            warnings.extend(local_warnings)

        if matches:
            if prefer_latest:
                matches.sort(key=lambda item: (item[0], item[1], str(item[2])), reverse=True)
            else:
                matches.sort(key=lambda item: (item[0], str(item[2])), reverse=True)
            resolved[canonical] = matches[0][2]
            if len(matches) > 1:
                warnings.append(
                    f"{display_dataset(canonical)}: selected {matches[0][2]} from {len(matches)} candidates "
                    f"(score={matches[0][0]})."
                )
        else:
            msg = f"No predictions.jsonl found for dataset={dataset}, prompt={prompt_profile}, rendering={rendering_profile}."
            if strict:
                raise SystemExit(msg)
            warnings.append(msg)

    return resolved, warnings


def parse_pred_mapping(values: Sequence[str] | None) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for value in values or []:
        if "=" not in value:
            raise SystemExit(f"--pred must be dataset=path, got: {value}")
        dataset, path = value.split("=", 1)
        out[canonical_dataset(dataset)] = Path(path)
    return out


def as_number(value: Any) -> float | None:
    if value in {None, "", "NA"}:
        return None
    try:
        number = float(value)
    except Exception:
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def fmt(value: Any, digits: int = 4) -> str:
    number = as_number(value)
    if number is None:
        return "NA"
    return f"{number:.{digits}f}"


def fmt1(value: Any) -> str:
    return fmt(value, 1)


def mean(values: Sequence[Any]) -> float | None:
    nums = [as_number(v) for v in values]
    nums = [v for v in nums if v is not None]
    if not nums:
        return None
    return statistics.fmean(nums)


def metric_value(result: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = result.get(key)
        if value not in {None, ""}:
            return value
    return "NA"


def dataset_summary(dataset: str, path: Path, prompt_profile: str) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    rows = read_jsonl(path)
    result = evaluate_predictions(rows, dataset=canonical_dataset(dataset), prompt_profile=prompt_profile)
    row = {
        "dataset": display_dataset(dataset),
        "n": result.get("n", "NA"),
        "EM": metric_value(result, "EM", "em"),
        "F1": metric_value(result, "F1", "f1"),
        "Recall@5": metric_value(result, "Recall@5", "support_title_recall", "SupportRecall"),
        "avg_context_tokens": metric_value(result, "avg_context_tokens", "context_tokens", "CtxTok"),
        "F1_per_1k_context_tokens": metric_value(result, "F1_per_1k_context_tokens"),
        "retrieval_ms": metric_value(result, "retrieval_ms", "retrieval_latency_ms"),
        "generation_ms": metric_value(result, "generation_ms", "generation_latency_ms"),
        "bridge_connected_rate": metric_value(result, "bridge_connected_rate"),
        "answer_slot_aligned_rate": metric_value(result, "answer_slot_aligned_rate"),
        "chain_complete_v2_rate": metric_value(result, "chain_complete_v2_rate"),
        "anchor_connected_chain_complete_rate": metric_value(result, "anchor_connected_chain_complete_rate"),
        "multi_anchor_bundle_rate": metric_value(result, "multi_anchor_bundle_rate"),
        "avg_residual_coverage_count": metric_value(result, "avg_residual_coverage_count"),
        "query_anchor_coverage_rate": metric_value(result, "query_anchor_coverage_rate"),
        "answer_in_evidence_bundles": metric_value(result, "answer_in_evidence_bundles", "answer_in_context"),
        "answer_in_rendered_context": metric_value(result, "answer_in_rendered_context"),
        "avg_rendered_bundle_count": metric_value(result, "avg_rendered_bundle_count"),
        "avg_rendered_sentence_count": metric_value(result, "avg_rendered_sentence_count"),
        "predictions_path": str(path),
    }
    return row, result, rows


def conditioned_rows(dataset: str, result: Mapping[str, Any]) -> list[dict[str, Any]]:
    per = list(result.get("per_example") or [])
    out: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        present = [x for x in per if as_number(x.get(condition)) is not None]
        ones = [x for x in present if float(x.get(condition) or 0.0) > 0.0]
        zeros = [x for x in present if float(x.get(condition) or 0.0) <= 0.0]
        f1_one = mean([x.get("f1") for x in ones])
        f1_zero = mean([x.get("f1") for x in zeros])
        ctx_one = mean([x.get("context_tokens") for x in ones])
        ctx_zero = mean([x.get("context_tokens") for x in zeros])
        delta = None if f1_one is None or f1_zero is None else f1_one - f1_zero
        out.append(
            {
                "dataset": display_dataset(dataset),
                "condition": condition,
                "n_condition_1": len(ones),
                "n_condition_0": len(zeros),
                "mean_F1_condition_1": f1_one if f1_one is not None else "NA",
                "mean_F1_condition_0": f1_zero if f1_zero is not None else "NA",
                "delta_F1": delta if delta is not None else "NA",
                "mean_context_tokens_condition_1": ctx_one if ctx_one is not None else "NA",
                "mean_context_tokens_condition_0": ctx_zero if ctx_zero is not None else "NA",
            }
        )
    return out


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "NA") for col in columns})


def markdown_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    if not rows:
        return "_No rows._\n"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "NA")
            if isinstance(value, float):
                value = fmt(value)
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def average_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = [
        "F1",
        "Recall@5",
        "avg_context_tokens",
        "F1_per_1k_context_tokens",
        "bridge_connected_rate",
        "answer_slot_aligned_rate",
        "chain_complete_v2_rate",
        "multi_anchor_bundle_rate",
        "avg_residual_coverage_count",
        "answer_in_rendered_context",
    ]
    averaged: dict[str, Any] = {"dataset": "average"}
    for key in keys:
        value = mean([row.get(key) for row in rows])
        averaged[key] = value if value is not None else "NA"
    return averaged


def load_ablation(path: str | None) -> Any:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    if p.suffix.lower() == ".json":
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    with p.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def compact_ablation_note(ablation_data: Any) -> str:
    if not ablation_data:
        return "supported by Table~\\ref{tab:ablation_components}"
    return "supported by the supplied component-ablation file"


def latex_table(rows: Sequence[Mapping[str, Any]], ablation_data: Any) -> str:
    avg = average_summary(rows)
    bridge = fmt(avg.get("bridge_connected_rate"))
    slot = fmt(avg.get("answer_slot_aligned_rate"))
    multi = fmt(avg.get("multi_anchor_bundle_rate"))
    f1_eff = fmt(avg.get("F1_per_1k_context_tokens"))
    ctx = fmt1(avg.get("avg_context_tokens"))
    note = compact_ablation_note(ablation_data)
    return "\n".join(
        [
            "\\begin{table*}[t]",
            "\\centering",
            "\\small",
            "\\setlength{\\tabcolsep}{3.5pt}",
            "\\renewcommand{\\arraystretch}{1.08}",
            "\\begin{tabular}{p{0.21\\textwidth} p{0.35\\textwidth} p{0.34\\textwidth}}",
            "\\toprule",
            "\\textbf{Design principle}",
            "& \\textbf{Operationalization in ACE-RAG}",
            "& \\textbf{Empirical diagnostic} \\\\",
            "\\midrule",
            "Coverage is not answerability",
            "& ACE-RAG prioritizes compact evidence-chain organization rather than only maximizing broad evidence coverage.",
            "& Main results show that higher R@5 does not always yield higher F1; here avg. F1/1K context is " + f1_eff + ". \\\\",
            "Mention-edge expansion",
            "& Seed nodes are bound to doc anchors and expanded through mention edges to construct local chain prefixes.",
            "& Avg. bridge-connected rate = " + bridge + "; " + note + ". \\\\",
            "Residual query completion",
            "& Query-slot filling removes already grounded query terms and selects mention-side evidence with residual query cues.",
            "& Avg. answer-slot aligned rate = " + slot + "; " + note + ". \\\\",
            "Multi-anchor preservation",
            "& Anchor Bundle preserves evidence from multiple query-matched doc anchors.",
            "& Avg. multi-anchor bundle rate = " + multi + "; " + note + ". \\\\",
            "Budget preservation",
            "& Compact rendering serializes deduplicated evidence chains into short contexts.",
            "& Avg. context tokens = " + ctx + " and avg. F1/1K context = " + f1_eff + ". \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Design principles behind ACE-RAG and their empirical diagnostics.}",
            "\\label{tab:design_principles}",
            "\\end{table*}",
            "",
        ]
    )


def summary_markdown(
    rows: Sequence[Mapping[str, Any]],
    conditioned: Sequence[Mapping[str, Any]],
    resolved: Mapping[str, Path],
    warnings: Sequence[str],
    ablation_data: Any,
) -> str:
    avg = average_summary(rows)
    note = compact_ablation_note(ablation_data)
    compact_columns = [
        "dataset",
        "n",
        "EM",
        "F1",
        "Recall@5",
        "avg_context_tokens",
        "F1_per_1k_context_tokens",
        "bridge_connected_rate",
        "answer_slot_aligned_rate",
        "multi_anchor_bundle_rate",
        "answer_in_rendered_context",
    ]
    conditioned_columns = [
        "dataset",
        "condition",
        "n_condition_1",
        "n_condition_0",
        "mean_F1_condition_1",
        "mean_F1_condition_0",
        "delta_F1",
        "mean_context_tokens_condition_1",
        "mean_context_tokens_condition_0",
    ]
    lines = [
        "# ACE-RAG Design Principle Diagnostics",
        "",
        "These diagnostics are post-hoc analyses. They are not used by ACE-RAG during retrieval, chain construction, rendering, or generation.",
        "",
        "## Resolved Prediction Paths",
        "",
    ]
    for dataset, path in resolved.items():
        lines.append(f"- {display_dataset(dataset)}: `{path}`")
    if warnings:
        lines.extend(["", "## Warnings", ""])
        lines.extend([f"- {w}" for w in warnings])
    lines.extend(["", "## Per-Dataset Diagnostic Summary", "", markdown_table(rows, compact_columns)])
    lines.extend(["", "## Average Diagnostic Summary", "", markdown_table([avg], ["dataset"] + [c for c in compact_columns if c not in {"dataset", "n", "EM"}])])
    lines.extend(["", "## Conditioned Diagnostics", "", "Conditioned diagnostics are post-hoc associations, not causal estimates.", "", markdown_table(conditioned, conditioned_columns)])
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "### Coverage is not answerability",
            "",
            "The diagnostics should be read together with the main common-prompt table: broad support-title coverage is not sufficient for answer F1. ACE-RAG's compact chain rendering emphasizes answerable evidence organization rather than only maximizing R@5.",
            "",
            "### Mention-edge expansion",
            "",
            f"The average bridge-connected rate is {fmt(avg.get('bridge_connected_rate'))}. This retrieval-side diagnostic, together with the w/o Mention Edge ablation " + note + ", supports the role of mention-edge expansion in local chain construction.",
            "",
            "### Residual query completion",
            "",
            f"The average answer-slot aligned rate is {fmt(avg.get('answer_slot_aligned_rate'))}, and average residual coverage count is {fmt(avg.get('avg_residual_coverage_count'))}. These are post-hoc indicators that residual query cues are often represented in the rendered evidence. The w/o Query Slot ablation is " + note + ".",
            "",
            "### Multi-anchor preservation",
            "",
            f"The average multi-anchor bundle rate is {fmt(avg.get('multi_anchor_bundle_rate'))}. This is most relevant for multi-hop comparison datasets such as 2Wiki, where preserving multiple query-matched anchors can avoid losing one side of the comparison.",
            "",
            "### Budget preservation",
            "",
            f"The average compact context length is {fmt1(avg.get('avg_context_tokens'))} tokens with F1/1K context of {fmt(avg.get('F1_per_1k_context_tokens'))}. This supports using compact rendering as a budget-preserving serialization of the same retrieved evidence bundles.",
            "",
        ]
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize post-hoc ACE-RAG design-principle diagnostics.")
    parser.add_argument("--output-root", default="outputs", help="Root directory to recursively search for predictions.jsonl.")
    parser.add_argument("--pred", action="append", default=[], help="Explicit dataset=path mapping. Can be repeated.")
    parser.add_argument("--datasets", nargs="+", default=["hotpotqa", "2wikimultihopqa", "musique", "popqa"])
    parser.add_argument("--prompt-profile", default="common_qa")
    parser.add_argument("--rendering-profile", default="top3_chain_dedup")
    parser.set_defaults(prefer_latest=True, include_conditioned=True)
    parser.add_argument("--prefer-latest", dest="prefer_latest", action="store_true")
    parser.add_argument("--no-prefer-latest", dest="prefer_latest", action="store_false")
    parser.add_argument("--out-dir", default="outputs/design_principles")
    parser.add_argument("--include-conditioned", dest="include_conditioned", action="store_true")
    parser.add_argument("--no-include-conditioned", dest="include_conditioned", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--ablation-table")
    parser.add_argument("--ablation-json")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    explicit = parse_pred_mapping(args.pred)
    warnings: list[str] = []

    if explicit:
        resolved = explicit
        for dataset, path in resolved.items():
            if not path.exists():
                msg = f"Explicit predictions path for {display_dataset(dataset)} does not exist: {path}"
                if args.strict:
                    raise SystemExit(msg)
                warnings.append(msg)
    else:
        resolved, warnings = find_predictions(
            output_root=Path(args.output_root),
            datasets=args.datasets,
            prompt_profile=args.prompt_profile,
            rendering_profile=args.rendering_profile,
            prefer_latest=args.prefer_latest,
            strict=args.strict,
        )

    print("Resolved prediction files:")
    for dataset in args.datasets:
        canonical = canonical_dataset(dataset)
        print(f"- {display_dataset(canonical)}: {resolved.get(canonical, 'MISSING')}")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")
    if args.dry_run:
        return

    out_dir = ensure_dir(args.out_dir)
    ablation_data = load_ablation(args.ablation_json) or load_ablation(args.ablation_table)

    diagnostic_rows: list[dict[str, Any]] = []
    conditioned: list[dict[str, Any]] = []
    raw_summaries: dict[str, Any] = {}

    for dataset, path in resolved.items():
        if not path.exists():
            continue
        row, result, _ = dataset_summary(dataset, path, args.prompt_profile)
        diagnostic_rows.append(row)
        raw_summaries[display_dataset(dataset)] = {k: v for k, v in result.items() if k != "per_example"}
        if args.include_conditioned:
            conditioned.extend(conditioned_rows(dataset, result))

    if args.strict and len(diagnostic_rows) < len(args.datasets):
        raise SystemExit(f"Expected {len(args.datasets)} datasets but summarized {len(diagnostic_rows)}.")

    condition_columns = [
        "dataset",
        "condition",
        "n_condition_1",
        "n_condition_0",
        "mean_F1_condition_1",
        "mean_F1_condition_0",
        "delta_F1",
        "mean_context_tokens_condition_1",
        "mean_context_tokens_condition_0",
    ]

    outputs = {
        "diagnostics_csv": out_dir / "diagnostics_by_dataset.csv",
        "diagnostics_json": out_dir / "diagnostics_by_dataset.json",
        "conditioned_csv": out_dir / "conditioned_diagnostics.csv",
        "latex": out_dir / "design_principles_table.tex",
        "summary": out_dir / "design_principles_summary.md",
        "manifest": out_dir / "run_manifest.json",
    }

    write_csv(outputs["diagnostics_csv"], diagnostic_rows, DIAGNOSTIC_COLUMNS)
    dump_json({"rows": diagnostic_rows, "raw_summaries": raw_summaries}, outputs["diagnostics_json"])
    write_csv(outputs["conditioned_csv"], conditioned, condition_columns)
    outputs["latex"].write_text(latex_table(diagnostic_rows, ablation_data), encoding="utf-8")
    outputs["summary"].write_text(
        summary_markdown(diagnostic_rows, conditioned, resolved, warnings, ablation_data),
        encoding="utf-8",
    )
    dump_json(
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "args": vars(args),
            "resolved_paths": {display_dataset(k): str(v) for k, v in resolved.items()},
            "warnings": warnings,
            "outputs": {k: str(v) for k, v in outputs.items()},
            "post_hoc_notice": "Diagnostics are post-hoc analyses and are not used by ACE-RAG during retrieval, chain construction, rendering, or generation.",
        },
        outputs["manifest"],
    )

    print(f"Wrote diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
