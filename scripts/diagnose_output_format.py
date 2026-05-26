#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_failures import find_all_latest, find_latest_prediction, infer_dataset, infer_prompt
from utils.eval_metrics import answer_contains, exact_match, row_token_metrics
from utils.generation import has_idk_phrase, is_insufficient_prediction
from utils.io_utils import dump_json, ensure_dir, read_jsonl, write_jsonl
from utils.text import normalize_answer


PREFIX_RE = re.compile(r"^\s*(answer|final answer|the answer is|it is|based on)\s*[:,-]?\s+", re.I)
MARKDOWN_RE = re.compile(r"(^|\n)\s*(#{1,6}\s+|[-*]\s+|\d+\.\s+|```|\*\*)")
CITATION_RE = re.compile(r"\[[^\]]+\]|\(\s*(?:source|citation|context|evidence)[^)]+\)", re.I)
EXPLANATION_RE = re.compile(r"\b(because|based on|according to|context|provided|evidence|therefore|so the answer)\b", re.I)


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def avg(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))


def median(values: Sequence[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def word_count(text: Any) -> int:
    return len(str(text or "").split())


def is_yes_no_question(answers: Sequence[Any]) -> bool:
    norms={normalize_answer(str(x)) for x in answers if str(x).strip()}
    return bool(norms) and norms.issubset({"yes","no"})


def row_format_metrics(row: Mapping[str,Any]) -> dict[str,Any]:
    raw=str(row.get("raw_prediction", row.get("prediction", "")) or "")
    answers=[str(x) for x in row.get("answers",[]) if str(x).strip()]
    tok=row_token_metrics(row)
    answer_hit=bool(answer_contains(raw, answers))
    em=bool(exact_match(raw, answers))
    yn=is_yes_no_question(answers)
    pred_norm=normalize_answer(raw)
    return {
        "id":row.get("id"),
        "dataset":row.get("dataset"),
        "prompt_profile":row.get("prompt_profile"),
        "prediction":raw,
        "answers":answers,
        "prediction_tokens":int(tok.get("completion_tokens",0) or 0),
        "prediction_words":word_count(raw),
        "long_answer":int(tok.get("completion_tokens",0) or 0)>16 or word_count(raw)>12,
        "prefix":bool(PREFIX_RE.search(raw)),
        "markdown":bool(MARKDOWN_RE.search(raw)),
        "citation":bool(CITATION_RE.search(raw)),
        "explanation":bool(EXPLANATION_RE.search(raw)),
        "yes_no_format_error":yn and pred_norm not in {"yes","no"},
        "insufficient":is_insufficient_prediction(raw),
        "idk":has_idk_phrase(raw),
        "answer_in_prediction":answer_hit,
        "em":em,
        "formatting_loss":answer_hit and not em,
    }


def summarize(rows: Sequence[Mapping[str,Any]], path: Path) -> tuple[dict[str,Any], list[dict[str,Any]]]:
    per=[row_format_metrics(row) for row in rows]
    n=max(1,len(per))
    pred_tokens=[float(x["prediction_tokens"]) for x in per]
    summary={
        "dataset":infer_dataset(path, rows),
        "prompt_profile":infer_prompt(rows),
        "source_predictions":str(path),
        "n":len(per),
        "avg_prediction_tokens":avg(pred_tokens),
        "median_prediction_tokens":median(pred_tokens),
        "long_answer_rate":sum(1.0 if x["long_answer"] else 0.0 for x in per)/n,
        "prefix_rate":sum(1.0 if x["prefix"] else 0.0 for x in per)/n,
        "markdown_rate":sum(1.0 if x["markdown"] else 0.0 for x in per)/n,
        "citation_rate":sum(1.0 if x["citation"] else 0.0 for x in per)/n,
        "explanation_rate":sum(1.0 if x["explanation"] else 0.0 for x in per)/n,
        "yes_no_format_error_rate":sum(1.0 if x["yes_no_format_error"] else 0.0 for x in per)/n,
        "insufficient_rate":sum(1.0 if x["insufficient"] else 0.0 for x in per)/n,
        "answer_in_prediction":sum(1.0 if x["answer_in_prediction"] else 0.0 for x in per)/n,
        "formatting_loss_rate":sum(1.0 if x["formatting_loss"] else 0.0 for x in per)/n,
        "idk_rate":sum(1.0 if x["idk"] else 0.0 for x in per)/n,
    }
    return summary, per


def fmt(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value,float) else str(value)


def markdown(summary: Mapping[str,Any]) -> str:
    keys=[
        "n","avg_prediction_tokens","median_prediction_tokens","long_answer_rate",
        "prefix_rate","markdown_rate","citation_rate","explanation_rate",
        "yes_no_format_error_rate","insufficient_rate","answer_in_prediction",
        "formatting_loss_rate","idk_rate",
    ]
    lines=[
        "# Output Format Diagnostic",
        "",
        f"- dataset: {summary.get('dataset')}",
        f"- prompt_profile: {summary.get('prompt_profile')}",
        f"- source: {summary.get('source_predictions')}",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key in keys:
        lines.append(f"| {key} | {fmt(summary.get(key,0))} |")
    return "\n".join(lines)+"\n"


def evaluate_path(path: Path, analysis_dir: Path) -> dict[str,Any]:
    rows=read_jsonl(path)
    summary, per=summarize(rows,path)
    stem=f"output_format_diagnostic_{summary['dataset']}_{summary['prompt_profile']}"
    dump_json(summary, analysis_dir / f"{stem}.json")
    (analysis_dir / f"{stem}.md").write_text(markdown(summary), encoding="utf-8")
    examples=[x for x in per if x["formatting_loss"] or x["long_answer"] or x["prefix"] or x["markdown"] or x["citation"] or x["explanation"]]
    write_jsonl(examples[:200], analysis_dir / f"output_format_examples_{summary['dataset']}_{summary['prompt_profile']}.jsonl")
    print(markdown(summary))
    print(f"wrote: {analysis_dir}")
    return summary


def find_latest_prediction_any(output_root: Path, dataset: str | None, prompt_profile: str | None, prefer_full_context: bool = False) -> Path:
    candidates=[]
    for path in output_root.rglob("predictions.jsonl"):
        try:
            first=None
            with open(path,"r",encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        first=json.loads(line)
                        break
        except Exception:
            continue
        if not first:
            continue
        if dataset and str(first.get("dataset") or infer_dataset(path,[first]))!=dataset:
            continue
        if prompt_profile and str(first.get("prompt_profile") or "UNKNOWN")!=prompt_profile:
            continue
        if prefer_full_context and (
            first.get("context_truncation_enabled")
            or first.get("top_bundles") is not None
            or first.get("context_token_budget") is not None
            or str(first.get("compaction_profile") or "none") != "none"
        ):
            continue
        ablation=str(first.get("ablation_variant") or "")
        residual=str((first.get("retrieval_diagnostics",{}) or {}).get("residual_selection_variant") or first.get("residual_selection_variant") or "")
        pref=2 if ablation in {"","core_qmrag_mainline"} and residual in {"","residual_lexical"} else 1 if ablation in {"","core_qmrag_mainline"} else 0
        candidates.append((pref,path.stat().st_mtime,str(path),path))
    if not candidates:
        raise SystemExit(f"No matching predictions.jsonl found for dataset={dataset!r} prompt_profile={prompt_profile!r}")
    return max(candidates)[3]


def main() -> None:
    parser=argparse.ArgumentParser(description="Diagnose output formatting errors without changing main metrics.")
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--prompt-profile", default=None)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--all-latest", action="store_true")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--analysis-root", default="outputs/analysis")
    parser.add_argument("--analysis-dir", default=None)
    args=parser.parse_args()
    analysis_dir=Path(args.analysis_dir) if args.analysis_dir else Path(args.analysis_root)/now_timestamp()
    ensure_dir(analysis_dir)
    if args.all_latest:
        paths=find_all_latest(Path(args.output_root))
    elif args.predictions:
        paths=[Path(args.predictions)]
    else:
        paths=[find_latest_prediction_any(Path(args.output_root), args.dataset, args.prompt_profile, prefer_full_context=args.prompt_profile=="common_qa")]
    summaries=[evaluate_path(path,analysis_dir) for path in paths]
    dump_json({"summaries":summaries}, analysis_dir/"output_format_diagnostic_summary.json")


if __name__=="__main__":
    main()
