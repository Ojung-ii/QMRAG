#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import subprocess
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from openai import AsyncOpenAI

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.eval_metrics import evaluate_predictions, is_insufficient_prediction, summary_markdown
from utils.generation import (
    ACE_NATIVE_PROMPT_VARIANTS,
    build_prompt,
    count_tokens,
    extract_usage_token_counts,
    resolve_ace_native_prompt_profile,
)
from utils.io_utils import dump_json, ensure_dir, read_jsonl, write_jsonl


DEFAULT_PROMPTS = (
    "p2_relaxed_chain",
    "p5_hippo_cot_style",
    "p6_aggressive_span",
    "p7_anchor_priority",
    "p8_r0_section_aware",
    "p9_hippo_style_answer_only",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generation-only HotpotQA ACE-RAG prompt sprint.")
    parser.add_argument("--input-predictions-jsonl", required=True)
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--prompt-variants", nargs="+", default=list(DEFAULT_PROMPTS), choices=ACE_NATIVE_PROMPT_VARIANTS)
    parser.add_argument(
        "--prompt-profiles",
        nargs="+",
        default=None,
        help="Optional generic prompt profiles such as common_qa. When set, this overrides --prompt-variants.",
    )
    parser.add_argument("--renderer-variant", default="r0_current")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--generation-only", action="store_true", default=True)
    parser.add_argument("--context-field", default="rendered_context")
    parser.add_argument("--question-field", default="question")
    parser.add_argument("--gold-field", default="gold_answer")
    parser.add_argument("--output-root", default="outputs/hotpotqa_f1_sprint")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--stage-name", default="prompt_sprint_n1000")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--base-urls", nargs="+", default=None)
    parser.add_argument("--openai-base-url", default=None)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="auto")
    parser.add_argument("--max-output-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-concurrency", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--retry", type=int, default=2)
    return parser.parse_args()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""


async def resolve_model(client: AsyncOpenAI, configured: str) -> str:
    if configured and configured.lower() not in {"", "auto"}:
        return configured
    models = await client.models.list()
    if not models.data:
        raise RuntimeError("OpenAI-compatible /v1/models returned no models")
    return models.data[0].id


def answers_for_row(row: Mapping[str, Any], gold_field: str) -> list[str]:
    answers = row.get("answers")
    if isinstance(answers, list) and answers:
        return [str(x) for x in answers if str(x).strip()]
    if row.get(gold_field) is not None:
        return [str(row.get(gold_field))]
    if row.get("answer") is not None:
        return [str(row.get("answer"))]
    return []


def postprocess_prediction(raw: str, variant: str) -> str:
    text = str(raw or "").strip()
    if variant == "p5_hippo_cot_style" and "Answer:" in text:
        text = text.rsplit("Answer:", 1)[-1].strip()
    return text.strip().strip("`").strip()


def usage_dict(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}
    out: dict[str, int] = {}
    for attr, key in (
        ("prompt_tokens", "prompt_tokens"),
        ("completion_tokens", "completion_tokens"),
        ("total_tokens", "total_tokens"),
    ):
        value = getattr(usage, attr, None)
        if value is not None:
            out[key] = int(value)
    return out


async def call_llm(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    args: argparse.Namespace,
    retry: int,
) -> dict[str, Any]:
    req = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_output_tokens,
    }
    last_error: Exception | None = None
    for attempt in range(retry + 1):
        started = time.perf_counter()
        try:
            resp = await client.chat.completions.create(**req)
            raw = resp.choices[0].message.content or ""
            return {
                "raw": raw,
                "usage": usage_dict(getattr(resp, "usage", None)),
                "latency_s": time.perf_counter() - started,
                "error": "",
            }
        except Exception as exc:
            last_error = exc
            if attempt >= retry:
                break
            await asyncio.sleep(1.0 + attempt)
    return {
        "raw": "",
        "usage": {},
        "latency_s": 0.0,
        "error": repr(last_error),
    }


async def run_sprint(args: argparse.Namespace) -> Path:
    input_path = Path(args.input_predictions_jsonl)
    source_rows = read_jsonl(input_path)
    if args.limit:
        source_rows = source_rows[: args.limit]
    if not source_rows:
        raise SystemExit(f"No rows loaded from {input_path}")

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.output_root) / run_id
    stage_root = run_root / args.stage_name
    ensure_dir(stage_root)
    ensure_dir(Path(args.output_root))
    (Path(args.output_root) / "latest_run.txt").write_text(str(run_root), encoding="utf-8")

    base_urls = args.base_urls or ([args.openai_base_url] if args.openai_base_url else ["http://localhost:8013/v1"])
    clients = [AsyncOpenAI(base_url=url, api_key=args.api_key, timeout=args.timeout) for url in base_urls]
    models = [await resolve_model(client, args.model) for client in clients]

    semaphore = asyncio.Semaphore(max(1, args.max_concurrency))
    completed = 0
    prompt_jobs = list(args.prompt_profiles or args.prompt_variants)
    generic_prompt_mode = bool(args.prompt_profiles)
    total = len(source_rows) * len(prompt_jobs)
    rows_by_variant: dict[str, list[dict[str, Any]]] = {variant: [] for variant in prompt_jobs}
    lock = asyncio.Lock()

    async def worker(task_index: int, variant: str, row_index: int, row: Mapping[str, Any]) -> None:
        nonlocal completed
        prompt_profile = variant if generic_prompt_mode else resolve_ace_native_prompt_profile(variant)
        question = str(row.get(args.question_field) or "")
        context = str(row.get(args.context_field) or "")
        prompt = build_prompt(question, context, prompt_profile)
        endpoint_index = task_index % len(clients)
        async with semaphore:
            result = await call_llm(clients[endpoint_index], models[endpoint_index], prompt, args, args.retry)
        processed = postprocess_prediction(result["raw"], variant)
        usage = result.get("usage") or {}
        usage_counts = extract_usage_token_counts(usage)
        prompt_tokens = int(usage_counts.get("prompt_tokens") or count_tokens(prompt))
        completion_tokens = int(usage_counts.get("completion_tokens") or count_tokens(processed))
        total_tokens = int(usage_counts.get("total_tokens") or (prompt_tokens + completion_tokens))
        answers = answers_for_row(row, args.gold_field)
        out = dict(row)
        out.update(
            {
                "dataset": args.dataset,
                "id": row.get("id", row_index),
                "question": question,
                "answers": answers,
                "gold_answer": answers[0] if answers else "",
                "raw_llm_output": result["raw"],
                "raw_prediction": processed,
                "prediction": processed,
                "prompt_profile": prompt_profile,
                "ace_native_prompt_variant": "" if generic_prompt_mode else variant,
                "ace_renderer_variant": args.renderer_variant,
                "top_k": args.top_k,
                "top_bundles": args.top_k,
                "generation_only": True,
                "prompt_experiment_type": "generic_generation_only" if generic_prompt_mode else "hotpotqa_f1_sprint_generation_only",
                "prompt": prompt,
                "final_prompt_text": prompt,
                "prompt_hash": __import__("hashlib").sha256(prompt.encode("utf-8")).hexdigest(),
                "generation_provider": "openai_compatible_async",
                "llm_provider": "openai_compatible_async",
                "llm_model": models[endpoint_index],
                "llm_endpoint": base_urls[endpoint_index],
                "llm_usage": usage,
                "llm_usage_prompt_tokens": prompt_tokens,
                "llm_usage_completion_tokens": completion_tokens,
                "llm_usage_total_tokens": total_tokens,
                "input_prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_llm_tokens": total_tokens,
                "prompt_tokens": prompt_tokens,
                "context_tokens": int(row.get("context_tokens") or row.get("rendered_context_tokens") or 0),
                "generation_latency_s": float(result.get("latency_s") or 0.0),
                "generation_error": result.get("error") or "",
                "source_predictions_path": str(input_path),
                "source_prompt_profile": row.get("prompt_profile"),
                "source_ace_native_prompt_variant": row.get("ace_native_prompt_variant"),
                "source_prediction": row.get("prediction"),
                "source_raw_prediction": row.get("raw_prediction"),
            }
        )
        async with lock:
            rows_by_variant[variant].append(out)
            completed += 1
            if completed % max(1, args.progress_every) == 0 or completed == total:
                print(f"progress {completed}/{total}", flush=True)

    tasks = []
    task_index = 0
    for variant in prompt_jobs:
        for row_index, row in enumerate(source_rows):
            tasks.append(asyncio.create_task(worker(task_index, variant, row_index, row)))
            task_index += 1
    await asyncio.gather(*tasks)

    manifest = {
        "input_predictions_jsonl": str(input_path),
        "output_root": str(run_root),
        "stage_root": str(stage_root),
        "dataset": args.dataset,
        "prompt_variants": list(args.prompt_variants),
        "prompt_profiles": list(args.prompt_profiles or []),
        "renderer_variant": args.renderer_variant,
        "top_k": args.top_k,
        "n": len(source_rows),
        "base_urls": base_urls,
        "models": models,
        "max_concurrency": args.max_concurrency,
        "max_output_tokens": args.max_output_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "git_commit": git_commit(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(__import__("sys").argv),
    }
    dump_json(manifest, stage_root / "run_manifest.json")

    summary_rows = []
    for variant, rows in rows_by_variant.items():
        rows.sort(key=lambda item: str(item.get("id")))
        out_dir = stage_root / variant
        ensure_dir(out_dir)
        write_jsonl(rows, out_dir / "predictions.jsonl")
        prompt_profile = variant if generic_prompt_mode else resolve_ace_native_prompt_profile(variant)
        result = evaluate_predictions(rows, dataset=args.dataset, prompt_profile=prompt_profile)
        result.update(
            {
                "dataset": args.dataset,
                "method": "ACE-RAG",
                "prompt_setting": prompt_profile,
                "ace_native_prompt_variant": "" if generic_prompt_mode else variant,
                "ace_renderer_variant": args.renderer_variant,
                "top_bundles": args.top_k,
                "qa_top_k": args.top_k,
                "generation_only": True,
                "source_predictions_path": str(input_path),
                "output_path": str(out_dir / "predictions.jsonl"),
                "git_commit": manifest["git_commit"],
                "command": manifest["command"],
                "base_urls": base_urls,
                "model": ",".join(models),
                "evidence_bundles_hash_match_rate": 1.0,
                "empty_answer_rate": sum(1 for row in rows if not str(row.get("prediction") or "").strip()) / max(1, len(rows)),
            }
        )
        avg_input = result.get("avg_input_prompt_tokens")
        avg_context = result.get("avg_context_tokens")
        result["avg_input_tokens"] = avg_input
        result["avg_prompt_tokens"] = avg_input
        result["avg_prompt_overhead_tokens"] = (
            float(avg_input) - float(avg_context)
            if avg_input is not None and avg_context is not None
            else None
        )
        result["generation_ms"] = result.get("generation_latency_ms")
        result["retrieval_ms"] = result.get("retrieval_latency_ms")
        result["total_ms"] = result.get("latency_ms")
        dump_json(result, out_dir / "rag_summary.json")
        (out_dir / "eval_summary.md").write_text(summary_markdown(args.dataset, result), encoding="utf-8")
        summary_rows.append(
            {
                "prompt_variant": variant,
                "EM": result.get("EM"),
                "F1": result.get("F1"),
                "InputTok": result.get("avg_input_prompt_tokens"),
                "Insuff": result.get("insufficient_rate"),
                "output_path": str(out_dir / "predictions.jsonl"),
            }
        )

    with (stage_root / "quick_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["prompt_variant", "EM", "F1", "InputTok", "Insuff", "output_path"])
        writer.writeheader()
        writer.writerows(summary_rows)
    return run_root


def main() -> None:
    args = parse_args()
    root = asyncio.run(run_sprint(args))
    print(f"RUN_ROOT={root}")


if __name__ == "__main__":
    main()
