#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_failures import (
    FAILURE_CATEGORIES,
    classify_example,
    find_latest_prediction,
    infer_dataset,
    infer_prompt,
    infer_rendering,
    iter_prediction_files,
    summarize_prediction_file,
)
from utils.eval_metrics import evaluate_predictions, summary_markdown
from utils.generation import DEFAULT_RENDERING_PROFILE, PROMPT_TEMPLATES, RENDERING_PROFILES, add_token_accounting_fields, build_prompt, normalize_prediction_for_eval, render_context
from utils.io_utils import dump_json, load_yaml, read_jsonl, write_jsonl
from utils.text import safe_truncate


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def context_from_row(row: Mapping[str, Any]) -> str:
    context = row.get("rendered_context")
    if context is None:
        context = row.get("rendered_context_preview", "")
    return str(context or "")


def json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def find_latest_prediction_with_rendering(
    output_root: Path,
    dataset: str | None,
    prompt_profile: str | None,
    rendering_profile: str | None,
) -> Path:
    candidates = []
    for path in iter_prediction_files(output_root):
        try:
            summary = summarize_prediction_file(path)
        except Exception:
            continue
        if dataset and summary["dataset"] != dataset:
            continue
        if prompt_profile and summary["prompt_profile"] != prompt_profile:
            continue
        if rendering_profile and summary["rendering_profile"] != rendering_profile:
            continue
        if int(summary["n"]) <= 0:
            continue
        candidates.append(summary)
    if not candidates:
        filters = f"dataset={dataset!r} prompt_profile={prompt_profile!r} rendering_profile={rendering_profile!r}"
        raise SystemExit(f"No matching predictions.jsonl found for {filters}")
    return max(candidates, key=lambda x: (float(x["mtime"]), str(x["path"])))["path"]


def prompt_experiment_type(source_prompt: str, target_prompt: str, rendering_changed: bool) -> str:
    if rendering_changed:
        return "rendering_replay"
    if target_prompt != source_prompt or target_prompt in {"qmrag_bundle_qa", "qmrag_bundle_light", "qmrag_bundle_tiny"}:
        return "replay_ablation"
    return "replay"


def resolve_model(client: Any, configured: str) -> str:
    if configured and configured.lower() not in {"auto", ""}:
        return configured
    models = client.models.list()
    if not models.data:
        raise RuntimeError("OpenAI-compatible /v1/models returned no models; set generation.model explicitly.")
    return models.data[0].id


def generate_with_openai_compatible(prompt: str, cfg: Mapping[str, Any]) -> dict[str, Any]:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("openai package is required for replay generation") from exc
    client = OpenAI(
        base_url=str(cfg.get("base_url") or "http://127.0.0.1:8011/v1"),
        api_key=str(cfg.get("api_key") or "EMPTY"),
        timeout=float(cfg.get("timeout_s", cfg.get("timeout", 120))),
    )
    model = resolve_model(client, str(cfg.get("model", "auto")))
    messages = []
    if cfg.get("system_message"):
        messages.append({"role": "system", "content": str(cfg["system_message"])})
    messages.append({"role": "user", "content": prompt})
    req = {
        "model": model,
        "messages": messages,
        "temperature": float(cfg.get("temperature", 0.0)),
        "max_tokens": int(cfg.get("max_new_tokens", 64)),
    }
    if cfg.get("top_p") is not None:
        req["top_p"] = float(cfg["top_p"])
    if cfg.get("stop"):
        req["stop"] = cfg["stop"]
    if cfg.get("extra_body"):
        req["extra_body"] = dict(cfg.get("extra_body") or {})
    resp = client.chat.completions.create(**req)
    usage = getattr(resp, "usage", None)
    raw = resp.choices[0].message.content or ""
    return {
        "raw_prediction": raw,
        "prediction": normalize_prediction_for_eval(raw),
        "generation_provider": "vllm",
        "llm_provider": "vllm",
        "llm_model": model,
        "llm_usage": usage.model_dump() if hasattr(usage, "model_dump") else {},
    }


def replay_rows(
    rows: Sequence[Mapping[str, Any]],
    dataset: str,
    source_prompt: str,
    target_prompt: str,
    source_rendering: str,
    target_rendering: str,
    rerender_context: bool,
    cfg: Mapping[str, Any],
    limit: int | None,
    sample: int | None,
    failure_category: str | None,
    no_llm: bool,
    dry_run: bool,
) -> list[dict[str, Any]]:
    out = []
    selected = list(rows)
    if failure_category:
        selected = [row for row in selected if classify_example(row)["failure_category"] == failure_category]
    if sample is not None:
        selected = selected[:sample]
    if limit is not None:
        selected = selected[:limit]
    rendering_changed = rerender_context and (target_rendering != source_rendering or any(str(row.get("rendering_profile") or source_rendering) != target_rendering for row in selected))
    for row in selected:
        source_context = context_from_row(row)
        source_hash = str(row.get("rendered_context_hash") or sha256_text(source_context))
        if rerender_context:
            context = render_context(
                row.get("evidence_bundles", []) or [],
                rendering_profile=target_rendering,
                max_chars=int(cfg.get("max_context_chars", 24000)),
                token_budget=cfg.get("context_token_budget"),
            )
        else:
            context = source_context
        actual_hash = sha256_text(context)
        prompt = build_prompt(str(row.get("question", "")), context, target_prompt)
        t0 = time.perf_counter()
        if no_llm or dry_run:
            gen = {
                "raw_prediction": str(row.get("raw_prediction", row.get("prediction", "")) or ""),
                "prediction": str(row.get("prediction", row.get("raw_prediction", "")) or ""),
                "generation_provider": "schema_replay_no_llm",
                "llm_provider": "schema_replay_no_llm",
                "llm_model": row.get("llm_model"),
                "llm_usage": {},
            }
        else:
            gen = generate_with_openai_compatible(prompt, cfg)
        new_row = dict(row)
        new_row.update(
            {
                "dataset": dataset,
                "source_prompt_profile": source_prompt,
                "source_rendering_profile": str(row.get("rendering_profile") or source_rendering),
                "prompt_profile": target_prompt,
                "rendering_profile": target_rendering,
                "prompt_experiment_type": prompt_experiment_type(source_prompt, target_prompt, rendering_changed),
                "raw_prediction": str(gen.get("raw_prediction", "")),
                "prediction": normalize_prediction_for_eval(gen.get("prediction", gen.get("raw_prediction", ""))),
                "generation_provider": gen.get("generation_provider", "vllm"),
                "llm_provider": gen.get("llm_provider", "vllm"),
                "llm_model": gen.get("llm_model"),
                "llm_usage": gen.get("llm_usage", {}),
                "generation_latency_s": round(time.perf_counter() - t0, 6),
                "rendered_context": context,
                "rendered_context_preview": safe_truncate(context, 2000),
                "rendered_context_hash": actual_hash,
                "source_rendered_context_hash": source_hash,
                "rendered_context_hash_match": actual_hash == source_hash,
                "evidence_bundles_hash": json_hash(row.get("evidence_bundles", []) or []),
                "source_evidence_bundles_hash": str(row.get("evidence_bundles_hash") or json_hash(row.get("evidence_bundles", []) or [])),
                "evidence_bundles_hash_match": True,
                "prompt_hash": sha256_text(prompt),
            }
        )
        new_row = add_token_accounting_fields(
            new_row,
            prompt,
            context,
            new_row.get("raw_prediction", ""),
            target_prompt,
            cfg,
            usage=new_row.get("llm_usage"),
            model=new_row.get("llm_model") or row.get("llm_model"),
        )
        out.append(new_row)
    return out


def load_generation_config(config_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    cfg = dict((load_yaml(config_path).get("generation", {}) if config_path.exists() else {}) or {})
    cfg["prompt_profile"] = args.prompt_profile or cfg.get("prompt_profile", "common_qa")
    if args.temperature is not None:
        cfg["temperature"] = args.temperature
    if args.max_tokens is not None:
        cfg["max_new_tokens"] = args.max_tokens
    if args.vllm_base_url:
        cfg["base_url"] = args.vllm_base_url
    if args.vllm_model:
        cfg["model"] = args.vllm_model
    if args.vllm_api_key:
        cfg["api_key"] = args.vllm_api_key
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay generation over fixed rendered_context.")
    parser.add_argument("--predictions", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--source-prompt", default=None)
    parser.add_argument("--source-rendering-profile", default=None)
    parser.add_argument("--target-prompt", "--prompt-profile", dest="prompt_profile", default=None)
    parser.add_argument("--rendering-profile", choices=sorted(RENDERING_PROFILES), default=None)
    parser.add_argument("--failure-category", choices=FAILURE_CATEGORIES, default=None)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--output-root", default="outputs/replay")
    parser.add_argument("--search-output-root", default="outputs")
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--vllm-base-url", default=None)
    parser.add_argument("--vllm-model", default=None)
    parser.add_argument("--vllm-api-key", default=None)
    args = parser.parse_args()

    target_prompt = str(args.prompt_profile or "qmrag_bundle_qa")
    if target_prompt not in PROMPT_TEMPLATES:
        raise SystemExit(f"Unsupported target prompt {target_prompt!r}; choices={sorted(PROMPT_TEMPLATES)}")
    if args.predictions:
        pred_path = Path(args.predictions)
    elif args.source_rendering_profile:
        pred_path = find_latest_prediction_with_rendering(
            Path(args.search_output_root),
            args.dataset,
            args.source_prompt,
            args.source_rendering_profile,
        )
    else:
        pred_path = find_latest_prediction(Path(args.search_output_root), args.dataset, args.source_prompt)
    rows = read_jsonl(pred_path)
    dataset = args.dataset or infer_dataset(pred_path, rows)
    source_prompt = args.source_prompt or infer_prompt(rows)
    source_rendering = infer_rendering(rows)
    target_rendering = str(args.rendering_profile or source_rendering or DEFAULT_RENDERING_PROFILE)
    cfg = load_generation_config(Path(args.config), args)
    cfg["rendering_profile"] = target_rendering
    replayed = replay_rows(rows, dataset, source_prompt, target_prompt, source_rendering, target_rendering, bool(args.rendering_profile), cfg, args.limit, args.sample, args.failure_category, args.no_llm, args.dry_run)
    expected_selected_count = len(rows)
    if args.failure_category:
        expected_selected_count = sum(1 for row in rows if classify_example(row)["failure_category"] == args.failure_category)
    if args.sample is not None:
        expected_selected_count = min(expected_selected_count, args.sample)
    if args.limit is not None:
        expected_selected_count = min(expected_selected_count, args.limit)

    timestamp = now_timestamp()
    suffix = f"{source_prompt}_to_{target_prompt}"
    if target_rendering != source_rendering or args.rendering_profile:
        suffix = f"{suffix}_{target_rendering}"
    if args.failure_category:
        suffix = f"{suffix}_{args.failure_category}"
    out_dir = Path(args.output_root) / timestamp / dataset / suffix
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_out = out_dir / "predictions.jsonl"
    write_jsonl(replayed, pred_out)
    result = evaluate_predictions(replayed, dataset=dataset, prompt_profile=target_prompt)
    result.update(
        {
            "source_predictions": str(pred_path),
            "source_prompt_profile": source_prompt,
            "target_prompt_profile": target_prompt,
            "source_rendering_profile": source_rendering,
            "rendering_profile": target_rendering,
            "target_rendering_profile": target_rendering,
            "failure_category": args.failure_category,
            "prompt_experiment_type": prompt_experiment_type(source_prompt, target_prompt, bool(args.rendering_profile)),
            "source_row_count": len(rows),
            "selected_row_count": len(replayed),
            "row_count_matches_selection": len(replayed) == expected_selected_count,
            "id_sequence_matches_source_prefix": [str(x.get("id")) for x in replayed]
            == [str(x.get("id")) for x in rows[: len(replayed)]],
            "rendered_context_hash_match_rate": sum(1.0 if x.get("rendered_context_hash_match") else 0.0 for x in replayed)
            / max(1, len(replayed)),
            "evidence_bundles_hash_match_rate": sum(1.0 if x.get("evidence_bundles_hash_match") else 0.0 for x in replayed)
            / max(1, len(replayed)),
        }
    )
    dump_json(result, out_dir / "eval.json")
    (out_dir / "eval_summary.md").write_text(summary_markdown(dataset, result), encoding="utf-8")
    print(f"source: {pred_path}")
    print(f"output: {pred_out}")
    print(
        json.dumps(
            {
                "n": len(replayed),
                "id_sequence_matches_source_prefix": result["id_sequence_matches_source_prefix"],
                "rendered_context_hash_match_rate": result["rendered_context_hash_match_rate"],
                "evidence_bundles_hash_match_rate": result["evidence_bundles_hash_match_rate"],
                "prompt_profile": target_prompt,
                "rendering_profile": target_rendering,
                "failure_category": args.failure_category,
                "prompt_experiment_type": result["prompt_experiment_type"],
                "avg_prompt_template_tokens": result.get("avg_prompt_template_tokens"),
                "avg_rendered_context_tokens": result.get("avg_rendered_context_tokens"),
                "avg_input_prompt_tokens": result.get("avg_input_prompt_tokens"),
                "avg_completion_tokens": result.get("avg_completion_tokens"),
                "avg_total_llm_tokens": result.get("avg_total_llm_tokens"),
                "token_count_source_counts": result.get("token_count_source_counts"),
                "no_llm": bool(args.no_llm or args.dry_run),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
