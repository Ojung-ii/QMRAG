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

from scripts.analyze_failures import find_latest_prediction, infer_dataset, infer_prompt
from utils.eval_metrics import evaluate_predictions, summary_markdown
from utils.generation import PROMPT_TEMPLATES, build_prompt, normalize_prediction_for_eval
from utils.io_utils import dump_json, load_yaml, read_jsonl, write_jsonl
from utils.text import safe_truncate, token_count


def now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def context_from_row(row: Mapping[str, Any]) -> str:
    context = row.get("rendered_context")
    if context is None:
        context = row.get("rendered_context_preview", "")
    return str(context or "")


def prompt_experiment_type(prompt_profile: str) -> str:
    if prompt_profile == "common_qa":
        return "main_comparison"
    if prompt_profile == "qmrag_bundle_qa":
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
    cfg: Mapping[str, Any],
    limit: int | None,
    no_llm: bool,
    dry_run: bool,
) -> list[dict[str, Any]]:
    out = []
    selected = list(rows[:limit] if limit is not None else rows)
    for row in selected:
        context = context_from_row(row)
        expected_hash = str(row.get("rendered_context_hash") or sha256_text(context))
        actual_hash = sha256_text(context)
        prompt = build_prompt(str(row.get("question", "")), context, target_prompt)
        t0 = time.perf_counter()
        if no_llm or dry_run:
            gen = {
                "raw_prediction": str(row.get("raw_prediction", row.get("prediction", "")) or ""),
                "prediction": str(row.get("prediction", row.get("raw_prediction", "")) or ""),
                "generation_provider": "schema_replay_no_llm",
                "llm_provider": "schema_replay_no_llm",
                "llm_model": None,
                "llm_usage": {},
            }
        else:
            gen = generate_with_openai_compatible(prompt, cfg)
        new_row = dict(row)
        new_row.update(
            {
                "dataset": dataset,
                "source_prompt_profile": source_prompt,
                "prompt_profile": target_prompt,
                "prompt_experiment_type": prompt_experiment_type(target_prompt),
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
                "source_rendered_context_hash": expected_hash,
                "rendered_context_hash_match": actual_hash == expected_hash,
                "rendered_context_tokens": token_count(context),
                "prompt_hash": sha256_text(prompt),
            }
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
    parser.add_argument("--target-prompt", "--prompt-profile", dest="prompt_profile", default=None)
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
    else:
        pred_path = find_latest_prediction(Path(args.search_output_root), args.dataset, args.source_prompt)
    rows = read_jsonl(pred_path)
    dataset = args.dataset or infer_dataset(pred_path, rows)
    source_prompt = args.source_prompt or infer_prompt(rows)
    cfg = load_generation_config(Path(args.config), args)
    replayed = replay_rows(rows, dataset, source_prompt, target_prompt, cfg, args.limit, args.no_llm, args.dry_run)

    timestamp = now_timestamp()
    out_dir = Path(args.output_root) / timestamp / dataset / f"{source_prompt}_to_{target_prompt}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_out = out_dir / "predictions.jsonl"
    write_jsonl(replayed, pred_out)
    result = evaluate_predictions(replayed, dataset=dataset, prompt_profile=target_prompt)
    result.update(
        {
            "source_predictions": str(pred_path),
            "source_prompt_profile": source_prompt,
            "target_prompt_profile": target_prompt,
            "prompt_experiment_type": "replay_ablation",
            "row_count_matches_source": len(replayed) == (len(rows) if args.limit is None else min(len(rows), args.limit)),
            "id_sequence_matches_source": [str(x.get("id")) for x in replayed]
            == [str(x.get("id")) for x in rows[: len(replayed)]],
            "rendered_context_hash_match_rate": sum(1.0 if x.get("rendered_context_hash_match") else 0.0 for x in replayed)
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
                "id_sequence_matches_source": result["id_sequence_matches_source"],
                "rendered_context_hash_match_rate": result["rendered_context_hash_match_rate"],
                "prompt_profile": target_prompt,
                "prompt_experiment_type": "replay_ablation",
                "no_llm": bool(args.no_llm or args.dry_run),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
