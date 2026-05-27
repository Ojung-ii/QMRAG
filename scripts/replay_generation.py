#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
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
from utils.generation import (
    ACE_NATIVE_PROMPT_VARIANTS,
    ACE_RENDERER_VARIANTS,
    infer_ace_native_prompt_variant,
    resolve_ace_native_prompt_profile,
    COMPACTION_PROFILES,
    DEFAULT_RENDERING_PROFILE,
    PROMPT_TEMPLATES,
    RENDERING_PROFILES,
    _load_counting_tokenizer,
    add_token_accounting_fields,
    build_prompt,
    count_tokens,
    normalize_prediction_for_eval,
    render_context,
    render_context_with_metadata,
)
from utils.io_utils import dump_json, load_yaml, read_jsonl, write_jsonl
from utils.text import normalize_answer, token_count, safe_truncate


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

def first_jsonl_row(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    return json.loads(line)
    except Exception:
        return None
    return None

def source_preference(first: Mapping[str, Any]) -> int:
    ablation=str(first.get("ablation_variant") or "")
    residual=str((first.get("retrieval_diagnostics",{}) or {}).get("residual_selection_variant") or first.get("residual_selection_variant") or "")
    if ablation in {"", "core_ace_rag_mainline"} and residual in {"", "residual_lexical"}:
        return 2
    if ablation in {"", "core_ace_rag_mainline"}:
        return 1
    return 0


def find_latest_prediction_with_rendering(
    output_root: Path,
    dataset: str | None,
    prompt_profile: str | None,
    rendering_profile: str | None,
    exclude_context_truncation: bool = True,
) -> Path:
    candidates = []
    for path in iter_prediction_files(output_root):
        first = first_jsonl_row(path)
        if not first:
            continue
        row_dataset = str(first.get("dataset") or infer_dataset(path, [first]))
        row_prompt = str(first.get("prompt_profile") or "")
        row_rendering = str(first.get("rendering_profile") or "structured_chain")
        if dataset and row_dataset != dataset:
            continue
        if prompt_profile and row_prompt != prompt_profile:
            continue
        if rendering_profile and row_rendering != rendering_profile:
            continue
        if exclude_context_truncation:
            if (
                first.get("context_truncation_enabled")
                or first.get("top_bundles") is not None
                or first.get("context_token_budget") is not None
                or str(first.get("compaction_profile") or "none") != "none"
            ):
                continue
        candidates.append({"path": path, "mtime": path.stat().st_mtime, "source_preference": source_preference(first)})
    if not candidates:
        filters = f"dataset={dataset!r} prompt_profile={prompt_profile!r} rendering_profile={rendering_profile!r}"
        raise SystemExit(f"No matching predictions.jsonl found for {filters}")
    return max(candidates, key=lambda x: (int(x.get("source_preference",0)), float(x["mtime"]), str(x["path"])))["path"]


def prompt_experiment_type(source_prompt: str, target_prompt: str, rendering_changed: bool, context_truncation: bool = False, context_compaction: bool = False) -> str:
    if target_prompt == "strict_short_qa" and not context_compaction and not context_truncation and not rendering_changed:
        return "format_ablation"
    if target_prompt in {"ace_rag_compact_chain_qa", "ace_rag_compact_chain_light", "ace_rag_compact_chain_short_qa"}:
        return "compact_prompt_ablation"
    if target_prompt.startswith("ace_rag_native_"):
        return "ace_native_prompt_ablation"
    if target_prompt in {"ace_rag_bundle_qa", "ace_rag_bundle_light", "ace_rag_bundle_tiny", "ace_rag_bundle_short_qa"}:
        return "ablation"
    if context_compaction:
        return "context_compaction_replay"
    if context_truncation:
        return "context_budget_replay"
    if rendering_changed:
        return "rendering_replay"
    if target_prompt != source_prompt:
        return "replay_ablation"
    return "replay"


def ordered_bundles_for_context(bundles: Sequence[Mapping[str, Any]], ordering_source: str) -> list[Mapping[str, Any]]:
    if ordering_source == "raw_score":
        return sorted(
            list(bundles),
            key=lambda bundle: float(bundle.get("score", 0.0) or 0.0),
            reverse=True,
        )
    return list(bundles)


def render_context_with_truncation(
    bundles: Sequence[Mapping[str, Any]],
    target_rendering: str,
    cfg: Mapping[str, Any],
    top_bundles: int | None,
    context_token_budget: int | None,
    ordering_source: str,
    compaction_profile: str = "none",
    max_sentences_per_bundle: int = 3,
    ace_renderer_variant: str = "r0_current",
) -> tuple[str, list[Mapping[str, Any]], int, dict[str, Any]]:
    original_bundles = list(bundles)
    ordered = ordered_bundles_for_context(original_bundles, ordering_source)
    candidates = ordered[: max(0, int(top_bundles))] if top_bundles is not None else ordered
    max_chars = int(cfg.get("max_context_chars", 24000))
    token_budget = cfg.get("context_token_budget")
    if str(compaction_profile or "none") == "full_structured_budget" and context_token_budget is not None:
        tokenizer, _counter_source = _load_counting_tokenizer(cfg, cfg.get("model"))
        effective_budget = int(context_token_budget)
        context = ""
        stats: dict[str, Any] = {}
        for _ in range(8):
            context, stats = render_context_with_metadata(
                candidates,
                rendering_profile=target_rendering,
                max_chars=max_chars,
                token_budget=effective_budget,
                compaction_profile=compaction_profile,
                max_sentences_per_bundle=max_sentences_per_bundle,
                ace_renderer_variant=ace_renderer_variant,
            )
            actual_tokens = count_tokens(context, tokenizer)
            if actual_tokens <= int(context_token_budget) or effective_budget <= 1:
                break
            scale = max(0.2, min(0.95, (float(context_token_budget) / max(1.0, float(actual_tokens))) * 0.92))
            next_budget = max(1, int(effective_budget * scale))
            if next_budget >= effective_budget:
                next_budget = effective_budget - 1
            effective_budget = next_budget
        stats["effective_context_token_budget"] = effective_budget
        rendered_count = int(stats.get("rendered_bundle_count", 0) or 0)
        dropped_count = max(0, len(original_bundles) - rendered_count)
        stats["dropped_bundle_count"] = dropped_count
        return context, list(candidates), dropped_count, stats
    if context_token_budget is None:
        context, stats = render_context_with_metadata(
            candidates,
            rendering_profile=target_rendering,
            max_chars=max_chars,
            token_budget=token_budget,
            compaction_profile=compaction_profile,
            max_sentences_per_bundle=max_sentences_per_bundle,
            ace_renderer_variant=ace_renderer_variant,
        )
        return context, list(candidates), max(0, len(original_bundles) - len(candidates)), stats
    selected: list[Mapping[str, Any]] = []
    context = ""
    stats: dict[str, Any] = {}
    for bundle in candidates:
        trial = selected + [bundle]
        trial_context, trial_stats = render_context_with_metadata(
            trial,
            rendering_profile=target_rendering,
            max_chars=max_chars,
            token_budget=None,
            compaction_profile=compaction_profile,
            max_sentences_per_bundle=max_sentences_per_bundle,
            ace_renderer_variant=ace_renderer_variant,
        )
        trial_tokens = token_count(trial_context)
        if trial_tokens > int(context_token_budget) and selected:
            break
        selected = trial
        context = trial_context
        stats = trial_stats
        if trial_tokens >= int(context_token_budget):
            break
    if not selected and candidates:
        selected = [candidates[0]]
        context, stats = render_context_with_metadata(
            selected,
            rendering_profile=target_rendering,
            max_chars=max_chars,
            token_budget=None,
            compaction_profile=compaction_profile,
            max_sentences_per_bundle=max_sentences_per_bundle,
            ace_renderer_variant=ace_renderer_variant,
        )
    return context, selected, max(0, len(original_bundles) - len(selected)), stats


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
    top_bundles: int | None = None,
    context_token_budget: int | None = None,
    ordering_source: str = "current",
    compaction_profile: str = "none",
    max_sentences_per_bundle: int = 3,
    ace_native_prompt_variant: str | None = None,
    ace_renderer_variant: str = "r0_current",
    save_final_prompt: bool = False,
) -> list[dict[str, Any]]:
    out = []
    selected = list(rows)
    if failure_category:
        selected = [row for row in selected if classify_example(row)["failure_category"] == failure_category]
    if sample is not None:
        selected = selected[:sample]
    if limit is not None:
        selected = selected[:limit]
    compaction_profile = str(compaction_profile or "none")
    effective_top_bundles = 3 if compaction_profile in {"top3_chain_dedup", "top3_chain_dedup_no_sources"} and top_bundles is None else top_bundles
    context_truncation = effective_top_bundles is not None or context_token_budget is not None or ordering_source != "current"
    context_compaction = compaction_profile != "none"
    rendering_changed = rerender_context and (target_rendering != source_rendering or any(str(row.get("rendering_profile") or source_rendering) != target_rendering for row in selected))
    for row in selected:
        row_cfg: Mapping[str, Any] = cfg
        if str(cfg.get("model", "auto") or "auto").lower() in {"", "auto"} and row.get("llm_model"):
            tmp_cfg = dict(cfg)
            tmp_cfg["model"] = row.get("llm_model")
            row_cfg = tmp_cfg
        source_context = context_from_row(row)
        source_hash = str(row.get("rendered_context_hash") or sha256_text(source_context))
        source_context_tokens = row.get("rendered_context_tokens")
        if source_context_tokens is None:
            source_context_tokens = token_count(source_context)
        source_input_tokens = row.get("input_prompt_tokens")
        original_bundle_count = len(row.get("evidence_bundles", []) or [])
        rendered_bundles = list(row.get("evidence_bundles", []) or [])
        dropped_bundle_count = 0
        compaction_stats: dict[str, Any] = {}
        if context_truncation or context_compaction:
            context, rendered_bundles, dropped_bundle_count, compaction_stats = render_context_with_truncation(
                row.get("evidence_bundles", []) or [],
                target_rendering,
                row_cfg,
                effective_top_bundles,
                context_token_budget,
                ordering_source,
                compaction_profile=compaction_profile,
                max_sentences_per_bundle=max_sentences_per_bundle,
                ace_renderer_variant=ace_renderer_variant,
            )
        elif rerender_context:
            context, compaction_stats = render_context_with_metadata(
                row.get("evidence_bundles", []) or [],
                rendering_profile=target_rendering,
                max_chars=int(row_cfg.get("max_context_chars", 24000)),
                token_budget=row_cfg.get("context_token_budget"),
                compaction_profile="none",
                max_sentences_per_bundle=max_sentences_per_bundle,
                ace_renderer_variant=ace_renderer_variant,
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
                "ace_native_prompt_variant": ace_native_prompt_variant or infer_ace_native_prompt_variant(target_prompt),
                "ace_renderer_variant": ace_renderer_variant,
                "rendering_profile": target_rendering,
                "prompt_experiment_type": prompt_experiment_type(source_prompt, target_prompt, rendering_changed, context_truncation, context_compaction),
                "context_truncation_enabled": context_truncation,
                "context_compaction_enabled": context_compaction,
                "compaction_profile": compaction_profile,
                "max_sentences_per_bundle": int(max_sentences_per_bundle or 3),
                "top_bundles": effective_top_bundles,
                "context_token_budget": context_token_budget,
                "ordering_source": ordering_source,
                "original_bundle_count": original_bundle_count,
                "rendered_bundle_count": int(compaction_stats.get("rendered_bundle_count", len(rendered_bundles)) or 0),
                "dropped_bundle_count": int(compaction_stats.get("dropped_bundle_count", dropped_bundle_count) or 0),
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
        if save_final_prompt:
            new_row["final_prompt_text"] = prompt
            new_row["prompt"] = prompt
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
        answers = row.get("answers")
        if isinstance(answers, list) and answers:
            new_row["gold_answer"] = str(answers[0])
        elif row.get("answer") is not None:
            new_row["gold_answer"] = str(row.get("answer"))
        new_row["context_tokens"] = int(new_row.get("rendered_context_tokens", 0) or 0)
        new_row["prompt_tokens"] = int(new_row.get("input_prompt_tokens", 0) or 0)
        new_row["top_k"] = effective_top_bundles
        if source_input_tokens is None:
            try:
                source_input_tokens = token_count(build_prompt(str(row.get("question", "")), source_context, source_prompt))
            except Exception:
                source_input_tokens = None
        source_input_value = float(source_input_tokens or 0.0)
        new_row["source_rendered_context_tokens"] = int(source_context_tokens or 0)
        new_row["source_input_prompt_tokens"] = int(source_input_value or 0)
        new_row["actual_context_tokens"] = int(new_row.get("rendered_context_tokens", 0) or 0)
        if context_token_budget is not None:
            saturation_target = min(float(context_token_budget), float(source_context_tokens or context_token_budget))
            new_row["budget_saturated"] = float(new_row["actual_context_tokens"]) >= 0.95 * max(1.0, saturation_target)
        else:
            new_row["budget_saturated"] = False
        new_row["token_reduction_rate"] = 1.0 - float(new_row.get("input_prompt_tokens", 0.0) or 0.0) / max(1e-9, source_input_value) if source_input_value > 0 else 0.0
        source_context_token_value = float(source_context_tokens or 0.0)
        new_row["compaction_token_reduction_rate"] = 1.0 - float(new_row.get("rendered_context_tokens", 0.0) or 0.0) / max(1e-9, source_context_token_value) if source_context_token_value > 0 else 0.0
        new_row["rendered_sentence_count"] = int(compaction_stats.get("rendered_sentence_count", 0) or 0)
        new_row["avg_rendered_sentences_per_bundle"] = float(compaction_stats.get("avg_sentences_per_bundle", 0.0) or 0.0)
        new_row["rendered_chain_count"] = int(compaction_stats.get("rendered_chain_count", 0) or 0)
        new_row["rendered_chain_sentence_count"] = int(compaction_stats.get("rendered_chain_sentence_count", 0) or 0)
        new_row["rendered_support_count"] = int(compaction_stats.get("rendered_support_count", compaction_stats.get("support_sentence_count", 0)) or 0)
        new_row["rendered_source_count"] = int(compaction_stats.get("rendered_source_count", 0) or 0)
        new_row["support_sentence_count"] = int(compaction_stats.get("support_sentence_count", 0) or 0)
        new_row["fallback_used"] = bool(compaction_stats.get("fallback_used", False))
        new_row["fallback_rate"] = 1.0 if new_row["fallback_used"] else 0.0
        new_row["dropped_sentence_count"] = int(compaction_stats.get("dropped_sentence_count", 0) or 0)
        new_row["dropped_chain_sentence_count"] = int(compaction_stats.get("dropped_chain_sentence_count", 0) or 0)
        new_row["dropped_support_count"] = int(compaction_stats.get("dropped_support_count", 0) or 0)
        new_row["dropped_source_count"] = int(compaction_stats.get("dropped_source_count", 0) or 0)
        new_row["duplicate_removed_count"] = int(compaction_stats.get("duplicate_removed_count", 0) or 0)
        new_row["source_removed_count"] = int(compaction_stats.get("source_removed_count", 0) or 0)
        new_row["metadata_removed_count"] = int(compaction_stats.get("metadata_removed_count", 0) or 0)
        out.append(new_row)
    return out


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "UNKNOWN"


def exact_insufficient_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    target = normalize_answer("insufficient information")
    return sum(
        1.0
        if normalize_answer(str(row.get("raw_prediction", row.get("prediction", "")) or "")) == target
        else 0.0
        for row in rows
    ) / max(1, len(rows))


def empty_answer_rate(rows: Sequence[Mapping[str, Any]]) -> float:
    return sum(
        1.0 if not str(row.get("raw_prediction", row.get("prediction", "")) or "").strip() else 0.0
        for row in rows
    ) / max(1, len(rows))


def run_command() -> str:
    return " ".join([sys.executable, *sys.argv])


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
    parser.add_argument("--ace-native-prompt-variant", choices=ACE_NATIVE_PROMPT_VARIANTS, default=None)
    parser.add_argument("--ace-renderer-variant", choices=ACE_RENDERER_VARIANTS, default="r0_current")
    parser.add_argument("--rendering-profile", choices=sorted(RENDERING_PROFILES), default=None)
    parser.add_argument("--failure-category", choices=FAILURE_CATEGORIES, default=None)
    parser.add_argument("--top-bundles", type=int, default=None)
    parser.add_argument("--context-token-budget", type=int, default=None)
    parser.add_argument("--ordering-source", choices=["current", "raw_score"], default="current")
    parser.add_argument("--compaction-profile", choices=COMPACTION_PROFILES, default="none")
    parser.add_argument("--max-sentences-per-bundle", type=int, default=3)
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--output-root", default="outputs/replay")
    parser.add_argument("--output-dir", default=None, help="Write this replay directly to the given directory.")
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
    parser.add_argument("--save-rendered-context", action="store_true")
    parser.add_argument("--save-final-prompt", action="store_true")
    args = parser.parse_args()

    ace_variant = args.ace_native_prompt_variant
    target_prompt = str(args.prompt_profile or "ace_rag_bundle_qa")
    if ace_variant:
        variant_prompt = resolve_ace_native_prompt_profile(ace_variant)
        if args.prompt_profile and args.prompt_profile != variant_prompt:
            raise SystemExit(
                f"--target-prompt {args.prompt_profile!r} conflicts with "
                f"--ace-native-prompt-variant {ace_variant!r} ({variant_prompt!r})"
            )
        target_prompt = variant_prompt
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
    cfg["prompt_profile"] = target_prompt
    cfg["rendering_profile"] = target_rendering
    cfg["ace_renderer_variant"] = args.ace_renderer_variant
    effective_top_bundles = 3 if str(args.compaction_profile or "none") in {"top3_chain_dedup", "top3_chain_dedup_no_sources"} and args.top_bundles is None else args.top_bundles
    context_truncation = effective_top_bundles is not None or args.context_token_budget is not None or args.ordering_source != "current"
    context_compaction = str(args.compaction_profile or "none") != "none"
    renderer_changed = str(args.ace_renderer_variant or "r0_current") != "r0_current"
    replayed = replay_rows(
        rows,
        dataset,
        source_prompt,
        target_prompt,
        source_rendering,
        target_rendering,
        bool(args.rendering_profile),
        cfg,
        args.limit,
        args.sample,
        args.failure_category,
        args.no_llm,
        args.dry_run,
        effective_top_bundles,
        args.context_token_budget,
        args.ordering_source,
        args.compaction_profile,
        args.max_sentences_per_bundle,
        ace_variant,
        args.ace_renderer_variant,
        args.save_final_prompt,
    )
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
    if context_truncation:
        if effective_top_bundles is not None:
            suffix = f"{suffix}_top{effective_top_bundles}"
        if args.context_token_budget is not None:
            suffix = f"{suffix}_ctx{args.context_token_budget}"
        if args.ordering_source != "current":
            suffix = f"{suffix}_{args.ordering_source}"
    if context_compaction:
        suffix = f"{suffix}_{args.compaction_profile}"
        if args.compaction_profile in {"sentence_cap", "sentence_cap_no_sources"}:
            suffix = f"{suffix}{args.max_sentences_per_bundle}"
    if renderer_changed:
        suffix = f"{suffix}_{args.ace_renderer_variant}"
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / timestamp / dataset / suffix
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_out = out_dir / "predictions.jsonl"
    write_jsonl(replayed, pred_out)
    result = evaluate_predictions(replayed, dataset=dataset, prompt_profile=target_prompt)
    result.update(
        {
            "source_predictions": str(pred_path),
            "source_prompt_profile": source_prompt,
            "target_prompt_profile": target_prompt,
            "ace_native_prompt_variant": ace_variant or infer_ace_native_prompt_variant(target_prompt),
            "ace_renderer_variant": args.ace_renderer_variant,
            "source_rendering_profile": source_rendering,
            "rendering_profile": target_rendering,
            "target_rendering_profile": target_rendering,
            "failure_category": args.failure_category,
            "prompt_experiment_type": prompt_experiment_type(source_prompt, target_prompt, bool(args.rendering_profile), context_truncation, context_compaction),
            "context_truncation_enabled": context_truncation,
            "context_compaction_enabled": context_compaction,
            "compaction_profile": str(args.compaction_profile or "none"),
            "max_sentences_per_bundle": int(args.max_sentences_per_bundle or 3),
            "top_bundles": effective_top_bundles,
            "context_token_budget": args.context_token_budget,
            "ordering_source": args.ordering_source,
            "avg_rendered_bundle_count": sum(float(x.get("rendered_bundle_count", 0.0) or 0.0) for x in replayed) / max(1, len(replayed)),
            "avg_dropped_bundle_count": sum(float(x.get("dropped_bundle_count", 0.0) or 0.0) for x in replayed) / max(1, len(replayed)),
            "avg_actual_context_tokens": sum(float(x.get("actual_context_tokens", x.get("rendered_context_tokens", 0.0)) or 0.0) for x in replayed) / max(1, len(replayed)),
            "budget_saturated_rate": sum(float(1.0 if x.get("budget_saturated") else 0.0) for x in replayed) / max(1, len(replayed)),
            "token_reduction_rate": sum(float(x.get("token_reduction_rate", 0.0) or 0.0) for x in replayed) / max(1, len(replayed)),
            "avg_rendered_sentence_count": sum(float(x.get("rendered_sentence_count", 0.0) or 0.0) for x in replayed) / max(1, len(replayed)),
            "avg_sentences_per_bundle": sum(float(x.get("avg_rendered_sentences_per_bundle", 0.0) or 0.0) for x in replayed) / max(1, len(replayed)),
            "avg_rendered_chain_count": sum(float(x.get("rendered_chain_count", 0.0) or 0.0) for x in replayed) / max(1, len(replayed)),
            "avg_rendered_chain_sentence_count": sum(float(x.get("rendered_chain_sentence_count", 0.0) or 0.0) for x in replayed) / max(1, len(replayed)),
            "avg_rendered_support_count": sum(float(x.get("rendered_support_count", 0.0) or 0.0) for x in replayed) / max(1, len(replayed)),
            "avg_rendered_source_count": sum(float(x.get("rendered_source_count", 0.0) or 0.0) for x in replayed) / max(1, len(replayed)),
            "avg_support_sentence_count": sum(float(x.get("support_sentence_count", 0.0) or 0.0) for x in replayed) / max(1, len(replayed)),
            "fallback_rate": sum(float(1.0 if x.get("fallback_used") else 0.0) for x in replayed) / max(1, len(replayed)),
            "avg_dropped_sentence_count": sum(float(x.get("dropped_sentence_count", 0.0) or 0.0) for x in replayed) / max(1, len(replayed)),
            "avg_dropped_chain_sentence_count": sum(float(x.get("dropped_chain_sentence_count", 0.0) or 0.0) for x in replayed) / max(1, len(replayed)),
            "avg_dropped_support_count": sum(float(x.get("dropped_support_count", 0.0) or 0.0) for x in replayed) / max(1, len(replayed)),
            "avg_dropped_source_count": sum(float(x.get("dropped_source_count", 0.0) or 0.0) for x in replayed) / max(1, len(replayed)),
            "avg_duplicate_removed_count": sum(float(x.get("duplicate_removed_count", 0.0) or 0.0) for x in replayed) / max(1, len(replayed)),
            "avg_source_removed_count": sum(float(x.get("source_removed_count", 0.0) or 0.0) for x in replayed) / max(1, len(replayed)),
            "avg_metadata_removed_count": sum(float(x.get("metadata_removed_count", 0.0) or 0.0) for x in replayed) / max(1, len(replayed)),
            "avg_compaction_token_reduction_rate": sum(float(x.get("compaction_token_reduction_rate", 0.0) or 0.0) for x in replayed) / max(1, len(replayed)),
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
    result["insufficient_information_rate"] = exact_insufficient_rate(replayed)
    result["empty_answer_rate"] = empty_answer_rate(replayed)
    dump_json(result, out_dir / "eval.json")
    (out_dir / "eval_summary.md").write_text(summary_markdown(dataset, result), encoding="utf-8")
    rag_summary = {
        "dataset": dataset,
        "method": "ACE-RAG",
        "prompt_setting": target_prompt,
        "ace_native_prompt_variant": ace_variant or infer_ace_native_prompt_variant(target_prompt),
        "ace_renderer_variant": args.ace_renderer_variant,
        "top_bundles": effective_top_bundles,
        "qa_top_k": effective_top_bundles,
        "compaction_profile": str(args.compaction_profile or "none"),
        "n": result.get("n"),
        "EM": result.get("em"),
        "F1": result.get("f1"),
        "Recall@5": result.get("support_title_recall"),
        "avg_context_tokens": result.get("avg_rendered_context_tokens", result.get("avg_context_tokens", result.get("context_tokens"))),
        "avg_prompt_text_tokens": result.get("avg_input_prompt_tokens"),
        "avg_prompt_tokens": result.get("avg_input_prompt_tokens"),
        "avg_prompt_input_tokens": result.get("avg_input_prompt_tokens"),
        "F1_per_1k_context_tokens": result.get("F1_per_1k_context_tokens"),
        "F1_per_1k_prompt_tokens": result.get("F1_per_1k_input_prompt_tokens"),
        "retrieval_ms": result.get("retrieval_latency_ms"),
        "generation_ms": result.get("generation_latency_ms"),
        "total_ms": result.get("latency_ms"),
        "insufficient_information_rate": result.get("insufficient_information_rate"),
        "empty_answer_rate": result.get("empty_answer_rate"),
        "output_path": str(pred_out),
        "eval_path": str(out_dir / "eval.json"),
        "git_commit": git_commit(),
        "command_used": run_command(),
        "source_predictions": str(pred_path),
        "prompt_template_preview": PROMPT_TEMPLATES[target_prompt][:2000],
        "prompt_hash_example": replayed[0].get("prompt_hash") if replayed else None,
        "evidence_bundles_hash_match_rate": result.get("evidence_bundles_hash_match_rate"),
        "rendered_context_hash_match_rate": result.get("rendered_context_hash_match_rate"),
    }
    dump_json(rag_summary, out_dir / "rag_summary.json")
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
                "ace_native_prompt_variant": ace_variant or infer_ace_native_prompt_variant(target_prompt),
                "ace_renderer_variant": args.ace_renderer_variant,
                "rendering_profile": target_rendering,
                "failure_category": args.failure_category,
                "prompt_experiment_type": result["prompt_experiment_type"],
                "context_truncation_enabled": context_truncation,
                "context_compaction_enabled": context_compaction,
                "compaction_profile": str(args.compaction_profile or "none"),
                "max_sentences_per_bundle": int(args.max_sentences_per_bundle or 3),
                "top_bundles": effective_top_bundles,
                "context_token_budget": args.context_token_budget,
                "ordering_source": args.ordering_source,
                "avg_rendered_bundle_count": result.get("avg_rendered_bundle_count"),
                "avg_dropped_bundle_count": result.get("avg_dropped_bundle_count"),
                "avg_actual_context_tokens": result.get("avg_actual_context_tokens"),
                "budget_saturated_rate": result.get("budget_saturated_rate"),
                "token_reduction_rate": result.get("token_reduction_rate"),
                "avg_rendered_sentence_count": result.get("avg_rendered_sentence_count"),
                "avg_sentences_per_bundle": result.get("avg_sentences_per_bundle"),
                "avg_rendered_chain_count": result.get("avg_rendered_chain_count"),
                "avg_rendered_chain_sentence_count": result.get("avg_rendered_chain_sentence_count"),
                "avg_rendered_support_count": result.get("avg_rendered_support_count"),
                "avg_rendered_source_count": result.get("avg_rendered_source_count"),
                "avg_support_sentence_count": result.get("avg_support_sentence_count"),
                "fallback_rate": result.get("fallback_rate"),
                "avg_dropped_sentence_count": result.get("avg_dropped_sentence_count"),
                "avg_dropped_chain_sentence_count": result.get("avg_dropped_chain_sentence_count"),
                "avg_dropped_support_count": result.get("avg_dropped_support_count"),
                "avg_dropped_source_count": result.get("avg_dropped_source_count"),
                "avg_duplicate_removed_count": result.get("avg_duplicate_removed_count"),
                "avg_source_removed_count": result.get("avg_source_removed_count"),
                "avg_metadata_removed_count": result.get("avg_metadata_removed_count"),
                "avg_compaction_token_reduction_rate": result.get("avg_compaction_token_reduction_rate"),
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
