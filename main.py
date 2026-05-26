from __future__ import annotations
import argparse, copy, hashlib, json, shutil, time
from pathlib import Path
from typing import Any, Dict, Mapping
from tabulate import tabulate
from tqdm import tqdm
from utils.data_loaders import load_dataset
from utils.eval_metrics import evaluate_predictions, summary_markdown
from utils.generation import ACE_NATIVE_PROMPT_VARIANTS, ACE_RENDERER_VARIANTS, COMPACTION_PROFILES, DEFAULT_PROMPT_PROFILE, DEFAULT_RENDERING_PROFILE, PROMPT_TEMPLATES, RENDERING_PROFILES, add_token_accounting_fields, generate_answer, infer_ace_native_prompt_variant, normalize_prediction_for_eval, resolve_ace_native_prompt_profile
from utils.indexing import LightweightEPCIndexer, ensure_mention_bridge_index
from utils.embedding import build_or_load_dense_indexes
from utils.io_utils import ExperimentLogger, dump_json, dump_yaml, ensure_dir, load_yaml, now_timestamp, to_jsonable
from utils.retrieval import QueryMedoidRetriever
from utils.text import safe_truncate
from utils.timing import TimingRecorder

def parse_args():
    p=argparse.ArgumentParser(description="QMRAG: Query-conditioned Medoid RAG runner")
    p.add_argument("--config", default="config/default.yaml"); p.add_argument("--datasets", nargs="+", default=None); p.add_argument("--mode", choices=["all","index","eval"], default="all")
    p.add_argument("--timestamp", default=None); p.add_argument("--limit", type=int, default=None); p.add_argument("--corpus-limit", type=int, default=None); p.add_argument("--output-root", default=None)
    p.add_argument("--reindex", "--force-reindex", action="store_true", dest="reindex"); p.add_argument("--rebuild-embeddings", action="store_true")
    p.add_argument("--no-embed", "--no-embedding", action="store_true", dest="no_embed"); p.add_argument("--no-llm", action="store_true")
    p.add_argument("--vllm-base-url", default=None); p.add_argument("--vllm-model", default=None); p.add_argument("--embedding-model-path", default=None); p.add_argument("--embedding-device", default=None); p.add_argument("--embedding-batch-size", type=int, default=None)
    p.add_argument("--prompt-profile", choices=sorted(PROMPT_TEMPLATES), default=None)
    p.add_argument("--ace-native-prompt-variant", choices=ACE_NATIVE_PROMPT_VARIANTS, default=None)
    p.add_argument("--ace-renderer-variant", choices=ACE_RENDERER_VARIANTS, default=None)
    p.add_argument("--rendering-profile", choices=sorted(RENDERING_PROFILES), default=None)
    p.add_argument("--compaction-profile", choices=sorted(COMPACTION_PROFILES), default=None)
    p.add_argument("--retrieval-variant", choices=["full_hetero","prop_text_only","prop_parent_anchor","prop_parent_mention_bidirectional"], default=None)
    p.add_argument("--seed-selection-variant", choices=["medoid_current","top_relevance","anchor_first","chain_potential"], default=None)
    p.add_argument("--residual-selection", choices=["residual_lexical","bridge_fullquery","residual_dense_only","residual_hybrid_lex_first","residual_dense_fallback","residual_unified_alignment"], default=None)
    p.add_argument("--ablation-variant", default=None)
    p.add_argument("--candidate-pool-size", type=int, default=None, help="Enable candidate cap ablation with this total candidate count.")
    p.add_argument("--stable-dedup-before-seed", action="store_true", help="Enable stable candidate dedup before seed selection as an ablation flag.")
    p.add_argument("--enable-timing", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()

def apply_overrides(cfg: Dict[str,Any], args) -> Dict[str,Any]:
    cfg=copy.deepcopy(cfg)
    gen_cfg=cfg.setdefault("generation",{})
    if args.ace_native_prompt_variant:
        variant_prompt=resolve_ace_native_prompt_profile(args.ace_native_prompt_variant)
        if args.prompt_profile and args.prompt_profile != variant_prompt:
            raise ValueError(f"--prompt-profile {args.prompt_profile!r} conflicts with --ace-native-prompt-variant {args.ace_native_prompt_variant!r} ({variant_prompt!r})")
        prompt_profile=variant_prompt
        gen_cfg["ace_native_prompt_variant"]=args.ace_native_prompt_variant
    else:
        prompt_profile=args.prompt_profile or gen_cfg.get("prompt_profile") or DEFAULT_PROMPT_PROFILE
        variant=infer_ace_native_prompt_variant(prompt_profile)
        if variant:
            gen_cfg["ace_native_prompt_variant"]=variant
    if prompt_profile not in PROMPT_TEMPLATES:
        raise ValueError(f"Unsupported prompt_profile={prompt_profile!r}; choices={sorted(PROMPT_TEMPLATES)}")
    gen_cfg["prompt_profile"]=prompt_profile
    if args.ace_renderer_variant:
        gen_cfg["ace_renderer_variant"]=args.ace_renderer_variant
    if args.rendering_profile:
        gen_cfg["rendering_profile"]=args.rendering_profile
    if args.compaction_profile:
        gen_cfg["compaction_profile"]=args.compaction_profile
    if args.output_root: cfg.setdefault("run",{})["output_root"]=args.output_root
    if args.no_llm: gen_cfg["provider"]="none"
    if args.vllm_base_url: gen_cfg.update({"provider":"vllm","base_url":args.vllm_base_url})
    if args.vllm_model: gen_cfg.update({"provider":"vllm","model":args.vllm_model})
    if args.no_embed:
        cfg.setdefault("indexing",{}).setdefault("embedding",{})["enabled"]=False; cfg.setdefault("retrieval",{}).setdefault("dense",{})["enabled"]=False; cfg.setdefault("retrieval",{}).setdefault("embedding",{})["enabled"]=False
    if args.embedding_model_path:
        cfg.setdefault("indexing",{}).setdefault("embedding",{})["model_path"]=args.embedding_model_path; cfg.setdefault("retrieval",{}).setdefault("embedding",{})["model_path"]=args.embedding_model_path
    if args.embedding_device:
        cfg.setdefault("indexing",{}).setdefault("embedding",{})["device"]=args.embedding_device; cfg.setdefault("retrieval",{}).setdefault("embedding",{})["device"]=args.embedding_device
    if args.embedding_batch_size:
        cfg.setdefault("indexing",{}).setdefault("embedding",{})["batch_size"]=args.embedding_batch_size; cfg.setdefault("retrieval",{}).setdefault("embedding",{})["batch_size"]=args.embedding_batch_size
    if args.retrieval_variant:
        cfg.setdefault("retrieval",{})["retrieval_variant"]=args.retrieval_variant
    if args.seed_selection_variant:
        cfg.setdefault("retrieval",{})["seed_selection_variant"]=args.seed_selection_variant
    if args.residual_selection:
        cfg.setdefault("retrieval",{}).setdefault("bridge",{})["selection"]=args.residual_selection
        cfg.setdefault("retrieval",{})["residual_selection"]=args.residual_selection
    if args.ablation_variant:
        cfg.setdefault("run",{})["ablation_variant"]=args.ablation_variant
    if args.candidate_pool_size is not None:
        cap_cfg=cfg.setdefault("retrieval",{}).setdefault("candidate_cap",{})
        cap_cfg["enabled"]=True
        cap_cfg["total_candidates"]=int(args.candidate_pool_size)
    if args.stable_dedup_before_seed:
        cfg.setdefault("retrieval",{}).setdefault("optimization",{})["stable_dedup_before_seed"]=True
    if args.enable_timing:
        cfg.setdefault("run",{})["enable_timing"]=True
    idx_emb=cfg.get("indexing",{}).get("embedding",{})
    cfg.setdefault("retrieval",{})["embedding"]={**idx_emb, **cfg.get("retrieval",{}).get("embedding",{})}
    return cfg

INDEX_REQUIRED_FILES=("chunks.jsonl","propositions.jsonl","entities.json","index_meta.json")
EMBEDDING_ARTIFACTS={"dense","prop_embeddings.npy","chunk_embeddings.npy","embedding_meta.json"}

def index_exists(d: Path) -> bool:
    return all((d/x).exists() for x in INDEX_REQUIRED_FILES)

def as_index_dir(path: Path) -> Path | None:
    if index_exists(path):
        return path
    nested=path/"index"
    if index_exists(nested):
        return nested
    return None

def latest_dataset_index(
    dataset: str,
    output_root: Path,
    current_index_dir: Path,
    retrieval_cfg: Mapping[str,Any] | None = None,
    prefer_dense: bool = False,
) -> Path | None:
    candidates=[]; seen=set(); current_resolved=current_index_dir.resolve()
    search_paths=list((output_root/dataset/"indexing").glob("*"))
    search_paths.extend(output_root.glob(f"*/{dataset}/index"))
    for raw_path in search_paths:
        index_dir=as_index_dir(raw_path)
        if index_dir is None:
            continue
        try:
            resolved=index_dir.resolve()
        except FileNotFoundError:
            continue
        if resolved==current_resolved or str(resolved) in seen:
            continue
        seen.add(str(resolved)); candidates.append(index_dir)
    if not candidates:
        return None
    if prefer_dense:
        dense_candidates=[p for p in candidates if dense_indexes_ready(p,retrieval_cfg or {})]
        if dense_candidates:
            candidates=dense_candidates
    return max(candidates, key=lambda p: ((p/"index_meta.json").stat().st_mtime, str(p)))

def dense_indexes_ready(index_dir: Path, retrieval_cfg: Mapping[str,Any]) -> bool:
    dense_cfg=retrieval_cfg.get("dense",{}) if retrieval_cfg else {}
    if not dense_cfg.get("enabled",False):
        return True
    dense_dir=index_dir/"dense"
    units=dense_cfg.get("units",["proposition","chunk"])
    return all((dense_dir/f"{unit}_embeddings.npy").exists() and (dense_dir/f"{unit}_ids.json").exists() for unit in units)

def copy_index_tree(source: Path, target: Path, include_embeddings: bool=True) -> None:
    ensure_dir(target)
    for name in INDEX_REQUIRED_FILES:
        shutil.copy2(source/name, target/name)
    for child in source.iterdir():
        if child.name in INDEX_REQUIRED_FILES:
            continue
        if not include_embeddings and child.name in EMBEDDING_ARTIFACTS:
            continue
        dest=target/child.name
        if child.is_dir():
            shutil.copytree(child,dest,dirs_exist_ok=True)
        elif child.is_file():
            shutil.copy2(child,dest)

def build_or_load_index(dataset: str, docs, cfg: Mapping[str,Any], index_dir: Path, output_root: Path, logger: ExperimentLogger, force=False, rebuild_embeddings=False):
    def ensure_bridge(idx: Dict[str,Any], path: Path, force_bridge: bool=False) -> Dict[str,Any]:
        return ensure_mention_bridge_index(idx,path,cfg,logger,force=force_bridge)
    if not force and index_exists(index_dir):
        logger.log(f"Loading EPC index: {index_dir}")
        with logger.time_block("index.load", dataset=dataset): idx=LightweightEPCIndexer.load(index_dir)
        idx=ensure_bridge(idx,index_dir)
        logger.log("Index meta: "+json.dumps(to_jsonable(idx.get("meta",{})),ensure_ascii=False)[:1500])
        return idx,index_dir,{"index_source":"current","index_dir":str(index_dir)}
    prefer_dense=bool(cfg.get("retrieval",{}).get("dense",{}).get("enabled",False)) and not rebuild_embeddings
    latest=None if force else latest_dataset_index(dataset,output_root,index_dir,cfg.get("retrieval",{}),prefer_dense=prefer_dense)
    if latest is not None:
        if rebuild_embeddings:
            logger.log(f"Reusing latest EPC index for {dataset}: source={latest} target={index_dir} rebuild_embeddings=True")
            with logger.time_block("index.reuse_latest", dataset=dataset, source=str(latest), target=str(index_dir), include_embeddings=False):
                copy_index_tree(latest,index_dir,include_embeddings=False)
            with logger.time_block("index.load", dataset=dataset): idx=LightweightEPCIndexer.load(index_dir)
            idx=ensure_bridge(idx,index_dir)
            logger.log("Index meta: "+json.dumps(to_jsonable(idx.get("meta",{})),ensure_ascii=False)[:1500])
            return idx,index_dir,{"index_source":"copied_latest","source_index_dir":str(latest),"index_dir":str(index_dir)}
        logger.log(f"Loading latest index for {dataset}: {latest}")
        with logger.time_block("index.load_latest", dataset=dataset, source=str(latest)): idx=LightweightEPCIndexer.load(latest)
        idx=ensure_bridge(idx,latest)
        logger.log("Index meta: "+json.dumps(to_jsonable(idx.get("meta",{})),ensure_ascii=False)[:1500])
        return idx,latest,{"index_source":"latest","source_index_dir":str(latest),"index_dir":str(latest)}
    if force:
        logger.log(f"Reindex requested; building EPC index for {dataset}: docs={len(docs)} target={index_dir}")
    else:
        logger.log(f"No reusable EPC index found; building EPC index for {dataset}: docs={len(docs)} target={index_dir}")
    index_cfg={**dict(cfg.get("indexing",{}) or {}),"bridge":dict(cfg.get("retrieval",{}).get("bridge",{}) or {})}
    with logger.time_block("index.build_epc", dataset=dataset, num_docs=len(docs)): idx=LightweightEPCIndexer(index_cfg, logger).build(docs)
    with logger.time_block("index.save_epc", dataset=dataset): LightweightEPCIndexer.save(idx,index_dir)
    idx=ensure_bridge(idx,index_dir)
    logger.log("Index meta: "+json.dumps(to_jsonable(idx.get("meta",{})),ensure_ascii=False)[:1500])
    return idx,index_dir,{"index_source":"built","index_dir":str(index_dir)}

def append_line(fh,row): fh.write(json.dumps(to_jsonable(row),ensure_ascii=False)+"\n"); fh.flush()

def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()

def add_generation_logging_fields(row: Dict[str,Any], gen: Mapping[str,Any], cfg: Mapping[str,Any]) -> Dict[str,Any]:
    log_cfg=cfg.get("logging",{}) if cfg else {}
    rendered_context=str(gen.get("rendered_context") or "")
    prompt=str(gen.get("prompt") or "")
    prompt_profile=str(row.get("prompt_profile") or gen.get("prompt_profile") or DEFAULT_PROMPT_PROFILE)
    raw_prediction=str(row.get("raw_prediction", gen.get("raw_prediction", gen.get("prediction",""))) or "")
    usage=gen.get("usage") or gen.get("llm_usage") or row.get("llm_usage")
    model=gen.get("model") or gen.get("llm_model") or row.get("llm_model")
    preview_chars=int(log_cfg.get("rendered_context_preview_chars",2000) or 2000)
    row=add_token_accounting_fields(row,prompt,rendered_context,raw_prediction,prompt_profile,cfg.get("generation",{}) if cfg else {},usage=usage,model=model)
    row["rendered_context_preview"]=safe_truncate(rendered_context, preview_chars)
    row["rendered_context_hash"]=sha256_text(rendered_context)
    row["prompt_hash"]=sha256_text(prompt)
    if bool(log_cfg.get("save_rendered_context",False)):
        row["rendered_context"]=rendered_context
    if bool(log_cfg.get("save_full_prompt",False)):
        row["prompt"]=prompt
    return row

def has_predictions(out_dir: Path) -> bool:
    return (out_dir/"predictions.jsonl").exists() or (out_dir/"예측결과.jsonl").exists()

def copy_compat_outputs(out_dir: Path, compat_dir: Path) -> None:
    if out_dir.resolve()==compat_dir.resolve():
        return
    ensure_dir(compat_dir)
    for name in ("config.yaml","predictions.jsonl","예측결과.jsonl","eval.json","eval_summary.md","timing_events.jsonl","timing_summary.json","timing_summary.md","runtime_diagnostics_summary.md"):
        src=out_dir/name
        if src.exists():
            shutil.copy2(src,compat_dir/name)

def copy_compat_index(index_dir: Path, compat_dir: Path, logger: ExperimentLogger) -> None:
    compat_index_dir=compat_dir/"index"
    if index_dir.resolve()==compat_index_dir.resolve():
        return
    logger.log(f"Copying compatibility index artifacts: source={index_dir} target={compat_index_dir}")
    with logger.time_block("index.copy_compat", source=str(index_dir), target=str(compat_index_dir)):
        copy_index_tree(index_dir,compat_index_dir,include_embeddings=False)

def result_prompt_profile(rows, fallback=None) -> str:
    for row in rows:
        if row.get("prompt_profile"):
            return str(row["prompt_profile"])
    return str(fallback or DEFAULT_PROMPT_PROFILE)

def prompt_experiment_type(prompt_profile: str, rendering_profile: str | None = None) -> str:
    if str(rendering_profile or DEFAULT_RENDERING_PROFILE) != DEFAULT_RENDERING_PROFILE:
        return "rendering_ablation"
    if prompt_profile=="common_qa":
        return "main_comparison"
    if prompt_profile=="strict_short_qa":
        return "format_ablation"
    if prompt_profile in {"qmrag_compact_chain_qa","qmrag_compact_chain_light","qmrag_compact_chain_short_qa"}:
        return "compact_prompt_ablation"
    if prompt_profile.startswith("acerag_native_"):
        return "ace_native_prompt_ablation"
    if prompt_profile in {"qmrag_bundle_qa","qmrag_bundle_light","qmrag_bundle_tiny","qmrag_bundle_short_qa"}:
        return "ablation"
    return "unknown"

def log_retrieval_summary(preds: list[Mapping[str,Any]], logger: ExperimentLogger) -> None:
    if not preds:
        return
    def avg(key: str) -> float:
        vals=[float((row.get("retrieval_diagnostics",{}) or {}).get(key,0.0) or 0.0) for row in preds]
        return sum(vals)/max(1,len(vals))
    summary={
        "avg_bridge_title_count":round(avg("bridge_title_count"),6),
        "avg_bridge_bundle_count":round(avg("bridge_bundle_count"),6),
        "bridge_connected_rate":round(sum(1.0 if (row.get("retrieval_diagnostics",{}) or {}).get("has_bridge_connected") else 0.0 for row in preds)/max(1,len(preds)),6),
        "answer_slot_aligned_rate":round(sum(1.0 if (row.get("retrieval_diagnostics",{}) or {}).get("has_answer_slot_aligned") else 0.0 for row in preds)/max(1,len(preds)),6),
        "chain_complete_v2_rate":round(sum(1.0 if (row.get("retrieval_diagnostics",{}) or {}).get("has_chain_complete_v2") else 0.0 for row in preds)/max(1,len(preds)),6),
        "anchor_connected_chain_complete_rate":round(sum(1.0 if (row.get("retrieval_diagnostics",{}) or {}).get("has_anchor_connected_chain_complete") else 0.0 for row in preds)/max(1,len(preds)),6),
        "anchor_mismatch_chain_rate":round(sum(1.0 if (row.get("retrieval_diagnostics",{}) or {}).get("has_anchor_mismatch_chain") else 0.0 for row in preds)/max(1,len(preds)),6),
        "multi_anchor_bundle_rate":round(sum(1.0 if (row.get("retrieval_diagnostics",{}) or {}).get("has_multi_anchor_bundle") else 0.0 for row in preds)/max(1,len(preds)),6),
        "generic_relation_top1_rate":round(sum(1.0 if (row.get("retrieval_diagnostics",{}) or {}).get("generic_relation_top1") else 0.0 for row in preds)/max(1,len(preds)),6),
        "query_anchor_coverage_rate":round(avg("query_anchor_coverage"),6),
        "avg_residual_coverage_count":round(avg("avg_residual_coverage_count"),6),
        "chain_complete_rate":round(sum(1.0 if (row.get("retrieval_diagnostics",{}) or {}).get("has_chain_complete") else 0.0 for row in preds)/max(1,len(preds)),6),
        "retrieval_ms":round(avg("timings.total_retrieval_s"),6),
        "context_tokens":round(avg("context_tokens"),6),
    }
    # timings are nested, compute total retrieval separately.
    totals=[float(((row.get("retrieval_diagnostics",{}) or {}).get("timings",{}) or {}).get("total_retrieval_s",0.0) or 0.0)*1000.0 for row in preds]
    summary["retrieval_ms"]=round(sum(totals)/max(1,len(totals)),6)
    logger.log("Retrieval summary: "+json.dumps(summary,ensure_ascii=False))
    logger.event({"event":"retrieval.summary",**summary})

def runtime_diagnostics_markdown(dataset: str, result: Mapping[str,Any], timing_summary: Mapping[str,Any] | None = None) -> str:
    timing_summary=timing_summary or {}
    stages=dict(timing_summary.get("stages",{}) or {})
    timing_rows=[]
    for stage,row in sorted(stages.items(), key=lambda kv: float((kv[1] or {}).get("total_ms",0.0) or 0.0), reverse=True):
        timing_rows.append((stage,row))
    metric_rows=[
        ("dataset",dataset),
        ("retrieval_variant",result.get("retrieval_variant","full_hetero")),
        ("seed_selection_variant",result.get("seed_selection_variant","medoid_current")),
        ("n",result.get("n",0)),
        ("retrieval_ms",f"{result.get('retrieval_latency_ms',0):.3f}"),
        ("seed_selection_ms",f"{result.get('seed_selection_ms',0):.3f}"),
        ("num_query_embedding_calls",f"{result.get('num_query_embedding_calls',0):.3f}"),
        ("num_dense_search_calls",f"{result.get('num_dense_search_calls',0):.3f}"),
        ("num_bm25_search_calls",f"{result.get('num_bm25_search_calls',0):.3f}"),
        ("num_title_search_calls",f"{result.get('num_title_search_calls',0):.3f}"),
        ("num_chunk_search_calls",f"{result.get('num_chunk_search_calls',0):.3f}"),
        ("num_proposition_search_calls",f"{result.get('num_proposition_search_calls',0):.3f}"),
        ("raw_candidate_count",f"{result.get('raw_candidate_count',0):.3f}"),
        ("unique_candidate_count",f"{result.get('unique_candidate_count',0):.3f}"),
        ("duplicate_candidate_count",f"{result.get('duplicate_candidate_count',0):.3f}"),
        ("candidate_merge_reduction_rate",f"{result.get('candidate_merge_reduction_rate',0):.4f}"),
        ("num_candidate_score_computations",f"{result.get('num_candidate_score_computations',0):.3f}"),
        ("num_candidate_score_cache_hits",f"{result.get('num_candidate_score_cache_hits',0):.3f}"),
        ("num_bridge_title_lookups",f"{result.get('num_bridge_title_lookups',0):.3f}"),
        ("num_bridge_title_cache_hits",f"{result.get('num_bridge_title_cache_hits',0):.3f}"),
        ("num_bridge_prop_score_computations",f"{result.get('num_bridge_prop_score_computations',0):.3f}"),
        ("num_bridge_prop_score_cache_hits",f"{result.get('num_bridge_prop_score_cache_hits',0):.3f}"),
        ("unique_bridge_title_count",f"{result.get('unique_bridge_title_count',0):.3f}"),
        ("duplicate_bridge_title_count",f"{result.get('duplicate_bridge_title_count',0):.3f}"),
        ("num_pairwise_similarity_computations",f"{result.get('num_pairwise_similarity_computations',0):.3f}"),
        ("num_pairwise_similarity_cache_hits",f"{result.get('num_pairwise_similarity_cache_hits',0):.3f}"),
        ("pairwise_matrix_size",f"{result.get('pairwise_matrix_size',0):.3f}"),
        ("candidate_count_by_type",json.dumps(result.get("candidate_count_by_type",{}),ensure_ascii=False)),
        ("seed_unit_type_distribution",json.dumps(result.get("seed_unit_type_distribution",{}),ensure_ascii=False)),
    ]
    lines=["# Runtime Diagnostics Summary","",f"- dataset: {dataset}","", "## Duplicate And Cache Diagnostics", "", "| metric | value |", "|---|---:|"]
    lines.extend(f"| {k} | {v} |" for k,v in metric_rows)
    lines.extend(["", "## Timing Bottlenecks", "", "| stage | count | total_ms | mean_ms | p50_ms | p95_ms | max_ms | extra_mean |", "|---|---:|---:|---:|---:|---:|---:|---|"])
    for stage,row in timing_rows:
        extra=json.dumps(row.get("extra_mean",{}),ensure_ascii=False,sort_keys=True)
        lines.append(f"| {stage} | {row.get('count',0)} | {row.get('total_ms',0):.3f} | {row.get('mean_ms',0):.3f} | {row.get('p50_ms',0):.3f} | {row.get('p95_ms',0):.3f} | {row.get('max_ms',0):.3f} | {extra} |")
    return "\n".join(lines)+"\n"

def eval_only(dataset: str, out_dir: Path, logger: ExperimentLogger, prompt_profile: str | None = None, compat_dir: Path | None = None):
    path=out_dir/"예측결과.jsonl" if (out_dir/"예측결과.jsonl").exists() else out_dir/"predictions.jsonl"
    rows=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    prompt_profile=result_prompt_profile(rows,prompt_profile)
    with logger.time_block("eval", dataset=dataset, n=len(rows)):
        res=evaluate_predictions(rows,dataset=dataset,prompt_profile=prompt_profile); dump_json(res,out_dir/"eval.json"); (out_dir/"eval_summary.md").write_text(summary_markdown(dataset,res),encoding="utf-8")
    if compat_dir is not None:
        copy_compat_outputs(out_dir,compat_dir)
    return {"dataset":dataset, **{k:v for k,v in res.items() if k!="per_example"}}

def run_dataset(dataset: str, cfg: Dict[str,Any], args, timestamp: str):
    output_root=Path(cfg.get("run",{}).get("output_root","outputs"))
    index_target_dir=output_root/dataset/"indexing"/timestamp
    out_dir=ensure_dir(output_root/dataset/"eval"/timestamp)
    compat_dir=output_root/timestamp/dataset
    if args.mode=="eval" and not has_predictions(out_dir) and has_predictions(compat_dir):
        out_dir=compat_dir
    logger=ExperimentLogger(out_dir, echo=bool(cfg.get("run",{}).get("echo_logs",True)))
    timing=TimingRecorder(out_dir, enabled=bool(cfg.get("run",{}).get("enable_timing",False) or getattr(args,"enable_timing",False)))
    logger.log(f"Run dataset={dataset} mode={args.mode} timestamp={timestamp}")
    logger.log(f"Eval output: {out_dir}")
    logger.log(f"Index target: {index_target_dir}")
    dump_yaml(cfg,out_dir/"config.yaml")
    prompt_profile=str(cfg.get("generation",{}).get("prompt_profile") or DEFAULT_PROMPT_PROFILE)
    ace_native_prompt_variant=str(cfg.get("generation",{}).get("ace_native_prompt_variant") or infer_ace_native_prompt_variant(prompt_profile) or "")
    rendering_profile=str(cfg.get("generation",{}).get("rendering_profile") or DEFAULT_RENDERING_PROFILE)
    compaction_profile=str(cfg.get("generation",{}).get("compaction_profile") or "none")
    ace_renderer_variant=str(cfg.get("generation",{}).get("ace_renderer_variant") or "r0_current")
    ablation_variant=str(cfg.get("run",{}).get("ablation_variant") or "")
    if args.mode=="eval": return eval_only(dataset,out_dir,logger,prompt_profile,compat_dir)
    ds_cfg=cfg.get("datasets",{}).get(dataset)
    if not ds_cfg: raise ValueError(f"Dataset {dataset} not in config")
    with timing.time_block(dataset=dataset, stage="data_load", query_id=None) as tdata:
        with logger.time_block("data.load", dataset=dataset): qas,docs=load_dataset(dataset,ds_cfg,args.limit,args.corpus_limit)
        tdata["num_items_out"]=len(qas)
    logger.log(f"Loaded QA={len(qas)} docs={len(docs)}")
    with timing.time_block(dataset=dataset, stage="index_load", query_id=None, num_items_in=len(docs)) as tidx:
        idx,index_dir,index_info=build_or_load_index(dataset,docs,cfg,index_target_dir,output_root,logger,args.reindex,args.rebuild_embeddings)
        tidx["num_items_out"]=len(idx.get("propositions",[]) or [])
    if index_dir.resolve()==index_target_dir.resolve():
        dump_yaml(cfg,index_target_dir/"config.yaml")
    logger.log(f"Using index_dir={index_dir} source={index_info.get('index_source')}")
    dense_indexes={}
    if cfg.get("retrieval",{}).get("dense",{}).get("enabled",False) and not args.no_embed:
        dense_force=args.rebuild_embeddings or args.reindex
        if not dense_force and not dense_indexes_ready(index_dir,cfg.get("retrieval",{})) and index_dir.resolve()!=index_target_dir.resolve():
            logger.log(f"Latest index is missing dense artifacts; copying EPC index for dense build: source={index_dir} target={index_target_dir}")
            with logger.time_block("index.copy_for_dense", dataset=dataset, source=str(index_dir), target=str(index_target_dir)):
                copy_index_tree(index_dir,index_target_dir,include_embeddings=False)
            dump_yaml(cfg,index_target_dir/"config.yaml")
            with logger.time_block("index.load", dataset=dataset): idx=LightweightEPCIndexer.load(index_target_dir)
            idx=ensure_mention_bridge_index(idx,index_target_dir,cfg,logger)
            index_info={**index_info,"index_source":"copied_latest_for_dense","source_index_dir":str(index_dir),"index_dir":str(index_target_dir)}
            index_dir=index_target_dir
        logger.log("Building/loading dense indexes")
        with timing.time_block(dataset=dataset, stage="embedding_load", query_id=None, num_items_in=len(idx.get("propositions",[]) or [])) as temb:
            with logger.time_block("index.build_dense_indexes", dataset=dataset):
                dense_indexes=build_or_load_dense_indexes(idx,cfg.get("retrieval",{}),index_dir,logger,force=dense_force)
            temb["num_items_out"]=len(dense_indexes)
        logger.log(f"Dense indexes ready: units={list(dense_indexes.keys())}")
    copy_compat_index(index_dir,compat_dir,logger)
    if args.mode=="index":
        timing.write_summary()
        return {"dataset":dataset,"n":0,"status":"indexed","prompt_profile":prompt_profile,"rendering_profile":rendering_profile,"compaction_profile":compaction_profile,"retrieval_variant":str(cfg.get("retrieval",{}).get("retrieval_variant","full_hetero")),"seed_selection_variant":str(cfg.get("retrieval",{}).get("seed_selection_variant","medoid_current")),"index_dir":str(index_dir),**index_info,"index_meta":idx.get("meta",{})}
    retriever=QueryMedoidRetriever(idx,cfg.get("retrieval",{}),dense_indexes,logger); preds=[]; pko=out_dir/"예측결과.jsonl"; pen=out_dir/"predictions.jsonl"
    for p in [pko,pen]:
        if p.exists(): p.unlink()
    with logger.time_block("run.examples", dataset=dataset, n=len(qas)):
        with open(pko,"a",encoding="utf-8") as fko, open(pen,"a",encoding="utf-8") as fen:
            for qa in tqdm(qas, desc=dataset, ncols=100):
                try:
                    with logger.time_block("retrieve.one", dataset=dataset, qid=qa.id): ret=retriever.retrieve(qa.question, qa.metadata, dataset=dataset, query_id=qa.id, timing_recorder=timing)
                    t=time.perf_counter()
                    with logger.time_block("generate.one", dataset=dataset, qid=qa.id): gen=generate_answer(qa.question, ret["evidence_bundles"], cfg.get("generation",{}))
                    gen_timings=dict(gen.get("generation_stage_timings_s",{}) or {})
                    timing.record_duration(dataset=dataset, query_id=qa.id, stage="context_rendering", duration_s=float(gen_timings.get("context_rendering",0.0) or 0.0), num_items_in=len(ret.get("evidence_bundles",[]) or []), num_items_out=1, extra={"retrieval_variant":str(cfg.get("retrieval",{}).get("retrieval_variant","full_hetero"))})
                    timing.record_duration(dataset=dataset, query_id=qa.id, stage="generation", duration_s=float(gen_timings.get("generation",gen.get("generation_latency_s",0.0)) or 0.0), num_items_in=1, num_items_out=1, extra={"provider":gen.get("generation_provider") or gen.get("llm_provider")})
                    raw_prediction=str(gen.get("raw_prediction",gen.get("prediction","")) or "")
                    prediction=normalize_prediction_for_eval(gen.get("prediction",raw_prediction))
                    generation_provider=gen.get("generation_provider") or gen.get("llm_provider") or cfg.get("generation",{}).get("provider")
                    row_prompt_profile=str(gen.get("prompt_profile",prompt_profile))
                    row_rendering_profile=str(gen.get("rendering_profile",rendering_profile) or DEFAULT_RENDERING_PROFILE)
                    row_compaction_profile=str(gen.get("compaction_profile",compaction_profile) or "none")
                    retrieval_variant=str(cfg.get("retrieval",{}).get("retrieval_variant","full_hetero"))
                    seed_selection_variant=str(cfg.get("retrieval",{}).get("seed_selection_variant","medoid_current"))
                    ret["diagnostics"]["ablation_variant"]=ablation_variant
                    row={"dataset":dataset,"id":qa.id,"question":qa.question,"raw_prediction":raw_prediction,"prediction":prediction,"answers":qa.answers,"support_titles":qa.support_titles,"support_facts":qa.support_facts,"prompt_profile":row_prompt_profile,"ace_native_prompt_variant":ace_native_prompt_variant or infer_ace_native_prompt_variant(row_prompt_profile),"ace_renderer_variant":str(gen.get("ace_renderer_variant") or ace_renderer_variant),"rendering_profile":row_rendering_profile,"compaction_profile":row_compaction_profile,"context_compaction_enabled":row_compaction_profile!="none","prompt_experiment_type":prompt_experiment_type(row_prompt_profile,row_rendering_profile),"retrieval_variant":retrieval_variant,"seed_selection_variant":seed_selection_variant,"ablation_variant":ablation_variant,"generation_provider":generation_provider,"evidence_bundles":ret["evidence_bundles"],"seeds":ret["seeds"],"retrieval_diagnostics":ret["diagnostics"],"generation_latency_s":float(gen.get("generation_latency_s",round(time.perf_counter()-t,6)) or 0.0),"llm_provider":generation_provider,"llm_model":gen.get("model"),"llm_usage":gen.get("usage")}
                    row.update(dict(gen.get("context_stats",{}) or {}))
                    if gen.get("generation_error"): row["generation_error"]=gen.get("generation_error")
                    row=add_generation_logging_fields(row,gen,cfg)
                except Exception as e:
                    logger.event({"event":"example.error","dataset":dataset,"qid":qa.id,"error":repr(e)})
                    if not args.continue_on_error: raise
                    generation_provider=cfg.get("generation",{}).get("provider")
                    row={"dataset":dataset,"id":qa.id,"question":qa.question,"raw_prediction":"","prediction":"","answers":qa.answers,"support_titles":qa.support_titles,"prompt_profile":prompt_profile,"ace_native_prompt_variant":ace_native_prompt_variant or infer_ace_native_prompt_variant(prompt_profile),"ace_renderer_variant":ace_renderer_variant,"rendering_profile":rendering_profile,"compaction_profile":compaction_profile,"context_compaction_enabled":compaction_profile!="none","prompt_experiment_type":prompt_experiment_type(prompt_profile,rendering_profile),"retrieval_variant":str(cfg.get("retrieval",{}).get("retrieval_variant","full_hetero")),"seed_selection_variant":str(cfg.get("retrieval",{}).get("seed_selection_variant","medoid_current")),"ablation_variant":ablation_variant,"generation_provider":generation_provider,"error":repr(e),"evidence_bundles":[],"seeds":[],"retrieval_diagnostics":{"ablation_variant":ablation_variant,"candidate_count":0,"seed_count":0,"bundle_count":0,"context_tokens":0,"timings":{}},"generation_latency_s":0.0,"llm_provider":generation_provider}
                    row=add_generation_logging_fields(row,{"rendered_context":"","prompt":""},cfg)
                preds.append(row)
                with timing.time_block(dataset=dataset, query_id=qa.id, stage="write_outputs", num_items_in=1) as twrite:
                    append_line(fko,row); append_line(fen,row)
                    twrite["num_items_out"]=2
    with timing.time_block(dataset=dataset, stage="evaluation", query_id=None, num_items_in=len(preds)):
        with logger.time_block("eval", dataset=dataset, n=len(preds)):
            log_retrieval_summary(preds,logger)
            res=evaluate_predictions(preds,dataset=dataset,prompt_profile=prompt_profile); res["index_dir"]=str(index_dir); res["index_source"]=index_info.get("index_source"); res["bridge_config"]=dict(cfg.get("retrieval",{}).get("bridge",{}) or {})
            res["retrieval_variant"]=str(cfg.get("retrieval",{}).get("retrieval_variant","full_hetero"))
            res["seed_selection_variant"]=str(cfg.get("retrieval",{}).get("seed_selection_variant","medoid_current"))
            res["ablation_variant"]=ablation_variant
            res["ace_native_prompt_variant"]=ace_native_prompt_variant or infer_ace_native_prompt_variant(prompt_profile)
            res["compaction_profile"]=compaction_profile
            res["ace_renderer_variant"]=ace_renderer_variant
            res["context_compaction_enabled"]=compaction_profile!="none"
            if index_info.get("source_index_dir"): res["source_index_dir"]=index_info.get("source_index_dir")
            dump_json(res,out_dir/"eval.json"); (out_dir/"eval_summary.md").write_text(summary_markdown(dataset,res),encoding="utf-8")
    timing_summary=timing.write_summary()
    if timing_summary:
        res["timing_summary"]=timing_summary
    (out_dir/"runtime_diagnostics_summary.md").write_text(runtime_diagnostics_markdown(dataset,res,timing_summary),encoding="utf-8")
    copy_compat_outputs(out_dir,compat_dir)
    return {"dataset":dataset, **{k:v for k,v in res.items() if k!="per_example"}}

def main():
    args=parse_args(); cfg=apply_overrides(load_yaml(args.config), args); timestamp=args.timestamp or now_timestamp(); datasets=args.datasets or list(cfg.get("datasets",{}).keys()); rows=[]
    for ds in datasets: rows.append(run_dataset(ds,cfg,args,timestamp))
    if rows:
        table=[]
        for s in rows: table.append({"dataset":s.get("dataset"),"prompt":s.get("prompt_profile","-"),"n":s.get("n",0),"EM":f"{s.get('em',0):.4f}" if "em" in s else "-","F1":f"{s.get('f1',0):.4f}" if "f1" in s else "-","AnsContains":f"{s.get('answer_contains',0):.4f}" if "answer_contains" in s else "-","SupportRecall":f"{s.get('support_title_recall',0):.4f}" if "support_title_recall" in s else "-","SR/1kTok":f"{s.get('support_recall_per_1k_tokens',0):.4f}" if "support_recall_per_1k_tokens" in s else "-","CtxTok":f"{s.get('context_tokens',0):.1f}" if "context_tokens" in s else "-","LatencyMs":f"{s.get('latency_ms',0):.1f}" if "latency_ms" in s else "-","DenseRate":f"{s.get('dense_enabled_rate',0):.2f}" if "dense_enabled_rate" in s else "-"})
        print("\n"+tabulate(table, headers="keys", tablefmt="github"))
if __name__=="__main__": main()
