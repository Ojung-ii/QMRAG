from __future__ import annotations
import argparse, copy, json, shutil, time
from pathlib import Path
from typing import Any, Dict, Mapping
from tabulate import tabulate
from tqdm import tqdm
from utils.data_loaders import load_dataset
from utils.eval_metrics import evaluate_predictions, summary_markdown
from utils.generation import DEFAULT_PROMPT_PROFILE, PROMPT_TEMPLATES, generate_answer
from utils.indexing import LightweightEPCIndexer
from utils.embedding import build_or_load_dense_indexes
from utils.io_utils import ExperimentLogger, dump_json, dump_yaml, ensure_dir, load_yaml, now_timestamp, to_jsonable
from utils.retrieval import QueryMedoidRetriever

def parse_args():
    p=argparse.ArgumentParser(description="QMRAG: Query-conditioned Medoid RAG runner")
    p.add_argument("--config", default="config/default.yaml"); p.add_argument("--datasets", nargs="+", default=None); p.add_argument("--mode", choices=["all","index","eval"], default="all")
    p.add_argument("--timestamp", default=None); p.add_argument("--limit", type=int, default=None); p.add_argument("--corpus-limit", type=int, default=None); p.add_argument("--output-root", default=None)
    p.add_argument("--reindex", "--force-reindex", action="store_true", dest="reindex"); p.add_argument("--rebuild-embeddings", action="store_true")
    p.add_argument("--no-embed", "--no-embedding", action="store_true", dest="no_embed"); p.add_argument("--no-llm", action="store_true")
    p.add_argument("--vllm-base-url", default=None); p.add_argument("--vllm-model", default=None); p.add_argument("--embedding-model-path", default=None); p.add_argument("--embedding-device", default=None); p.add_argument("--embedding-batch-size", type=int, default=None)
    p.add_argument("--prompt-profile", choices=sorted(PROMPT_TEMPLATES), default=None)
    p.add_argument("--continue-on-error", action="store_true")
    return p.parse_args()

def apply_overrides(cfg: Dict[str,Any], args) -> Dict[str,Any]:
    cfg=copy.deepcopy(cfg)
    if args.output_root: cfg.setdefault("run",{})["output_root"]=args.output_root
    if args.no_llm: cfg.setdefault("generation",{})["provider"]="none"
    if args.vllm_base_url: cfg.setdefault("generation",{}).update({"provider":"vllm","base_url":args.vllm_base_url})
    if args.vllm_model: cfg.setdefault("generation",{}).update({"provider":"vllm","model":args.vllm_model})
    if args.prompt_profile: cfg.setdefault("generation",{})["prompt_profile"]=args.prompt_profile
    if args.no_embed:
        cfg.setdefault("indexing",{}).setdefault("embedding",{})["enabled"]=False; cfg.setdefault("retrieval",{}).setdefault("dense",{})["enabled"]=False; cfg.setdefault("retrieval",{}).setdefault("embedding",{})["enabled"]=False
    if args.embedding_model_path:
        cfg.setdefault("indexing",{}).setdefault("embedding",{})["model_path"]=args.embedding_model_path; cfg.setdefault("retrieval",{}).setdefault("embedding",{})["model_path"]=args.embedding_model_path
    if args.embedding_device:
        cfg.setdefault("indexing",{}).setdefault("embedding",{})["device"]=args.embedding_device; cfg.setdefault("retrieval",{}).setdefault("embedding",{})["device"]=args.embedding_device
    if args.embedding_batch_size:
        cfg.setdefault("indexing",{}).setdefault("embedding",{})["batch_size"]=args.embedding_batch_size; cfg.setdefault("retrieval",{}).setdefault("embedding",{})["batch_size"]=args.embedding_batch_size
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

def latest_dataset_index(dataset: str, output_root: Path, current_index_dir: Path) -> Path | None:
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
    if not force and index_exists(index_dir):
        logger.log(f"Loading EPC index: {index_dir}")
        with logger.time_block("index.load", dataset=dataset): idx=LightweightEPCIndexer.load(index_dir)
        logger.log("Index meta: "+json.dumps(to_jsonable(idx.get("meta",{})),ensure_ascii=False)[:1500])
        return idx,index_dir,{"index_source":"current","index_dir":str(index_dir)}
    latest=None if force else latest_dataset_index(dataset,output_root,index_dir)
    if latest is not None:
        if rebuild_embeddings:
            logger.log(f"Reusing latest EPC index for {dataset}: source={latest} target={index_dir} rebuild_embeddings=True")
            with logger.time_block("index.reuse_latest", dataset=dataset, source=str(latest), target=str(index_dir), include_embeddings=False):
                copy_index_tree(latest,index_dir,include_embeddings=False)
            with logger.time_block("index.load", dataset=dataset): idx=LightweightEPCIndexer.load(index_dir)
            logger.log("Index meta: "+json.dumps(to_jsonable(idx.get("meta",{})),ensure_ascii=False)[:1500])
            return idx,index_dir,{"index_source":"copied_latest","source_index_dir":str(latest),"index_dir":str(index_dir)}
        logger.log(f"Loading latest index for {dataset}: {latest}")
        with logger.time_block("index.load_latest", dataset=dataset, source=str(latest)): idx=LightweightEPCIndexer.load(latest)
        logger.log("Index meta: "+json.dumps(to_jsonable(idx.get("meta",{})),ensure_ascii=False)[:1500])
        return idx,latest,{"index_source":"latest","source_index_dir":str(latest),"index_dir":str(latest)}
    if force:
        logger.log(f"Reindex requested; building EPC index for {dataset}: docs={len(docs)} target={index_dir}")
    else:
        logger.log(f"No reusable EPC index found; building EPC index for {dataset}: docs={len(docs)} target={index_dir}")
    with logger.time_block("index.build_epc", dataset=dataset, num_docs=len(docs)): idx=LightweightEPCIndexer(cfg.get("indexing",{}), logger).build(docs)
    with logger.time_block("index.save_epc", dataset=dataset): LightweightEPCIndexer.save(idx,index_dir)
    logger.log("Index meta: "+json.dumps(to_jsonable(idx.get("meta",{})),ensure_ascii=False)[:1500])
    return idx,index_dir,{"index_source":"built","index_dir":str(index_dir)}

def append_line(fh,row): fh.write(json.dumps(to_jsonable(row),ensure_ascii=False)+"\n"); fh.flush()

def result_prompt_profile(rows, fallback=None) -> str:
    for row in rows:
        if row.get("prompt_profile"):
            return str(row["prompt_profile"])
    return str(fallback or DEFAULT_PROMPT_PROFILE)

def eval_only(dataset: str, out_dir: Path, logger: ExperimentLogger, prompt_profile: str | None = None):
    path=out_dir/"예측결과.jsonl" if (out_dir/"예측결과.jsonl").exists() else out_dir/"predictions.jsonl"
    rows=[json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    prompt_profile=result_prompt_profile(rows,prompt_profile)
    with logger.time_block("eval", dataset=dataset, n=len(rows)):
        res=evaluate_predictions(rows); res["prompt_profile"]=prompt_profile; dump_json(res,out_dir/"eval.json"); (out_dir/"eval_summary.md").write_text(summary_markdown(dataset,res),encoding="utf-8")
    return {"dataset":dataset, **{k:v for k,v in res.items() if k!="per_example"}}

def run_dataset(dataset: str, cfg: Dict[str,Any], args, timestamp: str):
    output_root=Path(cfg.get("run",{}).get("output_root","outputs"))
    index_target_dir=output_root/dataset/"indexing"/timestamp
    out_dir=ensure_dir(output_root/dataset/"eval"/timestamp)
    logger=ExperimentLogger(out_dir, echo=bool(cfg.get("run",{}).get("echo_logs",True)))
    logger.log(f"Run dataset={dataset} mode={args.mode} timestamp={timestamp}")
    logger.log(f"Eval output: {out_dir}")
    logger.log(f"Index target: {index_target_dir}")
    dump_yaml(cfg,out_dir/"config.yaml")
    prompt_profile=str(cfg.get("generation",{}).get("prompt_profile") or DEFAULT_PROMPT_PROFILE)
    if args.mode=="eval": return eval_only(dataset,out_dir,logger,prompt_profile)
    ds_cfg=cfg.get("datasets",{}).get(dataset)
    if not ds_cfg: raise ValueError(f"Dataset {dataset} not in config")
    with logger.time_block("data.load", dataset=dataset): qas,docs=load_dataset(dataset,ds_cfg,args.limit,args.corpus_limit)
    logger.log(f"Loaded QA={len(qas)} docs={len(docs)}")
    idx,index_dir,index_info=build_or_load_index(dataset,docs,cfg,index_target_dir,output_root,logger,args.reindex,args.rebuild_embeddings)
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
            index_info={**index_info,"index_source":"copied_latest_for_dense","source_index_dir":str(index_dir),"index_dir":str(index_target_dir)}
            index_dir=index_target_dir
        logger.log("Building/loading dense indexes")
        with logger.time_block("index.build_dense_indexes", dataset=dataset):
            dense_indexes=build_or_load_dense_indexes(idx,cfg.get("retrieval",{}),index_dir,logger,force=dense_force)
        logger.log(f"Dense indexes ready: units={list(dense_indexes.keys())}")
    if args.mode=="index": return {"dataset":dataset,"n":0,"status":"indexed","prompt_profile":prompt_profile,"index_dir":str(index_dir),**index_info,"index_meta":idx.get("meta",{})}
    retriever=QueryMedoidRetriever(idx,cfg.get("retrieval",{}),dense_indexes,logger); preds=[]; pko=out_dir/"예측결과.jsonl"; pen=out_dir/"predictions.jsonl"
    for p in [pko,pen]:
        if p.exists(): p.unlink()
    with logger.time_block("run.examples", dataset=dataset, n=len(qas)):
        with open(pko,"a",encoding="utf-8") as fko, open(pen,"a",encoding="utf-8") as fen:
            for qa in tqdm(qas, desc=dataset, ncols=100):
                try:
                    with logger.time_block("retrieve.one", dataset=dataset, qid=qa.id): ret=retriever.retrieve(qa.question, qa.metadata)
                    t=time.perf_counter()
                    with logger.time_block("generate.one", dataset=dataset, qid=qa.id): gen=generate_answer(qa.question, ret["evidence_bundles"], cfg.get("generation",{}))
                    row={"id":qa.id,"question":qa.question,"prediction":gen.get("prediction",""),"answers":qa.answers,"support_titles":qa.support_titles,"support_facts":qa.support_facts,"prompt_profile":gen.get("prompt_profile",prompt_profile),"evidence_bundles":ret["evidence_bundles"],"seeds":ret["seeds"],"retrieval_diagnostics":ret["diagnostics"],"generation_latency_s":round(time.perf_counter()-t,6),"llm_provider":gen.get("llm_provider"),"llm_model":gen.get("model"),"llm_usage":gen.get("usage")}
                except Exception as e:
                    logger.event({"event":"example.error","dataset":dataset,"qid":qa.id,"error":repr(e)})
                    if not args.continue_on_error: raise
                    row={"id":qa.id,"question":qa.question,"prediction":"","answers":qa.answers,"support_titles":qa.support_titles,"prompt_profile":prompt_profile,"error":repr(e),"evidence_bundles":[],"seeds":[],"retrieval_diagnostics":{"candidate_count":0,"seed_count":0,"bundle_count":0,"context_tokens":0,"timings":{}},"generation_latency_s":0.0,"llm_provider":cfg.get("generation",{}).get("provider")}
                preds.append(row); append_line(fko,row); append_line(fen,row)
    with logger.time_block("eval", dataset=dataset, n=len(preds)):
        res=evaluate_predictions(preds); res["prompt_profile"]=prompt_profile; res["index_dir"]=str(index_dir); res["index_source"]=index_info.get("index_source")
        if index_info.get("source_index_dir"): res["source_index_dir"]=index_info.get("source_index_dir")
        dump_json(res,out_dir/"eval.json"); (out_dir/"eval_summary.md").write_text(summary_markdown(dataset,res),encoding="utf-8")
    return {"dataset":dataset, **{k:v for k,v in res.items() if k!="per_example"}}

def main():
    args=parse_args(); cfg=apply_overrides(load_yaml(args.config), args); timestamp=args.timestamp or now_timestamp(); datasets=args.datasets or list(cfg.get("datasets",{}).keys()); rows=[]
    for ds in datasets: rows.append(run_dataset(ds,cfg,args,timestamp))
    if rows:
        table=[]
        for s in rows: table.append({"dataset":s.get("dataset"),"prompt":s.get("prompt_profile","-"),"n":s.get("n",0),"EM":f"{s.get('em',0):.4f}" if "em" in s else "-","F1":f"{s.get('f1',0):.4f}" if "f1" in s else "-","AnsContains":f"{s.get('answer_contains',0):.4f}" if "answer_contains" in s else "-","SupportRecall":f"{s.get('support_title_recall',0):.4f}" if "support_title_recall" in s else "-","SR/1kTok":f"{s.get('support_recall_per_1k_tokens',0):.4f}" if "support_recall_per_1k_tokens" in s else "-","CtxTok":f"{s.get('context_tokens',0):.1f}" if "context_tokens" in s else "-","LatencyMs":f"{s.get('latency_ms',0):.1f}" if "latency_ms" in s else "-","DenseRate":f"{s.get('dense_enabled_rate',0):.2f}" if "dense_enabled_rate" in s else "-"})
        print("\n"+tabulate(table, headers="keys", tablefmt="github"))
if __name__=="__main__": main()
