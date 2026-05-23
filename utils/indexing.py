from __future__ import annotations
import json, time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence
from tqdm import tqdm
from .data_loaders import CorpusDoc
from .embeddings import DenseMatrixIndex, build_embedder
from .io_utils import dump_json, ensure_dir, read_jsonl, write_jsonl
from .text import normalize_text, pack_sentences, sentence_split, token_count

class LightweightEPCIndexer:
    def __init__(self, cfg: Mapping[str, Any], logger: Optional[Any]=None):
        self.cfg=cfg; self.logger=logger
    def build(self, docs: Sequence[CorpusDoc]) -> Dict[str, Any]:
        t0=time.perf_counter(); max_chunk_tokens=int(self.cfg.get("max_chunk_tokens",240)); overlap=int(self.cfg.get("chunk_overlap_sentences",0)); min_prop=int(self.cfg.get("min_prop_tokens",2))
        chunks=[]; props=[]; ent_map={}
        if self.logger:
            self.logger.log(f"Indexing EPC docs: n={len(docs)}")
        it=tqdm(docs, desc="index.docs", ncols=100) if self.cfg.get("show_progress",True) else docs
        for d_i,doc in enumerate(it):
            title=normalize_text(doc.title or f"doc_{d_i}"); text=normalize_text(doc.text)
            if not text: continue
            ent=ent_map.setdefault(title.lower(), {"entity_id":f"e_{len(ent_map)}","title":title,"doc_ids":[],"chunk_ids":[],"proposition_ids":[]})
            ent["doc_ids"].append(str(doc.doc_id))
            sents=[s for s in sentence_split(text) if token_count(s)>=min_prop] or [text]
            cursor=0
            for chunk_text in pack_sentences(sents, max_chunk_tokens, overlap):
                cid=f"c_{len(chunks)}"; chunk={"chunk_id":cid,"doc_id":str(doc.doc_id),"title":title,"text":chunk_text,"token_count":token_count(chunk_text),"proposition_ids":[],"metadata":dict(doc.metadata or {})}
                chunks.append(chunk); ent["chunk_ids"].append(cid)
                while cursor < len(sents):
                    s=sents[cursor]
                    if s not in chunk_text and chunk["proposition_ids"]: break
                    pid=f"p_{len(props)}"; prop={"prop_id":pid,"chunk_id":cid,"doc_id":str(doc.doc_id),"title":title,"text":s,"token_count":token_count(s),"entities":[title],"metadata":dict(doc.metadata or {})}
                    props.append(prop); chunk["proposition_ids"].append(pid); ent["proposition_ids"].append(pid); cursor+=1
                    if cursor>=len(sents) or sents[cursor] not in chunk_text: break
        meta={"num_docs":len(docs),"num_chunks":len(chunks),"num_propositions":len(props),"num_entities":len(ent_map),"max_chunk_tokens":max_chunk_tokens,"chunk_overlap_sentences":overlap,"build_seconds":round(time.perf_counter()-t0,6),"graph_type":"EPC(title-entity, sentence-proposition, packed-source-chunk)"}
        if self.logger:
            self.logger.log(
                "Indexed EPC docs: "
                f"docs={meta['num_docs']} chunks={meta['num_chunks']} "
                f"propositions={meta['num_propositions']} entities={meta['num_entities']} "
                f"seconds={meta['build_seconds']}"
            )
        return {"chunks":chunks,"propositions":props,"entities":list(ent_map.values()),"meta":meta}
    @staticmethod
    def save(index: Mapping[str, Any], index_dir: str | Path) -> None:
        p=ensure_dir(index_dir); write_jsonl(index.get("chunks",[]), p/"chunks.jsonl"); write_jsonl(index.get("propositions",[]), p/"propositions.jsonl"); dump_json(index.get("entities",[]), p/"entities.json"); dump_json(index.get("meta",{}), p/"index_meta.json")
    @staticmethod
    def load(index_dir: str | Path) -> Dict[str, Any]:
        p=Path(index_dir); return {"chunks":read_jsonl(p/"chunks.jsonl"),"propositions":read_jsonl(p/"propositions.jsonl"),"entities":json.loads((p/"entities.json").read_text(encoding="utf-8")),"meta":json.loads((p/"index_meta.json").read_text(encoding="utf-8"))}

def maybe_build_dense_embeddings(index: Dict[str,Any], index_dir: str | Path, cfg: Mapping[str,Any], logger: Optional[Any]=None, force: bool=False) -> Dict[str,Any]:
    emb_cfg=cfg.get("embedding",{}) if cfg else {}
    p=ensure_dir(index_dir); prop_path=p/"prop_embeddings.npy"; chunk_path=p/"chunk_embeddings.npy"; meta_path=p/"embedding_meta.json"
    if not emb_cfg.get("enabled", True):
        index["meta"]={**index.get("meta",{}),"embedding_enabled":False}; dump_json(index["meta"], p/"index_meta.json"); return index
    if not force and prop_path.exists() and chunk_path.exists() and meta_path.exists():
        index["meta"]={**index.get("meta",{}),"embedding_enabled":True,"embedding_meta":json.loads(meta_path.read_text(encoding="utf-8"))}; return index
    t0=time.perf_counter(); embedder=build_embedder(emb_cfg); dtype=str(emb_cfg.get("save_dtype", emb_cfg.get("dtype","float16")))
    prop_texts=[f"{x.get('title','')}: {x.get('text','')}" for x in index.get("propositions",[])]
    chunk_texts=[f"{x.get('title','')}: {x.get('text','')}" for x in index.get("chunks",[])]
    if logger: logger.log(f"Encoding dense proposition embeddings n={len(prop_texts)} model={emb_cfg.get('model_path')}")
    prop_emb=embedder.encode_passages(prop_texts) if prop_texts else None
    if logger: logger.log(f"Encoding dense chunk embeddings n={len(chunk_texts)} model={emb_cfg.get('model_path')}")
    chunk_emb=embedder.encode_passages(chunk_texts) if chunk_texts else None
    if prop_emb is not None: DenseMatrixIndex.save(prop_path, prop_emb, dtype)
    if chunk_emb is not None: DenseMatrixIndex.save(chunk_path, chunk_emb, dtype)
    meta={"enabled":True,"provider":emb_cfg.get("provider"),"model_path":emb_cfg.get("model_path"),"num_prop_embeddings":len(prop_texts),"num_chunk_embeddings":len(chunk_texts),"dim":int(prop_emb.shape[1]) if prop_emb is not None and len(prop_emb) else 0,"save_dtype":dtype,"build_seconds":round(time.perf_counter()-t0,6)}
    dump_json(meta, meta_path); index["meta"]={**index.get("meta",{}),"embedding_enabled":True,"embedding_meta":meta}; dump_json(index["meta"], p/"index_meta.json"); return index
LightweightGraphIndexer=LightweightEPCIndexer
