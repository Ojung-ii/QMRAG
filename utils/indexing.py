from __future__ import annotations
import json, re, string, time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence
from tqdm import tqdm
from .data_loaders import CorpusDoc
from .embeddings import DenseMatrixIndex, build_embedder
from .io_utils import dump_json, ensure_dir, read_jsonl, write_jsonl
from .text import normalize_text, pack_sentences, sentence_split, token_count

BRIDGE_REQUIRED_FILES=(
    "title_aliases.json",
    "prop_to_mentioned_titles.json",
    "title_to_mentioning_props.json",
    "title_mentions.jsonl",
)

GENERIC_ALIASES={
    "film","song","album","town","county","city","president","queen","war","battle",
    "person","politician","actor","actress","writer","author","united states","new york",
}
_PUNCT_TABLE=str.maketrans({c:" " for c in string.punctuation})

def _alias_text(text: str, *, drop_parenthetical: bool=False, lowercase: bool=True) -> str:
    s=normalize_text(text)
    if drop_parenthetical:
        s=re.sub(r"\([^)]*\)", " ", s)
    if lowercase:
        s=s.lower()
    s=s.translate(_PUNCT_TABLE)
    return re.sub(r"\s+", " ", s).strip()

def normalize_alias(text: str) -> str:
    return _alias_text(text, drop_parenthetical=False, lowercase=True)

def mention_surface(text: str) -> str:
    return _alias_text(text, drop_parenthetical=False, lowercase=False)

def title_aliases(title: str) -> list[str]:
    raw=normalize_text(title)
    aliases={normalize_alias(raw)}
    aliases.add(_alias_text(raw, drop_parenthetical=True, lowercase=True))
    return [a for a in aliases if a]

def alias_token_len(alias: str) -> int:
    return len(alias.split())

def valid_alias(alias: str, min_alias_tokens: int) -> bool:
    if alias_token_len(alias)<min_alias_tokens:
        return False
    return alias not in GENERIC_ALIASES

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
        bridge=self._build_mention_bridge(props, list(ent_map.values()))
        meta={**meta, **bridge.get("meta",{})}
        if self.logger:
            self.logger.log(
                "Indexed EPC docs: "
                f"docs={meta['num_docs']} chunks={meta['num_chunks']} "
                f"propositions={meta['num_propositions']} entities={meta['num_entities']} "
                f"seconds={meta['build_seconds']}"
            )
        return {"chunks":chunks,"propositions":props,"entities":list(ent_map.values()),"meta":meta,**{k:v for k,v in bridge.items() if k!="meta"}}

    def _build_mention_bridge(self, props: Sequence[Mapping[str,Any]], entities: Sequence[Mapping[str,Any]]) -> Dict[str,Any]:
        cfg=dict(self.cfg.get("bridge",{}) or {})
        enabled=bool(cfg.get("enabled",True))
        t0=time.perf_counter()
        if self.logger:
            self.logger.event({"event":"bridge.index.start","enabled":enabled})
            self.logger.log(f"Building mention bridge index: enabled={enabled}")
        empty_meta={"mention_bridge_enabled":enabled,"num_title_aliases":0,"num_ambiguous_aliases":0,"num_title_mentions":0,"num_props_with_mentions":0,"mention_build_seconds":0.0}
        if not enabled:
            return {"title_aliases":{},"prop_to_mentioned_titles":{},"title_to_mentioning_props":{},"title_mentions":[],"meta":empty_meta}
        min_alias_tokens=int(cfg.get("min_alias_tokens",2))
        max_mentions_per_prop=int(cfg.get("max_mentions_per_prop",3))
        skip_ambiguous=bool(cfg.get("skip_ambiguous_aliases",True))
        remove_self=bool(cfg.get("remove_self_mentions",True))
        alias_to_titles: Dict[str,set[str]]={}
        for ent in entities:
            title=str(ent.get("title","")).strip()
            if not title:
                continue
            for alias in title_aliases(title):
                if valid_alias(alias,min_alias_tokens):
                    alias_to_titles.setdefault(alias,set()).add(title)
        title_aliases_json={a:sorted(titles) for a,titles in sorted(alias_to_titles.items())}
        ambiguous={a for a,titles in alias_to_titles.items() if len(titles)>1}
        aliases_by_len: Dict[int,dict[str,list[str]]]={}
        max_len=0
        for alias,titles in alias_to_titles.items():
            n=alias_token_len(alias)
            max_len=max(max_len,n)
            aliases_by_len.setdefault(n,{})[alias]=sorted(titles)
        max_len=min(max_len,8)
        title_mentions=[]; prop_to_titles: Dict[str,list[str]]={}; title_to_props: Dict[str,list[str]]={}
        show_progress=bool(self.cfg.get("show_progress",True))
        it=tqdm(props, desc="bridge.mentions", ncols=100) if show_progress else props
        for prop in it:
            prop_id=str(prop.get("prop_id","")); source_title=str(prop.get("title",""))
            toks=normalize_alias(str(prop.get("text",""))).split()
            surface_toks=mention_surface(str(prop.get("text",""))).split()
            if not prop_id or not toks:
                continue
            hits=[]
            limit_len=min(max_len, len(toks))
            for n in range(max(1,min_alias_tokens),limit_len+1):
                table=aliases_by_len.get(n)
                if not table:
                    continue
                for i in range(0,len(toks)-n+1):
                    mention=" ".join(toks[i:i+n])
                    titles=table.get(mention)
                    if not titles:
                        continue
                    is_ambiguous=mention in ambiguous
                    if skip_ambiguous and is_ambiguous:
                        continue
                    surface=" ".join(surface_toks[i:i+n]) if i+n<=len(surface_toks) else mention
                    for title in titles:
                        if remove_self and title==source_title:
                            continue
                        hits.append((mention,title,surface,is_ambiguous))
            if not hits:
                continue
            seen=set(); kept=[]
            for mention,title,surface,is_ambiguous in sorted(hits, key=lambda x:(x[3], -alias_token_len(x[0]), x[1])):
                key=(mention,title)
                if key in seen:
                    continue
                seen.add(key); kept.append((mention,title,surface,is_ambiguous))
                if len(kept)>=max_mentions_per_prop:
                    break
            for mention,title,surface,is_ambiguous in kept:
                title_mentions.append({"prop_id":prop_id,"source_title":source_title,"mentioned_title":title,"mention":surface,"mention_norm":mention,"ambiguous":bool(is_ambiguous)})
                prop_to_titles.setdefault(prop_id,[])
                if title not in prop_to_titles[prop_id]:
                    prop_to_titles[prop_id].append(title)
                title_to_props.setdefault(title,[])
                if prop_id not in title_to_props[title]:
                    title_to_props[title].append(prop_id)
        prop_ids={str(p.get("prop_id","")) for p in props}
        invalid_ids=sorted(pid for pid in prop_to_titles if pid not in prop_ids)
        if invalid_ids:
            raise ValueError(f"Bridge prop_id mismatch: first_invalid={invalid_ids[:5]}")
        meta={"mention_bridge_enabled":enabled,"num_title_aliases":len(title_aliases_json),"num_ambiguous_aliases":len(ambiguous),"num_title_mentions":len(title_mentions),"num_props_with_mentions":len(prop_to_titles),"mention_build_seconds":round(time.perf_counter()-t0,6)}
        if self.logger:
            self.logger.log(
                "Built mention bridge index: "
                f"aliases={meta['num_title_aliases']} ambiguous={meta['num_ambiguous_aliases']} "
                f"mentions={meta['num_title_mentions']} props_with_mentions={meta['num_props_with_mentions']} "
                f"seconds={meta['mention_build_seconds']}"
            )
            self.logger.event({"event":"bridge.index.end",**meta})
            self.logger.log(f"Validated mention bridge prop ids: props_with_mentions={len(prop_to_titles)} mismatches=0")
        return {"title_aliases":title_aliases_json,"prop_to_mentioned_titles":prop_to_titles,"title_to_mentioning_props":title_to_props,"title_mentions":title_mentions,"meta":meta}
    @staticmethod
    def save(index: Mapping[str, Any], index_dir: str | Path) -> None:
        p=ensure_dir(index_dir); meta={**dict(index.get("meta",{}) or {}),"index_dir":str(p)}
        write_jsonl(index.get("chunks",[]), p/"chunks.jsonl"); write_jsonl(index.get("propositions",[]), p/"propositions.jsonl"); dump_json(index.get("entities",[]), p/"entities.json"); dump_json(meta, p/"index_meta.json")
        _save_bridge_files(index, p)
    @staticmethod
    def load(index_dir: str | Path) -> Dict[str, Any]:
        p=Path(index_dir)
        meta=json.loads((p/"index_meta.json").read_text(encoding="utf-8"))
        meta["index_dir"]=str(p)
        out={"chunks":read_jsonl(p/"chunks.jsonl"),"propositions":read_jsonl(p/"propositions.jsonl"),"entities":json.loads((p/"entities.json").read_text(encoding="utf-8")),"meta":meta,"index_dir":str(p)}
        _load_bridge_files(out, p)
        return out

def bridge_files_ready(index_dir: str | Path) -> bool:
    p=Path(index_dir)
    return all((p/name).exists() for name in BRIDGE_REQUIRED_FILES)

def _save_bridge_files(index: Mapping[str, Any], index_dir: Path) -> None:
    dump_json(index.get("title_aliases",{}), index_dir/"title_aliases.json")
    dump_json(index.get("prop_to_mentioned_titles",{}), index_dir/"prop_to_mentioned_titles.json")
    dump_json(index.get("title_to_mentioning_props",{}), index_dir/"title_to_mentioning_props.json")
    write_jsonl(index.get("title_mentions",[]), index_dir/"title_mentions.jsonl")

def _load_bridge_files(index: Dict[str, Any], index_dir: Path) -> Dict[str, Any]:
    index["title_aliases"]=json.loads((index_dir/"title_aliases.json").read_text(encoding="utf-8")) if (index_dir/"title_aliases.json").exists() else {}
    index["prop_to_mentioned_titles"]=json.loads((index_dir/"prop_to_mentioned_titles.json").read_text(encoding="utf-8")) if (index_dir/"prop_to_mentioned_titles.json").exists() else {}
    index["title_to_mentioning_props"]=json.loads((index_dir/"title_to_mentioning_props.json").read_text(encoding="utf-8")) if (index_dir/"title_to_mentioning_props.json").exists() else {}
    index["title_mentions"]=read_jsonl(index_dir/"title_mentions.jsonl") if (index_dir/"title_mentions.jsonl").exists() else []
    return index

def _bridge_cfg(cfg: Mapping[str,Any]) -> Dict[str,Any]:
    if "retrieval" in cfg:
        return dict((cfg.get("retrieval",{}) or {}).get("bridge",{}) or {})
    return dict(cfg.get("bridge",{}) or {})

def _indexing_cfg(cfg: Mapping[str,Any]) -> Dict[str,Any]:
    if "indexing" in cfg:
        return dict(cfg.get("indexing",{}) or {})
    return dict(cfg or {})

def ensure_mention_bridge_index(index: Dict[str,Any], index_dir: str | Path, cfg: Mapping[str,Any], logger: Optional[Any]=None, force: bool=False) -> Dict[str,Any]:
    bridge_cfg=_bridge_cfg(cfg)
    enabled=bool(bridge_cfg.get("enabled",False))
    p=ensure_dir(index_dir)
    index["index_dir"]=str(p)
    index["meta"]={**dict(index.get("meta",{}) or {}),"index_dir":str(p)}
    if not enabled:
        return index
    if not force and bridge_files_ready(p):
        _load_bridge_files(index,p)
        if logger:
            logger.log(f"Bridge index present: {p}")
        return index
    if logger:
        missing=[name for name in BRIDGE_REQUIRED_FILES if not (p/name).exists()]
        logger.log(f"Bridge index missing or rebuild requested; building mention bridge: index_dir={p} missing={missing} force={force}")
    build_cfg={**_indexing_cfg(cfg),"bridge":bridge_cfg}
    bridge=LightweightEPCIndexer(build_cfg, logger)._build_mention_bridge(index.get("propositions",[]), index.get("entities",[]))
    index.update({k:v for k,v in bridge.items() if k!="meta"})
    index["meta"]={**dict(index.get("meta",{}) or {}), **dict(bridge.get("meta",{}) or {}), "index_dir":str(p)}
    dump_json(index["meta"], p/"index_meta.json")
    _save_bridge_files(index,p)
    return index

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
