from __future__ import annotations

import math
import json
import random
import re
import time
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .embedding import DenseIndex, encode_query
from .io_utils import read_jsonl
try:
    from .embeddings import DenseMatrixIndex, build_embedder
except Exception:  # pragma: no cover
    DenseMatrixIndex = None
    build_embedder = None
from .text import jaccard, token_count, tokenize


RESIDUAL_STOPWORDS={
    "a","an","the","is","are","was","were","be","been","being","am",
    "do","does","did","of","for","to","in","on","at","by","with","from",
    "and","or","as","that","this","these","those","it","its","his","her",
    "their","there","one","which","who","whom","whose","what","than","then",
    "also","into","about","between","among","under","over","after","before",
}

GENERIC_RELATION_TITLES={
    "place of birth","place of origin","date of birth","date of death",
    "occupation","country","county","city","prime minister","president",
    "film","song","album",
}

ANSWER_SLOT_TERMS={
    "when","date","year","time","where","place","location","birth","born",
    "birthplace","die","died","death","dead","prime","minister","known",
    "author","written","wrote","write",
}

TERM_VARIANTS={
    "die":{"die","died","death","dead","deaths","d"},
    "died":{"die","died","death","dead","deaths","d"},
    "death":{"die","died","death","dead","deaths","d"},
    "birth":{"birth","born","birthplace","birthplaces"},
    "born":{"birth","born","birthplace","birthplaces"},
    "birthplace":{"birth","born","birthplace","birthplaces"},
    "place":{"place","location","located","where","born_in"},
    "when":{"when","date","year","time","january","february","march","april","may","june","july","august","september","october","november","december"},
    "date":{"when","date","year","time","january","february","march","april","may","june","july","august","september","october","november","december"},
    "minister":{"minister","ministry"},
    "prime":{"prime"},
    "known":{"known","notable","famous","recognized","recognised"},
    "performer":{"performer","singer","rapper","artist","vocalist"},
}


def _norm_token(token: str) -> str:
    t=str(token or "").lower().strip()
    if t.endswith("'s"):
        t=t[:-2]
    return t


def _norm_tokens(text: Any) -> List[str]:
    raw=str(text or "")
    toks=[t for t in (_norm_token(x) for x in tokenize(raw)) if t]
    if re.search(r"\bborn\s+(?:in|at)\b", raw, flags=re.I):
        toks.append("born_in")
    return toks


def _term_variants(term: str) -> set[str]:
    t=_norm_token(term)
    return set(TERM_VARIANTS.get(t,{t}))


def _residual_groups(terms: Sequence[str]) -> List[set[str]]:
    groups=[]; seen=set()
    for term in terms:
        t=_norm_token(term)
        if not t or t in seen:
            continue
        seen.add(t); groups.append(_term_variants(t))
    return groups


def _coverage_count(terms: Sequence[str], text: Any) -> int:
    toks=set(_norm_tokens(text))
    return sum(1 for group in _residual_groups(terms) if toks & group)


def _lexical_bm25_score(terms: Sequence[str], text: Any, idf: Optional[Mapping[str,float]]=None, avgdl: float=20.0) -> float:
    toks=_norm_tokens(text)
    if not toks:
        return 0.0
    tf=Counter(toks); dl=len(toks); score=0.0; k1=1.2; b=0.75
    for group in _residual_groups(terms):
        freq=sum(tf.get(v,0) for v in group)
        if freq<=0:
            continue
        term_idf=max((float(idf.get(v,1.0)) for v in group), default=1.0) if idf else 1.0
        denom=freq+k1*(1-b+b*dl/max(1e-9,avgdl))
        score+=term_idf*(freq*(k1+1))/max(1e-9,denom)
    return float(score)


def _bridge_rank_text(prop: Mapping[str,Any], prev_prop: Optional[Mapping[str,Any]]) -> str:
    text=str(prop.get("text",""))
    if not prev_prop:
        return text
    if str(prev_prop.get("chunk_id",""))!=str(prop.get("chunk_id","")):
        return text
    prev_text=str(prev_prop.get("text",""))
    if re.search(r"\((?:d|b|c)\.\s*$", prev_text, flags=re.I):
        return f"{prev_text} {text}".strip()
    return text


def build_residual_query(question: str, anchor_title: str, bridge_title: str, seed_prop_text: str) -> List[str]:
    q_terms=_norm_tokens(question)
    seed_toks=set(_norm_tokens(seed_prop_text))
    remove=set(_norm_tokens(anchor_title)) | set(_norm_tokens(bridge_title))
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9'_-]+(?:\s+[A-Z][A-Za-z0-9'_-]+)+", str(seed_prop_text or "")):
        phrase=" ".join(_norm_tokens(match.group(0)))
        if phrase and phrase not in GENERIC_RELATION_TITLES:
            remove.update(phrase.split())
    terms=[]; seen=set()
    for term in q_terms:
        if term in RESIDUAL_STOPWORDS or term in remove or len(term)<=1:
            continue
        if term not in ANSWER_SLOT_TERMS and seed_toks & _term_variants(term):
            continue
        if term not in seen:
            seen.add(term); terms.append(term)
    if not terms:
        terms=[t for t in q_terms if t not in RESIDUAL_STOPWORDS] or q_terms
    return terms


class BM25Index:
    def __init__(self, docs: Sequence[Mapping[str, Any]], text_key: str = "text", k1: float = 1.5, b: float = 0.75):
        self.docs = list(docs)
        self.text_key = text_key
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(str(d.get(text_key, "")) + " " + str(d.get("title", ""))) for d in self.docs]
        self.doc_lens = [len(x) for x in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(1, len(self.doc_lens))
        self.tfs = [Counter(toks) for toks in self.doc_tokens]
        df: Counter[str] = Counter()
        for toks in self.doc_tokens:
            df.update(set(toks))
        n_docs = len(self.docs)
        self.idf = {t: math.log(1.0 + (n_docs - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def topk(self, query: str, k: int) -> List[Tuple[int, float]]:
        q = tokenize(query)
        if not q:
            return []
        pairs: List[Tuple[int, float]] = []
        for i, (tf, dl) in enumerate(zip(self.tfs, self.doc_lens)):
            s = 0.0
            for t in q:
                if t not in tf:
                    continue
                freq = tf[t]
                denom = freq + self.k1 * (1 - self.b + self.b * dl / max(1e-9, self.avgdl))
                s += self.idf.get(t, 0.0) * (freq * (self.k1 + 1)) / max(1e-9, denom)
            if s > 0:
                pairs.append((i, float(s)))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:k]


def _rank_norm(keys: Sequence[Any]) -> Dict[Any, float]:
    return {key: 1.0 / (rank + 1) for rank, key in enumerate(keys)}


@dataclass
class RetrievalDiagnostics:
    candidate_count: int
    seed_count: int
    bundle_count: int
    context_tokens: int
    timings: Dict[str, float]
    unit_counts: Dict[str, int]
    dense_enabled: bool


class QueryMedoidRetriever:
    def __init__(self, index: Mapping[str, Any], cfg: Mapping[str, Any], dense_indexes: Optional[Mapping[str, DenseIndex]] = None, logger: Optional[Any] = None):
        self.index = index
        self.cfg = cfg
        self.logger = logger
        self.dense_indexes = dict(dense_indexes or {}) if isinstance(dense_indexes, Mapping) else {}
        self.index_dir = Path(dense_indexes) if dense_indexes is not None and not isinstance(dense_indexes, Mapping) else None
        self._dense_prop_matrix = None
        self._dense_chunk_matrix = None
        self._query_embedder = None
        if self.index_dir is not None and bool(cfg.get("dense", {}).get("enabled", False)) and DenseMatrixIndex is not None:
            if (self.index_dir / "prop_embeddings.npy").exists():
                self._dense_prop_matrix = DenseMatrixIndex.from_npy(self.index_dir / "prop_embeddings.npy")
            if (self.index_dir / "chunk_embeddings.npy").exists():
                self._dense_chunk_matrix = DenseMatrixIndex.from_npy(self.index_dir / "chunk_embeddings.npy")
            if (self._dense_prop_matrix is not None or self._dense_chunk_matrix is not None) and build_embedder is not None:
                self._query_embedder = build_embedder(cfg.get("embedding", {}))
        self.chunks = list(index.get("chunks", []))
        self.props = list(index.get("propositions", []))
        self.entities = list(index.get("entities", []))
        self.entity_by_title = {str(e["title"]).lower(): e for e in self.entities}
        self.chunk_by_id = {c["chunk_id"]: c for c in self.chunks}
        self.prop_by_id = {p["prop_id"]: p for p in self.props}
        self.props_by_title: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        self.chunks_by_title: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
        for p in self.props:
            self.props_by_title[str(p.get("title", ""))].append(p)
        for c in self.chunks:
            self.chunks_by_title[str(c.get("title", ""))].append(c)
        bm25_cfg = cfg.get("bm25", {})
        self.prop_bm25 = BM25Index(self.props, k1=float(bm25_cfg.get("k1", 1.5)), b=float(bm25_cfg.get("b", 0.75)))
        self.chunk_bm25 = BM25Index(self.chunks, k1=float(bm25_cfg.get("k1", 1.5)), b=float(bm25_cfg.get("b", 0.75)))
        self.rng = random.Random(int(cfg.get("random_seed", 13)))
        self._embed_cache: Dict[str, Any] = {}
        self.bridge_cfg=dict(cfg.get("bridge",{}) or {})
        raw_index_dir=index.get("index_dir") or (index.get("meta",{}) or {}).get("index_dir")
        self.bridge_index_dir=Path(raw_index_dir) if raw_index_dir else None
        self.bridge_requested=bool(self.bridge_cfg.get("enabled",False))
        self.bridge_index_loaded=False
        self.bridge_index_warning: Optional[str]=None
        self.prop_to_mentioned_titles: Dict[str,List[str]]={}
        self.title_to_mentioning_props: Dict[str,List[str]]={}
        self.title_mentions: List[Mapping[str,Any]]=[]
        self._load_bridge_index(index)
        self.title_mentions_by_prop: Dict[str,List[Mapping[str,Any]]] = defaultdict(list)
        for row in self.title_mentions:
            if row.get("prop_id"):
                self.title_mentions_by_prop[str(row["prop_id"])].append(row)
        self.bridge_enabled=self.bridge_requested and self.bridge_index_loaded and bool(self.prop_to_mentioned_titles)
        if self.bridge_requested and not self.bridge_enabled:
            self._warn_bridge(self.bridge_index_warning or "Bridge retrieval disabled because the mention bridge index is empty")

    def retrieve(self, question: str, metadata: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        timings: Dict[str, float] = {}
        t_all = time.perf_counter()
        metadata = metadata or {}
        t0 = time.perf_counter(); candidates = self._candidate_retrieval(question, metadata); timings["candidate_retrieval_s"] = time.perf_counter() - t0
        t0 = time.perf_counter(); seeds = self._sampled_medoid_seeding(candidates); timings["medoid_seeding_s"] = time.perf_counter() - t0
        t0 = time.perf_counter(); bundles = self._local_refinement(question, seeds, candidates); timings["local_refinement_s"] = time.perf_counter() - t0
        t0 = time.perf_counter(); bundles, context_tokens = self._budget_context(bundles); timings["budgeting_s"] = time.perf_counter() - t0
        timings["total_retrieval_s"] = time.perf_counter() - t_all
        diag = RetrievalDiagnostics(
            candidate_count=len(candidates),
            seed_count=len(seeds),
            bundle_count=len(bundles),
            context_tokens=context_tokens,
            timings={k: round(v, 6) for k, v in timings.items()},
            unit_counts={
                "propositions": sum(1 for c in candidates if c["unit"] == "proposition"),
                "chunks": sum(1 for c in candidates if c["unit"] == "chunk"),
                "entities": sum(1 for c in candidates if c["unit"] == "entity"),
            },
            dense_enabled=bool(self.dense_indexes) or self._dense_prop_matrix is not None or self._dense_chunk_matrix is not None,
        )
        bridge_titles={t for b in bundles for t in b.get("bridge_titles",[]) or []}
        bridge_connected_count=sum(1 for b in bundles if b.get("bridge_connected"))
        answer_slot_aligned_count=sum(1 for b in bundles if b.get("answer_slot_aligned"))
        chain_complete_v2_count=sum(1 for b in bundles if b.get("chain_complete_v2"))
        residual_counts=[float(b.get("residual_coverage_count",0.0) or 0.0) for b in bundles]
        diag_dict=diag.__dict__
        diag_dict.update({
            "bridge_enabled":self.bridge_enabled,
            "bridge_index_loaded":self.bridge_index_loaded,
            "ordering_mode":self.bridge_cfg.get("ordering","chain_aware") if self.bridge_enabled else "score",
            "bridge_title_count":len(bridge_titles),
            "bridge_bundle_count":sum(1 for b in bundles if b.get("has_bridge")),
            "chain_complete_count":sum(1 for b in bundles if b.get("chain_complete")),
            "has_chain_complete":any(bool(b.get("chain_complete")) for b in bundles),
            "bridge_connected_count":bridge_connected_count,
            "answer_slot_aligned_count":answer_slot_aligned_count,
            "chain_complete_v2_count":chain_complete_v2_count,
            "chain_complete_v2_rate":chain_complete_v2_count/max(1,len(bundles)),
            "avg_residual_coverage_count":sum(residual_counts)/max(1,len(residual_counts)),
            "has_bridge_connected":bridge_connected_count>0,
            "has_answer_slot_aligned":answer_slot_aligned_count>0,
            "has_chain_complete_v2":chain_complete_v2_count>0,
        })
        return {"question": question, "candidates": candidates, "seeds": seeds, "evidence_bundles": bundles, "diagnostics": diag.__dict__}

    def _warn_bridge(self, message: str) -> None:
        self.bridge_index_warning=message
        if self.logger:
            self.logger.log(f"WARNING: {message}")
            if hasattr(self.logger, "event"):
                self.logger.event({"event":"bridge.index.warning","message":message})

    def _load_bridge_index(self, index: Mapping[str, Any]) -> None:
        self.prop_to_mentioned_titles=dict(index.get("prop_to_mentioned_titles",{}) or {})
        self.title_to_mentioning_props=dict(index.get("title_to_mentioning_props",{}) or {})
        self.title_mentions=list(index.get("title_mentions",[]) or [])
        in_memory_loaded=bool(self.prop_to_mentioned_titles or self.title_to_mentioning_props or self.title_mentions)
        if self.bridge_index_dir is not None:
            paths={
                "prop_to_mentioned_titles": self.bridge_index_dir/"prop_to_mentioned_titles.json",
                "title_to_mentioning_props": self.bridge_index_dir/"title_to_mentioning_props.json",
                "title_mentions": self.bridge_index_dir/"title_mentions.jsonl",
            }
            missing=[str(path) for path in paths.values() if not path.exists()]
            if not missing:
                self.prop_to_mentioned_titles=json.loads(paths["prop_to_mentioned_titles"].read_text(encoding="utf-8"))
                self.title_to_mentioning_props=json.loads(paths["title_to_mentioning_props"].read_text(encoding="utf-8"))
                self.title_mentions=read_jsonl(paths["title_mentions"])
                self.bridge_index_loaded=True
            elif self.bridge_requested:
                self.bridge_index_loaded=in_memory_loaded
                self.bridge_index_warning=f"Bridge index files missing: {missing}"
        else:
            self.bridge_index_loaded=in_memory_loaded
            if self.bridge_requested and not in_memory_loaded:
                self.bridge_index_warning="Bridge index directory is unknown and no in-memory bridge index was provided"
        if self.bridge_index_loaded:
            prop_ids=set(self.prop_by_id)
            invalid=[pid for pid in self.prop_to_mentioned_titles if pid not in prop_ids]
            if invalid:
                self.bridge_index_loaded=False
                self.bridge_index_warning=f"Bridge prop_id mismatch: first_invalid={invalid[:5]}"

    def _add_candidate(self, merged: Dict[Tuple[str, str], Dict[str, Any]], cand: Dict[str, Any], component: str, score: float) -> None:
        key = (cand["unit"], cand["id"])
        if key not in merged:
            cand["score_components"] = {}
            merged[key] = cand
        else:
            merged[key].setdefault("raw_scores", {}).update(cand.get("raw_scores", {}))
        merged[key]["score_components"][component] = max(float(score), merged[key]["score_components"].get(component, float("-inf")))

    def _candidate_retrieval(self, question: str, metadata: Mapping[str, Any]) -> List[Dict[str, Any]]:
        top_k = int(self.cfg.get("candidate_top_k", 80))
        bm25_top_k = int(self.cfg.get("bm25", {}).get("top_k", top_k))
        dense_top_k = int(self.cfg.get("dense", {}).get("top_k", top_k))
        search_units = set(self.cfg.get("search_units", ["proposition", "chunk", "entity"]))
        weights = self.cfg.get("weights", {})
        w_bm25, w_dense, w_entity = float(weights.get("bm25", 0.35)), float(weights.get("dense", 0.55)), float(weights.get("entity", 0.10))
        merged: Dict[Tuple[str, str], Dict[str, Any]] = {}

        if "proposition" in search_units:
            prop_pairs = self.prop_bm25.topk(question, bm25_top_k)
            norm = _rank_norm([("proposition", self.props[i]["prop_id"]) for i, _ in prop_pairs])
            for idx, raw in prop_pairs:
                p = self.props[idx]
                self._add_candidate(merged, {"unit":"proposition","id":p["prop_id"],"title":p["title"],"text":p["text"],"chunk_id":p["chunk_id"],"tokens":p.get("token_count", token_count(p.get("text", ""))),"raw_scores":{"bm25":raw}}, "bm25", norm[("proposition", p["prop_id"])])
        if "chunk" in search_units:
            chunk_pairs = self.chunk_bm25.topk(question, max(1, bm25_top_k // 2))
            norm = _rank_norm([("chunk", self.chunks[i]["chunk_id"]) for i, _ in chunk_pairs])
            for idx, raw in chunk_pairs:
                c = self.chunks[idx]
                self._add_candidate(merged, {"unit":"chunk","id":c["chunk_id"],"title":c["title"],"text":c["text"],"chunk_id":c["chunk_id"],"tokens":c.get("token_count", token_count(c.get("text", ""))),"raw_scores":{"bm25":raw}}, "bm25", norm[("chunk", c["chunk_id"])])

        if self._dense_prop_matrix is not None or self._dense_chunk_matrix is not None:
            qv = self._query_embedder.encode_queries([question])[0] if self._query_embedder is not None else None
            if qv is not None and "proposition" in search_units and self._dense_prop_matrix is not None:
                pairs = self._dense_prop_matrix.search(qv, dense_top_k)
                norm = _rank_norm([("proposition", self.props[idx]["prop_id"]) for idx, _ in pairs])
                for idx, raw in pairs:
                    p = self.props[int(idx)]
                    self._add_candidate(merged, {"unit":"proposition","id":p["prop_id"],"title":p["title"],"text":p["text"],"chunk_id":p["chunk_id"],"tokens":p.get("token_count", token_count(p.get("text", ""))),"raw_scores":{"dense":raw}}, "dense", norm[("proposition", p["prop_id"])])
            if qv is not None and "chunk" in search_units and self._dense_chunk_matrix is not None:
                pairs = self._dense_chunk_matrix.search(qv, max(1, dense_top_k // 2))
                norm = _rank_norm([("chunk", self.chunks[idx]["chunk_id"]) for idx, _ in pairs])
                for idx, raw in pairs:
                    c = self.chunks[int(idx)]
                    self._add_candidate(merged, {"unit":"chunk","id":c["chunk_id"],"title":c["title"],"text":c["text"],"chunk_id":c["chunk_id"],"tokens":c.get("token_count", token_count(c.get("text", ""))),"raw_scores":{"dense":raw}}, "dense", norm[("chunk", c["chunk_id"])])
        elif self.dense_indexes:
            qvec = encode_query({"embedding": self.cfg.get("embedding", {})}, question, logger=self.logger, cache=self._embed_cache)
            if "proposition" in search_units and "proposition" in self.dense_indexes:
                pairs = self.dense_indexes["proposition"].search(qvec, dense_top_k)
                norm = _rank_norm([("proposition", pid) for pid, _ in pairs])
                for pid, raw in pairs:
                    p = self.prop_by_id.get(pid)
                    if p:
                        self._add_candidate(merged, {"unit":"proposition","id":pid,"title":p["title"],"text":p["text"],"chunk_id":p["chunk_id"],"tokens":p.get("token_count", token_count(p.get("text", ""))),"raw_scores":{"dense":raw}}, "dense", norm[("proposition", pid)])
            if "chunk" in search_units and "chunk" in self.dense_indexes:
                pairs = self.dense_indexes["chunk"].search(qvec, max(1, dense_top_k // 2))
                norm = _rank_norm([("chunk", cid) for cid, _ in pairs])
                for cid, raw in pairs:
                    c = self.chunk_by_id.get(cid)
                    if c:
                        self._add_candidate(merged, {"unit":"chunk","id":cid,"title":c["title"],"text":c["text"],"chunk_id":cid,"tokens":c.get("token_count", token_count(c.get("text", ""))),"raw_scores":{"dense":raw}}, "dense", norm[("chunk", cid)])

        if "entity" in search_units:
            anchors: List[str] = []
            for key in ("subj", "s_wiki_title", "title", "entity"):
                if metadata.get(key): anchors.append(str(metadata[key]))
            if bool(self.cfg.get("scan_titles_in_question", False)):
                q_l = question.lower(); max_scan = int(self.cfg.get("title_scan_limit", 50000))
                for i, (title_l, e) in enumerate(self.entity_by_title.items()):
                    if i >= max_scan: break
                    if title_l and title_l in q_l: anchors.append(str(e["title"]))
            for a in list(dict.fromkeys(anchors)):
                e = self.entity_by_title.get(a.lower())
                if not e: continue
                self._add_candidate(merged, {"unit":"entity","id":e["entity_id"],"title":e["title"],"text":e["title"],"chunk_id":None,"tokens":1,"raw_scores":{"entity":1.0}}, "entity", 1.0)
                for pid in e.get("proposition_ids", [])[: int(self.cfg.get("entity_anchor_prop_limit", 16))]:
                    p = self.prop_by_id.get(pid)
                    if p:
                        self._add_candidate(merged, {"unit":"proposition","id":pid,"title":p["title"],"text":p["text"],"chunk_id":p["chunk_id"],"tokens":p.get("token_count", token_count(p.get("text", ""))),"raw_scores":{"entity":1.0}}, "entity", 1.0)

        candidates = []
        for c in merged.values():
            comps = c.get("score_components", {})
            c["score"] = w_bm25 * comps.get("bm25", 0.0) + w_dense * comps.get("dense", 0.0) + w_entity * comps.get("entity", 0.0)
            candidates.append(c)
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[: max(top_k, int(self.cfg.get("min_candidate_pool", top_k)))]

    def _sampled_medoid_seeding(self, candidates: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        if not candidates: return []
        seed_count = min(int(self.cfg.get("seed_count", 4)), len(candidates))
        trials = int(self.cfg.get("seed_trials", 7)); sample_multiplier = int(self.cfg.get("sample_multiplier", 6))
        sample_size = min(len(candidates), max(seed_count, seed_count * sample_multiplier))
        top_pool = list(candidates[: int(self.cfg.get("seed_pool_top_k", len(candidates)))])
        weights = [max(1e-9, float(c.get("score", 0.0))) for c in top_pool]
        best: List[Dict[str, Any]] = []; best_obj = -1e18
        for _ in range(max(1, trials)):
            sample = self._weighted_sample_without_replacement(top_pool, weights, sample_size)
            seeds = self._greedy_mmr(sample, seed_count)
            obj = self._facility_objective(candidates, seeds)
            if obj > best_obj:
                best_obj = obj; best = [dict(s) for s in seeds]
        for rank, s in enumerate(best):
            s["seed_rank"] = rank + 1; s["seed_objective"] = round(best_obj, 6)
        return best

    def _weighted_sample_without_replacement(self, items: Sequence[Mapping[str, Any]], weights: Sequence[float], k: int) -> List[Mapping[str, Any]]:
        pool = list(zip(items, weights)); out: List[Mapping[str, Any]] = []
        for _ in range(min(k, len(pool))):
            total = sum(max(0.0, w) for _, w in pool)
            if total <= 0: idx = self.rng.randrange(len(pool))
            else:
                r = self.rng.random() * total; acc = 0.0; idx = 0
                for i, (_, w) in enumerate(pool):
                    acc += max(0.0, w)
                    if acc >= r: idx = i; break
            item, _ = pool.pop(idx); out.append(item)
        return out

    def _greedy_mmr(self, sample: Sequence[Mapping[str, Any]], seed_count: int) -> List[Mapping[str, Any]]:
        selected: List[Mapping[str, Any]] = []; remaining = list(sample)
        rel_scale = max((float(x.get("score", 0.0)) for x in sample), default=1.0) or 1.0
        diversity_weight = float(self.cfg.get("diversity_weight", 0.35))
        while remaining and len(selected) < seed_count:
            best_idx = 0; best_score = -1e18
            for i, cand in enumerate(remaining):
                rel = float(cand.get("score", 0.0)) / rel_scale
                div = 1.0 if not selected else 1.0 - max(jaccard(tokenize(cand.get("text", "")), tokenize(s.get("text", ""))) for s in selected)
                cost = float(cand.get("tokens", 0)) / max(1, int(self.cfg.get("context_token_budget", 3000)))
                score = rel + diversity_weight * div - cost
                if score > best_score: best_idx, best_score = i, score
            selected.append(remaining.pop(best_idx))
        return selected

    def _facility_objective(self, candidates: Sequence[Mapping[str, Any]], seeds: Sequence[Mapping[str, Any]]) -> float:
        if not seeds: return -1e18
        obj = 0.0
        for c in candidates:
            ctoks = tokenize(c.get("text", ""))
            max_sim = max(jaccard(ctoks, tokenize(s.get("text", ""))) for s in seeds)
            obj += float(c.get("score", 0.0)) * max_sim
        obj += sum(float(s.get("score", 0.0)) for s in seeds)
        obj -= sum(float(s.get("tokens", 0)) for s in seeds) / max(1, int(self.cfg.get("context_token_budget", 3000)))
        return obj

    def _local_refinement(self, question: str, seeds: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        if not seeds: return []
        refine_cfg = self.cfg.get("refine", {})
        per_entity_props = int(refine_cfg.get("per_entity_propositions", 4)); per_entity_chunks = int(refine_cfg.get("per_entity_chunks", 2)); max_bundles = int(refine_cfg.get("max_bundles", 6))
        q_toks = tokenize(question); bundles: List[Dict[str, Any]] = []; seen_titles = set()
        candidate_scores={(str(c.get("unit")),str(c.get("id"))):float(c.get("score",0.0)) for c in candidates}
        for s in seeds:
            title = str(s.get("title", ""))
            if not title or title in seen_titles: continue
            seen_titles.add(title)
            local_props = list(self.props_by_title.get(title, []))
            local_props.sort(key=lambda p: jaccard(q_toks, tokenize(p.get("text", ""))) + 0.05 * float(s.get("score", 0.0)), reverse=True)
            selected_props = local_props[:per_entity_props]
            selected_chunks: List[Mapping[str, Any]] = []; chunk_ids = set()
            for p in selected_props:
                cid = p.get("chunk_id")
                if cid and cid not in chunk_ids and cid in self.chunk_by_id:
                    selected_chunks.append(self.chunk_by_id[cid]); chunk_ids.add(cid)
            if len(selected_chunks) < per_entity_chunks:
                extra = list(self.chunks_by_title.get(title, [])); extra.sort(key=lambda c: jaccard(q_toks, tokenize(c.get("text", ""))), reverse=True)
                for c in extra:
                    if c.get("chunk_id") not in chunk_ids:
                        selected_chunks.append(c); chunk_ids.add(c.get("chunk_id"))
                    if len(selected_chunks) >= per_entity_chunks: break
            path = [{"type":"query","id":"q","text":question},{"type":"entity","id":title,"text":title}] + [{"type":"proposition","id":p["prop_id"],"text":p["text"]} for p in selected_props]
            score = float(s.get("score", 0.0)) + sum(jaccard(q_toks, tokenize(p.get("text", ""))) for p in selected_props)
            bundle={"bundle_id":f"b{len(bundles)}","anchor_title":title,"seed":dict(s),"propositions":[dict(p) for p in selected_props],"source_chunks":[dict(c) for c in selected_chunks],"evidence_path":path,"score":round(score,6),"diagnostics":{"num_props":len(selected_props),"num_chunks":len(selected_chunks),"refinement":"same-title entity hub expansion"}}
            self._apply_bridge_expansion(bundle, question, q_toks, selected_props, selected_chunks, candidate_scores)
            bundles.append(bundle)
        for c in candidates:
            if len(bundles) >= max_bundles: break
            title = c.get("title", "")
            if not title or title in seen_titles: continue
            seen_titles.add(title)
            chunk = self.chunk_by_id.get(c.get("chunk_id")) if c.get("chunk_id") else None
            prop = self.prop_by_id.get(c.get("id")) if c.get("unit") == "proposition" else None
            if prop:
                bprops=[dict(prop)]
            elif chunk and self._exact_anchor_match(question,str(title)) and not self._is_generic_relation_title(question,str(title)):
                chunk_props=[
                    self.prop_by_id[pid]
                    for pid in chunk.get("proposition_ids",[]) or []
                    if pid in self.prop_by_id
                ]
                chunk_props=[p for p in chunk_props if self.prop_to_mentioned_titles.get(str(p.get("prop_id",""))) or self.title_mentions_by_prop.get(str(p.get("prop_id","")))]
                chunk_props.sort(key=lambda p: jaccard(q_toks, tokenize(p.get("text",""))), reverse=True)
                bprops=[dict(p) for p in chunk_props[:max(1,min(2,per_entity_props))]]
            else:
                bprops=[]
            bchunks=[dict(chunk)] if chunk else ([dict(c)] if c.get("unit") == "chunk" else [])
            bundle={"bundle_id":f"b{len(bundles)}","anchor_title":title,"seed":dict(c),"propositions":bprops,"source_chunks":bchunks,"evidence_path":[{"type":"query","id":"q","text":question},{"type":"candidate","id":c.get("id"),"text":c.get("text","")}],"score":round(float(c.get("score",0.0)),6),"diagnostics":{"refinement":"direct-candidate bridge"}}
            self._apply_bridge_expansion(bundle, question, q_toks, bprops, bchunks, candidate_scores)
            bundles.append(bundle)
        bundles=self._order_bundles(question,bundles)
        return bundles[:max_bundles]

    def _mention_rows_for_prop(self, prop_id: str, source_title: str) -> List[Mapping[str,Any]]:
        if not self.bridge_enabled:
            return []
        max_mentions=int(self.bridge_cfg.get("max_mentions_per_prop",3))
        skip_ambiguous=bool(self.bridge_cfg.get("skip_ambiguous_aliases",True))
        remove_self=bool(self.bridge_cfg.get("remove_self_mentions",True))
        rows=[]
        raw_rows=list(self.title_mentions_by_prop.get(str(prop_id),[]))
        if not raw_rows:
            raw_rows=[
                {"prop_id":str(prop_id),"source_title":source_title,"mentioned_title":title,"mention":title,"mention_norm":str(title).lower(),"ambiguous":False}
                for title in self.prop_to_mentioned_titles.get(str(prop_id),[])
            ]
        for row in raw_rows[:max_mentions*4]:
            if skip_ambiguous and row.get("ambiguous"):
                continue
            title=str(row.get("mentioned_title",""))
            if remove_self and title==source_title:
                continue
            if not title:
                continue
            rows.append(row)
            if len(rows)>=max_mentions:
                break
        return rows

    def _bridge_source_props(self, base_props: Sequence[Mapping[str,Any]], anchor_title: str) -> List[Mapping[str,Any]]:
        out=[]; seen=set()
        for prop in base_props:
            pid=str(prop.get("prop_id",""))
            if pid:
                seen.add(pid); out.append(prop)
        max_scan=int(self.bridge_cfg.get("max_bridge_source_props_per_seed",16))
        for prop in self.props_by_title.get(anchor_title,[])[:max_scan]:
            pid=str(prop.get("prop_id",""))
            if pid and pid not in seen and self.prop_to_mentioned_titles.get(pid):
                seen.add(pid); out.append(prop)
        return out

    def _bridge_prop_score(self, prop: Mapping[str,Any], q_toks: Sequence[str], candidate_scores: Mapping[Tuple[str,str],float]) -> float:
        pid=str(prop.get("prop_id",""))
        return candidate_scores.get(("proposition",pid), jaccard(q_toks, tokenize(prop.get("text",""))))

    def _candidate_bridge_idf(self, candidates: Sequence[Mapping[str,Any]]) -> Dict[str,float]:
        df: Counter[str]=Counter()
        for prop in candidates:
            df.update(set(_norm_tokens(prop.get("text",""))))
        n=max(1,len(candidates))
        return {term:math.log(1.0+(n-f+0.5)/(f+0.5)) for term,f in df.items()}

    def _rank_bridge_props(self, candidates: Sequence[Mapping[str,Any]], residual_terms: Sequence[str], q_toks: Sequence[str]) -> List[Tuple[Mapping[str,Any],Dict[str,Any]]]:
        if not candidates:
            return []
        idf=self._candidate_bridge_idf(candidates)
        avgdl=sum(token_count(p.get("text","")) for p in candidates)/max(1,len(candidates))
        ranked=[]
        for idx,prop in enumerate(candidates):
            text=prop.get("text","")
            rank_text=_bridge_rank_text(prop, candidates[idx-1] if idx>0 else None)
            coverage=_coverage_count(residual_terms,rank_text)
            residual_score=_lexical_bm25_score(residual_terms,rank_text,idf,avgdl)
            original_score=jaccard(q_toks,_norm_tokens(rank_text))
            tok_count=int(prop.get("token_count",token_count(text)) or 0)
            rank_key=(coverage>0,coverage,residual_score,original_score,-tok_count)
            info={
                "residual_coverage_count":coverage,
                "residual_score":round(float(residual_score),6),
                "original_relevance_score":round(float(original_score),6),
                "token_count":tok_count,
                "rank_key":rank_key,
            }
            if rank_text!=text:
                info["display_text"]=rank_text
            ranked.append((prop,info))
        ranked.sort(key=lambda x:x[1]["rank_key"], reverse=True)
        return ranked

    def _is_generic_relation_title(self, question: str, title: str) -> bool:
        if not bool(self.bridge_cfg.get("demote_generic_relation_titles",True)):
            return False
        norm_title=" ".join(_norm_tokens(title))
        if not norm_title:
            return False
        if norm_title in GENERIC_RELATION_TITLES:
            return True
        q_norm=" ".join(_norm_tokens(question))
        return norm_title in q_norm and any(phrase in norm_title for phrase in GENERIC_RELATION_TITLES)

    def _finalize_bundle(self, question: str, bundle: Mapping[str,Any]) -> Dict[str,Any]:
        b=dict(bundle)
        is_generic=self._is_generic_relation_title(question,str(b.get("anchor_title","")))
        exact=self._exact_anchor_match(question,str(b.get("anchor_title","")))
        bridge_connected=bool(b.get("bridge_connected",b.get("has_bridge",False)))
        aligned=bool(b.get("answer_slot_aligned",float(b.get("residual_coverage_count",0.0) or 0.0)>0.0))
        chain_v2=bridge_connected and aligned
        b["is_generic_relation_title"]=is_generic
        b["exact_anchor_match"]=exact
        b["bridge_connected"]=bridge_connected
        b["answer_slot_aligned"]=aligned
        b["chain_complete_v2"]=chain_v2
        b["old_chain_complete"]=bool(b.get("old_chain_complete",b.get("chain_complete",False)))
        b["chain_complete"]=chain_v2
        if is_generic:
            b["ordering_group"]="generic_relation"
        elif chain_v2:
            b["ordering_group"]="chain_complete_v2"
        elif bridge_connected:
            b["ordering_group"]="bridge_connected"
        elif exact:
            b["ordering_group"]="anchor"
        elif b.get("propositions"):
            b["ordering_group"]="same_title"
        else:
            b["ordering_group"]="fallback"
        return b

    def _apply_bridge_expansion(self, bundle: Dict[str,Any], question: str, q_toks: Sequence[str], base_props: Sequence[Mapping[str,Any]], base_chunks: Sequence[Mapping[str,Any]], candidate_scores: Mapping[Tuple[str,str],float]) -> None:
        bundle.setdefault("bridge_titles",[])
        bundle.setdefault("has_bridge",False)
        bundle.setdefault("bridge_prop_count",0)
        bundle.setdefault("chain_complete",False)
        bundle.setdefault("bridge_connected",False)
        bundle.setdefault("answer_slot_aligned",False)
        bundle.setdefault("chain_complete_v2",False)
        bundle.setdefault("residual_coverage_count",0)
        bundle.setdefault("residual_query","")
        bundle.setdefault("residual_terms",[])
        bundle.setdefault("ordering_group","same_title")
        if not self.bridge_enabled or not base_props:
            return
        max_titles=int(self.bridge_cfg.get("max_bridge_titles_per_seed",2))
        max_props=int(self.bridge_cfg.get("max_bridge_props_per_title",2))
        anchor=str(bundle.get("anchor_title",""))
        bridge_rows=[]; seen_titles=set()
        for prop in self._bridge_source_props(base_props, anchor):
            for row in self._mention_rows_for_prop(str(prop.get("prop_id","")), str(prop.get("title",""))):
                title=str(row.get("mentioned_title",""))
                if title in seen_titles:
                    continue
                seen_titles.add(title); bridge_rows.append((prop,row))
                if len(bridge_rows)>=max_titles:
                    break
            if len(bridge_rows)>=max_titles:
                break
        if not bridge_rows:
            return
        existing_prop_ids={str(p.get("prop_id")) for p in bundle.get("propositions",[]) if p.get("prop_id")}
        existing_chunk_ids={str(c.get("chunk_id")) for c in bundle.get("source_chunks",[]) if c.get("chunk_id")}
        bridge_source_props=[]; bridge_props=[]; bridge_titles=[]; bridge_paths=[]; residual_terms_all=[]; total_residual_coverage=0
        for source_prop,row in bridge_rows:
            title=str(row.get("mentioned_title",""))
            residual_terms=build_residual_query(question, anchor, title, str(source_prop.get("text","")))
            candidates=list(self.props_by_title.get(title,[]))
            ranked_candidates=self._rank_bridge_props(candidates,residual_terms,q_toks)
            selected=[]
            selected_info=[]
            for prop,info in ranked_candidates:
                pid=str(prop.get("prop_id",""))
                if pid in existing_prop_ids:
                    continue
                selected.append(prop); selected_info.append(info); existing_prop_ids.add(pid)
                if len(selected)>=max_props:
                    break
            if not selected:
                continue
            bridge_coverage=sum(int(info.get("residual_coverage_count",0) or 0) for info in selected_info)
            total_residual_coverage+=bridge_coverage
            for term in residual_terms:
                if term not in residual_terms_all:
                    residual_terms_all.append(term)
            source_pid=str(source_prop.get("prop_id",""))
            if source_pid and source_pid not in existing_prop_ids:
                bridge_source_props.append(dict(source_prop)); existing_prop_ids.add(source_pid)
                source_cid=str(source_prop.get("chunk_id",""))
                if source_cid and source_cid not in existing_chunk_ids and source_cid in self.chunk_by_id:
                    bundle.setdefault("source_chunks",[]).append(dict(self.chunk_by_id[source_cid])); existing_chunk_ids.add(source_cid)
            bridge_titles.append(title)
            for prop,info in zip(selected,selected_info):
                prop_dict=dict(prop)
                prop_dict["bridge_selection"]={k:v for k,v in info.items() if k!="rank_key"}
                bridge_props.append(prop_dict)
                cid=str(prop.get("chunk_id",""))
                if cid and cid not in existing_chunk_ids and cid in self.chunk_by_id:
                    bundle.setdefault("source_chunks",[]).append(dict(self.chunk_by_id[cid])); existing_chunk_ids.add(cid)
                bridge_paths.append({"path_type":"mention_bridge","source_title":str(source_prop.get("title","")),"seed_prop":str(source_prop.get("text","")),"mention":row.get("mention"),"bridge_title":title,"bridge_prop":str(info.get("display_text") or prop.get("text","")),"seed_prop_id":source_prop.get("prop_id"),"bridge_prop_id":prop.get("prop_id"),"residual_query":" ".join(residual_terms),"residual_terms":residual_terms,"residual_coverage_count":info.get("residual_coverage_count",0),"residual_score":info.get("residual_score",0.0),"original_relevance_score":info.get("original_relevance_score",0.0)})
        if not bridge_props:
            return
        bridge_connected=bool(base_props and bridge_titles and bridge_props)
        answer_slot_aligned=total_residual_coverage>0
        old_chain_complete=bridge_connected
        bundle["propositions"]=list(bundle.get("propositions",[]))+bridge_source_props+bridge_props
        bundle["bridge_titles"]=bridge_titles
        bundle["has_bridge"]=True
        bundle["bridge_prop_count"]=len(bridge_props)
        bundle["bridge_connected"]=bridge_connected
        bundle["residual_terms"]=residual_terms_all
        bundle["residual_query"]=" ".join(residual_terms_all)
        bundle["residual_coverage_count"]=total_residual_coverage
        bundle["answer_slot_aligned"]=answer_slot_aligned
        bundle["chain_complete_v2"]=bridge_connected and answer_slot_aligned
        bundle["old_chain_complete"]=old_chain_complete
        bundle["chain_complete"]=bundle["chain_complete_v2"]
        bundle["ordering_group"]="chain_complete_v2" if bundle["chain_complete_v2"] else "bridge_connected"
        bundle["evidence_path"]=list(bundle.get("evidence_path",[]))+bridge_paths
        bundle["score"]=round(float(bundle.get("score",0.0))+sum(self._bridge_prop_score(p,q_toks,candidate_scores) for p in bridge_props),6)
        diag=dict(bundle.get("diagnostics",{}))
        diag.update({"bridge_titles":len(bridge_titles),"bridge_prop_count":len(bridge_props),"bridge_connected":bridge_connected,"answer_slot_aligned":answer_slot_aligned,"chain_complete_v2":bundle["chain_complete_v2"],"residual_coverage_count":total_residual_coverage,"residual_query":bundle["residual_query"],"refinement":"same-title + mention-bridge residual expansion"})
        bundle["diagnostics"]=diag

    def _exact_anchor_match(self, question: str, title: str) -> bool:
        q=set(_norm_tokens(question)); t=set(_norm_tokens(title))
        return bool(t and t.issubset(q))

    def _order_bundles(self, question: str, bundles: Sequence[Dict[str,Any]]) -> List[Dict[str,Any]]:
        ordering=str(self.bridge_cfg.get("ordering","chain_aware_v2"))
        finalized=[self._finalize_bundle(question,b) for b in bundles]
        if ordering not in {"chain_aware","chain_aware_v2"}:
            return sorted(finalized,key=lambda b:float(b.get("score",0.0)),reverse=True)
        def key(b: Mapping[str,Any]):
            cost=sum(token_count(c.get("text","")) for c in b.get("source_chunks",[]) or []) or sum(token_count(p.get("text","")) for p in b.get("propositions",[]) or [])
            is_generic=bool(b.get("is_generic_relation_title"))
            exact=bool(b.get("exact_anchor_match"))
            bridge_connected=bool(b.get("bridge_connected"))
            chain_v2=bool(b.get("chain_complete_v2"))
            if chain_v2 and not is_generic:
                priority=0
            elif bridge_connected and exact and not is_generic:
                priority=1
            elif exact and not is_generic:
                priority=2
            elif bridge_connected and not is_generic:
                priority=3
            elif b.get("propositions") and not is_generic:
                priority=4
            elif is_generic:
                priority=5
            else:
                priority=6
            return (
                -priority,
                float(b.get("residual_coverage_count",0.0) or 0.0),
                float(b.get("score",0.0)),
                -cost,
            )
        return sorted(finalized, key=key, reverse=True)

    def _budget_context(self, bundles: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
        budget = int(self.cfg.get("context_token_budget", self.cfg.get("refine", {}).get("context_token_budget", 3000)))
        out: List[Dict[str, Any]] = []; used = 0
        for b in bundles:
            b2 = dict(b); props = [dict(p) for p in b.get("propositions", [])]; chunks = [dict(c) for c in b.get("source_chunks", [])]
            cost = sum(token_count(c.get("text", "")) for c in chunks) or sum(token_count(p.get("text", "")) for p in props)
            if out and used + cost > budget: continue
            b2["context_tokens"] = cost; out.append(b2); used += cost
            if used >= budget: break
        return out, used
