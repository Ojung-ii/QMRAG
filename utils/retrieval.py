from __future__ import annotations

import math
import json
import random
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
        diag_dict=diag.__dict__
        diag_dict.update({
            "bridge_enabled":self.bridge_enabled,
            "bridge_index_loaded":self.bridge_index_loaded,
            "ordering_mode":self.bridge_cfg.get("ordering","chain_aware") if self.bridge_enabled else "score",
            "bridge_title_count":len(bridge_titles),
            "bridge_bundle_count":sum(1 for b in bundles if b.get("has_bridge")),
            "chain_complete_count":sum(1 for b in bundles if b.get("chain_complete")),
            "has_chain_complete":any(bool(b.get("chain_complete")) for b in bundles),
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
            bprops=[dict(prop)] if prop else []
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

    def _apply_bridge_expansion(self, bundle: Dict[str,Any], question: str, q_toks: Sequence[str], base_props: Sequence[Mapping[str,Any]], base_chunks: Sequence[Mapping[str,Any]], candidate_scores: Mapping[Tuple[str,str],float]) -> None:
        bundle.setdefault("bridge_titles",[])
        bundle.setdefault("has_bridge",False)
        bundle.setdefault("bridge_prop_count",0)
        bundle.setdefault("chain_complete",False)
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
        bridge_source_props=[]; bridge_props=[]; bridge_titles=[]; bridge_paths=[]
        for source_prop,row in bridge_rows:
            title=str(row.get("mentioned_title",""))
            candidates=list(self.props_by_title.get(title,[]))
            candidates.sort(key=lambda p:self._bridge_prop_score(p,q_toks,candidate_scores), reverse=True)
            selected=[]
            for prop in candidates:
                pid=str(prop.get("prop_id",""))
                if pid in existing_prop_ids:
                    continue
                selected.append(prop); existing_prop_ids.add(pid)
                if len(selected)>=max_props:
                    break
            if not selected:
                continue
            source_pid=str(source_prop.get("prop_id",""))
            if source_pid and source_pid not in existing_prop_ids:
                bridge_source_props.append(dict(source_prop)); existing_prop_ids.add(source_pid)
                source_cid=str(source_prop.get("chunk_id",""))
                if source_cid and source_cid not in existing_chunk_ids and source_cid in self.chunk_by_id:
                    bundle.setdefault("source_chunks",[]).append(dict(self.chunk_by_id[source_cid])); existing_chunk_ids.add(source_cid)
            bridge_titles.append(title)
            for prop in selected:
                bridge_props.append(dict(prop))
                cid=str(prop.get("chunk_id",""))
                if cid and cid not in existing_chunk_ids and cid in self.chunk_by_id:
                    bundle.setdefault("source_chunks",[]).append(dict(self.chunk_by_id[cid])); existing_chunk_ids.add(cid)
                bridge_paths.append({"path_type":"mention_bridge","source_title":str(source_prop.get("title","")),"seed_prop":str(source_prop.get("text","")),"mention":row.get("mention"),"bridge_title":title,"bridge_prop":str(prop.get("text","")),"seed_prop_id":source_prop.get("prop_id"),"bridge_prop_id":prop.get("prop_id")})
        if not bridge_props:
            return
        bundle["propositions"]=list(bundle.get("propositions",[]))+bridge_source_props+bridge_props
        bundle["bridge_titles"]=bridge_titles
        bundle["has_bridge"]=True
        bundle["bridge_prop_count"]=len(bridge_props)
        bundle["chain_complete"]=bool(base_props and bridge_titles and bridge_props)
        bundle["ordering_group"]="complete_bridge_chain" if bundle["chain_complete"] else "bridge_candidate"
        bundle["evidence_path"]=list(bundle.get("evidence_path",[]))+bridge_paths
        bundle["score"]=round(float(bundle.get("score",0.0))+sum(self._bridge_prop_score(p,q_toks,candidate_scores) for p in bridge_props),6)
        diag=dict(bundle.get("diagnostics",{}))
        diag.update({"bridge_titles":len(bridge_titles),"bridge_prop_count":len(bridge_props),"refinement":"same-title + mention-bridge expansion"})
        bundle["diagnostics"]=diag

    def _exact_anchor_match(self, question: str, title: str) -> bool:
        q=set(tokenize(question)); t=set(tokenize(title))
        return bool(t and t.issubset(q))

    def _order_bundles(self, question: str, bundles: Sequence[Dict[str,Any]]) -> List[Dict[str,Any]]:
        ordering=str(self.bridge_cfg.get("ordering","chain_aware"))
        if ordering!="chain_aware":
            return sorted(bundles,key=lambda b:float(b.get("score",0.0)),reverse=True)
        def key(b: Mapping[str,Any]):
            cost=sum(token_count(c.get("text","")) for c in b.get("source_chunks",[]) or []) or sum(token_count(p.get("text","")) for p in b.get("propositions",[]) or [])
            return (
                1 if b.get("chain_complete") else 0,
                1 if self._exact_anchor_match(question,str(b.get("anchor_title",""))) else 0,
                1 if b.get("has_bridge") else 0,
                float(b.get("score",0.0)),
                -cost,
            )
        return sorted([dict(b) for b in bundles], key=key, reverse=True)

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
