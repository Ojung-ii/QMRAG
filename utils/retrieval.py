from __future__ import annotations

import math
import random
import time
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .embedding import DenseIndex, encode_query
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
        return {"question": question, "candidates": candidates, "seeds": seeds, "evidence_bundles": bundles, "diagnostics": diag.__dict__}

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
            bundles.append({"bundle_id":f"b{len(bundles)}","anchor_title":title,"seed":dict(s),"propositions":[dict(p) for p in selected_props],"source_chunks":[dict(c) for c in selected_chunks],"evidence_path":path,"score":round(score,6),"diagnostics":{"num_props":len(selected_props),"num_chunks":len(selected_chunks),"refinement":"same-title entity hub expansion"}})
        for c in candidates:
            if len(bundles) >= max_bundles: break
            title = c.get("title", "")
            if not title or title in seen_titles: continue
            seen_titles.add(title)
            chunk = self.chunk_by_id.get(c.get("chunk_id")) if c.get("chunk_id") else None
            prop = self.prop_by_id.get(c.get("id")) if c.get("unit") == "proposition" else None
            bundles.append({"bundle_id":f"b{len(bundles)}","anchor_title":title,"seed":dict(c),"propositions":[dict(prop)] if prop else [],"source_chunks":[dict(chunk)] if chunk else ([dict(c)] if c.get("unit") == "chunk" else []),"evidence_path":[{"type":"query","id":"q","text":question},{"type":"candidate","id":c.get("id"),"text":c.get("text","")}],"score":round(float(c.get("score",0.0)),6),"diagnostics":{"refinement":"direct-candidate bridge"}})
        bundles.sort(key=lambda b: b["score"], reverse=True)
        return bundles[:max_bundles]

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
