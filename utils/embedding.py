from __future__ import annotations

import json
import gc
import shutil
import time
from pathlib import Path
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, List

import numpy as np
from tqdm import tqdm

from .io_utils import dump_json, ensure_dir, load_json


RECOMMENDED_NVEMBED_VERSIONS = {
    "torch": "2.2.0",
    "transformers": "4.42.4",
    "sentence-transformers": "2.7.0",
    "flash-attn": "2.2.0",
}


def _pkg_version(pkg: str) -> str:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return "not_installed"
    except Exception:
        return "unknown"


def dependency_report() -> Dict[str, str]:
    return {
        "torch": _pkg_version("torch"),
        "transformers": _pkg_version("transformers"),
        "sentence_transformers": _pkg_version("sentence-transformers"),
        "flash_attn": _pkg_version("flash-attn"),
        "numpy": _pkg_version("numpy"),
    }


def nvembed_dependency_report() -> Dict[str, Any]:
    installed = dependency_report()
    expected = {
        "torch": RECOMMENDED_NVEMBED_VERSIONS["torch"],
        "transformers": RECOMMENDED_NVEMBED_VERSIONS["transformers"],
        "sentence_transformers": RECOMMENDED_NVEMBED_VERSIONS["sentence-transformers"],
        "flash_attn": RECOMMENDED_NVEMBED_VERSIONS["flash-attn"],
    }
    return {
        "installed": installed,
        "recommended_for_nvembed_v2": expected,
        "strict_matches": {k: str(installed.get(k, "")).startswith(str(v)) for k, v in expected.items()},
    }


def _torch_dtype_from_string(name: Optional[str]):
    if not name or str(name).lower() in {"none", "null", "auto"}:
        return None
    import torch
    table = {
        "float16": torch.float16, "fp16": torch.float16,
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float32": torch.float32, "fp32": torch.float32,
    }
    key = str(name).lower()
    if key not in table:
        raise ValueError(f"Unsupported torch dtype: {name}")
    return table[key]


def normalize_matrix(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    norms = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return (x / norms).astype(np.float32)


class LocalEmbeddingModel:
    """Local embedding wrapper tuned for NV-Embed-v2.

    Default backend is HuggingFace Transformers because NVIDIA's model card
    documents `AutoModel.from_pretrained(..., trust_remote_code=True)` and
    `model.encode(texts, instruction=..., max_length=...)`. SentenceTransformers
    remains available as a fallback via `backend: sentence_transformers`.
    """

    def __init__(self, cfg: Mapping[str, Any], logger: Optional[Any] = None):
        self.cfg = dict(cfg or {})
        self.logger = logger
        self.enabled = bool(self.cfg.get("enabled", True))
        self.backend = str(self.cfg.get("backend", "transformers") or "transformers").lower()
        self.model_path = str(self.cfg.get("model_path") or self.cfg.get("model_name") or "")
        self.device = self.cfg.get("device") or None
        self.device_map = self.cfg.get("device_map", None)
        self.batch_size = int(self.cfg.get("batch_size", 2))
        self.normalize = bool(self.cfg.get("normalize", self.cfg.get("normalize_embeddings", True)))
        self.query_instruction = str(self.cfg.get("query_instruction", "") or "")
        self.passage_instruction = str(self.cfg.get("passage_instruction", "") or "")
        self.local_files_only = bool(self.cfg.get("local_files_only", True))
        self.trust_remote_code = bool(self.cfg.get("trust_remote_code", True))
        self.max_seq_length = int(self.cfg.get("max_seq_length", 2048) or 2048)
        self.add_eos = bool(self.cfg.get("add_eos", False))
        self.torch_dtype_name = self.cfg.get("torch_dtype") or self.cfg.get("model_kwargs", {}).get("torch_dtype")
        self._model: Any = None
        self._tokenizer: Any = None

    def load(self) -> None:
        if not self.enabled or self._model is not None:
            return
        if not self.model_path:
            raise ValueError("embedding.model_path must be set")
        if self.logger:
            self.logger.log(
                "Loading embedding model: "
                + json.dumps({
                    "backend": self.backend,
                    "model_path": self.model_path,
                    "device": self.device,
                    "device_map": self.device_map,
                    "dependency_report": nvembed_dependency_report(),
                }, ensure_ascii=False)
            )
        if self.backend in {"transformers", "hf", "auto"}:
            try:
                self._load_transformers()
                return
            except Exception:
                if self.backend != "auto":
                    raise
                if self.logger:
                    self.logger.log("Transformers backend failed; falling back to SentenceTransformers")
        self._load_sentence_transformers()

    def _load_transformers(self) -> None:
        from transformers import AutoModel
        dtype = _torch_dtype_from_string(self.torch_dtype_name)
        kwargs: Dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
        }
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        if self.device_map not in {None, "", "null"}:
            kwargs["device_map"] = self.device_map
        self._model = AutoModel.from_pretrained(self.model_path, **kwargs)
        if self.device and self.device_map in {None, "", "null"}:
            self._model = self._model.to(self.device)
        self._model.eval()
        try:
            self._tokenizer = self._model.tokenizer
            self._tokenizer.padding_side = "right"
        except Exception:
            self._tokenizer = None

    def _load_sentence_transformers(self) -> None:
        from sentence_transformers import SentenceTransformer
        model_kwargs = dict(self.cfg.get("model_kwargs", {}) or {})
        dtype = _torch_dtype_from_string(self.torch_dtype_name)
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        kwargs = {
            "device": self.device,
            "trust_remote_code": self.trust_remote_code,
            "local_files_only": self.local_files_only,
        }
        if model_kwargs:
            kwargs["model_kwargs"] = model_kwargs
        try:
            self._model = SentenceTransformer(self.model_path, **kwargs)
        except TypeError:
            self._model = SentenceTransformer(self.model_path, device=self.device, trust_remote_code=self.trust_remote_code)
        try:
            self._model.max_seq_length = self.max_seq_length
            self._model.tokenizer.padding_side = "right"
        except Exception:
            pass

    def _query_instruction(self) -> str:
        instr = self.query_instruction.strip()
        if not instr:
            return ""
        if instr.startswith("Instruct:"):
            return instr if instr.endswith("Query: ") else instr.rstrip() + "\nQuery: "
        return f"Instruct: {instr}\nQuery: "

    def _eos_token(self) -> str:
        try:
            tok = getattr(self._model, "tokenizer", None) or self._tokenizer
            return tok.eos_token or ""
        except Exception:
            return ""

    def _append_eos(self, texts: Sequence[str]) -> List[str]:
        if not self.add_eos:
            return [str(t) for t in texts]
        eos = self._eos_token()
        return [str(t) + eos for t in texts]

    def close(self) -> None:
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass

    def _show_progress(self, total_batches: int, progress_desc: Optional[str]) -> bool:
        if total_batches <= 1:
            return False
        default = bool(progress_desc)
        return bool(self.cfg.get("show_progress", self.cfg.get("show_progress_bar", default)))

    def encode(
        self,
        texts: Sequence[str],
        kind: str = "passage",
        is_query: Optional[bool] = None,
        progress_desc: Optional[str] = None,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        if is_query is not None:
            kind = "query" if is_query else "passage"
        self.load()
        assert self._model is not None
        instruction = self._query_instruction() if kind == "query" else self.passage_instruction
        raw_texts = self._append_eos([str(t) for t in texts])
        if self.backend in {"transformers", "hf", "auto"} and hasattr(self._model, "encode"):
            import torch
            outs = []
            total_batches = (len(raw_texts) + self.batch_size - 1) // self.batch_size
            batch_starts = range(0, len(raw_texts), self.batch_size)
            if self._show_progress(total_batches, progress_desc):
                batch_starts = tqdm(
                    batch_starts,
                    total=total_batches,
                    desc=progress_desc or f"embed.{kind}",
                    ncols=100,
                    unit="batch",
                )
            with torch.inference_mode():
                for i in batch_starts:
                    batch = raw_texts[i:i+self.batch_size]
                    emb = self._model.encode(batch, instruction=instruction, max_length=self.max_seq_length)
                    if hasattr(emb, "detach"):
                        emb = emb.detach().float().cpu().numpy()
                    outs.append(np.asarray(emb, dtype=np.float32))
            arr = np.vstack(outs) if outs else np.zeros((0, 0), dtype=np.float32)
        else:
            # SentenceTransformers uses prompt for query instruction. Passage prompt remains empty.
            prompt = instruction if kind == "query" else None
            kwargs = {
                "batch_size": self.batch_size,
                "convert_to_numpy": True,
                "normalize_embeddings": self.normalize,
                "show_progress_bar": self._show_progress(
                    (len(raw_texts) + self.batch_size - 1) // self.batch_size,
                    progress_desc,
                ),
            }
            if prompt:
                kwargs["prompt"] = prompt
            arr = self._model.encode(raw_texts, **kwargs)
            arr = np.asarray(arr, dtype=np.float32)
        return normalize_matrix(arr) if self.normalize else np.asarray(arr, dtype=np.float32)

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode(texts, kind="query")

    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        return self.encode(texts, kind="passage")


def unit_text(row: Mapping[str, Any]) -> str:
    title = str(row.get("title", ""))
    text = str(row.get("text", ""))
    return f"Title: {title}\nText: {text}" if title else text


class DenseIndex:
    def __init__(self, ids: Sequence[str], matrix: np.ndarray, id_to_row: Mapping[str, Mapping[str, Any]], unit: str):
        self.ids = list(ids)
        self.matrix = normalize_matrix(np.asarray(matrix, dtype=np.float32))
        self.id_to_row = id_to_row
        self.unit = unit

    @classmethod
    def build(
        cls,
        rows: Sequence[Mapping[str, Any]],
        id_key: str,
        unit: str,
        embedder: LocalEmbeddingModel,
        logger: Optional[Any] = None,
    ) -> "DenseIndex":
        ids = [str(r[id_key]) for r in rows]
        texts = [unit_text(r) for r in rows]
        if logger:
            logger.log(
                f"Encoding dense {unit} embeddings: n={len(texts)} "
                f"batch_size={embedder.batch_size}"
            )
        t0 = time.perf_counter()
        if logger and hasattr(logger, "time_block"):
            with logger.time_block("dense.encode", unit=unit, n=len(texts), batch_size=embedder.batch_size):
                mat = embedder.encode(texts, kind="passage", progress_desc=f"dense.{unit}")
        else:
            mat = embedder.encode(texts, kind="passage", progress_desc=f"dense.{unit}")
        if logger:
            dim = int(mat.shape[1]) if mat.ndim == 2 and mat.shape[0] else 0
            logger.log(
                f"Encoded dense {unit} embeddings: n={len(texts)} dim={dim} "
                f"seconds={time.perf_counter() - t0:.3f}"
            )
        return cls(ids=ids, matrix=mat, id_to_row={str(r[id_key]): dict(r) for r in rows}, unit=unit)

    @staticmethod
    def exists(out_dir: str | Path, name: str) -> bool:
        p = Path(out_dir)
        return (p / f"{name}_embeddings.npy").exists() and (p / f"{name}_ids.json").exists()

    def save(self, out_dir: str | Path, name: str) -> None:
        out = ensure_dir(out_dir)
        np.save(out / f"{name}_embeddings.npy", self.matrix)
        dump_json({"unit": self.unit, "ids": self.ids}, out / f"{name}_ids.json")

    @classmethod
    def load(cls, out_dir: str | Path, name: str, rows: Sequence[Mapping[str, Any]], id_key: str, unit: str) -> "DenseIndex":
        p = Path(out_dir)
        mat = np.load(p / f"{name}_embeddings.npy", mmap_mode=None)
        meta = load_json(p / f"{name}_ids.json")
        return cls(ids=meta["ids"], matrix=mat, id_to_row={str(r[id_key]): dict(r) for r in rows}, unit=unit)

    def search(self, query_vec: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        if self.matrix.size == 0 or not self.ids:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        if q.ndim == 2:
            q = q[0]
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        sims = self.matrix @ q
        k = min(int(top_k), len(sims))
        if k <= 0:
            return []
        idx = np.argpartition(-sims, k - 1)[:k]
        pairs = [(self.ids[int(i)], float(sims[int(i)])) for i in idx]
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs


def build_or_load_dense_indexes(index: Mapping[str, Any], cfg: Mapping[str, Any], index_dir: str | Path, logger: Optional[Any] = None, force: bool = False) -> Dict[str, DenseIndex]:
    dense_cfg = cfg.get("dense", {})
    if not dense_cfg.get("enabled", False):
        return {}
    embedder = LocalEmbeddingModel(cfg.get("embedding", {}), logger=logger)
    dense_dir = ensure_dir(Path(index_dir) / "dense")
    units = dense_cfg.get("units", ["proposition", "chunk"])
    out: Dict[str, DenseIndex] = {}
    try:
        if "proposition" in units:
            if not force and DenseIndex.exists(dense_dir, "proposition"):
                if logger: logger.log(f"Loading dense proposition index: {dense_dir}")
                out["proposition"] = DenseIndex.load(dense_dir, "proposition", index.get("propositions", []), "prop_id", "proposition")
            else:
                if logger: logger.log("Building dense proposition index")
                out["proposition"] = DenseIndex.build(index.get("propositions", []), "prop_id", "proposition", embedder, logger)
                out["proposition"].save(dense_dir, "proposition")
                if logger: logger.log(f"Saved dense proposition index: {dense_dir}")
        if "chunk" in units:
            if not force and DenseIndex.exists(dense_dir, "chunk"):
                if logger: logger.log(f"Loading dense chunk index: {dense_dir}")
                out["chunk"] = DenseIndex.load(dense_dir, "chunk", index.get("chunks", []), "chunk_id", "chunk")
            else:
                if logger: logger.log("Building dense chunk index")
                out["chunk"] = DenseIndex.build(index.get("chunks", []), "chunk_id", "chunk", embedder, logger)
                out["chunk"].save(dense_dir, "chunk")
                if logger: logger.log(f"Saved dense chunk index: {dense_dir}")
        dump_json({"enabled": True, "units": list(out.keys()), "model_path": cfg.get("embedding", {}).get("model_path")}, dense_dir / "dense_meta.json")
    finally:
        embedder.close()
    return out


def encode_query(cfg: Mapping[str, Any], query: str, logger: Optional[Any] = None, cache: Optional[Dict[str, LocalEmbeddingModel]] = None) -> np.ndarray:
    emb_cfg = cfg.get("embedding", {})
    key = json.dumps({k: str(v) for k, v in emb_cfg.items() if k != "model_kwargs"}, sort_keys=True)
    if cache is not None and key in cache:
        embedder = cache[key]
    else:
        embedder = LocalEmbeddingModel(emb_cfg, logger=logger)
        if cache is not None:
            cache[key] = embedder
    return embedder.encode([query], kind="query")


def patch_local_nvembed_config(model_path: str | Path) -> bool:
    """Patch local NV-Embed-v2 config.json _name_or_path to the snapshot path.

    NVIDIA's model card notes that local cache path issues can be resolved by
    setting config.json['_name_or_path'] to the local model path. This function
    creates config.json.bak before modifying the file. Returns True if changed.
    """
    p = Path(model_path).expanduser().resolve()
    cfg_path = p / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found under {p}")
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    target = str(p)
    if data.get("_name_or_path") == target:
        return False
    bak = cfg_path.with_suffix(".json.bak")
    if not bak.exists():
        shutil.copy2(cfg_path, bak)
    data["_name_or_path"] = target
    cfg_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return True


# Compatibility helpers for older scaffold modules.
def embedding_enabled(cfg: Mapping[str, Any]) -> bool:
    return bool((cfg or {}).get("enabled", False))


class VectorSearchIndex(DenseIndex):
    pass


def topk_inner_product(matrix: np.ndarray, query_vec: np.ndarray, k: int) -> List[Tuple[int, float]]:
    if matrix.size == 0:
        return []
    q = np.asarray(query_vec, dtype=np.float32)
    if q.ndim == 2:
        q = q[0]
    sims = np.asarray(matrix, dtype=np.float32) @ q
    kk = min(int(k), len(sims))
    if kk <= 0:
        return []
    idx = np.argpartition(-sims, kk - 1)[:kk]
    pairs = [(int(i), float(sims[int(i)])) for i in idx]
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs
