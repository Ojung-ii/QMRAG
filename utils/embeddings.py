from __future__ import annotations
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple
import numpy as np

def l2_normalize(x: np.ndarray, eps: float=1e-12) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)

class LocalNVEmbedder:
    def __init__(self, cfg: Mapping[str, Any]):
        self.cfg=dict(cfg)
        self.model_path=self.cfg.get("model_path") or self.cfg.get("name") or "nvidia/NV-Embed-v2"
        self.batch_size=int(self.cfg.get("batch_size",4))
        self.normalize=bool(self.cfg.get("normalize",True))
        self.query_instruction=str(self.cfg.get("query_instruction") or "Given a question, retrieve passages that answer the question")
        self.max_seq_length=int(self.cfg.get("max_seq_length",32768))
        self.device=self.cfg.get("device") or None
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise RuntimeError("sentence-transformers is required for local NV-Embed-v2. Install requirements or run --no-embed.") from e
        kwargs={"trust_remote_code": bool(self.cfg.get("trust_remote_code",True))}
        if self.device: kwargs["device"]=self.device
        self.model=SentenceTransformer(self.model_path, **kwargs)
        try: self.model.max_seq_length=self.max_seq_length
        except Exception: pass
        try: self.model.tokenizer.padding_side="right"
        except Exception: pass
    def _add_eos(self, texts: Sequence[str]) -> List[str]:
        eos=None
        try: eos=self.model.tokenizer.eos_token
        except Exception: pass
        if not bool(self.cfg.get("add_eos", True)) or not eos: return [str(x) for x in texts]
        return [str(x)+eos for x in texts]
    def encode_passages(self, texts: Sequence[str]) -> np.ndarray:
        if not texts: return np.zeros((0,0), dtype=np.float32)
        arr=self.model.encode(self._add_eos(texts), batch_size=self.batch_size, normalize_embeddings=self.normalize, convert_to_numpy=True, show_progress_bar=False)
        arr=np.asarray(arr, dtype=np.float32)
        return l2_normalize(arr) if self.normalize else arr
    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        if not texts: return np.zeros((0,0), dtype=np.float32)
        prompt=f"Instruct: {self.query_instruction}\nQuery: "
        try:
            arr=self.model.encode(self._add_eos(texts), batch_size=self.batch_size, prompt=prompt, normalize_embeddings=self.normalize, convert_to_numpy=True, show_progress_bar=False)
        except TypeError:
            arr=self.model.encode([prompt+t for t in self._add_eos(texts)], batch_size=self.batch_size, normalize_embeddings=self.normalize, convert_to_numpy=True, show_progress_bar=False)
        arr=np.asarray(arr, dtype=np.float32)
        return l2_normalize(arr) if self.normalize else arr

def build_embedder(cfg: Mapping[str, Any]) -> LocalNVEmbedder:
    provider=str(cfg.get("provider","local_nvembed")).lower()
    if provider not in {"local_nvembed","nvembed","sentence_transformers","sentence-transformers"}:
        raise ValueError(f"Unsupported embedding provider: {provider}")
    return LocalNVEmbedder(cfg)

class DenseMatrixIndex:
    def __init__(self, matrix: np.ndarray):
        self.matrix=matrix
    @classmethod
    def from_npy(cls, path: str | Path, mmap: bool=True) -> "DenseMatrixIndex":
        return cls(np.load(path, mmap_mode="r" if mmap else None))
    @staticmethod
    def save(path: str | Path, matrix: np.ndarray, dtype: str="float16") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.save(path, matrix.astype(np.float16 if dtype=="float16" else np.float32))
    def search(self, q: np.ndarray, top_k: int, batch_size: int=200000) -> List[Tuple[int,float]]:
        if self.matrix is None or self.matrix.size==0 or top_k<=0: return []
        q=np.asarray(q,dtype=np.float32).reshape(-1)
        n=self.matrix.shape[0]; k=min(top_k,n)
        best_i=[]; best_s=[]
        for start in range(0,n,max(1,batch_size)):
            block=np.asarray(self.matrix[start:start+batch_size], dtype=np.float32)
            scores=block@q
            kk=min(k, scores.shape[0])
            idx=np.argpartition(-scores, kk-1)[:kk]
            best_i.extend([start+int(i) for i in idx]); best_s.extend([float(scores[i]) for i in idx])
            if len(best_i)>k*8:
                order=np.argsort(-np.asarray(best_s))[:k]
                best_i=[best_i[int(i)] for i in order]; best_s=[best_s[int(i)] for i in order]
        order=np.argsort(-np.asarray(best_s))[:k]
        return [(best_i[int(i)], best_s[int(i)]) for i in order]
