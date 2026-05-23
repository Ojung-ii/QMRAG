from __future__ import annotations
import re, unicodedata
from typing import Iterable, List, Sequence, Set
_TOKEN_RE = re.compile(r"[A-Za-z0-9가-힣_]+(?:[-'][A-Za-z0-9가-힣_]+)?")
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'“‘(])")
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"\s+", " ", text).strip()
def normalize_answer(text: str) -> str:
    text = normalize_text(text).lower(); text = re.sub(r"\b(a|an|the)\b", " ", text); text = re.sub(r"[^0-9a-z가-힣]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()
def tokenize(text: str) -> List[str]: return [m.group(0).lower() for m in _TOKEN_RE.finditer(normalize_text(text))]
def token_count(text: str) -> int: return len(tokenize(text))
def sentence_split(text: str, min_chars: int=2) -> List[str]:
    text=normalize_text(text)
    if not text: return []
    parts=[]
    for block in re.split(r"\n+", text):
        block=block.strip()
        if block: parts.extend(_SENT_RE.split(block))
    return [p.strip() for p in parts if len(p.strip())>=min_chars] or [text]
def pack_sentences(sentences: Sequence[str], max_tokens: int, overlap_sentences: int=0) -> List[str]:
    chunks=[]; cur=[]; cur_tokens=0
    for sent in sentences:
        st=token_count(sent)
        if cur and cur_tokens+st>max_tokens:
            chunks.append(" ".join(cur).strip()); cur=cur[-overlap_sentences:] if overlap_sentences>0 else []; cur_tokens=sum(token_count(x) for x in cur)
        cur.append(sent); cur_tokens+=st
    if cur: chunks.append(" ".join(cur).strip())
    return chunks
def jaccard(a: str|Iterable[str], b: str|Iterable[str]) -> float:
    sa: Set[str]=set(tokenize(a) if isinstance(a,str) else a); sb: Set[str]=set(tokenize(b) if isinstance(b,str) else b)
    if not sa and not sb: return 1.0
    if not sa or not sb: return 0.0
    return len(sa&sb)/max(1,len(sa|sb))
def safe_truncate(text: str, max_chars: int) -> str:
    text=str(text or "")
    return text if len(text)<=max_chars else text[:max_chars-3].rstrip()+"..."
# compatibility aliases
split_sentences = sentence_split
jaccard_tokens = jaccard


def safe_truncate(text: str, max_chars: int) -> str:
    text = str(text or "")
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit("\n", 1)[0] or text[:max_chars]
