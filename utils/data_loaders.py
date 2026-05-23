from __future__ import annotations
import ast, json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
from .text import normalize_text

@dataclass
class CorpusDoc:
    doc_id: str
    title: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class QAExample:
    id: str
    question: str
    answers: List[str]
    support_titles: List[str] = field(default_factory=list)
    support_facts: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

def _read(path: str | Path) -> Any:
    p=Path(path)
    if not p.exists(): raise FileNotFoundError(f"Missing data file: {p}")
    text=p.read_text(encoding='utf-8').strip()
    if not text: return []
    if p.suffix.lower()=='.jsonl': return [json.loads(x) for x in text.splitlines() if x.strip()]
    try: return json.loads(text)
    except json.JSONDecodeError: return [json.loads(x) for x in text.splitlines() if x.strip()]

def _rows(obj: Any) -> List[Any]:
    if isinstance(obj,list): return obj
    if isinstance(obj,dict):
        for k in ('data','examples','rows'):
            if isinstance(obj.get(k),list): return obj[k]
        return list(obj.values())
    return list(obj)

def _answers(v: Any) -> List[str]:
    if v is None: return []
    if isinstance(v,list): return [str(x) for x in v if str(x).strip()]
    if isinstance(v,str):
        s=v.strip()
        if not s: return []
        for fn in (json.loads, ast.literal_eval):
            try:
                o=fn(s)
                if isinstance(o,list): return [str(x) for x in o if str(x).strip()]
                if isinstance(o,str): return [o]
            except Exception: pass
        return [x.strip() for x in s.split('|') if x.strip()] if '|' in s else [s]
    return [str(v)]

def _unique_docs(rows: Iterable[Mapping[str,Any]], prefix: str) -> List[CorpusDoc]:
    docs=[]; seen=set()
    for i,r in enumerate(rows):
        title=normalize_text(r.get('title') or r.get('wiki_title') or r.get('name') or f'{prefix}_{i}')
        text=normalize_text(r.get('text') or r.get('context') or r.get('paragraph_text') or r.get('paragraph') or '')
        if not text: continue
        key=(title,text)
        if key in seen: continue
        seen.add(key)
        docs.append(CorpusDoc(str(r.get('id') or r.get('idx') or f'{prefix}_{len(docs)}'), title, text, {k:v for k,v in r.items() if k not in {'text','context','paragraph_text','paragraph'}}))
    return docs

def _load_corpus(path: str | Path | None, corpus_limit: Optional[int], prefix: str) -> List[CorpusDoc]:
    if not path or not Path(path).exists(): return []
    rows=_rows(_read(path)); rows=rows[:corpus_limit] if corpus_limit else rows
    return _unique_docs(rows,prefix)

def load_popqa(cfg: Mapping[str,Any], limit: Optional[int]=None, corpus_limit: Optional[int]=None):
    qa_rows=_rows(_read(cfg.get('qa_path') or 'data/popqa/popqa.json')); qa_rows=qa_rows[:limit] if limit else qa_rows
    docs=_load_corpus(cfg.get('corpus_path'), corpus_limit, 'popqa_corpus')
    if not docs:
        paras=[]
        for r in qa_rows:
            for p in r.get('paragraphs',[]) or []: paras.append({'title':p.get('title'),'text':p.get('text')})
        docs=_unique_docs(paras,'popqa_para')
    qas=[]
    for i,r in enumerate(qa_rows):
        support=[]
        for p in r.get('paragraphs',[]) or []:
            if p.get('is_supporting') and p.get('title'): support.append(normalize_text(p.get('title')))
        if not support:
            for k in ('s_wiki_title','o_wiki_title'):
                if r.get(k): support.append(normalize_text(r[k]))
        qas.append(QAExample(str(r.get('id') or i), normalize_text(r.get('question','')), _answers(r.get('possible_answers') or r.get('answers') or r.get('answer') or r.get('obj')), list(dict.fromkeys(support)), [], dict(r)))
    return qas,docs

def _context_docs(rows: List[Mapping[str,Any]], prefix: str) -> List[CorpusDoc]:
    d=[]
    for i,r in enumerate(rows):
        for item in r.get('context') or r.get('contexts') or []:
            if isinstance(item,(list,tuple)) and len(item)>=2:
                title=str(item[0]); text=' '.join(item[1]) if isinstance(item[1],list) else str(item[1])
            elif isinstance(item,Mapping):
                title=str(item.get('title') or item.get('name') or f'doc_{len(d)}'); text=item.get('text') or ' '.join(item.get('sentences',[]))
            else: continue
            d.append({'id':f'{prefix}_{i}_{title}','title':title,'text':text})
    return _unique_docs(d,prefix)

def load_hotpotqa(cfg: Mapping[str,Any], limit: Optional[int]=None, corpus_limit: Optional[int]=None):
    rows=_rows(_read(cfg.get('qa_path') or 'data/hotpotqa/hotpotqa.json')); rows=rows[:limit] if limit else rows
    qas=[]
    for i,r in enumerate(rows):
        sf=r.get('supporting_facts',[]) or []; titles=[]
        for x in sf:
            if isinstance(x,(list,tuple)) and x: titles.append(normalize_text(x[0]))
            elif isinstance(x,Mapping) and x.get('title'): titles.append(normalize_text(x['title']))
        qas.append(QAExample(str(r.get('_id') or r.get('id') or i), normalize_text(r.get('question','')), _answers(r.get('answer') or r.get('answers')), list(dict.fromkeys(titles)), sf, dict(r)))
    docs=_load_corpus(cfg.get('corpus_path'), corpus_limit, 'hotpotqa_corpus') or _context_docs(rows,'hotpot')
    return qas,docs

def load_2wiki(cfg: Mapping[str,Any], limit: Optional[int]=None, corpus_limit: Optional[int]=None):
    rows=_rows(_read(cfg.get('qa_path') or 'data/2wiki/2wikimultihopqa.json')); rows=rows[:limit] if limit else rows
    qas=[]
    for i,r in enumerate(rows):
        titles=[]
        for key in ('supporting_facts','evidences','supporting_titles'):
            for x in r.get(key) or []:
                if isinstance(x,str): titles.append(normalize_text(x))
                elif isinstance(x,(list,tuple)) and x: titles.append(normalize_text(x[0]))
                elif isinstance(x,Mapping) and x.get('title'): titles.append(normalize_text(x['title']))
        qas.append(QAExample(str(r.get('_id') or r.get('id') or i), normalize_text(r.get('question','')), _answers(r.get('answer') or r.get('answers')), list(dict.fromkeys(titles)), r.get('supporting_facts',[]) or [], dict(r)))
    docs=_load_corpus(cfg.get('corpus_path'), corpus_limit, '2wiki_corpus') or _context_docs(rows,'2wiki')
    return qas,docs

def load_musique(cfg: Mapping[str,Any], limit: Optional[int]=None, corpus_limit: Optional[int]=None):
    rows=_rows(_read(cfg.get('qa_path') or 'data/musique/musique.json')); rows=rows[:limit] if limit else rows
    qas=[]; doc_rows=[]
    for i,r in enumerate(rows):
        titles=[]
        for j,p in enumerate(r.get('paragraphs') or r.get('context') or []):
            if not isinstance(p,Mapping): continue
            title=str(p.get('title') or p.get('idx') or f'para_{j}'); text=p.get('paragraph_text') or p.get('text') or p.get('context') or ''
            doc_rows.append({'id':f'mus_{i}_{j}','title':title,'text':text})
            if p.get('is_supporting') or p.get('supporting'): titles.append(normalize_text(title))
        ans=_answers(r.get('answer') or r.get('answers'))
        for a in _answers(r.get('answer_aliases')):
            if a not in ans: ans.append(a)
        qas.append(QAExample(str(r.get('id') or r.get('_id') or i), normalize_text(r.get('question','')), ans, list(dict.fromkeys(titles)), [], dict(r)))
    docs=_load_corpus(cfg.get('corpus_path'), corpus_limit, 'musique_corpus') or _unique_docs(doc_rows,'musique')
    return qas,docs

def load_dataset(dataset: str, cfg: Mapping[str,Any], limit: Optional[int]=None, corpus_limit: Optional[int]=None):
    dtype=str(cfg.get('type',dataset)).lower()
    if dtype=='popqa': return load_popqa(cfg,limit,corpus_limit)
    if dtype in {'hotpotqa','hotpot'}: return load_hotpotqa(cfg,limit,corpus_limit)
    if dtype in {'2wiki','2wikimultihopqa','twowiki'}: return load_2wiki(cfg,limit,corpus_limit)
    if dtype=='musique': return load_musique(cfg,limit,corpus_limit)
    raise ValueError(f'Unsupported dataset: {dataset}')
