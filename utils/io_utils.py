from __future__ import annotations
import contextlib, dataclasses, json, os, platform, re, resource, socket, sys, time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Mapping, MutableMapping
import yaml
def now_timestamp() -> str: return datetime.now().strftime("%Y%m%d_%H%M%S")
def ensure_dir(path: str|Path) -> Path: p=Path(path); p.mkdir(parents=True, exist_ok=True); return p
def _expand_env(obj: Any) -> Any:
    if isinstance(obj,str):
        def repl(m: re.Match[str]) -> str: return os.environ.get(m.group(1), m.group(2))
        obj=re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)[:-]-(.*?)\}", repl, obj); return os.path.expandvars(obj)
    if isinstance(obj,list): return [_expand_env(x) for x in obj]
    if isinstance(obj,dict): return {k:_expand_env(v) for k,v in obj.items()}
    return obj
def load_yaml(path: str|Path) -> Dict[str,Any]:
    with open(path,"r",encoding="utf-8") as f: return _expand_env(yaml.safe_load(f) or {})
def dump_yaml(obj: Mapping[str,Any], path: str|Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path,"w",encoding="utf-8") as f: yaml.safe_dump(dict(obj), f, allow_unicode=True, sort_keys=False)
def to_jsonable(obj: Any) -> Any:
    try:
        import numpy as np
        if isinstance(obj,np.ndarray): return obj.tolist()
        if isinstance(obj,(np.integer,)): return int(obj)
        if isinstance(obj,(np.floating,)): return float(obj)
    except Exception: pass
    if dataclasses.is_dataclass(obj): return dataclasses.asdict(obj)
    if isinstance(obj,Path): return str(obj)
    if isinstance(obj,Mapping): return {str(k):to_jsonable(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple,set)): return [to_jsonable(x) for x in obj]
    return obj
def dump_json(obj: Any, path: str|Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path,"w",encoding="utf-8") as f: json.dump(to_jsonable(obj), f, ensure_ascii=False, indent=2)
def write_jsonl(rows: Iterable[Mapping[str,Any]], path: str|Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path,"w",encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(to_jsonable(row),ensure_ascii=False)+"\n")
def read_jsonl(path: str|Path) -> list[dict[str,Any]]:
    rows=[]
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows
def deep_update(base: MutableMapping[str,Any], override: Mapping[str,Any]) -> MutableMapping[str,Any]:
    for k,v in override.items():
        if isinstance(v,Mapping) and isinstance(base.get(k),MutableMapping): deep_update(base[k], v)  # type: ignore[index]
        else: base[k]=v
    return base
def memory_snapshot_mb() -> Dict[str,float]:
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss; return {"max_rss_mb":round(float(rss/1024.0),3)}
class ExperimentLogger:
    def __init__(self,out_dir: str|Path,echo: bool=True):
        self.out_dir=ensure_dir(out_dir); self.log_dir=ensure_dir(self.out_dir/"logs"); self.echo=echo; self.log_path=self.log_dir/"run.log"; self.events_path=self.log_dir/"events.jsonl"; self.event({"event":"run.start","host":socket.gethostname(),"python":sys.version,"platform":platform.platform(),"pid":os.getpid()})
    def log(self,msg: str) -> None:
        line=f"[{datetime.now().isoformat(timespec='seconds')}] {msg}"
        with open(self.log_path,"a",encoding="utf-8") as f: f.write(line+"\n")
        if self.echo: print(line, flush=True)
    def event(self,payload: Mapping[str,Any]) -> None:
        row={"ts":datetime.now().isoformat(timespec="milliseconds"),**dict(payload),**memory_snapshot_mb()}
        with open(self.events_path,"a",encoding="utf-8") as f: f.write(json.dumps(to_jsonable(row),ensure_ascii=False)+"\n")
    @contextlib.contextmanager
    def time_block(self,event: str,**payload: Any) -> Iterator[None]:
        start=time.perf_counter(); self.event({"event":event+".start",**payload})
        try: yield
        except Exception as e:
            self.event({"event":event+".error","error":repr(e),**payload,"duration_s":round(time.perf_counter()-start,6)}); raise
        finally: self.event({"event":event+".end",**payload,"duration_s":round(time.perf_counter()-start,6)})

# Backward-compatible aliases used by different modules.
def load_json(path: str|Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def iter_jsonl(path: str|Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def load_json(path: str|Path) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
