from __future__ import annotations
import importlib, json, os, platform, sys
from pathlib import Path
EXPECTED={"torch":"2.2.0","transformers":"4.42.4","sentence_transformers":"2.7.0"}
def ver(pkg):
    try:
        m=importlib.import_module(pkg); return str(getattr(m,"__version__","unknown"))
    except Exception as e: return f"MISSING: {e!r}"
def main():
    rows={"python":sys.version.split()[0],"platform":platform.platform(),"torch":ver("torch"),"transformers":ver("transformers"),"sentence_transformers":ver("sentence_transformers"),"flash_attn":ver("flash_attn"),"faiss":ver("faiss"),"openai":ver("openai")}
    try:
        import torch
        rows.update({"cuda_available":str(torch.cuda.is_available()),"torch_cuda":str(getattr(torch.version,"cuda",None)),"gpu_count":str(torch.cuda.device_count())})
        if torch.cuda.is_available(): rows["gpu0"]=torch.cuda.get_device_name(0)
    except Exception: pass
    mp=Path(os.environ.get("NVEMBED_MODEL_PATH","/home/dilab/.cache/huggingface/models--nvidia--NV-Embed-v2/snapshots/3fa59658547db50a1e8e3346cf057fd0c77ed6ef/"))
    rows.update({"NVEMBED_MODEL_PATH":str(mp),"nvembed_path_exists":str(mp.exists()),"nvembed_config_exists":str((mp/"config.json").exists())})
    print(json.dumps(rows,indent=2,ensure_ascii=False))
    warn=[f"{p}: expected {e}, got {rows.get(p,'')}" for p,e in EXPECTED.items() if not rows.get(p,'').startswith(e)]
    if warn:
        print("\n[WARN] Version mismatch detected:")
        for w in warn: print("  -",w)
if __name__=="__main__": main()
