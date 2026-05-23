#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from utils.embedding import patch_local_nvembed_config
DEFAULT_PATH = "/home/dilab/.cache/huggingface/models--nvidia--NV-Embed-v2/snapshots/3fa59658547db50a1e8e3346cf057fd0c77ed6ef/"
def main() -> None:
    ap = argparse.ArgumentParser(description="Patch local NV-Embed-v2 config.json _name_or_path")
    ap.add_argument("--model-path", default=os.environ.get("NVEMBED_MODEL_PATH", DEFAULT_PATH))
    a = ap.parse_args()
    changed = patch_local_nvembed_config(a.model_path)
    print(f"patched={changed} model_path={Path(a.model_path).expanduser().resolve()}")
if __name__ == "__main__": main()
