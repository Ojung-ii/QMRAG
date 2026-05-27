#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.embedding import LocalEmbeddingModel, dependency_report

DEFAULT_PATH = "/home/dilab/.cache/huggingface/models--nvidia--NV-Embed-v2/snapshots/3fa59658547db50a1e8e3346cf057fd0c77ed6ef/"


def main() -> None:
    ap = argparse.ArgumentParser(description="Check local NV-Embed-v2 dependency and loadability for ACE-RAG.")
    ap.add_argument("--model-path", default=os.environ.get("NVEMBED_MODEL_PATH", DEFAULT_PATH))
    ap.add_argument("--device", default=os.environ.get("EMBED_DEVICE", "cuda"))
    ap.add_argument("--skip-load", action="store_true", help="Only check versions and local files.")
    ap.add_argument("--load", action="store_true", help="Explicitly load the model; default behavior unless --skip-load is set.")
    ap.add_argument("--encode", action="store_true", help="Encode a tiny query/passage pair after loading.")
    args = ap.parse_args()

    model_path = Path(args.model_path).expanduser()
    report = {
        "dependency_report": dependency_report(),
        "model_path": str(model_path),
        "path_exists": model_path.exists(),
        "config_exists": (model_path / "config.json").exists(),
        "device": args.device,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.skip_load:
        return

    cfg = {
        "enabled": True,
        "provider": "local_nvembed",
        "model_path": str(model_path),
        "device": args.device,
        "batch_size": 1,
        "normalize": True,
        "trust_remote_code": True,
        "local_files_only": True,
        "max_seq_length": 32768,
        "add_eos": True,
        "query_instruction": "Given a question, retrieve passages that answer the question",
        "passage_instruction": "",
        "model_kwargs": {"torch_dtype": "float16"},
    }
    embedder = LocalEmbeddingModel(cfg)
    if args.encode:
        q = embedder.encode(["What is George Rankin's occupation?"], kind="query")
        p = embedder.encode(["George James Rankin was an Australian soldier and politician."], kind="passage")
        print(json.dumps({"query_shape": list(q.shape), "passage_shape": list(p.shape), "query_norm": float((q[0] ** 2).sum() ** 0.5)}, indent=2))
    else:
        embedder.load()
        print("NV-Embed-v2 loaded successfully.")


if __name__ == "__main__":
    main()
