# ACE-RAG

**ACE-RAG** = Answerable Chain Evidence Retrieval for GraphRAG.

This repository implements ACE-RAG, a compact evidence-chain retrieval framework for GraphRAG:

1. Global Seed Search over anchor, chunk, and sentence/proposition views,
2. Local Refinement through mention-edge expansion and residual cue selection,
3. compact evidence-chain rendering with anchor bundling.

The code intentionally avoids broad graph expansion during retrieval. The index stays lightweight: title/entity anchors, sentence-level propositions, and packed source chunks.

## Repository layout

```text
main.py
README.md
requirements.txt
requriements.txt
requirements-flashattn-optional.txt
environment.yml
config/
  default.yaml
  smoke.yaml
  datasets/
utils/
  indexing.py
  retrieval.py
  retireval.py
  generation.py
  eval_metrics.py
  embedding.py
  data_loaders.py
  io_utils.py
  text.py
scripts/
  create_ace_rag_env.sh
  check_env.py
  check_nvembed_env.py
  check_vllm.py
  patch_nvembed_config.py
  prepare_data_layout.py
  run_dataset.sh
  run_all_vllm_nvembed.sh
  run_smoke.sh
  run_smoke_all.sh
  install_flash_attn_optional.sh
data/
  popqa/
  hotpotqa/
  2wiki/
  musique/
outputs/
cache/
```

## Data layout

Expected paths:

```text
data/popqa/popqa.json
data/popqa/popqa_corpus.json

data/hotpotqa/hotpotqa.json
data/hotpotqa/hotpotqa_corpus.json

data/2wiki/2wikimultihopqa.json
data/2wiki/2wikimultihopqa_corpus.json

data/musique/musique.json
data/musique/musique_corpus.json
```

From a flat directory containing the uploaded files, run:

```bash
python scripts/prepare_data_layout.py --src /path/to/flat/files --dst data
```

For the ChatGPT workspace layout used during development:

```bash
python scripts/prepare_data_layout.py --src /mnt/data --dst data
```

The repository zip keeps only `.gitkeep` files under `data/`; actual datasets are intentionally not committed by default.

## ACE-RAG conda environment

vLLM is assumed to be running in a separate environment. ACE-RAG only needs the client/runtime environment for retrieval, local embedding, and API calls.

Recommended setup:

```bash
bash scripts/create_ace_rag_env.sh
conda activate ACE-RAG
```

The script creates a Python 3.10 environment, installs PyTorch 2.2.0 with CUDA 12.1 through conda, installs the pinned ACE-RAG dependencies, and removes `transformer-engine` if it is present.

The NV-Embed-v2 compatibility pins are reflected in `environment.yml` and `requirements.txt`:

```text
pytorch==2.2.0
transformers==4.42.4
sentence-transformers==2.7.0
numpy<2
```

`flash-attn==2.2.0` is separated into `requirements-flashattn-optional.txt` because it often depends on the local CUDA compiler/toolchain. Install it only if your local NV-Embed-v2 snapshot requires it:

```bash
bash scripts/install_flash_attn_optional.sh
```

or during environment creation:

```bash
INSTALL_FLASH_ATTN=1 bash scripts/create_ace_rag_env.sh
```

## Environment variables

The default NV-Embed-v2 path is already set in `config/default.yaml`:

```text
/home/dilab/.cache/huggingface/models--nvidia--NV-Embed-v2/snapshots/3fa59658547db50a1e8e3346cf057fd0c77ed6ef/
```

You can still override it:

```bash
export NVEMBED_MODEL_PATH=/home/dilab/.cache/huggingface/models--nvidia--NV-Embed-v2/snapshots/3fa59658547db50a1e8e3346cf057fd0c77ed6ef/
export VLLM_BASE_URL=http://localhost:8011/v1
export VLLM_MODEL=auto
export VLLM_API_KEY=EMPTY
```

Check vLLM:

```bash
python scripts/check_vllm.py
```

Check dependency versions and local NV-Embed-v2 path without loading the large model:

```bash
python scripts/check_nvembed_env.py
```

Actually load NV-Embed-v2 and encode one query/passage pair:

```bash
python scripts/check_nvembed_env.py --load --encode
```

If local snapshot loading complains about `config.json` or `_name_or_path`, patch the local config:

```bash
python scripts/patch_nvembed_config.py
```

## Smoke test without LLM or embedding

```bash
bash scripts/run_smoke_all.sh 3
```

This validates loaders, indexing, retrieval, output writing, logging, and evaluation without GPU/model dependencies.

## Run with vLLM and NV-Embed-v2

All datasets:

```bash
bash scripts/run_all_vllm_nvembed.sh --limit 100
```

Single dataset:

```bash
bash scripts/run_dataset.sh popqa --limit 100 --reindex
bash scripts/run_dataset.sh hotpotqa --limit 100 --reindex
bash scripts/run_dataset.sh 2wiki --limit 100 --reindex
bash scripts/run_dataset.sh musique --limit 100 --reindex
```

Direct command:

```bash
python main.py \
  --config config/default.yaml \
  --datasets popqa hotpotqa 2wiki musique \
  --limit 100 \
  --reindex
```

Useful debugging flags:

```bash
--corpus-limit 1000       # build index on a smaller corpus slice
--no-embedding            # disable dense retrieval
--no-llm                  # use extractive fallback generation
--continue-on-error       # keep running after per-example failures
--mode index              # build index only
--mode eval               # evaluate existing predictions
--reindex                 # build a fresh index instead of reusing the latest dataset index
```

## Output structure

Indexing artifacts and evaluation artifacts are separated by dataset. By default, a new run reuses the latest completed dataset index. Use `--reindex` to build a fresh one.

```text
outputs/{dataset}/
  indexing/{timestamp}/
    chunks.jsonl
    propositions.jsonl
    entities.json
    index_meta.json
    dense/
      proposition_embeddings.npy
      proposition_ids.json
      chunk_embeddings.npy
      chunk_ids.json
    config.yaml
  eval/{timestamp}/
    예측결과.jsonl
    predictions.jsonl
    config.yaml
    eval.json
    eval_summary.md
    logs/
      run.log
      events.jsonl
```

The CLI also prints a markdown table with EM, F1, answer containment, support-title recall, support recall per 1k context tokens, context tokens, latency, and dense-retrieval usage rate.

## Git workflow

Recommended workflow:

```bash
unzip ACE-RAG.zip
cd ACE-RAG

git init
git add .
git commit -m "Initial ACE-RAG scaffold"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

On the server:

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd ACE-RAG
bash scripts/create_ace_rag_env.sh
conda activate ACE-RAG
python scripts/prepare_data_layout.py --src /path/to/data/files --dst data
python scripts/check_vllm.py
python scripts/check_nvembed_env.py --load --encode
bash scripts/run_all_vllm_nvembed.sh --limit 100
```

You can also unzip directly on the server and then `git init` there. GitHub-first is usually cleaner because the server run can then be reproduced with a fresh `git clone`, and outputs/data stay outside version control through `.gitignore`.
