from __future__ import annotations
import os
from openai import OpenAI
base=os.environ.get("VLLM_BASE_URL","http://localhost:8011/v1"); key=os.environ.get("VLLM_API_KEY","EMPTY")
client=OpenAI(base_url=base, api_key=key, timeout=30)
models=client.models.list()
print("base_url=",base); print("models=",[m.id for m in models.data])
