from __future__ import annotations
import argparse, shutil
from pathlib import Path
MAP={"popqa":[("popqa.json","popqa.json"),("popqa_corpus.json","popqa_corpus.json")],"hotpotqa":[("hotpotqa.json","hotpotqa.json"),("hotpotqa_corpus.json","hotpotqa_corpus.json")],"2wiki":[("2wikimultihopqa.json","2wikimultihopqa.json"),("2wikimultihopqa_corpus.json","2wikimultihopqa_corpus.json")],"musique":[("musique.json","musique.json"),("musique_corpus.json","musique_corpus.json")]}
ALIASES={"popqa.json":["popqa(1).json"],"popqa_corpus.json":["popqa_corpus(1).json"],"hotpotqa.json":["hotpotqa(1).json","hotpotqa_dev.json"],"hotpotqa_corpus.json":["hotpotqa_corpus(1).json"],"2wikimultihopqa.json":["2wikimultihopqa_dev.json"],"musique.json":["musique_dev.json","musique_dev.jsonl"]}
def find(src, name):
    for n in [name]+ALIASES.get(name,[]):
        p=src/n
        if p.exists(): return p
    return None
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--src",default="/mnt/data"); ap.add_argument("--dst",default="data"); ap.add_argument("--symlink",action="store_true"); a=ap.parse_args(); src=Path(a.src).resolve(); dst=Path(a.dst).resolve()
    for ds,pairs in MAP.items():
        d=dst/ds; d.mkdir(parents=True,exist_ok=True)
        for canonical,target in pairs:
            sp=find(src,canonical); tp=d/target
            if sp is None: print(f"[MISS] {ds}: {canonical}"); continue
            if tp.exists() or tp.is_symlink(): tp.unlink()
            if a.symlink: tp.symlink_to(sp); action="symlink"
            else: shutil.copy2(sp,tp); action="copy"
            print(f"[{action}] {sp} -> {tp}")
if __name__=="__main__": main()
