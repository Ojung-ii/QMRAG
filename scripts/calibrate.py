from __future__ import annotations
import argparse, subprocess, sys
def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset",default="popqa"); p.add_argument("--limit",default="50"); args=p.parse_args()
    cmd=[sys.executable,"main.py","--config","config/smoke.yaml","--datasets",args.dataset,"--limit",args.limit,"--no-llm","--no-embedding","--reindex"]
    print("RUN"," ".join(cmd)); subprocess.run(cmd,check=True)
if __name__=="__main__": main()
