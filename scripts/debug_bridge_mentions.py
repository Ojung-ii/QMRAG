#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


GOLDEN_CASES=[
    ("hotpotqa", "The Newcomers (film)", "Chris Evans", "Chris Evans (actor)"),
    ("hotpotqa", "Distribution of Industry Act 1950", "Clement Attlee", "Clement Attlee"),
    ("2wiki", "Lothair II", "Ermengarde of Tours", "Ermengarde of Tours"),
    ("2wiki", "Changed It", "Nicki Minaj", "Nicki Minaj"),
]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows=[]
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_index_dirs(output_root: Path) -> Iterable[Path]:
    for path in sorted(output_root.rglob("title_mentions.jsonl")):
        yield path.parent


def dataset_for_index(index_dir: Path) -> str:
    parts=index_dir.parts
    if "outputs" in parts:
        rest=parts[parts.index("outputs")+1:]
        if len(rest)>=3 and rest[1]=="indexing":
            return rest[0]
        if len(rest)>=3 and rest[2]=="index":
            return rest[1]
        if rest:
            return rest[0]
    return "UNKNOWN"


def case_matches(row: Dict[str, Any], source: str, mention: str, target: str) -> bool:
    source_ok=source.lower() in str(row.get("source_title","")).lower()
    target_ok=target.lower()==str(row.get("mentioned_title","")).lower()
    mention_text=str(row.get("mention",""))+" "+str(row.get("mention_norm",""))
    mention_ok=mention.lower() in mention_text.lower()
    return source_ok and target_ok and mention_ok


def main() -> None:
    parser=argparse.ArgumentParser(description="Check golden mention-bridge cases in built indexes.")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--limit", type=int, default=5)
    args=parser.parse_args()

    for index_dir in iter_index_dirs(Path(args.output_root)):
        dataset=dataset_for_index(index_dir)
        if args.dataset and dataset != args.dataset:
            continue
        rows=read_jsonl(index_dir/"title_mentions.jsonl")
        print(f"index_dir={index_dir} dataset={dataset} mentions={len(rows)}")
        for case_dataset, source, mention, target in GOLDEN_CASES:
            if dataset != case_dataset:
                continue
            matches=[r for r in rows if case_matches(r,source,mention,target)]
            status="FOUND" if matches else "missing"
            print(f"  {status}: {source} -> {mention} -> {target} n={len(matches)}")
            for row in matches[:max(0,args.limit)]:
                print("    "+json.dumps(row,ensure_ascii=False))


if __name__=="__main__":
    main()
