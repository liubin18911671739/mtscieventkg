"""
Convert extracted events into timestamped triples and synthetic diffusion edges.
Outputs data/kg/triples.tsv.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import yaml
from tqdm import tqdm

from utils.kg import event_to_triples, generate_diffusion_edges
from utils.text import ensure_dir
from utils.runtime import track_stats

CFG = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
EVENT_DIR = Path("data/extracted")
OUT_DIR = Path("data/kg")
ensure_dir(OUT_DIR)


def load_events(lang_files: List[Path]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for path in lang_files:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))
    return events


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build temporal knowledge graph triples.")
    parser.add_argument("--languages", nargs="+", default=CFG["languages"])
    parser.add_argument("--max-diffusion", type=int, default=400, help="Max synthetic diffusion edges")
    return parser.parse_args()


def main() -> None:
    with track_stats() as stats:
        args = parse_args()
        lang_files = [EVENT_DIR / f"events_{l}.jsonl" for l in args.languages]
        if not any(p.exists() for p in lang_files):
            print("[warn] no events found. Running heuristic extractor on dummy corpus.")
            # run extractor pipeline quickly
            import os

            os.system("python src/01_build_corpus.py --force-dummy")
            os.system("python src/02_llm_event_extract.py --provider local")

        lang_files = [p for p in lang_files if p.exists()]
        events = load_events(lang_files)
        triples = []
        for ev in tqdm(events, desc="events->triples"):
            triples.extend(event_to_triples(ev))
        # diffusion edges
        triples.extend(generate_diffusion_edges(events, max_edges=args.max_diffusion))

        out_path = OUT_DIR / "triples.tsv"
        with out_path.open("w", encoding="utf-8") as f:
            for s, r, o, t, lang in triples:
                f.write(f"{s}\t{r}\t{o}\t{t}\t{lang}\n")
        print(f"wrote {out_path} with {len(triples)} triples")
    print(f"[stats] {stats.as_dict()}")


if __name__ == "__main__":
    main()
