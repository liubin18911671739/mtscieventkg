"""
Normalize OpenAlex/arXiv JSONL into corpus_{lang}.jsonl with key fields.
Generates dummy data automatically when raw files are missing.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any

import yaml
from tqdm import tqdm

from utils.text import maybe_generate_dummy_raw, ensure_dir
from utils.runtime import track_stats

CFG = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
RAW_DIR = Path("data/raw")
CORPUS_DIR = RAW_DIR / "corpus"
ensure_dir(CORPUS_DIR)


def _from_inverted_index(idx: Dict[str, Any]) -> str:
    """Rebuild abstract from OpenAlex inverted index."""
    if not idx:
        return ""
    max_pos = max(p for positions in idx.values() for p in positions)
    tokens = [""] * (max_pos + 1)
    for word, positions in idx.items():
        for p in positions:
            tokens[p] = word
    return " ".join(tokens)


def normalize(work: Dict[str, Any]) -> Dict[str, Any]:
    abstract = work.get("abstract") or _from_inverted_index(work.get("abstract_inverted_index", {}))
    title = work.get("title") or work.get("display_name") or ""
    return {
        "id": work.get("id"),
        "lang": work.get("language"),
        "year": work.get("publication_year"),
        "title": (title or "").strip(),
        "abstract": (abstract or "").strip(),
        "host_venue": (work.get("host_venue") or {}).get("display_name"),
        "authors": [
            a["author"]["display_name"]
            for a in work.get("authorships", [])
            if a.get("author") and a["author"].get("display_name")
        ],
        "institutions": list(
            {
                i["display_name"]
                for a in work.get("authorships", [])
                for i in a.get("institutions", [])
                if i.get("display_name")
            }
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build normalized corpus JSONL per language.")
    parser.add_argument("--languages", nargs="+", default=CFG["languages"])
    parser.add_argument("--force-dummy", action="store_true", help="Generate dummy raw data before processing")
    return parser.parse_args()


def main() -> None:
    with track_stats() as stats:
        args = parse_args()
        if args.force_dummy or not list(RAW_DIR.glob("openalex_*.jsonl")):
            maybe_generate_dummy_raw(RAW_DIR, CFG)

        for lang in args.languages:
            inp = RAW_DIR / f"openalex_{lang}.jsonl"
            out = CORPUS_DIR / f"corpus_{lang}.jsonl"
            if not inp.exists():
                print(f"[warn] missing {inp}, skipping.")
                continue
            with inp.open("r", encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
                for line in tqdm(fin, desc=f"normalize-{lang}"):
                    work = json.loads(line)
                    rec = normalize(work)
                    if rec["title"] and rec["abstract"]:
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"wrote {out}")
    print(f"[stats] {stats.as_dict()}")


if __name__ == "__main__":
    main()
