"""
Download OpenAlex works by concept and language, or generate dummy data when offline.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Any

import requests
import yaml
from tqdm import tqdm

from utils.text import maybe_generate_dummy_raw, ensure_dir
from utils.runtime import track_stats

CFG = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
RAW_DIR = Path("data/raw")
ensure_dir(RAW_DIR)


def fetch_openalex(concept: str, year_from: int, year_to: int, lang: str, max_pages: int) -> Any:
    cursor = "*"
    page = 0
    while True:
        if page >= max_pages:
            break
        params = {
            "filter": f"concepts.display_name.search:{concept},from_publication_date:{year_from}-01-01,to_publication_date:{year_to}-12-31,language:{lang}",
            "per-page": CFG["openalex"]["per_page"],
            "cursor": cursor,
        }
        resp = requests.get(CFG["openalex"]["api_base"], params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        for w in payload.get("results", []):
            yield w
        cursor = payload.get("meta", {}).get("next_cursor")
        page += 1
        if not cursor:
            break
        time.sleep(CFG["openalex"].get("sleep", 0.5))


def download_one_lang(lang: str, args: argparse.Namespace) -> Path:
    out_file = RAW_DIR / f"openalex_{lang}.jsonl"
    if out_file.exists() and not args.overwrite:
        return out_file
    year_from, year_to = args.year_from, args.year_to
    total = 0
    with out_file.open("w", encoding="utf-8") as f:
        for concept in args.concepts:
            for w in tqdm(
                fetch_openalex(concept, year_from, year_to, lang, args.max_pages),
                desc=f"{concept}-{lang}",
            ):
                f.write(json.dumps(w, ensure_ascii=False) + "\n")
                total += 1
    if total == 0:
        out_file.unlink(missing_ok=True)
    return out_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download OpenAlex metadata by concept/language.")
    parser.add_argument("--concept", dest="concepts", action="append", default=None, help="Concept keywords")
    parser.add_argument("--year-from", type=int, default=CFG["time_range"][0])
    parser.add_argument("--year-to", type=int, default=CFG["time_range"][1])
    parser.add_argument("--languages", nargs="+", default=CFG["languages"])
    parser.add_argument("--max-pages", type=int, default=CFG["openalex"]["max_pages"])
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--force-dummy", action="store_true", help="Skip download and generate dummy data")
    return parser.parse_args()


def main() -> None:
    with track_stats() as stats:
        args = parse_args()
        if args.concepts is None:
            args.concepts = CFG["domain_concepts"]
        if args.force_dummy:
            maybe_generate_dummy_raw(RAW_DIR, CFG)
            print("Dummy raw data generated.")
            print(f"[stats] {stats.as_dict()}")
            return

        created = []
        for lang in args.languages:
            try:
                created.append(download_one_lang(lang, args))
            except Exception as exc:  # network fallback
                print(f"[warn] download failed for {lang}: {exc}. Generating dummy instead.")
        remaining = [p for p in created if p is not None and p.exists()]
        if not remaining:
            maybe_generate_dummy_raw(RAW_DIR, CFG)
            print("No data downloaded; dummy raw data generated.")
        else:
            print(f"Saved: {remaining}")
    print(f"[stats] {stats.as_dict()}")


if __name__ == "__main__":
    main()
