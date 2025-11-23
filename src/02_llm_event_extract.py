"""
Extract scientific events (problem, method, data_or_material, finding, application, year)
from corpus JSONL. Supports Zhipu QingYan (智普清言), OpenAI API, or a lightweight local heuristic extractor.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import List, Dict, Any

import requests
import yaml
from tqdm import tqdm

from utils.prompt_templates import build_prompt
from utils.text import fix_json_list, ensure_dir, maybe_generate_dummy_raw
from utils.runtime import track_stats

CFG = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
CORPUS_DIR = Path("data/raw/corpus")
OUT_DIR = Path("data/extracted")
ensure_dir(OUT_DIR)


def call_openai(prompt: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=CFG["llm"]["model"],
        temperature=CFG["llm"]["temperature"],
        max_tokens=CFG["llm"]["max_tokens"],
        top_p=CFG["llm"].get("top_p", 1.0),
        messages=[
            {"role": "system", "content": CFG["llm"].get("system_prompt", "")},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content


def call_zhipu(prompt: str) -> str:
    api_key = os.environ.get(CFG["llm"].get("api_key_env", "ZHIPUAI_API_KEY"))
    if not api_key:
        raise RuntimeError("ZHIPUAI_API_KEY not set")
    payload = {
        "model": CFG["llm"]["model"],
        "messages": [
            {"role": "system", "content": CFG["llm"].get("system_prompt", "")},
            {"role": "user", "content": prompt},
        ],
        "temperature": CFG["llm"]["temperature"],
        "top_p": CFG["llm"].get("top_p", 0.9),
        "max_tokens": CFG["llm"]["max_tokens"],
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.post(CFG["zhipu"]["api_base"], headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        return json.dumps(data, ensure_ascii=False)
    message = choices[0].get("message") or {}
    content = message.get("content") or choices[0].get("content") or ""
    if isinstance(content, list):  # streaming-like parts
        content = "".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in content])
    return content


def heuristic_extract(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Local offline extractor that uses simple keyword heuristics."""
    title = doc.get("title", "")
    abstract = doc.get("abstract", "")
    text = f"{title}. {abstract}".lower()
    methods = list({m for m in re.findall(r"(model|network|algorithm|framework)", text)})
    data = list({m for m in re.findall(r"(dataset|data|corpus|simulation)", text)})
    apps = list({m for m in re.findall(r"(application|robotics|biology|materials?)", text)})
    finding = " ".join(title.split()[:12]) or "key finding"
    return [
        {
            "problem": title[:200],
            "method": methods or ["baseline method"],
            "data_or_material": data or ["synthetic data"],
            "finding": finding,
            "application": apps or ["general science"],
            "year": doc.get("year"),
            "confidence": 0.5,
        }
    ]


def extract_events(doc: Dict[str, Any], provider: str) -> List[Dict[str, Any]]:
    if provider in {"openai", "zhipu"}:
        prompt = build_prompt(doc.get("title", ""), doc.get("abstract", ""), doc.get("lang", ""), doc.get("year", ""))
        try:
            raw = call_openai(prompt) if provider == "openai" else call_zhipu(prompt)
        except Exception as exc:
            print(f"[warn] {provider} call failed: {exc}. Falling back to heuristic.")
            return heuristic_extract(doc)
        parsed = fix_json_list(raw)
        if not parsed:
            parsed = heuristic_extract(doc)
        return parsed
    return heuristic_extract(doc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM-based scientific event extraction.")
    parser.add_argument("--provider", choices=["zhipu", "openai", "local"], default=CFG["llm"]["provider"])
    parser.add_argument("--languages", nargs="+", default=CFG["languages"])
    parser.add_argument("--limit", type=int, default=None, help="Limit number of papers per language")
    parser.add_argument("--force-dummy", action="store_true", help="Generate dummy corpus if missing")
    return parser.parse_args()


def main() -> None:
    with track_stats() as stats:
        args = parse_args()
        if args.force_dummy and not list(CORPUS_DIR.glob("corpus_*.jsonl")):
            maybe_generate_dummy_raw(Path("data/raw"), CFG)
            os.system("python src/01_build_corpus.py")

        for lang in args.languages:
            inp = CORPUS_DIR / f"corpus_{lang}.jsonl"
            out = OUT_DIR / f"events_{lang}.jsonl"
            if not inp.exists():
                print(f"[warn] missing {inp}, skipping.")
                continue
            with inp.open("r", encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
                for idx, line in enumerate(tqdm(fin, desc=f"extract-{lang}")):
                    if args.limit and idx >= args.limit:
                        break
                    doc = json.loads(line)
                    events = extract_events(doc, args.provider)
                    for ev in events:
                        ev["doc_id"] = doc.get("id")
                        ev["lang"] = doc.get("lang", lang)
                        ev["year"] = ev.get("year") or doc.get("year")
                        fout.write(json.dumps(ev, ensure_ascii=False) + "\n")
            print(f"wrote {out}")
    print(f"[stats] {stats.as_dict()}")


if __name__ == "__main__":
    main()
