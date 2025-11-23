import json
import random
import string
from pathlib import Path
from typing import Iterable, List, Dict, Any

try:
    import ujson as _json
except ImportError:  # pragma: no cover - fallback
    _json = json


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield _json.loads(line)


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(_json.dumps(rec, ensure_ascii=False) + "\n")


def fix_json_list(text: str) -> List[Dict[str, Any]]:
    """Best-effort JSON list parser for slightly malformed outputs."""
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, list) else []
    except Exception:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                return []
    return []


def _rand_words(n: int, lang: str) -> str:
    vocab = ["quantum", "graph", "neural", "diffusion", "model", "data", "science", "learning", "knowledge", "event"]
    words = [random.choice(vocab) for _ in range(n)]
    return f"[{lang}] " + " ".join(words)


def generate_dummy_papers(
    num: int,
    langs: List[str],
    time_range: List[int],
    concepts: List[str],
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """Create synthetic OpenAlex-like metadata per language."""
    random.seed(seed)
    start, end = time_range
    dummy = {l: [] for l in langs}
    for lang in langs:
        for idx in range(num):
            year = random.randint(start, end)
            concept = random.choice(concepts)
            paper_id = f"W{lang}{year}{idx:05d}"
            title = _rand_words(6, lang) + f" on {concept}"
            abstract = _rand_words(40, lang)
            authors = [f"{lang.upper()} Author {i}" for i in range(1, 4)]
            insts = [f"{lang.upper()} Institute {random.randint(1,5)}"]
            dummy[lang].append(
                {
                    "id": paper_id,
                    "display_name": title,
                    "title": title,
                    "abstract": abstract,
                    "language": lang,
                    "publication_year": year,
                    "host_venue": {"display_name": f"{concept} Journal"},
                    "concepts": [{"display_name": concept}],
                    "authorships": [
                        {
                            "author": {"display_name": a},
                            "institutions": [{"display_name": inst} for inst in insts],
                        }
                        for a in authors
                    ],
                }
            )
    return dummy


def maybe_generate_dummy_raw(raw_dir: Path, cfg: Dict[str, Any]) -> List[Path]:
    """Generate dummy raw OpenAlex-like files when missing."""
    ensure_dir(raw_dir)
    existing = list(raw_dir.glob("openalex_*.jsonl"))
    if existing:
        return existing

    dummy = generate_dummy_papers(
        num=cfg["data"]["dummy_records"],
        langs=cfg["languages"],
        time_range=cfg["time_range"],
        concepts=cfg["domain_concepts"],
        seed=cfg["data"].get("seed", 42),
    )
    out_paths = []
    for lang, records in dummy.items():
        path = raw_dir / f"openalex_{lang}.jsonl"
        write_jsonl(path, records)
        out_paths.append(path)
    return out_paths
