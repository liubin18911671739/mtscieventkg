import hashlib
import random
from typing import Dict, List, Tuple, Any


Triple = Tuple[str, str, str, int, str]


def make_event_id(doc_id: str, finding: str) -> str:
    """Stable event identifier from doc id and finding text."""
    h = hashlib.sha1((doc_id + finding).encode("utf-8")).hexdigest()[:10]
    return f"event::{doc_id}::{h}"


def event_to_triples(event: Dict[str, Any]) -> List[Triple]:
    """Convert one extracted event dict into timestamped triples."""
    triples: List[Triple] = []
    ev_id = make_event_id(str(event.get("doc_id")), event.get("finding", ""))
    year = int(event.get("year", 0) or 0)
    lang = event.get("lang", "unk")

    for m in event.get("method", []) or []:
        triples.append((ev_id, "propose", f"method::{m}", year, lang))
    for d in event.get("data_or_material", []) or []:
        triples.append((ev_id, "use", f"data::{d}", year, lang))
    finding = event.get("finding")
    if finding:
        triples.append((ev_id, "support", f"finding::{finding}", year, lang))
    for app in event.get("application", []) or []:
        triples.append((ev_id, "apply", f"application::{app}", year, lang))
    return triples


def generate_diffusion_edges(events: List[Dict[str, Any]], max_edges: int = 400) -> List[Triple]:
    """Create synthetic cite_diffuse/improve edges across events to enable model training."""
    triples: List[Triple] = []
    if len(events) < 2:
        return triples
    pairs = []
    for i, src in enumerate(events):
        for j, tgt in enumerate(events):
            if i == j:
                continue
            year_src, year_tgt = int(src.get("year", 0) or 0), int(tgt.get("year", 0) or 0)
            if year_tgt <= year_src:
                continue
            # quick overlap heuristic
            overlap = set(src.get("method", []) or []) & set(tgt.get("method", []) or [])
            find_overlap = src.get("finding") and tgt.get("finding") and src["finding"] == tgt["finding"]
            if overlap or find_overlap:
                score = len(overlap) + (1 if find_overlap else 0)
                pairs.append((score, src, tgt))
    random.shuffle(pairs)
    pairs = sorted(pairs, key=lambda x: -x[0])[:max_edges]
    for _, src, tgt in pairs:
        sid = make_event_id(str(src.get("doc_id")), src.get("finding", ""))
        tid = make_event_id(str(tgt.get("doc_id")), tgt.get("finding", ""))
        year = int(tgt.get("year", 0) or 0)
        lang = tgt.get("lang", "unk")
        triples.append((sid, "cite_diffuse", tid, year, lang))
        triples.append((sid, "improve", tid, year, lang))
    return triples
