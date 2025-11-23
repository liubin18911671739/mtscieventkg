from collections import Counter
from typing import Iterable, List, Dict, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score


def precision_recall_f1(pred: Iterable[str], gold: Iterable[str]) -> Dict[str, float]:
    pset, gset = set(pred), set(gold)
    tp = len(pset & gset)
    precision = tp / (len(pset) + 1e-9)
    recall = tp / (len(gset) + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return {"precision": precision, "recall": recall, "f1": f1}


def hits_at_k(scores: List[float], labels: List[int], ks: List[int]) -> Dict[int, float]:
    order = np.argsort(scores)[::-1]
    labels = np.asarray(labels)[order]
    out = {}
    for k in ks:
        topk = labels[:k] if k <= len(labels) else labels
        out[k] = float(topk.sum() > 0) if len(topk) else 0.0
    return out


def mean_reciprocal_rank(scores: List[float], labels: List[int]) -> float:
    order = np.argsort(scores)[::-1]
    labels = np.asarray(labels)[order]
    for idx, lab in enumerate(labels, start=1):
        if lab == 1:
            return 1.0 / idx
    return 0.0


def dfa_metrics(scores: List[float], labels: List[int], ks: List[int]) -> Dict[str, float]:
    """Diffusion forecasting accuracy metrics."""
    metrics = {}
    try:
        metrics["auc"] = roc_auc_score(labels, scores)
    except Exception:
        metrics["auc"] = 0.0
    metrics["mrr"] = mean_reciprocal_rank(scores, labels)
    for k, v in hits_at_k(scores, labels, ks).items():
        metrics[f"hits@{k}"] = v
    return metrics


def language_link_index(edges: List[Tuple[str, str, str]]) -> float:
    """
    LLI: ratio of cross-language edges in the graph.
    edges: list of (lang_u, lang_v, relation)
    """
    if not edges:
        return 0.0
    cross = sum(1 for u, v, _ in edges if u != v)
    return cross / len(edges)


def community_bridge_centrality(edge_index: List[Tuple[str, str]]) -> Dict[str, float]:
    """Simple CBC: betweenness-like centrality from edge counts."""
    deg = Counter()
    for u, v in edge_index:
        deg[u] += 1
        deg[v] += 1
    total = sum(deg.values()) + 1e-9
    return {n: d / total for n, d in deg.items()}
