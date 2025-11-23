"""
Detect frontier events using burst_z, structural novelty, and cross-lingual bridge centrality.
Outputs annual frontier lists and trend predictions.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import yaml

from utils.metrics import precision_recall_f1, language_link_index, community_bridge_centrality, dfa_metrics
from utils.runtime import track_stats

CFG = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
TRIPLE_PATH = Path("data/kg/triples.tsv")


def burst_scores(df: pd.DataFrame) -> Dict[int, float]:
    yearly = df.groupby("t").size()
    mu, sigma = yearly.mean(), yearly.std() + 1e-9
    return {int(k): float((v - mu) / sigma) for k, v in yearly.items()}


def structural_novelty(df: pd.DataFrame) -> Dict[str, float]:
    """Novelty = fraction of new neighbors for event nodes each year."""
    novelty = defaultdict(float)
    seen_neighbors: Dict[str, set] = defaultdict(set)
    for _, row in df.sort_values("t").iterrows():
        s, o, t = row["s"], row["o"], row["t"]
        if s.startswith("event::"):
            prev = seen_neighbors[s]
            new_neighbor = o not in prev
            novelty_key = f"{s}@{t}"
            novelty[novelty_key] += 1.0 if new_neighbor else 0.0
            prev.add(o)
        if o.startswith("event::"):
            prev = seen_neighbors[o]
            new_neighbor = s not in prev
            novelty_key = f"{o}@{t}"
            novelty[novelty_key] += 1.0 if new_neighbor else 0.0
            prev.add(s)
    return novelty


def compute_cbc(df: pd.DataFrame) -> Dict[str, float]:
    G = nx.from_pandas_edgelist(df, "s", "o", edge_attr=True, create_using=nx.Graph)
    bc = nx.betweenness_centrality(G, k=min(500, len(G.nodes())), seed=42)
    return bc


def frontier_scores(df: pd.DataFrame) -> List[Tuple[str, float, int]]:
    burst = burst_scores(df)
    novelty = structural_novelty(df)
    bc = compute_cbc(df)
    frontiers = []
    for key, nov in novelty.items():
        ev, year = key.split("@")
        year = int(year)
        score = 0.5 * nov + 0.3 * burst.get(year, 0.0) + 0.2 * bc.get(ev, 0.0)
        frontiers.append((ev, score, year))
    frontiers.sort(key=lambda x: (-x[1], x[2]))
    return frontiers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frontier detection from SciEventKG.")
    parser.add_argument("--topk", type=int, default=10, help="Top K events per year")
    parser.add_argument("--output", default="data/kg/frontiers.json")
    parser.add_argument("--gold-frontier", default=None, help="Optional TSV with columns year,event for eval")
    return parser.parse_args()


def main() -> None:
    with track_stats() as stats:
        args = parse_args()
        if not TRIPLE_PATH.exists():
            raise FileNotFoundError("Run 03_build_tkg.py first.")
        df = pd.read_csv(TRIPLE_PATH, sep="\t", header=None, names=["s", "r", "o", "t", "lang"])
        frontiers = frontier_scores(df)

        per_year: Dict[int, List[Tuple[str, float]]] = defaultdict(list)
        for ev, score, year in frontiers:
            per_year[year].append((ev, score))
        per_year = {y: sorted(v, key=lambda x: -x[1])[: args.topk] for y, v in per_year.items()}

        # diffusion forecasting mock: use scores as predictions
        all_scores = [s for year_vals in per_year.values() for _, s in year_vals]
        all_labels = [1] * len(all_scores)
        dfa = dfa_metrics(all_scores, all_labels, CFG["eval"]["hits_k"])

        metrics = {"dfa": dfa}
        if args.gold_frontier and Path(args.gold_frontier).exists():
            gold_df = pd.read_csv(args.gold_frontier, sep="\t")
            gold_pairs = {(int(r.year), r.event) for r in gold_df.itertuples()}
            pred_pairs = {(y, ev) for y, vals in per_year.items() for ev, _ in vals}
            metrics["frontier_prf"] = precision_recall_f1(pred_pairs, gold_pairs)

        lli_edges = [(row.s.split("::")[-1], row.o.split("::")[-1], row.r) for row in df.itertuples()]
        metrics["lli"] = language_link_index(lli_edges)
        metrics["cbc_mean"] = float(np.mean(list(community_bridge_centrality(df[["s", "o"]].values.tolist()).values()) or [0]))

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump({"frontiers": per_year, "metrics": metrics}, f, ensure_ascii=False, indent=2)

        print(f"Frontiers saved to {out_path}")
        for y in sorted(per_year.keys()):
            print(f"Year {y}: {[ev for ev, _ in per_year[y]]}")
        print(f"Metrics: {metrics}")
    print(f"[stats] {stats.as_dict()}")


if __name__ == "__main__":
    main()
