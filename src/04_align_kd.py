"""
Cross-lingual entity alignment using multilingual embeddings + nearest neighbors.
Outputs align_pairs.tsv for KD.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from utils.embedding import encode_texts
from utils.text import ensure_dir
from utils.runtime import track_stats
from utils.metrics import precision_recall_f1

CFG = yaml.safe_load(open("config.yaml", "r", encoding="utf-8"))
TRIPLE_PATH = Path("data/kg/triples.tsv")
OUT_DIR = Path("data/kg")
ensure_dir(OUT_DIR)


def load_entities(triple_path: Path) -> Tuple[pd.DataFrame, List[str], Dict[str, List[str]]]:
    df = pd.read_csv(triple_path, sep="\t", header=None, names=["s", "r", "o", "t", "lang"])
    entities = pd.Index(pd.concat([df["s"], df["o"]]).unique())
    ent_langs: Dict[str, List[str]] = {e: [] for e in entities}
    for _, row in df.iterrows():
        ent_langs[row["s"]].append(row["lang"])
        ent_langs[row["o"]].append(row["lang"])
    ent_langs = {k: list(set(v)) for k, v in ent_langs.items()}
    return df, list(entities), ent_langs


def build_alignment(
    entities: List[str],
    ent_langs: Dict[str, List[str]],
    embeddings: np.ndarray,
    threshold: float,
    knn: int,
) -> List[Tuple[str, str, float]]:
    pairs = []
    en_idx = [i for i, e in enumerate(entities) if "en" in ent_langs.get(e, [])]
    if not en_idx:
        return pairs
    en_emb = embeddings[en_idx]
    en_norm = en_emb / (np.linalg.norm(en_emb, axis=1, keepdims=True) + 1e-9)
    for lang in CFG["languages"]:
        if lang == "en":
            continue
        tgt_idx = [i for i, e in enumerate(entities) if lang in ent_langs.get(e, []) and "en" not in ent_langs.get(e, [])]
        if not tgt_idx:
            continue
        tgt_emb = embeddings[tgt_idx]
        tgt_norm = tgt_emb / (np.linalg.norm(tgt_emb, axis=1, keepdims=True) + 1e-9)
        sim = tgt_norm @ en_norm.T
        for row, i_ent in zip(sim, tgt_idx):
            topk_idx = np.argsort(row)[::-1][:knn]
            for j in topk_idx:
                score = float(row[j])
                if score >= threshold:
                    pairs.append((entities[i_ent], entities[en_idx[j]], score))
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-lingual entity alignment via multilingual embeddings.")
    parser.add_argument("--model", default=CFG["embedding"]["model_name"], help="Embedding model name")
    parser.add_argument("--threshold", type=float, default=CFG["embedding"]["align_threshold"])
    parser.add_argument("--knn", type=int, default=CFG["embedding"]["knn"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--eval-file", type=str, default=None, help="Optional TSV with columns tgt,src,label for alignment eval")
    return parser.parse_args()


def main() -> None:
    with track_stats() as stats:
        args = parse_args()
        if not TRIPLE_PATH.exists():
            raise FileNotFoundError("Run 03_build_tkg.py first.")

        df, entities, ent_langs = load_entities(TRIPLE_PATH)
        texts = [e.split("::", 1)[-1] for e in entities]
        embeddings = encode_texts(
            texts,
            args.model,
            batch_size=CFG["embedding"]["batch_size"],
            normalize=CFG["embedding"]["normalize"],
            device=args.device,
        )
        pairs = build_alignment(entities, ent_langs, embeddings, args.threshold, args.knn)
        out_path = OUT_DIR / "align_pairs.tsv"
        pd.DataFrame(pairs, columns=["tgt", "src", "score"]).to_csv(out_path, sep="\t", index=False)
        print(f"wrote {out_path} with {len(pairs)} pairs using {args.model}")

        if args.eval_file:
            gold = pd.read_csv(args.eval_file, sep="\t")
            pred_set = {(t, s) for t, s, _ in pairs}
            gold_pos = {(r.tgt, r.src) for r in gold.itertuples() if getattr(r, "label", 1) == 1}
            metrics = precision_recall_f1(pred_set, gold_pos)
            print(f"alignment eval: {metrics}")
    print(f"[stats] {stats.as_dict()}")


if __name__ == "__main__":
    main()
