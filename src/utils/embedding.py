import math
import random
from typing import Iterable, List, Optional

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None


def load_model(model_name: str, device: str = "cpu") -> Optional["SentenceTransformer"]:
    if SentenceTransformer is None:
        return None
    try:
        return SentenceTransformer(model_name, device=device)
    except Exception:
        return None


def encode_texts(
    texts: List[str],
    model_name: str,
    batch_size: int = 64,
    normalize: bool = True,
    device: str = "cpu",
) -> np.ndarray:
    """Encode texts with SentenceTransformer or return random unit vectors as a fallback."""
    model = load_model(model_name, device=device)
    if model is None:
        dim = 384
        rng = np.random.default_rng(0)
        emb = rng.standard_normal((len(texts), dim))
    else:
        emb = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=False,
            device=device,
        )
    if normalize:
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
        emb = emb / norms
    return np.asarray(emb)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b.T
