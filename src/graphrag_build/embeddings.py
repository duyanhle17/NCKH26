"""
Unified embedding module.

Embeds text using SentenceTransformer (works with any HuggingFace model).
Returns numpy ndarray of float32 normalized vectors.
"""
import logging
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("GRAPHRAG_BUILD")


def load_embedder(model_name: str) -> SentenceTransformer:
    """Load a SentenceTransformer embedding model."""
    logger.info(f"Loading embedding model: {model_name}")
    return SentenceTransformer(model_name)


def embed_texts(
    embedder: SentenceTransformer,
    texts: List[str],
    batch_size: int = 16,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Embed a list of texts into normalized float32 vectors.

    Args:
        embedder: SentenceTransformer model
        texts: list of strings to embed
        batch_size: batch size for encoding
        show_progress: show progress bar

    Returns:
        np.ndarray of shape (len(texts), dim), dtype float32, L2-normalized
    """
    if not texts:
        raise ValueError("embed_texts received an empty list.")

    vecs = []
    total = len(texts)
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        v = embedder.encode(
            batch,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vecs.append(np.asarray(v, dtype="float32"))
        if show_progress and (i + batch_size) % (batch_size * 10) == 0:
            logger.info(f"  Embedded {min(i + batch_size, total)}/{total}")

    result = np.vstack(vecs)
    logger.info(f"  Embedding complete: {result.shape[0]} vectors, dim={result.shape[1]}")
    return result
