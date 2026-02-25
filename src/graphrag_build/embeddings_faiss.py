from typing import List, Optional, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def embed_passages(embedder: SentenceTransformer, texts: List[str], batch_size: int) -> np.ndarray:
    if not texts:
        raise ValueError("embed_passages got empty list.")
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        v = embedder.encode(batch, normalize_embeddings=True)
        vecs.append(np.asarray(v, dtype="float32"))
    return np.vstack(vecs)

def build_faiss_ip(embeddings: np.ndarray) -> faiss.Index:
    d = embeddings.shape[1]
    idx = faiss.IndexFlatIP(d)
    idx.add(embeddings.astype("float32"))
    return idx