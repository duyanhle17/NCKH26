"""
I/O helpers for saving pipeline artifacts.
No hard dependency on any specific vector backend.
"""
import json, pickle
from datetime import datetime
from pathlib import Path
import numpy as np


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, obj) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def save_json_compact(path: Path, obj) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")


def save_pickle(path: Path, obj) -> None:
    ensure_dir(path.parent)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def save_npy(path: Path, arr: np.ndarray) -> None:
    ensure_dir(path.parent)
    np.save(str(path), arr)


def build_meta(
    config,
    passages_n: int,
    chunks_n: int,
    kg_nodes: int,
    kg_edges: int,
    entities_n: int,
):
    return {
        "created_at": datetime.now().isoformat(),
        "build_mode": (
            f"TXT→PASSAGES(legal-split)→CHUNK(token)→"
            f"{config.embed_model}→{config.vector_backend.upper()}"
            f"(chunks+entities)→KG"
        ),
        "dataset_dir": str(config.dataset_dir),
        "cache_dir": str(config.cache_dir),
        "embedding_model": config.embed_model,
        "vector_backend": config.vector_backend,
        "chunking": {
            "max_token_size": config.max_token_size,
            "overlap_token_size": config.overlap_token_size,
            "min_chunk_chars": config.min_chunk_chars,
        },
        "counts": {
            "passages": passages_n,
            "chunks": chunks_n,
            "kg_nodes": kg_nodes,
            "kg_edges": kg_edges,
            "entities": entities_n,
        },
    }