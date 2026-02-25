from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

@dataclass
class BuildConfig:
    # ── Dataset ──────────────────────────────────────────────────────────
    dataset_dir: Path = Path("./dataset")

    # ── Output directories ──────────────────────────────────────────────
    work_dir: Path = Path("./work")
    cache_dir: Path = Path("./artifacts/graphrag")

    # ── Embedding model ─────────────────────────────────────────────────
    # Snowflake Arctic Embed M — multilingual, works well for Vietnamese
    embed_model: str = "Snowflake/snowflake-arctic-embed-m"
    batch_embed: int = 16

    # ── Vector storage backend ──────────────────────────────────────────
    # Choose one: "faiss" | "chromadb"
    # (uncomment "milvus" or "zvec" in vector_store.py to enable those)
    vector_backend: str = "chromadb"

    # Milvus-specific (only used when vector_backend="milvus")
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "graphrag_chunks"

    # ── Chunking parameters ─────────────────────────────────────────────
    max_token_size: int = 512
    overlap_token_size: int = 64
    min_chunk_chars: int = 120

    # ── KG extraction parameters ────────────────────────────────────────
    top_k_terms_per_chunk: int = 24
    min_term_len: int = 3
    max_term_words: int = 10
    cooc_window: int = 2
    prune_min_cooc_weight: int = 2