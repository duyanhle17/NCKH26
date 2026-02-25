from dataclasses import dataclass
from pathlib import Path

@dataclass
class BuildConfig:
    # ✅ NEW: dataset folder (multiple .txt)
    dataset_dir: Path = Path("./dataset")

    # outputs
    work_dir: Path = Path("./work")
    cache_dir: Path = Path("./artifacts/graphrag_bge")

    embed_model: str = "BAAI/bge-large-en-v1.5"
    batch_embed: int = 16

    max_token_size: int = 750
    overlap_token_size: int = 80
    min_chunk_chars: int = 120

    top_k_terms_per_chunk: int = 24
    min_term_len: int = 3
    max_term_words: int = 10
    cooc_window: int = 2
    prune_min_cooc_weight: int = 2