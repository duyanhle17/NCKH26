#!/usr/bin/env python3
"""
GraphRAG Build CLI — unified entry point for the build pipeline.

Usage:
    python build.py                                         # defaults: snowflake + faiss
    python build.py --backend chromadb                      # use ChromaDB
    python build.py --backend zvec                          # use ZVec (macOS/Linux)
    python build.py --backend milvus --milvus-uri http://localhost:19530
    python build.py --model intfloat/multilingual-e5-large  # custom model
    python build.py --help                                  # show all options

Backend compatibility:
    ┌────────────┬─────────────┬────────────────┬──────────────────────────┐
    │ Backend    │ Platform    │ Python         │ Install                  │
    ├────────────┼─────────────┼────────────────┼──────────────────────────┤
    │ faiss      │ All         │ 3.8+           │ pip install faiss-cpu    │
    │ chromadb   │ All         │ 3.9 - 3.13     │ pip install chromadb     │
    │ milvus     │ All (client)│ 3.8+           │ pip install pymilvus     │
    │ zvec       │ Linux/macOS │ 3.10 - 3.12    │ pip install zvec         │
    └────────────┴─────────────┴────────────────┴──────────────────────────┘
"""
import sys
import argparse
from pathlib import Path

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    parser = argparse.ArgumentParser(
        description="GraphRAG Build Pipeline — build embeddings + KG from legal documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Embedding model ──────────────────────────────────────────────────
    parser.add_argument(
        "--model", "-m",
        default="Snowflake/snowflake-arctic-embed-m",
        help="HuggingFace embedding model name (default: Snowflake/snowflake-arctic-embed-m)",
    )

    # ── Vector backend ───────────────────────────────────────────────────
    parser.add_argument(
        "--backend", "-b",
        default="faiss",
        choices=["faiss", "chromadb", "milvus", "zvec"],
        help="Vector database backend (default: faiss)",
    )

    # ── Directories ──────────────────────────────────────────────────────
    parser.add_argument(
        "--dataset", "-d",
        default="./dataset",
        help="Path to dataset directory containing .txt files (default: ./dataset)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output artifacts directory (default: ./artifacts/graphrag_{backend})",
    )

    # ── Chunking parameters ──────────────────────────────────────────────
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens per chunk (default: 512)")
    parser.add_argument("--overlap-tokens", type=int, default=64, help="Overlap tokens between chunks (default: 64)")
    parser.add_argument("--min-chunk-chars", type=int, default=120, help="Min chars per chunk (default: 120)")
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size (default: 16)")

    # ── Milvus-specific ──────────────────────────────────────────────────
    parser.add_argument("--milvus-uri", default="http://localhost:19530", help="Milvus server URI")
    parser.add_argument("--milvus-collection", default="graphrag_chunks", help="Milvus collection name prefix")

    args = parser.parse_args()

    # ── Validate backend availability ────────────────────────────────────
    backend = args.backend
    _check_backend(backend)

    # ── Enable backend in vector_store if needed ─────────────────────────
    _ensure_backend_registered(backend)

    # ── Build config ─────────────────────────────────────────────────────
    from graphrag_build.config import BuildConfig

    output_dir = args.output or f"./artifact_{backend}"

    config = BuildConfig(
        dataset_dir=Path(args.dataset),
        cache_dir=Path(output_dir),
        embed_model=args.model,
        vector_backend=backend,
        batch_embed=args.batch_size,
        max_token_size=args.max_tokens,
        overlap_token_size=args.overlap_tokens,
        min_chunk_chars=args.min_chunk_chars,
        milvus_uri=args.milvus_uri,
        milvus_collection=args.milvus_collection,
    )

    # ── Print config ─────────────────────────────────────────────────────
    print("=" * 60)
    print("  GraphRAG Build Pipeline")
    print("=" * 60)
    print(f"  Embedding model : {config.embed_model}")
    print(f"  Vector backend  : {config.vector_backend}")
    print(f"  Dataset         : {config.dataset_dir}")
    print(f"  Output          : {config.cache_dir}")
    print(f"  Max tokens      : {config.max_token_size}")
    print(f"  Overlap tokens  : {config.overlap_token_size}")
    print(f"  Min chunk chars : {config.min_chunk_chars}")
    print("=" * 60)

    # ── Run pipeline ─────────────────────────────────────────────────────
    from graphrag_build.pipeline import run_build
    run_build(config)


def _check_backend(backend: str):
    """Check if the required packages for the backend are installed."""
    checks = {
        "faiss": ("faiss", "pip install faiss-cpu"),
        "chromadb": ("chromadb", "pip install chromadb"),
        "milvus": ("pymilvus", "pip install pymilvus"),
        "zvec": ("zvec", "pip install zvec"),
    }
    pkg, install_cmd = checks[backend]
    try:
        __import__(pkg)
    except ImportError:
        print(f"\n❌ Backend '{backend}' requires package '{pkg}'")
        print(f"   Install: {install_cmd}")
        print(f"   Or install from requirements: pip install -r requirements/{backend}.txt")

        # Platform-specific hints
        import platform
        if backend == "zvec" and platform.system() == "Windows":
            print(f"\n   ⚠️  ZVec does NOT support Windows.")
            print(f"   Use WSL (Ubuntu) or switch to macOS/Linux.")
        if backend == "chromadb":
            v = sys.version_info
            if v >= (3, 14):
                print(f"\n   ⚠️  ChromaDB does not support Python {v.major}.{v.minor} yet.")
                print(f"   Use Python 3.13 or earlier (e.g. .venv313).")
        sys.exit(1)


def _ensure_backend_registered(backend: str):
    """
    Dynamically register the backend if it's commented out in _BACKENDS.
    This allows milvus/zvec to work without editing vector_store.py.
    """
    from graphrag_build.vector_store import _BACKENDS

    if backend in _BACKENDS:
        return  # already registered

    # Dynamically import and register
    if backend == "milvus":
        from graphrag_build.vector_store import MilvusVectorStore
        _BACKENDS["milvus"] = MilvusVectorStore
        print(f"  ℹ️  Dynamically enabled '{backend}' backend")

    elif backend == "zvec":
        from graphrag_build.vector_store import ZVecVectorStore
        _BACKENDS["zvec"] = ZVecVectorStore
        print(f"  ℹ️  Dynamically enabled '{backend}' backend")

    elif backend == "chromadb":
        from graphrag_build.vector_store import ChromaDBVectorStore
        _BACKENDS["chromadb"] = ChromaDBVectorStore
        print(f"  ℹ️  Dynamically enabled '{backend}' backend")


if __name__ == "__main__":
    main()
