"""
Vector store abstraction layer.

Provides a unified interface for storing and querying vector embeddings
across multiple backends:
  - faiss    : Local, in-process (no server needed)
  - chromadb : Embedded, persistent, cross-platform (recommended for Windows)
  - milvus   : Distributed, server-based (for production scale)
  - zvec     : Embedded, lightweight (Alibaba — Linux/macOS only)

Usage:
    store = create_vector_store("chromadb", save_dir=Path("./artifacts"))
    store.add("chunks", embeddings, metadata_list)
    store.save()

    # Later, for query:
    store = create_vector_store("chromadb", save_dir=Path("./artifacts"))
    store.load()
    distances, indices = store.search("chunks", query_vector, top_k=5)
"""
import logging
import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("GRAPHRAG_BUILD")


class VectorStore(ABC):
    """Abstract base class for vector storage backends."""

    @abstractmethod
    def add(self, collection: str, embeddings: np.ndarray,
            metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """Add vectors to a named collection."""
        ...

    @abstractmethod
    def search(self, collection: str, query: np.ndarray,
               top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors.

        Args:
            collection: name of the collection to search
            query: query vector(s), shape (n_queries, dim)
            top_k: number of results per query

        Returns:
            (distances, indices) — both shape (n_queries, top_k)
        """
        ...

    @abstractmethod
    def save(self) -> None:
        """Persist the store to disk."""
        ...

    @abstractmethod
    def load(self) -> None:
        """Load the store from disk."""
        ...

    @abstractmethod
    def collection_size(self, collection: str) -> int:
        """Return the number of vectors in a collection."""
        ...


# ═══════════════════════════════════════════════════════════════════════════
# FAISS Backend
# ═══════════════════════════════════════════════════════════════════════════

class FaissVectorStore(VectorStore):
    """
    FAISS-based vector store — local, in-process, no server needed.
    Uses IndexFlatIP (inner product = cosine similarity for normalized vectors).
    """

    def __init__(self, save_dir: Path, **kwargs):
        import faiss  # lazy import
        self._faiss = faiss
        self.save_dir = save_dir
        self._indices: Dict[str, Any] = {}
        self._embeddings: Dict[str, np.ndarray] = {}

    def add(self, collection: str, embeddings: np.ndarray,
            metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        emb = embeddings.astype("float32")
        dim = emb.shape[1]
        idx = self._faiss.IndexFlatIP(dim)
        idx.add(emb)
        self._indices[collection] = idx
        self._embeddings[collection] = emb
        logger.info(f"[FAISS] Added {emb.shape[0]} vectors to '{collection}' (dim={dim})")

    def search(self, collection: str, query: np.ndarray,
               top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        idx = self._indices.get(collection)
        if idx is None:
            raise KeyError(f"Collection '{collection}' not found in FAISS store")
        q = query.astype("float32")
        if q.ndim == 1:
            q = q.reshape(1, -1)
        distances, indices = idx.search(q, top_k)
        return distances, indices

    def save(self) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        for name, idx in self._indices.items():
            idx_path = self.save_dir / f"faiss_{name}.index"
            self._faiss.write_index(idx, str(idx_path))
            logger.info(f"[FAISS] Saved index → {idx_path}")
        for name, emb in self._embeddings.items():
            emb_path = self.save_dir / f"embeddings_{name}.npy"
            np.save(str(emb_path), emb)
            logger.info(f"[FAISS] Saved embeddings → {emb_path}")

    def load(self) -> None:
        for idx_path in self.save_dir.glob("faiss_*.index"):
            name = idx_path.stem.replace("faiss_", "")
            self._indices[name] = self._faiss.read_index(str(idx_path))
            logger.info(f"[FAISS] Loaded index '{name}' from {idx_path}")
        for emb_path in self.save_dir.glob("embeddings_*.npy"):
            name = emb_path.stem.replace("embeddings_", "")
            self._embeddings[name] = np.load(str(emb_path))
            logger.info(f"[FAISS] Loaded embeddings '{name}' from {emb_path}")

    def collection_size(self, collection: str) -> int:
        idx = self._indices.get(collection)
        return idx.ntotal if idx else 0


# ═══════════════════════════════════════════════════════════════════════════
# ChromaDB Backend
# ═══════════════════════════════════════════════════════════════════════════

class ChromaDBVectorStore(VectorStore):
    """
    ChromaDB-based vector store — embedded, persistent, cross-platform.
    Works on Windows/Linux/macOS, no server needed.
    Uses cosine similarity (inner product for normalized vectors).

    pip install chromadb
    """

    def __init__(self, save_dir: Path, **kwargs):
        import chromadb
        self._chromadb = chromadb
        self.save_dir = save_dir
        self._db_path = save_dir / "chromadb"
        self._db_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._db_path))
        self._collections: Dict[str, Any] = {}
        self._embeddings: Dict[str, np.ndarray] = {}
        logger.info(f"[ChromaDB] Opened persistent store at {self._db_path}")

    def add(self, collection: str, embeddings: np.ndarray,
            metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        emb = embeddings.astype("float32")
        n, dim = emb.shape

        # Delete collection if it exists (recreate)
        try:
            self._client.delete_collection(name=collection)
        except Exception:
            pass

        col = self._client.create_collection(
            name=collection,
            metadata={"hnsw:space": "ip"},  # inner product = cosine for normalized
        )

        # ChromaDB expects list of lists + string IDs
        ids = [str(i) for i in range(n)]
        emb_list = emb.tolist()

        # Insert in batches (ChromaDB has a batch limit ~41666)
        batch_size = 5000
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            col.add(
                ids=ids[start:end],
                embeddings=emb_list[start:end],
                metadatas=[{"idx": i} for i in range(start, end)] if metadata is None else metadata[start:end],
            )

        self._collections[collection] = col
        self._embeddings[collection] = emb
        logger.info(f"[ChromaDB] Added {n} vectors to '{collection}' (dim={dim})")

        # Also save raw npy for interoperability
        np.save(str(self.save_dir / f"embeddings_{collection}.npy"), emb)

    def search(self, collection: str, query: np.ndarray,
               top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        col = self._collections.get(collection)
        if col is None:
            col = self._client.get_collection(name=collection)
            self._collections[collection] = col

        q = query.astype("float32")
        if q.ndim == 1:
            q = q.reshape(1, -1)

        results = col.query(
            query_embeddings=q.tolist(),
            n_results=top_k,
            include=["distances"],
        )

        # ChromaDB returns distances (lower = closer for L2, but we use IP)
        # For IP space, ChromaDB returns distances where higher = more similar
        all_distances = []
        all_indices = []
        for i in range(len(results["ids"])):
            ids = results["ids"][i]
            dists = results["distances"][i] if results["distances"] else [0.0] * len(ids)
            indices = [int(id_str) for id_str in ids]
            all_distances.append(dists)
            all_indices.append(indices)

        return (
            np.array(all_distances, dtype="float32"),
            np.array(all_indices, dtype="int64"),
        )

    def save(self) -> None:
        # ChromaDB PersistentClient auto-persists
        logger.info(f"[ChromaDB] Data auto-persisted at {self._db_path}")

    def load(self) -> None:
        # Re-open client (it auto-loads from disk)
        self._client = self._chromadb.PersistentClient(path=str(self._db_path))
        # Load embeddings
        for emb_path in self.save_dir.glob("embeddings_*.npy"):
            name = emb_path.stem.replace("embeddings_", "")
            if name not in self._embeddings:
                self._embeddings[name] = np.load(str(emb_path))
        logger.info(f"[ChromaDB] Loaded store from {self._db_path}")

    def collection_size(self, collection: str) -> int:
        col = self._collections.get(collection)
        if col is None:
            try:
                col = self._client.get_collection(name=collection)
                self._collections[collection] = col
            except Exception:
                return 0
        return col.count()


# ═══════════════════════════════════════════════════════════════════════════
# Milvus Backend  (requires running Milvus server + pip install pymilvus)
# ═══════════════════════════════════════════════════════════════════════════

class MilvusVectorStore(VectorStore):
    """
    Milvus-based vector store — distributed, server-based.
    Requires a running Milvus instance.

    pip install pymilvus
    """

    def __init__(self, save_dir: Path, uri: str = "http://localhost:19530",
                 collection_prefix: str = "graphrag", **kwargs):
        from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
        self._pymilvus = {
            "connections": connections,
            "Collection": Collection,
            "FieldSchema": FieldSchema,
            "CollectionSchema": CollectionSchema,
            "DataType": DataType,
            "utility": utility,
        }
        self.uri = uri
        self.prefix = collection_prefix
        self.save_dir = save_dir  # for metadata json backup
        self._collections: Dict[str, Any] = {}

        # Connect
        connections.connect("default", uri=uri)
        logger.info(f"[Milvus] Connected to {uri}")

    def _get_collection_name(self, collection: str) -> str:
        return f"{self.prefix}_{collection}"

    def add(self, collection: str, embeddings: np.ndarray,
            metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        M = self._pymilvus
        col_name = self._get_collection_name(collection)
        dim = embeddings.shape[1]

        # Drop if exists
        if M["utility"].has_collection(col_name):
            M["utility"].drop_collection(col_name)

        # Create schema
        fields = [
            M["FieldSchema"](name="id", dtype=M["DataType"].INT64,
                             is_primary=True, auto_id=True),
            M["FieldSchema"](name="embedding", dtype=M["DataType"].FLOAT_VECTOR, dim=dim),
        ]
        schema = M["CollectionSchema"](fields=fields, description=f"GraphRAG {collection}")
        col = M["Collection"](name=col_name, schema=schema)

        # Insert
        emb_list = embeddings.astype("float32").tolist()
        col.insert([emb_list])
        col.flush()

        # Build index
        index_params = {
            "metric_type": "IP",
            "index_type": "IVF_FLAT",
            "params": {"nlist": min(128, max(1, embeddings.shape[0] // 10))},
        }
        col.create_index("embedding", index_params)
        col.load()

        self._collections[collection] = col
        logger.info(f"[Milvus] Added {embeddings.shape[0]} vectors to '{col_name}' (dim={dim})")

        # Also save raw embeddings to disk as backup
        self.save_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(self.save_dir / f"embeddings_{collection}.npy"), embeddings)

    def search(self, collection: str, query: np.ndarray,
               top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        col = self._collections.get(collection)
        if col is None:
            col_name = self._get_collection_name(collection)
            col = self._pymilvus["Collection"](col_name)
            col.load()
            self._collections[collection] = col

        q = query.astype("float32")
        if q.ndim == 1:
            q = q.reshape(1, -1)

        results = col.search(
            data=q.tolist(),
            anns_field="embedding",
            param={"metric_type": "IP", "params": {"nprobe": 16}},
            limit=top_k,
        )

        distances = np.array([[hit.distance for hit in r] for r in results], dtype="float32")
        indices = np.array([[hit.id for hit in r] for r in results], dtype="int64")
        return distances, indices

    def save(self) -> None:
        logger.info("[Milvus] Data persisted in Milvus server (+ npy backups on disk)")

    def load(self) -> None:
        logger.info("[Milvus] Collections are loaded on-demand from server")

    def collection_size(self, collection: str) -> int:
        col = self._collections.get(collection)
        if col is None:
            return 0
        return col.num_entities


# ═══════════════════════════════════════════════════════════════════════════
# ZVec Backend
# ═══════════════════════════════════════════════════════════════════════════

class ZVecVectorStore(VectorStore):
    """
    ZVec-based vector store — embedded, in-process, lightweight.
    Alibaba's "SQLite of vector databases", built on Proxima engine.

    pip install zvec
    """

    def __init__(self, save_dir: Path, **kwargs):
        import zvec as _zvec
        self._zvec = _zvec
        self.save_dir = save_dir
        self._dbs: Dict[str, Any] = {}
        self._embeddings: Dict[str, np.ndarray] = {}  # keep raw for search result mapping

    def _db_path(self, collection: str) -> str:
        return str(self.save_dir / f"zvec_{collection}.db")

    def add(self, collection: str, embeddings: np.ndarray,
            metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        import zvec

        emb = embeddings.astype("float32")
        dim = emb.shape[1]
        n = emb.shape[0]
        self._embeddings[collection] = emb

        db_path = self._db_path(collection)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Create / open database
        db = zvec.Database(db_path)

        # Define schema with a vector column and an index column
        schema = zvec.Schema()
        schema.add_field("vec", zvec.FieldType.VECTOR, dim=dim)
        schema.add_field("idx", zvec.FieldType.INT64)

        col = db.create_collection(collection, schema, exist_ok=True)

        # Insert documents
        docs = []
        for i in range(n):
            doc = {"vec": emb[i].tolist(), "idx": i}
            docs.append(doc)
        col.insert(docs)

        self._dbs[collection] = db
        logger.info(f"[ZVec] Added {n} vectors to '{collection}' (dim={dim}) → {db_path}")

        # Also save raw npy for compatibility
        np.save(str(self.save_dir / f"embeddings_{collection}.npy"), emb)

    def search(self, collection: str, query: np.ndarray,
               top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        import zvec

        db = self._dbs.get(collection)
        if db is None:
            db_path = self._db_path(collection)
            db = zvec.Database(db_path)
            self._dbs[collection] = db

        q = query.astype("float32")
        if q.ndim == 1:
            q = q.reshape(1, -1)

        all_distances = []
        all_indices = []
        col = db.get_collection(collection)

        for qv in q:
            results = col.search(qv.tolist(), top_k=top_k, field="vec")
            dists = [r["_score"] for r in results]
            idxs = [r["idx"] for r in results]
            all_distances.append(dists)
            all_indices.append(idxs)

        return (
            np.array(all_distances, dtype="float32"),
            np.array(all_indices, dtype="int64"),
        )

    def save(self) -> None:
        logger.info("[ZVec] Data auto-persisted to .db files")

    def load(self) -> None:
        # ZVec databases are opened on-demand
        for db_path in self.save_dir.glob("zvec_*.db"):
            collection = db_path.stem.replace("zvec_", "")
            if collection not in self._dbs:
                import zvec
                self._dbs[collection] = zvec.Database(str(db_path))
                logger.info(f"[ZVec] Loaded '{collection}' from {db_path}")
        # Load embeddings
        for emb_path in self.save_dir.glob("embeddings_*.npy"):
            name = emb_path.stem.replace("embeddings_", "")
            if name not in self._embeddings:
                self._embeddings[name] = np.load(str(emb_path))

    def collection_size(self, collection: str) -> int:
        emb = self._embeddings.get(collection)
        return emb.shape[0] if emb is not None else 0


# ═══════════════════════════════════════════════════════════════════════════
# Factory — choose your backend here
# ═══════════════════════════════════════════════════════════════════════════

_BACKENDS = {
    # ── Active backends ──────────────────────────────────────────────────
    "faiss":    FaissVectorStore,       # Local, in-process (pip install faiss-cpu)
    "chromadb": ChromaDBVectorStore,    # Embedded, cross-platform (pip install chromadb)

    # ── Uncomment to enable (requires additional setup) ─────────────────
    # "milvus": MilvusVectorStore,      # Needs running Milvus server + pip install pymilvus
    # "zvec":   ZVecVectorStore,        # Linux/macOS only, Python 3.10-3.12 + pip install zvec
}


def create_vector_store(backend: str, save_dir: Path, **kwargs) -> VectorStore:
    """
    Factory function to create a vector store backend.

    Args:
        backend: one of "faiss", "chromadb" (or "milvus", "zvec" if enabled)
        save_dir: directory to save/load persistent data
        **kwargs: backend-specific options (e.g. uri for Milvus)

    Returns:
        VectorStore instance

    Example:
        store = create_vector_store("chromadb", save_dir=Path("./artifacts"))
        store.add("chunks", embeddings)
        store.save()
    """
    backend = backend.lower().strip()
    if backend not in _BACKENDS:
        available = ", ".join(sorted(_BACKENDS.keys()))
        raise ValueError(
            f"Unknown vector backend '{backend}'. Available: {available}"
        )

    logger.info(f"Creating vector store: backend={backend}, dir={save_dir}")
    return _BACKENDS[backend](save_dir=save_dir, **kwargs)
