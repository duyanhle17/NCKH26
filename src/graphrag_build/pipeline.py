"""
GraphRAG Build Pipeline

Orchestrates the full build process:
  1. Load .txt documents
  2. Split into passages (legal-structure-aware)
  3. Chunk passages by token budget
  4. Extract entities + build knowledge graph
  5. Embed chunks & entities
  6. Store in vector database (FAISS / Milvus / ZVec)
  7. Save all artifacts
"""
import logging

from .config import BuildConfig
from .dataset_loader import load_txt_documents
from .passages import docs_to_passages
from .chunking import build_chunks
from .entities_kg import build_kg_llm
from .embeddings import load_embedder, embed_texts
from .vector_store import create_vector_store
from .io_artifacts import (
    ensure_dir, save_json, save_json_compact, save_pickle, save_npy, build_meta
)

logger = logging.getLogger("GRAPHRAG_BUILD")


def run_build(config: BuildConfig) -> None:
    logging.basicConfig(level=logging.INFO)

    ensure_dir(config.work_dir)
    ensure_dir(config.cache_dir)

    # ── 1. Load dataset ──────────────────────────────────────────────────
    logger.info("📚 Load dataset txt files ...")
    docs = load_txt_documents(config.dataset_dir)
    logger.info(f"✅ docs={len(docs)} from {config.dataset_dir}")

    # ── 2. Passages ──────────────────────────────────────────────────────
    logger.info("🧱 docs → passages ...")
    passages = docs_to_passages(docs)
    if not passages:
        raise ValueError("Passages=0. Dataset split failed.")
    logger.info(f"✅ passages={len(passages)}")

    # ── 3. Chunking ──────────────────────────────────────────────────────
    logger.info("✂️  passages → chunks ...")
    chunks_meta, dataset = build_chunks(
        passages,
        embed_model=config.embed_model,
        max_tokens=config.max_token_size,
        overlap_tokens=config.overlap_token_size,
        min_chunk_chars=config.min_chunk_chars,
    )
    if not dataset:
        raise ValueError("Chunks=0 after chunking.")
    logger.info(f"✅ chunks={len(dataset)}")
    
    # Lưu sớm chunk ra file để user có thể check kể cả khi KG chưa build xong
    logger.info("📁 saving chunk artifacts early ...")
    save_json_compact(config.cache_dir / "chunks.json", dataset)
    save_json(config.cache_dir / "chunks_meta.json", chunks_meta)

    # ── 4. KG extraction (LLM based) ─────────────────────────────────────
    logger.info(f"🧠 LLM entities + KG (model={config.llm_model}) ...")
    
    # Tạo thư mục checkpoint
    checkpoint_dir = config.work_dir / f"checkpoint_{config.vector_backend}"
    ensure_dir(checkpoint_dir)

    kg, chunk_entities, entity_to_chunks, all_entities, all_relationships = build_kg_llm(
        dataset,
        llm_model=config.llm_model,
        entity_types=config.entity_types,
        max_workers=config.max_workers,
        batch_size=config.batch_size,
        checkpoint_dir=str(checkpoint_dir),
        yake_top_k=config.yake_top_k,
        yake_lang=config.yake_lang
    )
    
    entities = sorted(list(entity_to_chunks.keys()))
    logger.info(
        f"✅ kg_nodes={kg.number_of_nodes()} "
        f"kg_edges={kg.number_of_edges()} entities={len(entities)}"
    )

    # ── 5. Embedding ─────────────────────────────────────────────────────
    logger.info(f"🧩 embed chunks + entities (model={config.embed_model}) ...")
    embedder = load_embedder(config.embed_model)

    chunk_emb = embed_texts(embedder, dataset, batch_size=config.batch_embed)
    ent_emb = None
    if entities:
        ent_emb = embed_texts(embedder, entities, batch_size=config.batch_embed)

    # ── 6. Vector store ──────────────────────────────────────────────────
    logger.info(f"💾 store vectors (backend={config.vector_backend}) ...")

    store_kwargs = {}
    if config.vector_backend == "milvus":
        store_kwargs["uri"] = config.milvus_uri
        store_kwargs["collection_prefix"] = config.milvus_collection

    store = create_vector_store(
        backend=config.vector_backend,
        save_dir=config.cache_dir,
        **store_kwargs,
    )

    store.add("chunks", chunk_emb)
    if ent_emb is not None:
        store.add("entities", ent_emb)
    store.save()

    logger.info(
        f"✅ vectors stored: chunks={store.collection_size('chunks')}"
        + (f", entities={store.collection_size('entities')}" if ent_emb is not None else "")
    )

    # ── 7. Save artifacts ────────────────────────────────────────────────
    logger.info("📁 saving artifacts ...")

    save_json_compact(config.cache_dir / "chunks.json", dataset)
    save_json(config.cache_dir / "chunks_meta.json", chunks_meta)
    save_json_compact(
        config.cache_dir / "chunk_entities.json",
        [sorted(list(s)) for s in chunk_entities],
    )
    save_pickle(config.cache_dir / "kg.pkl", kg)
    save_json_compact(config.cache_dir / "entities.json", entities)
    save_json_compact(
        config.cache_dir / "entity_to_chunks.json",
        {k: sorted(list(v)) for k, v in entity_to_chunks.items()},
    )
    save_json_compact(config.cache_dir / "all_entities_raw.json", all_entities)
    save_json_compact(config.cache_dir / "all_relationships_raw.json", all_relationships)

    meta = build_meta(
        config=config,
        passages_n=len(passages),
        chunks_n=len(dataset),
        kg_nodes=kg.number_of_nodes(),
        kg_edges=kg.number_of_edges(),
        entities_n=len(entities),
    )
    save_json(config.cache_dir / "meta.json", meta)

    logger.info(f"✅ DONE. artifacts at {config.cache_dir}")
    logger.info(
        f"   Model: {config.embed_model} | Backend: {config.vector_backend} | "
        f"Chunks: {len(dataset)} | Entities: {len(entities)}"
    )