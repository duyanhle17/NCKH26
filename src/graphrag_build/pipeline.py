import logging
from sentence_transformers import SentenceTransformer

from .config import BuildConfig
from .dataset_loader import load_txt_documents
from .passages import docs_to_passages
from .chunking import build_chunks
from .entities_kg import build_kg
from .embeddings_faiss import embed_passages, build_faiss_ip
from .io_artifacts import (
    ensure_dir, save_json, save_json_compact, save_pickle, save_faiss, save_npy, build_meta
)

logger = logging.getLogger("GRAPHRAG_BUILD")

def run_build(config: BuildConfig) -> None:
    logging.basicConfig(level=logging.INFO)

    ensure_dir(config.work_dir)
    ensure_dir(config.cache_dir)

    logger.info("📚 Load dataset txt files ...")
    docs = load_txt_documents(config.dataset_dir)
    logger.info(f"✅ docs={len(docs)} from {config.dataset_dir}")

    logger.info("🧱 docs -> passages ...")
    passages = docs_to_passages(docs)
    if not passages:
        raise ValueError("Passages=0. Dataset split failed.")
    logger.info(f"✅ passages={len(passages)}")

    logger.info("✂️ passages -> chunks ...")
    chunks_meta, dataset = build_chunks(
        passages,
        embed_model=config.embed_model,
        max_tokens=config.max_token_size,
        overlap_tokens=config.overlap_token_size,
        min_chunk_chars=config.min_chunk_chars
    )
    if not dataset:
        raise ValueError("Chunks=0 after chunking.")
    logger.info(f"✅ chunks={len(dataset)}")

    logger.info("🧠 entities + KG ...")
    kg, chunk_entities, entity_to_chunks = build_kg(
        dataset,
        top_k_terms_per_chunk=config.top_k_terms_per_chunk,
        max_term_words=config.max_term_words,
        cooc_window=config.cooc_window,
        prune_min_cooc_weight=config.prune_min_cooc_weight
    )
    entities = sorted(list(entity_to_chunks.keys()))
    logger.info(f"✅ kg_nodes={kg.number_of_nodes()} kg_edges={kg.number_of_edges()} entities={len(entities)}")

    logger.info("🧩 embed chunks -> faiss ...")
    embedder = SentenceTransformer(config.embed_model)
    chunk_emb = embed_passages(embedder, dataset, batch_size=config.batch_embed)
    chunk_index = build_faiss_ip(chunk_emb)

    ent_index = None
    ent_emb = None
    if entities:
        ent_emb = embed_passages(embedder, entities, batch_size=config.batch_embed)
        ent_index = build_faiss_ip(ent_emb)

    # ---- SAVE
    save_json_compact(config.cache_dir / "chunks.json", dataset)
    save_json(config.cache_dir / "chunks_meta.json", chunks_meta)

    save_faiss(config.cache_dir / "faiss_chunks.index", chunk_index)
    save_npy(config.cache_dir / "embeddings_chunks.npy", chunk_emb)

    save_json_compact(config.cache_dir / "chunk_entities.json", [sorted(list(s)) for s in chunk_entities])
    save_pickle(config.cache_dir / "kg.pkl", kg)
    save_json_compact(config.cache_dir / "entities.json", entities)
    save_json_compact(config.cache_dir / "entity_to_chunks.json", {k: sorted(list(v)) for k, v in entity_to_chunks.items()})

    if ent_index is not None and ent_emb is not None:
        save_faiss(config.cache_dir / "faiss_entities.index", ent_index)
        save_npy(config.cache_dir / "embeddings_entities.npy", ent_emb)

    meta = build_meta(
        config=config,
        md_path=None,  # dataset mode (no md)
        passages_n=len(passages),
        chunks_n=len(dataset),
        kg_nodes=kg.number_of_nodes(),
        kg_edges=kg.number_of_edges(),
        entities_n=len(entities)
    )
    save_json(config.cache_dir / "meta.json", meta)

    logger.info("✅ DONE. artifacts at %s", str(config.cache_dir))