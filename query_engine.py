import os, json, pickle, time
from typing import List, Dict, Tuple
import numpy as np
import networkx as nx
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ============================================================
# LOAD ARTIFACTS
# ============================================================
CACHE_DIR_V2 = "./artifact_faiss"
FORCE_RELOAD = True

if FORCE_RELOAD or "QUERY_READY" not in globals():
    print("🔄 Loading GraphRAG artifacts...")
    
    # Load metadata
    with open(os.path.join(CACHE_DIR_V2, "meta.json"), "r", encoding="utf-8") as f:
        META_V2 = json.load(f)
    
    # Load chunks
    with open(os.path.join(CACHE_DIR_V2, "chunks.json"), "r", encoding="utf-8") as f:
        CHUNKS_V2 = json.load(f)
    
    # Load KG
    with open(os.path.join(CACHE_DIR_V2, "kg.pkl"), "rb") as f:
        KG_V2 = pickle.load(f)
    
    # Load FAISS indices
    CHUNK_INDEX_V2 = faiss.read_index(os.path.join(CACHE_DIR_V2, "faiss_chunks.index"))
    
    entity_index_path = os.path.join(CACHE_DIR_V2, "faiss_entities.index")
    if os.path.exists(entity_index_path):
        ENTITY_INDEX_V2 = faiss.read_index(entity_index_path)
        with open(os.path.join(CACHE_DIR_V2, "entities.json"), "r", encoding="utf-8") as f:
            ENTITY_NAMES_V2 = json.load(f)
    else:
        ENTITY_INDEX_V2 = None
        ENTITY_NAMES_V2 = []
    
    # Load embedder
    EMBEDDER_V2 = SentenceTransformer(META_V2["embedding_model"])
    
    QUERY_READY = True
    
    print("✅ Artifacts loaded!")
    print(f"   Chunks: {len(CHUNKS_V2)}")
    print(f"   KG: {KG_V2.number_of_nodes()} nodes, {KG_V2.number_of_edges()} edges")
    print(f"   Entity VDB: {len(ENTITY_NAMES_V2)} entities")

# ============================================================
# LLM CLIENT (reuse Kimi K2 Instruct)
# ============================================================
# Bạn có thể dùng 'moonshotai/kimi-k2-instruct' (hoặc model cụ thể nếu nvidia có)
LLM_MODEL_QUERY = "moonshotai/kimi-k2-instruct"
RATE_LIMIT_RPM = 40

if not os.getenv("NVAPI_KEY"):
    raise RuntimeError("Missing NVAPI_KEY env var. export NVAPI_KEY='nvapi-...'")

query_llm_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVAPI_KEY")
)

# Simple rate limiter for query
last_llm_call = 0
def call_llm_query(prompt: str, temperature: float = 0.2, max_tokens: int = 800) -> str:
    global last_llm_call
    
    # Rate limit
    min_interval = 60.0 / RATE_LIMIT_RPM
    elapsed = time.time() - last_llm_call
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    
    try:
        resp = query_llm_client.chat.completions.create(
            model=LLM_MODEL_QUERY,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        last_llm_call = time.time()
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ LLM error: {e}")
        return ""

# ============================================================
# QUERY FUNCTIONS
# ============================================================

def embed_query(query: str) -> np.ndarray:
    """Embed query"""
    vec = EMBEDDER_V2.encode([f"query: {query}"], normalize_embeddings=True)
    return vec.astype("float32")

def search_entities(query: str, top_k: int = 10) -> List[Tuple[str, float]]:
    """Search entities bằng VectorDB"""
    if ENTITY_INDEX_V2 is None or not ENTITY_NAMES_V2:
        return []
    
    q_vec = embed_query(query)
    D, I = ENTITY_INDEX_V2.search(q_vec, top_k)
    
    results = []
    for j in range(min(top_k, len(I[0]))):
        idx = int(I[0][j])
        score = float(D[0][j])
        if idx >= 0 and idx < len(ENTITY_NAMES_V2):
            results.append((ENTITY_NAMES_V2[idx], score))
    return results

def search_chunks(query: str, top_k: int = 10) -> List[Tuple[int, float]]:
    """Search chunks bằng FAISS"""
    q_vec = embed_query(query)
    D, I = CHUNK_INDEX_V2.search(q_vec, top_k)
    return [(int(I[0][j]), float(D[0][j])) for j in range(min(top_k, len(I[0]))) if int(I[0][j]) >= 0]

def get_entity_info(entity_name: str) -> Dict:
    """Get entity info từ KG"""
    if not KG_V2.has_node(entity_name):
        return None
    return {
        "name": entity_name,
        **KG_V2.nodes[entity_name]
    }

def get_entity_relationships(entity_name: str, include_incoming: bool = True) -> List[Dict]:
    """Get relationships của entity"""
    rels = []
    
    # Outgoing edges
    for _, tgt, data in KG_V2.out_edges(entity_name, data=True):
        rels.append({
            "source": entity_name,
            "target": tgt,
            "direction": "outgoing",
            **data
        })
    
    # Incoming edges
    if include_incoming:
        for src, _, data in KG_V2.in_edges(entity_name, data=True):
            rels.append({
                "source": src,
                "target": entity_name,
                "direction": "incoming",
                **data
            })
    
    return rels

def multi_hop_expansion(seed_entities: List[str], max_hops: int = 2, max_neighbors: int = 5) -> Dict:
    """
    Multi-hop expansion từ seed entities.
    Returns entities và relationships trong subgraph.
    """
    visited_entities = set(seed_entities)
    all_relationships = []
    
    current_frontier = set(seed_entities)
    
    for hop in range(max_hops):
        next_frontier = set()
        
        for entity in current_frontier:
            if not KG_V2.has_node(entity):
                continue
            
            # Get neighbors (both directions)
            neighbors = list(KG_V2.successors(entity)) + list(KG_V2.predecessors(entity))
            
            # Sort by edge weight
            weighted_neighbors = []
            for nb in neighbors:
                if nb in visited_entities:
                    continue
                edge_data = KG_V2.get_edge_data(entity, nb) or KG_V2.get_edge_data(nb, entity)
                weight = edge_data.get("weight", 0.5) if edge_data else 0.5
                weighted_neighbors.append((nb, weight))
            
            weighted_neighbors.sort(key=lambda x: x[1], reverse=True)
            
            for nb, _ in weighted_neighbors[:max_neighbors]:
                next_frontier.add(nb)
                visited_entities.add(nb)
                
                # Add relationship info
                rels = get_entity_relationships(nb, include_incoming=False)
                all_relationships.extend(rels)
        
        current_frontier = next_frontier
        if not current_frontier:
            break
    
    # Get entity infos
    entities_info = []
    for ent in visited_entities:
        info = get_entity_info(ent)
        if info:
            entities_info.append(info)
    
    return {
        "entities": entities_info,
        "relationships": all_relationships
    }

# ============================================================
# MAIN QUERY FUNCTION
# ============================================================

def query_graphrag_v2(
    question: str,
    top_k_entities: int = 8,
    top_k_chunks: int = 6,
    max_hops: int = 2,
    verbose: bool = True
) -> Tuple[str, Dict]:
    debug_info = {}
    
    # ===== Step 1: Entity Search =====
    entity_results = search_entities(question, top_k=top_k_entities)
    seed_entities = [e for e, _ in entity_results]
    debug_info["matched_entities"] = entity_results
    
    if verbose:
        print(f"🔍 Step 1: Matched {len(seed_entities)} entities")
        for ent, score in entity_results[:5]:
            print(f"   • {ent} (score: {score:.3f})")
    
    # ===== Step 2: Multi-hop Expansion =====
    expansion = multi_hop_expansion(seed_entities, max_hops=max_hops)
    debug_info["expansion"] = {
        "num_entities": len(expansion["entities"]),
        "num_relationships": len(expansion["relationships"])
    }
    
    if verbose:
        print(f"🕸️  Step 2: Expanded to {len(expansion['entities'])} entities, {len(expansion['relationships'])} relationships")
    
    # ===== Step 3: Chunk Retrieval =====
    chunk_results = search_chunks(question, top_k=top_k_chunks)
    retrieved_chunks = [CHUNKS_V2[idx] for idx, _ in chunk_results if idx < len(CHUNKS_V2)]
    debug_info["chunk_indices"] = [idx for idx, _ in chunk_results]
    
    if verbose:
        print(f"📄 Step 3: Retrieved {len(retrieved_chunks)} chunks")
    
    # ===== Step 4: Build Context =====
    # Format entities
    entities_context = "ENTITIES liên quan:\n"
    for ent in expansion["entities"][:15]:
        entities_context += f"• {ent['name']} ({ent.get('entity_type', 'UNKNOWN')}): {ent.get('description', '')[:200]}\n"
    
    # Format relationships
    rels_context = "\nRELATIONSHIPS:\n"
    seen_rels = set()
    for rel in expansion["relationships"][:20]:
        rel_key = (rel["source"], rel["target"])
        if rel_key in seen_rels:
            continue
        seen_rels.add(rel_key)
        rels_context += f"• {rel['source']} --[{rel.get('relation', 'liên_quan')}]--> {rel['target']}"
        if rel.get("description"):
            rels_context += f" ({rel['description'][:100]})"
        rels_context += "\n"
    
    # Format chunks
    chunks_context = "\nSOURCE TEXTS:\n"
    for i, chunk in enumerate(retrieved_chunks):
        chunks_context += f"[Chunk {i+1}]\n{chunk[:800]}\n\n"
    
    # Combine context
    full_context = entities_context + rels_context + chunks_context
    debug_info["context_length"] = len(full_context)
    
    # ===== Step 5: LLM Answer =====
    prompt = f"""Bạn là chuyên gia về pháp luật Việt Nam. Dựa vào CONTEXT dưới đây để trả lời CÂU HỎI. 
CONTEXT bao gồm các Entity, Relationship và các đoạn văn bản pháp luật trích xuất.

Quy tắc:
1. TRẢ LỜI NGẮN GỌN, CHÍNH XÁC, DỰA TRÊN CONTEXT.
2. NÓI RÕ THEO VĂN BẢN (HOẶC ĐIỀU LUẬT) NÀO NẾU CÓ.
3. KHÔNG BỊA ĐẶT. NẾU KHÔNG TÌM THẤY TRONG CONTEXT, HÃY TRẢ LỜI "Không tìm thấy".

CONTEXT:
{full_context}

CÂU HỎI: {question}

TRẢ LỜI:"""
    
    if verbose:
        print(f"🤖 Step 5: Calling LLM (Kimi K2 Instruct)...")
    
    answer = call_llm_query(prompt, temperature=0.1, max_tokens=1024)
    
    return answer, debug_info

# ============================================================
# MAIN EXECUTABLE
# ============================================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔍 GRAPHRAG V2 - QUERY ENGINE READY (Kimi K2 Instruct)")
    print("="*60 + "\n")

    test_questions = [
        "Chi phí trực tiếp trong dự toán xây lắp trước thuế theo Thông tư 01/1999/TT-BXD bao gồm các chi phí nào?",
        "Theo Thông tư 01/1999/TT-BXD, giá trị dự toán xây lắp sau thuế bao gồm những thành phần nào?",
    ]

    for q in test_questions:
        print(f"❓ CÂU HỎI: {q}")
        print("-" * 50)
        
        answer, debug = query_graphrag_v2(q, verbose=True)
        
        print("\n📝 TRẢ LỜI:")
        print(answer)
        print("\n" + f"📊 Debug: {debug['expansion']['num_entities']} entities, "
              f"{debug['expansion']['num_relationships']} rels, "
              f"context: {debug['context_length']} chars\n")
        print("="*60 + "\n")
