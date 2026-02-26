import os, json, pickle, time
from typing import List, Dict, Tuple
import numpy as np
import networkx as nx
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from tqdm import tqdm

# ============================================================
# LOAD ARTIFACTS
# ============================================================
CACHE_DIR_V2 = "./artifact_faiss"

print("🔄 Loading GraphRAG artifacts...")

with open(os.path.join(CACHE_DIR_V2, "meta.json"), "r", encoding="utf-8") as f:
    META_V2 = json.load(f)

with open(os.path.join(CACHE_DIR_V2, "chunks.json"), "r", encoding="utf-8") as f:
    CHUNKS_V2 = json.load(f)

with open(os.path.join(CACHE_DIR_V2, "kg.pkl"), "rb") as f:
    KG_V2 = pickle.load(f)

# Build entity to chunks mapping for Hybrid Search
# KG_V2 is a DiGraph where nodes contain 'source_chunks'
ENTITY_TO_CHUNKS = {}
for node, data in KG_V2.nodes(data=True):
    ENTITY_TO_CHUNKS[node] = data.get("source_chunks", [])

CHUNK_INDEX_V2 = faiss.read_index(os.path.join(CACHE_DIR_V2, "faiss_chunks.index"))

entity_index_path = os.path.join(CACHE_DIR_V2, "faiss_entities.index")
if os.path.exists(entity_index_path):
    ENTITY_INDEX_V2 = faiss.read_index(entity_index_path)
    with open(os.path.join(CACHE_DIR_V2, "entities.json"), "r", encoding="utf-8") as f:
        ENTITY_NAMES_V2 = json.load(f)
else:
    ENTITY_INDEX_V2 = None
    ENTITY_NAMES_V2 = []

EMBEDDER_V2 = SentenceTransformer(META_V2["embedding_model"])

print("✅ Artifacts loaded!")

# ============================================================
# LLM CLIENT
# ============================================================
LLM_MODEL_QUERY = "moonshotai/kimi-k2-instruct"
if not os.getenv("NVAPI_KEY"):
    raise RuntimeError("Missing NVAPI_KEY env var.")

query_llm_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("NVAPI_KEY")
)

def call_llm_query(prompt: str, temperature: float = 0.1, max_tokens: int = 1024) -> str:
    try:
        resp = query_llm_client.chat.completions.create(
            model=LLM_MODEL_QUERY,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ LLM error: {e}")
        return ""

# ============================================================
# SEARCH FUNCTIONS
# ============================================================

def embed_query(query: str) -> np.ndarray:
    vec = EMBEDDER_V2.encode([f"query: {query}"], normalize_embeddings=True)
    return vec.astype("float32")

def search_chunks_direct(query: str, top_k: int = 5) -> List[int]:
    """Retrieve chunks directly using string embedding"""
    q_vec = embed_query(query)
    D, I = CHUNK_INDEX_V2.search(q_vec, top_k)
    return [int(idx) for idx in I[0] if idx >= 0 and idx < len(CHUNKS_V2)]

def search_entities(query: str, top_k: int = 15) -> List[Tuple[str, float]]:
    """Retrieve entities directly using string embedding"""
    if ENTITY_INDEX_V2 is None: return []
    q_vec = embed_query(query)
    D, I = ENTITY_INDEX_V2.search(q_vec, top_k)
    return [(ENTITY_NAMES_V2[idx], float(D[0][j])) for j, idx in enumerate(I[0]) if 0 <= idx < len(ENTITY_NAMES_V2)]

def get_entity_relationships(entity_name: str) -> List[Dict]:
    """Get all outgoing and incoming relationships for an entity"""
    rels = []
    for _, tgt, data in KG_V2.out_edges(entity_name, data=True):
        rels.append({"src": entity_name, "tgt": tgt, **data})
    for src, _, data in KG_V2.in_edges(entity_name, data=True):
        rels.append({"src": src, "tgt": entity_name, **data})
    return rels

def hybrid_query_engine(question: str, top_k_entities: int = 5, top_k_chunks: int = 5) -> Tuple[str, Dict]:
    """
    HYBRID SEARCH PIPELINE:
    1. Lấy top K entities từ query.
    2. Từ các entities đó, trích xuất tất cả các chunks liên quan (source_chunks của entity).
    3. Lấy thêm top M chunks trực tiếp từ query.
    4. Trộn chung các chunks tìm được.
    5. Trích xuất tất cả các mối quan hệ liên quan đến top K entities.
    6. Đưa tất cả vào LLM để answer.
    """
    debug_info = {}
    
    # 1. Direct Chunk Search (Semantic Search)
    direct_chunks = search_chunks_direct(question, top_k=5)
    
    # 2. Search Entities
    entity_results = search_entities(question, top_k=5)
    matched_entities = [e for e, score in entity_results]
    
    # 3. Get Chunks from Entities (Entity -> Chunk traversal)
    entity_linked_chunks = []
    for ent in matched_entities:
        for c_idx in ENTITY_TO_CHUNKS.get(ent, []):
            if c_idx < len(CHUNKS_V2) and c_idx not in direct_chunks and c_idx not in entity_linked_chunks:
                entity_linked_chunks.append(c_idx)
                
    # Giới hạn lấy tối đa 2 chunk từ entities (mà chưa có trong direct_chunks)
    entity_linked_chunks = entity_linked_chunks[:2]
    
    # 4. Merge Chunks (Total max 7: 5 direct + 2 entity)
    final_chunk_indices = direct_chunks + entity_linked_chunks
    
    # 5. Extract Relationships cho matched entities
    rels_context = ""
    seen_rels = set()
    for ent in matched_entities:
        for rel in get_entity_relationships(ent):
            rel_key = (rel["src"], rel["tgt"])
            if rel_key not in seen_rels:
                seen_rels.add(rel_key)
                rel_name = rel.get("relation", "liên_quan")
                desc = f" ({rel['description']})" if rel.get("description") else ""
                rels_context += f"• {rel['src']} --[{rel_name}]--> {rel['tgt']}{desc}\n"
    
    # Generate Context String
    ent_context_str = "\n".join([f"• {e}" for e in matched_entities])
    chunk_context_str = "\n".join([f"[Chunk {i+1}]: {CHUNKS_V2[idx]}" for i, idx in enumerate(final_chunk_indices)])
    
    context_str = f"""--- KHÁI NIỆM & THỰC THỂ CÓ LIÊN QUAN ---
{ent_context_str}

--- CÁC MỐI QUAN HỆ TRONG ĐỒ THỊ ---
{rels_context}

--- CÁC TRÍCH ĐOẠN VĂN BẢN (CHUNKS) ---
{chunk_context_str}"""

    # 6. LLM Call
    prompt = f"""Bạn là chuyên gia về pháp luật Việt Nam. Dựa vào CONTEXT dưới đây để trả lời CÂU HỎI. 

Quy tắc:
1. TRẢ LỜI NGẮN GỌN, CHÍNH XÁC, DỰA TRÊN CONTEXT CUNG CẤP BÊN DƯỚI.
2. NẾU KHÔNG TÌM THẤY THÔNG TIN TRONG CONTEXT, HÃY TRẢ LỜI "Không tìm thấy thông tin liên quan".
3. TRÍCH DẪN ĐIỀU LUẬT (NẾU CÓ).

CONTEXT:
{context_str}

CÂU HỎI: {question}

TRẢ LỜI:"""

    answer = call_llm_query(prompt)
    
    debug_info["context_recall"] = context_str
    debug_info["num_entities"] = len(matched_entities)
    debug_info["num_chunks"] = len(final_chunk_indices)
    debug_info["num_relationships"] = len(seen_rels)
    
    return answer, debug_info

# ============================================================
# EVALUATION BATCH SCRIPT
# ============================================================
if __name__ == "__main__":
    import sys
    
    EVAL_FILE = "eval-thue.json"
    OUTPUT_FILE = "eval-thue-output.json"
    
    if not os.path.exists(EVAL_FILE):
        print(f"File {EVAL_FILE} not found!")
        sys.exit(1)
        
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
        
    print(f"🚀 Bắt đầu đánh giá cho {len(eval_data)} câu hỏi...")
    
    results = []
    for i, item in enumerate(tqdm(eval_data)):
        q = item.get("query", "")
        # Tăng thời gian chờ lên 4.5 giây để tránh lỗi Rate Limit (429) do đánh giá 50 câu liên tục
        time.sleep(4.5) 
        
        my_answer, debug = hybrid_query_engine(q)
        
        res = {
            "id": i + 1,
            "type": item.get("type", ""),
            "query": q,
            "expected_answer": item.get("expected_answer", ""),
            "my_answer": my_answer,
            "debug_info": {
                "num_entities": debug["num_entities"],
                "num_chunks": debug["num_chunks"],
                "num_relationships": debug["num_relationships"],
                "context_recall": debug["context_recall"]
            }
        }
        results.append(res)
        
        # Save tiến trình liên tục để phòng hờ bị ngắt
        with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
            json.dump(results, out_f, ensure_ascii=False, indent=2)
            
    print(f"\n✅ Đã hoàn thành! Kết quả được lưu tại {OUTPUT_FILE}")
