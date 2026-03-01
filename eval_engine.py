import os, json, pickle, time
from typing import List, Dict, Tuple
import numpy as np
import networkx as nx
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from tqdm import tqdm
from rank_bm25 import BM25Okapi

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

print("🔄 Building BM25 Index...")
def tokenize_vi(text: str) -> List[str]:
    import string
    text = str(text).lower()
    for p in string.punctuation:
        text = text.replace(p, " ")
    return text.split()

TOKENIZED_CHUNKS = [tokenize_vi(c) for c in CHUNKS_V2]
BM25_INDEX = BM25Okapi(TOKENIZED_CHUNKS)

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

def search_chunks_bm25(query: str, top_k: int = 5) -> List[int]:
    """Retrieve chunks using BM25 keyword matching"""
    tokenized_query = tokenize_vi(query)
    scores = BM25_INDEX.get_scores(tokenized_query)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [int(idx) for idx in top_indices if scores[idx] > 0]

def hybrid_query_engine(question: str, top_k_chunks: int = 6) -> Tuple[str, Dict]:
    """
    LOCAL GRAPHRAG SUMMARY: Lấy chunk từ Semantic + BM25 + Entities,
    gộp lại lấy top_k=6, sau đó bổ sung Entities/Relationships để làm Context.
    """
    debug_info = {}
    
    # 1. Lấy chunks bằng Semantic FAISS
    semantic_chunks = search_chunks_direct(question, top_k=4)
    
    # 2. Lấy chunks bằng BM25 (tốt cho exact match số liệu, năm, Phụ lục)
    bm25_chunks = search_chunks_bm25(question, top_k=4)
    
    # 3. Lấy chunks từ việc search Entity trực tiếp bằng câu hỏi
    entity_results = search_entities(question, top_k=3)
    entity_chunks = []
    for ent, _ in entity_results:
        entity_chunks.extend(ENTITY_TO_CHUNKS.get(ent, []))
    
    # Gộp lại (ưu tiên Semantic -> BM25 -> Entity Chunks)
    combined_chunks = []
    seen = set()
    for c in semantic_chunks + bm25_chunks + entity_chunks:
        if c not in seen:
            seen.add(c)
            combined_chunks.append(c)
    
    # Cắt lấy đúng top_k (6 chunk theo yêu cầu)
    final_chunk_indices = combined_chunks[:top_k_chunks]
    
    # 4. GraphRAG: Tự động tải Entities & Relationships từ các chunks đã chọn
    chunk_entities = set()
    for c_idx in final_chunk_indices:
        for ent, c_list in ENTITY_TO_CHUNKS.items():
            if c_idx in c_list:
                chunk_entities.add(ent)
                
    combined_entities = list(chunk_entities)
    
    rels_context = ""
    seen_rels = set()
    for ent in combined_entities:
        for rel in get_entity_relationships(ent):
            rel_key = (rel["src"], rel["tgt"])
            if rel_key not in seen_rels:
                seen_rels.add(rel_key)
                rel_name = rel.get("relation", "liên_quan")
                desc = f" ({rel['description']})" if rel.get("description") else ""
                rels_context += f"• {rel['src']} --[{rel_name}]--> {rel['tgt']}{desc}\n"
    
    # Generate Context String
    ent_context_str = "\n".join([f"• {e}" for e in combined_entities[:15]])
    chunk_context_str = "\n".join([f"[Chunk {i+1}]: {CHUNKS_V2[idx]}" for i, idx in enumerate(final_chunk_indices)])
    
    context_str = f"""--- KHÁI NIỆM & THỰC THỂ (TỪ ĐỒ THỊ GRAPHRAG) ---
{ent_context_str}

--- MỐI QUAN HỆ (TỪ ĐỒ THỊ GRAPHRAG) ---
{rels_context}

--- CÁC TRÍCH ĐOẠN VĂN BẢN (CHUNKS) ---
{chunk_context_str}"""

    prompt = f"""Bạn là trợ lý pháp lý chuyên về văn bản pháp luật Việt Nam.

QUY TẮC:
1) CHỈ dùng thông tin trong CONTEXT bên dưới. KHÔNG dùng kiến thức ngoài.
2) Có thể suy luận logic/tính toán dựa trên quy định trong CONTEXT.
3) Nếu CONTEXT không đủ thông tin, trả lời: "Không tìm thấy thông tin liên quan"

TRẢ LỜI NGẮN GỌN dưới dạng 1 đoạn văn (passage), gồm: câu trả lời + cơ sở pháp lý + trích dẫn ngắn. Không dùng heading, không bullet dài.

CONTEXT:
{context_str}

CÂU HỎI: {question}

TRẢ LỜI:"""

    answer = call_llm_query(prompt)
    
    debug_info["context_recall"] = context_str
    debug_info["num_entities"] = 0
    debug_info["num_chunks"] = len(final_chunk_indices)
    debug_info["num_relationships"] = 0
    
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
        q_text = item.get("query", "")
        case_text = item.get("case", "")
        if case_text:
            q = f"[Tình huống thực tế]: {case_text}\n[Câu hỏi]: {q_text}"
        else:
            q = q_text
            
        # Tăng thời gian chờ lên 4.5 giây để tránh lỗi Rate Limit (429) do đánh giá 50 câu liên tục
        time.sleep(4.5) 
        
        my_answer, debug = hybrid_query_engine(q)
        
        res = {
            "id": i + 1,
            "type": item.get("type", ""),
            "query": q_text,
            "case": case_text,
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
