# GraphRAG Build Pipeline — Văn bản Pháp luật Việt Nam

Hệ thống **GraphRAG** (Graph-based Retrieval-Augmented Generation) xây dựng pipeline xử lý văn bản pháp luật Việt Nam, từ dữ liệu thô đến Knowledge Graph + Vector Database, phục vụ hỏi đáp pháp lý tự động.

## 📋 Tổng quan

| Thành phần | Mô tả |
|:---|:---|
| **Dataset** | 11.280 văn bản pháp luật (Thuế - Phí - Lệ phí), 20 loại văn bản |
| **Pipeline** | Load → Passages → Chunking → KG Extraction → Embedding → Vector Store |
| **KG Backend** | NIM API (Kimi K2 Instruct) hoặc GPT Proxy (ChatGPT Plus) |
| **Vector Backend** | FAISS, ChromaDB, Milvus, ZVec |
| **Embedding** | Snowflake Arctic Embed M (mặc định) |

---

## 🚀 Cài đặt

```bash
# Clone repo
git clone <repo-url>
cd NCKH26

# Tạo virtual environment
python -m venv venv
source venv/bin/activate

# Cài đặt dependencies (chọn 1 trong các backend)
pip install -r requirements/base.txt
pip install -r requirements/faiss.txt      # FAISS (mặc định)
# pip install -r requirements/chromadb.txt # ChromaDB
# pip install -r requirements/milvus.txt   # Milvus
# pip install -r requirements/zvec.txt     # ZVec
```

---

## ⚡ Sử dụng

### Build Pipeline (CLI)

```bash
# Chạy mặc định: Snowflake embedding + FAISS
python build.py

# Tuỳ chọn backend
python build.py --backend chromadb
python build.py --backend milvus --milvus-uri http://localhost:19530

# Tuỳ chọn embedding model
python build.py --model intfloat/multilingual-e5-large

# Xem tất cả options
python build.py --help
```

### Build Pipeline (Python)

```python
from src.graphrag_build.config import BuildConfig
from src.graphrag_build.pipeline import run_build

# Mặc định: NIM API + FAISS
config = BuildConfig()
run_build(config)

# Dùng GPT proxy để build KG
config = BuildConfig(
    kg_backend="gpt",       # "nim" hoặc "gpt"
    max_workers=4,
    gpt_model="gpt-5.1",
)
run_build(config)
```

### Chuyển đổi KG Backend

Trong `src/graphrag_build/config.py`, đổi `kg_backend`:

| Giá trị | Backend | Yêu cầu |
|:---|:---|:---|
| `"nim"` (mặc định) | NVIDIA NIM API (Kimi K2 Instruct) | `export NVAPI_KEY="nvapi-..."` |
| `"gpt"` | GPT Proxy local (ChatGPT Plus) | Proxy chạy tại `localhost:8317` |

---

## 📁 Cấu trúc dự án

```
NCKH26/
├── build.py                    # CLI entry point
├── eval_engine.py              # Evaluation engine (hybrid query + LLM)
├── dataset/                    # Văn bản pháp luật (.txt)
│   └── thue_phi_le_phi/        # 20 thể loại, 11.280 files
│       ├── Thông_tư/
│       ├── Nghị_định/
│       ├── Quyết_định/
│       └── ...
├── src/graphrag_build/         # Core pipeline modules
│   ├── config.py               # Cấu hình (embedding, KG backend, chunking)
│   ├── pipeline.py             # Orchestrator chính
│   ├── dataset_loader.py       # Load .txt files
│   ├── passages.py             # Chia văn bản → passages (theo Điều/Khoản)
│   ├── chunking.py             # Passages → chunks (theo token budget)
│   ├── entities_kg.py          # KG extraction via NIM API (Kimi K2)
│   ├── gpt_kg.py               # KG extraction via GPT Proxy (ChatGPT Plus)
│   ├── embeddings.py           # Sentence-Transformers embedding
│   ├── vector_store.py         # Vector DB plugin (FAISS/ChromaDB/Milvus/ZVec)
│   ├── io_artifacts.py         # Save/load artifacts (JSON, Pickle, Numpy)
│   └── utils_text.py           # Text utilities
├── artifact_faiss/             # Output artifacts (sau khi build)
├── requirements/               # Dependencies theo backend
│   ├── base.txt
│   ├── faiss.txt
│   ├── chromadb.txt
│   ├── milvus.txt
│   └── zvec.txt
├── STRUCTURE.md                # Giải thích cấu trúc project
├── DATASET_ANALYSIS.md         # Phân tích chi tiết dataset
└── eval-thue.json              # Bộ câu hỏi đánh giá
```

---

## 🔧 Pipeline Flow

```
1. Load Dataset        dataset/*.txt
       ↓
2. Passages            Chia theo cấu trúc pháp lý (Điều/Khoản/Điểm)
       ↓
3. Chunking            Token budget (700 tokens, overlap 128)
       ↓
4. KG Extraction       NIM API hoặc GPT Proxy
   ├── Entities        Trích xuất thực thể (cơ quan, văn bản, thuế phí...)
   └── Triples         source → relation → target
       ↓
5. Embedding           Sentence-Transformers → vectors
       ↓
6. Vector Store        FAISS / ChromaDB / Milvus / ZVec
       ↓
7. Artifacts           chunks.json, kg.pkl, entities.json, ...
```

---

## 📊 Evaluation

```bash
# Chạy đánh giá trên bộ câu hỏi
python eval_engine.py
```

Evaluation engine sử dụng **hybrid query**: Semantic Search + BM25 + Entity Search + KG Relationships, sau đó gọi LLM để sinh câu trả lời.

---

## 🔑 API Keys

### NIM API (Kimi K2)
```bash
export NVAPI_KEY="nvapi-your-key-here"
```

### GPT Proxy
GPT proxy cần đang chạy tại `http://localhost:8317`. Cấu hình trong `config.py`:
```python
gpt_base_url = "http://localhost:8317/v1"
gpt_api_key = "proxypal-local"
gpt_model = "gpt-5.1"
```

---

## 📦 Output Artifacts

Sau khi build xong, các file output nằm trong `artifact_{backend}/`:

| File | Nội dung |
|:---|:---|
| `chunks.json` | Danh sách tất cả chunks (text) |
| `chunks_meta.json` | Metadata của chunks (source file, passage ID...) |
| `kg.pkl` | Knowledge Graph (NetworkX DiGraph) |
| `entities.json` | Danh sách entities đã merge |
| `entity_to_chunks.json` | Mapping entity → chunk indices |
| `chunk_entities.json` | Mapping chunk → entity names |
| `all_entities_raw.json` | Entities thô (chưa merge) |
| `all_relationships_raw.json` | Relationships thô (chưa merge) |
| `meta.json` | Thông tin build (model, backend, stats) |
| `chunks.index` / `entities.index` | FAISS indices |
