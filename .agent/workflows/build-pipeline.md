---
description: How to build the GraphRAG pipeline with different vector backends
---

# GraphRAG Build Pipeline — Backend Selection Guide

## Quick Start

```bash
# Default: FAISS backend (works everywhere)
python build.py

# ChromaDB backend (needs Python ≤3.13)
.venv313/Scripts/python.exe build.py --backend chromadb    # Windows
.venv313/bin/python build.py --backend chromadb            # macOS/Linux

# ZVec backend (macOS/Linux only, Python 3.10-3.12)
python build.py --backend zvec

# Milvus backend (needs running Milvus server)
python build.py --backend milvus --milvus-uri http://localhost:19530
```

## Full Options

```bash
python build.py \
    --backend faiss \
    --model Snowflake/snowflake-arctic-embed-m \
    --dataset ./dataset \
    --output ./artifacts/graphrag_faiss \
    --max-tokens 512 \
    --overlap-tokens 64 \
    --min-chunk-chars 120
```

## Backend Compatibility

| Backend    | Platform        | Python      | Install command          |
|------------|-----------------|-------------|--------------------------|
| `faiss`    | All             | 3.8+        | `pip install faiss-cpu`  |
| `chromadb` | All             | 3.9 - 3.13  | `pip install chromadb`   |
| `milvus`   | All (client)    | 3.8+        | `pip install pymilvus`   |
| `zvec`     | Linux/macOS     | 3.10 - 3.12 | `pip install zvec`       |

## Setup for Each Backend

### FAISS (default — works on Windows/Python 3.14)
```bash
pip install -r requirements/faiss.txt
python build.py --backend faiss
```

### ChromaDB (needs Python ≤3.13 on Windows)
// turbo
```bash
# Windows: use existing .venv313
.venv313\Scripts\pip.exe install -r requirements/chromadb.txt
.venv313\Scripts\python.exe build.py --backend chromadb
```
```bash
# macOS/Linux: create venv if needed
python3.13 -m venv .venv313
.venv313/bin/pip install -r requirements/chromadb.txt
.venv313/bin/python build.py --backend chromadb
```

### ZVec (macOS/Linux only)
```bash
# macOS (Apple Silicon)
python3.12 -m venv .venv312
.venv312/bin/pip install -r requirements/zvec.txt
.venv312/bin/python build.py --backend zvec
```

### Milvus (needs server)
```bash
# 1. Start Milvus server (Docker)
docker run -d --name milvus -p 19530:19530 milvusdb/milvus:latest

# 2. Install client
pip install -r requirements/milvus.txt

# 3. Build
python build.py --backend milvus --milvus-uri http://localhost:19530
```

## Switching Between Backends

All backends produce the same output structure in `artifacts/graphrag_{backend}/`:
- `chunks.json` — chunk texts
- `chunks_meta.json` — chunk metadata
- `embeddings_chunks.npy` — raw embeddings (numpy, interoperable)
- `entities.json` — extracted entities
- `kg.pkl` — knowledge graph
- `meta.json` — build metadata

The `.npy` embeddings files are always saved regardless of backend,
so you can switch backends later without re-embedding.

## Custom Embedding Model

```bash
# Use a different model
python build.py --model intfloat/multilingual-e5-large
python build.py --model BAAI/bge-m3
python build.py --model Snowflake/snowflake-arctic-embed-l
```
