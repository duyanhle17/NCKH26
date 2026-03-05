# NCKH26 - GraphRAG Build & Evaluation Pipeline

Dự án này cung cấp quy trình xây dựng GraphRAG từ các văn bản pháp luật và công cụ đánh giá (evaluation engine) sử dụng mô hình ngôn ngữ lớn (LLM).

## 1. Cài đặt môi trường

### Yêu cầu hệ thống
- Python 3.10 trở lên.
- Khuyên dùng môi trường ảo (venv hoặc conda).

### Cài đặt thư viện dependencies
Chạy lệnh sau để cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

Hoặc nếu bạn muốn cài đặt theo từng backend cụ thể (ví dụ FAISS):
```bash
pip install -r requirements/base.txt -r requirements/faiss.txt
```

---

## 2. Cấu hình khóa API (NVAPI_KEY)

Dự án sử dụng NVIDIA API cho các tác vụ LLM. Bạn cần export biến môi trường `NVAPI_KEY`.

### Trên macOS / Linux (Terminal)
Chạy lệnh sau trong terminal (hoặc thêm vào file `~/.zshrc` hoặc `~/.bashrc` để dùng lâu dài):
```bash
export NVAPI_KEY="your_api_key_here"
```

### Trên Windows (Command Prompt)
```cmd
set NVAPI_KEY=your_api_key_here
```

### Trên Windows (PowerShell)
```powershell
$env:NVAPI_KEY="your_api_key_here"
```

---

## 3. Hướng dẫn chạy Pipeline

Quy trình gồm 2 bước chính: Xây dựng Index và Chạy Evaluation.

### Bước 1: Xây dựng Index (build.py)
Công cụ này sẽ xử lý dữ liệu trong thư mục `dataset/`, thực hiện chunking, tạo embedding và xây dựng Knowledge Graph.

- **Chạy mặc định (FAISS):**
  ```bash
  python build.py
  ```
- **Sử dụng backend khác (ví dụ ChromaDB):**
  ```bash
  python build.py --backend chromadb
  ```
- **Tùy chỉnh model embedding:**
  ```bash
  python build.py --model intfloat/multilingual-e5-large
  ```

Sau khi chạy xong, các file artifacts sẽ được tạo trong thư mục `artifact_faiss/` (hoặc tương ứng với backend bạn chọn).

### Bước 2: Chạy công cụ đánh giá (eval_engine.py)
Sau khi đã có artifacts từ Bước 1, bạn có thể chạy engine để thực hiện truy vấn và đánh giá.

```bash
python eval_engine.py
```

*Lưu ý: `eval_engine.py` mặc định tìm dữ liệu trong `./artifact_faiss`. Nếu bạn dùng backend khác, hãy đảm bảo đường dẫn trong code trỏ đúng vị trí.*

---

## Cấu trúc thư mục chính
- `src/graphrag_build/`: Chứa mã nguồn core cho việc xây dựng pipeline.
- `dataset/`: Thư mục chứa các tệp văn bản đầu vào (.txt, .json).
- `artifact_faiss/`: Kết quả sau khi chạy `build.py` (embeddings, metadata, KG).
- `eval_engine.py`: Script dùng để chạy thử nghiệm và đánh giá chất lượng RAG.
