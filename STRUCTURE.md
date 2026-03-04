# Dự án GraphRAG - Cấu trúc thư mục và tệp tin

Tài liệu này giải thích vai trò của các tệp tin và thư mục trong hệ thống GraphRAG Build Pipeline dành cho văn bản pháp luật Việt Nam.

## 📁 Thư mục gốc

| Tên tệp/thư mục | Mô tả |
| :--- | :--- |
| `build.py` | **Điểm nhập CLI chính.** Sử dụng script này để chạy toàn bộ quá trình build pipeline. Hỗ trợ chọn backend (faiss/chromadb/...) và model. |
| `dataset/` | Nơi lưu trữ các văn bản pháp luật gốc (định dạng `.txt`). |
| `requirements/` | Chứa các file cài đặt thư viện cho từng loại backend khác nhau (`base.txt`, `faiss.txt`, `chromadb.txt`, ...). |
| `src/` | Chứa mã nguồn chính của dự án. |
| `artifacts/` | Thư mục lưu trữ kết quả đầu ra sau khi build (Embedding, Metadata, Knowledge Graph). |
| `test_*.py` | Các script kiểm thử độc lập cho từng thành phần (Chunking, Embedding). |
| `.agent/workflows/` | Chứa tài liệu hướng dẫn quy trình build (`build-pipeline.md`). |

## 📁 Thư mục `src/graphrag_build/`

Đây là trung tâm xử lý dữ liệu của hệ thống:

| Tên tệp | Vai trò |
| :--- | :--- |
| `pipeline.py` | **Trình điều phối (Orchestrator).** Kết nối các bước từ lúc nạp dữ liệu đến khi lưu vào cơ sở dữ liệu vector. |
| `config.py` | Định nghĩa các tham số cấu hình như kích thước chunk, độ chồng lấp (overlap), và tham số KG. |
| `passages.py` | Chia văn bản pháp luật thành các **đoạn văn (passages)** dựa trên cấu trúc Điều/Khoản để bảo toàn ngữ nghĩa. |
| `chunking.py` | Chia nhỏ các đoạn văn thành **chunk** dựa trên số lượng token, đảm bảo không cắt giữa câu và có overlap hợp lý. |
| `entities_kg.py` | Trích xuất các thực thể pháp lý và xây dựng **Knowledge Graph** (Đồ thị tri thức) dựa trên sự đồng xuất hiện. |
| `embeddings.py` | Xử lý việc chuyển đổi văn bản thành vector (Embedding) sử dụng Sentence-Transformers. |
| `vector_store.py` | **Cơ chế Plugin cho Vector DB.** Hỗ trợ lưu trữ vào FAISS, ChromaDB, Milvus hoặc ZVec với cùng một giao diện. |
| `dataset_loader.py` | Hỗ trợ nạp các tệp văn bản từ thư mục `dataset/`. |
| `io_artifacts.py` | Các hàm bổ trợ để lưu trữ và nạp các tệp kết quả (JSON, Pickle, Numpy). |
| `utils_text.py` | Các tiện ích xử lý và chuẩn hóa văn bản. |

---
**Lưu ý:** Để chạy hệ thống, hãy sử dụng file `build.py`. Ví dụ: `python build.py --backend faiss`.

---------------------------------------------

1. Chunking: Với bộ data mới, có nhiều file --> load như nào
    + Load các file data --> chia cắt nhỏ --> mỗi một passage có tiêu đ, ID riêng ( Ví dụ: Thông tư II Chương 3,....)
    + Chunking ( mỗi chunk nó sẽ có ID riêng của chunk đó : tên của Passage nó thuộc)
    + tìm cách eval chunk
2. Embedding : Snowflake/ BGE BAAI m3
3. Tuần này thử : faiss/ chromaDB
    + Làm các nào để kiểm tra độ chính xác của vectorDB, check vectorDB đang lưu faiss như nào.
    + Nếu mà storage đang ổn hay ko ổn, thì nó dựa theo cái gì (model embedding --> số chiều của model đó)
4. Xây dựng KG --> prompt như nào cho LLM để xây ( ưu tiên nếu có dạng triples ( head,relation,tail))
    --> Làm thế nào để lưu KG ( hiện tại đang sử dụng với Networkx), tìm hiểu cách để lưu vào Neo4j

5. Evaluation
