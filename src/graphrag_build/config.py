from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, List

@dataclass
class BuildConfig:
    # ── Dataset ──────────────────────────────────────────────────────────
    dataset_dir: Path = Path("./dataset")

    # ── Output directories ──────────────────────────────────────────────
    work_dir: Path = Path("./work")
    cache_dir: Path | None = None

    # ── 1. EMBEDDING MODEL CHOICE ───────────────────────────────────────
    # To use a different model, change this string to any HF model name.
    # Recommended for Vietnamese:
    #   - "Snowflake/snowflake-arctic-embed-m" (Current)
    #   - "intfloat/multilingual-e5-large"
    #   - "BAAI/bge-m3"
    embed_model: str = "Snowflake/snowflake-arctic-embed-m"
    batch_embed: int = 16

    # ── 2. STORAGE BACKEND CHOICE ───────────────────────────────────────
    # Change this to switch your database.
    # Options: "faiss", "chromadb", "milvus", "zvec"
    # Note: your files will be saved in ./artifact_{vector_backend}/
    vector_backend: str = "faiss" 

    # Milvus-specific (only used when vector_backend="milvus")
    milvus_uri: str = "http://localhost:19530"
    milvus_collection: str = "graphrag_chunks"

    # ── Chunking parameters ─────────────────────────────────────────────
    max_token_size: int = 700
    overlap_token_size: int = 128
    min_chunk_chars: int = 120

    # ── 3. LLM KG EXTRACTION PARAMETERS ─────────────────────────────────
    # Model LLM — Kimi K2 Instruct chạy qua NVIDIA NIM API
    # Để dùng model này, set: export NVAPI_KEY="nvapi-..."
    llm_model: str = "moonshotai/kimi-k2-instruct"

    # max_workers: số luồng song song gọi LLM cùng lúc.
    # Với Kimi K2 NVIDIA NIM: bạn đã test thành công với 20 workers, có thể tăng lên nữa
    # nếu bị rate limit thì backoff tự động sẽ xử lý.
    max_workers: int = 2  # Set xuống 2 vì Kimi API quota đang chặn (Too Many Requests), có thể 10

    # batch_size: số chunks xử lý trong 1 vòng lặp trước khi save checkpoint.
    # VD: 200 chunks = cứ 200 chunks lại lưu tiến độ một lần.
    # Nếu bị ngắt giữa chừng, lần sau sẻ tiếp tục từ batch chưa xong.
    batch_size: int = 200

    # Entity types phù hợp với dạng văn bản pháp luật Việt Nam (Thông tư, Nghị định...)
    entity_types: List[str] = field(default_factory=lambda: [
        "CƠ_QUAN_BAN_HÀNH",
        "VĂN_BẢN_PHÁP_LUẬT",
        "LOẠI_THUẾ_PHÍ",
        "MỨC_THU_PHÍ",
        "ĐỐI_TƯỢNG_ÁP_DỤNG",
        "HOẠT_ĐỘNG_ĐƯỢC_ĐIỀU_CHỈNH",
        "ĐIỀU_KHOẢN",
        "THỜI_HẠN_HIỆU_LỰC",
    ])

    # YAKE supplement — bổ sung keyword nếu LLM bỏ sót
    yake_top_k: int = 5       # số keyword YAKE bổ sung tối đa mỗi chunk
    yake_lang: str = "vi"     # ngôn ngữ: "vi" cho tiếng Việt

    def __post_init__(self):
        if self.cache_dir is None:
            self.cache_dir = Path(f"./artifact_{self.vector_backend}")