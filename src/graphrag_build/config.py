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

    # ── Chunking parameters (Unified Semantic Chunking) ────────────────
    # Chiến lược: gom unit pháp lý tuần tự (greedy buffer accumulation).
    # Flush buffer khi >= min_chunk_tokens VÀ unit tiếp theo sẽ vượt max_chunk_tokens.
    # Unit đơn lẻ > max_chunk_tokens → force-split thành ~target_chunk_tokens.
    # Chunk cuối < tail_merge_threshold → gộp vào chunk trước.
    min_chunk_tokens: int = 800       # Buffer >= 800 mới được flush
    target_chunk_tokens: int = 1000   # Mục tiêu khi force-split unit quá lớn
    max_chunk_tokens: int = 1200      # Cận trên cứng mỗi chunk
    tail_merge_threshold: int = 400   # Chunk cuối < 400 tokens → gộp vào chunk trước
    min_chunk_chars: int = 120        # Bỏ chunk quá ngắn (< 120 ký tự)

    # ── 3. KG EXTRACTION BACKEND ──────────────────────────────────────────
    # Options: "nim" (NVIDIA NIM API / Kimi K2), "gpt" (GPT proxy local)
    # Đổi giá trị này để chuyển đổi giữa 2 backend.
    kg_backend: Literal["nim", "gpt"] = "nim"

    # ── 3a. NIM API settings (kg_backend="nim") ──────────────────────────
    # Model LLM — Kimi K2 Instruct chạy qua NVIDIA NIM API
    # Để dùng model này, set: export NVAPI_KEY="nvapi-..."
    llm_model: str = "moonshotai/kimi-k2-instruct"

    # ── 3b. GPT proxy settings (kg_backend="gpt") ────────────────────────
    # GPT-5 qua ChatGPT Plus proxy. Proxy phải đang chạy trước khi build.
    gpt_base_url: str = "http://localhost:8317/v1"
    gpt_api_key: str = "proxypal-local"
    gpt_model: str = "gpt-5.2-codex"

    # ── 4. NEO4J GRAPH DATABASE ─────────────────────────────────────────
    # Set neo4j_enabled=True để export KG sang Neo4j sau khi build.
    # Cần chạy Neo4j trước: docker compose -f docker-compose.neo4j.yml up -d
    neo4j_enabled: bool = True
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "graphrag2026"

    # ── 3c. Common KG parameters ─────────────────────────────────────────
    # max_workers: số luồng song song gọi LLM cùng lúc.
    # NIM: 2 workers (API quota hạn chế). GPT proxy: có thể 4-8 workers.
    max_workers: int = 10

    # batch_size: số chunks xử lý trong 1 vòng lặp trước khi save checkpoint.
    # Nếu bị ngắt giữa chừng, lần sau sẽ tiếp tục từ batch chưa xong.
    batch_size: int = 50

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