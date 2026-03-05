"""
gpt_kg.py — Build Knowledge Graph qua GPT proxy (ChatGPT Plus)

Self-contained module: có đầy đủ prompt, parse, merge logic.
Tương thích hoàn toàn với pipeline.py (cùng return signature với entities_kg.build_kg_llm).

Flow: chunks đã có sẵn → Extract Entities → Extract Triples (Relationships) → Build KG (nx.DiGraph)

Để chạy: set kg_backend="gpt" trong BuildConfig.
Yêu cầu: GPT proxy phải đang chạy ở gpt_base_url (default: localhost:8317).
"""

import os, json, re, time, gc, logging
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

import networkx as nx
from openai import OpenAI

logger = logging.getLogger("KG_GPT")

# =========================
# Adaptive rate limiter
# =========================
class AdaptiveRateLimiter:
    def __init__(self):
        self.lock = Lock()
        self.request_count = 0
        self.start_time = time.time()
        self.consecutive_errors = 0
        self.backoff_until = 0

    def wait(self):
        with self.lock:
            now = time.time()
            if now < self.backoff_until:
                sleep_time = self.backoff_until - now
                logger.info(f"⏳ GPT backoff: sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
            self.request_count += 1

    def report_success(self):
        with self.lock:
            self.consecutive_errors = 0

    def report_error(self, is_rate_limit: bool = False):
        with self.lock:
            self.consecutive_errors += 1
            if is_rate_limit:
                backoff = min(2 ** self.consecutive_errors, 60)
            else:
                backoff = min(2 ** self.consecutive_errors, 15)
            self.backoff_until = time.time() + backoff
            logger.warning(f"⚠️ GPT error! Backoff {backoff}s (attempt {self.consecutive_errors})")

    def get_stats(self) -> str:
        elapsed = time.time() - self.start_time
        rpm = self.request_count / (elapsed / 60) if elapsed > 0 else 0
        return f"Requests: {self.request_count} | Elapsed: {elapsed:.1f}s | Rate: {rpm:.1f} req/min"


rate_limiter = AdaptiveRateLimiter()

# =========================
# GPT client
# =========================
def make_gpt_client(base_url: str, api_key: str) -> OpenAI:
    """Tạo client kết nối GPT qua proxy local."""
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
    )


def call_gpt(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 3000,
) -> str:
    """Gọi GPT proxy với retry + adaptive rate limiting."""
    for attempt in range(5):
        try:
            rate_limiter.wait()
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Bạn là chuyên gia phân tích văn bản pháp luật Việt Nam. "
                            "Luôn trả về JSON array theo đúng format yêu cầu, "
                            "không giải thích, không markdown."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            rate_limiter.report_success()
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            s = str(e).lower()
            is_rate = "429" in s or "rate" in s or "too many" in s
            rate_limiter.report_error(is_rate_limit=is_rate)
            if "timeout" in s or "connection" in s:
                wait = 2 ** attempt
                logger.warning(f"Connection error (attempt {attempt+1}), retry in {wait}s: {e}")
                time.sleep(wait)
            else:
                logger.warning(f"GPT call failed (attempt {attempt+1}): {e}")
                time.sleep(1)
    logger.error("GPT call failed after 5 attempts, returning empty string.")
    return ""


# =========================
# Prompts (Vietnamese legal document — Thông tư, Nghị định, ...)
# =========================

ENTITY_EXTRACTION_PROMPT = """Bạn là chuyên gia phân tích văn bản pháp luật Việt Nam, chuyên về Thông tư, Nghị định, Quyết định.

Hãy trích xuất TẤT CẢ entities quan trọng từ đoạn văn bản pháp luật dưới đây.

Các loại entity CẦN trích xuất ({entity_types}):
- CƠ_QUAN_BAN_HÀNH: Bộ, Ủy ban, cơ quan nhà nước ban hành văn bản (VD: Bộ Tài chính, Bộ Xây dựng)
- VĂN_BẢN_PHÁP_LUẬT: Thông tư, Nghị định, Pháp lệnh, Quyết định kèm số hiệu (VD: Thông tư số 01/1999/TT-BXD)
- LOẠI_THUẾ_PHÍ: Tên loại thuế hoặc phí, lệ phí (VD: Thuế giá trị gia tăng, Lệ phí cấp phép, Phí môi giới)
- MỨC_THU_PHÍ: Mức phí cụ thể bằng số, tỷ lệ phần trăm (VD: 0,75% trị giá giao dịch, 500.000 đồng)
- ĐỐI_TƯỢNG_ÁP_DỤNG: Tổ chức, cá nhân hoặc loại phương tiện chịu sự điều chỉnh (VD: doanh nghiệp xây dựng, xe quân sự)
- HOẠT_ĐỘNG_ĐƯỢC_ĐIỀU_CHỈNH: Hành vi, hoạt động kinh tế được quy định (VD: môi giới chứng khoán, lập dự toán xây lắp)
- ĐIỀU_KHOẢN: Điều, khoản, mục, điểm cụ thể (VD: Điều 13, Điều 14 Nghị định 25/2004/NĐ-CP)
- THỜI_HẠN_HIỆU_LỰC: Ngày bắt đầu hoặc kết thúc hiệu lực (VD: có hiệu lực từ ngày 01/01/1999)

VĂN BẢN:
{chunk_text}

Quy tắc:
- Trích xuất name ĐÚNG như trong văn bản, không viết tắt tùy tiện
- Với văn bản pháp luật: luôn lấy đủ số hiệu (VD: "Nghị định số 28/1998/NĐ-CP")
- Với mức phí/thuế: lấy cả giá trị và đơn vị (VD: "0,75% trị giá giao dịch")
- description: tóm tắt ngắn gọn vai trò hoặc nội dung trong ngữ cảnh này

Trả về JSON array (không giải thích, không markdown):
[
  {{"name": "...", "type": "...", "description": "..."}},
  ...
]
"""

RELATIONSHIP_EXTRACTION_PROMPT = """Bạn là chuyên gia phân tích văn bản pháp luật Việt Nam.

Dựa trên đoạn văn bản và danh sách entities đã trích xuất, hãy xác định các MỐI QUAN HỆ giữa các entities.
ƯU TIÊN tạo các TRIPLES (subject → predicate → object) rõ ràng, có thể truy vấn được.

ENTITIES:
{entities_json}

VĂN BẢN:
{chunk_text}

Các loại relation phù hợp với văn bản pháp luật VN:
- ban_hành_bởi       : văn bản được ban hành bởi cơ quan
- căn_cứ             : văn bản/quy định dựa trên văn bản khác
- sửa_đổi_bổ_sung    : văn bản sửa đổi văn bản cũ
- quy_định_mức       : văn bản/loại phí có mức thu cụ thể
- áp_dụng_cho        : quy định áp dụng cho đối tượng nào
- điều_chỉnh         : văn bản điều chỉnh hoạt động nào
- thu_bởi            : loại phí được thu bởi cơ quan nào
- nộp_vào            : tiền phí nộp vào ngân sách/tài khoản nào
- có_hiệu_lực_từ     : thời điểm văn bản có hiệu lực
- thay_thế           : văn bản mới thay thế văn bản cũ
- hướng_dẫn_thi_hành : văn bản hướng dẫn thi hành văn bản khác
- bao_gồm            : đối tượng/nội dung bao gồm thành phần con
- liên_quan          : quan hệ chung khi không xác định được loại cụ thể

Quy tắc:
- MỖI relationship là 1 TRIPLE: source (chủ thể) → relation (vị ngữ) → target (đối tượng)
- Chỉ tạo relationship nếu có bằng chứng rõ ràng trong văn bản
- source và target phải KHỚP CHÍNH XÁC với name trong danh sách entities
- Ưu tiên tạo nhiều triples đa dạng để KG phong phú, dễ truy vấn
- weight: 0.9=rất chắc chắn, 0.7=chắc chắn, 0.5=có thể

Trả về JSON array (không giải thích, không markdown):
[
  {{"source": "...", "target": "...", "relation": "...", "description": "...", "weight": 0.8}},
  ...
]
"""

# =========================
# Helpers
# =========================
def parse_json_safe(text: str) -> List[Dict[str, Any]]:
    text = (text or "").strip()
    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        text = m.group()
    try:
        arr = json.loads(text)
        return arr if isinstance(arr, list) else []
    except Exception:
        return []


def norm_entity_name(name: str) -> str:
    return (name or "").strip().upper()


def extract_terms_yake(text: str, top_k: int = 10, lan: str = "vi") -> List[str]:
    """
    YAKE hỗ trợ: chỉ để bổ sung entities bị sót.
    - lan: "vi" (VN) hoặc "en"
    """
    try:
        import yake
        kw = yake.KeywordExtractor(lan=lan, n=3, top=top_k, dedupLim=0.9, windowsSize=1)
        kws = kw.extract_keywords(text)
        kws = sorted(kws, key=lambda x: x[1])[:top_k]
        out = []
        for k, _ in kws:
            k = (k or "").strip()
            if len(k) >= 3:
                out.append(k)
        return out
    except Exception:
        return []


# =========================
# Chunk processing
# =========================
def process_single_chunk(
    client: OpenAI,
    gpt_model: str,
    chunk: str,
    chunk_idx: int,
    entity_types: List[str],
    yake_top_k: int,
    yake_lang: str,
    max_text_chars: int = 6000,  # GPT context lớn hơn NIM → cho nhiều text hơn
) -> Tuple[int, List[Dict[str, Any]], List[Dict[str, Any]]]:

    chunk_short = chunk[:max_text_chars]

    # ---- Entities via GPT
    prompt_ent = ENTITY_EXTRACTION_PROMPT.format(
        entity_types=", ".join(entity_types),
        chunk_text=chunk_short,
    )
    ent_text = call_gpt(client, gpt_model, prompt_ent, temperature=0.1, max_tokens=2500)
    ents_raw = parse_json_safe(ent_text)

    entities: List[Dict[str, Any]] = []
    for e in ents_raw:
        if not isinstance(e, dict):
            continue
        name = norm_entity_name(e.get("name", ""))
        if not name:
            continue
        entities.append({
            "name": name,
            "type": e.get("type", "UNKNOWN"),
            "description": (e.get("description", "") or "").strip(),
            "source_chunk": chunk_idx,
        })

    # ---- YAKE supplement (optional)
    yake_terms = extract_terms_yake(chunk_short, top_k=yake_top_k, lan=yake_lang) if yake_top_k > 0 else []
    existing = {e["name"] for e in entities}
    for t in yake_terms:
        n = norm_entity_name(t)
        if n and n not in existing:
            entities.append({
                "name": n,
                "type": "TERM_YAKE",
                "description": "Keyword extracted by YAKE",
                "source_chunk": chunk_idx,
            })
            existing.add(n)

    # ---- Relationships (Triples) via GPT
    relationships: List[Dict[str, Any]] = []
    if entities:
        entities_json = json.dumps(
            [{"name": e["name"], "type": e["type"]} for e in entities],
            ensure_ascii=False,
            indent=2,
        )
        prompt_rel = RELATIONSHIP_EXTRACTION_PROMPT.format(
            entities_json=entities_json,
            chunk_text=chunk_short,
        )
        rel_text = call_gpt(client, gpt_model, prompt_rel, temperature=0.1, max_tokens=2500)
        rels_raw = parse_json_safe(rel_text)

        names = {e["name"] for e in entities}
        for r in rels_raw:
            if not isinstance(r, dict):
                continue
            src = norm_entity_name(r.get("source", ""))
            tgt = norm_entity_name(r.get("target", ""))
            if not src or not tgt or src == tgt:
                continue
            if src not in names or tgt not in names:
                # chỉ giữ rel mà node có trong chunk list
                continue
            try:
                w = float(r.get("weight", 0.5))
            except Exception:
                w = 0.5
            relationships.append({
                "source": src,
                "target": tgt,
                "relation": (r.get("relation", "liên_quan") or "liên_quan").strip(),
                "description": (r.get("description", "") or "").strip(),
                "weight": max(0.1, min(1.0, w)),
                "source_chunk": chunk_idx,
            })

    return chunk_idx, entities, relationships


# =========================
# Build KG (parallel + checkpoint)
# =========================
def build_kg_gpt(
    dataset: List[str],
    *,
    gpt_model: str = "gpt-5.1",
    gpt_base_url: str = "http://localhost:8317/v1",
    gpt_api_key: str = "proxypal-local",
    entity_types: List[str],
    max_workers: int = 4,
    batch_size: int = 200,
    checkpoint_dir: Optional[str] = None,
    yake_top_k: int = 5,
    yake_lang: str = "vi",
) -> Tuple[nx.DiGraph, List[Set[str]], Dict[str, Set[int]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build KG using GPT proxy. Same return signature as build_kg_llm().

    Flow:
      1. Mỗi chunk → GPT extract entities (JSON)
      2. Mỗi chunk → GPT extract triples/relationships (JSON)
      3. Merge tất cả entities → nodes
      4. Merge tất cả triples → edges
      5. Return (kg, chunk_entities, entity_to_chunks, all_entities, all_relationships)
    """

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
        progress_file = os.path.join(checkpoint_dir, "progress.json")
    else:
        progress_file = None

    client = make_gpt_client(base_url=gpt_base_url, api_key=gpt_api_key)

    total = len(dataset)
    num_batches = (total + batch_size - 1) // batch_size

    # resume from checkpoint
    start_batch = 0
    if progress_file and os.path.exists(progress_file):
        with open(progress_file, "r", encoding="utf-8") as f:
            prog = json.load(f)
            start_batch = int(prog.get("completed_batches", 0))

    logger.info(
        f"🚀 GPT KG build | model={gpt_model} | chunks={total} | "
        f"batches={num_batches} | workers={max_workers} | resume_batch={start_batch}"
    )

    all_entities: List[Dict[str, Any]] = []
    all_relationships: List[Dict[str, Any]] = []
    chunk_entities_map: List[List[Dict[str, Any]]] = [[] for _ in range(total)]

    for b in range(start_batch, num_batches):
        bs = b * batch_size
        be = min(bs + batch_size, total)
        logger.info(f"📦 Batch {b+1}/{num_batches} chunks {bs}-{be-1}")

        args_list = [(dataset[i], i) for i in range(bs, be)]
        batch_results: Dict[int, Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]] = {}

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {
                ex.submit(
                    process_single_chunk,
                    client,
                    gpt_model,
                    chunk,
                    idx,
                    entity_types,
                    yake_top_k,
                    yake_lang,
                ): idx
                for chunk, idx in args_list
            }
            done = 0
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    _, ents, rels = fut.result()
                    batch_results[idx] = (ents, rels)
                except Exception as e:
                    logger.error(f"Chunk {idx} failed: {e}")
                    batch_results[idx] = ([], [])
                done += 1
                if done % 10 == 0 or done == (be - bs):
                    elapsed = time.time() - t0
                    rate = done / elapsed if elapsed > 0 else 0
                    logger.info(f"   [{done}/{be-bs}] {rate:.2f} chunks/s | {rate_limiter.get_stats()}")

        # collect + checkpoint
        batch_entities = []
        batch_relationships = []

        for idx in range(bs, be):
            ents, rels = batch_results.get(idx, ([], []))
            batch_entities.extend(ents)
            batch_relationships.extend(rels)
            chunk_entities_map[idx] = ents

        all_entities.extend(batch_entities)
        all_relationships.extend(batch_relationships)

        if checkpoint_dir:
            batch_file = os.path.join(checkpoint_dir, f"batch_{b:04d}.json")
            with open(batch_file, "w", encoding="utf-8") as f:
                json.dump({
                    "batch_idx": b,
                    "start_chunk": bs,
                    "end_chunk": be,
                    "entities": batch_entities,
                    "relationships": batch_relationships,
                }, f, ensure_ascii=False)

            with open(progress_file, "w", encoding="utf-8") as f:
                json.dump({
                    "completed_batches": b + 1,
                    "total_batches": num_batches,
                    "total_entities": len(all_entities),
                    "total_relationships": len(all_relationships),
                }, f, ensure_ascii=False)

        # memory
        del batch_results, batch_entities, batch_relationships
        gc.collect()

    # ========= Merge entities =========
    entity_data = defaultdict(lambda: {"types": [], "descriptions": [], "source_chunks": set()})
    for e in all_entities:
        name = e["name"]
        entity_data[name]["types"].append(e.get("type", "UNKNOWN"))
        if e.get("description"):
            entity_data[name]["descriptions"].append(e["description"])
        entity_data[name]["source_chunks"].add(int(e.get("source_chunk", -1)))

    kg = nx.DiGraph()
    for name, data in entity_data.items():
        # choose most common type
        counts = defaultdict(int)
        for t in data["types"]:
            counts[t] += 1
        main_type = max(counts, key=counts.get) if counts else "UNKNOWN"

        # merge a few unique descriptions
        uniq_desc = []
        for d in data["descriptions"]:
            if d not in uniq_desc:
                uniq_desc.append(d)
            if len(uniq_desc) >= 3:
                break

        kg.add_node(
            name,
            entity_type=main_type,
            description=" | ".join(uniq_desc),
            source_chunks=sorted(list(data["source_chunks"])),
        )

    # ========= Merge relationships (triples) =========
    edge_data = defaultdict(lambda: {"relations": [], "descriptions": [], "weights": [], "source_chunks": set()})
    for r in all_relationships:
        key = (r["source"], r["target"], r.get("relation", "liên_quan"))
        edge_data[key]["relations"].append(r.get("relation", "liên_quan"))
        if r.get("description"):
            edge_data[key]["descriptions"].append(r["description"])
        edge_data[key]["weights"].append(float(r.get("weight", 0.5)))
        edge_data[key]["source_chunks"].add(int(r.get("source_chunk", -1)))

    for (src, tgt, rel), data in edge_data.items():
        if not kg.has_node(src) or not kg.has_node(tgt):
            continue

        # pick most frequent rel label
        counts = defaultdict(int)
        for r in data["relations"]:
            counts[r] += 1
        main_rel = max(counts, key=counts.get) if counts else rel

        uniq_desc = []
        for d in data["descriptions"]:
            if d not in uniq_desc:
                uniq_desc.append(d)
            if len(uniq_desc) >= 2:
                break

        avg_weight = sum(data["weights"]) / max(1, len(data["weights"]))

        kg.add_edge(
            src,
            tgt,
            relation=main_rel,
            description=" | ".join(uniq_desc),
            weight=avg_weight,
            source_chunks=sorted(list(data["source_chunks"])),
        )

    # ========= Build chunk_entities + entity_to_chunks =========
    chunk_entities: List[Set[str]] = []
    entity_to_chunks: Dict[str, Set[int]] = defaultdict(set)

    for idx, ents in enumerate(chunk_entities_map):
        names = set(e["name"] for e in ents if e.get("name"))
        chunk_entities.append(names)
        for n in names:
            entity_to_chunks[n].add(idx)

    logger.info(f"✅ KG built (GPT): nodes={kg.number_of_nodes()} edges={kg.number_of_edges()}")
    return kg, chunk_entities, dict(entity_to_chunks), all_entities, all_relationships
