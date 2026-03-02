"""
gpt_kg.py — Dùng GPT-5 (proxy localhost:8317) để build Knowledge Graph
Chạy: python gpt_kg.py
"""

from openai import OpenAI
from pathlib import Path
from datetime import datetime

# ── CẤU HÌNH ──────────────────────────────────────────────────────
BASE_URL   = "http://localhost:8317/v1"
API_KEY    = "proxypal-local"
MODEL      = "gpt-5.1"
INPUT_DIR  = Path(".")
OUTPUT_DIR = Path("kg_outputs")
MAX_CHARS  = 8000

SYSTEM_PROMPT = """Bạn là chuyên gia phân tích văn bản pháp lý và xây dựng knowledge graph.
Nhiệm vụ: Đọc văn bản và trích xuất tri thức có cấu trúc.
Chỉ trả về nội dung Markdown, không giải thích thêm."""

KG_PROMPT_TEMPLATE = """
Từ văn bản pháp lý sau, hãy xây dựng Knowledge Graph dạng Markdown với các phần:

## 1. Thực thể chính (Entities)
Danh sách các đối tượng quan trọng (cơ quan, người, khái niệm, loại tài sản...)

## 2. Quan hệ (Relations / Triples)
Dạng: [Chủ thể] → [Quan hệ] → [Đối tượng]

## 3. Quy định & Ràng buộc (Rules)
Các điều khoản, thời hạn, điều kiện quan trọng

## 4. Tóm tắt (Summary)
Mô tả ngắn gọn văn bản là gì, điều chỉnh gì

─── NỘI DUNG VĂN BẢN ───────────────────────────────────
{content}
─────────────────────────────────────────────────────────
"""


def build_kg(client: OpenAI, text: str) -> str:
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + "\n\n[... Văn bản bị cắt bớt ...]"

    prompt = KG_PROMPT_TEMPLATE.format(content=text)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.3,
    )

    content = response.choices[0].message.content
    usage   = response.usage
    print(f"   Tokens: {usage.prompt_tokens} prompt + {usage.completion_tokens} completion")
    return content


def process_files():
    txt_files = sorted(INPUT_DIR.glob("*.txt"))
    if not txt_files:
        print("⚠️  Không tìm thấy file .txt nào.")
        return

    print(f"📂 Tìm thấy {len(txt_files)} file")
    print(f"🤖 Model: {MODEL} (GPT proxy)\n")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    OUTPUT_DIR.mkdir(exist_ok=True)
    all_summaries = []

    for i, txt_path in enumerate(txt_files, 1):
        print(f"[{i}/{len(txt_files)}] 📄 {txt_path.name}")

        try:
            text = txt_path.read_text(encoding="utf-8", errors="replace")
            kg_content = build_kg(client, text)

            out_name = txt_path.stem + "_gpt_kg.md"
            out_path = OUTPUT_DIR / out_name
            header = (
                f"# Knowledge Graph: {txt_path.name}\n"
                f"_Model: {MODEL} | Tạo lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n---\n\n"
            )
            out_path.write_text(header + kg_content, encoding="utf-8")
            print(f"   ✅ Đã lưu → {out_path}\n")

            all_summaries.append(f"## {txt_path.name}\n\n{kg_content}")

        except Exception as e:
            print(f"   ❌ Lỗi: {e}\n")
            continue

    if all_summaries:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        merged = OUTPUT_DIR / f"ALL_KG_gpt_{ts}.md"
        header = (
            f"# Knowledge Graph — Tổng hợp ({MODEL})\n"
            f"_Tạo lúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n---\n\n"
        )
        merged.write_text(header + "\n\n---\n\n".join(all_summaries), encoding="utf-8")
        print(f"📦 File tổng hợp → {merged}")

    print("\n🎉 Hoàn tất!")


if __name__ == "__main__":
    process_files()