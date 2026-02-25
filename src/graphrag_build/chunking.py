import re, hashlib
from typing import Any, Dict, List
from transformers import AutoTokenizer
from .utils_text import normalize_text

def mdhash_id(text: str, prefix="chunk-") -> str:
    h = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
    return f"{prefix}{h}"

class TokenizerWrapper:
    def __init__(self, tok):
        self.tok = tok
    def encode(self, text: str):
        return self.tok.encode(text, add_special_tokens=False)
    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True)

# ── Vietnamese legal document header/footer stripping ─────────────────────────

# Regex to find the start of actual legal content.
# We look for document type keywords that signal the beginning of the real text.
_RE_DOC_TITLE = re.compile(
    r"^\s*(THÔNG TƯ|NGHỊ ĐỊNH|QUYẾT ĐỊNH|LUẬT|PHÁP LỆNH|CHỈ THỊ|CÔNG VĂN|HƯỚNG DẪN)\s*$",
    re.IGNORECASE | re.MULTILINE
)

# Regex for the first substantive paragraph (Căn cứ ..., Xét đề nghị ..., Thực hiện ...)
_RE_FIRST_SUBSTANCE = re.compile(
    r"^\s*(Căn cứ|Xét đề nghị|Thực hiện|Để\s|Theo đề nghị|Bộ trưởng|Chủ tịch)",
    re.IGNORECASE | re.MULTILINE
)

# Footer patterns (applied on the joined tail block)
_RE_FOOTER_BLOCK = re.compile(
    r"(?:"
    r"Nơi\s+nhận\s*:"     # "Nơi nhận:" section
    r"|KT\.\s*(?:BỘ TRƯỞNG|THỦ TRƯỞNG|CHỦ TỊCH)"  # Acting minister
    r"|\(Đã\s+ký"         # "(Đã ký)" possibly across lines
    r"|^\s*THỨ\s+TRƯỞNG\s*$"
    r"|^\s*BỘ\s+TRƯỞNG\s*$"
    r"|^\s*CHỦ\s+TỊCH\s*$"
    r")",
    re.IGNORECASE | re.MULTILINE
)


def strip_header_footer(text: str) -> str:
    """
    Remove standard Vietnamese legal document header and footer noise.
    Uses a block-based approach:
      - Header: find the document title line (THÔNG TƯ, NGHỊ ĐỊNH, ...) and start from there.
      - Footer: scan backwards from the end for signature / 'Nơi nhận' blocks.
    """
    lines = text.split("\n")
    total = len(lines)

    # ── HEADER: find start of real content ──
    # Strategy: join the first ~40 lines and look for document title keyword
    head_block = "\n".join(lines[:min(40, total)])
    start = 0

    m_title = _RE_DOC_TITLE.search(head_block)
    if m_title:
        # Start from the title line itself (THÔNG TƯ, etc.)
        title_offset = head_block[:m_title.start()].count("\n")
        start = title_offset

    # ── FOOTER: find end of real content ──
    end = total
    # Scan a generous range to catch "Nơi nhận:" which can be far from the end
    tail_start = max(total - 60, start)
    tail_block = "\n".join(lines[tail_start:])

    # First, look specifically for "Nơi nhận:" — it always starts the footer section
    m_noi_nhan = re.search(r"Nơi\s+nhận\s*:", tail_block, re.IGNORECASE)
    m_footer = _RE_FOOTER_BLOCK.search(tail_block)

    # Use whichever comes first (Nơi nhận typically precedes signature)
    match_to_use = None
    if m_noi_nhan and m_footer:
        match_to_use = m_noi_nhan if m_noi_nhan.start() <= m_footer.start() else m_footer
    elif m_noi_nhan:
        match_to_use = m_noi_nhan
    elif m_footer:
        match_to_use = m_footer

    if match_to_use:
        # Walk back from footer match to include any preceding person-name lines
        footer_line_offset = tail_block[:match_to_use.start()].count("\n")
        footer_abs = tail_start + footer_line_offset

        # Scan upward past empty lines and short name-only lines (e.g. "Trương\n  Chí Trung")
        candidate = footer_abs
        for i in range(footer_abs - 1, max(footer_abs - 5, start) - 1, -1):
            stripped = lines[i].strip()
            if not stripped:
                candidate = i
                continue
            # Short lines just before signature are likely person names
            if len(stripped) <= 40 and not re.search(r"[\.;:,]$", stripped):
                candidate = i
            else:
                break
        end = candidate

    result = "\n".join(lines[start:end]).strip()
    # Fallback: nếu strip quá nhiều, giữ nguyên
    if len(result) < len(text) * 0.3:
        return text.strip()
    return result

# ── Vietnamese legal structure splitting ──────────────────────────────────────
# Order matters: more specific patterns first

# Phần / Chương / Mục (Roman numerals or keywords)
_RE_PHAN = re.compile(
    r"(?im)(?=^\s*(?:"
    # I-, II-, III- (old style sections)
    r"(?:(?:IX|IV|V?I{0,3})[\-\.]\s+[A-ZÀ-Ỹ])"
    r"|"
    # "Phần thứ nhất", "Chương I", "Mục 1"
    r"(?:(?:PHẦN|Phần|CHƯƠNG|Chương|MỤC|Mục)\s+)"
    r"))"
)

# Điều (Article)
_RE_DIEU = re.compile(
    r"(?im)(?=^\s*Điều\s+\d+)"
)

# Khoản: 1., 2., 1-, 2-, 1.1., 1.1-, 1.1.1-
_RE_KHOAN = re.compile(
    r"(?im)(?=^\s*\d+(?:\.\d+)*[\.\-\)]\s+)"
)

# Điểm: a), b), c), ... đ), e), ...  or a., b., c.
_RE_DIEM = re.compile(
    r"(?im)(?=^\s*[a-zđ][\)\.\-]\s+)"
)

# Bullet points: - text, + text
_RE_BULLET = re.compile(
    r"(?im)(?=^\s*[\-\+]\s+[A-ZÀ-Ỹa-zà-ỹĐđ])"
)

def split_legal_units(text: str) -> list[str]:
    """
    Split Vietnamese legal text into semantic units respecting the hierarchy:
    Phần/Chương > Điều > Khoản > Điểm > Bullets

    Tries from coarsest to finest; stops at the level that produces
    a reasonable number of pieces (>= 2).
    """
    text = text.strip()
    if not text:
        return []

    for pattern in [_RE_PHAN, _RE_DIEU, _RE_KHOAN, _RE_DIEM, _RE_BULLET]:
        parts = pattern.split(text)
        parts = [p.strip() for p in parts if p and p.strip()]
        if len(parts) >= 2:
            return parts

    # Fallback: split by double newline paragraphs
    parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if len(parts) >= 2:
        return parts

    return [text]


def pack_units_to_token_chunks(tw: TokenizerWrapper, units: list[str], max_tokens: int, overlap_tokens: int) -> list[list[int]]:
    out_tokens = []
    cur_tokens = []

    for unit in units:
        ut = tw.encode(unit)
        if not ut:
            continue

        if len(ut) > max_tokens:
            if cur_tokens:
                out_tokens.append(cur_tokens)
                cur_tokens = []

            start = 0
            while start < len(ut):
                piece = ut[start:start + max_tokens]
                out_tokens.append(piece)
                if len(piece) <= 1:
                    break
                eff_ov = min(overlap_tokens, len(piece) - 1)
                start += max(1, len(piece) - eff_ov)
            continue

        if len(cur_tokens) + len(ut) <= max_tokens:
            cur_tokens.extend(ut)
        else:
            if cur_tokens:
                out_tokens.append(cur_tokens)

            if overlap_tokens > 0 and out_tokens:
                prev = out_tokens[-1]
                eff_ov = min(overlap_tokens, len(prev) - 1) if len(prev) > 1 else 0
                cur_tokens = prev[-eff_ov:] + ut if eff_ov > 0 else ut
            else:
                cur_tokens = ut

    if cur_tokens:
        out_tokens.append(cur_tokens)

    return out_tokens

def build_chunks(passages: List[Dict[str, Any]], embed_model: str, max_tokens: int, overlap_tokens: int, min_chunk_chars: int):
    tokenizer = AutoTokenizer.from_pretrained(embed_model, use_fast=True)
    tw = TokenizerWrapper(tokenizer)

    chunks_meta = []
    dataset = []
    seen_ids = set()

    for pi, p in enumerate(passages):
        doc_key = f'{p.get("doc_id","doc")}::passage-{pi}'
        text = (p.get("content") or "").strip()
        if not text:
            continue

        # Strip header/footer noise from each passage
        text = strip_header_footer(text)
        if not text:
            continue

        units = split_legal_units(text)
        token_chunks = pack_units_to_token_chunks(tw, units, max_tokens=max_tokens, overlap_tokens=overlap_tokens)

        for ci, tk in enumerate(token_chunks):
            chunk_text = normalize_text(tw.decode(tk))
            if len(chunk_text) < min_chunk_chars:
                continue
            cid = mdhash_id(chunk_text)
            if cid in seen_ids:
                continue
            seen_ids.add(cid)

            meta = {
                "id": cid,
                "full_doc_id": doc_key,
                "chunk_order_index": ci,
                "tokens": len(tk),
                "path": p.get("path", ""),
                "content": chunk_text
            }
            chunks_meta.append(meta)
            dataset.append(chunk_text)

    return chunks_meta, dataset