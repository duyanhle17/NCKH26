"""Chunking module – Unified Parser & Accumulator for Vietnamese legal documents.

Flow:
  dataset_loader.py  → trả document nguyên vẹn (1 dict / file)
  passages.py        → normalize text, pass through
  chunking.py (HERE) → nhận diện cấu trúc 6 cấp (Phần > Chương > Mục > Điều
                        > Khoản > Điểm), greedy buffer accumulation 800-1200
                        tokens, tail-merge orphan < 400 tokens.

Strategy:
  - Line-by-line parsing toàn bộ document, nhận diện 6 cấp hierarchy.
  - Hard-stop tại ranh giới ngữ nghĩa (Phần/Chương/Mục/Điều/Khoản/Điểm).
  - Greedy Buffer: gom unit tuần tự, flush khi >= 800 tokens VÀ thêm unit
    tiếp theo sẽ vượt 1200.
  - Tail merge: chunk cuối < 400 tokens → gộp vào chunk trước đó.
  - Force-split unit đơn lẻ > 1200 → ~1000 tokens mỗi phần.
  - Semantic Range ID: chunk_[doc_slug]_[start_node]_to_[end_node].
  - Fallback paragraph-based cho văn bản flat (Công điện, Lệnh, Sắc lệnh).
"""

import re
import unicodedata
from typing import Any, Dict, List, Tuple

from transformers import AutoTokenizer

from .utils_text import normalize_text


# ── Default constants (fallback nếu không truyền tham số) ─────────────────────

_DEFAULT_MIN_TOKENS: int = 800
_DEFAULT_TARGET_TOKENS: int = 1000
_DEFAULT_MAX_TOKENS: int = 1200
_DEFAULT_TAIL_MERGE_THRESHOLD: int = 400


# ── Deterministic Semantic ID ─────────────────────────────────────────────────

def slugify(text: str) -> str:
    """Chuyển text tiếng Việt thành slug ASCII lowercase."""
    s = text.replace("đ", "d").replace("Đ", "d")
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def _generate_range_id(
    doc_number: str,
    start_marker: str,
    end_marker: str = "",
    sub_idx: int = 0,
    total_sub: int = 1,
) -> str:
    """
    Sinh Semantic Range ID xác định.

    Format: ``chunk_[doc_slug]_[start_slug]_to_[end_slug]``
    Sub-chunk thêm ``_p[index]``.
    """
    doc_slug = slugify(doc_number) if doc_number else "unknown"
    start_slug = slugify(start_marker) if start_marker else "root"
    end_slug = slugify(end_marker) if end_marker else ""

    if not end_slug or start_slug == end_slug:
        cid = f"chunk_{doc_slug}_{start_slug}"
    else:
        cid = f"chunk_{doc_slug}_{start_slug}_to_{end_slug}"

    if total_sub > 1:
        cid += f"_p{sub_idx}"
    return cid


# ── Tokenizer wrapper ────────────────────────────────────────────────────────

class TokenizerWrapper:
    def __init__(self, tok: Any) -> None:
        self.tok = tok

    def encode(self, text: str) -> List[int]:
        return self.tok.encode(text, add_special_tokens=False)

    def decode(self, ids: List[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=True)


# ── Corrupted-table detection & collapse ──────────────────────────────────────

# ── Corrupted-table detection & collapse ──────────────────────────────────────

def _is_corrupted_block(text: str, short_line_threshold: int = 12,
                         min_short_ratio: float = 0.60) -> bool:
    lines = [l for l in text.split("\n") if l.strip()]
    if len(lines) < 6:
        return False
    short_count = sum(1 for l in lines if len(l.strip()) <= short_line_threshold)
    return (short_count / len(lines)) >= min_short_ratio


def _collapse_corrupted_block(text: str) -> str:
    tokens = [l.strip() for l in text.split("\n") if l.strip()]
    return " ".join(tokens)


# ── Sentence splitting (cho force-split & oversized units) ────────────────────

def _split_sentences(text: str) -> List[str]:
    """Split text thành sentence/clause segments (multi-level)."""
    # 1. Legal sub-markers
    _RE_SENT_SPLIT = re.compile(
        r"(?="
        r"(?:^|\n)\s*"
        r"(?:"
        r"\d+(?:\.\d+)*[.\-\)]\s"
        r"|[a-zđ][.\)]\s"
        r"|[IVXLC]+[.\)]\s"
        r"|Điều\s+\d+"
        r"|Khoản\s+\d+"
        r"|Mục\s+[IVXLC\d]"
        r")"
        r")",
        re.MULTILINE,
    )
    parts = _RE_SENT_SPLIT.split(text)
    parts = [p for p in parts if p and p.strip()]
    if len(parts) >= 2:
        return parts

    # 2. Double newline
    parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if len(parts) >= 2:
        return parts

    # 3. Punctuation + newline/spaces
    parts = re.split(r"(?<=[.;:])\s*\n|\s{2,}", text)
    parts = [p for p in parts if p and p.strip()]
    if len(parts) >= 2:
        return parts

    # 4. Vietnamese sentence boundary
    parts = re.split(r"(?<=[.;])\s+(?=[A-ZÀ-ỸĐ\-])", text)
    parts = [p for p in parts if p and p.strip()]
    if len(parts) >= 2:
        return parts

    return [text]


# ── 6-level hierarchy regex ──────────────────────────────────────────────────
# Phần > Chương > Mục > Điều > Khoản > Điểm

_RE_PHU_LUC = re.compile(
    r"^\s*(?:PHỤ\s+LỤC|Phụ\s+lục)[\s:0-9A-Za-zÀ-ỹ\-\.]*", re.IGNORECASE
)
_RE_PHAN = re.compile(
    r"^\s*(?:PHẦN|Phần)\s+([IVXLCDM]+|\d+)", re.IGNORECASE
)
_RE_CHUONG = re.compile(
    r"^\s*(?:CHƯƠNG|Chương)\s+([IVXLCDM]+|\d+)", re.IGNORECASE
)
_RE_MUC = re.compile(
    r"^\s*(?:MỤC|Mục)\s+(\d+|[IVXLCDM]+)", re.IGNORECASE
)
_RE_DIEU = re.compile(r"^\s*(?:ĐIỀU|Điều)\s+(\d+)", re.MULTILINE)
_RE_KHOAN_EXPLICIT = re.compile(
    r"^\s*(?:Khoản|KHOẢN)\s+(\d+)", re.MULTILINE | re.IGNORECASE
)
_RE_SUB_ITEM = re.compile(r"^\s*(\d+\.\d+(?:\.\d+)*)[\.\-\)]*\s", re.MULTILINE)
_RE_KHOAN_NUM = re.compile(r"^\s*(\d+)[\.\-\)]\s", re.MULTILINE)
_RE_DIEM = re.compile(r"^\s*([a-zđ])[\.\)]\s", re.MULTILINE)

# Quick check: text chứa bất kỳ marker pháp lý nào?
_RE_ANY_LEGAL = re.compile(
    r"(?:^|\n)\s*(?:"
    r"(?:PHẦN|Phần)\s+[IVXLCDM\d]"
    r"|(?:CHƯƠNG|Chương)\s+[IVXLCDM\d]"
    r"|(?:MỤC|Mục)\s+[IVXLCDM\d]"
    r"|(?:ĐIỀU|Điều)\s+\d+"
    r"|(?:Khoản|KHOẢN)\s+\d+"
    r"|\d+\.\d+(?:\.\d+)*[\.\-\)]*\s"
    r"|\d+[\.\-\)]\s"
    r"|[a-zđ][\.\)]\s"
    r")",
    re.MULTILINE,
)


# ── Marker classification (6 levels) ─────────────────────────────────────────

def _classify_marker(line: str) -> Tuple[str, str]:
    """
    Nhận diện marker pháp lý 6 cấp ở đầu dòng.

    Returns ``(level, label)``:
      - ``("phu_luc", "Phụ lục I")``
      - ``("phan",    "Phần II")``
      - ``("chuong",  "Chương III")``
      - ``("muc",     "Mục 1")``
      - ``("dieu",    "Điều 5")``
      - ``("khoan",   "Khoản 3")``
      - ``("diem",    "Điểm a")``
      - ``("sub_item","1.2.3")``
      - ``("", "")`` nếu không có marker.
    """
    stripped = line.strip()

    m = _RE_PHU_LUC.match(stripped)
    if m:
        return "phu_luc", stripped

    m = _RE_PHAN.match(stripped)
    if m:
        return "phan", f"Phần {m.group(1)}"

    m = _RE_CHUONG.match(stripped)
    if m:
        return "chuong", f"Chương {m.group(1)}"

    m = _RE_MUC.match(stripped)
    if m:
        return "muc", f"Mục {m.group(1)}"

    m = _RE_DIEU.match(stripped)
    if m:
        return "dieu", f"Điều {m.group(1)}"

    m = _RE_KHOAN_EXPLICIT.match(stripped)
    if m:
        return "khoan", f"Khoản {m.group(1)}"

    # Sub-item trước khoản-số (1.1 vs 1.)
    m = _RE_SUB_ITEM.match(stripped)
    if m:
        return "sub_item", m.group(1)

    m = _RE_KHOAN_NUM.match(stripped)
    if m:
        return "khoan", f"Khoản {m.group(1)}"

    m = _RE_DIEM.match(stripped)
    if m:
        return "diem", f"Điểm {m.group(1)}"

    return "", ""


def _has_legal_markers(text: str) -> bool:
    return bool(_RE_ANY_LEGAL.search(text))


# ── Pre-processing ────────────────────────────────────────────────────────────

def _clean_document_text(text: str) -> str:
    """Collapse corrupted PDF table fragments, giữ cấu trúc paragraph."""
    paragraphs = re.split(r"\n{2,}", text)
    cleaned: List[str] = []
    for para in paragraphs:
        if _is_corrupted_block(para):
            collapsed = _collapse_corrupted_block(para)
            if collapsed:
                cleaned.append(collapsed)
        else:
            cleaned.append(para)

    result: List[str] = []
    i = 0
    while i < len(cleaned):
        run_end = i
        while run_end < len(cleaned) and len(cleaned[run_end].strip()) <= 15:
            run_end += 1
        if run_end - i >= 6:
            collapsed = " ".join(
                cleaned[j].strip() for j in range(i, run_end) if cleaned[j].strip()
            )
            if collapsed:
                result.append(collapsed)
            i = run_end
        else:
            result.append(cleaned[i])
            i += 1
    return "\n\n".join(result)


# ── Line-by-line parser → semantic units ──────────────────────────────────────

# Thứ tự cấp: cao → thấp (dùng để reset state)
_LEVEL_ORDER: List[str] = ["phu_luc", "phan", "chuong", "muc", "dieu", "khoan", "diem", "sub_item"]


def _parse_document_to_units(text: str) -> List[Dict[str, Any]]:
    """
    Parse toàn bộ document line-by-line, tách thành semantic units.

    Mỗi unit: ``{"level": str, "marker": str, "content": str, "hierarchy_path": str}``.

    Hierarchy state 6 cấp: phu_luc / phan / chuong / muc / dieu / khoan / diem / sub_item.
    Khi gặp cấp X → reset tất cả cấp thấp hơn X.
    """
    lines = text.split("\n")
    units: List[Dict[str, Any]] = []

    # Hierarchy state
    hier: Dict[str, str] = {
        "phu_luc": "", "phan": "", "chuong": "", "muc": "",
        "dieu": "", "khoan": "", "diem": "", "sub_item": "",
    }

    cur_level = ""
    cur_marker = ""
    cur_lines: List[str] = []

    def _build_path() -> str:
        parts: List[str] = []
        for key in _LEVEL_ORDER:
            if hier[key]:
                parts.append(hier[key])
        return " > ".join(parts) if parts else ""

    def _update_hier(level: str, marker: str) -> None:
        """Cập nhật hierarchy: set level hiện tại, reset các cấp thấp hơn."""
        idx = _LEVEL_ORDER.index(level)
        hier[level] = marker
        for lower in _LEVEL_ORDER[idx + 1:]:
            hier[lower] = ""

    def _flush_unit() -> None:
        nonlocal cur_level, cur_marker, cur_lines
        if not cur_lines:
            return
        content = "\n".join(cur_lines).strip()
        if not content:
            cur_lines = []
            return
        units.append({
            "level": cur_level,
            "marker": cur_marker,
            "content": content,
            "hierarchy_path": _build_path(),
        })
        cur_lines = []

    for line in lines:
        level, marker = _classify_marker(line)
        if level:
            # Flush unit trước đó
            _flush_unit()
            # Cập nhật hierarchy state
            _update_hier(level, marker)
            cur_level = level
            cur_marker = marker
            cur_lines = [line]
        else:
            cur_lines.append(line)

    _flush_unit()
    return units


# ── Token-based splitting ─────────────────────────────────────────────────────

def _split_text_at_tokens(
    text: str, tw: TokenizerWrapper, n_tokens: int
) -> Tuple[str, str]:
    """Cắt text thành (first_part, remainder) sao cho first_part ~ n_tokens."""
    total = len(tw.encode(text))
    if total <= n_tokens:
        return text, ""

    segments = _split_sentences(text)
    if len(segments) >= 2:
        first_parts: List[str] = []
        toks = 0
        split_idx = 0
        for i, seg in enumerate(segments):
            st = len(tw.encode(seg))
            if toks + st <= n_tokens:
                first_parts.append(seg)
                toks += st
                split_idx = i + 1
            else:
                break
        if first_parts:
            return "\n".join(first_parts), "\n".join(segments[split_idx:])

    # Fallback: word-level split
    words = text.split()
    first_words: List[str] = []
    t = 0
    for w in words:
        wt = len(tw.encode(w))
        if t + wt + 1 <= n_tokens:
            first_words.append(w)
            t += wt + 1
        else:
            break
    rest_words = words[len(first_words):]
    return " ".join(first_words), " ".join(rest_words)


def _split_oversized_unit(
    text: str, tw: TokenizerWrapper, target_tokens: int, max_tokens: int
) -> List[str]:
    """Chia unit > max_tokens thành ~ target_tokens mỗi phần."""
    if len(tw.encode(text)) <= max_tokens:
        return [text]

    segments = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if len(segments) < 2:
        segments = _split_sentences(text)

    refined: List[str] = []
    for seg in segments:
        if len(tw.encode(seg)) <= max_tokens:
            refined.append(seg)
        else:
            words = seg.split()
            buf: List[str] = []
            buf_len = 0
            for w in words:
                wt = len(tw.encode(w))
                if buf_len + wt + 1 <= max_tokens:
                    buf.append(w)
                    buf_len += wt + 1
                else:
                    if buf:
                        refined.append(" ".join(buf))
                    buf = [w]
                    buf_len = wt
            if buf:
                refined.append(" ".join(buf))

    chunks: List[str] = []
    cur_parts: List[str] = []
    cur_toks = 0
    for seg in refined:
        st = len(tw.encode(seg))
        if cur_toks + st <= target_tokens:
            cur_parts.append(seg)
            cur_toks += st
        else:
            if cur_parts:
                chunks.append("\n\n".join(cur_parts))
            cur_parts = [seg]
            cur_toks = st
    if cur_parts:
        chunks.append("\n\n".join(cur_parts))
    return chunks


# ── Greedy Buffer Accumulation ────────────────────────────────────────────────

def _accumulate_blocks(
    units: List[Dict[str, Any]],
    tw: TokenizerWrapper,
    min_tokens: int = _DEFAULT_MIN_TOKENS,
    target_tokens: int = _DEFAULT_TARGET_TOKENS,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    tail_merge_threshold: int = _DEFAULT_TAIL_MERGE_THRESHOLD,
) -> List[Dict[str, Any]]:
    """
    Greedy Buffer Accumulation — gom unit tuần tự, flush khi đạt ngưỡng.

    Quy tắc:
    1. buffer + next_unit <= max_tokens → nạp tiếp.
    2. Tràn max_tokens:
       - Buffer >= min_tokens → flush, bắt đầu buffer mới.
       - Buffer < min_tokens  → force-split unit để lấp đầy.
    3. Unit đơn lẻ > max_tokens → split thành ~ target_tokens.
    4. Tail merge: chunk cuối < tail_merge_threshold → gộp vào chunk trước.

    Returns list block dict:
      content, start_marker, end_marker, start_level, end_level, hierarchy_path, tokens.
    """
    if not units:
        return []

    blocks: List[Dict[str, Any]] = []

    buf_texts: List[str] = []
    buf_info: List[Dict[str, str]] = []
    buf_toks: int = 0

    def _flush() -> None:
        nonlocal buf_texts, buf_info, buf_toks
        if not buf_texts:
            return
        seen = []
        for info in buf_info:
            hp = info.get("hierarchy_path", "")
            if hp and hp not in seen:
                seen.append(hp)
        merged_path = " | ".join(seen)
        blocks.append({
            "content": "\n\n".join(buf_texts),
            "start_marker": buf_info[0]["marker"],
            "end_marker": buf_info[-1]["marker"],
            "start_level": buf_info[0]["level"],
            "end_level": buf_info[-1]["level"],
            "hierarchy_path": merged_path,
            "tokens": buf_toks,
        })
        buf_texts = []
        buf_info = []
        buf_toks = 0

    def _make_info(unit: Dict[str, Any]) -> Dict[str, str]:
        return {
            "level": unit["level"],
            "marker": unit["marker"],
            "hierarchy_path": unit.get("hierarchy_path", ""),
        }

    for unit in units:
        u_text = unit["content"]
        u_toks = len(tw.encode(u_text))
        u_info = _make_info(unit)

        # Case 3: unit > max_tokens → split riêng
        if u_toks > max_tokens:
            _flush()
            sub_chunks = _split_oversized_unit(u_text, tw, target_tokens, max_tokens)
            for sc in sub_chunks:
                blocks.append({
                    "content": sc,
                    "start_marker": unit["marker"],
                    "end_marker": unit["marker"],
                    "start_level": unit["level"],
                    "end_level": unit["level"],
                    "hierarchy_path": unit.get("hierarchy_path", ""),
                    "tokens": len(tw.encode(sc)),
                })
            continue

        # Case 1: fits in buffer
        if buf_toks + u_toks <= max_tokens:
            buf_texts.append(u_text)
            buf_info.append(u_info)
            buf_toks += u_toks
            continue

        # Case 2: overflow
        if buf_toks >= min_tokens:
            _flush()
            buf_texts = [u_text]
            buf_info = [u_info]
            buf_toks = u_toks
        else:
            # Force-split unit
            space = target_tokens - buf_toks
            if space < 50:
                space = max_tokens - buf_toks
            first_part, remainder = _split_text_at_tokens(u_text, tw, space)

            if first_part.strip():
                buf_texts.append(first_part)
                buf_info.append(u_info)
                buf_toks += len(tw.encode(first_part))

            _flush()

            if remainder.strip():
                buf_texts = [remainder]
                buf_info = [_make_info(unit)]
                buf_toks = len(tw.encode(remainder))

    _flush()

    # ── Tail merge: chunk cuối < threshold → gộp vào chunk trước ─────────
    if len(blocks) >= 2 and blocks[-1]["tokens"] < tail_merge_threshold:
        tail = blocks.pop()
        blocks[-1]["content"] += "\n\n" + tail["content"]
        blocks[-1]["end_marker"] = tail["end_marker"]
        blocks[-1]["end_level"] = tail["end_level"]
        blocks[-1]["tokens"] += tail["tokens"]

    return blocks


# ── Fallback: chunk flat text ─────────────────────────────────────────────────

def _chunk_flat_text(
    text: str, tw: TokenizerWrapper, max_tokens: int
) -> List[str]:
    """Chunk theo paragraph rồi gom đến max_tokens (cho Công điện, Lệnh…)."""
    if len(tw.encode(text)) <= max_tokens:
        return [text]

    segments = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if len(segments) < 2:
        segments = _split_sentences(text)

    refined: List[str] = []
    for seg in segments:
        if len(tw.encode(seg)) <= max_tokens:
            refined.append(seg)
        else:
            words = seg.split()
            buf: List[str] = []
            buf_len = 0
            for w in words:
                wt = len(tw.encode(w))
                if buf_len + wt + 1 <= max_tokens:
                    buf.append(w)
                    buf_len += wt + 1
                else:
                    if buf:
                        refined.append(" ".join(buf))
                    buf = [w]
                    buf_len = wt
            if buf:
                refined.append(" ".join(buf))

    chunks: List[str] = []
    cur_parts: List[str] = []
    cur_toks = 0
    for seg in refined:
        st = len(tw.encode(seg))
        if cur_toks + st <= max_tokens:
            cur_parts.append(seg)
            cur_toks += st
        else:
            if cur_parts:
                chunks.append("\n\n".join(cur_parts))
            cur_parts = [seg]
            cur_toks = st
    if cur_parts:
        chunks.append("\n\n".join(cur_parts))
    return chunks


# ── Public helpers used by passages.py ─────────────────────────────────────────

def strip_header_footer(text: str) -> str:
    """Public alias for _clean_document_text (collapse corrupted PDF fragments)."""
    return _clean_document_text(text)


def split_legal_units(text: str) -> List[str]:
    """Parse text into semantic legal units, returning their content strings."""
    if _has_legal_markers(text):
        units = _parse_document_to_units(text)
        return [u["content"] for u in units if u.get("content", "").strip()]
    # Flat text fallback: split on double newlines
    parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    return parts if parts else [text]


# ── Entry point ───────────────────────────────────────────────────────────────

def build_chunks(
    passages: List[Dict[str, Any]],
    embed_model: str,
    min_chunk_tokens: int = _DEFAULT_MIN_TOKENS,
    target_chunk_tokens: int = _DEFAULT_TARGET_TOKENS,
    max_chunk_tokens: int = _DEFAULT_MAX_TOKENS,
    tail_merge_threshold: int = _DEFAULT_TAIL_MERGE_THRESHOLD,
    min_chunk_chars: int = 120,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    **Unified Parser & Accumulator** cho văn bản pháp luật Việt Nam.

    Nhận danh sách document nguyên vẹn (1 dict/file từ dataset_loader),
    parse cấu trúc 6 cấp, gom greedy buffer theo token budget:
      - min_chunk_tokens: buffer >= giá trị này mới được flush (default 800)
      - target_chunk_tokens: mục tiêu khi force-split unit quá lớn (default 1000)
      - max_chunk_tokens: cận trên cứng mỗi chunk (default 1200)
      - tail_merge_threshold: chunk cuối < giá trị này → gộp vào chunk trước (default 400)
    """
    tokenizer = AutoTokenizer.from_pretrained(embed_model, use_fast=True)
    tw = TokenizerWrapper(tokenizer)

    chunks_meta: List[Dict[str, Any]] = []
    dataset: List[str] = []
    seen_ids: set = set()

    for doc in passages:
        doc_id = doc.get("doc_id", "")
        raw_content = (doc.get("content") or "").strip()
        if not raw_content:
            continue

        meta_info = doc.get("metadata", {})
        doc_number = meta_info.get("doc_number", "")

        # Tiền xử lý: collapse bảng PDF bị vỡ
        text = _clean_document_text(raw_content)
        if not text:
            continue

        # ── Quyết định chiến lược chunk ──────────────────────────────────
        if _has_legal_markers(text):
            # ── A. Unified semantic parsing ──────────────────────────────
            units = _parse_document_to_units(text)

            # Greedy Buffer Accumulation (dùng tham số từ config)
            blocks = _accumulate_blocks(
                units, tw,
                min_tokens=min_chunk_tokens,
                target_tokens=target_chunk_tokens,
                max_tokens=max_chunk_tokens,
                tail_merge_threshold=tail_merge_threshold,
            )

            for bi, block in enumerate(blocks):
                chunk_text = normalize_text(block["content"])

                if len(chunk_text) < min_chunk_chars:
                    if len(chunk_text) < 30:
                        continue

                start_marker = block["start_marker"]
                end_marker = block["end_marker"]
                hierarchy_path = block["hierarchy_path"]
                semantic_level = block["start_level"] or "preamble"

                cid = _generate_range_id(doc_number, start_marker, end_marker)
                orig_cid = cid
                dedup_counter = 1
                while cid in seen_ids:
                    cid = f"{orig_cid}_{dedup_counter}"
                    dedup_counter += 1
                seen_ids.add(cid)

                chunks_meta.append({
                    "id": cid,
                    "doc_id": doc_id,
                    "hierarchy_path": hierarchy_path,
                    "semantic_level": semantic_level,
                    "start_marker": start_marker,
                    "end_marker": end_marker,
                    "chunk_index": bi,
                    "tokens": len(tw.encode(chunk_text)),
                    "path": doc.get("path", ""),
                    "doc_number": doc_number,
                    "content": chunk_text,
                })
                dataset.append(chunk_text)

        else:
            # ── B. Flat-text fallback (Công điện, Lệnh, Sắc lệnh) ──────
            text_chunks = _chunk_flat_text(text, tw, max_chunk_tokens)

            for ci, raw_text in enumerate(text_chunks):
                chunk_text = normalize_text(raw_text)

                if len(chunk_text) < min_chunk_chars:
                    if len(text_chunks) > 1 or len(chunk_text) < 30:
                        continue

                cid = _generate_range_id(doc_number, f"para_{ci}")
                orig_cid = cid
                dedup_counter = 1
                while cid in seen_ids:
                    cid = f"{orig_cid}_{dedup_counter}"
                    dedup_counter += 1
                seen_ids.add(cid)

                chunks_meta.append({
                    "id": cid,
                    "doc_id": doc_id,
                    "hierarchy_path": "Văn bản",
                    "semantic_level": "paragraph",
                    "start_marker": "",
                    "end_marker": "",
                    "chunk_index": ci,
                    "tokens": len(tw.encode(chunk_text)),
                    "path": doc.get("path", ""),
                    "doc_number": doc_number,
                    "content": chunk_text,
                })
                dataset.append(chunk_text)

    return chunks_meta, dataset