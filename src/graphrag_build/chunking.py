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

# Regex to find document type keywords that signal the beginning of real content
_RE_DOC_TITLE = re.compile(
    r"^\s*(THÔNG TƯ|NGHỊ ĐỊNH|QUYẾT ĐỊNH|LUẬT|PHÁP LỆNH|CHỈ THỊ|CÔNG VĂN|HƯỚNG DẪN)\s*$",
    re.IGNORECASE | re.MULTILINE
)

# Regex for the first substantive paragraph
_RE_FIRST_SUBSTANCE = re.compile(
    r"^\s*(Căn cứ|Xét đề nghị|Thực hiện|Để\s|Theo đề nghị|Bộ trưởng|Chủ tịch)",
    re.IGNORECASE | re.MULTILINE
)

# Footer patterns – ordered from earliest to latest in a doc
_RE_FOOTER_BLOCK = re.compile(
    r"(?:"
    r"Nơi\s+nhận\s*:"          # "Nơi nhận:" section
    r"|KT\.\s*(?:BỘ TRƯỞNG|THỦ TRƯỞNG|CHỦ TỊCH)"   # Acting minister
    r"|\(Đã\s+ký"              # "(Đã ký)"
    r"|^[\s]*THỨ\s+TRƯỞNG\s*$"
    r"|^[\s]*BỘ\s+TRƯỞNG\s*$"
    r"|^[\s]*CHỦ\s+TỊCH\s*$"
    r")",
    re.IGNORECASE | re.MULTILINE
)

# Prefix injected by dataset_loader: "[filename.txt]\n"
_RE_DATASET_PREFIX = re.compile(r"^\s*\[[^\]]+\]\s*\n", re.MULTILINE)


def strip_header_footer(text: str) -> str:
    """
    Remove standard Vietnamese legal document header and footer noise.

    Also strips the [filename.txt] prefix that dataset_loader prepends.
    Uses a block-based approach:
      - Header: find the document title line (THÔNG TƯ, NGHỊ ĐỊNH, ...) and
                start from there; if not found, fall back to first substance line.
      - Footer: scan backwards from the end for signature / 'Nơi nhận' blocks.
    """
    # Strip dataset_loader prefix "[01_1999_TT-BXD.txt]\n" if present
    text = _RE_DATASET_PREFIX.sub("", text, count=1)

    lines = text.split("\n")
    total = len(lines)

    # ── HEADER: find start of real content ──
    head_block = "\n".join(lines[:min(40, total)])
    start = 0

    m_title = _RE_DOC_TITLE.search(head_block)
    if m_title:
        title_offset = head_block[:m_title.start()].count("\n")
        start = title_offset
    else:
        # Fallback: find first substantive sentence
        m_sub = _RE_FIRST_SUBSTANCE.search(head_block)
        if m_sub:
            start = head_block[:m_sub.start()].count("\n")

    # ── FOOTER: find end of real content ──
    end = total
    tail_start = max(total - 60, start)
    tail_block = "\n".join(lines[tail_start:])

    m_noi_nhan = re.search(r"Nơi\s+nhận\s*:", tail_block, re.IGNORECASE)
    m_footer = _RE_FOOTER_BLOCK.search(tail_block)

    match_to_use = None
    if m_noi_nhan and m_footer:
        match_to_use = m_noi_nhan if m_noi_nhan.start() <= m_footer.start() else m_footer
    elif m_noi_nhan:
        match_to_use = m_noi_nhan
    elif m_footer:
        match_to_use = m_footer

    if match_to_use:
        footer_line_offset = tail_block[:match_to_use.start()].count("\n")
        footer_abs = tail_start + footer_line_offset

        # Walk back past blank lines and short name-only lines (person names)
        candidate = footer_abs
        for i in range(footer_abs - 1, max(footer_abs - 5, start) - 1, -1):
            stripped = lines[i].strip()
            if not stripped:
                candidate = i
                continue
            if len(stripped) <= 40 and not re.search(r"[\.;:,]$", stripped):
                candidate = i
            else:
                break
        end = candidate

    result = "\n".join(lines[start:end]).strip()
    # Safety: if strip removed >70% of text, it was wrong — keep original
    if len(result) < len(text) * 0.3:
        return text.strip()
    return result

# ── Corrupted-table / appendix detection ──────────────────────────────────────

def _is_corrupted_block(text: str, short_line_threshold: int = 12,
                         min_short_ratio: float = 0.60) -> bool:
    """
    Return True when a block looks like a PDF/Word table that was pasted as
    plain text — characterised by a very high proportion of very short lines
    (each cell/formula-token on its own line).
    """
    lines = [l for l in text.split("\n") if l.strip()]
    if len(lines) < 6:
        return False
    short_count = sum(1 for l in lines if len(l.strip()) <= short_line_threshold)
    return (short_count / len(lines)) >= min_short_ratio


def collapse_corrupted_block(text: str) -> str:
    """
    Flatten a corrupted block into a single line by joining non-empty lines
    with spaces, preserving as much readable content as possible.
    """
    tokens = [l.strip() for l in text.split("\n") if l.strip()]
    return " ".join(tokens)

# ── Vietnamese legal structure splitting ──────────────────────────────────────
# Order matters: more specific patterns first

# Phần / Chương / Mục (Roman numerals or keywords)
_RE_PHAN = re.compile(
    r"(?im)(?=^\s*(?:"
    # I-, II-, III- (old style sections), require at least 2 chars after dash
    r"(?:(?:IX|IV|V?I{0,3})[\-\.]\s+[A-ZÀ-Ỹ]\S)"
    r"|"
    # "Phần thứ nhất", "Chương I", "Mục 1."
    r"(?:(?:PHẦN|Phần|CHƯƠNG|Chương|MỤC|Mục)\s+)"
    r"))"
)

# Điều (Article)
_RE_DIEU = re.compile(
    r"(?im)(?=^\s*Điều\s+\d+)"
)

# Khoản: 1., 2., 1-, 1.1., etc.
# Guard: must be followed by a space then at least one word character
# to avoid matching single-digit garbage lines in corrupt tables.
_RE_KHOAN = re.compile(
    r"(?im)(?=^\s*\d+(?:\.\d+)*[\.\-\)]\s+\w)"
)

# Điểm: a), b), c), đ), e),  or  a., b., c.
_RE_DIEM = re.compile(
    r"(?im)(?=^\s*[a-zđ][\)\.\-]\s+)"
)

# Bullet points: - text, + text
_RE_BULLET = re.compile(
    r"(?im)(?=^\s*[\-\+]\s+[A-ZÀ-Ỹa-zà-ỹĐđ])"
)


# Minimum characters for a unit to stand alone; smaller ones get merged
_MIN_UNIT_CHARS: int = 80
# Minimum run of consecutive tiny paragraphs (each < 20 chars) to trigger
# group-collapse of PDF-table artefacts
_CORRUPT_RUN_LEN: int = 4
_CORRUPT_PARA_MAX: int = 20


def _collapse_tiny_para_runs(paragraphs: list[str]) -> list[str]:
    """
    After normalize_text, corrupted PDF tables appear as many consecutive
    tiny paragraphs (each < _CORRUPT_PARA_MAX chars). This function detects
    runs of >= _CORRUPT_RUN_LEN such paragraphs and collapses them into one.
    """
    result: list[str] = []
    i = 0
    while i < len(paragraphs):
        # Measure how long a run of tiny paragraphs starts here
        run_end = i
        while run_end < len(paragraphs) and len(paragraphs[run_end].strip()) <= _CORRUPT_PARA_MAX:
            run_end += 1
        run_len = run_end - i
        if run_len >= _CORRUPT_RUN_LEN:
            # Collapse the run into a single line
            collapsed = " ".join(paragraphs[j].strip() for j in range(i, run_end) if paragraphs[j].strip())
            if collapsed:
                result.append(collapsed)
            i = run_end
        else:
            result.append(paragraphs[i])
            i += 1
    return result


def _merge_tiny_units(units: list[str]) -> list[str]:
    """
    Merge units that are too short to carry semantic meaning into the
    adjacent unit. Prefer merging forward (into next unit); if it's the
    last unit, merge backward.
    """
    if not units:
        return units
    merged: list[str] = []
    for u in units:
        if merged and len(u) < _MIN_UNIT_CHARS:
            merged[-1] = merged[-1] + "\n\n" + u
        else:
            merged.append(u)
    return merged


def split_legal_units(text: str) -> list[str]:
    """
    Split Vietnamese legal text into semantic units respecting the hierarchy:
    Phần/Chương > Điều > Khoản > Điểm > Bullets

    Tries from coarsest to finest; stops at the level that produces
    a reasonable number of pieces (>= 2).

    Corrupted blocks (PDF-table artefacts) are collapsed into a single line
    before being returned, so downstream chunking handles them cleanly.
    """
    text = text.strip()
    if not text:
        return []

    # ── Pre-process: collapse corrupted runs of tiny paragraphs ──────────────
    # normalize_text already reduced consecutive blanks to \n\n, so each
    # table cell / formula token is its own tiny paragraph.
    paragraphs = re.split(r"\n{2,}", text)

    # Step 1: collapse per-paragraph corruption (multi-line formulas in 1 para)
    cleaned: list[str] = []
    for para in paragraphs:
        if _is_corrupted_block(para):
            collapsed = collapse_corrupted_block(para)
            if collapsed:
                cleaned.append(collapsed)
        else:
            cleaned.append(para)

    # Step 2: collapse runs of consecutive tiny paragraphs
    cleaned = _collapse_tiny_para_runs(cleaned)
    text = "\n\n".join(cleaned)

    # ── Legal-structure splitting ─────────────────────────────────────────────
    for pattern in [_RE_PHAN, _RE_DIEU, _RE_KHOAN, _RE_DIEM, _RE_BULLET]:
        parts = pattern.split(text)
        parts = [p.strip() for p in parts if p and p.strip()]
        if len(parts) >= 2:
            return _merge_tiny_units(parts)

    # Fallback: split by double newline paragraphs
    parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if len(parts) >= 2:
        return _merge_tiny_units(parts)

    return [text]


def _split_sentences(text: str) -> list[str]:
    """
    Split a text block into sentences / clause-level segments.

    Priority split points (in order):
      1. Legal clause markers on their own line:  1.1-  1.2-  a)  b)  I.  II.
      2. Sentence-ending punctuation followed by whitespace: .  ;  :
      3. Double newline (paragraph break)

    Returns a list of non-empty segments preserving original text.
    """
    # First try splitting by legal sub-markers (e.g. "1.1-", "1.2-", "a)", "I.")
    # These often appear at the start of a line or after a newline
    _RE_SENT_SPLIT = re.compile(
        r"(?="                                      # lookahead – keep the marker in next segment
        r"(?:^|\n)\s*"                              # start-of-line or after newline
        r"(?:"
        r"\d+(?:\.\d+)*[\.\-\)]\s"                  # 1.  1.1-  2)  etc.
        r"|[a-dđ][\.\)]\s"                          # a)  b.  c)  d)  đ)
        r"|[IVXLC]+[\.\)]\s"                        # I.  II.  III)  IV.
        r"|Điều\s+\d+"                              # Điều 1, Điều 2
        r"|Khoản\s+\d+"                             # Khoản 1
        r"|Mục\s+[IVXLC\d]"                         # Mục I, Mục 1
        r")"
        r")",
        re.MULTILINE
    )
    parts = _RE_SENT_SPLIT.split(text)
    parts = [p for p in parts if p and p.strip()]
    if len(parts) >= 2:
        return parts

    # Fallback: split at sentence-ending punctuation (. ; :) followed by space/newline
    _RE_SENTENCE_END = re.compile(
        r"(?<=[\.;:])"           # after . ; :
        r"(?:\s*\n|\s{2,})"     # followed by newline or 2+ spaces
    )
    parts = _RE_SENTENCE_END.split(text)
    parts = [p for p in parts if p and p.strip()]
    if len(parts) >= 2:
        return parts

    # Last resort: split by double newline
    parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    return parts if len(parts) >= 2 else [text]


def _get_overlap_text(text: str, tw: TokenizerWrapper, budget: int) -> str:
    """
    Extract the last N sentences from `text` that fit within `budget` tokens.
    Returns the overlap text to prepend to the next chunk.
    """
    if budget <= 0:
        return ""
    # Split into sentences and take from the end
    sents = _split_sentences(text)
    ov_parts: list[str] = []
    ov_toks = 0
    for s in reversed(sents):
        st = len(tw.encode(s))
        if ov_toks + st <= budget:
            ov_parts.insert(0, s)
            ov_toks += st
        else:
            break
    return "\n\n".join(ov_parts) if ov_parts else ""


def pack_units_to_text_chunks(
    tw: TokenizerWrapper,
    units: list[str],
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """
    Pack semantic units into chunks by token-budget, returning ORIGINAL TEXT.

    Strategy:
    - Count tokens per unit using the tokenizer.
    - Combine units greedily until the budget is reached.
    - For overlap: repeat the last N full units (or sentences, for oversized
      units) whose combined token count is <= overlap_tokens.
    - Oversized units (individually > max_tokens) are split at sentence /
      clause boundaries (legal markers like 1., 2., I., II.) rather than
      mid-word — this preserves semantic integrity.
    """
    # Pre-compute token counts so we don't tokenize twice
    unit_toks: list[int] = []
    for u in units:
        tids = tw.encode(u)
        unit_toks.append(len(tids))

    out_texts: list[str] = []
    buf_units: list[str] = []
    buf_toks: int = 0

    def flush():
        nonlocal buf_units, buf_toks
        if buf_units:
            out_texts.append("\n\n".join(buf_units))
        buf_units = []
        buf_toks = 0

    def overlap_units() -> tuple[list[str], int]:
        """Return tail units from the last chunk that fit in overlap budget."""
        if not out_texts or overlap_tokens <= 0:
            return [], 0
        last_chunk_units = out_texts[-1].split("\n\n")
        ov_buf, ov_toks = [], 0
        for u in reversed(last_chunk_units):
            t = len(tw.encode(u))
            if ov_toks + t <= overlap_tokens:
                ov_buf.insert(0, u)
                ov_toks += t
            else:
                break
        return ov_buf, ov_toks

    for unit, ntok in zip(units, unit_toks):
        if ntok == 0:
            continue

        if ntok > max_tokens:
            # Oversized unit — split at sentence / legal-clause boundaries
            flush()
            sentences = _split_sentences(unit)
            seg_parts: list[str] = []
            seg_toks = 0

            for sent in sentences:
                st = len(tw.encode(sent))
                if st == 0:
                    continue

                if seg_toks + st > max_tokens and seg_parts:
                    # Emit current segment
                    chunk_text = "\n\n".join(seg_parts)
                    out_texts.append(chunk_text)
                    # Overlap: take last sentence(s) that fit in budget
                    ov_text = _get_overlap_text(chunk_text, tw, overlap_tokens)
                    if ov_text:
                        ov_tok = len(tw.encode(ov_text))
                        seg_parts = [ov_text, sent]
                        seg_toks = ov_tok + st
                    else:
                        seg_parts = [sent]
                        seg_toks = st
                else:
                    seg_parts.append(sent)
                    seg_toks += st

            if seg_parts:
                out_texts.append("\n\n".join(seg_parts))
            continue

        if buf_toks + ntok <= max_tokens:
            buf_units.append(unit)
            buf_toks += ntok
        else:
            flush()
            ov_units, ov_toks = overlap_units()
            buf_units = ov_units + [unit]
            buf_toks = ov_toks + ntok

    flush()
    return out_texts


# Keep old name as alias for any external callers
def pack_units_to_token_chunks(tw: TokenizerWrapper, units: list[str],
                               max_tokens: int, overlap_tokens: int) -> list[list[int]]:
    """Deprecated: use pack_units_to_text_chunks instead."""
    texts = pack_units_to_text_chunks(tw, units, max_tokens, overlap_tokens)
    return [tw.encode(t) for t in texts]


def split_into_sections(text: str) -> List[Dict[str, str]]:
    """
    Chia văn bản theo các mục/điều/khoản/phần lớn
    Nhận diện các tiêu đề section trong văn bản pháp luật Việt Nam
    """
    section_patterns = [
        r'^(Điều\s+\d+[\.\:])',              # Điều 1. / Điều 1:
        r'^([IVXLC]+[\.\-]\s)',               # I. / II- / III.
        r'^(\d+[\.\)]\s)',                     # 1. / 2) 
        r'^(\d+\.\d+[\.\-]\s)',               # 1.1. / 2.3-
        r'^(Chương\s+[IVXLC\d]+)',            # Chương I / Chương 1
        r'^(Phần\s+[IVXLC\d]+)',             # Phần I / Phần 1
        r'^(Mục\s+[IVXLC\d]+)',              # Mục I / Mục 1
        r'^(PHỤ LỤC)',                        # PHỤ LỤC
    ]
    
    sections = []
    current_section = ""
    current_header = ""
    
    for line in text.split('\n'):
        stripped = line.strip()
        is_header = False
        
        for pattern in section_patterns:
            if re.match(pattern, stripped, re.IGNORECASE):
                if current_section.strip():
                    sections.append({
                        "header": current_header,
                        "content": current_section.strip()
                    })
                current_header = stripped
                current_section = line + '\n'
                is_header = True
                break
        
        if not is_header:
            current_section += line + '\n'
            
    if current_section.strip():
        sections.append({
            "header": current_header,
            "content": current_section.strip()
        })
    return sections

def chunk_text_by_tokens(text: str, tw: TokenizerWrapper, max_tokens: int, overlap_tokens: int) -> List[str]:
    """
    Chia văn bản thành các chunk theo token size với overlap.
    Ưu tiên cắt tại các ranh giới câu/đoạn.
    """
    if len(tw.encode(text)) <= max_tokens:
        return [text]
        
    chunks = []
    # Chia theo câu (dùng dấu chấm, dấu chấm phẩy) HOẶC xuống dòng
    sentences = [s.strip() for s in re.split(r'(?<=[.;:!\?])\s+|\n+', text) if s.strip()]
    
    current_chunk = ""
    
    for sentence in sentences:
        sent_toks = len(tw.encode(sentence))
        curr_toks = len(tw.encode(current_chunk)) if current_chunk else 0
        
        # Nếu thêm câu hiện tại vẫn trong giới hạn max_tokens
        if curr_toks + sent_toks + 1 <= max_tokens:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
        else:
            # Lưu chunk hiện tại
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # Nếu câu đơn lẻ dài hơn max_tokens, cắt cứng theo từ
            if sent_toks > max_tokens:
                words = sentence.split()
                current_chunk = ""
                for word in words:
                    wt = len(tw.encode(word))
                    curr_t = len(tw.encode(current_chunk)) if current_chunk else 0
                    if curr_t + wt + 1 <= max_tokens:
                        current_chunk = current_chunk + " " + word if current_chunk else word
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = word
            else:
                # Tạo overlap: lấy phần cuối của chunk trước dựa vào decode token
                if chunks:
                    last_chunk = chunks[-1]
                    last_ids = tw.encode(last_chunk)
                    if len(last_ids) > overlap_tokens:
                        overlap_ids = last_ids[-overlap_tokens:]
                        overlap_text = tw.decode(overlap_ids)
                        # Cắt overlap tại ranh giới từ để tránh mất nghĩa
                        space_idx = overlap_text.find(' ')
                        if space_idx > 0:
                            overlap_text = overlap_text[space_idx + 1:]
                    else:
                        overlap_text = last_chunk
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
                    
    # Lưu chunk cuối cùng
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return chunks

def build_chunks(passages: List[Dict[str, Any]], embed_model: str, max_tokens: int, overlap_tokens: int, min_chunk_chars: int):
    tokenizer = AutoTokenizer.from_pretrained(embed_model, use_fast=True)
    tw = TokenizerWrapper(tokenizer)

    chunks_meta = []
    dataset = []
    seen_ids = set()
    chunk_id_counter = 0

    for pi, p in enumerate(passages):
        doc_key = f'{p.get("doc_id","doc")}::passage-{pi}'
        text = (p.get("content") or "").strip()
        if not text:
            continue

        text = strip_header_footer(text)
        if not text:
            continue

        # Áp dụng chiến lược chia Session -> Chunks mới của user
        sections = split_into_sections(text)
        
        for section in sections:
            section_content = section["content"]
            section_header = section["header"]
            
            text_chunks = chunk_text_by_tokens(section_content, tw, max_tokens, overlap_tokens)

            for ci, raw_text in enumerate(text_chunks):
                # Nối header vào nội dung để làm rõ ngữ cảnh cho LLM/Vector
                if section_header:
                    full_raw_text = f"{section_header} {raw_text}"
                else:
                    full_raw_text = raw_text
                    
                chunk_text = normalize_text(full_raw_text)
                
                # Nếu chunk quá nhỏ so với min_chunk_chars:
                # Nhưng nếu nó là chunk DUY NHẤT của một Section (ví dụ Điều 1 ngắn), ta vẫn GIỮ LẠI ngọai trừ nó quá ngắn (<30 chars)
                if len(chunk_text) < min_chunk_chars:
                    if len(text_chunks) > 1 or len(chunk_text) < 30:
                        continue
                        
                cid = mdhash_id(chunk_text)
                if cid in seen_ids:
                    continue
                seen_ids.add(cid)

                tok_count = len(tw.encode(full_raw_text))

                meta = {
                    "id": cid,
                    "full_doc_id": doc_key,
                    "section_header": section_header,
                    "chunk_index": ci,
                    "total_chunks_in_section": len(text_chunks),
                    "tokens": tok_count,
                    "path": p.get("path", ""),
                    "content": chunk_text
                }
                chunks_meta.append(meta)
                dataset.append(chunk_text)
                chunk_id_counter += 1

    return chunks_meta, dataset