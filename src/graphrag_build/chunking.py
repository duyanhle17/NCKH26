"""
Chunking module – Simplified pipeline for Vietnamese legal documents.

Flow:
  dataset_loader.py  → chia passages theo cấu trúc (Phần/Chương/Mục/Điều/Phụ lục)
  passages.py        → giữ nguyên passages đã chia, chỉ normalize
  chunking.py (HERE) → nhận passages, chia thành chunks theo token budget
                        mỗi chunk giữ lại passage_id (context_title) từ dataset_loader

Overlap: 100-150 tokens (configurable)
"""

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


# ── Corrupted-table detection & collapse ──────────────────────────────────────

def _is_corrupted_block(text: str, short_line_threshold: int = 12,
                         min_short_ratio: float = 0.60) -> bool:
    """
    Return True when a block looks like a PDF/Word table pasted as plain text
    — high proportion of very short lines.
    """
    lines = [l for l in text.split("\n") if l.strip()]
    if len(lines) < 6:
        return False
    short_count = sum(1 for l in lines if len(l.strip()) <= short_line_threshold)
    return (short_count / len(lines)) >= min_short_ratio


def _collapse_corrupted_block(text: str) -> str:
    """Flatten a corrupted block into a single line."""
    tokens = [l.strip() for l in text.split("\n") if l.strip()]
    return " ".join(tokens)


# ── Sentence / clause splitting for overlap & oversized units ─────────────────

def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences / clause-level segments for fine-grained chunking.
    Chiến lược chia nhiều tầng:
    1. Legal sub-markers (1.1- , Điều, Khoản...)
    2. Double newline (paragraph break)
    3. Dấu chấm/chấm phẩy + newline hoặc 2+ khoảng trắng
    4. Dấu chấm câu thông thường (. ; ) theo sau bởi 1 khoảng trắng
    """
    # 1. Try legal sub-markers first
    _RE_SENT_SPLIT = re.compile(
        r"(?="
        r"(?:^|\n)\s*"
        r"(?:"
        r"\d+(?:\.\d+)*[.\-\)]\s"     # 1. 1.1- 2) etc.
        r"|[a-dđ][.\)]\s"             # a) b. c)
        r"|[IVXLC]+[.\)]\s"           # I. II. III)
        r"|Điều\s+\d+"                # Điều 1
        r"|Khoản\s+\d+"               # Khoản 1
        r"|Mục\s+[IVXLC\d]"           # Mục I, Mục 1
        r")"
        r")",
        re.MULTILINE
    )
    parts = _RE_SENT_SPLIT.split(text)
    parts = [p for p in parts if p and p.strip()]
    if len(parts) >= 2:
        return parts

    # 2. Double newline (paragraph break)
    parts = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if len(parts) >= 2:
        return parts

    # 3. Split at sentence-ending punctuation + newline or 2+ spaces
    _RE_SENTENCE_END = re.compile(
        r"(?<=[.;:])"
        r"(?:\s*\n|\s{2,})"
    )
    parts = _RE_SENTENCE_END.split(text)
    parts = [p for p in parts if p and p.strip()]
    if len(parts) >= 2:
        return parts

    # 4. Final fallback: dấu chấm câu tiếng Việt thông thường
    #    Split sau dấu chấm (.) khi theo sau bởi chữ hoa hoặc dấu gạch
    _RE_VN_SENTENCE = re.compile(
        r'(?<=[.;])\s+(?=[A-ZÀ-ỸĐ\-])'
    )
    parts = _RE_VN_SENTENCE.split(text)
    parts = [p for p in parts if p and p.strip()]
    if len(parts) >= 2:
        return parts

    return [text]


# ── Main chunking logic ──────────────────────────────────────────────────────

def _clean_passage_text(text: str) -> str:
    """
    Pre-process passage text: collapse corrupted PDF table fragments
    nhưng giữ nguyên cấu trúc paragraph cho văn bản bình thường.
    """
    paragraphs = re.split(r"\n{2,}", text)
    cleaned = []
    for para in paragraphs:
        if _is_corrupted_block(para):
            collapsed = _collapse_corrupted_block(para)
            if collapsed:
                cleaned.append(collapsed)
        else:
            cleaned.append(para)

    # Collapse runs of consecutive tiny paragraphs (< 15 chars)
    # Chỉ collapse khi có >= 6 dòng ngắn liên tiếp (dấu hiệu bảng PDF)
    result = []
    i = 0
    while i < len(cleaned):
        run_end = i
        while run_end < len(cleaned) and len(cleaned[run_end].strip()) <= 15:
            run_end += 1
        run_len = run_end - i
        if run_len >= 6:
            # Đây là bảng PDF bị vỡ → collapse thành 1 dòng
            collapsed = " ".join(cleaned[j].strip() for j in range(i, run_end) if cleaned[j].strip())
            if collapsed:
                result.append(collapsed)
            i = run_end
        else:
            result.append(cleaned[i])
            i += 1
    return "\n\n".join(result)


def _chunk_text_by_tokens(
    text: str,
    tw: TokenizerWrapper,
    max_tokens: int,
    overlap_tokens: int,
) -> List[str]:
    """
    Chia văn bản thành chunks theo token budget.
    Overlap: lấy câu/đoạn cuối chunk trước prepend vào chunk sau.
    """
    total_toks = len(tw.encode(text))
    if total_toks <= max_tokens:
        return [text]

    sentences = _split_sentences(text)
    
    # 1. Tránh trường hợp có câu quá dài (vượt max_tokens)
    # Ta cắt câu dài thành các cụm từ (nối bằng dấu cách) và coi như nó là 1 câu
    refined_sentences = []
    for sent in sentences:
        st = len(tw.encode(sent))
        if st <= max_tokens:
            refined_sentences.append(sent)
        else:
            words = sent.split()
            current_pseudo = []
            curr_len = 0
            for w in words:
                wt = len(tw.encode(w))
                if curr_len + wt + 1 <= max_tokens:
                    current_pseudo.append(w)
                    curr_len += wt + 1
                else:
                    if current_pseudo:
                        refined_sentences.append(" ".join(current_pseudo))
                    current_pseudo = [w]
                    curr_len = wt
            if current_pseudo:
                refined_sentences.append(" ".join(current_pseudo))

    # 2. Gom các câu vào các chunks
    chunks = []
    current_parts = []
    current_toks = 0

    for sent in refined_sentences:
        st = len(tw.encode(sent))
        
        if current_toks + st <= max_tokens:
            current_parts.append(sent)
            current_toks += st
        else:
            # Ghi nhận chunk vừa rồi
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                
            # Tạo đoạn overlap bằng cách lấy đuôi của chunk trước
            ov_parts = []
            ov_toks = 0
            for s in reversed(current_parts):
                s_tok = len(tw.encode(s))
                if ov_toks + s_tok <= overlap_tokens:
                    ov_parts.insert(0, s)
                    ov_toks += s_tok
                else:
                    break
            
            # Khởi tạo chunk tiếp theo: Overlap + Câu mới hiện tại
            current_parts = ov_parts + [sent]
            current_toks = ov_toks + st

    # Xả bộ nhớ chunk cuối
    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


def build_chunks(
    passages: List[Dict[str, Any]],
    embed_model: str,
    max_tokens: int,
    overlap_tokens: int,
    min_chunk_chars: int,
):
    """
    Entry point: nhận passages từ pipeline, trả về (chunks_meta, dataset).

    Mỗi chunk giữ lại:
    - passage_id (doc_id từ dataset_loader, chứa cấu trúc Phần/Chương/Điều)
    - context_title (tiêu đề passage, ví dụ "Phụ lục 1 - Chương 2")
    - content: text chunk đã normalize + prepend context_title
    """
    tokenizer = AutoTokenizer.from_pretrained(embed_model, use_fast=True)
    tw = TokenizerWrapper(tokenizer)

    chunks_meta = []
    dataset = []
    seen_ids = set()

    for pi, p in enumerate(passages):
        passage_id = p.get("doc_id", f"doc::passage-{pi}")
        text = (p.get("content") or "").strip()
        if not text:
            continue

        # Lấy context_title từ metadata (đã set bởi dataset_loader)
        meta_info = p.get("metadata", {})
        context_title = meta_info.get("context_title", "")

        # Clean corrupted table artifacts
        text = _clean_passage_text(text)
        if not text:
            continue

        # Chunk by token budget
        text_chunks = _chunk_text_by_tokens(text, tw, max_tokens, overlap_tokens)

        for ci, raw_text in enumerate(text_chunks):
            # Prepend context_title để mỗi chunk biết nó thuộc đoạn nào
            if context_title:
                full_text = f"[Passage: {context_title}]\n{raw_text}"
            else:
                full_text = raw_text

            chunk_text = normalize_text(full_text)

            # Filter quá nhỏ
            if len(chunk_text) < min_chunk_chars:
                if len(text_chunks) > 1 or len(chunk_text) < 30:
                    continue

            cid = mdhash_id(chunk_text)
            if cid in seen_ids:
                continue
            seen_ids.add(cid)

            tok_count = len(tw.encode(chunk_text))

            meta = {
                "id": cid,
                "passage_id": passage_id,
                "context_title": context_title,
                "chunk_index": ci,
                "total_chunks_in_passage": len(text_chunks),
                "tokens": tok_count,
                "path": p.get("path", ""),
                "doc_number": meta_info.get("doc_number", ""),
                "content": chunk_text,
            }
            chunks_meta.append(meta)
            dataset.append(chunk_text)

    return chunks_meta, dataset