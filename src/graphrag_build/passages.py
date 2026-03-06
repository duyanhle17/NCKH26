import re
from typing import Any, Dict, List
from .utils_text import normalize_text

# ── Public import from chunking  (lazy to avoid circular) ─────────────────────
def _get_legal_splitters():
    from .chunking import strip_header_footer, split_legal_units
    return strip_header_footer, split_legal_units


# Maximum characters we allow in a single passage before we force a new one.
# Kept generous so that a full Điều (which may be long) fits in one passage.
MAX_PASSAGE_CHARS: int = 6000
# Minimum chars for a passage to be worth keeping
MIN_PASSAGE_CHARS: int = 120


def doc_to_passages(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Split a document into semantically meaningful passages.

    Strategy (fixes the old char-count slicing that broke Điều/Khoản):
    1. Strip header / footer noise.
    2. Split by legal structure hierarchy (Phần > Chương > Điều > Khoản > Điểm).
    3. Pack consecutive units into passages that stay under MAX_PASSAGE_CHARS
       WITHOUT ever cutting mid-unit: if a single unit already exceeds the
       budget, it becomes its own passage (oversized but semantically complete).
    """
    strip_header_footer, split_legal_units = _get_legal_splitters()

    raw = doc["content"]
    text = normalize_text(raw)

    # 1. Strip header / footer
    clean = strip_header_footer(text)
    if not clean:
        clean = text  # fallback: keep original

    # 2. Split into legal units
    units = split_legal_units(clean)

    passages: List[Dict[str, Any]] = []
    buf: List[str] = []
    buf_len: int = 0
    pi: int = 0

    def flush():
        nonlocal buf, buf_len, pi
        if not buf:
            return
        block = normalize_text("\n\n".join(buf))
        if len(block) >= MIN_PASSAGE_CHARS:
            passages.append({
                "doc_id": doc["doc_id"],
                "path": f'{doc["path"]} > PASSAGE_{pi}',
                "content": block,
            })
            pi += 1
        buf = []
        buf_len = 0

    for unit in units:
        unit = normalize_text(unit)
        if not unit:
            continue
        ulen = len(unit)

        if ulen > MAX_PASSAGE_CHARS:
            # Oversized unit (e.g. long Điều or corrupt appendix block):
            # flush current buffer first, then push unit as its own passage.
            flush()
            block = normalize_text(unit)
            if len(block) >= MIN_PASSAGE_CHARS:
                passages.append({
                    "doc_id": doc["doc_id"],
                    "path": f'{doc["path"]} > PASSAGE_{pi}',
                    "content": block,
                })
                pi += 1
        elif buf_len + ulen > MAX_PASSAGE_CHARS:
            flush()
            buf = [unit]
            buf_len = ulen
        else:
            buf.append(unit)
            buf_len += ulen

    flush()

    # Fallback: no passages created (very short doc)
    if not passages:
        block = normalize_text(clean)
        if len(block) >= MIN_PASSAGE_CHARS:
            passages.append({
                "doc_id": doc["doc_id"],
                "path": f'{doc["path"]} > FULL_DOC',
                "content": block,
            })

    return passages


def docs_to_passages(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    all_passages: List[Dict[str, Any]] = []
    for d in docs:
        all_passages.extend(doc_to_passages(d))
    return all_passages