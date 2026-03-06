import re
from typing import Any, Dict, List
from .utils_text import normalize_text
from .chunking import strip_header_footer, split_legal_units

MAX_PASSAGE_CHARS = 10000
MIN_PASSAGE_CHARS = 100

def doc_to_passages(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Pass-through for the Unified Parser.
    Just normalizes the text and passes the whole document to chunking.py.
    """
    raw = doc.get("content", "")
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
                "metadata": doc.get("metadata", {}),
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
                    "metadata": doc.get("metadata", {}),
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
                "metadata": doc.get("metadata", {}),
            })

    return passages


def docs_to_passages(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    all_passages: List[Dict[str, Any]] = []
    for d in docs:
        all_passages.extend(doc_to_passages(d))
    return all_passages