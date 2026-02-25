import re
from typing import Any, Dict, List
from .utils_text import normalize_text

# Bạn có thể giữ các regex Article/Chapter như cũ,
# nhưng dataset VN có thể không có "Article". Vậy ta dùng fallback an toàn.

PAGE_MARK = re.compile(r"^---\s*end\s*of\s*page\s*=\s*\d+\s*---\s*$", re.IGNORECASE)

def doc_to_passages(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Split theo đoạn trống lớn để tạo passages.
    Nếu không split được thì trả về 1 passage FULL_DOC.
    """
    text = doc["content"]
    parts = [normalize_text(p) for p in re.split(r"\n{2,}", text) if normalize_text(p)]
    passages: List[Dict[str, Any]] = []

    # ghép lại thành blocks đủ dài (tránh quá vụn)
    buf = []
    cur_len = 0
    def flush(idx: int):
        nonlocal buf, cur_len
        if not buf:
            return
        block = normalize_text("\n\n".join(buf))
        if len(block) >= 200:
            passages.append({
                "doc_id": doc["doc_id"],
                "path": f'{doc["path"]} > PASSAGE_{idx}',
                "content": block,
            })
        buf = []
        cur_len = 0

    pi = 0
    for p in parts:
        if cur_len + len(p) <= 3500:
            buf.append(p)
            cur_len += len(p)
        else:
            flush(pi)
            pi += 1
            buf = [p]
            cur_len = len(p)

    flush(pi)

    if not passages:
        passages.append({
            "doc_id": doc["doc_id"],
            "path": f'{doc["path"]} > FULL_DOC',
            "content": normalize_text(text),
        })
    return passages

def docs_to_passages(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    all_passages: List[Dict[str, Any]] = []
    for d in docs:
        all_passages.extend(doc_to_passages(d))
    return all_passages