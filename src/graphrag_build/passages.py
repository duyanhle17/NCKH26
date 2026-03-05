"""
Passages module – Bridge between dataset_loader and chunking.

Since dataset_loader now returns whole documents (1 dict per file),
this module simply normalizes the text and passes through.
chunking.py handles all structural parsing internally.
"""
from typing import Any, Dict, List
from .utils_text import normalize_text


MIN_PASSAGE_CHARS: int = 50


def doc_to_passages(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize a single document dict and pass through."""
    text = normalize_text(doc.get("content", ""))

    if len(text) < MIN_PASSAGE_CHARS:
        return []

    return [{
        "doc_id": doc["doc_id"],
        "path": doc.get("path", ""),
        "content": text,
        "metadata": doc.get("metadata", {}),
    }]


def docs_to_passages(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    all_passages: List[Dict[str, Any]] = []
    for d in docs:
        all_passages.extend(doc_to_passages(d))
    return all_passages