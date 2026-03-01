"""
Passages module – Bridge between dataset_loader and chunking.

Since dataset_loader now splits documents into structured passages
(by Phần/Chương/Mục/Điều/Phụ lục) with context_title metadata,
this module simply normalizes the text and passes through.
"""
from typing import Any, Dict, List
from .utils_text import normalize_text


# Minimum chars for a passage to be worth keeping
MIN_PASSAGE_CHARS: int = 50


def doc_to_passages(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert a single document (already structured by dataset_loader)
    into a list of passages.

    Since dataset_loader already splits by legal structure, we just
    normalize and pass through. Each doc IS already a passage.
    """
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