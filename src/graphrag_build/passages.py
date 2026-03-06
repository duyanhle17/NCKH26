import re
from typing import Any, Dict, List
from .utils_text import normalize_text

def doc_to_passages(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Pass-through for the Unified Parser.
    Just normalizes the text and passes the whole document to chunking.py.
    """
    raw = doc.get("content", "")
    text = normalize_text(raw)
    
    # Return as a single "passage" so chunking.py can handle the full context
    return [{
        "doc_id": doc["doc_id"],
        "path": doc["path"],
        "content": text,
    }]

def docs_to_passages(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    all_passages: List[Dict[str, Any]] = []
    for d in docs:
        all_passages.extend(doc_to_passages(d))
    return all_passages