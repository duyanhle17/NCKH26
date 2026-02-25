from pathlib import Path
from typing import Any, Dict, List, Tuple
from .utils_text import normalize_text

def list_txt_files(dataset_dir: Path) -> List[Path]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir.resolve()}")
    files = sorted([p for p in dataset_dir.rglob("*.txt") if p.is_file()])
    if not files:
        raise ValueError(f"No .txt files found in: {dataset_dir.resolve()}")
    return files

def load_txt_documents(dataset_dir: Path) -> List[Dict[str, Any]]:
    """
    Output: list of dict(doc_id, path, content)
    - doc_id: file stem
    - path: relative path inside dataset_dir
    - content: normalized full text with header prefix for traceability
    """
    docs = []
    files = list_txt_files(dataset_dir)
    for fp in files:
        raw = fp.read_text(encoding="utf-8", errors="ignore")
        text = normalize_text(raw)
        if len(text) < 50:
            continue
        rel = str(fp.relative_to(dataset_dir))
        doc_id = fp.stem
        docs.append({
            "doc_id": doc_id,
            "path": f"DATASET::{rel}",
            "content": f"[{rel}]\n{text}",
        })
    if not docs:
        raise ValueError("All txt docs are empty/too short after normalization.")
    return docs