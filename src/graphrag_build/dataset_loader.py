import os
import re
from pathlib import Path
from typing import Any, Dict, List

def list_txt_files(dataset_dir: Path) -> List[Path]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset_dir not found: {dataset_dir.resolve()}")
    files = sorted([p for p in dataset_dir.rglob("*.txt") if p.is_file()])
    if not files:
        raise ValueError(f"No .txt files found in: {dataset_dir.resolve()}")
    return files

def clean_text_advanced(text: str) -> str:
    """
    Tiền xử lý văn bản chuyên sâu:
    - Thay thế nhiều ngắt dòng bằng đoạn \n\n
    - Strip từng dòng
    - Bỏ các dòng toàn ký tự (*)
    - Bỏ khoảng trắng thừa
    """
    text = text.replace("\r", "\n")
    # Thay thế nhiều dấu xuống dòng liên tiếp bằng 2 dấu xuống dòng
    text = re.sub(r'\n\s*\n', '\n\n', text)
    # Loại bỏ khoảng trắng đầu/cuối mỗi dòng
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)
    # Loại bỏ các dòng chỉ chứa dấu *
    text = re.sub(r'\n\*+\n', '\n', text)
    text = re.sub(r'^\*+$', '', text, flags=re.MULTILINE)
    # Loại bỏ khoảng trắng thừa
    text = re.sub(r'[ \t]+', ' ', text)
    # Loại bỏ dòng trống thừa
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def extract_doc_metadata(filename: str, content: str) -> Dict[str, str]:
    """
    Trích xuất metadata từ tên file và nội dung văn bản.
    """
    base_name = os.path.splitext(filename)[0]
    # Loại bỏ hậu tố _1, _2...
    base_name_clean = re.sub(r'_(\d+)$', '', base_name)
    doc_number = base_name_clean.replace('_', '/')

    issuing_body = ""
    for line in content.split('\n')[:10]:
        line = line.strip()
        if line and line != "********" and "CỘNG" not in line and "Độc lập" not in line:
            issuing_body = line
            break

    return {
        "doc_number": doc_number,
        "issuing_body": issuing_body,
    }

def load_txt_documents(dataset_dir: Path) -> List[Dict[str, Any]]:
    docs = []
    files = list_txt_files(dataset_dir)
    
    for fp in files:
        # Xử lý nhiều Encoding khác nhau
        try:
            raw = fp.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            try:
                raw = fp.read_text(encoding='utf-8-sig')
            except UnicodeDecodeError:
                raw = fp.read_text(encoding='latin-1', errors="ignore")
                
        text = clean_text_advanced(raw)
        
        if len(text) < 50:
            continue
            
        rel = str(fp.relative_to(dataset_dir))
        doc_id = fp.stem
        filename = fp.name
        
        meta = extract_doc_metadata(filename, text)
        
        docs.append({
            "doc_id": doc_id,
            "path": f"DATASET::{rel}",
            "content": f"[{rel}]\n{text}",
            "metadata": meta,
        })
        
    if not docs:
        raise ValueError("All txt docs are empty/too short after normalization.")
    return docs