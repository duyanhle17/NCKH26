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

def split_document_into_passages(text: str, doc_id: str, rel: str, filename: str, meta: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Chia văn bản gốc thành các đoạn (passages) dựa trên cấu trúc văn bản pháp luật.
    Lưu giữ ngữ cảnh cấu trúc (Phụ lục, Phần, Chương, Mục, Điều) để làm title cho các chunk.
    """
    # Nhận diện cấp bậc tiêu đề
    pat_phuluc = re.compile(r'^\s*(PHỤ\s+LỤC|Phụ\s+lục)[\s:0-9A-Za-z\-\.]*', re.IGNORECASE)
    pat_phan = re.compile(r'^\s*(PHẦN|Phần)\s+([IVXLCDM]+|\d+)', re.IGNORECASE)
    pat_chuong = re.compile(r'^\s*(CHƯƠNG|Chương)\s+([IVXLCDM]+|\d+)', re.IGNORECASE)
    pat_muc = re.compile(r'^\s*(MỤC|Mục)\s+(\d+|[IVXLCDM]+)', re.IGNORECASE)
    pat_dieu = re.compile(r'^\s*(ĐIỀU|Điều)\s+\d+', re.IGNORECASE)
    # Old-style Roman numeral sections: "I-", "II-", "III.", "IV." (uppercase only, followed by - or .)
    pat_roman_section = re.compile(r'^\s*((?:IX|IV|V?I{1,3})[\-\.]\s+[A-ZÀ-Ỹ])', re.MULTILINE)
    
    passages = []
    lines = text.split('\n')
    
    # State tracking
    context = {"Phụ lục": "", "Phần": "", "Chương": "", "Mục": "", "Điều": ""}
    
    current_buffer = []
    
    def flush():
        content_text = "\n".join(current_buffer).strip()
        current_buffer.clear()
        
        if len(content_text) < 10:
            return
            
        parts = []
        if context["Phụ lục"]: parts.append(context["Phụ lục"])
        if context["Phần"]: parts.append(context["Phần"])
        if context["Chương"]: parts.append(context["Chương"])
        if context["Mục"]: parts.append(context["Mục"])
        if context["Điều"]: parts.append(context["Điều"])
        
        title = " - ".join(parts) if parts else "Phần mở đầu"
        
        doc_meta = meta.copy()
        doc_meta["context_title"] = title
        
        passage_id = title.replace(' ', '_').replace('-', '_')
        passage_id = re.sub(r'[^A-Za-z0-9_À-ỹ]', '', passage_id)
        if not passage_id:
            passage_id = "Phan_mo_dau"
            
        passages.append({
            "doc_id": f"{doc_id}_{passage_id}",
            "path": f"DATASET::{rel}",
            "content": f"[{filename}]\n[ID Passage: {title}]\n{content_text}",
            "metadata": doc_meta,
        })

    for line in lines:
        stripped = line.strip()
        if not stripped:
            current_buffer.append(line)
            continue
            
        is_heading = False
        
        if pat_phuluc.match(stripped):
            flush()
            context["Phụ lục"] = stripped
            context["Phần"] = ""
            context["Chương"] = ""
            context["Mục"] = ""
            context["Điều"] = ""
            is_heading = True
        elif pat_phan.match(stripped):
            flush()
            context["Phần"] = stripped
            context["Chương"] = ""
            context["Mục"] = ""
            context["Điều"] = ""
            is_heading = True
        elif pat_chuong.match(stripped):
            flush()
            context["Chương"] = stripped
            context["Mục"] = ""
            context["Điều"] = ""
            is_heading = True
        elif pat_muc.match(stripped):
            flush()
            context["Mục"] = stripped
            context["Điều"] = ""
            is_heading = True
        elif pat_roman_section.match(stripped):
            flush()
            context["Mục"] = stripped
            context["Điều"] = ""
            is_heading = True
        elif pat_dieu.match(stripped):
            flush()
            context["Điều"] = stripped
            is_heading = True
            
        current_buffer.append(line)
        
    flush()
    return passages

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
        
        # Gọi hàm chia văn bản theo cấu trúc
        file_passages = split_document_into_passages(text, doc_id, rel, filename, meta)
        docs.extend(file_passages)
        
    if not docs:
        raise ValueError("All txt docs are empty/too short after normalization.")
    return docs