import sys
from pathlib import Path

# Thêm thư mục src vào sys.path để import
sys.path.insert(0, "src")

from graphrag_build.dataset_loader import load_txt_documents
from graphrag_build.passages import docs_to_passages
from graphrag_build.chunking import build_chunks
from graphrag_build.config import BuildConfig

def main():
    config = BuildConfig()
    
    print("1. Đang load documents...")
    docs = load_txt_documents(Path("./dataset"))
    print(f"   Tìm thấy {len(docs)} documents.")
    
    print("\n2. Đang chia thành passages (dựa trên cấu trúc)...")
    passages = docs_to_passages(docs)
    print(f"   Tạo được {len(passages)} passages.")
    
    meta_info_list = []
    for p in passages:
        title = p.get('metadata', {}).get('context_title', 'No Title')
        meta_info_list.append(title)
        
    print("\n3. Đang chunking (theo token budget)...")
    # Gọi hàm build_chunks từ chunking.py
    chunks_meta, dataset = build_chunks(
        passages=passages,
        embed_model=config.embed_model,
        max_tokens=config.max_token_size,
        overlap_tokens=config.overlap_token_size,
        min_chunk_chars=config.min_chunk_chars
    )
    
    print(f"   Tạo được {len(dataset)} chunks.")
    
    output_file = "test_chunks_output.txt"
    print(f"\n4. Ghi chi tiết các chunks ra file {output_file} để dễ xem...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"TỔNG SỐ PASSAGES: {len(passages)}\n")
        f.write(f"TỔNG SỐ CHUNKS: {len(dataset)}\n")
        f.write("="*80 + "\n\n")
        
        for i, (meta, chunk_text) in enumerate(zip(chunks_meta, dataset)):
            f.write(f"--- CHUNK #{i+1} ---\n")
            f.write(f"ID: {meta['id']}\n")
            f.write(f"Passage ID: {meta['passage_id']}\n")
            f.write(f"Context Title: {meta.get('context_title', '')}\n")
            f.write(f"Chunk Index: {meta['chunk_index']}/{meta['total_chunks_in_passage']-1}\n")
            f.write(f"Tokens: {meta['tokens']}\n")
            f.write(f"Nội dung:\n{chunk_text}\n")
            f.write("-" * 80 + "\n\n")
            
    print(f"Hoàn thành! Bạn hãy mở file {output_file} để xem kết quả chunking.")

if __name__ == "__main__":
    main()
