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
    print(f"   Tìm thấy {len(docs)} documents (files).")
    
    print("\n2. Đang normalize documents...")
    passages = docs_to_passages(docs)
    print(f"   Sau normalize: {len(passages)} documents.")
        
    print("\n3. Đang chunking (unified parser & accumulator)...")
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
        f.write(f"TỔNG SỐ DOCUMENTS: {len(passages)}\n")
        f.write(f"TỔNG SỐ CHUNKS: {len(dataset)}\n")
        f.write("="*80 + "\n\n")
        
        for i, (meta, chunk_text) in enumerate(zip(chunks_meta, dataset)):
            f.write(f"--- CHUNK #{i+1} ---\n")
            f.write(f"ID: {meta['id']}\n")
            f.write(f"Doc ID: {meta['doc_id']}\n")
            f.write(f"Hierarchy Path: {meta.get('hierarchy_path', '')}\n")
            f.write(f"Semantic Level: {meta.get('semantic_level', '')}\n")
            f.write(f"Start Marker: {meta.get('start_marker', '')}\n")
            f.write(f"End Marker: {meta.get('end_marker', '')}\n")
            f.write(f"Chunk Index: {meta['chunk_index']}\n")
            f.write(f"Tokens: {meta['tokens']}\n")
            f.write(f"Nội dung:\n{chunk_text}\n")
            f.write("-" * 80 + "\n\n")
            
    print(f"Hoàn thành! Bạn hãy mở file {output_file} để xem kết quả chunking.")

    # Xuất thống kê chunk options
    if chunks_meta:
        tokens_list = [meta['tokens'] for meta in chunks_meta if 'tokens' in meta]
        if tokens_list:
            total_chunks = len(tokens_list)
            max_tokens = max(tokens_list)
            min_tokens = min(tokens_list)
            avg_tokens = sum(tokens_list) / total_chunks
            
            # Tính quartile thủ công
            sorted_tokens = sorted(tokens_list)
            
            def get_percentile(data, percentile):
                n = len(data)
                k = (n - 1) * percentile
                f = int(k)
                c = k - f
                if f + 1 < n:
                    return data[f] + (data[f + 1] - data[f]) * c
                else:
                    return data[f]
            
            q1 = get_percentile(sorted_tokens, 0.25)
            median_tokens = get_percentile(sorted_tokens, 0.50)
            q3 = get_percentile(sorted_tokens, 0.75)
            
            summary_file = "chunk_summary.txt"
            with open(summary_file, "w", encoding="utf-8") as fs:
                fs.write("=== BÁO CÁO THỐNG KÊ CHUNKING ===\n")
                fs.write(f"Tổng số chunk:           {total_chunks}\n")
                fs.write(f"Độ dài lớn nhất:         {max_tokens} tokens\n")
                fs.write(f"Độ dài nhỏ nhất:         {min_tokens} tokens\n")
                fs.write(f"Độ dài trung bình:       {avg_tokens:.2f} tokens\n")
                fs.write(f"Trung vị (Median / Q2):  {median_tokens:.2f} tokens\n")
                fs.write(f"Tứ phân vị Q1 (25%):     {q1:.2f} tokens\n")
                fs.write(f"Tứ phân vị Q3 (75%):     {q3:.2f} tokens\n")
                
                # Phân phối theo khoảng token
                ranges = [(0, 400, "Dưới 400"), (400, 800, "400 - 800"), (800, 1000, "800 - 1000"), (1000, 1201, "1000 - 1200")]
                fs.write("\n=== PHÂN PHỐI THEO KHOẢNG TOKEN ===\n")
                for lo, hi, label in ranges:
                    count = sum(1 for t in tokens_list if lo <= t < hi)
                    pct = count / total_chunks * 100
                    fs.write(f"  {label:>12}: {count:>6} chunks ({pct:5.1f}%)\n")

                # Tìm chunk nhỏ nhất và ghi nội dung
                min_idx = tokens_list.index(min_tokens)
                fs.write(f"\n--- CHUNK NHỎ NHẤT (#{min_idx+1}, {min_tokens} tokens) ---\n")
                fs.write(f"ID: {chunks_meta[min_idx]['id']}\n")
                fs.write(f"Hierarchy Path: {chunks_meta[min_idx].get('hierarchy_path', '')}\n")
                fs.write(f"Nội dung:\n{dataset[min_idx]}\n")
            
            print(f"Đã tạo file thống kê: {summary_file}")

if __name__ == "__main__":
    main()
