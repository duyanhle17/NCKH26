import json

with open("artifact_faiss/chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

print(f"Total chunks: {len(chunks)}")

# Print ALL chunks that contain any reference to PHU LUC 
for i, c in enumerate(chunks):
    text = c if isinstance(c, str) else str(c)
    lower = text.lower()
    if "phụ lục" in lower or "63,5" in text or "63.5" in text or "định mức" in lower or "lắp đặt điện" in lower:
        print(f"\n{'='*80}")
        print(f"CHUNK [{i}] (len={len(text)})")
        print(f"{'='*80}")
        print(text[:800])
        if len(text) > 800:
            print(f"\n... (truncated, total {len(text)} chars)")

print("\n\n--- Also check chunks_meta.json for PHU LUC passages ---")
with open("artifact_faiss/chunks_meta.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

for m in meta:
    if "phụ lục" in m.get("passage_id","").lower() or "phụ lục" in m.get("context_title","").lower():
        print(f"  Chunk idx={m.get('chunk_index')}, passage_id={m.get('passage_id')}, title={m.get('context_title')}, tokens={m.get('n_tokens')}")
