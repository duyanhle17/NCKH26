import re, hashlib
from typing import Any, Dict, List
from transformers import AutoTokenizer
from .utils_text import normalize_text

def mdhash_id(text: str, prefix="chunk-") -> str:
    h = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
    return f"{prefix}{h}"

class TokenizerWrapper:
    def __init__(self, tok):
        self.tok = tok
    def encode(self, text: str):
        return self.tok.encode(text, add_special_tokens=False)
    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True)

def split_legal_units(text: str) -> list[str]:
    # vẫn hỗ trợ 1., (a), (i)... (để sau bạn nâng cho VN "Điều/Khoản/Điểm")
    parts = re.split(r"(?im)(?=^\s*(?:\d+\.|\([a-z]\)|\([ivxlcdm]+\))\s+)", text.strip())
    parts = [p.strip() for p in parts if p and p.strip()]
    if len(parts) <= 1:
        return [text.strip()]
    return parts

def pack_units_to_token_chunks(tw: TokenizerWrapper, units: list[str], max_tokens: int, overlap_tokens: int) -> list[list[int]]:
    out_tokens = []
    cur_tokens = []

    for unit in units:
        ut = tw.encode(unit)
        if not ut:
            continue

        if len(ut) > max_tokens:
            if cur_tokens:
                out_tokens.append(cur_tokens)
                cur_tokens = []

            start = 0
            while start < len(ut):
                piece = ut[start:start + max_tokens]
                out_tokens.append(piece)
                if len(piece) <= 1:
                    break
                eff_ov = min(overlap_tokens, len(piece) - 1)
                start += max(1, len(piece) - eff_ov)
            continue

        if len(cur_tokens) + len(ut) <= max_tokens:
            cur_tokens.extend(ut)
        else:
            if cur_tokens:
                out_tokens.append(cur_tokens)

            if overlap_tokens > 0 and out_tokens:
                prev = out_tokens[-1]
                eff_ov = min(overlap_tokens, len(prev) - 1) if len(prev) > 1 else 0
                cur_tokens = prev[-eff_ov:] + ut if eff_ov > 0 else ut
            else:
                cur_tokens = ut

    if cur_tokens:
        out_tokens.append(cur_tokens)

    return out_tokens

def build_chunks(passages: List[Dict[str, Any]], embed_model: str, max_tokens: int, overlap_tokens: int, min_chunk_chars: int):
    tokenizer = AutoTokenizer.from_pretrained(embed_model, use_fast=True)
    tw = TokenizerWrapper(tokenizer)

    chunks_meta = []
    dataset = []
    seen_ids = set()

    for pi, p in enumerate(passages):
        doc_key = f'{p.get("doc_id","doc")}::passage-{pi}'
        text = (p.get("content") or "").strip()
        if not text:
            continue

        units = split_legal_units(text)
        token_chunks = pack_units_to_token_chunks(tw, units, max_tokens=max_tokens, overlap_tokens=overlap_tokens)

        for ci, tk in enumerate(token_chunks):
            chunk_text = normalize_text(tw.decode(tk))
            if len(chunk_text) < min_chunk_chars:
                continue
            cid = mdhash_id(chunk_text)
            if cid in seen_ids:
                continue
            seen_ids.add(cid)

            meta = {
                "id": cid,
                "full_doc_id": doc_key,
                "chunk_order_index": ci,
                "tokens": len(tk),
                "path": p.get("path", ""),
                "content": chunk_text
            }
            chunks_meta.append(meta)
            dataset.append(chunk_text)

    return chunks_meta, dataset