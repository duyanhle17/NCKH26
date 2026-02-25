import re
from typing import Dict, List, Set, Tuple, Any
import networkx as nx

STOP_EN = set("""
the a an and or of to in on for with by from as at into over under about
shall should may must means meaning
""".split())

LEGAL_MARKERS = [
    r"(Article\s+\d+)",
    r"(CHAPTER\s+[IVXLCDM]+)",
    r"(SECTION\s+\d+)",
    r"(ANNEX\b[^.\n]{0,80})",
    r"(Regulation\s+\(EU\)\s*\d{4}/\d+)",
]
DEFINITION_PATS = [
    r'“([^”]{2,120})”\s+means\s+([^.;]{10,500})',
    r'"([^"]{2,120})"\s+means\s+([^.;]{10,500})',
    r'“([^”]{2,120})”\s+shall\s+mean\s+([^.;]{10,500})',
    r'"([^"]{2,120})"\s+shall\s+mean\s+([^.;]{10,500})',
]

MIN_TERM_LEN = 3
MAX_TERM_WORDS = 10

def norm_term(t: str) -> str:
    t = (t or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = t.strip("“”\"'’ ")
    return t

def extract_markers(text: str) -> List[str]:
    out = []
    for pat in LEGAL_MARKERS:
        out.extend([x.strip() for x in re.findall(pat, text, flags=re.IGNORECASE)])
    return out

def extract_definitions(text: str) -> List[Tuple[str, str]]:
    defs = []
    for pat in DEFINITION_PATS:
        for m in re.finditer(pat, text, flags=re.IGNORECASE):
            term = m.group(1).strip()
            dfn  = m.group(2).strip()
            if 2 < len(term) < 120 and 10 < len(dfn) < 800:
                defs.append((term, dfn))
    return defs

def extract_terms_yake(text: str, top_k: int, max_term_words: int) -> List[str]:
    try:
        import yake
        kw = yake.KeywordExtractor(lan="en", n=max_term_words, top=top_k, dedupLim=0.9, windowsSize=1)
        kws = kw.extract_keywords(text)
        kws = sorted(kws, key=lambda x: x[1])[:top_k]
        return [k.strip() for k,_ in kws if k and len(k.strip()) >= MIN_TERM_LEN]
    except Exception:
        return []

def classify_type(term: str) -> str:
    t = term.lower()
    if re.match(r"^article\s+\d+", t): return "MARKER"
    if re.match(r"^chapter\s+", t): return "MARKER"
    if re.match(r"^section\s+\d+", t): return "MARKER"
    if re.match(r"^annex\b", t): return "MARKER"
    return "TERM"

def extract_entities(text: str, chunk_idx: int, top_k: int, max_term_words: int) -> Dict[str, Any]:
    markers = extract_markers(text)
    defs = extract_definitions(text)
    yk = extract_terms_yake(text, top_k=top_k, max_term_words=max_term_words)

    raw = []
    raw.extend(markers)
    raw.extend([t for t,_ in defs])
    raw.extend(yk)

    terms = []
    seen = set()
    for t in raw:
        tc = norm_term(t)
        if not tc:
            continue
        if len(tc) < MIN_TERM_LEN:
            continue
        if len(tc.split()) > max_term_words:
            continue
        if tc in STOP_EN:
            continue
        if tc.isdigit():
            continue
        if tc not in seen:
            seen.add(tc)
            terms.append(tc)
        if len(terms) >= top_k:
            break

    entities = [{"term": t, "type": classify_type(t), "chunk_idx": chunk_idx} for t in terms]
    return {"terms": terms, "entities": entities, "definitions": defs}

def build_kg(
    dataset: List[str],
    top_k_terms_per_chunk: int,
    max_term_words: int,
    cooc_window: int,
    prune_min_cooc_weight: int
) -> Tuple[nx.DiGraph, List[Set[str]], Dict[str, Set[int]]]:

    kg = nx.DiGraph()
    chunk_entities: List[Set[str]] = []
    entity_to_chunks: Dict[str, Set[int]] = {}

    for i, chunk in enumerate(dataset):
        ex = extract_entities(chunk, chunk_idx=i, top_k=top_k_terms_per_chunk, max_term_words=max_term_words)
        ents = set(ex["terms"])
        chunk_entities.append(ents)

        for t in ents:
            entity_to_chunks.setdefault(t, set()).add(i)

        for e in ex["entities"]:
            t = e["term"]
            if not kg.has_node(t):
                kg.add_node(t, type=e.get("type", "TERM"))

        for term, dfn in ex["definitions"]:
            a = norm_term(term)
            if not a:
                continue
            if not kg.has_node(a):
                kg.add_node(a, type="DEF_TERM")
            prev = kg.nodes[a].get("definition", "")
            if len(dfn) > len(prev):
                kg.nodes[a]["definition"] = dfn

        ordered = ex["terms"]
        for a_idx, a in enumerate(ordered):
            for b_idx in range(a_idx+1, min(len(ordered), a_idx+1+cooc_window)):
                b = ordered[b_idx]
                if a == b:
                    continue
                if kg.has_edge(a, b):
                    kg[a][b]["weight"] = kg[a][b].get("weight", 1) + 1
                else:
                    kg.add_edge(a, b, relation="co_occurs_with", weight=1, source_chunk=i)

    kg.remove_edges_from([
        (u, v) for u, v, d in kg.edges(data=True)
        if d.get("relation") == "co_occurs_with" and d.get("weight", 1) < prune_min_cooc_weight
    ])
    return kg, chunk_entities, entity_to_chunks