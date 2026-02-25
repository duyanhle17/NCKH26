import re

def normalize_text(s: str) -> str:
    s = (s or "").replace("\r", "\n")
    s = s.encode("utf-8", errors="surrogatepass").decode("utf-8", errors="ignore")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()