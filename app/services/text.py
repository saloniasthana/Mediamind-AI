import re
from collections import Counter


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9]+", text.lower()) if t not in STOPWORDS]


def split_chunks(text: str, size: int = 700, overlap: int = 120) -> list[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    chunks: list[str] = []
    step = max(size - overlap, 1)
    for start in range(0, len(clean), step):
        chunks.append(clean[start : start + size])
        if start + size >= len(clean):
            break
    return chunks


def summarize_text(text: str, max_sentences: int = 4) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip())
    sentences = [s for s in sentences if s]
    if len(sentences) <= max_sentences:
        return " ".join(sentences)
    frequencies = Counter(tokenize(text))
    ranked = sorted(
        enumerate(sentences),
        key=lambda item: sum(frequencies[t] for t in tokenize(item[1])),
        reverse=True,
    )
    selected = sorted(index for index, _ in ranked[:max_sentences])
    return " ".join(sentences[index] for index in selected)


def keyword_score(query: str, text: str) -> float:
    query_terms = set(tokenize(query))
    text_terms = set(tokenize(text))
    if not query_terms or not text_terms:
        return 0.0
    return len(query_terms & text_terms) / len(query_terms)
