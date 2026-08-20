"""2000-character word-aligned chunking with optional overlap (design doc §7.1)."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    text: str
    doc_id: str
    chunk_idx: int
    char_start: int
    char_end: int


def _normalize(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n")).strip()


def chunk_document(doc_id: str, text: str, max_chars: int = 2000,
                   overlap_chars: int = 200) -> list[Chunk]:
    """Split at word boundaries: scan to ``max_chars`` then backtrack to the
    last whitespace; next chunk starts ``overlap_chars`` (word-aligned) back."""
    text = _normalize(text)
    chunks: list[Chunk] = []
    start, idx = 0, 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        if end < n:
            back = text.rfind(" ", start, end)
            if back > start:
                end = back
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(id=f"{doc_id}::{idx}", text=chunk_text, doc_id=doc_id,
                                chunk_idx=idx, char_start=start, char_end=end))
            idx += 1
        if end >= n:
            break
        nxt = end - overlap_chars
        if overlap_chars > 0 and nxt > start:
            fwd = text.find(" ", max(nxt, 0))
            nxt = fwd + 1 if (fwd != -1 and fwd + 1 < end) else end + 1
        else:
            nxt = end + 1
        start = max(nxt, start + 1)
    return chunks


_WORD_RE = re.compile(r"[a-z0-9]+")


def corpus_idf(texts: list[str]) -> dict[str, float]:
    """Smoothed IDF over lowercased word tokens (design doc §7.3)."""
    n = len(texts)
    df: Counter[str] = Counter()
    for t in texts:
        df.update(set(_WORD_RE.findall(t.lower())))
    return {w: math.log((1 + n) / (1 + c)) + 1.0 for w, c in df.items()}
