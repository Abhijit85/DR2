"""Fused-relevance retrieval over ChromaDB (design doc §5 step 5, §7.4).

rel(chunk) = alpha * cos(e_chunk, e_query) + beta * cos(e_chunk, e_salience)
then MMR diversification, permanent dedup against already-injected chunk ids.
BM25 mode mirrors SARDI's retriever for parity runs.

The embedder is injectable: any callable ``list[str] -> np.ndarray``. Use
``SentenceTransformerEmbedder`` in production and ``HashEmbedder`` in tests.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from ..config import RetrievalConfig
from .chunker import Chunk


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder — CPU-only tests/dev."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def __call__(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim))
        for i, t in enumerate(texts):
            for w in t.lower().split():
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
                out[i, (h // self.dim) % self.dim] += 0.5
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", device: str | None = None):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)

    def __call__(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True))


@dataclass
class Retrieved:
    chunk: Chunk
    score: float


class VectorStore:
    """Thin ChromaDB wrapper with client-side fusion (two queries + merge)."""

    def __init__(self, path: str, collection: str, embedder):
        import chromadb
        self.client = chromadb.PersistentClient(path=path)
        self.col = self.client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"})
        self.embed = embedder

    def add(self, chunks: list[Chunk]) -> None:
        B = 256
        for i in range(0, len(chunks), B):
            batch = chunks[i:i + B]
            self.col.add(
                ids=[c.id for c in batch],
                documents=[c.text for c in batch],
                embeddings=self.embed([c.text for c in batch]).tolist(),
                metadatas=[{"doc_id": c.doc_id, "chunk_idx": c.chunk_idx,
                            "char_start": c.char_start, "char_end": c.char_end}
                           for c in batch],
            )

    def count(self) -> int:
        return self.col.count()

    def _query(self, emb: np.ndarray, n: int) -> dict[str, tuple[Chunk, float]]:
        n = min(n, max(self.col.count(), 1))
        res = self.col.query(query_embeddings=[emb.tolist()], n_results=n,
                             include=["documents", "metadatas", "distances"])
        out = {}
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                        res["metadatas"][0], res["distances"][0]):
            chunk = Chunk(id=cid, text=doc, doc_id=meta["doc_id"],
                          chunk_idx=meta["chunk_idx"], char_start=meta["char_start"],
                          char_end=meta["char_end"])
            out[cid] = (chunk, 1.0 - float(dist))   # cosine distance -> similarity
        return out

    def fused_search(self, query: str, salience_text: str, cfg: RetrievalConfig,
                     exclude: set[str]) -> list[Retrieved]:
        n_cand = cfg.k * cfg.candidates_multiplier
        by_q = self._query(self.embed([query])[0], n_cand)
        by_s = (self._query(self.embed([salience_text])[0], n_cand)
                if salience_text.strip() else {})
        fused: dict[str, tuple[Chunk, float]] = {}
        for cid in set(by_q) | set(by_s):
            chunk = (by_q.get(cid) or by_s.get(cid))[0]
            sq = by_q.get(cid, (None, 0.0))[1]
            ss = by_s.get(cid, (None, 0.0))[1]
            fused[cid] = (chunk, cfg.alpha * sq + cfg.beta * ss)
        pool = [Retrieved(c, s) for cid, (c, s) in fused.items() if cid not in exclude]
        pool.sort(key=lambda r: -r.score)
        return _mmr(pool, self.embed, cfg.mmr_lambda, cfg.k)


def _mmr(pool: list[Retrieved], embed, lam: float, k: int) -> list[Retrieved]:
    if len(pool) <= k:
        return pool[:k]
    embs = embed([r.chunk.text for r in pool])
    chosen: list[int] = []
    while len(chosen) < k and len(chosen) < len(pool):
        best_i, best_v = -1, -np.inf
        for i in range(len(pool)):
            if i in chosen:
                continue
            div = max((float(embs[i] @ embs[j]) for j in chosen), default=0.0)
            v = lam * pool[i].score - (1 - lam) * div
            if v > best_v:
                best_i, best_v = i, v
        chosen.append(best_i)
    return [pool[i] for i in chosen]


class BM25Store:
    """SARDI-parity sparse retriever (rank_bm25); same fused_search interface."""

    def __init__(self, chunks: list[Chunk]):
        from rank_bm25 import BM25Okapi
        self.chunks = chunks
        self.bm25 = BM25Okapi([c.text.lower().split() for c in chunks])

    def fused_search(self, query: str, salience_text: str, cfg: RetrievalConfig,
                     exclude: set[str]) -> list[Retrieved]:
        q = (query + " " + salience_text).lower().split()
        scores = self.bm25.get_scores(q)
        order = np.argsort(-scores)
        out = []
        for i in order:
            if self.chunks[i].id in exclude:
                continue
            out.append(Retrieved(self.chunks[i], float(scores[i])))
            if len(out) >= cfg.k:
                break
        return out
