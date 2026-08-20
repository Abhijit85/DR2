#!/usr/bin/env python3
"""Ingest a corpus into ChromaDB + IDF sidecar (design doc §7, build step 1).

Usage:
  python scripts/ingest_corpus.py --input docs/ --store ./ergo_store
  python scripts/ingest_corpus.py --input corpus.jsonl --store ./ergo_store \
      --embedder bge   # requires sentence-transformers + GPU/CPU model download

Input: a directory of .txt/.md files, or a JSONL with {"id":..., "text":...}.
"""
import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ergo.rag import HashEmbedder, VectorStore, chunk_document, corpus_idf


def load_docs(path: Path):
    if path.is_dir():
        for p in sorted(path.rglob("*")):
            if p.suffix.lower() in (".txt", ".md"):
                yield p.stem, p.read_text(errors="ignore")
    else:
        for line in path.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                yield str(d["id"]), d["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--store", default="./ergo_store")
    ap.add_argument("--collection", default="documents")
    ap.add_argument("--chunk-chars", type=int, default=2000)
    ap.add_argument("--overlap-chars", type=int, default=200)
    ap.add_argument("--embedder", choices=["hash", "bge", "minilm"], default="bge")
    args = ap.parse_args()

    if args.embedder == "hash":
        embedder = HashEmbedder()
    else:
        from ergo.rag import SentenceTransformerEmbedder
        name = ("BAAI/bge-base-en-v1.5" if args.embedder == "bge"
                else "sentence-transformers/all-MiniLM-L6-v2")
        embedder = SentenceTransformerEmbedder(name)

    store = VectorStore(args.store, args.collection, embedder)
    all_chunks, texts = [], []
    for doc_id, text in load_docs(Path(args.input)):
        chunks = chunk_document(doc_id, text, args.chunk_chars, args.overlap_chars)
        all_chunks.extend(chunks)
        texts.extend(c.text for c in chunks)
    store.add(all_chunks)
    idf = corpus_idf(texts)
    Path(args.store, "idf.json").write_text(json.dumps(idf))
    print(f"ingested {len(all_chunks)} chunks -> {args.store} "
          f"(collection={args.collection}); idf.json written "
          f"({len(idf)} terms)")


if __name__ == "__main__":
    main()
