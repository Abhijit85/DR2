#!/usr/bin/env python3
"""Probe dense retrieval recall on HotpotQA-style dev questions.

Expected JSONL schema:
  {"question": str, "gold_titles": [str, ...]}

A hit is counted when any retrieved chunk's inferred article title matches a gold
supporting title. The current local HotpotQA store writes hashed doc ids, so we
recover the title from the chunk text prefix.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ergo import ErgoConfig
from ergo.rag import SentenceTransformerEmbedder, VectorStore


def load_dev(path: str | Path, limit: int):
    with Path(path).open() as fh:
        for i, line in enumerate(fh):
            if i >= limit:
                break
            row = json.loads(line)
            yield row['question'], {_norm(t) for t in row.get('gold_titles', [])}


def _norm(text: str) -> str:
    return re.sub(r'\s+', ' ', text.strip().lower())


def _chunk_title(text: str) -> str:
    line = text.splitlines()[0].strip()
    if '  ' in line:
        line = line.split('  ', 1)[0]
    return _norm(line)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--store', required=True)
    ap.add_argument('--dev', required=True)
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--k', type=int, default=3)
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    cfg = ErgoConfig()
    cfg.retrieval.k = args.k
    store = VectorStore(args.store, 'documents', SentenceTransformerEmbedder(device=args.device))

    total = 0
    hits = 0
    for question, gold_titles in load_dev(args.dev, args.n):
        retrieved = store.fused_search(question, question, cfg.retrieval, exclude=set())
        titles = {_chunk_title(r.chunk.text) for r in retrieved}
        total += 1
        if gold_titles & titles:
            hits += 1

    score = hits / max(total, 1)
    verdict = 'PASS' if score >= 0.6 else 'FAIL'
    print(f'gold-hit@{args.k}={score:.3f}  hits={hits}/{total}  {verdict}')


if __name__ == '__main__':
    main()
