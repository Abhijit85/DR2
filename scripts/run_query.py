#!/usr/bin/env python3
"""Run one query through the full ERGO loop.

GPU (real backbone):
  python scripts/run_query.py --store ./ergo_store --backbone dream \
      --query "Who wrote the paper that introduced masked diffusion LMs?"

CPU smoke test (mock backbone; --facts maps evidence substrings to answers):
  python scripts/run_query.py --backbone mock --query "capital of France" \
      --mock-doc "Paris is the capital of France since 987." \
      --mock-fact "capital of France=Paris"
"""
import argparse
import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ergo import ErgoConfig, ErgoOrchestrator
from ergo.rag import HashEmbedder, VectorStore, chunk_document, corpus_idf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True)
    ap.add_argument("--store", default="./ergo_store")
    ap.add_argument("--backbone", choices=["mock", "llada", "dream", "dream_native", "sardi_dream"], default="dream")
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--mock-doc", action="append", default=[])
    ap.add_argument("--mock-fact", action="append", default=[])
    args = ap.parse_args()

    cfg = ErgoConfig.from_yaml(args.config) if args.config else ErgoConfig()

    if args.backbone == "mock":
        from ergo.backbones import MockBackbone
        from ergo.tokenization import SimpleTokenizer
        tok = SimpleTokenizer()
        facts = dict(f.split("=", 1) for f in args.mock_fact)
        backbone = MockBackbone(tok, facts)
        embedder = HashEmbedder()
        store = VectorStore("./mock_store", "documents", embedder)
        chunks = [c for i, d in enumerate(args.mock_doc)
                  for c in chunk_document(f"doc{i}", d)]
        if chunks:
            store.add(chunks)
        idf = corpus_idf([c.text for c in chunks])
        cfg.canvas.len_thought, cfg.canvas.len_answer = 16, 12
        cfg.canvas.len_observation, cfg.canvas.len_critique = 24, 12
        cfg.canvas.len_action_input = 8
    else:
        from ergo.backbones import load_dream, load_dream_native, load_llada, load_sardi_dream
        if args.backbone == "dream":
            backbone = load_dream(device=args.device, adapter_path=args.adapter)
        elif args.backbone == "dream_native":
            backbone = load_dream_native(device=args.device, adapter_path=args.adapter)
        elif args.backbone == "sardi_dream":
            backbone = load_sardi_dream(device=args.device, adapter_path=args.adapter)
        else:
            backbone = load_llada(device=args.device, adapter_path=args.adapter)
        tok = backbone.tok
        from ergo.rag import SentenceTransformerEmbedder
        embedder = SentenceTransformerEmbedder(device=None)
        store = VectorStore(args.store, "documents", embedder)
        idf_path = Path(args.store, "idf.json")
        idf = json.loads(idf_path.read_text()) if idf_path.exists() else {}

    orch = ErgoOrchestrator(backbone, tok, store, cfg, idf=idf)
    result = orch.run(args.query)

    print(f"\nANSWER: {result.answer}")
    print(f"terminated_by={result.terminated_by}  retrievals={result.n_retrievals}")
    for log in result.cycles:
        rb = f" rollback={log.rollback.n_rolled}/{log.rollback.n_committed}" if log.rollback else ""
        gap = f" gap_fire={log.gap.fire}({log.gap.rule})" if log.gap else ""
        print(f"  cycle {log.cycle}: action={log.action} conf={log.mean_answer_conf:.2f}"
              f" rel={log.mean_answer_rel:.2f}{gap}{rb} chunks={log.retrieved_ids}")


if __name__ == "__main__":
    main()
