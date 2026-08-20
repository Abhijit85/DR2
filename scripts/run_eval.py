#!/usr/bin/env python3
"""Benchmark runner skeleton for the 5 SARDI-parity datasets (blueprint §6).

Datasets load via HuggingFace `datasets`; per-dataset corpora must be ingested
first with ingest_corpus.py. This is deliberately a thin skeleton: dataset-
specific corpus preparation (esp. CofCA / SynthWorlds-RM counterfactual
corpora) is left as clearly-marked TODOs for the experiment phase.

  python scripts/run_eval.py --dataset hotpotqa --backbone dream \
      --store ./stores/hotpotqa --limit 200 --out results/hotpotqa_dream.jsonl
"""
import argparse
import json
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ergo import ErgoConfig, ErgoOrchestrator
from ergo.metrics import copy_rate, exact_match, f1

DATASETS = {
    # name -> (hf id, config, question field, answer field)
    "hotpotqa": ("hotpot_qa", "distractor", "question", "answer"),
    "2wikimultihopqa": ("xanhho/2WikiMultihopQA", None, "question", "answer"),
    "musique": ("dgslibisey/MuSiQue", None, "question", "answer"),
    # TODO(counterfactual): CofCA and SynthWorlds-RM per SARDI's released corpora
    "cofca": (None, None, "question", "answer"),
    "synthworlds-rm": (None, None, "question", "answer"),
}


def load_items(name: str, split: str, limit: int):
    hf_id, hf_cfg, qf, af = DATASETS[name]
    if hf_id is None:
        raise SystemExit(f"{name}: wire up SARDI's released corpus/questions first (TODO)")
    from datasets import load_dataset
    ds = load_dataset(hf_id, hf_cfg, split=split) if hf_cfg else load_dataset(hf_id, split=split)
    for i, row in enumerate(ds):
        if i >= limit:
            break
        gold = row[af]
        if isinstance(gold, dict):   # some sets nest answers
            gold = gold.get("text", gold.get("answer", ""))
        if isinstance(gold, list):
            gold = gold[0] if gold else ""
        yield row[qf], str(gold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--backbone", choices=["llada", "dream"], required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = ErgoConfig.from_yaml(args.config) if args.config else ErgoConfig()
    cfg.seed = args.seed

    from ergo.backbones import load_dream, load_llada
    backbone = (load_dream(device=args.device) if args.backbone == "dream"
                else load_llada(device=args.device))
    from ergo.rag import SentenceTransformerEmbedder, VectorStore
    store = VectorStore(args.store, "documents", SentenceTransformerEmbedder())
    idf_path = Path(args.store, "idf.json")
    idf = json.loads(idf_path.read_text()) if idf_path.exists() else {}
    orch = ErgoOrchestrator(backbone, backbone.tok, store, cfg, idf=idf)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ems, f1s = [], []
    with out.open("w") as fh:
        for q, gold in load_items(args.dataset, args.split, args.limit):
            t0 = time.time()
            res = orch.run(q, seed=args.seed)
            ctx = "\n".join(it.text for it in res.evidence.in_context())
            rec = {
                "question": q, "gold": gold, "pred": res.answer,
                "em": exact_match(res.answer, gold), "f1": f1(res.answer, gold),
                "copy_rate": copy_rate(res.answer, ctx),
                "n_retrievals": res.n_retrievals,
                "terminated_by": res.terminated_by,
                "rollbacks": [l.rollback.n_rolled for l in res.cycles if l.rollback],
                "gap_fires": [bool(l.gap.fire) for l in res.cycles if l.gap],
                "wall_s": round(time.time() - t0, 2),
            }
            ems.append(rec["em"]); f1s.append(rec["f1"])
            fh.write(json.dumps(rec) + "\n")
    n = len(ems)
    print(f"{args.dataset} n={n}  EM={sum(ems)/max(n,1):.3f}  F1={sum(f1s)/max(n,1):.3f}")


if __name__ == "__main__":
    main()
