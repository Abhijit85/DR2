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
import hashlib
import json
import time
from pathlib import Path

import pyarrow.parquet as pq

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ergo import ErgoConfig, ErgoOrchestrator

CALIBRATION_CODE_MARKER = "absorb_fix1"
from ergo.metrics import copy_rate, exact_match, f1
from ergo.trigger import ConformalCalibrator

DATASETS = {
    # name -> (hf id, config, question field, answer field)
    "hotpotqa": ("hotpot_qa", "distractor", "question", "answer"),
    "2wikimultihopqa": ("xanhho/2WikiMultihopQA", None, "question", "answer"),
    "musique": ("dgslibisey/MuSiQue", None, "question", "answer"),
    # TODO(counterfactual): CofCA and SynthWorlds-RM per SARDI's released corpora
    "cofca": (None, None, "question", "answer"),
    "synthworlds-rm": (None, None, "question", "answer"),
}

LOCAL_DATASET_FILES = {
    ("hotpotqa", "validation"): Path("datasets/hotpot_qa/distractor/validation-00000-of-00001.parquet"),
    ("hotpotqa", "train"): Path("datasets/hotpot_qa/distractor/train-00000-of-00002.parquet"),
}


def calibration_signature(cfg: ErgoConfig) -> dict:
    payload = {
        "code_marker": CALIBRATION_CODE_MARKER,
        "canvas": dict(vars(cfg.canvas)),
        "trigger": dict(vars(cfg.trigger)),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "sha256": hashlib.sha256(blob).hexdigest()}


def _iter_local_rows(path: Path):
    table = pq.read_table(path)
    for row in table.to_pylist():
        yield row


def load_items(name: str, split: str, limit: int, offset: int = 0):
    local_path = LOCAL_DATASET_FILES.get((name, split))
    if local_path and local_path.exists():
        qf = DATASETS[name][2]
        af = DATASETS[name][3]
        for i, row in enumerate(_iter_local_rows(local_path)):
            if i < offset:
                continue
            if i >= offset + limit:
                break
            gold = row[af]
            if isinstance(gold, dict):
                gold = gold.get("text", gold.get("answer", ""))
            if isinstance(gold, list):
                gold = gold[0] if gold else ""
            yield row[qf], str(gold)
        return

    hf_id, hf_cfg, qf, af = DATASETS[name]
    if hf_id is None:
        raise SystemExit(f"{name}: wire up SARDI's released corpus/questions first (TODO)")
    from datasets import load_dataset
    ds = load_dataset(hf_id, hf_cfg, split=split) if hf_cfg else load_dataset(hf_id, split=split)
    for i, row in enumerate(ds):
        if i < offset:
            continue
        if i >= offset + limit:
            break
        gold = row[af]
        if isinstance(gold, dict):
            gold = gold.get("text", gold.get("answer", ""))
        if isinstance(gold, list):
            gold = gold[0] if gold else ""
        yield row[qf], str(gold)


def _load_calibrator(path: str | None, cfg: ErgoConfig) -> ConformalCalibrator | None:
    if not path:
        return None
    cal_path = Path(path)
    if not cal_path.exists():
        raise SystemExit(f"calibration file not found: {cal_path}")
    raw = json.loads(cal_path.read_text())
    scores = raw.get("null_scores", raw if isinstance(raw, list) else None)
    if not isinstance(scores, list):
        raise SystemExit("calibration file must be a JSON list or {\"null_scores\": [...]} object")
    expected = calibration_signature(cfg)
    actual = raw.get("compat") if isinstance(raw, dict) else None
    if actual != expected:
        raise SystemExit(
            "calibration artifact is incompatible with the current trigger/canvas regime; "
            f"expected compat {expected}, got {actual}"
        )
    calibrator = ConformalCalibrator()
    calibrator.add_null_scores(scores)
    return calibrator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--backbone", choices=["llada", "dream", "dream_native", "sardi_dream"], required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--device-map", default=None)
    ap.add_argument("--max-memory", default=None,
                    help='JSON object, e.g. {"0":"35GiB","1":"35GiB"}')
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--calibration", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = ErgoConfig.from_yaml(args.config) if args.config else ErgoConfig()
    cfg.seed = args.seed

    max_memory = json.loads(args.max_memory) if args.max_memory else None
    if isinstance(max_memory, dict):
        normalized = {}
        for key, value in max_memory.items():
            if isinstance(key, str) and key.isdigit():
                normalized[int(key)] = value
            else:
                normalized[key] = value
        max_memory = normalized
    common_kw = dict(
        device=args.device,
        device_map=args.device_map,
        max_memory=max_memory,
        adapter_path=args.adapter,
    )

    from ergo.backbones import load_dream, load_dream_native, load_llada, load_sardi_dream
    if args.backbone == "dream":
        backbone = load_dream(**common_kw)
    elif args.backbone == "dream_native":
        backbone = load_dream_native(**common_kw)
    elif args.backbone == "llada":
        backbone = load_llada(**common_kw)
    else:
        backbone = load_sardi_dream(**common_kw)
    from ergo.rag import SentenceTransformerEmbedder, VectorStore
    store = VectorStore(args.store, "documents", SentenceTransformerEmbedder())
    idf_path = Path(args.store, "idf.json")
    idf = json.loads(idf_path.read_text()) if idf_path.exists() else {}
    calibrator = _load_calibrator(args.calibration, cfg)
    orch = ErgoOrchestrator(backbone, backbone.tok, store, cfg, idf=idf, calibrator=calibrator)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ems, f1s = [], []
    with out.open("w") as fh:
        for q, gold in load_items(args.dataset, args.split, args.limit, offset=args.offset):
            t0 = time.time()
            res = orch.run(q, seed=args.seed)
            ctx = "\n".join(it.text for it in res.evidence.in_context())
            cycles = []
            for log in res.cycles:
                gap = log.gap
                rollback = log.rollback
                cycles.append({
                    "cycle": log.cycle,
                    "action": log.action,
                    "vetoed": log.vetoed,
                    "mean_answer_conf": round(float(log.mean_answer_conf), 6),
                    "mean_answer_rel": round(float(log.mean_answer_rel), 6),
                    "denoise_s": round(float(log.denoise_s), 6),
                    "decision_s": round(float(log.decision_s), 6),
                    "retrieval_s": round(float(log.retrieval_s), 6),
                    "rescore_s": round(float(log.rescore_s), 6),
                    "rollback_s": round(float(log.rollback_s), 6),
                    "closure_s": round(float(log.closure_s), 6),
                    "retrieved_ids": list(log.retrieved_ids),
                    "retrieval_scores": [round(float(s), 6) for s in log.retrieval_scores],
                    "gap_fire": bool(gap.fire) if gap else False,
                    "gap_rule": gap.rule if gap else "none",
                    "gap_focus_words": list(gap.focus_words) if gap else [],
                    "gap_p_value": (round(float(gap.p_value), 6)
                                    if gap and gap.p_value is not None else None),
                    "gap_top_score": (round(float(gap.top_span.score), 6)
                                      if gap and gap.top_span is not None else None),
                    "gap_all_scores": ([round(float(s), 6) for s in gap.all_scores]
                                       if gap else []),
                    "rollback_n": rollback.n_rolled if rollback else 0,
                    "rollback_n_candidates": rollback.n_candidates if rollback else 0,
                    "rollback_n_committed": rollback.n_committed if rollback else 0,
                    "rollback_n_lrt": rollback.n_lrt if rollback else 0,
                    "rollback_n_contradicted": rollback.n_contradicted if rollback else 0,
                    "rollback_capped": bool(rollback.capped) if rollback else False,
                    "anchor_remasked": int(log.anchor_remasked),
                    "anchor_candidates": int(log.anchor_candidates),
                })
            rec = {
                "question": q, "gold": gold, "pred": res.answer,
                "em": exact_match(res.answer, gold), "f1": f1(res.answer, gold),
                "copy_rate": copy_rate(res.answer, ctx),
                "n_retrievals": res.n_retrievals,
                "terminated_by": res.terminated_by,
                "rollbacks": [c["rollback_n"] for c in cycles if c["rollback_n"] > 0],
                "gap_fires": [c["gap_fire"] for c in cycles],
                "cycles": cycles,
                "wall_s": round(time.time() - t0, 2),
            }
            ems.append(rec["em"])
            f1s.append(rec["f1"])
            fh.write(json.dumps(rec) + "\n")
    n = len(ems)
    print(f"{args.dataset} n={n}  EM={sum(ems)/max(n,1):.3f}  F1={sum(f1s)/max(n,1):.3f}")


if __name__ == "__main__":
    main()
