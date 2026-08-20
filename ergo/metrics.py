"""Metrics: EM/F1 (SQuAD-style), Copy Rate, RSD (design doc §2.1, §12)."""
from __future__ import annotations

import re
import string
from collections import Counter

import numpy as np


def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred: str, gold: str) -> float:
    return float(_normalize(pred) == _normalize(gold))


def f1(pred: str, gold: str) -> float:
    p, g = _normalize(pred).split(), _normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def copy_rate(answer: str, context: str) -> float:
    """Fraction of answer words appearing verbatim in the provided context
    (SPREAD's grounding metric)."""
    aw = _normalize(answer).split()
    if not aw:
        return 0.0
    cw = set(_normalize(context).split())
    return sum(w in cw for w in aw) / len(aw)


def rsd(answer: str, embed) -> float:
    """Response Semantic Drift: mean cosine distance between consecutive
    sentences (SPREAD).  ``embed``: list[str] -> np.ndarray (normalized)."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
    if len(sents) < 2:
        return 0.0
    E = embed(sents)
    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-9)
    dists = [1.0 - float(E[i] @ E[i + 1]) for i in range(len(E) - 1)]
    return float(np.mean(dists))
