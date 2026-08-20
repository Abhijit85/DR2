"""Salient-word extraction (design doc §5 step 4).

Word score: s(w) = conf(w) * Rel(w) * idf(w), pooled across trajectories.
A word's conf/Rel is the MINIMUM over its subword pieces (a word is only as
certain as its weakest piece); pooling sums scores over trajectories; any
single word's pooled score is capped at 2x the median to stop one hallucinated
rare token from steering retrieval alone.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .backbones.base import Snapshot
from .tokenization import TokenizerLike

STOPWORDS = set("""a an the of to in on for and or but is are was were be been being with as at by
from that this these those it its if then than so such not no nor do does did done can could will
would should may might must have has had i you he she we they what which who whom when where why
how all any both each few more most other some own same very s t just don now""".split())


def _merge_words(tok: TokenizerLike, ids: np.ndarray, positions: np.ndarray,
                 conf: np.ndarray, rel: np.ndarray) -> list[tuple[str, float, float]]:
    """Merge subword pieces into (word, min_conf, min_rel) triples."""
    words: list[tuple[str, float, float]] = []
    cur, c, r = [], 1.0, 1.0
    id_list = ids.tolist()
    for j, p in enumerate(positions):
        piece = tok.tokens([id_list[j]])[0]
        clean = piece.strip("Ġ▁").replace("##", "")
        if tok.is_word_start(id_list, j) and cur:
            words.append(("".join(cur), c, r))
            cur, c, r = [], 1.0, 1.0
        cur.append(clean)
        c, r = min(c, float(conf[p])), min(r, float(rel[p]))
    if cur:
        words.append(("".join(cur), c, r))
    return words


def salient_words(
    snapshots: list[Snapshot],
    tok: TokenizerLike,
    idf: dict[str, float],
    top_m: int,
    exclude: set[str] | None = None,
    fields: tuple[str, ...] = ("thought", "action_input", "answer"),
) -> list[str]:
    exclude = {w.lower() for w in (exclude or set())}
    default_idf = float(np.median(list(idf.values()))) if idf else 1.0
    pooled: dict[str, float] = defaultdict(float)
    for snap in snapshots:
        for f in fields:
            pos = snap.canvas.field_positions(f)
            ids = snap.predicted_ids[pos]
            for word, c, r in _merge_words(tok, ids, pos, snap.confidence, snap.relevance):
                w = word.lower().strip(".,;:!?\"'()[]")
                if (not w or not w[0].isalnum() or w in STOPWORDS or w in exclude
                        or w.startswith("<") or w.startswith("[")):
                    continue
                pooled[w] += c * r * idf.get(w, default_idf)
    if not pooled:
        return []
    scores = np.array(list(pooled.values()))
    cap = 2.0 * float(np.median(scores))
    ranked = sorted(pooled.items(), key=lambda kv: -min(kv[1], cap))
    return [w for w, _ in ranked[:top_m]]
