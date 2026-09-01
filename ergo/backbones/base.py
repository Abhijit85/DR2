"""Backbone protocol + SPREAD/CAP reveal policies (design doc §2.1, §3.1).

``spread_select`` preserves the original relevance-ranked reveal order. The
manual answer decoder now composes that order with a confidence gate so low-
confidence masked positions are not committed in bulk from independent
marginals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

from ..canvas import Canvas


@dataclass
class Snapshot:
    """Last-denoising-step state for one trajectory (design doc §3.1)."""
    canvas: Canvas
    predicted_ids: np.ndarray      # (L,) argmax token per position (incl. still-masked)
    confidence: np.ndarray         # (L,) prob of committed/argmax token
    relevance: np.ndarray          # (L,) Rel(i,q) in (0,1)
    entropy: np.ndarray            # (L,) normalized predictive entropy in [0,1]
    margin: np.ndarray             # (L,) top-2 margin uncertainty m_i in [0,1]

    def mean_answer_confidence(self) -> float:
        pos = self.canvas.field_positions("answer")
        return float(self.confidence[pos].mean())

    def mean_answer_relevance(self) -> float:
        pos = self.canvas.field_positions("answer")
        return float(self.relevance[pos].mean())


@dataclass
class DenoiseResult:
    trajectories: list[Snapshot] = field(default_factory=list)


class DiffusionBackbone(Protocol):
    """Contract every adapter fulfils (design doc §3.1)."""

    def denoise(
        self,
        context: str,
        canvas: Canvas,
        num_steps: int,
        n_parallel: int,
        stop_after_field: str | None = None,
        seed: int = 0,
    ) -> DenoiseResult: ...

    def rescore(self, context: str, canvas: Canvas) -> Snapshot:
        """One forward pass, no sampling — used by the rollback pass (C1)."""
        ...


# --------------------------------------------------------------------- SPREAD
def relevance_scores(hidden: np.ndarray, h_q: np.ndarray, *, standardize: bool = True) -> np.ndarray:
    """Rel(i,q) from cosine similarity, optionally standardized per canvas.

    ``standardize=True`` preserves ordering (monotone transform after z-scoring)
    while making thresholded uses such as ``tau_rel`` meaningful within each
    canvas. ``False`` keeps the original ``sigmoid(cos)`` scale for ablations.
    """
    hn = hidden / (np.linalg.norm(hidden, axis=-1, keepdims=True) + 1e-9)
    qn = h_q / (np.linalg.norm(h_q) + 1e-9)
    sim = hn @ qn
    if standardize:
        sim = (sim - sim.mean()) / (sim.std() + 1e-6)
    return 1.0 / (1.0 + np.exp(-sim))


def uncertainty_scores(probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalized entropy and top-2 margin uncertainty per position.

    probs: (L, V) row-stochastic. Returns (H_hat, m) each (L,) in [0, 1].
    """
    V = probs.shape[-1]
    ent = -np.sum(np.where(probs > 0, probs * np.log(probs + 1e-12), 0.0), axis=-1)
    h_hat = ent / max(np.log(V), 1e-9)
    part = np.partition(probs, -2, axis=-1)
    p1, p2 = part[..., -1], part[..., -2]
    m = 1.0 - (p1 - p2)
    return np.clip(h_hat, 0, 1), np.clip(m, 0, 1)


def spread_select(
    canvas: Canvas,
    probs: np.ndarray,          # (L, V) predictive distributions this step
    rel: np.ndarray,            # (L,) Rel(i, q)
    k: int,
    restrict: Sequence[str] | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Positions to commit this step: top-k *masked* positions by relevance.

    Ties are broken by sampled noise so parallel trajectories diverge
    (design doc §5 step 2). Returns an index array (possibly empty).
    """
    mask = canvas.ids == canvas.mask_id
    if restrict is not None:
        allowed = np.zeros_like(mask)
        for f in restrict:
            allowed[canvas.spans[f]] = True
        mask = mask & allowed
    idx = np.nonzero(mask)[0]
    if len(idx) == 0 or k <= 0:
        return idx[:0]
    noise = (rng.random(len(idx)) if rng is not None else np.zeros(len(idx))) * 1e-6
    order = np.argsort(-(rel[idx] + noise))
    return idx[order[: min(k, len(idx))]]


def cap_select(
    canvas: Canvas,
    probs: np.ndarray,          # (L, V) predictive distributions this step
    rel: np.ndarray,            # (L,) Rel(i, q)
    k: int,
    tau: float,
    restrict: Sequence[str] | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Confidence-gated SPREAD selection.

    Parallel commits are allowed only for masked positions whose argmax
    confidence clears ``tau``. Those eligible positions are still ranked by
    relevance, preserving SPREAD ordering. If nothing is above threshold, we
    commit exactly one best masked position to avoid bulk low-confidence
    collapse.
    """
    mask = canvas.ids == canvas.mask_id
    if restrict is not None:
        allowed = np.zeros_like(mask)
        for f in restrict:
            allowed[canvas.spans[f]] = True
        mask = mask & allowed
    idx = np.nonzero(mask)[0]
    if len(idx) == 0 or k <= 0:
        return idx[:0]
    conf = probs[idx].max(axis=-1)
    noise = (rng.random(len(idx)) if rng is not None else np.zeros(len(idx))) * 1e-6
    eligible = conf >= tau
    if np.any(eligible):
        gated = idx[eligible]
        gated_noise = noise[eligible]
        order = np.argsort(-(rel[gated] + gated_noise))
        return gated[order[: min(k, len(gated))]]
    score = rel[idx] + 1e-3 * conf + noise
    return idx[np.array([int(np.argmax(score))], dtype=np.int64)]


def linear_step_budget(num_masked: int, steps_remaining: int) -> int:
    """Per-step commit count k = ceil(|M_t| / steps_remaining) (design doc §10)."""
    if steps_remaining <= 0:
        return num_masked
    return int(np.ceil(num_masked / steps_remaining))
