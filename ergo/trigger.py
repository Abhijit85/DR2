"""C2 — Knowledge-gap trigger (design doc §5.1; blueprint §5.2, Prop 3).

Gap score per span:  G(S) = Rel(S,q) * conf(F(S)) * u(D(S))
  frame F(S): positions with confidence >= span median (the stable scaffold)
  details D(S): the rest (the uncertain slots)
  u_i = 0.5 * (normalized entropy + top-2 margin uncertainty)

Firing rules:
  primary  — conformal: p-value against null-span calibration scores <= alpha_fire
             (distribution-free false-fire control; no lambda hyperparameter)
  fallback — outlier rule: max_S G(S) > mu_G + lambda_G * sigma_G
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .backbones.base import Snapshot
from .config import TriggerConfig

_BOUNDARY_TOKENS = {".", ",", ";", ":", "?", "!", "and", "but", "which", "that",
                    "because", "so", "then", "while", "although"}


@dataclass
class GapSpan:
    positions: np.ndarray
    score: float
    frame_words: list[str]


@dataclass
class GapDecision:
    fire: bool
    focus_words: list[str] = field(default_factory=list)
    top_span: GapSpan | None = None
    all_scores: list[float] = field(default_factory=list)
    rule: str = "none"            # "conformal" | "fallback" | "none"
    p_value: float | None = None


class ConformalCalibrator:
    """Split-conformal calibration over null-span gap scores (blueprint Prop 3)."""

    def __init__(self) -> None:
        self.null_scores: list[float] = []

    def add_null_scores(self, scores) -> None:
        self.null_scores.extend(float(s) for s in scores)

    @property
    def ready(self) -> bool:
        return len(self.null_scores) >= 20   # minimum for a meaningful quantile

    def p_value(self, score: float) -> float:
        """Conformal p-value: (1 + #{null >= score}) / (n + 1)."""
        arr = np.asarray(self.null_scores)
        return float((1 + int((arr >= score).sum())) / (len(arr) + 1))


def _segment_spans(snapshot: Snapshot, tokens: list[str], min_len: int) -> list[np.ndarray]:
    pos = snapshot.canvas.field_positions("answer")
    spans, cur = [], []
    for p in pos:
        tok = tokens[p].lower().strip("Ġ▁#")
        cur.append(p)
        if tok in _BOUNDARY_TOKENS or any(tok.endswith(b) for b in (".", ",", ";", "?", "!")):
            if len(cur) >= min_len:
                spans.append(np.array(cur))
            cur = []
    if len(cur) >= min_len:
        spans.append(np.array(cur))
    return spans


def gap_scores(snapshot: Snapshot, tok, cfg: TriggerConfig) -> list[GapSpan]:
    tokens = tok.tokens(snapshot.predicted_ids.tolist())
    u = 0.5 * (snapshot.entropy + snapshot.margin)
    out = []
    for span in _segment_spans(snapshot, tokens, cfg.min_span_tokens):
        conf = snapshot.confidence[span]
        med = np.median(conf)
        frame = span[conf >= med]
        details = span[conf < med]
        if len(details) == 0 or len(frame) == 0:
            continue
        g = float(snapshot.relevance[span].mean()
                  * snapshot.confidence[frame].mean()
                  * u[details].mean())
        frame_words = [tokens[p].strip("Ġ▁#") for p in frame
                       if tokens[p].strip("Ġ▁#").isalnum()]
        out.append(GapSpan(positions=span, score=g, frame_words=frame_words))
    return out


def decide(
    snapshots: list[Snapshot],
    tok,
    cfg: TriggerConfig,
    calibrator: ConformalCalibrator | None = None,
) -> GapDecision:
    """Pooled across the N parallel trajectories (design doc §5.1)."""
    spans: list[GapSpan] = []
    for s in snapshots:
        spans.extend(gap_scores(s, tok, cfg))
    if not spans:
        return GapDecision(fire=False)
    scores = [s.score for s in spans]
    top = max(spans, key=lambda s: s.score)

    use_conformal = cfg.use_conformal and calibrator is not None and calibrator.ready
    if use_conformal:
        p = calibrator.p_value(top.score)
        fire = p <= cfg.alpha_fire
        rule = "conformal"
    else:
        mu, sigma = float(np.mean(scores)), float(np.std(scores))
        fire = top.score > mu + cfg.lambda_g * sigma and sigma > 0
        rule, p = "fallback", None

    focus = list(dict.fromkeys(top.frame_words))[: cfg.max_focus_words]
    return GapDecision(fire=fire, focus_words=focus if fire else [],
                       top_span=top, all_scores=scores, rule=rule, p_value=p)
