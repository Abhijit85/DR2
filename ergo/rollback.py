"""C1 — Evidential rollback (design doc §5 step 6b; blueprint §5.1, Props 1-2).

Every commit is a lease. After each evidence injection, one rescoring forward
pass re-tests every committed answer token with the likelihood-ratio criterion

    rollback(i)  <=>  log p'_i(x_i) < log pi_i - delta
                   or Rel'(i, q) < tau_rel
                   or the supporting chunk was judged `contradicts`

with delta = log(1/alpha_roll): under the calibration idealization the
false-rollback probability is bounded by e^{-delta} = alpha_roll (Prop. 1),
anytime-valid across repeated re-tests (Prop. 2).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .backbones.base import Snapshot
from .canvas import Canvas
from .config import RollbackConfig


@dataclass
class RollbackReport:
    positions: np.ndarray                 # positions cleared to [MASK]
    revised_phrases: list[str] = field(default_factory=list)
    n_committed: int = 0
    n_candidates: int = 0                 # tokens that failed the test (pre-cap)
    capped: bool = False

    @property
    def n_rolled(self) -> int:
        return len(self.positions)


def rollback_pass(
    canvas: Canvas,
    rescored: Snapshot,
    cfg: RollbackConfig,
    tau_rel: float,
    contradicted_positions: np.ndarray | None = None,
    tokenizer=None,
) -> RollbackReport:
    """Apply the LRT rollback rule to committed answer tokens in-place.

    ``rescored`` must come from ``backbone.rescore(new_context, canvas)`` —
    its ``confidence`` is p'_i(x_i) and ``relevance`` is Rel'(i, q).
    """
    pos = canvas.field_positions("answer")
    committed = canvas.ids[pos] != canvas.mask_id
    cand_pos = pos[committed]
    if len(cand_pos) == 0:
        return RollbackReport(positions=cand_pos, n_committed=0)

    pi = np.clip(canvas.committed_prob[cand_pos], 1e-9, 1.0)
    p_prime = np.clip(rescored.confidence[cand_pos], 1e-12, 1.0)
    if cfg.temperature != 1.0:  # optional temperature scaling of p_theta
        p_prime = p_prime ** (1.0 / cfg.temperature)
        pi = pi ** (1.0 / cfg.temperature)

    llr = np.log(p_prime) - np.log(pi)
    lrt_fails = llr < -cfg.delta
    # uncapped arms: SPREAD low-relevance re-mask (standard policy, not shock)
    # and critique-flagged contradiction
    other_fails = rescored.relevance[cand_pos] < tau_rel
    if contradicted_positions is not None and len(contradicted_positions):
        other_fails |= np.isin(cand_pos, contradicted_positions)

    # the cap is a shock absorber for the LRT arm only: evidence injection may
    # not wipe more than cap_fraction of the draft in one cycle (but always >=1)
    capped = False
    max_roll = max(1, int(np.floor(cfg.cap_fraction * len(cand_pos))))
    lrt_pos = cand_pos[lrt_fails]
    if len(lrt_pos) > max_roll:
        order = np.argsort(p_prime[lrt_fails])   # lowest-p' offenders first
        lrt_pos = lrt_pos[order[:max_roll]]
        capped = True
    failing = np.union1d(lrt_pos, cand_pos[other_fails])
    fails = lrt_fails | other_fails              # pre-cap candidate count

    phrases = []
    if tokenizer is not None and len(failing):
        phrases = [tokenizer.decode(canvas.ids[failing].tolist())]
    canvas.remask(failing)
    return RollbackReport(positions=failing, revised_phrases=phrases,
                          n_committed=int(committed.sum()),
                          n_candidates=int(fails.sum()), capped=capped)
