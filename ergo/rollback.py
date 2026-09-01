"""C1 — Evidential rollback (design doc §5 step 6b; blueprint §5.1, Props 1-2).

Every commit is a lease. After each evidence injection, one rescoring forward
pass re-tests every committed answer token with the likelihood-ratio criterion

    rollback(i)  <=>  log p'_i(x_i) < log pi_i - delta
                   or the supporting chunk was judged `contradicts`

with delta = log(1/alpha_roll): under the calibration idealization the
false-rollback probability is bounded by e^{-delta} = alpha_roll (Prop. 1),
anytime-valid across repeated re-tests (Prop. 2). Relevance-based anchor
remasking is tracked separately in the orchestrator and is not part of C1.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .backbones.base import Snapshot
from .canvas import Canvas
from .config import RollbackConfig

@dataclass
class RollbackReport:
    positions: np.ndarray
    revised_phrases: list[str] = field(default_factory=list)
    n_committed: int = 0
    n_candidates: int = 0
    n_lrt: int = 0
    n_contradicted: int = 0
    capped: bool = False

    @property
    def n_rolled(self) -> int:
        return len(self.positions)


def rollback_pass(
    canvas: Canvas,
    before: Snapshot,
    after: Snapshot,
    cfg: RollbackConfig,
    contradicted_positions: np.ndarray | None = None,
    exempt_positions: np.ndarray | None = None,
    tokenizer=None,
) -> RollbackReport:
    pos = canvas.field_positions("answer")
    committed = canvas.ids[pos] != canvas.mask_id
    cand_pos = pos[committed]
    if len(cand_pos) == 0:
        return RollbackReport(positions=cand_pos, n_committed=0)

    exempt_mask = np.zeros(len(cand_pos), dtype=bool)
    if exempt_positions is not None and len(exempt_positions):
        exempt_mask = np.isin(cand_pos, exempt_positions)

    p_before = np.clip(before.confidence[cand_pos], 1e-12, 1.0)
    p_after = np.clip(after.confidence[cand_pos], 1e-12, 1.0)
    if cfg.temperature != 1.0:
        p_after = p_after ** (1.0 / cfg.temperature)
        p_before = p_before ** (1.0 / cfg.temperature)

    llr = np.log(p_after) - np.log(p_before)
    lrt_fails = llr < -cfg.delta

    contrad_fails = np.zeros(len(cand_pos), dtype=bool)
    if contradicted_positions is not None and len(contradicted_positions):
        contrad_fails = np.isin(cand_pos, contradicted_positions)

    capped = False
    max_lrt_roll = max(1, int(np.floor(cfg.cap_fraction * len(cand_pos))))
    lrt_pos = cand_pos[lrt_fails]
    if len(lrt_pos) > max_lrt_roll:
        order = np.argsort(p_after[lrt_fails])
        lrt_pos = lrt_pos[order[:max_lrt_roll]]
        capped = True

    contrad_pos = cand_pos[contrad_fails]
    failing = np.union1d(lrt_pos, contrad_pos)
    fails = lrt_fails | contrad_fails

    phrases = []
    if tokenizer is not None and len(failing):
        phrases = [tokenizer.decode(canvas.ids[failing].tolist())]
    canvas.remask(failing)
    return RollbackReport(
        positions=failing,
        revised_phrases=phrases,
        n_committed=int(committed.sum()),
        n_candidates=int(fails.sum()),
        n_lrt=int(lrt_fails.sum()),
        n_contradicted=int(contrad_fails.sum()),
        capped=capped,
    )
