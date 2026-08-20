"""The structured generation canvas (design doc §4).

Field labels are pre-committed tokens; contents start masked. ``[OBSERVATION]``
is orchestrator-committed each cycle; ``[ACTION]`` content is resolved against a
constrained candidate set (retrieve/memory/finish) by sequence log-probability.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from enum import Enum

import numpy as np

from .config import CanvasConfig
from .tokenization import TokenizerLike

FIELDS = ("thought", "action", "action_input", "observation", "critique", "answer")
LABELS = {
    "thought": "[THOUGHT]", "action": "[ACTION]", "action_input": "[ACTION_INPUT]",
    "observation": "[OBSERVATION]", "critique": "[CRITIQUE]", "answer": "[ANSWER]",
}


class Action(str, Enum):
    RETRIEVE = "retrieve"
    MEMORY = "memory"
    FINISH = "finish"


@dataclass
class Canvas:
    """Token-level canvas: ids + per-position metadata, all numpy arrays."""
    ids: np.ndarray                       # (L,) int
    spans: dict[str, slice]               # field -> content slice (labels excluded)
    mask_id: int
    committed_prob: np.ndarray | None = None   # (L,) prob at commit time (pi_i)
    action_candidates: dict[Action, list[int]] = dc_field(default_factory=dict)

    @classmethod
    def build(cls, tok: TokenizerLike, cfg: CanvasConfig) -> "Canvas":
        ids: list[int] = []
        spans: dict[str, slice] = {}
        lengths = {
            "thought": cfg.len_thought, "action": cfg.len_action,
            "action_input": cfg.len_action_input, "observation": cfg.len_observation,
            "critique": cfg.len_critique, "answer": cfg.len_answer,
        }
        for f in FIELDS:
            ids.extend(tok.encode(LABELS[f]))
            start = len(ids)
            ids.extend([tok.mask_id] * lengths[f])
            spans[f] = slice(start, len(ids))
        canvas = cls(
            ids=np.array(ids, dtype=np.int64), spans=spans, mask_id=tok.mask_id,
            committed_prob=np.zeros(len(ids), dtype=np.float64),
            action_candidates={a: tok.encode(a.value) for a in Action},
        )
        return canvas

    # ------------------------------------------------------------------ helpers
    def copy(self) -> "Canvas":
        return Canvas(self.ids.copy(), dict(self.spans), self.mask_id,
                      None if self.committed_prob is None else self.committed_prob.copy(),
                      dict(self.action_candidates))

    def masked(self, field: str | None = None) -> np.ndarray:
        """Boolean mask of still-masked positions (optionally within a field)."""
        m = self.ids == self.mask_id
        if field is not None:
            keep = np.zeros_like(m)
            keep[self.spans[field]] = True
            m = m & keep
        return m

    def field_positions(self, field: str) -> np.ndarray:
        return np.arange(self.spans[field].start, self.spans[field].stop)

    def fully_unmasked(self, field: str = "answer") -> bool:
        return not self.masked(field).any()

    def commit(self, positions: np.ndarray, token_ids: np.ndarray, probs: np.ndarray) -> None:
        self.ids[positions] = token_ids
        if self.committed_prob is not None:
            self.committed_prob[positions] = probs

    def remask(self, positions: np.ndarray) -> None:
        self.ids[positions] = self.mask_id
        if self.committed_prob is not None:
            self.committed_prob[positions] = 0.0

    def remask_fields(self, fields: tuple[str, ...] = ("thought", "action", "action_input", "critique")) -> None:
        for f in fields:
            self.remask(self.field_positions(f))

    # ------------------------------------------------------------- observation
    def write_observation(self, tok: TokenizerLike, digest: str) -> None:
        """Orchestrator-committed digest; truncated/padded to the field length."""
        span = self.spans["observation"]
        length = span.stop - span.start
        obs_ids = tok.encode(digest)[:length]
        obs_ids += [tok.pad_id] * (length - len(obs_ids))
        self.ids[span] = np.array(obs_ids, dtype=np.int64)
        if self.committed_prob is not None:
            self.committed_prob[span] = 1.0

    # ------------------------------------------------------------------ action
    def resolve_action(self, seq_logprob_fn) -> Action:
        """Pick the constrained action by candidate-sequence log-probability.

        ``seq_logprob_fn(positions, candidate_ids) -> float`` scores committing
        ``candidate_ids`` at the first ``len(candidate_ids)`` action positions.
        Robust to multi-token action words under BPE tokenizers.
        """
        pos = self.field_positions("action")
        best, best_lp = Action.RETRIEVE, -np.inf
        for act, cand in self.action_candidates.items():
            cand = cand[: len(pos)]
            lp = seq_logprob_fn(pos[: len(cand)], np.array(cand, dtype=np.int64))
            if lp > best_lp:
                best, best_lp = act, lp
        return best

    def commit_action(self, tok: TokenizerLike, action: Action) -> None:
        pos = self.field_positions("action")
        cand = self.action_candidates[action][: len(pos)]
        ids = np.full(len(pos), tok.pad_id, dtype=np.int64)
        ids[: len(cand)] = cand
        self.ids[pos] = ids
        if self.committed_prob is not None:
            self.committed_prob[pos] = 1.0

    # ------------------------------------------------------------------- text
    def field_text(self, tok: TokenizerLike, field: str) -> str:
        span_ids = self.ids[self.spans[field]]
        return tok.decode([i for i in span_ids.tolist() if i != self.mask_id])
