"""Deterministic mock dLLM for CPU development and unit tests.

Simulates a masked-diffusion backbone with a scripted "knowledge model":
facts it only *knows* when supporting evidence text is present in the context.
Faithful to the real adapters' interface — Snapshot arrays, SPREAD ordering,
partial denoising, rescoring for rollback — so the orchestrator logic exercised
in tests is byte-identical to what runs on GPU.
"""
from __future__ import annotations

import numpy as np

from ..canvas import Action, Canvas
from ..tokenization import SimpleTokenizer
from .base import DenoiseResult, Snapshot, linear_step_budget, spread_select

_FILLER = "unknown"


class MockBackbone:
    def __init__(
        self,
        tok: SimpleTokenizer,
        facts: dict[str, str],
        *,
        known_conf: float = 0.95,
        unknown_conf: float = 0.30,
        answer_steps: int = 32,   # steps to fully demask the answer region
    ) -> None:
        self.tok = tok
        self.facts = facts        # {evidence-substring: answer text}
        self.known_conf = known_conf
        self.unknown_conf = unknown_conf
        self.answer_steps = answer_steps

    # ------------------------------------------------------------------ world
    def _lookup(self, context: str) -> str | None:
        for key, answer in self.facts.items():
            if key.lower() in context.lower():
                return answer
        return None

    def _query_words(self, context: str) -> list[str]:
        marker = "USER QUERY:"
        if marker in context:
            tail = context.split(marker, 1)[1].lstrip()
            first_line = tail.splitlines()[0] if tail.splitlines() else ""
        else:
            first_line = context.splitlines()[0] if context.splitlines() else context
        return first_line.split()[:12]

    # ------------------------------------------------------------- target ids
    def _target_for(self, canvas: Canvas, context: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(target_ids, conf, rel) for every canvas position."""
        L = len(canvas.ids)
        target = canvas.ids.copy()
        conf = np.full(L, 0.9)
        rel = np.full(L, 0.4)
        answer = self._lookup(context)
        known = answer is not None
        qwords = set(w.lower() for w in self._query_words(context))

        def fill(fieldname: str, text: str, c: float, r: float) -> None:
            pos = canvas.field_positions(fieldname)
            ids = self.tok.encode(text)[: len(pos)]
            ids += [self.tok.pad_id] * (len(pos) - len(ids))
            target[pos] = ids
            conf[pos] = c
            rel[pos] = r

        thought = ("the evidence provided is sufficient to answer"
                   if known else "need to look up missing details about " +
                   " ".join(sorted(qwords)[:3]))
        fill("thought", thought, 0.85, 0.7)
        fill("action", (Action.FINISH if known else Action.RETRIEVE).value, 0.99, 0.9)
        fill("action_input", " ".join(sorted(qwords)[:4]) or "details", 0.8, 0.8)
        if known:
            verdicts = "[1] useful grounds answer stated directly"
            fill("critique", verdicts, 0.8, 0.7)
            fill("answer", answer, self.known_conf, 0.92)
        else:
            fill("critique", "", 0.5, 0.4)
            fill("answer", " ".join([_FILLER] * 6), self.unknown_conf, 0.45)
        return target, conf, rel

    def _snapshot(self, canvas: Canvas, target, conf, rel) -> Snapshot:
        h_hat = np.clip(1.0 - conf, 0.02, 0.98)
        margin = np.clip(1.0 - conf * 0.9, 0.02, 1.0)
        return Snapshot(canvas=canvas, predicted_ids=target,
                        confidence=conf.copy(), relevance=rel.copy(),
                        entropy=h_hat, margin=margin)

    # -------------------------------------------------------------- interface
    def denoise(self, context, canvas, num_steps, n_parallel, stop_after_field=None, seed=0):
        result = DenoiseResult()
        for t in range(n_parallel):
            rng = np.random.default_rng(seed * 1000 + t)
            cv = canvas.copy()
            target, conf, rel = self._target_for(cv, context)
            # control blocks resolve first (semi-AR ordering)
            for f in ("thought", "action", "action_input", "critique"):
                pos = cv.field_positions(f)
                m = cv.ids[pos] == cv.mask_id
                cv.commit(pos[m], target[pos][m], conf[pos][m])
            if stop_after_field not in ("action", "action_input"):
                # answer region: SPREAD-ordered partial commit over the budget
                for _ in range(num_steps):
                    masked = int(cv.masked("answer").sum())
                    if masked == 0:
                        break
                    k = linear_step_budget(masked, self.answer_steps)
                    probs = self._prob_matrix(cv, target, conf)
                    take = spread_select(cv, probs, rel, k, restrict=("answer",), rng=rng)
                    cv.commit(take, target[take], conf[take])
            result.trajectories.append(self._snapshot(cv, target, conf, rel))
        return result

    def rescore(self, context, canvas):
        """One 'forward pass' under the (possibly updated) context — for C1."""
        cv = canvas.copy()
        target, conf, rel = self._target_for(cv, context)
        # committed tokens that CONTRADICT the evidence-informed target get
        # crushed probability; agreeing ones get the evidence-informed conf.
        pos = cv.field_positions("answer")
        committed = cv.ids[pos] != cv.mask_id
        agree = cv.ids[pos] == target[pos]
        conf_out, rel_out = conf.copy(), rel.copy()
        conf_out[pos[committed & agree]] = conf[pos][committed & agree]
        conf_out[pos[committed & ~agree]] = 0.02
        rel_out[pos[committed & ~agree]] = 0.2
        snap = self._snapshot(cv, target, conf_out, rel_out)
        return snap

    def _prob_matrix(self, canvas: Canvas, target, conf) -> np.ndarray:
        V = max(self.tok.vocab_size, 8)
        L = len(canvas.ids)
        probs = np.full((L, V), 0.0)
        rest = (1.0 - conf)[:, None] / (V - 1)
        probs += rest
        probs[np.arange(L), np.clip(target, 0, V - 1)] = conf
        return probs / probs.sum(axis=1, keepdims=True)
