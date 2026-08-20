"""Shared HuggingFace adapter for masked-diffusion LMs (LLaDA / Dream).

Implements the manual denoising loop with the SPREAD reveal policy
(design doc §9): at each step, one forward pass over [context ; canvas]
yields logits AND final-layer hidden states; h_q is mean-pooled over the
query(+evidence) span; Rel = sigmoid(cos(h_i, h_q)); the top-k masked
positions by Rel are committed. Control fields (thought/action/action_input/
critique) are decoded before the answer region (semi-AR field ordering).

We deliberately bypass model-specific generate()/diffusion_generate() so both
backbones run the *identical* selection rule — required for a clean paper
comparison. Torch is imported lazily; this module is unused in CPU tests.
"""
from __future__ import annotations

import numpy as np

from ..canvas import Action, Canvas
from ..tokenization import HFTokenizerWrapper
from .base import (DenoiseResult, Snapshot, linear_step_budget,
                   relevance_scores, spread_select, uncertainty_scores)

CONTROL_FIELDS = ("thought", "action", "action_input", "critique")
QUERY_MARKERS = ("USER QUERY:", "PRIOR CYCLES:")


class HFMaskedDiffusionBackbone:
    def __init__(
        self,
        model_name: str,
        mask_id: int | None = None,
        device: str = "cuda",
        dtype: str = "bfloat16",
        temperature: float = 0.0,
        apply_chat_template: bool = False,
        trust_remote_code: bool = True,
    ):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.torch = torch
        self.device = device
        self.hf_tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.tok = HFTokenizerWrapper(self.hf_tok, mask_id=mask_id)
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=trust_remote_code,
            torch_dtype=getattr(torch, dtype)).to(device).eval()
        self.temperature = temperature
        self.apply_chat_template = apply_chat_template

    # ---------------------------------------------------------------- forward
    def _forward(self, context_ids: list[int], canvas_ids: np.ndarray):
        """One pass -> (probs [L_canvas, V], hidden [L_canvas, d], h_q [d])."""
        torch = self.torch
        full = torch.tensor([context_ids + canvas_ids.tolist()], device=self.device)
        with torch.no_grad():
            out = self.model(full, output_hidden_states=True)
        n_ctx = len(context_ids)
        logits = out.logits[0, n_ctx:, :].float()
        if self.temperature and self.temperature > 0:
            logits = logits / self.temperature
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        hidden_all = out.hidden_states[-1][0].float().cpu().numpy()
        hidden = hidden_all[n_ctx:]
        h_q = hidden_all[self._query_span(context_ids)].mean(axis=0)
        return probs, hidden, h_q

    def _query_span(self, context_ids: list[int]) -> slice:
        """Token span between QUERY_MARKERS: query + evidence header -> h_q
        (design doc §5 step 2: evidence reshapes the relevance landscape)."""
        text = self.hf_tok.decode(context_ids)
        lo = text.find(QUERY_MARKERS[0])
        hi = text.find(QUERY_MARKERS[1])
        if lo == -1:
            return slice(0, len(context_ids))
        lo_ids = len(self.hf_tok.encode(text[:lo], add_special_tokens=False))
        hi_ids = (len(self.hf_tok.encode(text[:hi], add_special_tokens=False))
                  if hi != -1 else len(context_ids))
        return slice(lo_ids, max(hi_ids, lo_ids + 1))

    def _encode_context(self, context: str) -> list[int]:
        if self.apply_chat_template:
            return self.hf_tok.apply_chat_template(
                [{"role": "user", "content": context}],
                add_generation_prompt=True, tokenize=True)
        return self.hf_tok.encode(context, add_special_tokens=True)

    # ---------------------------------------------------------------- denoise
    def denoise(self, context, canvas, num_steps, n_parallel,
                stop_after_field=None, seed=0) -> DenoiseResult:
        ctx_ids = self._encode_context(context)
        result = DenoiseResult()
        for t in range(n_parallel):
            rng = np.random.default_rng(seed * 1000 + t)
            cv = canvas.copy()
            steps_left = num_steps
            # phase 1: control fields (semi-AR ordering; action constrained)
            steps_left = self._decode_fields(ctx_ids, cv, CONTROL_FIELDS, steps_left, rng,
                                             resolve_action=True)
            # phase 2: answer region, unless asked to stop at the action block
            if stop_after_field not in ("action", "action_input") and steps_left > 0:
                self._decode_fields(ctx_ids, cv, ("answer",), steps_left, rng)
            probs, hidden, h_q = self._forward(ctx_ids, cv.ids)
            result.trajectories.append(self._snapshot(cv, probs, hidden, h_q))
        return result

    def _decode_fields(self, ctx_ids, cv: Canvas, fields, steps: int, rng,
                       resolve_action: bool = False) -> int:
        total_masked = int(sum(cv.masked(f).sum() for f in fields))
        if total_masked == 0:
            return steps
        for step in range(steps):
            masked = int(sum(cv.masked(f).sum() for f in fields))
            if masked == 0:
                return steps - step
            probs, hidden, h_q = self._forward(ctx_ids, cv.ids)
            rel = relevance_scores(hidden, h_q)
            if resolve_action and cv.masked("action").any():
                self._commit_action(cv, probs)
            k = linear_step_budget(int(sum(cv.masked(f).sum() for f in fields)),
                                   steps - step)
            take = spread_select(cv, probs, rel, k, restrict=fields, rng=rng)
            if len(take) == 0:
                return steps - step
            ids = probs[take].argmax(axis=-1)
            cv.commit(take, ids.astype(np.int64), probs[take, ids])
        return 0

    def _commit_action(self, cv: Canvas, probs: np.ndarray) -> None:
        """Constrained decoding: score candidate sequences by log-prob (§4)."""
        def seq_logprob(positions, cand_ids):
            return float(np.sum(np.log(
                probs[positions, cand_ids] + 1e-12)))
        action = cv.resolve_action(seq_logprob)
        cv.commit_action(self.tok, action)

    # ---------------------------------------------------------------- rescore
    def rescore(self, context, canvas) -> Snapshot:
        ctx_ids = self._encode_context(context)
        probs, hidden, h_q = self._forward(ctx_ids, canvas.ids)
        return self._snapshot(canvas.copy(), probs, hidden, h_q, use_committed=True)

    def _snapshot(self, cv: Canvas, probs, hidden, h_q, use_committed=False) -> Snapshot:
        rel = relevance_scores(hidden, h_q)
        h_hat, margin = uncertainty_scores(probs)
        pred = probs.argmax(axis=-1).astype(np.int64)
        conf = probs[np.arange(len(pred)), pred]
        if use_committed:   # p'_i evaluated AT the committed token (for the LRT)
            committed = cv.ids != cv.mask_id
            conf = np.where(committed,
                            probs[np.arange(len(cv.ids)), np.clip(cv.ids, 0, probs.shape[1] - 1)],
                            conf)
            pred = np.where(committed, cv.ids, pred)
        return Snapshot(canvas=cv, predicted_ids=pred, confidence=conf,
                        relevance=rel, entropy=h_hat, margin=margin)


class LLaDABackbone(HFMaskedDiffusionBackbone):
    """LLaDA-8B-Instruct (GSAI-ML). Mask token id 126336 (their reference code)."""

    def __init__(self, model_name: str = "GSAI-ML/LLaDA-8B-Instruct", **kw):
        kw.setdefault("mask_id", 126336)
        kw.setdefault("apply_chat_template", True)
        super().__init__(model_name, **kw)


class DreamBackbone(HFMaskedDiffusionBackbone):
    """Dream-v0-Instruct-7B (Dream-org). Mask id from its tokenizer."""

    def __init__(self, model_name: str = "Dream-org/Dream-v0-Instruct-7B", **kw):
        kw.setdefault("apply_chat_template", True)
        super().__init__(model_name, **kw)
