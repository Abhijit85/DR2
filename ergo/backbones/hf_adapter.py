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

import os
import types
import numpy as np

try:
    import torch as _torch_compat
    if not hasattr(_torch_compat.distributed, 'tensor'):
        class _DummyDTensor:
            pass
        _torch_compat.distributed.tensor = types.SimpleNamespace(DTensor=_DummyDTensor)
except Exception:
    pass

from ..canvas import Action, Canvas
from ..tokenization import HFTokenizerWrapper
from .base import (DenoiseResult, Snapshot, cap_select, linear_step_budget,
                   relevance_scores, spread_select, uncertainty_scores)

CONTROL_FIELDS = ("thought", "action", "action_input", "critique")
QUERY_MARKERS = ("USER QUERY:", "PRIOR CYCLES:")


def _find_subsequence(haystack: list[int], needle: list[int]) -> int:
    if not needle or len(needle) > len(haystack):
        return -1
    last = len(haystack) - len(needle) + 1
    for i in range(last):
        if haystack[i:i + len(needle)] == needle:
            return i
    return -1


def _resolve_model_source(model_name: str, *, require_local: bool = False) -> str:
    """Prefer a locally cached HF snapshot when one is already available."""
    if os.path.isdir(model_name):
        return model_name

    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(model_name, local_files_only=True)
    except Exception:
        if require_local:
            raise FileNotFoundError(
                f"No local Hugging Face snapshot found for {model_name!r}. "
                "Pre-download it into the HF cache or pass a local path."
            ) from None
        return model_name


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
        logits_shift: bool = False,
        bidirectional_mask_4d: bool = False,
        parallel_commit_threshold: float = 0.9,
        adapter_path: str | None = None,
        device_map: str | None = None,
        max_memory: dict[str, str] | None = None,
    ):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.requested_device = device
        require_local = os.environ.get("ERGO_HF_LOCAL_ONLY", "").lower() in {"1", "true", "yes"}
        model_source = _resolve_model_source(model_name, require_local=require_local)

        self.model_name = model_name
        self.model_source = model_source
        self.hf_tok = AutoTokenizer.from_pretrained(
            model_source,
            trust_remote_code=trust_remote_code,
            local_files_only=require_local,
        )
        self.tok = HFTokenizerWrapper(self.hf_tok, mask_id=mask_id)
        load_kw = dict(
            trust_remote_code=trust_remote_code,
            local_files_only=require_local,
            torch_dtype=getattr(torch, dtype),
        )
        if device_map:
            load_kw["device_map"] = device_map
        if max_memory:
            load_kw["max_memory"] = max_memory
        self.model = AutoModel.from_pretrained(model_source, **load_kw)
        if adapter_path:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path, is_trainable=False)
        if device_map:
            self.device = self._infer_dispatch_device(self.model, fallback=device)
            self.model = self.model.eval()
        else:
            self.device = device
            self.model = self.model.to(device).eval()
        self.temperature = temperature
        self.apply_chat_template = apply_chat_template
        self.logits_shift = logits_shift
        self.bidirectional_mask_4d = bidirectional_mask_4d
        self.parallel_commit_threshold = parallel_commit_threshold
        banned = set(getattr(self.tok, "special_ids", set()))
        banned.update(getattr(self.tok, "continuation_ids", set()))
        self.content_banned_ids = sorted(i for i in banned if 0 <= i < self.tok.vocab_size)

    @staticmethod
    def _infer_dispatch_device(model, fallback: str) -> str:
        devmap = getattr(model, "hf_device_map", None) or {}
        for target in devmap.values():
            if isinstance(target, str) and target.startswith("cuda"):
                return target
        try:
            return str(next(model.parameters()).device)
        except Exception:
            return fallback

    # ---------------------------------------------------------------- forward
    def _forward(self, context_ids: list[int], canvas_ids: np.ndarray):
        probs, hidden, h_q = self._forward_batch(context_ids, canvas_ids[None, :])
        return probs[0], hidden[0], h_q[0]

    def _forward_batch(self, context_ids: list[int], canvas_ids: np.ndarray):
        """Batched pass -> probs [B,L,V], hidden [B,L,d], h_q [B,d]."""
        torch = self.torch
        if canvas_ids.ndim != 2:
            raise ValueError(f"canvas_ids must be rank-2, got {canvas_ids.shape}")
        batch = canvas_ids.shape[0]
        ctx = np.asarray(context_ids, dtype=np.int64)
        ctx_block = np.broadcast_to(ctx, (batch, len(ctx)))
        full_ids = np.concatenate([ctx_block, canvas_ids], axis=1)
        full = torch.tensor(full_ids, device=self.device)
        if self.bidirectional_mask_4d:
            attn_2d = torch.ones((batch, full.shape[1]), dtype=torch.bool, device=self.device)
            attn = torch.logical_and(attn_2d[:, None, None, :], attn_2d[:, None, :, None])
        else:
            attn = None
        with torch.no_grad():
            out = self.model(full, attention_mask=attn, output_hidden_states=True)
        logits_all = out.logits.float()
        hidden_all = out.hidden_states[-1].float()
        if self.logits_shift:
            logits_all = torch.cat([logits_all[:, :1, :], logits_all[:, :-1, :]], dim=1)
            hidden_all = torch.cat([hidden_all[:, :1, :], hidden_all[:, :-1, :]], dim=1)
        n_ctx = len(context_ids)
        logits = logits_all[:, n_ctx:, :]
        if self.temperature and self.temperature > 0:
            logits = logits / self.temperature
        if self.content_banned_ids:
            logits[..., self.content_banned_ids] = float("-inf")
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        hidden_all = hidden_all.cpu().numpy()
        hidden = hidden_all[:, n_ctx:, :]
        q_span = self._query_span(context_ids)
        h_q = hidden_all[:, q_span, :].mean(axis=1)
        return probs, hidden, h_q

    def _query_span(self, context_ids: list[int]) -> slice:
        """Token span between QUERY_MARKERS inside the templated ids."""
        lo_marker = self.hf_tok.encode(QUERY_MARKERS[0], add_special_tokens=False)
        hi_marker = self.hf_tok.encode(QUERY_MARKERS[1], add_special_tokens=False)
        lo = _find_subsequence(context_ids, lo_marker)
        hi = _find_subsequence(context_ids, hi_marker)
        if lo == -1:
            return slice(0, len(context_ids))
        start = lo + len(lo_marker)
        end = hi if hi != -1 and hi > start else len(context_ids)
        return slice(start, max(end, start + 1))

    def _encode_context(self, context: str) -> list[int]:
        if self.apply_chat_template:
            return self.hf_tok.apply_chat_template(
                [{"role": "user", "content": context}],
                add_generation_prompt=True, tokenize=True)
        return self.hf_tok.encode(context, add_special_tokens=True)

    # ---------------------------------------------------------------- denoise
    def denoise(self, context, canvas, num_steps, n_parallel,
                stop_after_field=None, seed=0, control_sweeps: int = 4) -> DenoiseResult:
        ctx_ids = self._encode_context(context)
        cvs = [canvas.copy() for _ in range(n_parallel)]
        rngs = [np.random.default_rng(seed * 1000 + t) for t in range(n_parallel)]
        steps_left = num_steps

        # phase 1: control fields use a fixed small sweep cap, not the full per-cycle budget
        control_steps = min(steps_left, max(0, control_sweeps))
        if control_steps > 0:
            used = self._decode_fields_batch(
                ctx_ids, cvs, CONTROL_FIELDS, control_steps, rngs, resolve_action=True)
            steps_left -= used

        # phase 2: answer region gets the remaining budget
        if stop_after_field not in ("action", "action_input") and steps_left > 0:
            self._decode_fields_batch(ctx_ids, cvs, ("answer",), steps_left, rngs)

        probs, hidden, h_q = self._forward_batch(
            ctx_ids, np.stack([cv.ids for cv in cvs], axis=0))
        result = DenoiseResult()
        for i, cv in enumerate(cvs):
            result.trajectories.append(self._snapshot(cv, probs[i], hidden[i], h_q[i]))
        return result

    def _decode_fields_batch(self, ctx_ids, cvs: list[Canvas], fields, steps: int, rngs,
                             resolve_action: bool = False) -> int:
        if steps <= 0:
            return 0
        steps_used = 0
        for step in range(steps):
            masked_counts = [int(sum(cv.masked(f).sum() for f in fields)) for cv in cvs]
            if max(masked_counts, default=0) == 0:
                break
            probs, hidden, h_q = self._forward_batch(
                ctx_ids, np.stack([cv.ids for cv in cvs], axis=0))
            rel = np.stack([relevance_scores(hidden[i], h_q[i]) for i in range(len(cvs))], axis=0)
            steps_used += 1
            any_take = False
            for i, cv in enumerate(cvs):
                masked = masked_counts[i]
                if masked == 0:
                    continue
                if resolve_action and cv.masked("action").any():
                    self._commit_action(cv, probs[i])
                k = linear_step_budget(masked, steps - step)
                if tuple(fields) == ("answer",):
                    take = cap_select(
                        cv, probs[i], rel[i], k,
                        tau=self.parallel_commit_threshold,
                        restrict=fields, rng=rngs[i],
                    )
                else:
                    take = spread_select(cv, probs[i], rel[i], k, restrict=fields, rng=rngs[i])
                if len(take) == 0:
                    continue
                ids = probs[i, take].argmax(axis=-1)
                cv.commit(take, ids.astype(np.int64), probs[i, take, ids])
                any_take = True
            if not any_take:
                break
        return steps_used

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
        kw.setdefault("logits_shift", False)
        super().__init__(model_name, **kw)


class DreamBackbone(HFMaskedDiffusionBackbone):
    """Dream-v0-Instruct-7B (Dream-org). AR head predicts token i from position i-1."""

    def __init__(self, model_name: str = "Dream-org/Dream-v0-Instruct-7B", **kw):
        kw.setdefault("apply_chat_template", True)
        kw.setdefault("logits_shift", True)
        kw.setdefault("bidirectional_mask_4d", True)
        super().__init__(model_name, **kw)
