"""Native Dream backend using Dream's own ``diffusion_generate`` sampler.

This backend avoids free-running text continuation parsing. It appends ERGO's
structured canvas directly to the prompt, lets Dream fill only the masked canvas
positions, and then slices the generated ids back into the canvas by position.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from ..canvas import Canvas
from .base import DenoiseResult
from .hf_adapter import DreamBackbone, _resolve_model_source


def _ensure_generation_config(model_source: str) -> str:
    """Dream remote code requires generation_config.json, but the cache may not ship it."""
    source = Path(model_source)
    gen_cfg = source / "generation_config.json"
    if gen_cfg.exists():
        return str(source)
    config = json.loads((source / "config.json").read_text())
    shim_dir = Path("artifacts/dream_native_shim")
    shim_dir.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = shim_dir / item.name
        if target.exists() or target.is_symlink():
            continue
        os.symlink(item, target)
    payload = {
        "temperature": 0.0,
        "top_p": None,
        "top_k": None,
        "max_length": 20,
        "max_new_tokens": None,
        "eps": 1e-3,
        "steps": 512,
        "alg": "origin",
        "alg_temp": None,
        "num_return_sequences": 1,
        "return_dict_in_generate": False,
        "output_history": False,
        "mask_token_id": config["mask_token_id"],
        "pad_token_id": config["pad_token_id"],
        "bos_token_id": config["bos_token_id"],
        "eos_token_id": config["eos_token_id"],
        "transformers_version": config.get("transformers_version", "4.46.2"),
    }
    gen_cfg = shim_dir / "generation_config.json"
    gen_cfg.write_text(json.dumps(payload, indent=2) + "\n")
    return str(shim_dir)


class NativeDreamBackbone(DreamBackbone):
    """Dream backend that uses native diffusion_generate over the positional canvas."""

    def __init__(
        self,
        model_name: str = "Dream-org/Dream-v0-Instruct-7B",
        native_alg: str = "origin",
        native_threshold: float | None = None,
        **kw,
    ):
        resolved = _resolve_model_source(model_name, require_local=False)
        shimmed = _ensure_generation_config(resolved)
        kw.setdefault("apply_chat_template", True)
        kw.setdefault("logits_shift", True)
        kw.setdefault("bidirectional_mask_4d", True)
        super().__init__(model_name=shimmed, **kw)
        self.native_alg = native_alg
        self.native_threshold = native_threshold

    def _build_inputs(self, context: str, canvas: Canvas, batch: int):
        torch = self.torch
        ctx_ids = self._encode_context(context)
        body = np.concatenate([np.asarray(ctx_ids, dtype=np.int64), canvas.ids, np.array([self.tok.pad_id], dtype=np.int64)])
        full = np.broadcast_to(body, (batch, len(body))).copy()
        attn = np.ones_like(full, dtype=np.int64)
        attn[:, -1] = 0  # one ignored dummy slot keeps Dream's max_length > input length constraint satisfied
        return ctx_ids, torch.tensor(full, device=self.device), torch.tensor(attn, device=self.device)

    def _absorb(self, template: Canvas, generated_ids: np.ndarray) -> Canvas:
        cv = template.copy()
        cv.ids = generated_ids.astype(np.int64, copy=True)

        # Native Dream often uses eos/pad-like tokens as variable-length filler.
        # In ERGO's fixed answer canvas these should remain maskable content slots,
        # not count as a committed answer, otherwise closure stops early and the
        # rescoring path assigns them exact zero confidence after special-id bans.
        non_content_ids = getattr(self.tok, "non_content_ids", set())
        if non_content_ids:
            answer_pos = cv.field_positions("answer")
            bad = np.isin(cv.ids[answer_pos], list(non_content_ids))
            if np.any(bad):
                cv.remask(answer_pos[bad])

        if cv.committed_prob is not None:
            cv.committed_prob[:] = np.where(cv.ids == cv.mask_id, 0.0, 1.0)
        return cv

    def denoise(self, context, canvas, num_steps, n_parallel, stop_after_field=None, seed=0, control_sweeps: int = 4):
        del stop_after_field, control_sweeps, seed
        ctx_ids, inputs, attention_mask = self._build_inputs(context, canvas, max(1, n_parallel))
        output = self.model.diffusion_generate(
            inputs=inputs,
            attention_mask=attention_mask,
            max_length=inputs.shape[1] + 1,
            steps=max(1, num_steps),
            temperature=0.0,
            alg=self.native_alg,
            num_return_sequences=1,
            **({"threshold": self.native_threshold} if self.native_threshold is not None else {}),
        )
        seqs = output.sequences if hasattr(output, "sequences") else output
        start = len(ctx_ids)
        stop = start + len(canvas.ids)
        result = DenoiseResult()
        for seq in seqs:
            cv = self._absorb(canvas, seq[start:stop].detach().cpu().numpy())
            snap = self.rescore(context, cv)
            result.trajectories.append(snap)
        return result



class SARDIDreamBackbone(NativeDreamBackbone):
    """SARDI-patched Dream checkpoint using its native confidence-threshold sampler."""

    def __init__(self, model_name: str = "pauljngr/sardi-dream-7b", native_threshold: float = 0.95, **kw):
        kw.setdefault("apply_chat_template", False)
        super().__init__(
            model_name=model_name,
            native_alg="confidence_threshold",
            native_threshold=native_threshold,
            **kw,
        )
