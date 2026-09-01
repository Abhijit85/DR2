#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from ergo.backbones.hf_adapter import DreamBackbone


def shift_sequence(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x[:, :1, ...], x[:, :-1, ...]], dim=1)


def build_prompt_ids(backbone: DreamBackbone, query: str, masks: int) -> tuple[list[int], np.ndarray]:
    ctx = backbone._encode_context(query)
    suffix = np.full((masks,), backbone.tok.mask_id, dtype=np.int64)
    return ctx, suffix


def make_mask(kind: str, batch: int, full_len: int, device: str):
    if kind == 'none':
        return None
    if kind == '2d':
        return torch.ones((batch, full_len), dtype=torch.bool, device=device)
    if kind == '4d-bidir':
        mask2d = torch.ones((batch, full_len), dtype=torch.bool, device=device)
        return torch.logical_and(mask2d[:, None, None, :], mask2d[:, None, :, None])
    raise ValueError(kind)


def run_variant(backbone: DreamBackbone, ctx_ids: list[int], canvas_ids: np.ndarray, mask_kind: str, shift: bool):
    full_ids = np.concatenate([np.asarray(ctx_ids, dtype=np.int64), canvas_ids], axis=0)[None, :]
    full = torch.tensor(full_ids, device=backbone.device)
    attn = make_mask(mask_kind, 1, full.shape[1], backbone.device)
    with torch.no_grad():
        out = backbone.model(full, attention_mask=attn, output_hidden_states=True)
    logits = out.logits.float()
    if shift:
        logits = shift_sequence(logits)
    n_ctx = len(ctx_ids)
    logits = logits[:, n_ctx:, :]
    if backbone.content_banned_ids:
        logits[..., backbone.content_banned_ids] = float('-inf')
    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    pred = probs.argmax(axis=-1)
    toks = backbone.tok.tokens(pred.tolist())
    unique = len(set(pred.tolist()))
    adj = []
    for i in range(len(probs) - 1):
        adj.append(float(np.linalg.norm(probs[i] - probs[i + 1], ord=1)))
    sample_pos = [0, min(8, len(pred) - 1), min(16, len(pred) - 1), len(pred) - 1]
    top = {}
    for pos in sample_pos:
        idx = np.argsort(-probs[pos])[:3]
        top[pos] = [(backbone.tok.tokens([int(i)])[0], float(probs[pos, i])) for i in idx]
    return {
        'mask_kind': mask_kind,
        'shift': shift,
        'unique_argmax': unique,
        'span_len': int(len(pred)),
        'first_12_tokens': toks[:12],
        'mean_adj_l1': float(np.mean(adj)) if adj else 0.0,
        'top3': top,
        'mode_token': Counter(pred.tolist()).most_common(1)[0][0],
        'mode_token_str': backbone.tok.tokens([int(Counter(pred.tolist()).most_common(1)[0][0])])[0],
    }


def maybe_native(backbone: DreamBackbone, ctx_ids: list[int], masks: int):
    if not hasattr(backbone.model, 'diffusion_generate'):
        return 'native diffusion_generate unavailable'
    full = torch.tensor([ctx_ids], device=backbone.device)
    try:
        with torch.no_grad():
            out = backbone.model.diffusion_generate(
                input_ids=full,
                seq_len=full.shape[1] + masks,
            )
        if isinstance(out, torch.Tensor):
            ids = out[0].tolist()
        elif hasattr(out, 'sequences'):
            ids = out.sequences[0].tolist()
        else:
            return f'native output type {type(out).__name__}'
        text = backbone.tok.decode(ids)
        return text[:400]
    except Exception as e:
        return f'native diffusion_generate error: {type(e).__name__}: {e}'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--query', default='Were Scott Derrickson and Ed Wood of the same nationality?')
    ap.add_argument('--masks', type=int, default=32)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    backbone = DreamBackbone(device=args.device)
    ctx_ids, canvas_ids = build_prompt_ids(backbone, args.query, args.masks)

    print('query:', args.query)
    print('context_tokens:', len(ctx_ids), 'mask_suffix:', len(canvas_ids))
    for mask_kind in ('none', '2d', '4d-bidir'):
        for shift in (False, True):
            res = run_variant(backbone, ctx_ids, canvas_ids, mask_kind, shift)
            print(f"\nvariant mask={mask_kind} shift={shift}")
            print(' unique_argmax:', res['unique_argmax'], '/', res['span_len'])
            print(' mode_token:', res['mode_token_str'])
            print(' mean_adj_l1:', round(res['mean_adj_l1'], 6))
            print(' first_12_tokens:', res['first_12_tokens'])
            for pos, vals in res['top3'].items():
                pretty = ', '.join(f"{tok}:{prob:.4f}" for tok, prob in vals)
                print(f' top3@{pos}: {pretty}')

    print('\nnative diffusion_generate:')
    print(maybe_native(backbone, ctx_ids, args.masks))


if __name__ == '__main__':
    main()
