#!/usr/bin/env python3
from __future__ import annotations

import os
import argparse
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')
os.environ.setdefault('TORCH_COMPILE_DISABLE', '1')
import json
import math
import random
import types
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

DEFAULT_MODEL = 'pauljngr/sardi-dream-7b'

if not hasattr(torch.distributed, 'tensor'):
    class _DummyDTensor:
        pass
    torch.distributed.tensor = types.SimpleNamespace(DTensor=_DummyDTensor)


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def tokenize_rows(rows: list[dict], tokenizer, max_length: int) -> Dataset:
    data = []
    for row in rows:
        prompt_ids = tokenizer.encode(row['prompt'], add_special_tokens=False)
        response_ids = tokenizer.encode(row['response'], add_special_tokens=False)
        ids = (prompt_ids + response_ids)[:max_length]
        prompt_len = min(len(prompt_ids), len(ids))
        data.append({'input_ids': ids, 'prompt_len': prompt_len, 'task': row.get('task', 'unknown')})
    return Dataset.from_list(data)


@dataclass
class DiffusionSFTCollator:
    tokenizer: any
    mask_fraction: float
    min_masks: int = 1

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        pad_id = int(self.tokenizer.pad_token_id or 0)
        mask_id = int(getattr(self.tokenizer, 'mask_token_id', None))
        if mask_id is None:
            raise ValueError('Dream tokenizer must expose mask_token_id')
        max_len = max(len(f['input_ids']) for f in features)
        input_rows, label_rows, attn_rows = [], [], []
        for feat in features:
            ids = list(feat['input_ids'])
            prompt_len = int(feat['prompt_len'])
            labels = ids.copy()
            candidate_positions = list(range(prompt_len, len(ids)))
            if candidate_positions:
                n_mask = max(self.min_masks, math.ceil(len(candidate_positions) * self.mask_fraction))
                n_mask = min(n_mask, len(candidate_positions))
                masked = set(random.sample(candidate_positions, n_mask))
            else:
                masked = set()
            corrupted = [mask_id if i in masked else tok for i, tok in enumerate(ids)]
            labels = [-100 if i < prompt_len or i not in masked else tok for i, tok in enumerate(labels)]
            pad_n = max_len - len(ids)
            input_rows.append(corrupted + [pad_id] * pad_n)
            label_rows.append(labels + [-100] * pad_n)
            attn_rows.append([1] * len(ids) + [0] * pad_n)
        input_ids = torch.tensor(input_rows, dtype=torch.long)
        labels = torch.tensor(label_rows, dtype=torch.long)
        attn_2d = torch.tensor(attn_rows, dtype=torch.bool)
        attention_mask = torch.logical_and(attn_2d[:, None, None, :], attn_2d[:, None, :, None])
        return {'input_ids': input_ids, 'labels': labels, 'attention_mask': attention_mask}


def masked_ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    vocab = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab).float()
    flat_labels = labels.reshape(-1)
    keep = flat_labels != -100
    if not torch.any(keep):
        return flat_logits.new_zeros(())
    return F.cross_entropy(flat_logits[keep], flat_labels[keep])



class ManualAdamW:
    def __init__(self, params, lr: float, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.step_num = 0
        self.state = {}

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self):
        self.step_num += 1
        b1, b2 = self.betas
        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad.detach().float()
            state = self.state.setdefault(id(p), {
                'exp_avg': torch.zeros_like(p, dtype=torch.float32),
                'exp_avg_sq': torch.zeros_like(p, dtype=torch.float32),
            })
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            exp_avg.mul_(b1).add_(grad, alpha=1 - b1)
            exp_avg_sq.mul_(b2).addcmul_(grad, grad, value=1 - b2)
            bias_c1 = 1 - b1 ** self.step_num
            bias_c2 = 1 - b2 ** self.step_num
            denom = exp_avg_sq.sqrt().div_(math.sqrt(bias_c2)).add_(self.eps)
            step_size = self.lr / bias_c1
            update = exp_avg / denom
            if self.weight_decay:
                p.data.mul_(1 - self.lr * self.weight_decay)
            p.data.add_(update.to(dtype=p.dtype), alpha=-step_size)


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


def linear_warmup_decay(step: int, total_steps: int, warmup_ratio: float) -> float:
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    remain = max(total_steps - warmup_steps, 1)
    return max(0.0, (total_steps - step) / remain)


def sanitize_gradients(model) -> tuple[int, int]:
    bad_tensors = 0
    bad_values = 0
    for param in model.parameters():
        grad = param.grad
        if grad is None:
            continue
        bad = ~torch.isfinite(grad)
        if torch.any(bad):
            bad_tensors += 1
            bad_values += int(bad.sum().item())
            grad.data = torch.nan_to_num(grad.data, nan=0.0, posinf=0.0, neginf=0.0)
    return bad_tensors, bad_values


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-file', required=True)
    ap.add_argument('--model-name', default=DEFAULT_MODEL)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--max-length', type=int, default=2048)
    ap.add_argument('--mask-fraction', type=float, default=0.10)
    ap.add_argument('--per-device-batch-size', type=int, default=1)
    ap.add_argument('--gradient-accumulation-steps', type=int, default=16)
    ap.add_argument('--learning-rate', type=float, default=1e-5)
    ap.add_argument('--num-train-epochs', type=float, default=1.0)
    ap.add_argument('--max-steps', type=int, default=-1)
    ap.add_argument('--logging-steps', type=int, default=10)
    ap.add_argument('--save-steps', type=int, default=100)
    ap.add_argument('--warmup-ratio', type=float, default=0.03)
    ap.add_argument('--max-grad-norm', type=float, default=1.0)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load_jsonl(args.train_file)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    ds = tokenize_rows(rows, tokenizer, args.max_length)
    collator = DiffusionSFTCollator(tokenizer=tokenizer, mask_fraction=args.mask_fraction)
    loader = DataLoader(ds, batch_size=args.per_device_batch_size, shuffle=True, collate_fn=collator)

    model = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False

    peft_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias='none',
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    batches_per_epoch = max(1, math.ceil(len(ds) / args.per_device_batch_size))
    total_steps = args.max_steps if args.max_steps > 0 else math.ceil(batches_per_epoch * args.num_train_epochs / args.gradient_accumulation_steps)
    optimizer = ManualAdamW(model.parameters(), lr=args.learning_rate)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    data_iter = cycle(loader)
    optimizer.zero_grad()
    for step in range(total_steps):
        accum_loss = 0.0
        valid_microbatches = 0
        skipped_microbatches = 0
        for _ in range(args.gradient_accumulation_steps):
            batch = next(data_iter)
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop('labels')
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                out = model(**batch)
                loss_raw = masked_ce_loss(out.logits, labels)
            if not torch.isfinite(loss_raw):
                skipped_microbatches += 1
                optimizer.zero_grad()
                continue
            loss = loss_raw / args.gradient_accumulation_steps
            loss.backward()
            accum_loss += float(loss_raw.item())
            valid_microbatches += 1
        if valid_microbatches == 0:
            print(f'step={step + 1} loss=nan skipped={skipped_microbatches} reason=no_finite_microbatch', flush=True)
            optimizer.zero_grad()
            continue
        lr_scale = linear_warmup_decay(step, total_steps, args.warmup_ratio)
        optimizer.lr = args.learning_rate * lr_scale
        bad_grad_tensors, bad_grad_values = sanitize_gradients(model)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        if not torch.isfinite(grad_norm):
            print(
                f'step={step + 1} loss={accum_loss:.6f} lr={optimizer.lr:.6e} '
                f'grad_norm=nan skipped={skipped_microbatches} bad_grad_tensors={bad_grad_tensors} '
                f'bad_grad_values={bad_grad_values} reason=nonfinite_grad',
                flush=True,
            )
            optimizer.zero_grad()
            continue
        optimizer.step()
        optimizer.zero_grad()
        if (step + 1) % args.logging_steps == 0 or step == 0:
            print(
                f'step={step + 1} loss={accum_loss:.6f} lr={optimizer.lr:.6e} '
                f'grad_norm={float(grad_norm):.6f} valid_microbatches={valid_microbatches} '
                f'skipped_microbatches={skipped_microbatches} bad_grad_tensors={bad_grad_tensors} '
                f'bad_grad_values={bad_grad_values}',
                flush=True,
            )
        if (step + 1) % args.save_steps == 0:
            ckpt = outdir / f'checkpoint-{step + 1}'
            ckpt.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt)
            tokenizer.save_pretrained(ckpt)
    model.save_pretrained(outdir)
    tokenizer.save_pretrained(outdir)
    print(f'saved adapter to {outdir}', flush=True)


if __name__ == '__main__':
    main()
