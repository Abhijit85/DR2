"""ERGO configuration.

All defaults mirror the design document's hyperparameter table (design doc §10).
The two contribution knobs are *error rates*, not tuned thresholds:
  - alpha_roll  -> delta = log(1/alpha_roll)   (C1, false-rollback bound)
  - alpha_fire  -> conformal trigger level      (C2, false-fire control)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class CanvasConfig:
    len_thought: int = 96
    len_action: int = 6
    len_action_input: int = 24
    len_observation: int = 64
    len_critique: int = 32
    len_answer: int = 256


@dataclass
class TriggerConfig:
    """C2 — knowledge-gap trigger."""
    alpha_fire: float = 0.05        # conformal false-fire rate (primary rule)
    lambda_g: float = 1.5           # mu + lambda*sigma fallback rule
    max_focus_words: int = 6
    min_span_tokens: int = 3
    use_conformal: bool = True      # falls back automatically if no calibration set


@dataclass
class RollbackConfig:
    """C1 — evidential rollback."""
    alpha_roll: float = 0.135       # target false-rollback rate; delta = ln(1/alpha)
    cap_fraction: float = 0.30      # max fraction of anchors cleared per cycle
    temperature: float = 1.0        # optional temperature scaling of p_theta

    @property
    def delta(self) -> float:
        return math.log(1.0 / self.alpha_roll)


@dataclass
class RetrievalConfig:
    alpha: float = 0.4              # query-embedding weight
    beta: float = 0.6               # salience-embedding weight
    k: int = 3                      # chunks injected per retrieval
    candidates_multiplier: int = 4  # fetch 4k candidates before fusion
    mmr_lambda: float = 0.7
    evidence_token_budget: int = 2400
    relevance_floor: float = 0.35   # 2 consecutive below-floor cycles -> force finish
    chunk_chars: int = 2000
    chunk_overlap_chars: int = 200  # 0 for strict spec compliance
    bm25: bool = False              # use BM25 instead of dense (SARDI parity runs)


@dataclass
class GateConfig:
    evict_below: float = 0.3
    demote_below: float = 0.6
    verdict_weights: dict = field(default_factory=lambda: {
        "useful": 1.0, "partial": 0.6, "irrelevant": 0.2, "contradicts": 0.2,
    })


@dataclass
class LoopConfig:
    n_parallel: int = 4
    beam_keep: int = 1
    c_max: int = 4
    total_steps: int = 128
    tau_rel: float = 0.65           # SPREAD anchor threshold (harness, ablated)
    tau_done: float = 0.85
    salient_words: int = 8
    rsd_tighten: float = 0.05


@dataclass
class ErgoConfig:
    canvas: CanvasConfig = field(default_factory=CanvasConfig)
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    rollback: RollbackConfig = field(default_factory=RollbackConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    gate: GateConfig = field(default_factory=GateConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    seed: int = 0

    @property
    def steps_per_cycle(self) -> int:
        return max(1, self.loop.total_steps // (self.loop.c_max + 1))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ErgoConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        cfg = cls()
        for section, values in raw.items():
            if not hasattr(cfg, section):
                raise KeyError(f"unknown config section: {section}")
            obj = getattr(cfg, section)
            if isinstance(values, dict):
                for k, v in values.items():
                    if not hasattr(obj, k):
                        raise KeyError(f"unknown key {section}.{k}")
                    setattr(obj, k, v)
            else:
                setattr(cfg, section, values)
        return cfg

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(asdict(self), sort_keys=False))
