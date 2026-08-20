"""Per-chunk evidence gating (design doc §7.5) — Tier 1 (eviction/demotion).

g_d = clip(0.5 * verdict_d + 0.5 * sigmoid(agree_d), 0, 1)

Tier-2 attention-bias gating lives in the adapters (optional hooks); this
module is model-free and always available.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from ..config import GateConfig

_VERDICT_RE = re.compile(r"\[(\d+)\]\s*(useful|partial|irrelevant|contradicts)", re.I)


def parse_critique(text: str) -> dict[int, str]:
    """'[1] useful ... [2] irrelevant ...' -> {1: 'useful', 2: 'irrelevant'}"""
    return {int(m.group(1)): m.group(2).lower() for m in _VERDICT_RE.finditer(text)}


@dataclass
class EvidenceItem:
    chunk_id: str
    text: str
    source: str = ""
    gate: float = 1.0
    demoted: bool = False
    evicted: bool = False
    verdict: str | None = None


@dataclass
class EvidenceSet:
    """Monotone chunk-id set with per-chunk gates. Ids are never forgotten
    (permanent dedup) even when a chunk is evicted from the context."""
    items: dict[str, EvidenceItem] = field(default_factory=dict)
    seen_ids: set[str] = field(default_factory=set)

    def add(self, chunk_id: str, text: str, source: str = "") -> None:
        self.seen_ids.add(chunk_id)
        if chunk_id not in self.items:
            self.items[chunk_id] = EvidenceItem(chunk_id, text, source)

    def in_context(self) -> list[EvidenceItem]:
        live = [it for it in self.items.values() if not it.evicted]
        live.sort(key=lambda it: (it.demoted, ))    # demoted last, else insertion order
        return live

    def apply_verdicts(self, verdicts: dict[int, str], cfg: GateConfig,
                       agreement: dict[str, float] | None = None) -> list[str]:
        """Update gates from critique verdicts (+ optional agreement scores).
        Returns chunk_ids whose verdict was `contradicts` (feeds rollback)."""
        contradicts: list[str] = []
        live = self.in_context()
        for idx, verdict in verdicts.items():
            if not (1 <= idx <= len(live)):
                continue
            item = live[idx - 1]
            item.verdict = verdict
            v = cfg.verdict_weights.get(verdict, 0.6)
            a = 0.5
            if agreement and item.chunk_id in agreement:
                a = 1.0 / (1.0 + np.exp(-agreement[item.chunk_id]))
            item.gate = float(np.clip(0.5 * v + 0.5 * a, 0.0, 1.0))
            if verdict == "contradicts":
                contradicts.append(item.chunk_id)
            if item.gate < cfg.evict_below:
                item.evicted = True
            elif item.gate < cfg.demote_below:
                item.demoted = True
        return contradicts

    def render(self, token_budget: int, chars_per_token: float = 4.0) -> str:
        """Evidence section, newest-first within tier, budget-trimmed;
        demoted chunks are truncated to their first sentence (design doc §7.5)."""
        budget = int(token_budget * chars_per_token)
        lines, used = [], 0
        for i, it in enumerate(self.in_context(), start=1):
            text = it.text.split(". ")[0] + "." if it.demoted else it.text
            entry = f"[{i}] ({it.source}) {text}" if it.source else f"[{i}] {text}"
            if used + len(entry) > budget and lines:
                break
            lines.append(entry)
            used += len(entry)
        return "\n".join(lines) if lines else "(none)"
