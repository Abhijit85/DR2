"""Working + episodic memory (design doc §3.1 Memory Subsystem)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WorkingMemory:
    cycle_summaries: list[str] = field(default_factory=list)
    revisions: list[str] = field(default_factory=list)   # rollback log lines

    def add_cycle(self, thought: str, max_len: int = 140) -> None:
        one_line = " ".join(thought.split())[:max_len]
        if one_line:
            self.cycle_summaries.append(one_line)

    def add_revision(self, phrase: str) -> None:
        if phrase.strip():
            self.revisions.append(f"revised: {phrase.strip()[:80]}")

    def render(self) -> str:
        lines = [f"- cycle {i}: {s}" for i, s in enumerate(self.cycle_summaries)]
        lines += [f"- {r}" for r in self.revisions[-4:]]
        return "\n".join(lines) if lines else "(none)"


class EpisodicMemory:
    """Past thoughts/answers in a second vector collection (design doc §7.2).
    Same fused_search interface as the document store."""

    def __init__(self, store):
        self.store = store   # a VectorStore over the `episodic_memory` collection
        self._counter = 0

    def remember(self, kind: str, query: str, text: str) -> None:
        from .rag.chunker import Chunk
        self._counter += 1
        cid = f"epi::{kind}::{self._counter}"
        self.store.add([Chunk(id=cid, text=text, doc_id=f"episode:{query[:40]}",
                              chunk_idx=self._counter, char_start=0, char_end=len(text))])

    def fused_search(self, query, salience_text, cfg, exclude):
        return self.store.fused_search(query, salience_text, cfg, exclude)
