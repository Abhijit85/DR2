"""The ERGO loop (design doc §5; blueprint §0).

cycle:  partial denoise (SPREAD-guided, N parallel)
        -> hybrid decide (gap trigger fast path + ACTION vote slow path, veto)
        -> retrieve (fused query; critique-driven rewrite) | memory | finish
        -> evidential rollback pass (C1)
        -> gate evidence, re-mask low-relevance tokens, re-compose
closure: full denoise of remaining answer masks against all evidence.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .backbones.base import DiffusionBackbone, Snapshot
from .canvas import Action, Canvas
from .config import ErgoConfig
from .memory import WorkingMemory
from .rag.gating import EvidenceSet, parse_critique
from .rollback import RollbackReport, rollback_pass
from .salience import salient_words
from .tokenization import TokenizerLike
from .trigger import ConformalCalibrator, GapDecision, decide

SYSTEM_PROMPT = """You answer questions by iterative drafting. Your response canvas has six
fields. In [THOUGHT], reason about what the query needs and what evidence is
still missing. In [ACTION], write exactly one of: retrieve - you need
information from the document store; memory - you need something from earlier
in this session; finish - the evidence below is sufficient and [ANSWER] is
complete. In [ACTION_INPUT], name what you need to look up. In [CRITIQUE],
judge each numbered evidence item's usefulness for the query before relying
on it. In [ANSWER], draft the final answer. Ground every claim in the
EVIDENCE section; if evidence is insufficient, prefer retrieve over guessing."""


@dataclass
class CycleLog:
    cycle: int
    action: str
    gap: GapDecision | None = None
    vetoed: bool = False
    retrieved_ids: list[str] = field(default_factory=list)
    retrieval_scores: list[float] = field(default_factory=list)
    rollback: RollbackReport | None = None
    mean_answer_conf: float = 0.0
    mean_answer_rel: float = 0.0


@dataclass
class ErgoResult:
    answer: str
    cycles: list[CycleLog]
    evidence: EvidenceSet
    n_retrievals: int
    terminated_by: str


class ErgoOrchestrator:
    def __init__(
        self,
        backbone: DiffusionBackbone,
        tokenizer: TokenizerLike,
        doc_store,                       # VectorStore | BM25Store
        cfg: ErgoConfig,
        idf: dict[str, float] | None = None,
        episodic=None,
        calibrator: ConformalCalibrator | None = None,
    ):
        self.backbone = backbone
        self.tok = tokenizer
        self.docs = doc_store
        self.cfg = cfg
        self.idf = idf or {}
        self.episodic = episodic
        self.calibrator = calibrator

    # ------------------------------------------------------------ composition
    def compose_context(self, query: str, evidence: EvidenceSet,
                        memory: WorkingMemory, cycle: int) -> str:
        return (f"{SYSTEM_PROMPT}\n\n"
                f"USER QUERY:\n{query}\n\n"
                f"EVIDENCE (cycle {cycle}):\n"
                f"{evidence.render(self.cfg.retrieval.evidence_token_budget)}\n\n"
                f"PRIOR CYCLES:\n{memory.render()}\n\nRESPONSE:")

    def _query_span_text(self, query: str, evidence: EvidenceSet) -> str:
        """Query(+evidence header) text whose hidden states pool into h_q."""
        heads = [it.text.split(". ")[0] for it in evidence.in_context()[:3]]
        return query if not heads else query + " || " + " ".join(heads)

    # -------------------------------------------------------------- decisions
    def _vote(self, snapshots: list[Snapshot], cycle: int) -> Action:
        votes = []
        for s in snapshots:
            txt = s.canvas.field_text(self.tok, "action").strip().lower()
            for a in Action:
                if a.value in txt:
                    votes.append(a)
                    break
        if not votes:
            return Action.RETRIEVE if cycle < self.cfg.loop.c_max - 1 else Action.FINISH
        counts = Counter(votes)
        top = counts.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:   # tie
            return Action.RETRIEVE if cycle < self.cfg.loop.c_max - 1 else Action.FINISH
        return top[0][0]

    def _fuse(self, vote: Action, gap: GapDecision, mean_conf: float,
              cycle: int) -> tuple[Action, bool]:
        """Either path proposes retrieve; the ACTION vote can veto (§5 step 3)."""
        if vote == Action.FINISH and mean_conf >= self.cfg.loop.tau_done:
            return Action.FINISH, gap.fire            # confident finish wins; veto if gap fired
        if vote == Action.FINISH and gap.fire:
            return Action.RETRIEVE, False              # downgrade unconfident finish
        if gap.fire and vote != Action.MEMORY:
            return Action.RETRIEVE, False
        return vote, False

    # -------------------------------------------------------------------- run
    def run(self, query: str, seed: int | None = None) -> ErgoResult:
        cfg, tok = self.cfg, self.tok
        seed = self.cfg.seed if seed is None else seed
        evidence, memory = EvidenceSet(), WorkingMemory()
        canvas = Canvas.build(tok, cfg.canvas)
        canvas.write_observation(tok, "(none)")
        logs: list[CycleLog] = []
        low_rel_streak, terminated_by = 0, "c_max"
        best: Snapshot | None = None

        for c in range(cfg.loop.c_max):
            ctx = self.compose_context(query, evidence, memory, c)
            result = self.backbone.denoise(
                ctx, canvas, num_steps=cfg.steps_per_cycle,
                n_parallel=cfg.loop.n_parallel, seed=seed + c)
            snaps = result.trajectories
            best = max(snaps, key=lambda s: s.mean_answer_confidence())
            vote = self._vote(snaps, c)
            gap = decide(snaps, tok, cfg.trigger, self.calibrator)
            action, vetoed = self._fuse(vote, gap, best.mean_answer_confidence(), c)
            log = CycleLog(cycle=c, action=action.value, gap=gap, vetoed=vetoed,
                           mean_answer_conf=best.mean_answer_confidence(),
                           mean_answer_rel=best.mean_answer_relevance())
            logs.append(log)

            if action == Action.FINISH:
                terminated_by = "finish"
                canvas = best.canvas
                break

            # ---------------------------------------------------- retrieval
            store = self.docs if action == Action.RETRIEVE else (self.episodic or self.docs)
            prev_critique = parse_critique(best.canvas.field_text(tok, "critique"))
            mostly_irrelevant = (prev_critique and
                                 sum(v == "irrelevant" for v in prev_critique.values())
                                 > len(prev_critique) / 2)
            action_input = "" if mostly_irrelevant else best.canvas.field_text(tok, "action_input")
            words = salient_words(snaps, tok, self.idf, cfg.loop.salient_words,
                                  exclude=set(query.lower().split()))
            focus = " ".join(gap.focus_words + [action_input] + words).strip()
            hits = store.fused_search(query, focus, cfg.retrieval, exclude=evidence.seen_ids)
            for r in hits:
                evidence.add(r.chunk.id, r.chunk.text, source=r.chunk.doc_id)
            log.retrieved_ids = [r.chunk.id for r in hits]
            log.retrieval_scores = [r.score for r in hits]

            # exhaustion brakes (design doc §11)
            if not hits or max((r.score for r in hits), default=0.0) < cfg.retrieval.relevance_floor:
                low_rel_streak += 1
            else:
                low_rel_streak = 0
            if low_rel_streak >= 2:
                terminated_by = "exhausted"
                canvas = best.canvas
                break

            # -------------------------------------- C1: evidential rollback
            canvas = best.canvas
            contradict_ids = evidence.apply_verdicts(prev_critique, cfg.gate)
            new_ctx = self.compose_context(query, evidence, memory, c + 1)
            rescored = self.backbone.rescore(new_ctx, canvas)
            report = rollback_pass(canvas, rescored, cfg.rollback,
                                   tau_rel=cfg.loop.tau_rel, tokenizer=tok)
            log.rollback = report
            for ph in report.revised_phrases:
                memory.add_revision(ph)

            # -------------------------- SPREAD re-mask + re-compose (step 6)
            pos = canvas.field_positions("answer")
            committed = canvas.ids[pos] != canvas.mask_id
            low_rel = rescored.relevance[pos] < cfg.loop.tau_rel
            canvas.remask(pos[committed & low_rel])
            canvas.remask_fields()          # thought/action/action_input/critique
            digest = " | ".join(f"[{i}] {it.text[:80]}" for i, it in
                                enumerate(evidence.in_context()[:4], start=1)) or "(none)"
            canvas.write_observation(tok, digest)
            memory.add_cycle(best.canvas.field_text(tok, "thought"))

        # ------------------------------------------------------ closure pass
        if canvas.masked("answer").any():
            ctx = self.compose_context(query, evidence, memory, cfg.loop.c_max)
            result = self.backbone.denoise(ctx, canvas, num_steps=cfg.loop.total_steps,
                                           n_parallel=1, seed=seed + 999)
            canvas = result.trajectories[0].canvas
        answer = canvas.field_text(tok, "answer").replace("<pad>", "").strip()
        n_ret = sum(1 for l in logs if l.retrieved_ids)
        return ErgoResult(answer=answer, cycles=logs, evidence=evidence,
                          n_retrievals=n_ret, terminated_by=terminated_by)
