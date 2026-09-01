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
import time
from dataclasses import dataclass, field

import numpy as np

from .backbones.base import DiffusionBackbone, Snapshot
from .canvas import Action, Canvas
from .config import ErgoConfig
from .memory import WorkingMemory
from .postprocess import clean_answer
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
    anchor_remasked: int = 0
    anchor_candidates: int = 0
    mean_answer_conf: float = 0.0
    mean_answer_rel: float = 0.0
    denoise_s: float = 0.0
    decision_s: float = 0.0
    retrieval_s: float = 0.0
    rescore_s: float = 0.0
    rollback_s: float = 0.0
    closure_s: float = 0.0


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

    def _non_content_positions(self, canvas: Canvas) -> np.ndarray:
        pos = canvas.field_positions("answer")
        ids = canvas.ids[pos]
        non_content_ids = getattr(self.tok, "non_content_ids", set())
        if not non_content_ids:
            return pos[:0]
        keep = np.isin(ids, list(non_content_ids))
        return pos[keep]

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

    def _answer_stats(self, canvas: Canvas) -> tuple[int, float]:
        pos = canvas.field_positions("answer")
        ids = [int(i) for i in canvas.ids[pos] if int(i) != canvas.mask_id]
        ids = [i for i in ids if i not in getattr(self.tok, "non_content_ids", set())]
        if not ids:
            return 0, 0.0
        counts = Counter(ids)
        repeat_frac = max(counts.values()) / max(len(ids), 1)
        return len(ids), float(repeat_frac)

    def _is_degenerate_answer(self, canvas: Canvas, cycle: int) -> bool:
        tokens, repeat_frac = self._answer_stats(canvas)
        return (
            cycle + 1 >= self.cfg.loop.degenerate_min_cycles
            and tokens >= self.cfg.loop.degenerate_min_answer_tokens
            and repeat_frac >= self.cfg.loop.degenerate_repeat_frac
        )


    def _anchor_remask(self, canvas: Canvas, rescored: Snapshot, exempt_positions: np.ndarray) -> tuple[int, int]:
        pos = canvas.field_positions("answer")
        committed_mask = canvas.ids[pos] != canvas.mask_id
        if not np.any(committed_mask):
            return 0, 0
        non_content_ids = getattr(self.tok, "non_content_ids", set())
        content_mask = ~np.isin(canvas.ids[pos], list(non_content_ids)) if non_content_ids else np.ones(len(pos), dtype=bool)
        exempt = np.isin(pos, exempt_positions) if len(exempt_positions) else np.zeros(len(pos), dtype=bool)
        eligible = committed_mask & content_mask & ~exempt
        cand_pos = pos[eligible]
        if len(cand_pos) <= 1:
            return 0, len(cand_pos)
        keep_n = max(1, int(np.ceil(self.cfg.loop.anchor_keep_frac * len(cand_pos))))
        rel = rescored.relevance[cand_pos]
        order = np.argsort(-rel)
        keep = set(cand_pos[order[:keep_n]].tolist())
        remask = np.array([p for p in cand_pos if int(p) not in keep], dtype=np.int64)
        if len(remask):
            canvas.remask(remask)
        return int(len(remask)), int(len(cand_pos))

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

        closure_s = 0.0
        for c in range(cfg.loop.c_max):
            ctx = self.compose_context(query, evidence, memory, c)
            t_denoise = time.perf_counter()
            result = self.backbone.denoise(
                ctx, canvas, num_steps=cfg.steps_per_cycle,
                n_parallel=cfg.loop.n_parallel, seed=seed + c,
                control_sweeps=cfg.loop.control_sweeps)
            denoise_s = time.perf_counter() - t_denoise
            snaps = result.trajectories
            best = max(snaps, key=lambda s: s.mean_answer_confidence())
            t_decide = time.perf_counter()
            vote = self._vote(snaps, c)
            gap = decide(snaps, tok, cfg.trigger, self.calibrator)
            action, vetoed = self._fuse(vote, gap, best.mean_answer_confidence(), c)
            decision_s = time.perf_counter() - t_decide
            log = CycleLog(cycle=c, action=action.value, gap=gap, vetoed=vetoed,
                           mean_answer_conf=best.mean_answer_confidence(),
                           mean_answer_rel=best.mean_answer_relevance(),
                           denoise_s=denoise_s,
                           decision_s=decision_s)
            logs.append(log)

            if self._is_degenerate_answer(best.canvas, c):
                terminated_by = "degenerate"
                canvas = best.canvas
                break

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
            if evidence.in_context():
                action_input = "" if mostly_irrelevant else best.canvas.field_text(tok, "action_input")
                words = salient_words(snaps, tok, self.idf, cfg.loop.salient_words,
                                      exclude=set(query.lower().split()))
                focus = " ".join(gap.focus_words + [action_input] + words).strip()
            else:
                focus = " ".join(gap.focus_words).strip()
            t_retrieval = time.perf_counter()
            hits = store.fused_search(query, focus, cfg.retrieval, exclude=evidence.seen_ids)
            log.retrieval_s = time.perf_counter() - t_retrieval
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
            pre_ctx = ctx
            new_ctx = self.compose_context(query, evidence, memory, c + 1)
            t_rescore = time.perf_counter()
            before_rescored = self.backbone.rescore(pre_ctx, canvas)
            rescored = self.backbone.rescore(new_ctx, canvas)
            log.rescore_s = time.perf_counter() - t_rescore
            exempt_positions = self._non_content_positions(canvas)
            report = RollbackReport(positions=np.array([], dtype=np.int64))
            if cfg.rollback.enabled:
                t_rollback = time.perf_counter()
                report = rollback_pass(canvas, before_rescored, rescored, cfg.rollback,
                                       contradicted_positions=contradict_ids,
                                       exempt_positions=exempt_positions,
                                       tokenizer=tok)
                log.rollback_s = time.perf_counter() - t_rollback
                for ph in report.revised_phrases:
                    memory.add_revision(ph)
            else:
                log.rollback_s = 0.0
            log.rollback = report

            # -------------------------- SPREAD re-mask + re-compose (step 6)
            anchor_remasked, anchor_candidates = self._anchor_remask(canvas, rescored, exempt_positions)
            log.anchor_remasked = anchor_remasked
            log.anchor_candidates = anchor_candidates
            canvas.remask_fields()          # thought/action/action_input/critique
            digest = " | ".join(f"[{i}] {it.text[:80]}" for i, it in
                                enumerate(evidence.in_context()[:4], start=1)) or "(none)"
            canvas.write_observation(tok, digest)
            memory.add_cycle(best.canvas.field_text(tok, "thought"))

        # ------------------------------------------------------ closure pass
        if canvas.masked("answer").any():
            ctx = self.compose_context(query, evidence, memory, cfg.loop.c_max)
            t_closure = time.perf_counter()
            result = self.backbone.denoise(ctx, canvas, num_steps=(cfg.loop.closure_steps or cfg.loop.total_steps),
                                           n_parallel=1, seed=seed + 999,
                                           control_sweeps=cfg.loop.control_sweeps)
            closure_s = time.perf_counter() - t_closure
            canvas = result.trajectories[0].canvas
        answer = clean_answer(canvas.field_text(tok, "answer"))
        n_ret = sum(1 for l in logs if l.retrieved_ids)
        if logs:
            logs[-1].closure_s = closure_s
        return ErgoResult(answer=answer, cycles=logs, evidence=evidence,
                          n_retrievals=n_ret, terminated_by=terminated_by)
