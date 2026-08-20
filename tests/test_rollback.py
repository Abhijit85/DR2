import math

import numpy as np

from ergo.backbones.base import Snapshot
from ergo.canvas import Canvas
from ergo.config import CanvasConfig, RollbackConfig
from ergo.tokenization import SimpleTokenizer


def _small_canvas(tok):
    cfg = CanvasConfig(len_thought=4, len_action=2, len_action_input=4,
                       len_observation=4, len_critique=4, len_answer=8)
    return Canvas.build(tok, cfg)


def _snapshot_with(canvas, conf, rel):
    L = len(canvas.ids)
    return Snapshot(canvas=canvas, predicted_ids=canvas.ids.copy(),
                    confidence=conf, relevance=rel,
                    entropy=np.zeros(L), margin=np.zeros(L))


def _commit_answer(canvas, tok, words, prob):
    pos = canvas.field_positions("answer")[: len(words)]
    ids = np.array(tok.encode(" ".join(words)), dtype=np.int64)
    canvas.commit(pos, ids, np.full(len(pos), prob))
    return pos


def test_delta_is_log_inverse_alpha():
    assert math.isclose(RollbackConfig(alpha_roll=0.135).delta, math.log(1 / 0.135))
    assert math.isclose(RollbackConfig(alpha_roll=0.05).delta, math.log(20), rel_tol=1e-9)


def test_contradicted_token_rolls_back():
    from ergo.rollback import rollback_pass
    tok = SimpleTokenizer()
    cv = _small_canvas(tok)
    pos = _commit_answer(cv, tok, ["elsa", "einstein", "born"], prob=0.9)
    conf = np.full(len(cv.ids), 0.9)
    rel = np.full(len(cv.ids), 0.9)
    conf[pos[0]] = 0.01                      # evidence crushed p' for 'elsa'
    report = rollback_pass(cv, _snapshot_with(cv, conf, rel),
                           RollbackConfig(alpha_roll=0.135), tau_rel=0.5)
    assert report.n_rolled == 1
    assert cv.ids[pos[0]] == tok.mask_id     # cleared to MASK
    assert cv.ids[pos[1]] != tok.mask_id     # supported tokens survive


def test_llr_threshold_boundary():
    """Token survives iff log p' >= log pi - delta (strict inequality rule)."""
    from ergo.rollback import rollback_pass
    tok = SimpleTokenizer()
    delta = RollbackConfig(alpha_roll=0.135).delta
    for eps, expect_roll in ((-0.05, True), (0.05, False)):
        cv = _small_canvas(tok)
        pos = _commit_answer(cv, tok, ["fact"], prob=0.8)
        conf = np.full(len(cv.ids), 0.9)
        rel = np.full(len(cv.ids), 0.9)
        conf[pos[0]] = 0.8 * math.exp(-(delta - eps))
        report = rollback_pass(cv, _snapshot_with(cv, conf, rel),
                               RollbackConfig(alpha_roll=0.135), tau_rel=0.0)
        assert (report.n_rolled == 1) == expect_roll


def test_low_relevance_rolls_back_even_if_confident():
    from ergo.rollback import rollback_pass
    tok = SimpleTokenizer()
    cv = _small_canvas(tok)
    pos = _commit_answer(cv, tok, ["drifted", "phrase"], prob=0.95)
    conf = np.full(len(cv.ids), 0.95)
    rel = np.full(len(cv.ids), 0.9)
    rel[pos] = 0.2                            # SPREAD: low relevance -> re-mask
    report = rollback_pass(cv, _snapshot_with(cv, conf, rel),
                           RollbackConfig(), tau_rel=0.65)
    assert report.n_rolled == 2


def test_cap_limits_rollback_to_worst_offenders():
    from ergo.rollback import rollback_pass
    tok = SimpleTokenizer()
    cv = _small_canvas(tok)
    words = [f"w{i}" for i in range(8)]
    pos = _commit_answer(cv, tok, words, prob=0.9)
    conf = np.full(len(cv.ids), 0.9)
    rel = np.full(len(cv.ids), 0.9)
    conf[pos] = np.linspace(0.001, 0.02, len(pos))   # all fail the LRT
    cfg = RollbackConfig(cap_fraction=0.25)          # cap: 2 of 8
    report = rollback_pass(cv, _snapshot_with(cv, conf, rel), cfg, tau_rel=0.0)
    assert report.capped and report.n_rolled == 2
    # the two lowest-p' positions were the ones cleared
    assert set(report.positions) == set(pos[:2])
