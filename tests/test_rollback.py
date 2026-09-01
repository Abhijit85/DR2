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
    conf[pos[0]] = 0.01
    report = rollback_pass(cv, _snapshot_with(cv, np.full(len(cv.ids), 0.9), rel), _snapshot_with(cv, conf, rel),
                           RollbackConfig(alpha_roll=0.135), )
    assert report.n_rolled == 1
    assert cv.ids[pos[0]] == tok.mask_id
    assert cv.ids[pos[1]] != tok.mask_id


def test_llr_threshold_boundary():
    from ergo.rollback import rollback_pass
    tok = SimpleTokenizer()
    delta = RollbackConfig(alpha_roll=0.135).delta
    for eps, expect_roll in ((-0.05, True), (0.05, False)):
        cv = _small_canvas(tok)
        pos = _commit_answer(cv, tok, ["fact"], prob=0.8)
        conf = np.full(len(cv.ids), 0.9)
        rel = np.full(len(cv.ids), 0.9)
        conf[pos[0]] = 0.8 * math.exp(-(delta - eps))
        before_conf = np.full(len(cv.ids), 0.9)
        before_conf[pos[0]] = 0.8
        report = rollback_pass(cv, _snapshot_with(cv, before_conf, rel), _snapshot_with(cv, conf, rel),
                               RollbackConfig(alpha_roll=0.135), )
        assert (report.n_rolled == 1) == expect_roll


def test_cap_limits_rollback_to_worst_offenders():
    from ergo.rollback import rollback_pass
    tok = SimpleTokenizer()
    cv = _small_canvas(tok)
    words = [f"w{i}" for i in range(8)]
    pos = _commit_answer(cv, tok, words, prob=0.9)
    conf = np.full(len(cv.ids), 0.9)
    rel = np.full(len(cv.ids), 0.9)
    conf[pos] = np.linspace(0.001, 0.02, len(pos))
    cfg = RollbackConfig(cap_fraction=0.25)
    report = rollback_pass(cv, _snapshot_with(cv, np.full(len(cv.ids), 0.9), rel), _snapshot_with(cv, conf, rel), cfg, )
    assert report.capped and report.n_rolled == 2
    assert set(report.positions) == set(pos[:2])


def test_non_content_positions_do_not_affect_lrt_rollback():
    from ergo.rollback import rollback_pass
    tok = SimpleTokenizer()
    cv = _small_canvas(tok)
    pos = cv.field_positions("answer")[:4]
    cv.commit(pos, np.array([tok.pad_id, tok.pad_id, tok.encode("real")[0], tok.encode("text")[0]]), np.full(4, 0.9))
    conf = np.full(len(cv.ids), 0.95)
    rel = np.full(len(cv.ids), 0.2)
    report = rollback_pass(cv, _snapshot_with(cv, conf, rel), _snapshot_with(cv, conf, rel), RollbackConfig(),
                           exempt_positions=pos[:2])
    assert report.n_rolled == 0
    assert cv.ids[pos[2]] != tok.mask_id
    assert cv.ids[pos[3]] != tok.mask_id


def test_contradicted_positions_still_rollback_without_lrt_failure():
    from ergo.rollback import rollback_pass
    tok = SimpleTokenizer()
    cv = _small_canvas(tok)
    pos = _commit_answer(cv, tok, ["w0", "w1", "w2"], prob=0.95)
    conf = np.full(len(cv.ids), 0.95)
    rel = np.full(len(cv.ids), 0.1)
    report = rollback_pass(cv, _snapshot_with(cv, conf, rel), _snapshot_with(cv, conf, rel), RollbackConfig(),
                           contradicted_positions=np.array([pos[1]]))
    assert report.n_rolled == 1
    assert report.n_contradicted == 1
    assert cv.ids[pos[1]] == tok.mask_id
