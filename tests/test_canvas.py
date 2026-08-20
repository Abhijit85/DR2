import numpy as np

from ergo.canvas import Action, Canvas, FIELDS
from ergo.config import CanvasConfig
from ergo.tokenization import SimpleTokenizer


def _canvas(tok):
    cfg = CanvasConfig(len_thought=6, len_action=2, len_action_input=4,
                       len_observation=6, len_critique=4, len_answer=10)
    return Canvas.build(tok, cfg)


def test_labels_committed_contents_masked():
    tok = SimpleTokenizer()
    cv = _canvas(tok)
    for f in FIELDS:
        assert cv.masked(f).sum() == (cv.spans[f].stop - cv.spans[f].start)
    assert not (cv.ids[: cv.spans["thought"].start] == tok.mask_id).any()  # label tokens


def test_observation_is_system_committed():
    tok = SimpleTokenizer()
    cv = _canvas(tok)
    cv.write_observation(tok, "[1] Paris is the capital of France")
    assert not cv.masked("observation").any()
    assert "paris" in cv.field_text(tok, "observation").lower()


def test_action_resolution_constrained():
    tok = SimpleTokenizer()
    cv = _canvas(tok)
    # scorer prefers 'finish'
    def seq_lp(positions, cand):
        word = tok.decode(cand.tolist())
        return {"retrieve": -5.0, "memory": -9.0, "finish": -1.0}[word]
    act = cv.resolve_action(seq_lp)
    assert act == Action.FINISH
    cv.commit_action(tok, act)
    assert "finish" in cv.field_text(tok, "action")


def test_remask_fields_keeps_answer_anchors():
    tok = SimpleTokenizer()
    cv = _canvas(tok)
    apos = cv.field_positions("answer")[:3]
    cv.commit(apos, np.array(tok.encode("a b c")), np.array([0.9, 0.9, 0.9]))
    tpos = cv.field_positions("thought")[:2]
    cv.commit(tpos, np.array(tok.encode("x y")), np.array([0.9, 0.9]))
    cv.remask_fields()
    assert (cv.ids[cv.spans["thought"]] == tok.mask_id).all()
    assert (cv.ids[apos] != cv.mask_id).all()          # anchors preserved
    assert (cv.committed_prob[apos] == 0.9).all()       # pi_i preserved for the LRT
