import numpy as np

from ergo.backbones.base import Snapshot, relevance_scores
from ergo.canvas import Canvas
from ergo.config import CanvasConfig, TriggerConfig
from ergo.tokenization import SimpleTokenizer
from ergo.trigger import ConformalCalibrator, decide, gap_scores


def _make_snapshot(tok, answer_words, conf_vals, rel_vals):
    cfg = CanvasConfig(len_thought=4, len_action=2, len_action_input=4,
                       len_observation=4, len_critique=4,
                       len_answer=len(answer_words))
    cv = Canvas.build(tok, cfg)
    pos = cv.field_positions("answer")
    ids = np.array(tok.encode(" ".join(answer_words)), dtype=np.int64)
    L = len(cv.ids)
    pred = cv.ids.copy()
    pred[pos] = ids
    conf = np.full(L, 0.9)
    rel = np.full(L, 0.5)
    conf[pos] = conf_vals
    rel[pos] = rel_vals
    ent = np.clip(1 - conf, 0.01, 0.99)
    margin = np.clip(1 - conf, 0.01, 0.99)
    return Snapshot(canvas=cv, predicted_ids=pred, confidence=conf,
                    relevance=rel, entropy=ent, margin=margin)


def _gap_snapshot(tok):
    """Confident frame ('the prize was awarded in') + uncertain details."""
    words = "the prize was awarded to slotA in yearB .".split()
    conf = np.array([.95, .95, .95, .95, .9, .2, .9, .15, .9])
    rel = np.full(len(words), 0.9)
    return _make_snapshot(tok, words, conf, rel)


def _settled_snapshot(tok):
    words = "the prize was awarded to curie in 1911 .".split()
    conf = np.linspace(0.90, 0.96, len(words))   # slight variation: frame/detail split exists
    rel = np.full(len(words), 0.9)
    return _make_snapshot(tok, words, conf, rel)


def _padded_short_answer_snapshot(tok):
    cfg = CanvasConfig(len_thought=4, len_action=2, len_action_input=4,
                       len_observation=4, len_critique=4, len_answer=8)
    cv = Canvas.build(tok, cfg)
    pos = cv.field_positions("answer")
    ids = cv.ids.copy()
    answer_ids = np.array(tok.encode("october 1922"), dtype=np.int64)
    ids[pos[:2]] = answer_ids
    pred = ids.copy()
    L = len(cv.ids)
    conf = np.full(L, 0.95)
    conf[pos[0]] = 0.9
    conf[pos[1]] = 0.3
    rel = np.full(L, 0.9)
    ent = np.full(L, 0.2)
    margin = np.full(L, 0.2)
    return Snapshot(canvas=cv, predicted_ids=pred, confidence=conf,
                    relevance=rel, entropy=ent, margin=margin)


def test_gap_score_higher_for_uncertain_details():
    tok = SimpleTokenizer()
    g_gap = max(s.score for s in gap_scores(_gap_snapshot(tok), tok, TriggerConfig()))
    g_settled = max(s.score for s in gap_scores(_settled_snapshot(tok), tok, TriggerConfig()))
    assert g_gap > g_settled


def test_relevance_gates_stylistic_uncertainty():
    """High-entropy span that is NOT query-relevant scores lower."""
    tok = SimpleTokenizer()
    words = "the prize was awarded to slotA in yearB .".split()
    conf = np.array([.95, .95, .95, .95, .9, .2, .9, .15, .9])
    hi = gap_scores(_make_snapshot(tok, words, conf, np.full(9, 0.9)), tok, TriggerConfig())
    lo = gap_scores(_make_snapshot(tok, words, conf, np.full(9, 0.3)), tok, TriggerConfig())
    assert max(s.score for s in hi) > max(s.score for s in lo)


def test_conformal_fires_on_outlier_and_controls_nulls():
    tok = SimpleTokenizer()
    cal = ConformalCalibrator()
    rng = np.random.default_rng(0)
    null_scores = rng.uniform(0.0, 0.1, size=200)      # typical settled spans
    cal.add_null_scores(null_scores)
    cfg = TriggerConfig(alpha_fire=0.05, use_conformal=True)
    d = decide([_gap_snapshot(tok)], tok, cfg, cal)
    assert d.rule == "conformal" and d.fire and d.p_value <= 0.05
    assert d.focus_words                                # frame words become focus
    d2 = decide([_settled_snapshot(tok)], tok, cfg, cal)
    # settled canvas score sits inside/near the null distribution -> no fire
    assert not d2.fire


def test_fallback_rule_quiet_when_uniform():
    tok = SimpleTokenizer()
    cfg = TriggerConfig(use_conformal=False)
    # all-settled canvas: no outlier -> stays quiet
    d = decide([_settled_snapshot(tok)], tok, cfg, None)
    assert not d.fire


def test_conformal_pvalue_monotone():
    cal = ConformalCalibrator()
    cal.add_null_scores(np.linspace(0, 1, 100))
    assert cal.p_value(0.99) < cal.p_value(0.5) < cal.p_value(0.01)


def test_collect_null_scores_like_filter_prefers_strong_answers():
    rows = [
        {"em": 1.0, "f1": 1.0, "cycles": [{"gap_fire": False, "gap_all_scores": [0.1, 0.2]}]},
        {"em": 0.0, "f1": 0.7, "cycles": [{"gap_fire": False, "gap_all_scores": [0.3]}]},
        {"em": 0.0, "f1": 0.2, "cycles": [{"gap_fire": False, "gap_all_scores": [0.9]}]},
        {"em": 0.0, "f1": 0.8, "cycles": [{"gap_fire": True, "gap_all_scores": [0.8]}]},
    ]
    from scripts.summarize_pilot import collect_null_scores
    assert collect_null_scores(rows) == [0.1, 0.2, 0.3]


def test_standardized_relevance_preserves_order_and_spreads_scores():
    hidden = np.array([[1.0, 0.0], [0.8, 0.2], [0.2, 0.8], [-1.0, 0.0]], dtype=float)
    h_q = np.array([1.0, 0.0], dtype=float)
    raw = relevance_scores(hidden, h_q, standardize=False)
    std = relevance_scores(hidden, h_q, standardize=True)
    assert list(np.argsort(raw)) == list(np.argsort(std))
    assert (std.max() - std.min()) > (raw.max() - raw.min())
    assert std.min() < raw.min()


def test_short_answer_survives_span_segmentation_with_padding():
    tok = SimpleTokenizer()
    spans = gap_scores(_padded_short_answer_snapshot(tok), tok, TriggerConfig())
    assert len(spans) == 1
    assert len(spans[0].positions) == 2
