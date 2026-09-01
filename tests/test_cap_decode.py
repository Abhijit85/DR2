import numpy as np

from ergo.backbones.base import cap_select
from ergo.canvas import Canvas
from ergo.config import CanvasConfig
from ergo.tokenization import SimpleTokenizer


def _answer_probs(canvas: Canvas, conf: np.ndarray) -> np.ndarray:
    tok = SimpleTokenizer()
    answer = canvas.field_positions("answer")
    vocab = tok.vocab_size + len(answer) + 4
    probs = np.full((len(canvas.ids), vocab), 1e-6, dtype=np.float64)
    probs /= probs.sum(axis=1, keepdims=True)
    for j, pos in enumerate(answer):
        target_id = 2 + j
        row = np.full(vocab, (1.0 - conf[j]) / (vocab - 1), dtype=np.float64)
        row[target_id] = conf[j]
        probs[pos] = row
    return probs


def test_cap_select_limits_low_confidence_parallel_commits():
    tok = SimpleTokenizer()
    canvas = Canvas.build(tok, CanvasConfig(len_answer=4, len_thought=1, len_action=1,
                                            len_action_input=1, len_observation=1,
                                            len_critique=1))
    answer = canvas.field_positions("answer")
    conf = np.array([0.42, 0.41, 0.40, 0.39], dtype=np.float64)
    rel = np.zeros(len(canvas.ids), dtype=np.float64)
    rel[answer] = np.array([0.9, 0.7, 0.6, 0.5], dtype=np.float64)
    probs = _answer_probs(canvas, conf)

    take = cap_select(canvas, probs, rel, k=3, tau=0.9, restrict=("answer",))

    assert take.tolist() == [int(answer[0])]


def test_cap_select_allows_parallel_commits_only_above_threshold():
    tok = SimpleTokenizer()
    canvas = Canvas.build(tok, CanvasConfig(len_answer=4, len_thought=1, len_action=1,
                                            len_action_input=1, len_observation=1,
                                            len_critique=1))
    answer = canvas.field_positions("answer")
    conf = np.array([0.95, 0.91, 0.89, 0.93], dtype=np.float64)
    rel = np.zeros(len(canvas.ids), dtype=np.float64)
    rel[answer] = np.array([0.3, 0.8, 0.95, 0.6], dtype=np.float64)
    probs = _answer_probs(canvas, conf)

    take = cap_select(canvas, probs, rel, k=3, tau=0.9, restrict=("answer",))

    assert take.tolist() == [int(answer[1]), int(answer[3]), int(answer[0])]
