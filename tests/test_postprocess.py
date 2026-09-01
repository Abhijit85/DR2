from ergo.postprocess import clean_answer


def test_clean_answer_strips_prefix():
    assert clean_answer('ANSWER: October 1922') == 'October 1922'


def test_clean_answer_truncates_at_field_marker():
    assert clean_answer('October 1922 [ACTION_INPUT') == 'October 1922'


def test_clean_answer_returns_empty_for_marker_garbage():
    assert clean_answer('ANSWER: [ACTION_INPUT') == ''


def test_clean_answer_normalizes_whitespace():
    assert clean_answer('  Final Answer:  October   1922  ') == 'October 1922'
