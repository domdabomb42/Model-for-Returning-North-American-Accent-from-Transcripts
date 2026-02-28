from conase_geo.data.parse_tokens import parse_text_pos, parse_token_times_json


def test_parse_text_pos_handles_underscores_from_right() -> None:
    text = "new_york_NNP_1.23 welcome_UH_1.37 malformedtoken hi_there_UH_2.40"
    tokens = parse_text_pos(text)
    assert len(tokens) == 3
    assert tokens[0].token == "new_york"
    assert tokens[0].pos == "NNP"
    assert tokens[0].time == 1.23
    assert tokens[2].token == "hi_there"
    assert tokens[2].time == 2.40


def test_parse_token_times_json_supports_dicts_and_numbers() -> None:
    raw = '[{"token":"a","time":0.2}, 0.4, {"time":"0.9"}, {"bad":1}]'
    values = parse_token_times_json(raw)
    assert values == [0.2, 0.4, 0.9]
