import numpy as np

from conase_geo.text_utils import encode_text_to_token_ids, scores_to_probs, split_from_group, tokenize_text


def test_split_from_group_is_stable() -> None:
    a = split_from_group("channel_123", seed=42, val_fraction=0.1, test_fraction=0.1)
    b = split_from_group("channel_123", seed=42, val_fraction=0.1, test_fraction=0.1)
    assert a == b
    assert a in {"train", "val", "test"}


def test_scores_to_probs_binary_and_multiclass_shapes() -> None:
    binary_scores = np.array([0.0, 2.0, -2.0], dtype=np.float64)
    binary_probs = scores_to_probs(binary_scores)
    assert binary_probs.shape == (3, 2)
    assert np.allclose(binary_probs.sum(axis=1), 1.0)

    multi_scores = np.array([[1.0, 0.0, -1.0], [0.2, 0.3, 0.4]], dtype=np.float64)
    multi_probs = scores_to_probs(multi_scores)
    assert multi_probs.shape == (2, 3)
    assert np.allclose(multi_probs.sum(axis=1), 1.0)


def test_tokenize_and_encode_text_ids() -> None:
    tokens = tokenize_text("Y'all ready to GO @ home?")
    assert tokens[:3] == ["y'all", "ready", "to"]
    vocab = {"[PAD]": 0, "[UNK]": 1, "y'all": 2, "ready": 3, "go": 4}
    ids = encode_text_to_token_ids("Y'all ready to GO!", vocab=vocab, max_len=6)
    assert ids.shape == (6,)
    assert ids.tolist()[:4] == [2, 3, 1, 4]
