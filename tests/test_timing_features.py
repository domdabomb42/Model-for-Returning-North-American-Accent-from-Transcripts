from conase_geo.features.timing_features import compute_timing_features


def test_compute_timing_features_expected_stats() -> None:
    features = compute_timing_features(
        token_times=[0.1, 0.6, 1.0, 1.9],
        clip_start=0.0,
        clip_end=2.0,
        pause_threshold=0.35,
    )
    assert abs(features["words_per_second"] - 2.0) < 1e-6
    assert abs(features["pause_max"] - 0.9) < 1e-6
    assert features["pause_rate"] > 0.0
    assert features["rhythm_cv"] > 0.0


def test_compute_timing_features_empty_tokens() -> None:
    features = compute_timing_features(token_times=[], clip_start=0.0, clip_end=2.0, pause_threshold=0.35)
    assert features["words_per_second"] == 0.0
    assert features["pause_mean"] == 0.0
    assert features["n_tokens"] == 0.0
