from phishguard.evaluation.thresholds import (
    build_threshold_table,
    select_threshold_profiles,
)


def test_threshold_profiles_are_selected() -> None:
    targets = [0, 0, 0, 0, 1, 1, 1, 1]
    scores = [0.02, 0.05, 0.10, 0.40, 0.20, 0.60, 0.80, 0.95]

    table = build_threshold_table(targets, scores)
    profiles = select_threshold_profiles(table)

    assert set(profiles) == {
        "high_security",
        "balanced",
        "conservative",
    }

    for profile in profiles.values():
        assert 0.0 <= profile.threshold <= 1.0
        assert profile.metrics["n_samples"] == 8
        assert isinstance(profile.constraints_met, bool)
        assert profile.selection_note
