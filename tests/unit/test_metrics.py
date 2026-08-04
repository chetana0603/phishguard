import pytest

from phishguard.evaluation.metrics import (
    evaluate_binary_scores,
    expected_calibration_error,
)


def test_binary_metrics_match_known_confusion_matrix() -> None:
    metrics = evaluate_binary_scores(
        [0, 0, 1, 1],
        [0.1, 0.8, 0.9, 0.2],
        threshold=0.5,
    )
    assert metrics.true_positive == 1
    assert metrics.false_positive == 1
    assert metrics.true_negative == 1
    assert metrics.false_negative == 1
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == pytest.approx(0.5)
    assert metrics.false_positive_rate == pytest.approx(0.5)


def test_perfect_scores_have_zero_calibration_error() -> None:
    error = expected_calibration_error([0, 1], [0.0, 1.0], n_bins=2)
    assert error == pytest.approx(0.0)


def test_metrics_reject_out_of_range_scores() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        evaluate_binary_scores([0, 1], [0.1, 1.2], threshold=0.5)
