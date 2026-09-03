import numpy as np

from phishguard.evaluation.external_ood import (
    _threshold_metrics,
    expected_calibration_error,
)


def test_external_threshold_metrics() -> None:
    targets = np.array(
        [
            1,
            1,
            0,
            0,
        ],
        dtype=np.int8,
    )

    scores = np.array(
        [
            0.9,
            0.4,
            0.8,
            0.1,
        ],
        dtype=float,
    )

    metrics = _threshold_metrics(
        targets,
        scores,
        threshold=0.5,
    )

    assert metrics["true_positive"] == 1

    assert metrics["false_negative"] == 1

    assert metrics["true_negative"] == 1

    assert metrics["false_positive"] == 1

    assert metrics["recall_tpr"] == 0.5

    assert metrics["false_positive_rate"] == 0.5


def test_ece_is_zero_for_perfect_binary_probabilities() -> None:
    targets = np.array(
        [
            0,
            1,
            0,
            1,
        ],
        dtype=np.int8,
    )

    probabilities = np.array(
        [
            0.0,
            1.0,
            0.0,
            1.0,
        ],
        dtype=float,
    )

    assert (
        expected_calibration_error(
            targets,
            probabilities,
        )
        == 0.0
    )


def test_ece_rejects_invalid_probabilities() -> None:
    targets = np.array(
        [
            0,
            1,
        ],
        dtype=np.int8,
    )

    probabilities = np.array(
        [
            -0.1,
            1.1,
        ],
        dtype=float,
    )

    try:
        expected_calibration_error(
            targets,
            probabilities,
        )

    except ValueError:
        pass

    else:
        raise AssertionError("Expected ValueError for probabilities outside [0, 1].")
