import numpy as np
import pandas as pd
import pytest

from phishguard.training.calibration import (
    _grouped_fit_calibration_split,
    expected_calibration_error,
)


def test_ece_is_zero_for_perfect_binary_predictions() -> None:
    targets = np.array([0, 0, 1, 1], dtype=np.int8)
    probabilities = np.array([0.0, 0.0, 1.0, 1.0])

    assert expected_calibration_error(
        targets,
        probabilities,
    ) == pytest.approx(0.0)


def test_ece_rejects_invalid_probabilities() -> None:
    targets = np.array([0, 1], dtype=np.int8)
    probabilities = np.array([-0.1, 1.1])

    with pytest.raises(ValueError):
        expected_calibration_error(
            targets,
            probabilities,
        )


def test_grouped_calibration_split_has_no_domain_overlap() -> None:
    rows = []

    for domain_index in range(20):
        target = domain_index % 2

        for row_index in range(4):
            rows.append(
                {
                    "url_model_input": (f"https://domain{domain_index}.com/path{row_index}"),
                    "registered_domain": (f"domain{domain_index}.com"),
                    "target": target,
                }
            )

    frame = pd.DataFrame(rows)

    fit_frame, calibration_frame = _grouped_fit_calibration_split(frame)

    fit_domains = set(fit_frame["registered_domain"])
    calibration_domains = set(calibration_frame["registered_domain"])

    assert fit_domains.isdisjoint(calibration_domains)
    assert set(fit_frame["target"]) == {0, 1}
    assert set(calibration_frame["target"]) == {0, 1}
