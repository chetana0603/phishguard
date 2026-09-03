"""Tests for frozen-model inference."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from phishguard.inference.predictor import (
    MODEL_VERSION,
    PhishGuardPredictor,
)


def build_predictor(
    monkeypatch: pytest.MonkeyPatch,
    *,
    probabilities: np.ndarray,
    threshold: float = 0.75,
) -> PhishGuardPredictor:
    """Build predictor with a mocked estimator."""
    estimator = Mock()

    estimator.classes_ = np.array(
        [
            0,
            1,
        ]
    )

    estimator.predict_proba.return_value = probabilities

    monkeypatch.setattr(
        Path,
        "exists",
        lambda self: True,
    )

    monkeypatch.setattr(
        "phishguard.inference.predictor.joblib.load",
        lambda path: estimator,
    )

    return PhishGuardPredictor(
        model_path=Path("fake-model.joblib"),
        threshold=threshold,
    )


def test_predict_phishing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = build_predictor(
        monkeypatch,
        probabilities=np.array(
            [
                [
                    0.10,
                    0.90,
                ]
            ]
        ),
    )

    result = predictor.predict("https://example.com/login")

    assert result.prediction == 1
    assert result.label == "phishing"
    assert result.phishing_probability == pytest.approx(0.90)
    assert result.model_version == MODEL_VERSION


def test_predict_legitimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = build_predictor(
        monkeypatch,
        probabilities=np.array(
            [
                [
                    0.40,
                    0.60,
                ]
            ]
        ),
    )

    result = predictor.predict("https://example.com")

    assert result.prediction == 0
    assert result.label == "legitimate"


def test_threshold_is_inclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = build_predictor(
        monkeypatch,
        probabilities=np.array(
            [
                [
                    0.25,
                    0.75,
                ]
            ]
        ),
        threshold=0.75,
    )

    result = predictor.predict("https://example.com")

    assert result.prediction == 1


def test_predict_many(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = build_predictor(
        monkeypatch,
        probabilities=np.array(
            [
                [
                    0.80,
                    0.20,
                ],
                [
                    0.05,
                    0.95,
                ],
            ]
        ),
    )

    results = predictor.predict_many(
        [
            "https://example.com",
            "http://example.net/login",
        ]
    )

    assert len(results) == 2
    assert results[0].prediction == 0
    assert results[1].prediction == 1


def test_empty_batch_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = build_predictor(
        monkeypatch,
        probabilities=np.empty(
            (
                0,
                2,
            )
        ),
    )

    with pytest.raises(
        ValueError,
        match="At least one URL",
    ):
        predictor.predict_many([])


def test_blank_url_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictor = build_predictor(
        monkeypatch,
        probabilities=np.array(
            [
                [
                    0.5,
                    0.5,
                ]
            ]
        ),
    )

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        predictor.predict("   ")


def test_invalid_threshold_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        Path,
        "exists",
        lambda self: True,
    )

    with pytest.raises(
        ValueError,
        match=r"\[0, 1\]",
    ):
        PhishGuardPredictor(
            model_path=Path("fake-model.joblib"),
            threshold=1.5,
        )
