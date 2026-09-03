"""Inference interface for the frozen PhishGuard V3 model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

DEFAULT_MODEL_PATH = Path("artifacts/models/calibration_v3_rootcanon/sigmoid_grouped_cv.joblib")

MODEL_VERSION = "tfidf-logistic-v3-rootcanon"

FROZEN_THRESHOLD = 0.768113160039295


@dataclass(frozen=True)
class PredictionResult:
    """Prediction returned by the frozen PhishGuard model."""

    url: str
    phishing_probability: float
    prediction: int
    label: str
    threshold: float
    model_version: str

    def to_dict(self) -> dict[str, str | float | int]:
        """Return a JSON-serializable representation."""
        return asdict(self)


class PhishGuardPredictor:
    """Load and serve predictions from the frozen V3 classifier."""

    def __init__(
        self,
        *,
        model_path: Path = DEFAULT_MODEL_PATH,
        threshold: float = FROZEN_THRESHOLD,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must lie within [0, 1].")

        if not model_path.exists():
            raise FileNotFoundError(f"Frozen model not found at {model_path}.")

        self.model_path = model_path
        self.threshold = float(threshold)

        self.estimator: Any = joblib.load(model_path)

        self._positive_class_index = self._find_positive_class_index()

    def _find_positive_class_index(self) -> int:
        """Locate phishing class 1 in estimator classes."""
        classes = np.asarray(self.estimator.classes_)

        indices = np.flatnonzero(classes == 1)

        if len(indices) != 1:
            raise ValueError("Expected exactly one positive class labelled 1.")

        return int(indices[0])

    def predict_proba(
        self,
        urls: list[str],
    ) -> np.ndarray:
        """Return phishing probabilities for URL strings."""
        if not urls:
            raise ValueError("At least one URL is required.")

        cleaned_urls = []

        for url in urls:
            if not isinstance(
                url,
                str,
            ):
                raise TypeError("Every URL must be a string.")

            normalized_input = url.strip()

            if not normalized_input:
                raise ValueError("URL values must not be empty.")

            cleaned_urls.append(normalized_input)

        probabilities = np.asarray(
            self.estimator.predict_proba(cleaned_urls),
            dtype=float,
        )

        return probabilities[
            :,
            self._positive_class_index,
        ]

    def predict_many(
        self,
        urls: list[str],
    ) -> list[PredictionResult]:
        """Classify multiple URLs."""
        probabilities = self.predict_proba(urls)

        results = []

        for url, probability in zip(
            urls,
            probabilities,
            strict=True,
        ):
            prediction = int(probability >= self.threshold)

            results.append(
                PredictionResult(
                    url=url,
                    phishing_probability=float(probability),
                    prediction=prediction,
                    label=("phishing" if prediction == 1 else "legitimate"),
                    threshold=self.threshold,
                    model_version=MODEL_VERSION,
                )
            )

        return results

    def predict(
        self,
        url: str,
    ) -> PredictionResult:
        """Classify one URL."""
        return self.predict_many([url])[0]
