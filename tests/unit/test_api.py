"""Socket-free tests for the PhishGuard FastAPI application."""

from dataclasses import dataclass
from typing import Any

import pytest
from fastapi import FastAPI
from pydantic import ValidationError
from starlette.requests import Request

from phishguard.api.app import create_app
from phishguard.api.schemas import (
    MAX_BATCH_SIZE,
    BatchPredictRequest,
    PredictRequest,
)
from phishguard.inference import (
    FROZEN_THRESHOLD,
    MODEL_VERSION,
)


@dataclass(frozen=True)
class FakePrediction:
    """Fake predictor result for API tests."""

    url: str
    phishing_probability: float
    prediction: int
    label: str
    threshold: float
    model_version: str

    def to_dict(
        self,
    ) -> dict[str, str | float | int]:
        return {
            "url": self.url,
            "phishing_probability": self.phishing_probability,
            "prediction": self.prediction,
            "label": self.label,
            "threshold": self.threshold,
            "model_version": self.model_version,
        }


class FakePredictor:
    """Predictor stub used by API tests."""

    def predict(
        self,
        url: str,
    ) -> FakePrediction:
        if not url.strip():
            raise ValueError("URL values must not be empty.")

        return FakePrediction(
            url=url,
            phishing_probability=0.9,
            prediction=1,
            label="phishing",
            threshold=FROZEN_THRESHOLD,
            model_version=MODEL_VERSION,
        )

    def predict_many(
        self,
        urls: list[str],
    ) -> list[FakePrediction]:
        """Return fake batch predictions."""
        return [self.predict(url) for url in urls]


def _find_endpoint(
    application: FastAPI,
    *,
    path: str,
    method: str,
) -> Any:
    """Return a registered FastAPI endpoint."""
    for route in application.routes:
        route_path = getattr(
            route,
            "path",
            None,
        )

        methods = getattr(
            route,
            "methods",
            set(),
        )

        if route_path == path and method in methods:
            return route.endpoint

    raise AssertionError(f"Route {method} {path} not found.")


@pytest.fixture
def application() -> FastAPI:
    """Create API using a fake predictor."""
    return create_app(
        predictor=FakePredictor(),  # type: ignore[arg-type]
    )


def test_health(
    application: FastAPI,
) -> None:
    endpoint = _find_endpoint(
        application,
        path="/health",
        method="GET",
    )

    response = endpoint()

    assert response.status == "ok"


def test_model_info(
    application: FastAPI,
) -> None:
    endpoint = _find_endpoint(
        application,
        path="/model",
        method="GET",
    )

    response = endpoint()

    assert response.model_version == MODEL_VERSION

    assert response.threshold == FROZEN_THRESHOLD


def test_predict(
    application: FastAPI,
) -> None:
    endpoint = _find_endpoint(
        application,
        path="/predict",
        method="POST",
    )

    request = Request(
        {
            "type": "http",
            "app": application,
        }
    )

    payload = PredictRequest(url="https://example.com/login")

    response = endpoint(
        payload,
        request,
    )

    assert response.prediction == 1
    assert response.label == "phishing"

    assert response.phishing_probability == pytest.approx(0.9)

    assert response.model_version == MODEL_VERSION


def test_missing_url_rejected() -> None:
    with pytest.raises(
        ValidationError,
    ):
        PredictRequest.model_validate({})


def test_blank_url_rejected() -> None:
    with pytest.raises(
        ValidationError,
    ):
        PredictRequest(url="   ")


def test_batch_predict(
    application: FastAPI,
) -> None:
    endpoint = _find_endpoint(
        application,
        path="/predict/batch",
        method="POST",
    )

    request = Request(
        {
            "type": "http",
            "app": application,
        }
    )

    payload = BatchPredictRequest(
        urls=[
            "https://example.com",
            "https://example.org/login",
        ]
    )

    response = endpoint(
        payload,
        request,
    )

    assert response.count == 2
    assert len(response.predictions) == 2

    assert response.predictions[0].model_version == MODEL_VERSION

    assert response.predictions[1].model_version == MODEL_VERSION


def test_empty_batch_rejected() -> None:
    with pytest.raises(
        ValidationError,
    ):
        BatchPredictRequest(urls=[])


def test_batch_size_limit() -> None:
    with pytest.raises(
        ValidationError,
    ):
        BatchPredictRequest(urls=["https://example.com"] * (MAX_BATCH_SIZE + 1))
