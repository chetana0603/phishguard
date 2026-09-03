"""FastAPI application for PhishGuard inference."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from phishguard.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)
from phishguard.inference import (
    FROZEN_THRESHOLD,
    MODEL_VERSION,
    PhishGuardPredictor,
)


def create_app(
    predictor: PhishGuardPredictor | None = None,
) -> FastAPI:
    """Create the PhishGuard API application."""

    @asynccontextmanager
    async def lifespan(
        app: FastAPI,
    ) -> AsyncIterator[None]:
        if not hasattr(
            app.state,
            "predictor",
        ):
            app.state.predictor = PhishGuardPredictor()

        yield

    application = FastAPI(
        title="PhishGuard API",
        description=("URL-only phishing risk scoring using the frozen PhishGuard V3 model."),
        version="1.0.0",
        lifespan=lifespan,
    )
    if predictor is not None:
        application.state.predictor = predictor

    @application.get(
        "/health",
        response_model=HealthResponse,
    )
    def health() -> HealthResponse:
        """Return API health status."""
        return HealthResponse(status="ok")

    @application.get(
        "/model",
        response_model=ModelInfoResponse,
    )
    def model_info() -> ModelInfoResponse:
        """Return frozen model metadata."""
        return ModelInfoResponse(
            model_version=MODEL_VERSION,
            threshold=FROZEN_THRESHOLD,
            model_type=("character TF-IDF + Logistic Regression + grouped sigmoid calibration"),
        )

    @application.post(
        "/predict",
        response_model=PredictResponse,
    )
    def predict_url(
        payload: PredictRequest,
        request: Request,
    ) -> PredictResponse:
        """Score one URL using the frozen model."""
        try:
            result = request.app.state.predictor.predict(payload.url)
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise HTTPException(
                status_code=422,
                detail=str(exc),
            ) from exc

        return PredictResponse(**result.to_dict())

    return application


app = create_app()
