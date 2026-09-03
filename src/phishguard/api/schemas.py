"""Request and response schemas for the PhishGuard API."""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Single-URL phishing prediction request."""

    url: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="URL string to evaluate.",
    )


class PredictResponse(BaseModel):
    """Phishing prediction response."""

    url: str
    phishing_probability: float
    prediction: int
    label: str
    threshold: float
    model_version: str


class HealthResponse(BaseModel):
    """API health response."""

    status: str


class ModelInfoResponse(BaseModel):
    """Frozen model metadata."""

    model_version: str
    threshold: float
    model_type: str
