"""Request and response schemas for the PhishGuard API."""

from pydantic import BaseModel, Field, field_validator

MAX_BATCH_SIZE = 100


class PredictRequest(BaseModel):
    """Single-URL phishing prediction request."""

    url: str = Field(
        ...,
        min_length=1,
        max_length=4096,
        description="URL string to evaluate.",
    )

    @field_validator("url")
    @classmethod
    def reject_blank_url(
        cls,
        value: str,
    ) -> str:
        """Reject whitespace-only URL values."""
        if not value.strip():
            raise ValueError("URL must not be blank.")

        return value


class BatchPredictRequest(BaseModel):
    """Batch URL prediction request."""

    urls: list[str] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
    )

    @field_validator("urls")
    @classmethod
    def validate_urls(
        cls,
        values: list[str],
    ) -> list[str]:
        """Validate all URLs in a batch."""
        for value in values:
            if not value.strip():
                raise ValueError("Batch URLs must not be blank.")

            if len(value) > 4096:
                raise ValueError("Each URL must be at most 4096 characters.")

        return values


class PredictResponse(BaseModel):
    """Phishing prediction response."""

    url: str
    phishing_probability: float
    prediction: int
    label: str
    threshold: float
    model_version: str


class BatchPredictResponse(BaseModel):
    """Batch phishing prediction response."""

    count: int
    predictions: list[PredictResponse]


class HealthResponse(BaseModel):
    """API health response."""

    status: str


class ModelInfoResponse(BaseModel):
    """Frozen model metadata."""

    model_version: str
    threshold: float
    model_type: str
