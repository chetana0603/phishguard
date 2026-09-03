FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --frozen --no-dev

COPY artifacts/models/calibration_v3_rootcanon/sigmoid_grouped_cv.joblib artifacts/models/calibration_v3_rootcanon/sigmoid_grouped_cv.joblib

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "phishguard.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
