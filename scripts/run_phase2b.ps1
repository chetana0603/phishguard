$ErrorActionPreference = "Stop"

uv sync --locked
uv run ruff check . --no-cache
uv run ruff format --check .
uv run pytest --cov=phishguard --cov-report=term-missing
uv run python -m phishguard.training.tfidf_logistic
