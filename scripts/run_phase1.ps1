$ErrorActionPreference = "Stop"

uv sync --locked
uv run python -m phishguard.data.download
uv run python -m phishguard.data.prepare
uv run python -m phishguard.data.audit
uv run python -m phishguard.data.split
uv run ruff check .
uv run ruff format --check .
uv run pytest --cov=phishguard --cov-report=term-missing

Write-Host "Phase 1 completed successfully." -ForegroundColor Green
