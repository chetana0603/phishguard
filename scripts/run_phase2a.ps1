$ErrorActionPreference = "Stop"

Write-Host "Running Phase 2A: transparent rule baseline" -ForegroundColor Cyan

uv sync --locked
uv run ruff check .
uv run ruff format --check .
uv run pytest --cov=phishguard --cov-report=term-missing
uv run python -m phishguard.evaluation.rule_baseline

Write-Host "Phase 2A completed. Open reports/baselines/rule_baseline/report.md" -ForegroundColor Green
