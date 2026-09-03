$ErrorActionPreference = "Stop"

Write-Host "Running Phase 2C-B robustness evaluation..."

uv run python -m phishguard.evaluation.robustness

Write-Host "Phase 2C-B complete."