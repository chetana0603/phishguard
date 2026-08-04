# Apply the Phase 2A patch

1. Confirm Phase 1 is committed and pushed to `main`.
2. Create the branch:

```powershell
git switch -c phase2/rule-baseline
```

3. Extract this ZIP directly into the repository root and allow it to merge folders and overwrite:

- `src/phishguard/config.py`
- `src/phishguard/features/url_features.py`
- `src/phishguard/evaluation/metrics.py`

It does not replace the Phase 1 data scripts or your manual-download fallback.

4. Review `README_UPDATE.md` and apply its small root-README edits.
5. Run:

```powershell
uv run ruff format .
uv run ruff check .
uv run ruff format --check .
uv run pytest --cov=phishguard --cov-report=term-missing
uv run python -m phishguard.evaluation.rule_baseline
```

6. Open:

```powershell
code reports\baselines\rule_baseline\report.md
```

7. Review generated metrics before committing. Do not use `test.parquet` in this phase.
