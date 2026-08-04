# README update for Phase 2A

Change the current-status block near the top of the root README to:

```markdown
> **Current status:** Phase 2A — transparent URL-rule baseline evaluated on the
> domain-held-out validation split. The locked test set remains untouched.
```

Add this section after the Phase 1 generated-artifacts section:

````markdown
## Run Phase 2A

The rule baseline is intentionally simple and explainable. It establishes the comparison point
that statistical models must outperform; its additive score is not a calibrated probability.

```powershell
uv run python -m phishguard.evaluation.rule_baseline
```

Or run validation, tests, linting, and report generation together:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_phase2a.ps1
```

Generated outputs are written to `reports/baselines/rule_baseline/`. The locked test split is not
used during threshold selection.
````

In the Roadmap section, mark Phase 1 complete and Phase 2A active:

```markdown
- **Phase 1:** ✅ Reproducible data foundation
- **Phase 2A:** 🚧 Transparent rule baseline
- **Phase 2B:** Character-level TF-IDF logistic regression
- **Phase 3:** Engineered URL features and boosted-tree model
- **Phase 4:** Calibration, final threshold selection, and robustness evaluation
- **Phase 5:** FastAPI service and web interface
- **Phase 6:** Docker, model registry, monitoring, and cloud deployment
- **Phase 7:** Optional browser extension
```
