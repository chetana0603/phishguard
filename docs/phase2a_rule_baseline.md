# Phase 2A — Transparent Rule Baseline

## Purpose

This phase creates a deliberately simple, explainable comparison system before statistical
models are trained. It uses only the URL string and never visits the destination.

The rule score is **not** treated as a calibrated probability and is **not** a production security
control. Its purpose is to establish a measurable baseline that later models must outperform.

## Run

Ensure Phase 1 produced `data/processed/validation.parquet`, then run:

```powershell
uv run python -m phishguard.evaluation.rule_baseline
```

Or execute the complete quality and evaluation workflow:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_phase2a.ps1
```

## Outputs

```text
reports/baselines/rule_baseline/
├── report.md
├── metrics.json
├── threshold_table.csv
├── rule_trigger_counts.csv
├── precision_recall_curve.png
├── score_distribution.png
└── threshold_tradeoff.png
```

No raw URLs are written to reports.

## Validation-only threshold profiles

- **High security:** maximise recall while keeping validation false-positive rate at or below 10%.
- **Balanced:** minimise simulated cost with false negative cost 100 and false positive cost 2.
- **Conservative:** maximise precision while keeping validation false-positive rate at or below 2%.

The locked test set remains untouched.

## Completion criteria

- All tests and Ruff checks pass.
- Validation report and figures are generated.
- The report explicitly states that the baseline is uncalibrated and validation-only.
- No URL is opened or requested.
- Results are committed on `phase2/rule-baseline`, reviewed, and merged into `main`.
