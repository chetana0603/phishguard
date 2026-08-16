# Phase 2B — Character TF-IDF Logistic Regression

Phase 2B trains a sparse URL-only statistical classifier without opening any URL.

## Data policy

- Fit the vectorizer and classifier on `data/processed/train.parquet` only.
- Use `data/processed/validation.parquet` for candidate and threshold selection.
- Do not load `data/processed/test.parquet`.
- Keep `target = 1` for phishing and `target = 0` for legitimate URLs.

## Default search

The default command evaluates six laptop-friendly candidates. The implementation groups
candidates by vectorizer settings so multiple logistic-regression values can reuse the same
sparse TF-IDF matrices.

```powershell
uv run python -m phishguard.training.tfidf_logistic
```

The optional complete search evaluates 18 configurations:

```powershell
uv run python -m phishguard.training.tfidf_logistic --full-grid
```

## Fair baseline comparison

The report reads the Phase 2A balanced false-positive rate and reports the maximum model
recall attainable without exceeding that rate. This is a fairer comparison than comparing
unrelated default thresholds.

## Calibration status

Logistic regression provides probability-like scores, but Phase 2B only diagnoses calibration.
Formal calibration and reliability analysis belong to the next phase.
