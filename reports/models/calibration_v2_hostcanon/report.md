# Phase 2C-A — Probability Calibration

## Data policy

- Training rows: **141,090**
- Initial calibration comparison model-fit rows: **112,871**
- Initial calibration comparison calibration rows: **28,219**
- Validation rows: **47,030**
- Registered-domain overlap: **0**
- Locked test used: **False**

## Initial calibration comparison

| Method | AP | ROC AUC | Brier | Log loss | ECE |
|---|---:|---:|---:|---:|---:|
| isotonic | 0.9406 | 0.9386 | 0.08133 | 0.27045 | 0.01107 |
| sigmoid | 0.9439 | 0.9390 | 0.08148 | 0.27275 | 0.02075 |
| uncalibrated | 0.9439 | 0.9390 | 0.08511 | 0.28848 | 0.05326 |

## Preferred calibration method

- Method: **sigmoid**
- Status: **preferred pending Phase 2C-B robustness testing**
- Calibration uses five registered-domain-disjoint folds.
- With `ensemble=False`, the final underlying classifier is fitted on the complete training split.

### Grouped-CV sigmoid validation metrics

- Average precision: **0.9455**
- ROC AUC: **0.9407**
- Brier score: **0.08044**
- Log loss: **0.26791**
- ECE: **0.02363**

## Guardrails

- The Phase 2B scheme/www-neutral TF-IDF Logistic Regression model is used.
- Registered domains cannot cross model-fit/calibration or CV fold boundaries.
- The locked test set remains untouched.
- Sigmoid remains provisional until Phase 2C-B robustness evaluation.
