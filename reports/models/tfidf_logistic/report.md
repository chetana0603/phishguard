# Phase 2B — Character TF-IDF Logistic Regression

**Model:** `tfidf-logistic-v1`

**Training rows:** 141,090

**Validation rows:** 47,030

The locked test set was not loaded or evaluated.

## Selected configuration

- Candidate: `char25_c10_bal_scheme_www_neutral`
- Character n-grams: 2–5
- Logistic-regression C: 1.0
- Class weight: `balanced`
- Maximum TF-IDF features: 250,000
- Learned features: 250,000

## Validation ranking and probability diagnostics

- Average precision: **0.9455**
- ROC AUC: **0.9407**
- Brier score: **0.0831**
- Expected calibration error: **0.0498**

The raw logistic-regression probabilities are evaluated for calibration but are not yet treated as production-calibrated risk. Formal calibration is a later phase.

## Comparison at the rule baseline false-positive rate

- FPR limit: **1.63%**
- Selected threshold: **0.586886**
- Precision: **97.25%**
- Recall: **77.48%**
- Actual FPR: **1.63%**
- F1: **0.8625**

## Operating profiles selected on validation data

| Profile | Threshold | Precision | Recall | FPR | F1 | Constraints |
|---|---:|---:|---:|---:|---:|---|
| High Security | 0.447661 | 92.45% | 82.11% | 4.99% | 0.8697 | Met |
| Balanced | 0.447661 | 92.45% | 82.11% | 4.99% | 0.8697 | Met |
| Conservative | 0.973179 | 100.00% | 38.41% | 0.00% | 0.5551 | Met |

## Robustness candidate comparison

| Candidate | Scheme neutral | AP | ROC AUC | Recall at matched FPR | Actual FPR |
|---|---|---:|---:|---:|---:|
| `char25_c10_bal_raw` | No | 0.9990 | 0.9987 | 99.64% | 1.55% |
| `char25_c10_bal_scheme_www_neutral` | Yes | 0.9455 | 0.9407 | 77.48% | 1.63% |

The saved pipeline prefers the scheme-and-www-neutral candidate when available.
The raw candidate is retained as an in-dataset performance benchmark.

## Dataset shortcut diagnostic

A protocol-only detector that flags plain HTTP URLs as phishing achieved:

- Precision: **100.00%**
- Recall: **53.29%**
- False-positive rate: **0.00%**

All legitimate URLs in the current dataset use HTTPS. Consequently, protocol and
leading-www patterns are treated as collection-bias indicators rather than reliable
production phishing evidence.

## Runtime

- Final pipeline fit time: 26.621 seconds
- Validation scoring time: 4.724 seconds
- Mean validation latency: 0.1004 milliseconds per URL
- Throughput: 9955.92 URLs per second

## Saved outputs

- `model_comparison.csv`
- `validation_metrics.json`
- `threshold_table.csv`
- `top_character_ngrams.csv`
- `precision_recall_curve.png`
- `roc_curve.png`
- `calibration_curve.png`
- `score_distribution.png`
- `artifacts/models/tfidf_logistic/pipeline.joblib`

## Limitations

- Model selection and threshold selection both use the validation split.
- The test split remains locked until the final model family and calibration are fixed.
- Character n-grams can learn dataset-specific URL patterns and require drift monitoring.
- Global coefficients explain influential substrings but are not causal evidence.
- No URL was visited or downloaded during training or scoring.
