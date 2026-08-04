# Phase 2A — Rule Baseline Validation Report

**Model:** `rule-baseline-v1`

**Validation rows:** 47,030

This is a transparent heuristic comparison baseline, not a production detector.
The locked test set was not used.

## Operating profiles selected on validation data

| Profile | Threshold | Precision | Recall | FPR | F1 | Cost / 10,000 | Constraints |
|---|---:|---:|---:|---:|---:|---:|---|
| High Security | 0.01 | 96.71% | 64.50% | 1.63% | 0.7739 | 38040.61 | Met |
| Balanced | 0.01 | 96.71% | 64.50% | 1.63% | 0.7739 | 38040.61 | Met |
| Conservative | 0.07 | 97.29% | 18.45% | 0.38% | 0.3102 | 86998.94 | Met |

## Ranking and probability diagnostics

- Average precision: **0.7745**
- ROC AUC: **0.8120**
- Brier score: **0.3984**
- Expected calibration error: **0.4105**

The rule score is not a calibrated probability. Calibration is evaluated here only to
establish how much improvement later statistical models require.

## Balanced-profile confusion matrix

- True positives: 12,939
- False negatives: 7,121
- True negatives: 26,530
- False positives: 440

## Runtime

- Total scoring time: 3.779 seconds
- Mean latency: 0.0804 milliseconds per URL
- Throughput: 12445.23 URLs per second

## Most frequently triggered rules

| Rule | Trigger count |
|---|---:|
| `not_https` | 10,690 |
| `moderate_digit_ratio` | 1,731 |
| `deep_hostname` | 897 |
| `suspicious_hostname_term` | 706 |
| `suspicious_path_term` | 704 |
| `many_hostname_hyphens` | 582 |
| `high_digit_ratio` | 383 |
| `many_special_characters` | 323 |
| `many_hostname_levels` | 273 |
| `multiple_suspicious_path_terms` | 260 |
| `long_url` | 244 |
| `long_query` | 227 |
| `very_long_url` | 213 |
| `url_shortener` | 166 |
| `ip_hostname` | 136 |

### Profile-selection notes

- **High Security:** Selected from thresholds satisfying FPR <= 5% and precision >= 70%.
- **Balanced:** Selected by minimum simulated cost among thresholds satisfying FPR <= 5%, precision >= 70%, and recall >= 50%.
- **Conservative:** Selected from thresholds satisfying FPR <= 1% and recall >= 10%.

## Generated figures

- `precision_recall_curve.png`
- `score_distribution.png`
- `threshold_tradeoff.png`

## Limitations

- Rules are manually selected and are expected to miss novel phishing patterns.
- Several suspicious patterns also occur in legitimate URLs.
- The additive score is not an estimated phishing probability.
- Thresholds were selected on validation data only; they are not final test results.
- No URL was visited or downloaded during scoring.
