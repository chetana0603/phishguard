# Phase 2C-B — Robustness Evaluation

## Evaluation policy

- Validation rows: **47,030**
- Phase 2B matched FPR limit: **1.6314%**
- Calibrated operating threshold: **0.699137**
- Model: **grouped-CV sigmoid calibrated TF-IDF Logistic Regression**
- Locked test used: **False**

## Perturbation results

| Perturbation | Applicable | Mean drift | P95 drift | Max drift | Flip rate | Legit→Phish | Phish→Legit |
|---|---:|---:|---:|---:|---:|---:|---:|
| scheme_toggle | 47,030 | 0.000001 | 0.000000 | 0.030082 | 0.0000% | 0 | 0 |
| www_toggle | 47,030 | 0.000001 | 0.000000 | 0.030082 | 0.0000% | 0 | 0 |
| host_case | 46,894 | 0.215997 | 0.666097 | 0.973944 | 12.3598% | 624 | 5,172 |
| trailing_slash | 47,030 | 0.396919 | 0.893625 | 0.985179 | 44.9160% | 20,435 | 689 |
| query_order | 420 | 0.000339 | 0.001166 | 0.014012 | 0.0000% | 0 | 0 |

## Interpretation

- `scheme_toggle` and `www_toggle` are strict invariance checks because Phase 2B explicitly neutralizes those URL components.
- `host_case` is an invariance probe because DNS hostnames are case-insensitive.
- `trailing_slash` and `query_order` are sensitivity probes. They are not guaranteed to preserve semantics for every web application.
- Large probability drift is informative even when the binary decision does not cross the operating threshold.

## Safety and leakage controls

- The locked test split was not loaded.
- Raw phishing URLs are not written to the robustness example report; SHA-256 hashes are stored instead.
- The operating threshold was selected only from the existing validation split.

## Generated artifacts

- `robustness_summary.csv`
- `robustness_metrics.json`
- `largest_drift_examples.csv`
- `probability_drift.png`
- `report.md`
