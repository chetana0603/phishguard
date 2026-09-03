# Phase 2C-B — Robustness Evaluation

## Evaluation policy

- Validation rows: **47,030**
- Phase 2B matched FPR limit: **1.6314%**
- Calibrated operating threshold: **0.697774**
- Model: **grouped-CV sigmoid calibrated TF-IDF Logistic Regression**
- Locked test used: **False**

## Perturbation results

| Perturbation | Applicable | Mean drift | P95 drift | Max drift | Flip rate | Legit→Phish | Phish→Legit |
|---|---:|---:|---:|---:|---:|---:|---:|
| scheme_toggle | 47,030 | 0.000000 | 0.000000 | 0.000000 | 0.0000% | 0 | 0 |
| www_toggle | 47,030 | 0.000000 | 0.000000 | 0.000000 | 0.0000% | 0 | 0 |
| host_case | 46,894 | 0.000000 | 0.000000 | 0.000000 | 0.0000% | 0 | 0 |
| trailing_slash | 47,030 | 0.396542 | 0.893665 | 0.984971 | 44.8862% | 20,428 | 682 |
| query_order | 420 | 0.000341 | 0.001157 | 0.013949 | 0.0000% | 0 | 0 |

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
