# Phase 2C-D2 — V3 Internal Robustness

## Evaluation policy

- Validation rows: **47,030**
- Matched FPR limit: **1.6314%**
- V3 calibrated operating threshold: **0.768113**
- Model: **tfidf-logistic-v3 + root-path canonicalization + grouped-CV sigmoid**
- Threshold selected using the existing validation split only.
- External/OOD snapshot used: **False**
- Locked test loaded/scored: **False**

## Perturbation results

| Perturbation | Applicable | Mean drift | P95 drift | Max drift | Flip rate | Check |
|---|---:|---:|---:|---:|---:|---|
| scheme_toggle | 47,030 | 0.000000 | 0.000000 | 0.000000 | 0.0000% | PASS |
| www_toggle | 47,030 | 0.000000 | 0.000000 | 0.000000 | 0.0000% | PASS |
| host_case | 46,894 | 0.000000 | 0.000000 | 0.000000 | 0.0000% | PASS |
| root_slash | 41,774 | 0.000000 | 0.000000 | 0.000000 | 0.0000% | PASS |
| non_root_trailing_slash | 5,256 | 0.005136 | 0.025323 | 0.397530 | 0.7420% | diagnostic |
| query_order | 420 | 0.000402 | 0.001123 | 0.028582 | 0.2381% | diagnostic |

## Strict invariance checks

- `scheme_toggle`, `www_toggle`, `host_case`, and `root_slash` are strict V3 invariance probes.
- `root_slash` tests only equivalent root representations such as `example.com` and `example.com/`.
- Strict invariance tolerance: `1e-12`.
- Overall strict invariance status: **PASS**

## Sensitivity diagnostics

- `non_root_trailing_slash` deliberately keeps `/login` and `/login/` distinct.
- `query_order` remains a sensitivity diagnostic rather than a guaranteed semantic-preserving transformation.

## Safety and leakage controls

- The external/OOD snapshot was not loaded.
- The locked test split was not loaded.
- Raw phishing URLs are not written to the example report; SHA-256 hashes are stored instead.

## Generated artifacts

- `robustness_summary.csv`
- `robustness_metrics.json`
- `largest_drift_examples.csv`
- `probability_drift.png`
- `report.md`
