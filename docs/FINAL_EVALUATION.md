# PhishGuard Final Evaluation

## Final Model

**Model version:** `tfidf-logistic-v3-rootcanon`  
**Task:** Binary phishing URL classification  
**Frozen threshold:** `0.768113160039295`

The final PhishGuard model uses:

- URL normalization
- character-level TF-IDF features
- Logistic Regression
- five-fold `registered_domain`-grouped sigmoid calibration
- a validation-derived frozen decision threshold

The model was frozen before the final external evaluation and before
locked-test scoring.

No preprocessing, model, calibration, or threshold changes were made
after the final external results were observed.

---

## Evaluation Protocol

The internal dataset was split using `registered_domain` as the grouping
unit rather than using a random row split.

This prevents related URLs from the same registered domain from
appearing across train, validation, and test partitions.

Final split sizes:

| Split | Rows |
|---|---:|
| Train | 141,090 |
| Validation | 47,030 |
| Locked test | 47,030 |

The locked test was not model-scored until:

1. preprocessing was finalized
2. the final model was trained
3. calibration was finalized
4. the operating threshold was frozen
5. robustness testing was completed
6. fresh external/OOD evaluation was completed
7. model and evaluation artifact hashes were recorded

---

## Operating Point

The final threshold was selected on the validation set to match the
false-positive-rate operating point established by the earlier rule
baseline.

Frozen threshold:

```text
0.768113160039295
```

Validation FPR limit:

```text
1.6314%
```

The threshold was not tuned using external data or the locked test.

---

## Validation Performance

Final V3 calibrated validation metrics:

| Metric | Validation |
|---|---:|
| Average Precision | 0.9404 |
| ROC-AUC | 0.9364 |
| Brier score | 0.0872 |
| Log loss | 0.2864 |
| ECE | 0.0304 |

---

## Final Locked-Test Results

The frozen model was evaluated once on the 47,030-row locked test set.

### Confusion Matrix

| | Predicted Phishing | Predicted Legitimate |
|---|---:|---:|
| Actual Phishing | 14,746 | 5,314 |
| Actual Legitimate | 402 | 26,568 |

### Metrics

| Metric | Result |
|---|---:|
| Accuracy | **87.85%** |
| Precision | **97.35%** |
| Recall / TPR | **73.51%** |
| False-positive rate | **1.49%** |
| Specificity | **98.51%** |
| F1 | **0.8377** |
| ROC-AUC | **0.9394** |
| Average Precision | **0.9424** |
| Brier score | **0.085254** |
| Log loss | **0.282593** |
| ECE | **0.025873** |

The final test FPR of **1.49%** remained below the frozen validation
operating limit of **1.63%**.

---

## Validation vs Locked Test

| Metric | Validation | Locked Test |
|---|---:|---:|
| Average Precision | 0.9404 | 0.9424 |
| ROC-AUC | 0.9364 | 0.9394 |
| Brier score | 0.0872 | 0.0853 |
| Log loss | 0.2864 | 0.2826 |
| ECE | 0.0304 | 0.0259 |

The close validation-to-test results suggest that the
registered-domain-grouped internal evaluation was stable.

This does not imply equivalent performance on deployment traffic.

---

## Robustness Results

The final model was tested against controlled URL transformations.

Transformations that should be semantically equivalent were treated as
strict invariance requirements.

| Perturbation | Applicable Rows | Mean Score Drift | Prediction Flip Rate |
|---|---:|---:|---:|
| HTTP/HTTPS scheme | 47,030 | 0.000000 | 0.0000% |
| Leading `www.` | 47,030 | 0.000000 | 0.0000% |
| Hostname case | 46,894 | 0.000000 | 0.0000% |
| Root `/` representation | 41,774 | 0.000000 | 0.0000% |

All strict invariance checks passed.

Non-root trailing-slash and query-order perturbations were retained as
diagnostics because they may change URL semantics.

---

## Why V3 Was Necessary

An earlier external robustness evaluation exposed a representation
defect.

For Tranco root URLs, changing:

```text
https://example.com
```

to:

```text
https://example.com/
```

caused large prediction changes even though both represent the same
HTTP root resource.

The earlier model showed approximately:

```text
64.91% prediction flips
```

for this transformation.

V3 introduced root-only path canonicalization while preserving
potentially meaningful non-root path differences.

After the fix:

```text
Internal root-slash flip rate: 0.0000%
Fresh external flip rate:      0.0000%
Mean probability drift:        0.000000
```

---

## Fresh External/OOD Evaluation

A new external snapshot was collected on **2026-09-03**, after V3 had
been finalized.

The final external set contained:

| Source | Role | Rows |
|---|---|---:|
| PhishTank | phishing | 12,934 |
| Tranco | benign proxy | 12,594 |
| **Total** | | **25,528** |

Development-domain overlap was removed.

External examples sharing a registered domain with the locked internal
test set were also removed before model evaluation.

Only locked-test registered-domain identifiers were used for this
decontamination step. Test URLs, labels, features, predictions, and
metrics were not accessed.

### External Results

| Metric | Result |
|---|---:|
| PhishTank recall / TPR | **83.59%** |
| Tranco benign-proxy FPR | **18.46%** |
| Precision | 82.30% |
| F1 | 0.8294 |
| ROC-AUC | 0.9012 |
| Average Precision | 0.9167 |
| Brier score | 0.158137 |
| Log loss | 0.510190 |
| ECE | 0.136937 |

The model retained useful phishing ranking and recall under the external
distribution, but probability calibration degraded and the Tranco
benign-proxy false-positive rate increased substantially.

---

## Interpreting the Tranco Result

The **18.46% Tranco FPR should not be interpreted as a deployment
false-positive rate**.

Tranco provides popular registered domains and was used here as a
constructed benign stress-test proxy.

It does not represent the natural distribution of benign URLs seen by a
browser, enterprise gateway, email system, or production security
product.

The result instead demonstrates that:

- internal test performance does not fully describe OOD behaviour
- the model experiences meaningful distribution shift
- representative production traffic would be required before deployment
- internal FPR should not be presented as real-world FPR

---

## Dataset Bias Investigation

An earlier model achieved near-perfect internal validation performance.

Feature analysis and ablation testing showed that the source dataset
contained strong protocol-related collection bias.

In particular:

- legitimate URLs were strongly associated with HTTPS
- many phishing URLs were associated with HTTP

A protocol-only heuristic already separated a substantial portion of the
dataset.

Scheme information was therefore neutralized in the final model.

This deliberately reduced headline internal performance in exchange for
a more defensible evaluation.

---

## Final Interpretation

PhishGuard V3 demonstrates strong internal discrimination at the frozen
operating point:

- **97.35% precision**
- **73.51% phishing recall**
- **1.49% internal test FPR**
- **0.9394 ROC-AUC**
- **0.9424 Average Precision**

The model also satisfies the defined representation-invariance
requirements.

However, the external evaluation shows that performance degrades under
distribution shift, particularly for benign-proxy classification and
probability calibration.

Therefore, the final result should be interpreted as:

> A leakage-aware and robustness-tested URL-only phishing classifier
> with stable grouped internal evaluation, but measurable external
> distribution-shift limitations that would require further validation
> before production use.

---

## Reproducibility

**Frozen model version**

```text
tfidf-logistic-v3-rootcanon
```

**Frozen threshold**

```text
0.768113160039295
```

**Freeze Git commit**

```text
9198acad4f81016fba47d57ae74f434880a2d58b
```

**Model SHA-256**

```text
7df6dd6102d37f4d3358db5a7536b48395f1c1f610414e0d9c7c6fcb3463f058
```

**Environment**

```text
Python:       3.11.15
scikit-learn: 1.9.0
```

Detailed provenance is available in:

```text
reports/models/final_v3_freeze_manifest.json
```

Detailed final metrics are available in:

```text
reports/models/final_locked_test_v3/locked_test_metrics.json
```

---

## Final Status

**PhishGuard V3 is frozen.**

The locked test has been consumed.

Any future change to:

- preprocessing
- URL normalization
- TF-IDF configuration
- classifier configuration
- calibration
- operating threshold

must be treated as a new model version and evaluated under a new
development cycle.