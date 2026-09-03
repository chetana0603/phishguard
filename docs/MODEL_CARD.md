# PhishGuard Model Card

## Model Overview

**Model name:** PhishGuard V3  
**Model version:** `tfidf-logistic-v3-rootcanon`  
**Task:** Binary phishing URL classification  
**Positive class:** Phishing (`1`)  
**Negative class:** Legitimate (`0`)

PhishGuard V3 is a lightweight URL-only phishing classifier built using
character-level TF-IDF features and Logistic Regression.

The model was developed with an emphasis on:

- leakage-aware dataset splitting
- interpretable feature analysis
- probability calibration
- threshold-based operating-point selection
- URL representation robustness
- external/OOD evaluation
- reproducible final-model freezing

The model does not fetch webpages or inspect webpage content. Prediction
is based only on the supplied URL string.

---

## Intended Use

PhishGuard is intended as:

- a phishing-URL risk scoring component
- a lightweight first-stage screening model
- an example of leakage-aware ML evaluation
- a research/portfolio implementation for phishing detection
- a candidate component in a larger security pipeline

The output may be used as one signal among several when assessing a URL.

The model should not be treated as a complete security decision system.

---

## Out-of-Scope and Non-Intended Use

PhishGuard V3 should not be used as the sole basis for:

- blocking users or domains
- declaring a domain malicious
- incident-response decisions
- financial or account-security actions
- production security enforcement without further validation

The model does not inspect:

- webpage HTML
- JavaScript
- TLS certificates
- redirects
- page screenshots
- WHOIS information
- DNS history
- domain age
- network reputation
- email context
- user behaviour

A sophisticated phishing URL may therefore receive a low phishing score,
while an unusual legitimate URL may receive a high score.

---

## Model Architecture

The final classifier uses:

1. URL normalization
2. character-level TF-IDF
3. Logistic Regression
4. five-fold grouped sigmoid calibration
5. a frozen decision threshold

The final model uses the previously selected character TF-IDF
configuration with character n-grams and balanced Logistic Regression.

The classifier configuration was not changed after the external/OOD
evaluation.

---

## URL Normalization

Earlier experiments showed that URL formatting could produce artificial
model shortcuts.

The final V3 normalizer therefore canonicalizes representation elements
that should not influence phishing classification.

### Canonicalized

- HTTP/HTTPS scheme
- leading `www.`
- hostname letter case
- empty root path and `/`

For example:

```text
https://Example.com
http://www.example.com/
example.com
```

are normalized consistently with respect to scheme, hostname, `www.`,
and the root-path representation.

### Preserved

Potentially meaningful components remain distinct.

For example:

```text
example.com/login
example.com/login/
```

are not forced to the same representation.

Path, query, and fragment case are also preserved.

---

## Training Data

The primary development dataset was the UCI PhiUSIIL phishing URL
dataset.

The project uses a URL-only subset of the available information.

Internal target encoding:

- `1` = phishing
- `0` = legitimate

The original source label convention was converted during data
preparation.

Data preparation included:

- URL normalization
- invalid-value handling
- duplicate removal
- conflicting-label removal
- registered-domain extraction
- registered-domain-aware splitting

---

## Leakage Prevention

Random row splitting can allow highly related URLs from the same domain
to appear in both training and evaluation data.

PhishGuard instead uses `registered_domain` as the grouping unit.

The dataset was separated into approximately:

- 60% training
- 20% validation
- 20% locked test

Registered-domain overlap between these partitions was prevented.

The final model used:

- 141,090 training rows
- 47,030 validation rows
- 47,030 locked-test rows

The locked test was not model-scored until the model, calibration,
normalization, and operating threshold had been frozen.

---

## Collection-Bias Investigation

An early character TF-IDF model produced near-perfect validation
performance.

Feature inspection and ablation testing showed that this result was
partly driven by a protocol-format collection bias:

- legitimate examples were strongly associated with HTTPS
- many phishing examples used HTTP

A simple protocol-only heuristic already separated a large portion of
the dataset.

This demonstrated that the near-perfect model was exploiting a dataset
collection shortcut rather than only learning general phishing
characteristics.

Scheme information was therefore neutralized before final model
selection.

This reduced internal performance but produced a more defensible model.

---

## Calibration

Probability calibration was evaluated using:

- uncalibrated predictions
- sigmoid calibration
- isotonic calibration

The final model uses sigmoid calibration with five-fold
`StratifiedGroupKFold` splitting based on `registered_domain`.

`ensemble=False` was used for the final calibrated estimator.

Final validation probability-quality metrics:

| Metric | Validation |
|---|---:|
| Average Precision | 0.940383 |
| ROC-AUC | 0.936372 |
| Brier score | 0.087175 |
| Log loss | 0.286352 |
| ECE | 0.030398 |

---

## Operating Threshold

The final decision threshold is:

```text
0.768113160039295
```

The threshold was selected using the validation set to operate at the
previous rule-baseline false-positive-rate limit.

Validation FPR limit:

```text
1.6314%
```

The threshold was frozen before the final external evaluation and
locked-test scoring.

It was not retuned using either dataset.

---

## Robustness Evaluation

The model was evaluated using controlled URL perturbations.

### Strict Invariance Tests

The following transformations were expected not to alter predictions:

- HTTP/HTTPS scheme toggle
- leading `www.` toggle
- hostname case change
- equivalent root-path `/` representation

Final V3 results:

| Perturbation | Applicable Rows | Mean Drift | Max Drift | Flip Rate |
|---|---:|---:|---:|---:|
| Scheme toggle | 47,030 | 0.000000 | 0.000000 | 0.0000% |
| `www.` toggle | 47,030 | 0.000000 | 0.000000 | 0.0000% |
| Host case | 46,894 | 0.000000 | 0.000000 | 0.0000% |
| Root slash | 41,774 | 0.000000 | 0.000000 | 0.0000% |

All strict invariance tests passed.

### Sensitivity Diagnostics

Non-root trailing-slash changes and query-order changes were retained as
diagnostics rather than strict invariance requirements.

| Perturbation | Applicable Rows | Mean Drift | P95 Drift | Flip Rate |
|---|---:|---:|---:|---:|
| Non-root trailing slash | 5,256 | 0.005136 | 0.025323 | 0.7420% |
| Query order | 420 | 0.000402 | 0.001123 | 0.2381% |

---

## External/OOD Evaluation

A fresh external evaluation dataset was constructed on 2026-09-03
after V3 was finalized.

Sources:

- PhishTank verified-online phishing URLs
- Tranco popular domains as a benign proxy

Development-domain overlap was removed.

External examples overlapping the locked-test registered domains were
also removed before model evaluation.

The final external set contained:

- 12,934 PhishTank phishing URLs
- 12,594 Tranco benign-proxy URLs
- 25,528 total examples

The frozen model was evaluated once.

### External Results

| Metric | Result |
|---|---:|
| PhishTank recall / TPR | 83.59% |
| Tranco benign-proxy FPR | 18.46% |
| Precision | 82.30% |
| F1 | 0.8294 |
| ROC-AUC | 0.9012 |
| Average Precision | 0.9167 |
| Brier score | 0.158137 |
| Log loss | 0.510190 |
| ECE | 0.136937 |

The root-path robustness fix also generalized to the fresh external
data:

- baseline Tranco FPR: 18.4612%
- root-slash Tranco FPR: 18.4612%
- prediction flip rate: 0.0000%
- mean probability drift: 0.000000

---

## Interpretation of External Results

The external results demonstrate meaningful distribution shift.

In particular:

- ranking performance remained useful
- phishing recall remained substantial
- calibration degraded
- the Tranco benign-proxy false-positive rate was much higher than the
  internal false-positive rate

The Tranco value should **not** be interpreted as a real-world deployment
false-positive rate.

Tranco contains popular registered domains and is used here as a
constructed benign proxy. It does not represent the full distribution
of legitimate URLs encountered during normal browsing.

However, the result is evidence that the internal test distribution is
not sufficient to estimate deployment behaviour.

Production use would require evaluation on representative benign
traffic from the intended environment.

---

## Final Locked-Test Evaluation

The final frozen model was evaluated once on the locked internal test
set.

Test rows:

```text
47,030
```

### Confusion Matrix

| | Predicted Phishing | Predicted Legitimate |
|---|---:|---:|
| Actual Phishing | 14,746 | 5,314 |
| Actual Legitimate | 402 | 26,568 |

### Final Results

| Metric | Result |
|---|---:|
| Accuracy | 87.85% |
| Precision | 97.35% |
| Recall / TPR | 73.51% |
| False-positive rate | 1.49% |
| Specificity | 98.51% |
| F1 | 0.8377 |
| ROC-AUC | 0.9394 |
| Average Precision | 0.9424 |
| Brier score | 0.085254 |
| Log loss | 0.282593 |
| ECE | 0.025873 |

The test false-positive rate remained below the validation operating
limit of 1.63%.

Test ranking and calibration metrics were also close to validation
performance.

---

## Validation vs Locked Test

| Metric | Validation | Locked Test |
|---|---:|---:|
| Average Precision | 0.9404 | 0.9424 |
| ROC-AUC | 0.9364 | 0.9394 |
| Brier score | 0.0872 | 0.0853 |
| Log loss | 0.2864 | 0.2826 |
| ECE | 0.0304 | 0.0259 |

The similarity between validation and test performance supports the use
of registered-domain-grouped evaluation for this dataset.

It does not imply equivalent performance on unseen deployment
distributions.

---

## Known Limitations

### Dataset Collection Bias

Protocol-based collection bias was identified in the source dataset.

Although scheme information was neutralized, additional hidden
collection biases may remain.

### Distribution Shift

External evaluation produced significantly higher false-positive rates
and worse probability calibration than internal evaluation.

### URL-Only Model

The model does not use webpage or network information and cannot detect
many contextual phishing indicators.

### Benign External Proxy

Tranco is useful for stress testing but is not representative benign
production traffic.

### Concept Drift

Phishing strategies change over time. The model may degrade as attacker
behaviour, domains, URL structures, and infrastructure evolve.

### Threshold Dependence

The reported precision, recall, and FPR depend on the frozen operating
threshold.

Different deployment requirements may require a different operating
point, which should be validated as part of a new development and
evaluation cycle.

---

## Ethical and Security Considerations

False negatives may allow phishing URLs to pass through a screening
system.

False positives may incorrectly flag legitimate URLs.

For this reason, PhishGuard should be used as a supporting signal rather
than an autonomous security authority.

Phishing URLs used during evaluation should be treated as untrusted
strings and should not be opened or visited during analysis.

Stored reports avoid exposing clickable phishing URLs where practical.

---

## Reproducibility

Frozen model version:

```text
tfidf-logistic-v3-rootcanon
```

Frozen threshold:

```text
0.768113160039295
```

Freeze Git commit:

```text
9198acad4f81016fba47d57ae74f434880a2d58b
```

Frozen model SHA-256:

```text
7df6dd6102d37f4d3358db5a7536b48395f1c1f610414e0d9c7c6fcb3463f058
```

Environment at freeze:

```text
Python:       3.11.15
scikit-learn: 1.9.0
```

Additional provenance is stored in:

```text
reports/models/final_v3_freeze_manifest.json
```

---

## Model Status

**Status:** Frozen Phase 2 model

The locked test has been consumed.

No further changes to:

- model configuration
- URL normalization
- calibration
- operating threshold

should be reported as results for V3.

Any future model improvement should create a new model version and begin
a new evaluation cycle.