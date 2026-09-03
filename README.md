# PhishGuard

Leakage-aware and robustness-tested phishing URL detection using
character-level TF-IDF and Logistic Regression.

PhishGuard was built to explore a problem that is easy to underestimate:
a phishing classifier can achieve extremely high held-out accuracy while
still learning dataset-specific shortcuts or behaving unpredictably on
equivalent URL representations.

The project therefore focuses not only on model performance, but also on:

- leakage-aware evaluation
- dataset-bias discovery
- probability calibration
- operating-threshold selection
- adversarial-style robustness checks
- external/OOD validation
- reproducible model freezing

---

## Final Model

**Version:** `tfidf-logistic-v3-rootcanon`

The final pipeline uses:

```text
URL
 │
 ▼
URL normalization
 │
 ├── scheme neutralization
 ├── leading www. neutralization
 ├── hostname lowercase normalization
 └── equivalent root-path canonicalization
 │
 ▼
Character TF-IDF
 │
 ▼
Logistic Regression
 │
 ▼
5-fold domain-grouped sigmoid calibration
 │
 ▼
Frozen threshold
0.768113160039295
 │
 ▼
Phishing probability + classification
```

The classifier is URL-only. It does not fetch or visit webpages.

---

## Key Results

### Final Locked Test

The final model was evaluated once on a locked
**47,030-URL test set** after preprocessing, calibration, robustness
rules, and the operating threshold had been frozen.

| Metric | Result |
|---|---:|
| Precision | **97.35%** |
| Recall / TPR | **73.51%** |
| False-positive rate | **1.49%** |
| F1 | **0.8377** |
| ROC-AUC | **0.9394** |
| Average Precision | **0.9424** |
| Accuracy | **87.85%** |
| ECE | **0.0259** |

Confusion matrix:

| | Predicted Phishing | Predicted Legitimate |
|---|---:|---:|
| Actual Phishing | 14,746 | 5,314 |
| Actual Legitimate | 402 | 26,568 |

The final test FPR remained below the validation operating limit of
**1.63%**.

---

## Why the Project Is More Than a Classifier

An early character TF-IDF model achieved almost perfect validation
performance.

Instead of treating that as the final result, feature inspection and
ablation testing were used to determine what the model had actually
learned.

A strong shortcut was discovered:

```text
Legitimate URLs → strongly associated with HTTPS
Phishing URLs   → frequently associated with HTTP
```

A protocol-only heuristic could already classify a substantial portion
of the dataset.

This indicated **dataset collection bias**, rather than purely useful
phishing behaviour.

The final model therefore neutralizes scheme information even though
doing so reduces headline accuracy.

---

## Leakage-Aware Evaluation

Random URL-level splitting can put highly related URLs from the same
domain into both training and evaluation sets.

PhishGuard instead groups examples by:

```text
registered_domain
```

before splitting.

Final dataset sizes:

| Split | Rows |
|---|---:|
| Train | 141,090 |
| Validation | 47,030 |
| Locked test | 47,030 |

Registered-domain overlap between train, validation, and test partitions
is prevented.

---

## Robustness Testing

The model is tested against controlled transformations of the same URL.

Strict invariance is required when the transformation should not change
the meaning of the URL.

| Transformation | Prediction Flip Rate |
|---|---:|
| HTTP ↔ HTTPS | **0.0000%** |
| leading `www.` | **0.0000%** |
| hostname case | **0.0000%** |
| equivalent root `/` | **0.0000%** |

These checks were not added only as unit tests.

They were motivated by real model failures discovered during the
experiments.

---

## A Robustness Failure That Changed the Model

During an earlier external evaluation, the model behaved very
differently for:

```text
https://example.com
```

and:

```text
https://example.com/
```

even though they represent the same HTTP root resource.

The earlier model produced approximately:

```text
64.91% prediction flips
```

for this transformation on the external benign proxy.

V3 introduced **root-only path canonicalization**.

The fresh external evaluation then showed:

```text
Prediction flip rate: 0.0000%
Mean score drift:     0.000000
```

Non-root paths remain distinct:

```text
example.com/login
!=
example.com/login/
```

because those resources are not guaranteed to be equivalent.

---

## External / OOD Evaluation

After V3 was finalized, a fresh external snapshot was collected on
**2026-09-03**.

The evaluation used:

- **12,934** verified-online PhishTank URLs
- **12,594** Tranco popular-domain benign proxies
- **25,528** examples total

Development-domain overlap and locked-test-domain overlap were removed
before model scoring.

### External Results

| Metric | Result |
|---|---:|
| PhishTank recall | **83.59%** |
| Tranco benign-proxy FPR | **18.46%** |
| ROC-AUC | **0.9012** |
| Average Precision | **0.9167** |
| ECE | **0.1369** |

The model retained useful ranking and phishing recall, but calibration
and benign-proxy performance degraded under distribution shift.

### Important Interpretation

The **18.46% Tranco result is not a deployment FPR**.

Tranco popular domains are used as a benign stress-test proxy and are
not representative of normal browsing traffic.

The result instead demonstrates that strong internal test performance
does not guarantee equivalent behaviour on a different URL
distribution.

---

## Calibration

The final model uses five-fold sigmoid calibration with
`registered_domain` grouping.

Validation probability metrics:

| Metric | Result |
|---|---:|
| Brier score | 0.0872 |
| Log loss | 0.2864 |
| ECE | 0.0304 |

Locked-test values remained similar:

| Metric | Result |
|---|---:|
| Brier score | 0.0853 |
| Log loss | 0.2826 |
| ECE | 0.0259 |

External ECE increased to **0.1369**, providing additional evidence of
distribution shift.

---

## Evaluation Methodology

The final evaluation sequence was:

```text
Dataset preparation
        │
        ▼
Registered-domain grouped split
        │
        ▼
Rule baseline
        │
        ▼
TF-IDF + Logistic Regression
        │
        ▼
Feature / bias analysis
        │
        ▼
Scheme-neutral model
        │
        ▼
Grouped probability calibration
        │
        ▼
Robustness testing
        │
        ▼
Representation defects discovered
        │
        ▼
V3 root canonicalization
        │
        ▼
Internal robustness PASS
        │
        ▼
Fresh external/OOD evaluation
        │
        ▼
Freeze manifest + artifact hashes
        │
        ▼
Single locked-test evaluation
```

The locked test was not used for model selection or threshold tuning.

---

## Reproducibility

Frozen model:

```text
tfidf-logistic-v3-rootcanon
```

Frozen decision threshold:

```text
0.768113160039295
```

Model SHA-256:

```text
7df6dd6102d37f4d3358db5a7536b48395f1c1f610414e0d9c7c6fcb3463f058
```

Freeze commit:

```text
9198acad4f81016fba47d57ae74f434880a2d58b
```

Freeze environment:

```text
Python       3.11.15
scikit-learn 1.9.0
```

Detailed freeze metadata is stored in:

```text
reports/models/final_v3_freeze_manifest.json
```

---

## Project Structure

```text
phishguard/
│
├── src/phishguard/
│   ├── data/
│   │   ├── preparation
│   │   ├── external OOD construction
│   │   └── external decontamination
│   │
│   ├── models/
│   │   └── TF-IDF Logistic Regression
│   │
│   ├── training/
│   │   └── calibration
│   │
│   └── evaluation/
│       ├── metrics
│       ├── rule baseline
│       ├── robustness
│       └── external OOD evaluation
│
├── scripts/
│   ├── external snapshot preparation
│   ├── freeze-manifest generation
│   └── final locked-test evaluation
│
├── reports/
│   └── evaluation artifacts and metrics
│
├── tests/
│   └── automated unit/integration tests
│
└── docs/
    ├── EXPERIMENT_LOG.md
    ├── MODEL_CARD.md
    └── FINAL_EVALUATION.md
```

---

## Documentation

For detailed methodology and results:

- [Experiment Log](docs/EXPERIMENT_LOG.md)
- [Model Card](docs/MODEL_CARD.md)
- [Final Evaluation](docs/FINAL_EVALUATION.md)

The experiment log contains the complete development history, including
failed assumptions and model changes.

The model card describes intended use, limitations, bias, robustness,
and evaluation methodology.

The final evaluation document provides a concise summary of the frozen
V3 results.

---

## Development Setup

This project uses Python and `uv`.

Install dependencies:

```powershell
uv sync
```

Run the test suite:

```powershell
uv run pytest -q
```

Run lint checks:

```powershell
uv run ruff check .
```

Format Python code:

```powershell
uv run ruff format .
```

---

## Data

The project uses the UCI PhiUSIIL phishing URL dataset for model
development.

External robustness evaluation uses snapshots derived from:

- PhishTank
- Tranco

Raw external phishing feeds are not intended to be committed to the
repository.

Phishing URLs should always be treated as untrusted strings and should
not be opened during evaluation.

---

## Limitations

PhishGuard is deliberately a **URL-only model**.

It does not use:

- webpage content
- redirects
- DNS history
- WHOIS information
- TLS metadata
- domain reputation
- screenshots
- user context
- email context

External evaluation also shows measurable distribution shift.

The model should therefore be treated as a screening/risk-scoring
component rather than a complete phishing-defense system.

Representative deployment traffic would be required before production
use.

---

## Current Status

**V3 model development and evaluation are complete.**

```text
Model frozen:              Yes
Internal robustness:       Passed
Fresh external evaluation: Completed
Locked test evaluated:     Yes
Post-test threshold tuning: No
```

The V3 test set is now consumed.

Any future changes to preprocessing, model configuration, calibration,
or threshold will be developed as a new model version.