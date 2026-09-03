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
- robustness checks
- external/OOD validation
- reproducible model freezing
- tested inference serving
- containerized local inference

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

The final model was evaluated once on a locked **47,030-URL test set**
after preprocessing, calibration, robustness rules, and the operating
threshold had been frozen.

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
does not guarantee equivalent behaviour on a different URL distribution.

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

## Inference API

The frozen V3 model is served through a FastAPI inference service.

Available endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Service health |
| `/model` | GET | Frozen model metadata |
| `/predict` | POST | Score one URL |
| `/predict/batch` | POST | Score up to 100 URLs |
| `/docs` | GET | Interactive OpenAPI documentation |

The model is loaded once when the API starts and reused across requests.

### Single Prediction

Request:

```json
{
  "url": "https://example.com"
}
```

Example response:

```json
{
  "url": "https://example.com",
  "phishing_probability": 0.5044066778199805,
  "prediction": 0,
  "label": "legitimate",
  "threshold": 0.768113160039295,
  "model_version": "tfidf-logistic-v3-rootcanon"
}
```

### Batch Prediction

Request:

```json
{
  "urls": [
    "https://example.com",
    "https://example.com/",
    "http://www.example.com"
  ]
}
```

The batch endpoint uses the same vectorized model inference path and
accepts up to 100 URLs per request.

Equivalent root representations were verified to produce the same model
score through the containerized API:

```text
https://example.com
https://example.com/
http://www.example.com
```

This confirms that the V3 representation-invariance behaviour is
preserved after model serialization, API serving, and containerization.

---

## API Validation

The inference layer includes automated tests for:

- single-URL predictions
- batch predictions
- frozen threshold behaviour
- model-version metadata
- empty-input rejection
- whitespace-only URL rejection
- missing request fields
- maximum batch size enforcement
- health endpoint behaviour
- model metadata endpoint behaviour

The current automated test suite contains:

```text
92 passing tests
```

The API unit tests are designed to run without opening network sockets.
Actual HTTP inference was validated separately through the locally
running Docker container.

---

## Docker

The FastAPI service is packaged in a Linux Docker container and has been
validated locally.

### Build

The frozen model artifact must exist locally at:

```text
artifacts/models/calibration_v3_rootcanon/sigmoid_grouped_cv.joblib
```

Build the image:

```bash
docker build -t phishguard-api:v3 .
```

### Run

```bash
docker run --rm -p 8000:8000 phishguard-api:v3
```

The locally running API is then available at:

```text
http://127.0.0.1:8000
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### Health Check

The Docker image includes a container health check against:

```text
GET /health
```

A running container can be inspected with:

```bash
docker ps
```

or:

```bash
docker inspect --format='{{.State.Health.Status}}' phishguard-api
```

Expected status:

```text
healthy
```

### Containerized Inference Flow

```text
HTTP request
      │
      ▼
FastAPI
      │
      ▼
Pydantic validation
      │
      ▼
PhishGuardPredictor
      │
      ▼
Frozen V3 model
      │
      ▼
Probability + classification
      │
      ▼
JSON response
```

---

## Model Artifact Handling

The frozen `.joblib` model is intentionally excluded from normal Git
tracking through `.gitignore`.

Local Docker builds explicitly include the required frozen artifact in
the image while excluding unrelated development artifacts.

This keeps model-development artifacts out of the repository while
allowing the tested local container to run with the exact frozen V3
model.

A separate artifact or container-registry strategy would be required for
future cloud deployment. This is not implemented in V3.

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
│   ├── api/
│   │   ├── app.py
│   │   └── schemas.py
│   │
│   ├── inference/
│   │   └── predictor.py
│   │
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
├── artifacts/
│   └── models/
│       └── frozen local model artifacts
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
├── docs/
│   ├── EXPERIMENT_LOG.md
│   ├── MODEL_CARD.md
│   └── FINAL_EVALUATION.md
│
├── Dockerfile
├── .dockerignore
├── pyproject.toml
└── README.md
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

Check formatting:

```powershell
uv run ruff format --check .
```

Format Python code:

```powershell
uv run ruff format .
```

### Run the API without Docker

```powershell
uv run uvicorn phishguard.api.app:app --host 127.0.0.1 --port 8000
```

Depending on the host environment, local socket restrictions may prevent
Uvicorn from binding directly. The service was therefore also validated
inside Docker.

### Run with Docker

```powershell
docker build -t phishguard-api:v3 .
docker run --rm -p 8000:8000 phishguard-api:v3
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

External evaluation shows measurable distribution shift.

During qualitative containerized smoke testing, a legitimate SaaS-style
URL also received a high phishing score. This provides a concrete example
of the type of confident false positive that can occur outside the
development distribution.

The frozen V3 model was not modified or retuned in response to this
observation because the final test set had already been consumed.

The model should therefore be treated as a screening or risk-scoring
component rather than a complete phishing-defense system.

Representative deployment traffic and further OOD validation would be
required before production-facing use.

---

## Current Status

**V3 model development, evaluation, and local inference productization
are complete.**

```text
Model frozen:                  Yes
Internal robustness:           Passed
Fresh external evaluation:     Completed
Locked test evaluated:         Yes
Post-test threshold tuning:    No

Inference wrapper:             Completed
FastAPI service:               Completed
Single prediction endpoint:    Completed
Batch prediction endpoint:     Completed
Automated tests:               92 passed
Docker image build:            Completed
Docker health check:           Passed
Containerized HTTP inference:  Verified locally

Public/cloud deployment:       Not performed
Production deployment:         Not performed
```

The V3 test set is now consumed.

Any future changes to preprocessing, model configuration, calibration,
or threshold will be developed as a new model version.

The frozen V3 inference service has been validated locally through
Docker. Public deployment is intentionally deferred because external
evaluation identified meaningful distribution shift, and representative
deployment traffic would be required before production-facing use.
