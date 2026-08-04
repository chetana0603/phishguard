# PhishGuard

[![CI](https://github.com/chetana0603/phishguard/actions/workflows/ci.yml/badge.svg)](https://github.com/chetana0603/phishguard/actions/workflows/ci.yml)

PhishGuard is a production-oriented phishing URL detection project focused on leakage-safe
data preparation, calibrated risk scoring, explainable predictions, rigorous evaluation,
monitoring, CI/CD, and cloud deployment.

> **Current status:** Phase 2A — transparent URL-rule baseline evaluated on the
> domain-held-out validation split. The locked test set remains untouched.

## Why this project exists

Many phishing classifiers report high scores using random row splits, which can place URLs from
the same domain in both training and testing data. PhishGuard is designed to reduce this leakage
risk by grouping examples by registrable domain before model development.

Version 1 is deliberately URL-only. It never opens or requests the URLs in the dataset.

## Phase 1 deliverables

- Download UCI PhiUSIIL dataset 967 through code
- Convert the target so phishing is the positive class
- Normalise URLs conservatively for duplicate detection
- Remove conflicting-label duplicates
- Build public-suffix-aware domain groups
- Generate a reproducible data-quality audit
- Create deterministic 60/20/20 train, validation, and locked-test splits
- Verify split integrity through unit tests and CI

## Repository structure

```text
phishguard/
├── data/                    # Generated locally; excluded from Git
├── reports/                 # Generated audit and figures
├── scripts/                 # Reproducible PowerShell workflows
├── src/phishguard/
│   ├── config.py
│   ├── data/
│   │   ├── download.py
│   │   ├── prepare.py
│   │   ├── audit.py
│   │   └── split.py
│   ├── features/            # Phase 2
│   └── evaluation/          # Phase 2
├── tests/
├── pyproject.toml
└── uv.lock
```

## Dataset

The project uses the **PhiUSIIL Phishing URL (Website)** dataset from the UCI Machine Learning
Repository, dataset ID 967.

The original labels are:

- `1` = legitimate
- `0` = phishing

PhishGuard converts them internally to:

- `1` = phishing
- `0` = legitimate

The original dataset includes both URL-derived and webpage-source-derived variables. Phase 1
keeps the raw download for reproducibility but creates a separate URL-only modelling table.

## Setup

### Requirements

- Python 3.11+
- Git
- uv

Install uv if needed:

```powershell
py -m pip install uv
```

Create and synchronise the environment:

```powershell
uv sync
```

The first successful sync creates or updates `uv.lock`. Commit that lockfile.

## Run Phase 1

Run each step separately:

```powershell
uv run python -m phishguard.data.download
uv run python -m phishguard.data.prepare
uv run python -m phishguard.data.audit
uv run python -m phishguard.data.split
```

Or run the complete workflow:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_phase1.ps1
```

## Generated artefacts

```text
data/raw/phiusiil_raw.parquet
data/raw/phiusiil_metadata.json
data/interim/phiusiil_url_only.parquet
data/interim/conflicting_labels.parquet
data/interim/preparation_summary.json
data/processed/train.parquet
data/processed/validation.parquet
data/processed/test.parquet
data/processed/split_summary.json
reports/data_audit.md
reports/figures/*.png
```

Generated datasets are excluded from Git. The scripts and audit methodology remain reproducible.

## Run Phase 2A

The rule baseline is intentionally simple and explainable. It establishes the comparison point
that statistical models must outperform; its additive score is not a calibrated probability.

```powershell
uv run python -m phishguard.evaluation.rule_baseline
```

Or run validation, tests, linting, and report generation together:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_phase2a.ps1
```

Generated outputs are written to `reports/baselines/rule_baseline/`. The locked test split is not
used during threshold selection.

In the Roadmap section, mark Phase 1 complete and Phase 2A active:

```markdown
- **Phase 1:** ✅ Reproducible data foundation
- **Phase 2A:** 🚧 Transparent rule baseline
- **Phase 2B:** Character-level TF-IDF logistic regression
- **Phase 3:** Engineered URL features and boosted-tree model
- **Phase 4:** Calibration, final threshold selection, and robustness evaluation
- **Phase 5:** FastAPI service and web interface
- **Phase 6:** Docker, model registry, monitoring, and cloud deployment
- **Phase 7:** Optional browser extension```

## Quality checks

```powershell
uv run ruff check .
uv run ruff format --check .
uv run pytest --cov=phishguard --cov-report=term-missing
```

## Data-leakage controls

- Exact and normalised duplicate handling before splitting
- Conflicting labels removed and reported
- Registrable-domain grouping using a public-suffix-aware extractor
- No domain or normalised URL may cross split boundaries
- Locked test split is not used for model selection
- Deterministic random seed (`42`)

## Roadmap

- **Phase 1:** reproducible data foundation
- **Phase 2:** rule baseline and character-level TF-IDF logistic regression
- **Phase 3:** engineered URL features and boosted-tree model
- **Phase 4:** calibration, threshold selection, and robustness evaluation
- **Phase 5:** FastAPI service and web interface
- **Phase 6:** Docker, model registry, monitoring, and cloud deployment
- **Phase 7:** optional browser extension

## Safety

PhishGuard does not visit URLs during data preparation or Version 1 inference. Dataset URLs are
handled as untrusted strings and should not be converted into clickable links.

## Licence

Project code is released under the MIT License. The dataset remains subject to its own UCI/CC BY
4.0 terms and must be cited separately.
