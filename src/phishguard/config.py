"""Central project paths and reproducibility settings."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
BASELINE_REPORTS_DIR = REPORTS_DIR / "baselines"
RULE_BASELINE_REPORT_DIR = BASELINE_REPORTS_DIR / "rule_baseline"
MODEL_REPORTS_DIR = REPORTS_DIR / "models"
TFIDF_LOGISTIC_REPORT_DIR = MODEL_REPORTS_DIR / "tfidf_logistic"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_ARTIFACTS_DIR = ARTIFACTS_DIR / "models"
TFIDF_LOGISTIC_ARTIFACT_DIR = MODEL_ARTIFACTS_DIR / "tfidf_logistic"

RAW_DATA_PATH = RAW_DATA_DIR / "phiusiil_raw.parquet"
RAW_METADATA_PATH = RAW_DATA_DIR / "phiusiil_metadata.json"

PREPARED_DATA_PATH = INTERIM_DATA_DIR / "phiusiil_url_only.parquet"
CONFLICTING_LABELS_PATH = INTERIM_DATA_DIR / "conflicting_labels.parquet"
PREPARATION_SUMMARY_PATH = INTERIM_DATA_DIR / "preparation_summary.json"

TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.parquet"
VALIDATION_DATA_PATH = PROCESSED_DATA_DIR / "validation.parquet"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.parquet"
SPLIT_SUMMARY_PATH = PROCESSED_DATA_DIR / "split_summary.json"

DATA_AUDIT_REPORT_PATH = REPORTS_DIR / "data_audit.md"

RANDOM_STATE = 42
UCI_DATASET_ID = 967


def ensure_directories() -> None:
    """Create all generated-data, report, and artifact directories."""
    for directory in (
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
        BASELINE_REPORTS_DIR,
        RULE_BASELINE_REPORT_DIR,
        MODEL_REPORTS_DIR,
        TFIDF_LOGISTIC_REPORT_DIR,
        ARTIFACTS_DIR,
        MODEL_ARTIFACTS_DIR,
        TFIDF_LOGISTIC_ARTIFACT_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
