"""Central project paths and reproducibility settings."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

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
    """Create all generated-data and report directories."""
    for directory in (
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        REPORTS_DIR,
        FIGURES_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
