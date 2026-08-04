"""Download or locally load the PhiUSIIL phishing URL dataset."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from ucimlrepo import fetch_ucirepo

from phishguard.config import (
    RAW_DATA_PATH,
    RAW_METADATA_PATH,
    UCI_DATASET_ID,
    ensure_directories,
)

MANUAL_CSV_PATH = RAW_DATA_PATH.parent / "PhiUSIIL_Phishing_URL_Dataset.csv"


def _json_safe(value: Any) -> Any:
    """Convert metadata objects into JSON-serialisable values."""
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}

        if isinstance(value, (list, tuple)):
            return [_json_safe(item) for item in value]

        if hasattr(value, "to_dict"):
            return _json_safe(value.to_dict())

        return str(value)


def _find_column(
    dataframe: pd.DataFrame,
    expected_name: str,
) -> str:
    """Find a dataframe column using a case-insensitive match."""
    matching_columns = {str(column).strip().lower(): str(column) for column in dataframe.columns}

    column = matching_columns.get(expected_name.lower())

    if column is None:
        available = ", ".join(map(str, dataframe.columns))
        raise ValueError(f"Expected column '{expected_name}'. Available columns: {available}")

    return column


def _validate_raw_dataset(raw: pd.DataFrame) -> None:
    """Validate the minimum requirements of the raw dataset."""
    required_columns = {"URL", "label"}
    missing_columns = required_columns.difference(raw.columns)

    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing_columns)}")

    if raw.empty:
        raise ValueError("The dataset is empty.")

    if raw["URL"].isna().any():
        missing_count = int(raw["URL"].isna().sum())
        raise ValueError(f"The dataset contains {missing_count} missing URLs.")

    unique_labels = set(raw["label"].dropna().unique())

    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"Expected binary labels containing only 0 and 1. Found: {sorted(unique_labels)}"
        )


def _load_manual_csv() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the manually downloaded official UCI CSV."""
    print(f"Loading local dataset from {MANUAL_CSV_PATH}")

    source = pd.read_csv(MANUAL_CSV_PATH, low_memory=False)

    url_column = _find_column(source, "URL")
    label_column = _find_column(source, "label")

    raw = source.rename(
        columns={
            url_column: "URL",
            label_column: "label",
        }
    )

    feature_columns = [str(column) for column in raw.columns if column != "label"]

    metadata: dict[str, Any] = {
        "dataset_id": UCI_DATASET_ID,
        "dataset_name": "PhiUSIIL Phishing URL (Website)",
        "download_method": "manual_official_csv",
        "source_file": MANUAL_CSV_PATH.name,
        "rows": int(raw.shape[0]),
        "columns": int(raw.shape[1]),
        "feature_columns": feature_columns,
        "target_column": "label",
        "original_label_convention": {
            "0": "phishing",
            "1": "legitimate",
        },
    }

    return raw, metadata


def _load_using_ucimlrepo() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fetch the dataset through the official ucimlrepo package."""
    print(f"Fetching UCI dataset {UCI_DATASET_ID}")

    dataset = fetch_ucirepo(id=UCI_DATASET_ID)

    features = dataset.data.features.copy()
    targets = dataset.data.targets.copy()

    if "URL" not in features.columns:
        raise ValueError("Expected a 'URL' column in the UCI feature table.")

    if targets.shape[1] != 1:
        raise ValueError(f"Expected one target column, found {targets.shape[1]}.")

    target_column = targets.columns[0]
    targets = targets.rename(columns={target_column: "label"})

    if "label" in features.columns:
        raise ValueError("Feature table unexpectedly contains a 'label' column.")

    raw = pd.concat(
        [
            features.reset_index(drop=True),
            targets.reset_index(drop=True),
        ],
        axis=1,
    )

    metadata: dict[str, Any] = {
        "dataset_id": UCI_DATASET_ID,
        "dataset_name": "PhiUSIIL Phishing URL (Website)",
        "download_method": "ucimlrepo",
        "rows": int(raw.shape[0]),
        "columns": int(raw.shape[1]),
        "feature_columns": [str(column) for column in features.columns],
        "target_column": "label",
        "source_metadata": _json_safe(dataset.metadata),
        "variables": _json_safe(dataset.variables),
    }

    return raw, metadata


def download_dataset() -> pd.DataFrame:
    """Load the dataset and persist it as an unmodified parquet file."""
    ensure_directories()

    if MANUAL_CSV_PATH.exists():
        raw, metadata = _load_manual_csv()
    else:
        try:
            raw, metadata = _load_using_ucimlrepo()
        except (ConnectionError, OSError) as error:
            raise RuntimeError(
                "Could not connect to the UCI repository.\n"
                "Download the official dataset manually and place it at:\n"
                f"{MANUAL_CSV_PATH}"
            ) from error

    _validate_raw_dataset(raw)

    raw.to_parquet(RAW_DATA_PATH, index=False)

    RAW_METADATA_PATH.write_text(
        json.dumps(_json_safe(metadata), indent=2),
        encoding="utf-8",
    )

    print(f"Saved {len(raw):,} rows to {RAW_DATA_PATH}")
    print(f"Saved source metadata to {RAW_METADATA_PATH}")

    return raw


def main() -> None:
    """Run the dataset download or local-loading stage."""
    download_dataset()


if __name__ == "__main__":
    main()
