"""Create deterministic, domain-grouped train/validation/test splits."""

from __future__ import annotations

import json
from typing import Final

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from phishguard.config import (
    PREPARED_DATA_PATH,
    RANDOM_STATE,
    SPLIT_SUMMARY_PATH,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
    ensure_directories,
)

_N_SPLITS: Final[int] = 5
_TRAIN_FOLDS: Final[set[int]] = {0, 1, 2}
_VALIDATION_FOLD: Final[int] = 3
_TEST_FOLD: Final[int] = 4


def assign_splits(
    frame: pd.DataFrame,
    *,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Assign 60/20/20 splits while keeping each domain in exactly one split."""
    required = {"target", "split_group", "url_dedupe_key", "url_model_input"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Prepared data is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Cannot split an empty dataframe.")
    if frame["target"].nunique() != 2:
        raise ValueError("Expected exactly two target classes.")
    if frame["split_group"].isna().any():
        raise ValueError("split_group contains missing values.")

    splitter = StratifiedGroupKFold(
        n_splits=_N_SPLITS,
        shuffle=True,
        random_state=random_state,
    )

    fold_ids = np.full(len(frame), fill_value=-1, dtype=np.int8)
    for fold_id, (_, holdout_indices) in enumerate(
        splitter.split(frame, y=frame["target"], groups=frame["split_group"])
    ):
        fold_ids[holdout_indices] = fold_id

    if (fold_ids < 0).any():
        raise RuntimeError("At least one row was not assigned to a cross-validation fold.")

    split_names = np.select(
        [np.isin(fold_ids, list(_TRAIN_FOLDS)), fold_ids == _VALIDATION_FOLD],
        ["train", "validation"],
        default="test",
    )

    result = frame.copy()
    result["fold"] = fold_ids
    result["split"] = split_names

    validate_split_integrity(result)
    return result


def validate_split_integrity(frame: pd.DataFrame) -> None:
    """Raise an error if domains or deduplicated URLs cross split boundaries."""
    expected_splits = {"train", "validation", "test"}
    observed_splits = set(frame["split"].unique())
    if observed_splits != expected_splits:
        raise ValueError(f"Expected splits {expected_splits}, observed {observed_splits}.")

    for key in ("split_group", "url_dedupe_key"):
        overlap_count = frame.groupby(key)["split"].nunique().gt(1).sum()
        if overlap_count:
            raise ValueError(f"Found {overlap_count} {key} values appearing in multiple splits.")


def _split_summary(frame: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {
        "random_state": RANDOM_STATE,
        "method": "StratifiedGroupKFold",
        "n_splits": _N_SPLITS,
        "group_column": "split_group",
        "target_convention": {"0": "legitimate", "1": "phishing"},
        "splits": {},
    }

    split_details: dict[str, object] = {}
    for split_name in ("train", "validation", "test"):
        subset = frame[frame["split"] == split_name]
        split_details[split_name] = {
            "rows": int(len(subset)),
            "percentage_of_total": round(len(subset) / len(frame) * 100, 4),
            "phishing_rows": int((subset["target"] == 1).sum()),
            "legitimate_rows": int((subset["target"] == 0).sum()),
            "phishing_rate": round(float(subset["target"].mean()), 6),
            "unique_groups": int(subset["split_group"].nunique()),
        }
    summary["splits"] = split_details
    return summary


def split_dataset() -> pd.DataFrame:
    """Read prepared data, assign splits, and save locked parquet files."""
    ensure_directories()
    if not PREPARED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found at {PREPARED_DATA_PATH}. "
            "Run `python -m phishguard.data.prepare` first."
        )

    frame = pd.read_parquet(PREPARED_DATA_PATH)
    split_frame = assign_splits(frame)

    train = split_frame[split_frame["split"] == "train"].reset_index(drop=True)
    validation = split_frame[split_frame["split"] == "validation"].reset_index(drop=True)
    test = split_frame[split_frame["split"] == "test"].reset_index(drop=True)

    train.to_parquet(TRAIN_DATA_PATH, index=False)
    validation.to_parquet(VALIDATION_DATA_PATH, index=False)
    test.to_parquet(TEST_DATA_PATH, index=False)

    summary = _split_summary(split_frame)
    SPLIT_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Train: {len(train):,} rows -> {TRAIN_DATA_PATH}")
    print(f"Validation: {len(validation):,} rows -> {VALIDATION_DATA_PATH}")
    print(f"Locked test: {len(test):,} rows -> {TEST_DATA_PATH}")
    print(f"Saved split summary to {SPLIT_SUMMARY_PATH}")
    return split_frame


def main() -> None:
    split_dataset()


if __name__ == "__main__":
    main()
