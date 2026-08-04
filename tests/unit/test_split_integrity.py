"""Tests for leakage-safe group splitting."""

from __future__ import annotations

import pandas as pd

from phishguard.data.split import assign_splits, validate_split_integrity


def _values_by_split(frame: pd.DataFrame, column: str) -> dict[str, set[str]]:
    return {
        split: set(frame.loc[frame["split"] == split, column].astype(str))
        for split in ("train", "validation", "test")
    }


def test_no_domain_overlap_between_splits(grouped_binary_frame: pd.DataFrame) -> None:
    split_frame = assign_splits(grouped_binary_frame)
    groups = _values_by_split(split_frame, "split_group")

    assert groups["train"].isdisjoint(groups["validation"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["validation"].isdisjoint(groups["test"])


def test_no_normalised_url_overlap_between_splits(grouped_binary_frame: pd.DataFrame) -> None:
    split_frame = assign_splits(grouped_binary_frame)
    urls = _values_by_split(split_frame, "url_dedupe_key")

    assert urls["train"].isdisjoint(urls["validation"])
    assert urls["train"].isdisjoint(urls["test"])
    assert urls["validation"].isdisjoint(urls["test"])


def test_split_is_deterministic(grouped_binary_frame: pd.DataFrame) -> None:
    first = assign_splits(grouped_binary_frame)
    second = assign_splits(grouped_binary_frame)
    assert first[["url_dedupe_key", "split", "fold"]].equals(
        second[["url_dedupe_key", "split", "fold"]]
    )


def test_targets_are_binary(grouped_binary_frame: pd.DataFrame) -> None:
    split_frame = assign_splits(grouped_binary_frame)
    assert set(split_frame["target"].unique()) == {0, 1}


def test_no_missing_model_inputs(grouped_binary_frame: pd.DataFrame) -> None:
    split_frame = assign_splits(grouped_binary_frame)
    assert split_frame["url_model_input"].notna().all()
    assert split_frame["url_model_input"].str.len().gt(0).all()


def test_class_ratios_are_reasonably_similar(grouped_binary_frame: pd.DataFrame) -> None:
    split_frame = assign_splits(grouped_binary_frame)
    overall_rate = split_frame["target"].mean()
    split_rates = split_frame.groupby("split")["target"].mean()
    assert (split_rates - overall_rate).abs().max() <= 0.20


def test_validation_helper_accepts_valid_split(grouped_binary_frame: pd.DataFrame) -> None:
    split_frame = assign_splits(grouped_binary_frame)
    validate_split_integrity(split_frame)
