"""Tests for duplicate and conflict handling."""

import pandas as pd

from phishguard.data.prepare import prepare_dataframe


def test_conflicting_labels_are_removed() -> None:
    raw = pd.DataFrame(
        {
            "URL": [
                "https://Example.com/a#one",
                "https://example.com/a#two",
                "https://safe.example/b",
            ],
            "label": [0, 1, 1],
        }
    )

    prepared, conflicts, summary = prepare_dataframe(raw)

    assert len(conflicts) == 2
    assert len(prepared) == 1
    assert summary["conflicting_dedupe_keys_removed"] == 1


def test_same_label_duplicates_keep_one_row() -> None:
    raw = pd.DataFrame(
        {
            "URL": ["https://example.com/a#one", "https://EXAMPLE.com/a#two"],
            "label": [0, 0],
        }
    )

    prepared, conflicts, summary = prepare_dataframe(raw)

    assert conflicts.empty
    assert len(prepared) == 1
    assert summary["normalised_duplicates_removed"] == 1
