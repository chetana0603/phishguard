"""Shared pytest fixtures."""

from __future__ import annotations

import pandas as pd
import pytest


@pytest.fixture
def grouped_binary_frame() -> pd.DataFrame:
    """Return a balanced synthetic dataset with multiple rows per domain group."""
    rows: list[dict[str, object]] = []
    for group_index in range(30):
        target = group_index % 2
        domain = f"group-{group_index}.example"
        for row_index in range(3):
            url = f"https://{domain}/path/{row_index}?value={group_index}"
            rows.append(
                {
                    "url_original": url,
                    "url_model_input": url,
                    "url_dedupe_key": url,
                    "registered_domain": domain,
                    "is_valid_url": True,
                    "split_group": domain,
                    "target": target,
                }
            )
    return pd.DataFrame(rows)
