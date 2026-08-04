"""Generate a reproducible data-quality audit for the URL-only dataset."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from phishguard.config import (
    CONFLICTING_LABELS_PATH,
    DATA_AUDIT_REPORT_PATH,
    FIGURES_DIR,
    PREPARED_DATA_PATH,
    ensure_directories,
)


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a small dataframe as a Markdown table without extra dependencies."""
    if frame.empty:
        return "_No rows._"

    columns = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _save_histogram(values: Iterable[int], title: str, xlabel: str, output_name: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(list(values), bins=50)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("URL count")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / output_name, dpi=160)
    plt.close(figure)


def _save_class_distribution(frame: pd.DataFrame) -> None:
    counts = frame["target"].value_counts().reindex([0, 1], fill_value=0)
    labels = ["Legitimate (0)", "Phishing (1)"]

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.bar(labels, counts.values)
    axis.set_title("Class distribution")
    axis.set_ylabel("URL count")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "class_distribution.png", dpi=160)
    plt.close(figure)


def _save_domain_frequency(frame: pd.DataFrame) -> None:
    top_domains = frame["registered_domain"].fillna("<invalid>").value_counts().head(20)

    figure, axis = plt.subplots(figsize=(9, 6))
    axis.barh(top_domains.index.astype(str)[::-1], top_domains.values[::-1])
    axis.set_title("Most frequent registered domains")
    axis.set_xlabel("URL count")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "domain_frequency.png", dpi=160)
    plt.close(figure)


def generate_audit() -> str:
    """Create figures and a Markdown audit report from prepared data."""
    ensure_directories()
    if not PREPARED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Prepared dataset not found at {PREPARED_DATA_PATH}. "
            "Run `python -m phishguard.data.prepare` first."
        )

    frame = pd.read_parquet(PREPARED_DATA_PATH)
    conflicts = (
        pd.read_parquet(CONFLICTING_LABELS_PATH)
        if CONFLICTING_LABELS_PATH.exists()
        else pd.DataFrame()
    )

    frame = frame.copy()
    frame["url_length"] = frame["url_model_input"].str.len()
    frame["digit_count"] = frame["url_model_input"].str.count(r"\d")
    frame["symbol_count"] = frame["url_model_input"].str.count(r"[^A-Za-z0-9]")

    _save_class_distribution(frame)
    _save_histogram(
        frame["url_length"],
        title="URL length distribution",
        xlabel="Characters",
        output_name="url_length_distribution.png",
    )
    _save_domain_frequency(frame)

    class_summary = (
        frame.groupby("target")
        .agg(rows=("target", "size"), unique_domains=("split_group", "nunique"))
        .reset_index()
    )
    class_summary["class"] = class_summary["target"].map({0: "legitimate", 1: "phishing"})
    class_summary["percentage"] = (class_summary["rows"] / len(frame) * 100).round(2)
    class_summary = class_summary[["target", "class", "rows", "percentage", "unique_domains"]]

    top_domains = (
        frame.groupby(["target", "registered_domain"], dropna=False)
        .size()
        .rename("rows")
        .reset_index()
        .sort_values(["target", "rows"], ascending=[True, False])
        .groupby("target", as_index=False)
        .head(10)
    )
    top_domains["class"] = top_domains["target"].map({0: "legitimate", 1: "phishing"})
    top_domains = top_domains[["class", "registered_domain", "rows"]]

    numeric_summary = (
        frame[["url_length", "digit_count", "symbol_count"]]
        .describe(percentiles=[0.5, 0.9, 0.95, 0.99])
        .round(2)
        .reset_index()
        .rename(columns={"index": "statistic"})
    )

    report = f"""# PhishGuard Phase 1 Data Audit

## Dataset overview

- Prepared rows: **{len(frame):,}**
- Phishing rows: **{int((frame["target"] == 1).sum()):,}**
- Legitimate rows: **{int((frame["target"] == 0).sum()):,}**
- Missing model inputs: **{int(frame["url_model_input"].isna().sum()):,}**
- Invalid URL rows retained: **{int((~frame["is_valid_url"]).sum()):,}**
- Unique registered-domain split groups: **{frame["split_group"].nunique():,}**
- Conflicting-label rows removed: **{len(conflicts):,}**
- Exact duplicate model inputs remaining: **{int(frame["url_model_input"].duplicated().sum()):,}**
- Normalised duplicate keys remaining: **{int(frame["url_dedupe_key"].duplicated().sum()):,}**

## Target convention

- `1` = phishing
- `0` = legitimate

## Class summary

{_markdown_table(class_summary)}

## URL-shape summary

{_markdown_table(numeric_summary)}

## Most frequent domains by class

{_markdown_table(top_domains)}

## Generated figures

- `reports/figures/class_distribution.png`
- `reports/figures/url_length_distribution.png`
- `reports/figures/domain_frequency.png`

## Safety and scope notes

- No script in Phase 1 opens, renders, or requests any URL.
- The modelling table uses raw URL strings and features derived locally from
  those strings.
- Webpage-source features supplied by the original dataset are intentionally
  excluded from Version 1.
- Raw URLs are not printed in this report, reducing accidental click risk.
"""

    DATA_AUDIT_REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Saved data audit to {DATA_AUDIT_REPORT_PATH}")
    return report


def main() -> None:
    generate_audit()


if __name__ == "__main__":
    main()
