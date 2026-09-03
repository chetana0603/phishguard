"""Phase 2C-C2: remove locked-test domain overlap from external/OOD data."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from phishguard.config import TEST_DATA_PATH
from phishguard.data.external_ood import (
    EXTERNAL_DATA_PATH,
    EXTERNAL_REPORT_DIR,
)

EXTERNAL_EVAL_PATH = Path("data/external/processed") / "external_ood_eval.parquet"

DECONTAMINATION_REPORT_PATH = EXTERNAL_REPORT_DIR / "decontamination_report.json"


def _sha256_file(
    path: Path,
) -> str:
    """Return SHA-256 digest for a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        while True:
            chunk = file_handle.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def _canonical_domain(
    value: object,
) -> str | None:
    """Canonicalize a registered-domain value before hashing."""
    if pd.isna(value):
        return None

    text = str(value).strip().lower().rstrip(".")

    if not text:
        return None

    return text


def domain_sha256(
    value: object,
) -> str | None:
    """Hash a canonical registered domain without persisting raw test domains."""
    domain = _canonical_domain(value)

    if domain is None:
        return None

    return hashlib.sha256(domain.encode("utf-8")).hexdigest()


def _hash_set_fingerprint(
    hashes: set[str],
) -> str:
    """Fingerprint a set of hashes without storing its underlying values."""
    digest = hashlib.sha256()

    for value in sorted(hashes):
        digest.update(value.encode("ascii"))

        digest.update(b"\n")

    return digest.hexdigest()


def load_locked_test_domain_hashes(
    *,
    test_path: Path = TEST_DATA_PATH,
) -> tuple[
    set[str],
    dict[str, int],
]:
    """
    Read only the registered_domain column from the locked test set.

    No test URLs, labels, features, model scores, or predictions are loaded.
    """
    if not test_path.exists():
        raise FileNotFoundError(f"Locked test data not found at {test_path}.")

    # IMPORTANT:
    # Only the registered_domain column is loaded.
    frame = pd.read_parquet(
        test_path,
        columns=["registered_domain"],
    )

    hashes = {value for value in frame["registered_domain"].map(domain_sha256) if value is not None}

    summary = {
        "test_rows": int(len(frame)),
        "unique_test_domain_hashes": int(len(hashes)),
    }

    return hashes, summary


def _count_by_target(
    frame: pd.DataFrame,
) -> dict[str, int]:
    """Return row counts by target label."""
    counts = frame["target"].value_counts().sort_index()

    return {str(int(target)): int(count) for target, count in counts.items()}


def _count_by_source(
    frame: pd.DataFrame,
) -> dict[str, int]:
    """Return row counts by external source."""
    counts = frame["source"].astype(str).value_counts().sort_index()

    return {str(source): int(count) for source, count in counts.items()}


def decontaminate_external_dataset(
    *,
    external_path: Path = EXTERNAL_DATA_PATH,
    test_path: Path = TEST_DATA_PATH,
    output_path: Path = EXTERNAL_EVAL_PATH,
    report_path: Path = DECONTAMINATION_REPORT_PATH,
) -> dict[str, object]:
    """
    Remove external examples whose registered domains occur in the locked test set.

    The locked test set is not scored and its labels/URLs are never loaded.
    """
    if not external_path.exists():
        raise FileNotFoundError(f"External candidate dataset not found at {external_path}.")

    external = pd.read_parquet(external_path)

    required_columns = {
        "url_model_input",
        "target",
        "source",
        "registered_domain",
    }

    missing = required_columns.difference(external.columns)

    if missing:
        raise ValueError(f"External dataset missing required columns: {sorted(missing)}")

    if external.empty:
        raise ValueError("External dataset is empty.")

    invalid_targets = set(external["target"].dropna()).difference(
        {
            0,
            1,
        }
    )

    if invalid_targets:
        raise ValueError(f"External dataset contains invalid targets: {sorted(invalid_targets)}")

    test_domain_hashes, test_summary = load_locked_test_domain_hashes(test_path=test_path)

    external_hashes = external["registered_domain"].map(domain_sha256)

    if external_hashes.isna().any():
        bad_count = int(external_hashes.isna().sum())

        raise ValueError(f"External dataset contains {bad_count} invalid registered-domain values.")

    overlap_mask = external_hashes.isin(test_domain_hashes)

    overlap = external.loc[overlap_mask].copy()

    clean = external.loc[~overlap_mask].copy()

    if clean.empty:
        raise RuntimeError("All external rows overlapped with locked-test domains.")

    clean = clean.reset_index(drop=True)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    clean.to_parquet(
        output_path,
        index=False,
    )

    before_target_counts = _count_by_target(external)

    after_target_counts = _count_by_target(clean)

    overlap_target_counts = _count_by_target(overlap) if not overlap.empty else {}

    overlap_source_counts = _count_by_source(overlap) if not overlap.empty else {}

    after_source_counts = _count_by_source(clean)

    phishing_rows = int((clean["target"] == 1).sum())

    benign_rows = int((clean["target"] == 0).sum())

    phishing_prevalence = phishing_rows / len(clean)

    payload: dict[
        str,
        object,
    ] = {
        "phase": "2C-C2",
        "created_at_utc": (datetime.now(UTC).isoformat()),
        "purpose": (
            "Remove external/OOD registered-domain overlap "
            "with the locked internal test split before "
            "external model evaluation."
        ),
        "input_external_path": str(external_path),
        "input_external_sha256": (_sha256_file(external_path)),
        "output_external_path": str(output_path),
        "output_external_sha256": (_sha256_file(output_path)),
        "locked_test_access": {
            "registered_domain_column_loaded": True,
            "raw_domain_values_persisted": False,
            "url_column_loaded": False,
            "target_column_loaded": False,
            "other_feature_columns_loaded": False,
            "model_loaded": False,
            "predictions_generated": False,
            "metrics_calculated": False,
            "locked_test_scored": False,
        },
        "locked_test_domain_guard": {
            **test_summary,
            "domain_hash_set_fingerprint": (_hash_set_fingerprint(test_domain_hashes)),
        },
        "external_before": {
            "rows": int(len(external)),
            "target_counts": (before_target_counts),
            "source_counts": (_count_by_source(external)),
        },
        "overlap_removed": {
            "rows": int(len(overlap)),
            "target_counts": (overlap_target_counts),
            "source_counts": (overlap_source_counts),
        },
        "external_after": {
            "rows": int(len(clean)),
            "target_counts": (after_target_counts),
            "source_counts": (after_source_counts),
            "phishing_rows": (phishing_rows),
            "benign_proxy_rows": (benign_rows),
            "phishing_prevalence": float(phishing_prevalence),
        },
        "rebalance_after_filtering": False,
        "reason_not_rebalanced": (
            "All clean external examples are retained. "
            "Class-specific recall and false-positive rate "
            "will be reported separately during OOD evaluation."
        ),
        "raw_locked_test_domains_written_to_report": False,
        "locked_test_labels_used": False,
        "locked_test_urls_used": False,
        "locked_test_scored": False,
    }

    report_path.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Phase 2C-C2 locked-test domain contamination guard")

    print("=" * 74)

    print(f"External rows before: {len(external):,}")

    print(f"Locked-test rows inspected for registered_domain only: {test_summary['test_rows']:,}")

    print(f"Unique locked-test domain hashes: {test_summary['unique_test_domain_hashes']:,}")

    print(f"External rows overlapping locked-test domains: {len(overlap):,}")

    print(f"External rows after: {len(clean):,}")

    print(f"Remaining phishing rows: {phishing_rows:,}")

    print(f"Remaining benign-proxy rows: {benign_rows:,}")

    print("Locked-test URLs loaded: False")

    print("Locked-test labels loaded: False")

    print("Locked-test predictions generated: False")

    print("Locked test scored: False")

    return payload


def main() -> None:
    """Run Phase 2C-C2."""
    parser = argparse.ArgumentParser(
        description=("Remove locked-test domain overlap from external/OOD evaluation data.")
    )

    parser.add_argument(
        "--external-path",
        type=Path,
        default=EXTERNAL_DATA_PATH,
    )

    parser.add_argument(
        "--test-path",
        type=Path,
        default=TEST_DATA_PATH,
    )

    parser.add_argument(
        "--output-path",
        type=Path,
        default=EXTERNAL_EVAL_PATH,
    )

    parser.add_argument(
        "--report-path",
        type=Path,
        default=DECONTAMINATION_REPORT_PATH,
    )

    args = parser.parse_args()

    decontaminate_external_dataset(
        external_path=(args.external_path),
        test_path=(args.test_path),
        output_path=(args.output_path),
        report_path=(args.report_path),
    )


if __name__ == "__main__":
    main()
