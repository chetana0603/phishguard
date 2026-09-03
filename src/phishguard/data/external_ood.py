"""Phase 2C-C1: prepare external/OOD URL evaluation data."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

import pandas as pd
import tldextract

from phishguard.config import RANDOM_STATE, TRAIN_DATA_PATH, VALIDATION_DATA_PATH

PHISHTANK_BASE_URL = "http://data.phishtank.com/data"
TRANCO_LATEST_ID_URL = "https://tranco-list.eu/top-1m-id"
TRANCO_DOWNLOAD_URL = "https://tranco-list.eu/download/{list_id}/1000000"

EXTERNAL_RAW_DIR = Path("data/external/raw")
EXTERNAL_PROCESSED_DIR = Path("data/external/processed")
EXTERNAL_REPORT_DIR = Path("reports/external_ood")

EXTERNAL_DATA_PATH = EXTERNAL_PROCESSED_DIR / "external_ood_candidate.parquet"

DEFAULT_MAX_PER_CLASS = 30_000

_LIST_ID_RE = re.compile(r"^[A-Za-z0-9]+$")

# Disable live PSL fetching so registered-domain extraction
# uses tldextract's bundled snapshot and remains reproducible.
_TLD_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())


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


def _download_file(
    *,
    url: str,
    output_path: Path,
    user_agent: str,
    timeout_seconds: int = 120,
) -> dict[str, object]:
    """Download a source snapshot and return provenance metadata."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    request = Request(
        url,
        headers={
            "User-Agent": user_agent,
        },
    )

    digest = hashlib.sha256()
    byte_count = 0

    with urlopen(
        request,
        timeout=timeout_seconds,
    ) as response:
        status = getattr(
            response,
            "status",
            200,
        )

        last_modified = response.headers.get("Last-Modified")

        etag = response.headers.get("ETag")

        with output_path.open("wb") as file_handle:
            while True:
                chunk = response.read(1024 * 1024)

                if not chunk:
                    break

                file_handle.write(chunk)

                digest.update(chunk)

                byte_count += len(chunk)

    return {
        "url": url,
        "local_path": str(output_path),
        "http_status": int(status),
        "bytes": int(byte_count),
        "sha256": digest.hexdigest(),
        "last_modified": last_modified,
        "etag": etag,
    }


def _fetch_text(
    *,
    url: str,
    user_agent: str,
    timeout_seconds: int = 60,
) -> str:
    """Fetch a small text resource."""
    request = Request(
        url,
        headers={
            "User-Agent": user_agent,
        },
    )

    with urlopen(
        request,
        timeout=timeout_seconds,
    ) as response:
        content = response.read()

    return content.decode("utf-8").strip()


def _phishtank_download_url() -> str:
    """Build PhishTank feed URL with optional application key."""
    application_key = os.getenv(
        "PHISHTANK_APP_KEY",
        "",
    ).strip()

    if application_key:
        return f"{PHISHTANK_BASE_URL}/{application_key}/online-valid.csv"

    return f"{PHISHTANK_BASE_URL}/online-valid.csv"


def registered_domain(
    value: object,
) -> str | None:
    """Extract a reproducible registered domain from a URL or domain."""
    text = str(value).strip()

    if not text:
        return None

    try:
        parsed = urlsplit(text if "://" in text else f"//{text}")
    except ValueError:
        return None

    hostname = parsed.hostname

    if not hostname:
        return None

    hostname = hostname.strip().lower().rstrip(".")

    if not hostname:
        return None

    try:
        ipaddress.ip_address(hostname)

        return hostname

    except ValueError:
        pass

    extracted = _TLD_EXTRACTOR(hostname)

    domain = extracted.top_domain_under_public_suffix

    if domain:
        return domain.lower()

    # Keep unusual/internal hostnames instead of silently
    # converting them to empty strings.
    return hostname


def _development_domains(
    *,
    train_path: Path,
    validation_path: Path,
) -> set[str]:
    """Load train/validation registered domains for OOD decontamination."""
    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found at {train_path}.")

    if not validation_path.exists():
        raise FileNotFoundError(f"Validation data not found at {validation_path}.")

    train_domains = pd.read_parquet(
        train_path,
        columns=["registered_domain"],
    )

    validation_domains = pd.read_parquet(
        validation_path,
        columns=["registered_domain"],
    )

    combined = pd.concat(
        [
            train_domains,
            validation_domains,
        ],
        ignore_index=True,
    )

    return {
        str(value).strip().lower()
        for value in combined["registered_domain"].dropna()
        if str(value).strip()
    }


def _prepare_phishtank(
    *,
    path: Path,
    development_domains: set[str],
    max_rows: int,
) -> tuple[
    pd.DataFrame,
    dict[str, int],
]:
    """Prepare one current phishing URL per registered domain."""
    frame = pd.read_csv(
        path,
        dtype=str,
        on_bad_lines="skip",
    )

    required = {
        "phish_id",
        "url",
        "submission_time",
        "verified",
        "online",
    }

    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(f"PhishTank feed missing required columns: {sorted(missing)}")

    raw_rows = len(frame)

    verified = frame["verified"].fillna("").str.lower().eq("yes")

    online = frame["online"].fillna("").str.lower().eq("yes")

    frame = frame.loc[verified & online].copy()

    verified_online_rows = len(frame)

    frame["registered_domain"] = frame["url"].map(registered_domain)

    frame = frame.loc[frame["registered_domain"].notna()].copy()

    valid_domain_rows = len(frame)

    development_overlap_mask = frame["registered_domain"].isin(development_domains)

    development_overlap_removed = int(development_overlap_mask.sum())

    frame = frame.loc[~development_overlap_mask].copy()

    frame["submission_time_parsed"] = pd.to_datetime(
        frame["submission_time"],
        errors="coerce",
        utc=True,
    )

    frame["phish_id_numeric"] = pd.to_numeric(
        frame["phish_id"],
        errors="coerce",
    )

    frame = frame.sort_values(
        by=[
            "submission_time_parsed",
            "phish_id_numeric",
        ],
        ascending=[
            False,
            False,
        ],
        na_position="last",
    )

    before_domain_dedup = len(frame)

    # Prevent a single hosting/domain cluster from dominating
    # external recall.
    frame = frame.drop_duplicates(
        subset=["registered_domain"],
        keep="first",
    )

    duplicate_domains_removed = before_domain_dedup - len(frame)

    frame = frame.head(max_rows).copy()

    output = pd.DataFrame(
        {
            "url_model_input": (frame["url"].astype(str)),
            "target": 1,
            "source": "phishtank",
            "registered_domain": (frame["registered_domain"].astype(str)),
            "source_reference": (frame["phish_id"].astype(str)),
            "source_time": (frame["submission_time"].astype(str)),
        }
    )

    summary = {
        "raw_rows": int(raw_rows),
        "verified_online_rows": int(verified_online_rows),
        "valid_domain_rows": int(valid_domain_rows),
        "development_domain_overlap_removed": int(development_overlap_removed),
        "duplicate_domain_rows_removed": int(duplicate_domains_removed),
        "selected_rows": int(len(output)),
    }

    return output, summary


def _prepare_tranco(
    *,
    path: Path,
    development_domains: set[str],
    phishing_domains: set[str],
    max_rows: int,
) -> tuple[
    pd.DataFrame,
    dict[str, int],
]:
    """Prepare Tranco popular domains as a benign proxy."""
    frame = pd.read_csv(
        path,
        names=[
            "rank",
            "domain",
        ],
        header=None,
        dtype={
            "rank": str,
            "domain": str,
        },
        on_bad_lines="skip",
    )

    raw_rows = len(frame)

    frame["domain"] = frame["domain"].fillna("").str.strip().str.lower()

    frame = frame.loc[frame["domain"].ne("")].copy()

    frame["registered_domain"] = frame["domain"].map(registered_domain)

    frame = frame.loc[frame["registered_domain"].notna()].copy()

    valid_domain_rows = len(frame)

    development_overlap_mask = frame["registered_domain"].isin(development_domains)

    development_overlap_removed = int(development_overlap_mask.sum())

    frame = frame.loc[~development_overlap_mask].copy()

    phishing_conflict_mask = frame["registered_domain"].isin(phishing_domains)

    phishing_conflicts_removed = int(phishing_conflict_mask.sum())

    frame = frame.loc[~phishing_conflict_mask].copy()

    frame = frame.drop_duplicates(
        subset=["registered_domain"],
        keep="first",
    )

    frame["rank_numeric"] = pd.to_numeric(
        frame["rank"],
        errors="coerce",
    )

    frame = frame.sort_values(
        by="rank_numeric",
        ascending=True,
        na_position="last",
    )

    frame = frame.head(max_rows).copy()

    output = pd.DataFrame(
        {
            # Deliberately do NOT add a trailing slash.
            # Phase 2C-B found strong trailing-slash sensitivity.
            "url_model_input": ("https://" + frame["domain"].astype(str)),
            "target": 0,
            "source": ("tranco_benign_proxy"),
            "registered_domain": (frame["registered_domain"].astype(str)),
            "source_reference": (frame["rank"].astype(str)),
            "source_time": "",
        }
    )

    summary = {
        "raw_rows": int(raw_rows),
        "valid_domain_rows": int(valid_domain_rows),
        "development_domain_overlap_removed": int(development_overlap_removed),
        "phishtank_domain_conflicts_removed": int(phishing_conflicts_removed),
        "selected_rows": int(len(output)),
    }

    return output, summary


def build_external_candidate(
    *,
    phishtank_path: Path,
    tranco_path: Path,
    tranco_list_id: str,
    source_metadata: dict[
        str,
        object,
    ]
    | None = None,
    train_path: Path = TRAIN_DATA_PATH,
    validation_path: Path = VALIDATION_DATA_PATH,
    output_path: Path = EXTERNAL_DATA_PATH,
    report_dir: Path = EXTERNAL_REPORT_DIR,
    max_per_class: int = DEFAULT_MAX_PER_CLASS,
) -> dict[str, object]:
    """Create a balanced domain-disjoint external/OOD candidate."""
    if max_per_class < 1:
        raise ValueError("max_per_class must be positive.")

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    development_domains = _development_domains(
        train_path=train_path,
        validation_path=(validation_path),
    )

    phishing, phishing_summary = _prepare_phishtank(
        path=phishtank_path,
        development_domains=(development_domains),
        max_rows=max_per_class,
    )

    if phishing.empty:
        raise RuntimeError("No usable external phishing URLs remained after filtering.")

    phishing_domains = set(phishing["registered_domain"])

    # Build at most the same number of benign rows so
    # the combined diagnostic set does not become
    # class-dominated.
    benign_limit = min(
        max_per_class,
        len(phishing),
    )

    benign, benign_summary = _prepare_tranco(
        path=tranco_path,
        development_domains=(development_domains),
        phishing_domains=(phishing_domains),
        max_rows=benign_limit,
    )

    if benign.empty:
        raise RuntimeError("No usable Tranco benign-proxy domains remained.")

    # If Tranco yielded fewer rows than phishing,
    # trim phishing so the combined diagnostic set
    # remains balanced.
    final_per_class = min(
        len(phishing),
        len(benign),
    )

    phishing = phishing.head(final_per_class).copy()

    benign = benign.head(final_per_class).copy()

    combined = pd.concat(
        [
            phishing,
            benign,
        ],
        ignore_index=True,
    )

    combined = combined.sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    combined.to_parquet(
        output_path,
        index=False,
    )

    payload: dict[
        str,
        object,
    ] = {
        "phase": "2C-C1",
        "created_at_utc": (datetime.now(UTC).isoformat()),
        "locked_test_used": False,
        "locked_test_scored": False,
        "train_validation_domains": int(len(development_domains)),
        "tranco_list_id": (tranco_list_id),
        "max_per_class_requested": int(max_per_class),
        "final_rows_per_class": int(final_per_class),
        "final_total_rows": int(len(combined)),
        "target_convention": {
            "0": ("Tranco popular-domain benign proxy"),
            "1": ("PhishTank verified-online phishing URL"),
        },
        "phishtank": (phishing_summary),
        "tranco": (benign_summary),
        "source_metadata": (source_metadata or {}),
        "input_files": {
            "phishtank": {
                "path": str(phishtank_path),
                "sha256": (_sha256_file(phishtank_path)),
            },
            "tranco": {
                "path": str(tranco_path),
                "sha256": (_sha256_file(tranco_path)),
            },
        },
        "output_path": str(output_path),
        "output_sha256": (_sha256_file(output_path)),
        "limitations": [
            ("Tranco is a popular-domain proxy, not perfect benign ground truth."),
            ("Tranco examples are synthesized as https://<domain> root URLs."),
            ("Only one PhishTank URL per registered domain is retained."),
            (
                "The combined class balance is "
                "constructed for diagnostic evaluation "
                "and does not represent deployment prevalence."
            ),
            ("The locked internal test set has not been scored or used for model decisions."),
        ],
    }

    (report_dir / "external_dataset_provenance.json").write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Phase 2C-C1 external/OOD preparation")

    print("=" * 70)

    print(f"Development domains excluded: {len(development_domains):,}")

    print(f"PhishTank raw rows: {phishing_summary['raw_rows']:,}")

    print(
        "PhishTank development-domain "
        "overlap removed: "
        f"{phishing_summary['development_domain_overlap_removed']:,}"
    )

    print(
        "Tranco development-domain "
        "overlap removed: "
        f"{benign_summary['development_domain_overlap_removed']:,}"
    )

    print(
        "Tranco/PhishTank domain conflicts removed: "
        f"{benign_summary['phishtank_domain_conflicts_removed']:,}"
    )

    print(f"Final phishing rows: {final_per_class:,}")

    print(f"Final benign-proxy rows: {final_per_class:,}")

    print(f"Final total rows: {len(combined):,}")

    print(f"Tranco list ID: {tranco_list_id}")

    print("Locked test scored: False")

    return payload


def download_and_prepare_external_data(
    *,
    raw_dir: Path = EXTERNAL_RAW_DIR,
    output_path: Path = EXTERNAL_DATA_PATH,
    report_dir: Path = EXTERNAL_REPORT_DIR,
    max_per_class: int = DEFAULT_MAX_PER_CLASS,
) -> dict[str, object]:
    """Download reproducible source snapshots and build OOD candidate."""
    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    user_agent = os.getenv(
        "PHISHTANK_USER_AGENT",
        "phishguard-research/0.1",
    ).strip()

    if not user_agent:
        raise ValueError("PHISHTANK_USER_AGENT must not be empty.")

    tranco_list_id = _fetch_text(
        url=TRANCO_LATEST_ID_URL,
        user_agent=user_agent,
    )

    if not _LIST_ID_RE.fullmatch(tranco_list_id):
        raise RuntimeError(f"Unexpected Tranco list ID: {tranco_list_id!r}")

    phishtank_path = raw_dir / (f"phishtank_online_valid_{timestamp}.csv")

    tranco_path = raw_dir / (f"tranco_{tranco_list_id}_top1m.csv")

    phishtank_url = _phishtank_download_url()

    tranco_url = TRANCO_DOWNLOAD_URL.format(list_id=tranco_list_id)

    print("Downloading PhishTank snapshot...")

    phishtank_metadata = _download_file(
        url=phishtank_url,
        output_path=(phishtank_path),
        user_agent=user_agent,
    )

    print("Downloading Tranco snapshot...")

    tranco_metadata = _download_file(
        url=tranco_url,
        output_path=(tranco_path),
        user_agent=user_agent,
    )

    source_metadata = {
        "phishtank": (phishtank_metadata),
        "tranco": {
            **tranco_metadata,
            "list_id": (tranco_list_id),
            "latest_id_endpoint": (TRANCO_LATEST_ID_URL),
        },
    }

    return build_external_candidate(
        phishtank_path=(phishtank_path),
        tranco_path=(tranco_path),
        tranco_list_id=(tranco_list_id),
        source_metadata=(source_metadata),
        output_path=(output_path),
        report_dir=(report_dir),
        max_per_class=(max_per_class),
    )


def main() -> None:
    """Run Phase 2C-C1."""
    parser = argparse.ArgumentParser(
        description=("Prepare external/OOD PhishGuard evaluation data.")
    )

    parser.add_argument(
        "--max-per-class",
        type=int,
        default=DEFAULT_MAX_PER_CLASS,
        help=("Maximum number of domain-disjoint examples retained per class."),
    )

    args = parser.parse_args()

    download_and_prepare_external_data(max_per_class=(args.max_per_class))


if __name__ == "__main__":
    main()
