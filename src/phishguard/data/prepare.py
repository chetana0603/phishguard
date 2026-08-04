"""Prepare a safe, URL-only dataset for phishing-model development."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from dataclasses import dataclass
from typing import Final
from urllib.parse import SplitResult, urlsplit, urlunsplit

import pandas as pd
import tldextract

from phishguard.config import (
    CONFLICTING_LABELS_PATH,
    PREPARATION_SUMMARY_PATH,
    PREPARED_DATA_PATH,
    RAW_DATA_PATH,
    ensure_directories,
)

_EXPLICIT_SCHEME_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_HOST_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s")

# Use tldextract's bundled public-suffix snapshot. This avoids a network request at runtime.
_TLD_EXTRACTOR = tldextract.TLDExtract(suffix_list_urls=())


@dataclass(frozen=True)
class ParsedURL:
    """A URL parsed without ever opening or requesting it."""

    original: str
    parsed: SplitResult | None
    explicit_scheme: bool
    scheme_relative: bool

    @property
    def hostname(self) -> str | None:
        if self.parsed is None:
            return None
        try:
            return self.parsed.hostname
        except ValueError:
            return None


def parse_url(value: object) -> ParsedURL:
    """Parse a URL defensively, adding a temporary scheme when necessary."""
    if value is None or pd.isna(value):
        return ParsedURL(original="", parsed=None, explicit_scheme=False, scheme_relative=False)

    original = str(value).strip()
    if not original:
        return ParsedURL(original="", parsed=None, explicit_scheme=False, scheme_relative=False)

    explicit_scheme = bool(_EXPLICIT_SCHEME_RE.match(original))
    scheme_relative = original.startswith("//")

    if explicit_scheme:
        candidate = original
    elif scheme_relative:
        candidate = f"http:{original}"
    else:
        candidate = f"http://{original}"

    try:
        parsed = urlsplit(candidate)
    except ValueError:
        parsed = None

    return ParsedURL(
        original=original,
        parsed=parsed,
        explicit_scheme=explicit_scheme,
        scheme_relative=scheme_relative,
    )


def is_valid_url(value: object) -> bool:
    """Return whether a URL has a syntactically usable hostname."""
    parsed_url = parse_url(value)
    hostname = parsed_url.hostname
    return bool(hostname) and _HOST_WHITESPACE_RE.search(hostname) is None


def _normalised_netloc(parsed_url: ParsedURL) -> str | None:
    """Build a hostname-normalised netloc while preserving credentials when present."""
    parsed = parsed_url.parsed
    hostname = parsed_url.hostname
    if parsed is None or hostname is None:
        return None

    host = hostname.rstrip(".").lower()
    try:
        ip = ipaddress.ip_address(host)
        if ip.version == 6:
            host = f"[{host}]"
    except ValueError:
        pass

    try:
        port = parsed.port
    except ValueError:
        return None

    scheme = parsed.scheme.lower()
    is_default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    port_text = "" if port is None or is_default_port else f":{port}"

    userinfo = ""
    if parsed.username is not None:
        userinfo = parsed.username
        if parsed.password is not None:
            userinfo += f":{parsed.password}"
        userinfo += "@"

    return f"{userinfo}{host}{port_text}"


def normalise_url_for_dedupe(value: object) -> str:
    """Create a conservative canonical key for duplicate detection.

    The hostname and scheme are lowercased, fragments are removed, and default
    HTTP/HTTPS ports are removed. Path and query casing are preserved.
    """
    parsed_url = parse_url(value)
    if parsed_url.parsed is None:
        return parsed_url.original.split("#", maxsplit=1)[0]

    netloc = _normalised_netloc(parsed_url)
    if netloc is None:
        return parsed_url.original.split("#", maxsplit=1)[0]

    parsed = parsed_url.parsed
    path = parsed.path or ""
    query = parsed.query or ""

    if parsed_url.explicit_scheme:
        scheme = parsed.scheme.lower()
        return urlunsplit((scheme, netloc, path, query, ""))

    # Preserve the fact that the original value had no explicit scheme.
    return urlunsplit(("", netloc, path, query, ""))


def extract_registered_domain(value: object) -> str | None:
    """Extract a public-suffix-aware registrable domain for group splitting."""
    parsed_url = parse_url(value)
    hostname = parsed_url.hostname
    if not hostname:
        return None

    hostname = hostname.rstrip(".").lower()
    try:
        ipaddress.ip_address(hostname)
        return hostname
    except ValueError:
        pass

    extracted = _TLD_EXTRACTOR(hostname)
    registered = extracted.top_domain_under_public_suffix
    return registered or hostname


def convert_target(original_label: object) -> int:
    """Convert UCI labels so phishing is the positive class.

    UCI convention: 1 = legitimate, 0 = phishing.
    PhishGuard convention: 1 = phishing, 0 = legitimate.
    """
    try:
        value = int(original_label)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid source label: {original_label!r}") from exc

    if value not in {0, 1}:
        raise ValueError(f"Expected a binary source label, received {value}.")
    return 1 - value


def _invalid_split_group(dedupe_key: str) -> str:
    digest = hashlib.sha256(dedupe_key.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"invalid-{digest}"


def prepare_dataframe(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Transform raw UCI rows into the URL-only modelling table."""
    missing_columns = {"URL", "label"} - set(raw.columns)
    if missing_columns:
        raise ValueError(f"Raw data is missing required columns: {sorted(missing_columns)}")

    working = pd.DataFrame(
        {
            "url_original": raw["URL"].astype("string"),
            "source_label": raw["label"],
        }
    )
    working["url_model_input"] = working["url_original"].fillna("").str.strip()

    empty_mask = working["url_model_input"].eq("")
    empty_url_count = int(empty_mask.sum())
    working = working.loc[~empty_mask].copy()

    working["url_dedupe_key"] = working["url_model_input"].map(normalise_url_for_dedupe)
    working["registered_domain"] = working["url_model_input"].map(extract_registered_domain)
    working["is_valid_url"] = working["url_model_input"].map(is_valid_url)
    working["target"] = working["source_label"].map(convert_target).astype("int8")

    working["split_group"] = working.apply(
        lambda row: (
            row["registered_domain"]
            if isinstance(row["registered_domain"], str) and row["registered_domain"]
            else _invalid_split_group(str(row["url_dedupe_key"]))
        ),
        axis=1,
    )

    original_rows_after_empty_removal = len(working)

    label_counts = working.groupby("url_dedupe_key", dropna=False)["target"].nunique()
    conflicting_keys = set(label_counts[label_counts > 1].index)
    conflicts = working[working["url_dedupe_key"].isin(conflicting_keys)].copy()

    if conflicting_keys:
        working = working[~working["url_dedupe_key"].isin(conflicting_keys)].copy()

    before_deduplication = len(working)
    working = working.drop_duplicates(subset=["url_dedupe_key", "target"], keep="first")
    normalised_duplicates_removed = before_deduplication - len(working)

    working = working.sort_values("url_dedupe_key", kind="stable").reset_index(drop=True)
    conflicts = conflicts.sort_values("url_dedupe_key", kind="stable").reset_index(drop=True)

    summary = {
        "raw_rows": int(len(raw)),
        "empty_urls_removed": empty_url_count,
        "rows_after_empty_removal": int(original_rows_after_empty_removal),
        "conflicting_dedupe_keys_removed": int(len(conflicting_keys)),
        "conflicting_rows_removed": int(len(conflicts)),
        "normalised_duplicates_removed": int(normalised_duplicates_removed),
        "prepared_rows": int(len(working)),
        "invalid_url_rows_retained": int((~working["is_valid_url"]).sum()),
        "unique_split_groups": int(working["split_group"].nunique()),
    }
    return working, conflicts, summary


def prepare_dataset() -> pd.DataFrame:
    """Load the raw parquet file, prepare it, and write reproducible artefacts."""
    ensure_directories()
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {RAW_DATA_PATH}. "
            "Run `python -m phishguard.data.download` first."
        )

    raw = pd.read_parquet(RAW_DATA_PATH)
    prepared, conflicts, summary = prepare_dataframe(raw)

    prepared.to_parquet(PREPARED_DATA_PATH, index=False)
    conflicts.to_parquet(CONFLICTING_LABELS_PATH, index=False)
    PREPARATION_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved {len(prepared):,} prepared rows to {PREPARED_DATA_PATH}")
    print(f"Saved {len(conflicts):,} conflicting rows to {CONFLICTING_LABELS_PATH}")
    print(f"Saved preparation summary to {PREPARATION_SUMMARY_PATH}")
    return prepared


def main() -> None:
    prepare_dataset()


if __name__ == "__main__":
    main()
