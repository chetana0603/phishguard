"""Tests for URL parsing, normalisation, and target conversion."""

from phishguard.data.prepare import (
    convert_target,
    extract_registered_domain,
    is_valid_url,
    normalise_url_for_dedupe,
)


def test_hostname_is_case_insensitive() -> None:
    first = normalise_url_for_dedupe("https://Example.COM/Login")
    second = normalise_url_for_dedupe("https://example.com/Login")
    assert first == second


def test_path_case_is_preserved() -> None:
    upper = normalise_url_for_dedupe("https://example.com/Login")
    lower = normalise_url_for_dedupe("https://example.com/login")
    assert upper != lower


def test_fragment_is_removed_from_dedupe_key() -> None:
    with_fragment = normalise_url_for_dedupe("https://example.com/path?q=1#section")
    without_fragment = normalise_url_for_dedupe("https://example.com/path?q=1")
    assert with_fragment == without_fragment


def test_default_https_port_is_removed() -> None:
    with_port = normalise_url_for_dedupe("https://example.com:443/path")
    without_port = normalise_url_for_dedupe("https://example.com/path")
    assert with_port == without_port


def test_phishing_is_positive_class() -> None:
    assert convert_target(0) == 1
    assert convert_target(1) == 0


def test_registered_domain_uses_public_suffix() -> None:
    assert extract_registered_domain("https://login.accounts.example.co.uk/path") == "example.co.uk"


def test_ip_address_is_retained_as_group() -> None:
    assert extract_registered_domain("http://192.0.2.10/login") == "192.0.2.10"


def test_invalid_url_without_hostname() -> None:
    assert not is_valid_url("")
