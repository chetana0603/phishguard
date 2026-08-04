"""Safe URL-string feature extraction for heuristic and ML models."""

from __future__ import annotations

import ipaddress
import re
from dataclasses import asdict, dataclass
from typing import Final

from phishguard.data.prepare import extract_registered_domain, parse_url

_PERCENT_ENCODING_RE: Final[re.Pattern[str]] = re.compile(r"%[0-9A-Fa-f]{2}")
_SPECIAL_CHARACTER_RE: Final[re.Pattern[str]] = re.compile(r"[^A-Za-z0-9]")

SUSPICIOUS_KEYWORDS: Final[frozenset[str]] = frozenset(
    {
        "account",
        "auth",
        "banking",
        "confirm",
        "credential",
        "invoice",
        "login",
        "password",
        "payment",
        "recover",
        "secure",
        "session",
        "signin",
        "support",
        "unlock",
        "update",
        "verify",
        "wallet",
        "webscr",
    }
)

URL_SHORTENERS: Final[frozenset[str]] = frozenset(
    {
        "bit.ly",
        "buff.ly",
        "cutt.ly",
        "goo.gl",
        "is.gd",
        "ow.ly",
        "rebrand.ly",
        "shorturl.at",
        "t.co",
        "tiny.cc",
        "tinyurl.com",
    }
)


@dataclass(frozen=True)
class URLFeatures:
    """URL-only attributes derived without opening the URL."""

    is_valid_url: bool
    uses_https: bool
    has_non_http_scheme: bool
    has_ip_hostname: bool
    has_userinfo: bool
    has_punycode: bool
    has_non_default_port: bool
    uses_url_shortener: bool
    url_length: int
    hostname_length: int
    path_length: int
    query_length: int
    dot_count: int
    hostname_hyphen_count: int
    path_depth: int
    digit_ratio: float
    special_character_count: int
    percent_encoded_count: int
    suspicious_hostname_keyword_count: int
    suspicious_path_query_keyword_count: int

    def to_dict(self) -> dict[str, object]:
        """Return a serialisable representation."""
        return asdict(self)


def _keyword_count(text: str) -> int:
    lowered = text.lower()
    return sum(keyword in lowered for keyword in SUSPICIOUS_KEYWORDS)


def _is_ip_hostname(hostname: str) -> bool:
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        return False
    return True


def extract_url_features(value: object) -> URLFeatures:
    """Extract deterministic, URL-only features from an untrusted string."""
    parsed_url = parse_url(value)
    raw = parsed_url.original
    parsed = parsed_url.parsed
    hostname = (parsed_url.hostname or "").rstrip(".").lower()

    if parsed is None:
        scheme = ""
        path = ""
        query = ""
        netloc = ""
    else:
        scheme = parsed.scheme.lower()
        path = parsed.path or ""
        query = parsed.query or ""
        netloc = parsed.netloc or ""

    has_non_default_port = False
    if parsed is not None:
        try:
            port = parsed.port
        except ValueError:
            port = None
            has_non_default_port = True
        else:
            has_non_default_port = port is not None and not (
                (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
            )

    registered_domain = extract_registered_domain(raw)
    digits = sum(character.isdigit() for character in raw)
    digit_ratio = digits / len(raw) if raw else 0.0

    path_segments = [segment for segment in path.split("/") if segment]
    hostname_text = hostname
    path_query_text = f"{path}?{query}"

    return URLFeatures(
        is_valid_url=bool(hostname),
        uses_https=scheme == "https",
        has_non_http_scheme=bool(parsed_url.explicit_scheme and scheme not in {"http", "https"}),
        has_ip_hostname=_is_ip_hostname(hostname) if hostname else False,
        has_userinfo=(parsed.username is not None if parsed is not None else False)
        or "@" in netloc,
        has_punycode=any(label.startswith("xn--") for label in hostname.split(".")),
        has_non_default_port=has_non_default_port,
        uses_url_shortener=registered_domain in URL_SHORTENERS,
        url_length=len(raw),
        hostname_length=len(hostname),
        path_length=len(path),
        query_length=len(query),
        dot_count=hostname.count("."),
        hostname_hyphen_count=hostname.count("-"),
        path_depth=len(path_segments),
        digit_ratio=digit_ratio,
        special_character_count=len(_SPECIAL_CHARACTER_RE.findall(raw)),
        percent_encoded_count=len(_PERCENT_ENCODING_RE.findall(raw)),
        suspicious_hostname_keyword_count=_keyword_count(hostname_text),
        suspicious_path_query_keyword_count=_keyword_count(path_query_text),
    )
