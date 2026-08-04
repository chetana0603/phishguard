"""Transparent heuristic baseline for phishing URL detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from phishguard.features.url_features import URLFeatures, extract_url_features

RULE_BASELINE_VERSION = "rule-baseline-v1"
DEFAULT_THRESHOLD = 0.35


@dataclass(frozen=True)
class RuleContribution:
    """One triggered rule and its additive risk contribution."""

    name: str
    weight: float
    explanation: str


@dataclass(frozen=True)
class RulePrediction:
    """Risk score and explanations for one URL string."""

    score: float
    contributions: tuple[RuleContribution, ...]
    features: URLFeatures

    @property
    def triggered_rule_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.contributions)


def _append(
    contributions: list[RuleContribution],
    condition: bool,
    *,
    name: str,
    weight: float,
    explanation: str,
) -> None:
    if condition:
        contributions.append(
            RuleContribution(name=name, weight=weight, explanation=explanation)
        )


def score_features(features: URLFeatures) -> RulePrediction:
    """Assign an intentionally simple additive heuristic risk score."""
    contributions: list[RuleContribution] = []

    _append(
        contributions,
        not features.is_valid_url,
        name="invalid_url",
        weight=0.30,
        explanation="The input does not contain a syntactically usable hostname.",
    )
    _append(
        contributions,
        features.has_non_http_scheme,
        name="non_http_scheme",
        weight=0.30,
        explanation="The URL uses an explicit scheme other than HTTP or HTTPS.",
    )
    _append(
        contributions,
        features.has_ip_hostname,
        name="ip_hostname",
        weight=0.30,
        explanation="The hostname is an IP address rather than a named domain.",
    )
    _append(
        contributions,
        features.has_userinfo,
        name="userinfo_or_at_symbol",
        weight=0.25,
        explanation="The authority section contains user information or an @ symbol.",
    )
    _append(
        contributions,
        features.has_punycode,
        name="punycode_hostname",
        weight=0.20,
        explanation="The hostname contains a Punycode label.",
    )
    _append(
        contributions,
        features.suspicious_hostname_keyword_count >= 2,
        name="multiple_suspicious_hostname_terms",
        weight=0.22,
        explanation="The hostname contains multiple account or verification-related terms.",
    )
    _append(
        contributions,
        features.suspicious_hostname_keyword_count == 1,
        name="suspicious_hostname_term",
        weight=0.12,
        explanation="The hostname contains an account or verification-related term.",
    )
    _append(
        contributions,
        features.uses_url_shortener,
        name="url_shortener",
        weight=0.14,
        explanation="The URL uses a known shortening service that conceals the final domain.",
    )
    _append(
        contributions,
        features.has_non_default_port,
        name="non_default_port",
        weight=0.12,
        explanation="The URL specifies a non-default or malformed port.",
    )
    _append(
        contributions,
        features.url_length >= 200,
        name="very_long_url",
        weight=0.16,
        explanation="The URL is at least 200 characters long.",
    )
    _append(
        contributions,
        120 <= features.url_length < 200,
        name="long_url",
        weight=0.08,
        explanation="The URL is between 120 and 199 characters long.",
    )
    _append(
        contributions,
        features.dot_count >= 5,
        name="many_hostname_levels",
        weight=0.10,
        explanation="The hostname contains at least five dots.",
    )
    _append(
        contributions,
        features.dot_count == 4,
        name="deep_hostname",
        weight=0.05,
        explanation="The hostname contains four dots.",
    )
    _append(
        contributions,
        features.hostname_hyphen_count >= 3,
        name="many_hostname_hyphens",
        weight=0.08,
        explanation="The hostname contains at least three hyphens.",
    )
    _append(
        contributions,
        features.digit_ratio >= 0.30,
        name="high_digit_ratio",
        weight=0.10,
        explanation="At least 30% of URL characters are digits.",
    )
    _append(
        contributions,
        0.18 <= features.digit_ratio < 0.30,
        name="moderate_digit_ratio",
        weight=0.05,
        explanation="Between 18% and 30% of URL characters are digits.",
    )
    _append(
        contributions,
        features.percent_encoded_count >= 3,
        name="heavy_percent_encoding",
        weight=0.08,
        explanation="The URL contains at least three percent-encoded sequences.",
    )
    _append(
        contributions,
        features.suspicious_path_query_keyword_count >= 2,
        name="multiple_suspicious_path_terms",
        weight=0.08,
        explanation="The path or query contains multiple account-related terms.",
    )
    _append(
        contributions,
        features.suspicious_path_query_keyword_count == 1,
        name="suspicious_path_term",
        weight=0.03,
        explanation="The path or query contains an account-related term.",
    )
    _append(
        contributions,
        features.query_length >= 120,
        name="long_query",
        weight=0.05,
        explanation="The query string is at least 120 characters long.",
    )
    _append(
        contributions,
        features.special_character_count >= 20,
        name="many_special_characters",
        weight=0.05,
        explanation="The URL contains at least 20 non-alphanumeric characters.",
    )
    _append(
        contributions,
        features.is_valid_url and not features.uses_https,
        name="not_https",
        weight=0.02,
        explanation="The URL does not explicitly use HTTPS.",
    )

    score = min(sum(item.weight for item in contributions), 1.0)
    return RulePrediction(
        score=round(float(score), 6),
        contributions=tuple(contributions),
        features=features,
    )


def score_url(value: object) -> RulePrediction:
    """Extract features and score one URL without performing network access."""
    return score_features(extract_url_features(value))


def predict_scores(values: Iterable[object]) -> np.ndarray:
    """Return risk scores for an iterable of URL strings."""
    return np.asarray([score_url(value).score for value in values], dtype=float)


def predict(
    values: Iterable[object],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> np.ndarray:
    """Convert rule scores into binary phishing predictions."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")
    return (predict_scores(values) >= threshold).astype(np.int8)


def explain(value: object, *, limit: int = 5) -> Sequence[RuleContribution]:
    """Return the highest-weight triggered rules for one URL."""
    prediction = score_url(value)
    ordered = sorted(prediction.contributions, key=lambda item: item.weight, reverse=True)
    return tuple(ordered[:limit])
