"""Character-level TF-IDF logistic-regression phishing model."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from phishguard.config import RANDOM_STATE

TFIDF_LOGISTIC_VERSION = "tfidf-logistic-v2"

# Module-level constants
_SCHEME_RE = re.compile(
    r"^https?://",
    flags=re.IGNORECASE,
)


def _canonicalize_netloc(
    netloc: str,
    hostname: str,
) -> str:
    """Lowercase hostname while preserving userinfo and port."""
    user_info = ""

    if "@" in netloc:
        user_info, host_port = netloc.rsplit(
            "@",
            1,
        )
        user_info += "@"

    else:
        host_port = netloc

    # Preserve IPv6 brackets and any port suffix.
    if host_port.startswith("["):
        closing = host_port.find("]")

        if closing != -1:
            host = host_port[: closing + 1].lower()

            suffix = host_port[closing + 1 :]

            return f"{user_info}{host}{suffix}"

    host = hostname.lower()

    if host.startswith("www."):
        host = host[4:]

    suffix = ""

    if ":" in host_port:
        _, separator, candidate_port = host_port.rpartition(":")

        if separator and candidate_port.isdigit():
            suffix = f":{candidate_port}"

    return f"{user_info}{host}{suffix}"


def _canonicalize_root_path(
    path: str,
) -> str:
    """
    Canonicalize only the HTTP(S) root-path representation.

    An empty path and "/" are treated identically.

    Non-root trailing slashes are deliberately preserved:
    "/login" and "/login/" remain different strings.
    """
    if path == "/":
        return ""

    return path


def normalize_url_for_text_model(
    value: object,
) -> str:
    """
    Canonicalize URL text for the character TF-IDF model.

    The transformation:
    - removes HTTP/HTTPS scheme;
    - removes a leading www from the hostname;
    - lowercases only the hostname;
    - treats an empty root path and "/" identically;
    - preserves non-root path/query/fragment casing;
    - preserves non-root trailing slashes.
    """
    text = str(value).strip()

    if not text:
        return text

    scheme_neutral = _SCHEME_RE.sub(
        "",
        text,
    )

    try:
        parsed = urlsplit(f"//{scheme_neutral}")

    except ValueError:
        return scheme_neutral

    hostname = parsed.hostname

    if not parsed.netloc or not hostname:
        return scheme_neutral

    canonical_netloc = _canonicalize_netloc(
        parsed.netloc,
        hostname,
    )

    canonical_path = _canonicalize_root_path(parsed.path)

    canonical = urlunsplit(
        (
            "",
            canonical_netloc,
            canonical_path,
            parsed.query,
            parsed.fragment,
        )
    )

    if canonical.startswith("//"):
        canonical = canonical[2:]

    return canonical


TFIDF_LOGISTIC_VERSION = "tfidf-logistic-v3"


@dataclass(frozen=True)
class TfidfLogisticSpec:
    """Reproducible vectorizer and classifier configuration."""

    name: str
    ngram_min: int
    ngram_max: int
    c: float
    class_weight: str | None
    min_df: int = 3
    max_features: int = 250_000
    sublinear_tf: bool = True
    lowercase: bool = False
    scheme_neutral: bool = False

    def __post_init__(self) -> None:
        if self.ngram_min < 1 or self.ngram_max < self.ngram_min:
            raise ValueError("Invalid character n-gram range.")

        if self.c <= 0:
            raise ValueError("C must be positive.")

        if self.class_weight not in {None, "balanced"}:
            raise ValueError("class_weight must be None or 'balanced'.")

        if self.min_df < 1:
            raise ValueError("min_df must be at least 1.")

        if self.max_features < 1:
            raise ValueError("max_features must be positive.")

    @property
    def ngram_range(self) -> tuple[int, int]:
        return self.ngram_min, self.ngram_max

    @property
    def vectorizer_key(self) -> tuple[object, ...]:
        return (
            self.ngram_range,
            self.min_df,
            self.max_features,
            self.sublinear_tf,
            self.lowercase,
            self.scheme_neutral,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def compact_candidate_specs() -> list[TfidfLogisticSpec]:
    """Return a laptop-friendly validation search with one robustness ablation."""
    return [
        TfidfLogisticSpec("char35_c05", 3, 5, 0.5, None),
        TfidfLogisticSpec("char35_c10", 3, 5, 1.0, None),
        TfidfLogisticSpec("char35_c20_bal", 3, 5, 2.0, "balanced"),
        TfidfLogisticSpec("char36_c10", 3, 6, 1.0, None),
        TfidfLogisticSpec(
            name="char25_c10_bal_raw",
            ngram_min=2,
            ngram_max=5,
            c=1.0,
            class_weight="balanced",
        ),
        TfidfLogisticSpec(
            name="char25_c10_bal_scheme_www_neutral",
            ngram_min=2,
            ngram_max=5,
            c=1.0,
            class_weight="balanced",
            scheme_neutral=True,
        ),
        TfidfLogisticSpec("char46_c10", 4, 6, 1.0, None),
    ]


def full_candidate_specs() -> list[TfidfLogisticSpec]:
    """Return the complete 3 x 3 x 2 validation grid."""
    specs: list[TfidfLogisticSpec] = []
    for ngram_min, ngram_max in ((3, 5), (3, 6), (2, 5)):
        for c in (0.1, 1.0, 5.0):
            for class_weight in (None, "balanced"):
                weight_name = "none" if class_weight is None else "balanced"
                specs.append(
                    TfidfLogisticSpec(
                        name=f"char{ngram_min}{ngram_max}_c{c:g}_{weight_name}",
                        ngram_min=ngram_min,
                        ngram_max=ngram_max,
                        c=c,
                        class_weight=class_weight,
                    )
                )
    return specs


def build_vectorizer(spec: TfidfLogisticSpec) -> TfidfVectorizer:
    """Create a URL character n-gram vectorizer."""
    preprocessor = normalize_url_for_text_model if spec.scheme_neutral else None

    return TfidfVectorizer(
        analyzer="char",
        ngram_range=spec.ngram_range,
        min_df=spec.min_df,
        max_features=spec.max_features,
        sublinear_tf=spec.sublinear_tf,
        lowercase=spec.lowercase,
        dtype=np.float32,
        preprocessor=preprocessor,
    )


def build_classifier(spec: TfidfLogisticSpec) -> LogisticRegression:
    """Create a deterministic sparse logistic-regression classifier."""
    return LogisticRegression(
        C=spec.c,
        class_weight=spec.class_weight,
        max_iter=1_000,
        solver="liblinear",
        random_state=RANDOM_STATE,
    )


def build_pipeline(spec: TfidfLogisticSpec) -> Pipeline:
    """Create the complete training and inference pipeline."""
    return Pipeline(
        steps=[
            ("tfidf", build_vectorizer(spec)),
            ("classifier", build_classifier(spec)),
        ]
    )


def phishing_scores(pipeline: Pipeline, urls: list[str] | pd.Series) -> np.ndarray:
    """Return probability assigned to the phishing class (class 1)."""
    probabilities = pipeline.predict_proba(urls)
    classes = pipeline.named_steps["classifier"].classes_
    phishing_index = int(np.flatnonzero(classes == 1)[0])
    return probabilities[:, phishing_index].astype(float)


def top_character_ngrams(pipeline: Pipeline, *, top_n: int = 50) -> pd.DataFrame:
    """Return globally influential phishing and legitimate character n-grams."""
    if top_n < 1:
        raise ValueError("top_n must be positive.")

    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = vectorizer.get_feature_names_out()
    coefficients = classifier.coef_[0]

    phishing_indices = np.argsort(coefficients)[-top_n:][::-1]
    legitimate_indices = np.argsort(coefficients)[:top_n]

    rows: list[dict[str, object]] = []
    for direction, indices in (
        ("phishing", phishing_indices),
        ("legitimate", legitimate_indices),
    ):
        for rank, index in enumerate(indices, start=1):
            ngram = str(feature_names[index])
            rows.append(
                {
                    "direction": direction,
                    "rank": rank,
                    "ngram": ngram.encode("unicode_escape").decode("ascii"),
                    "coefficient": float(coefficients[index]),
                }
            )

    return pd.DataFrame(rows)
