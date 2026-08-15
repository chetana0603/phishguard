"""Character-level TF-IDF logistic-regression phishing model."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from phishguard.config import RANDOM_STATE

TFIDF_LOGISTIC_VERSION = "tfidf-logistic-v1"

# Module-level constants
_SCHEME_RE = re.compile(r"^https?://", flags=re.IGNORECASE)
_LEADING_WWW_RE = re.compile(r"^www\.", flags=re.IGNORECASE)


# Module-level function
def normalize_url_for_text_model(value: object) -> str:
    """Remove protocol and leading-www shortcuts from a URL string."""
    text = str(value).strip()
    text = _SCHEME_RE.sub("", text)
    text = _LEADING_WWW_RE.sub("", text)
    return text


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
