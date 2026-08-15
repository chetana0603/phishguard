import numpy as np

from phishguard.models.tfidf_logistic import (
    TfidfLogisticSpec,
    build_pipeline,
    normalize_url_for_text_model,
    phishing_scores,
    top_character_ngrams,
)


def test_scheme_neutral_url_normalization() -> None:
    assert normalize_url_for_text_model("https://www.example.com/login") == "example.com/login"

    assert normalize_url_for_text_model("HTTP://WWW.EXAMPLE.COM/path") == "EXAMPLE.COM/path"


def test_tfidf_logistic_pipeline_scores_urls() -> None:
    spec = TfidfLogisticSpec(
        name="test",
        ngram_min=2,
        ngram_max=4,
        c=1.0,
        class_weight=None,
        min_df=1,
        max_features=2_000,
    )
    urls = [
        "https://example.com/about",
        "https://docs.python.org/3/",
        "http://secure-login.example/verify-account",
        "http://192.168.1.10/update-password",
    ]
    targets = np.array([0, 0, 1, 1], dtype=np.int8)

    pipeline = build_pipeline(spec)
    pipeline.fit(urls, targets)
    scores = phishing_scores(pipeline, urls)

    assert scores.shape == (4,)
    assert np.all((scores >= 0.0) & (scores <= 1.0))


def test_top_character_ngrams_has_both_directions() -> None:
    spec = TfidfLogisticSpec(
        name="test",
        ngram_min=2,
        ngram_max=3,
        c=1.0,
        class_weight="balanced",
        min_df=1,
        max_features=1_000,
    )
    urls = [
        "https://example.com/home",
        "https://example.org/docs",
        "http://verify-login.bad/update",
        "http://account-security.bad/confirm",
    ]
    targets = [0, 0, 1, 1]

    pipeline = build_pipeline(spec)
    pipeline.fit(urls, targets)
    table = top_character_ngrams(pipeline, top_n=3)

    assert set(table["direction"]) == {"phishing", "legitimate"}
    assert len(table) == 6
