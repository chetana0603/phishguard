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

    assert normalize_url_for_text_model("HTTP://WWW.EXAMPLE.COM/path") == "example.com/path"


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


def test_normalizer_lowercases_hostname_only() -> None:
    result = normalize_url_for_text_model("HTTPS://WWW.Example.COM/Login?Token=ABC")

    assert result == "example.com/Login?Token=ABC"


def test_normalizer_preserves_path_and_query_case() -> None:
    result = normalize_url_for_text_model("https://Example.COM/MyPath?User=ABC")

    assert result == "example.com/MyPath?User=ABC"


def test_normalizer_removes_www_case_insensitively() -> None:
    result = normalize_url_for_text_model("https://WWW.Example.COM/login")

    assert result == "example.com/login"


def test_normalizer_scheme_invariance() -> None:
    http_result = normalize_url_for_text_model("http://Example.COM/login")

    https_result = normalize_url_for_text_model("https://Example.COM/login")

    assert http_result == https_result


def test_normalizer_hostname_case_invariance() -> None:
    lower = normalize_url_for_text_model("https://example.com/Login")

    upper = normalize_url_for_text_model("https://EXAMPLE.COM/Login")

    assert lower == upper


def test_normalizer_treats_empty_and_root_path_as_equal() -> None:
    without_slash = normalize_url_for_text_model("https://example.com")

    with_slash = normalize_url_for_text_model("https://example.com/")

    assert without_slash == "example.com"
    assert with_slash == "example.com"
    assert without_slash == with_slash


def test_normalizer_root_slash_invariance_with_query() -> None:
    without_slash = normalize_url_for_text_model("https://example.com?token=ABC")

    with_slash = normalize_url_for_text_model("https://example.com/?token=ABC")

    assert without_slash == with_slash == "example.com?token=ABC"


def test_normalizer_root_slash_invariance_with_fragment() -> None:
    without_slash = normalize_url_for_text_model("https://example.com#section")

    with_slash = normalize_url_for_text_model("https://example.com/#section")

    assert without_slash == with_slash == "example.com#section"


def test_normalizer_preserves_non_root_trailing_slash_difference() -> None:
    without_slash = normalize_url_for_text_model("https://example.com/login")

    with_slash = normalize_url_for_text_model("https://example.com/login/")

    assert without_slash == "example.com/login"
    assert with_slash == "example.com/login/"
    assert without_slash != with_slash


def test_normalizer_combines_all_required_invariances() -> None:
    first = normalize_url_for_text_model("HTTP://WWW.EXAMPLE.COM/")

    second = normalize_url_for_text_model("https://example.com")

    assert first == second == "example.com"
