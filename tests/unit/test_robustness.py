import pytest

from phishguard.evaluation.robustness import (
    perturb_host_case,
    perturb_non_root_trailing_slash,
    perturb_query_order,
    perturb_root_slash,
    perturb_scheme,
    perturb_trailing_slash,
    perturb_www,
)


def test_scheme_toggle_https_to_http() -> None:
    result = perturb_scheme("https://example.com/login?a=1")

    assert result == "http://example.com/login?a=1"


def test_scheme_toggle_http_to_https() -> None:
    result = perturb_scheme("http://example.com/login")

    assert result == "https://example.com/login"


def test_scheme_added_to_scheme_less_url() -> None:
    result = perturb_scheme("example.com/login")

    assert result == "https://example.com/login"


def test_www_toggle_adds_www() -> None:
    result = perturb_www("https://example.com/login")

    assert result == "https://www.example.com/login"


def test_www_toggle_removes_www() -> None:
    result = perturb_www("https://www.example.com/login")

    assert result == "https://example.com/login"


def test_host_case_changes_only_hostname() -> None:
    result = perturb_host_case("https://example.com/Login?Token=ABC")

    assert result is not None
    assert "/Login?Token=ABC" in result
    assert "EXAMPLE.COM" in result


def test_trailing_slash_addition() -> None:
    result = perturb_trailing_slash("https://example.com/login?a=1")

    assert result == "https://example.com/login/?a=1"


def test_trailing_slash_removal() -> None:
    result = perturb_trailing_slash("https://example.com/login/?a=1")

    assert result == "https://example.com/login?a=1"


def test_query_order_reversed() -> None:
    result = perturb_query_order("https://example.com/search?a=1&b=2&c=3")

    assert result == ("https://example.com/search?c=3&b=2&a=1")


def test_query_order_skips_duplicate_keys() -> None:
    result = perturb_query_order("https://example.com/search?a=1&a=2")

    assert result is None


def test_query_order_requires_multiple_parameters() -> None:
    result = perturb_query_order("https://example.com/search?a=1")

    assert result is None


@pytest.mark.parametrize(
    "value",
    [
        "",
        "not a valid url",
    ],
)
def test_invalid_url_perturbations_do_not_crash(
    value: str,
) -> None:
    assert perturb_scheme(value) is None
    assert perturb_www(value) is None
    assert perturb_host_case(value) is None
    assert perturb_trailing_slash(value) is None
    assert perturb_query_order(value) is None


def test_root_slash_perturbation_only_changes_root() -> None:
    assert perturb_root_slash("https://example.com") == "https://example.com/"

    assert perturb_root_slash("https://example.com/") == "https://example.com"

    assert perturb_root_slash("https://example.com/login") is None

    assert perturb_root_slash("https://example.com/login/") is None


def test_non_root_trailing_slash_excludes_root() -> None:
    assert perturb_non_root_trailing_slash("https://example.com") is None

    assert perturb_non_root_trailing_slash("https://example.com/") is None

    assert (
        perturb_non_root_trailing_slash("https://example.com/login") == "https://example.com/login/"
    )

    assert (
        perturb_non_root_trailing_slash("https://example.com/login/") == "https://example.com/login"
    )
