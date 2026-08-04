from phishguard.features.url_features import extract_url_features


def test_detects_ip_hostname() -> None:
    features = extract_url_features("http://192.168.1.10/login")
    assert features.has_ip_hostname


def test_detects_userinfo_and_at_symbol() -> None:
    features = extract_url_features("https://trusted.example@evil.test/login")
    assert features.has_userinfo


def test_detects_punycode() -> None:
    features = extract_url_features("https://xn--pple-43d.example/")
    assert features.has_punycode


def test_detects_shortener() -> None:
    features = extract_url_features("https://bit.ly/example")
    assert features.uses_url_shortener


def test_counts_suspicious_terms_separately() -> None:
    features = extract_url_features("https://secure-login.example/verify/account")
    assert features.suspicious_hostname_keyword_count == 2
    assert features.suspicious_path_query_keyword_count == 2
