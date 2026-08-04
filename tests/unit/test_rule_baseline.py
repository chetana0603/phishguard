import numpy as np
import pytest

from phishguard.models.rule_baseline import explain, predict, score_url


def test_obviously_suspicious_url_scores_above_simple_url() -> None:
    simple = score_url("https://example.com/about")
    suspicious = score_url("http://192.168.1.10/secure/login/verify?account=12345678901234567890")
    assert suspicious.score > simple.score


def test_score_is_bounded() -> None:
    prediction = score_url(
        "ftp://user:pass@192.168.1.10:9999/secure/login/verify/account/"
        "update/password?session=123456789012345678901234567890"
    )
    assert 0.0 <= prediction.score <= 1.0


def test_explanation_returns_high_weight_rules_first() -> None:
    contributions = explain("http://user@example.com/login")
    weights = [item.weight for item in contributions]
    assert weights == sorted(weights, reverse=True)


def test_predict_uses_supplied_threshold() -> None:
    values = ["https://example.com", "http://192.168.1.10/login"]
    predictions = predict(values, threshold=0.25)
    assert np.array_equal(predictions, np.asarray([0, 1], dtype=np.int8))


def test_predict_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="threshold"):
        predict(["https://example.com"], threshold=1.1)
