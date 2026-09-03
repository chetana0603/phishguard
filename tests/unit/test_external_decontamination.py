from pathlib import Path

import pandas as pd

from phishguard.data.external_decontamination import (
    decontaminate_external_dataset,
    domain_sha256,
    load_locked_test_domain_hashes,
)


def test_domain_sha256_is_case_insensitive() -> None:
    lower = domain_sha256("example.com")

    upper = domain_sha256("EXAMPLE.COM")

    trailing_dot = domain_sha256("Example.Com.")

    assert lower == upper
    assert lower == trailing_dot


def test_locked_test_loader_uses_registered_domains(
    tmp_path: Path,
) -> None:
    test_path = tmp_path / "test.parquet"

    frame = pd.DataFrame(
        {
            "registered_domain": [
                "example.com",
                "EXAMPLE.COM",
                "other.org",
            ],
            "target": [
                1,
                0,
                1,
            ],
            "url_model_input": [
                "https://example.com/a",
                "https://example.com/b",
                "https://other.org/",
            ],
        }
    )

    frame.to_parquet(
        test_path,
        index=False,
    )

    hashes, summary = load_locked_test_domain_hashes(test_path=test_path)

    assert len(hashes) == 2

    assert summary["test_rows"] == 3

    assert summary["unique_test_domain_hashes"] == 2


def test_decontamination_removes_test_domain_overlap(
    tmp_path: Path,
) -> None:
    external_path = tmp_path / "external.parquet"

    test_path = tmp_path / "test.parquet"

    output_path = tmp_path / "external_eval.parquet"

    report_path = tmp_path / "report.json"

    external = pd.DataFrame(
        {
            "url_model_input": [
                "https://bad-one.net/login",
                "https://overlap.com/phish",
                "https://safe-one.org",
                "https://safe-two.org",
            ],
            "target": [
                1,
                1,
                0,
                0,
            ],
            "source": [
                "phishtank",
                "phishtank",
                "tranco_benign_proxy",
                "tranco_benign_proxy",
            ],
            "registered_domain": [
                "bad-one.net",
                "overlap.com",
                "safe-one.org",
                "safe-two.org",
            ],
        }
    )

    test = pd.DataFrame(
        {
            "registered_domain": [
                "OVERLAP.COM",
                "test-only.net",
            ],
            "target": [
                0,
                1,
            ],
            "url_model_input": [
                "https://overlap.com/",
                "https://test-only.net/",
            ],
        }
    )

    external.to_parquet(
        external_path,
        index=False,
    )

    test.to_parquet(
        test_path,
        index=False,
    )

    payload = decontaminate_external_dataset(
        external_path=(external_path),
        test_path=(test_path),
        output_path=(output_path),
        report_path=(report_path),
    )

    result = pd.read_parquet(output_path)

    assert "overlap.com" not in set(result["registered_domain"])

    assert len(result) == 3

    assert payload["overlap_removed"]["rows"] == 1

    assert payload["locked_test_scored"] is False

    assert payload["locked_test_labels_used"] is False

    assert payload["locked_test_urls_used"] is False

    assert report_path.exists()
