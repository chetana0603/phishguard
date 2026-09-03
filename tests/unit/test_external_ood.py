from pathlib import Path

import pandas as pd

from phishguard.data.external_ood import (
    build_external_candidate,
    registered_domain,
)


def test_registered_domain_from_url() -> None:
    assert registered_domain("https://login.example.com/path") == "example.com"


def test_registered_domain_from_scheme_less_domain() -> None:
    assert registered_domain("www.example.org/path") == "example.org"


def test_registered_domain_handles_ip() -> None:
    assert registered_domain("http://192.0.2.10/login") == "192.0.2.10"


def test_external_candidate_removes_development_overlap_and_conflicts(
    tmp_path: Path,
) -> None:
    train = pd.DataFrame(
        {
            "registered_domain": [
                "seen.com",
            ]
        }
    )

    validation = pd.DataFrame(
        {
            "registered_domain": [
                "known.org",
            ]
        }
    )

    train_path = tmp_path / "train.parquet"

    validation_path = tmp_path / "validation.parquet"

    train.to_parquet(
        train_path,
        index=False,
    )

    validation.to_parquet(
        validation_path,
        index=False,
    )

    phishtank = pd.DataFrame(
        {
            "phish_id": [
                "3",
                "2",
                "1",
            ],
            "url": [
                "https://freshbad.net/login",
                "https://seen.com/phish",
                "https://anotherbad.org/auth",
            ],
            "submission_time": [
                "2026-08-16T10:00:00+00:00",
                "2026-08-16T09:00:00+00:00",
                "2026-08-16T08:00:00+00:00",
            ],
            "verified": [
                "yes",
                "yes",
                "yes",
            ],
            "online": [
                "yes",
                "yes",
                "yes",
            ],
        }
    )

    phishtank_path = tmp_path / "phishtank.csv"

    phishtank.to_csv(
        phishtank_path,
        index=False,
    )

    tranco_path = tmp_path / "tranco.csv"

    tranco_path.write_text(
        "\n".join(
            [
                "1,safe-example.com",
                "2,freshbad.net",
                "3,seen.com",
                "4,clean-example.org",
            ]
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "external.parquet"

    report_dir = tmp_path / "reports"

    payload = build_external_candidate(
        phishtank_path=(phishtank_path),
        tranco_path=(tranco_path),
        tranco_list_id="TEST1",
        train_path=train_path,
        validation_path=(validation_path),
        output_path=(output_path),
        report_dir=(report_dir),
        max_per_class=10,
    )

    result = pd.read_parquet(output_path)

    assert "seen.com" not in set(result["registered_domain"])

    assert result.groupby("target").size().nunique() == 1

    assert payload["locked_test_scored"] is False

    assert (report_dir / "external_dataset_provenance.json").exists()
