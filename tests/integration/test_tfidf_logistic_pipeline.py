from pathlib import Path

import pandas as pd

from phishguard.models.tfidf_logistic import TfidfLogisticSpec
from phishguard.training.tfidf_logistic import train_tfidf_logistic


def test_tfidf_logistic_training_generates_outputs(tmp_path: Path) -> None:
    train_path = tmp_path / "train.parquet"
    validation_path = tmp_path / "validation.parquet"
    report_dir = tmp_path / "report"
    artifact_dir = tmp_path / "artifact"

    train = pd.DataFrame(
        {
            "url_model_input": [
                "https://example.com/home",
                "https://docs.python.org/3/",
                "https://shop.example/products",
                "https://university.example/admissions",
                "http://verify-login.bad/update-account",
                "http://192.168.1.10/confirm-password",
                "https://secure-billing.bad/auth",
                "http://account-recovery.bad/verify",
            ],
            "target": [0, 0, 0, 0, 1, 1, 1, 1],
        }
    )
    validation = pd.DataFrame(
        {
            "url_model_input": [
                "https://example.net/about",
                "https://docs.example.net/help",
                "http://login-check.bad/confirm",
                "http://10.0.0.1/update-account",
            ],
            "target": [0, 0, 1, 1],
        }
    )
    train.to_parquet(train_path, index=False)
    validation.to_parquet(validation_path, index=False)

    spec = TfidfLogisticSpec(
        name="integration",
        ngram_min=2,
        ngram_max=4,
        c=1.0,
        class_weight=None,
        min_df=1,
        max_features=2_000,
    )
    payload = train_tfidf_logistic(
        train_path=train_path,
        validation_path=validation_path,
        report_dir=report_dir,
        artifact_dir=artifact_dir,
        candidate_specs=[spec],
    )

    assert payload["locked_test_used"] is False
    assert (report_dir / "report.md").exists()
    assert (report_dir / "model_comparison.csv").exists()
    assert (report_dir / "threshold_table.csv").exists()
    assert (report_dir / "top_character_ngrams.csv").exists()
    assert (artifact_dir / "pipeline.joblib").exists()
    assert (artifact_dir / "model_metadata.json").exists()
