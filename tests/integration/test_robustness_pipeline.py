import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from phishguard.evaluation.robustness import (
    run_robustness_evaluation,
)


def test_robustness_pipeline_creates_reports(
    tmp_path: Path,
) -> None:
    training_urls = [
        "https://good-one.com/home",
        "https://good-two.com/about",
        "https://good-three.com/products",
        "https://safe-example.org/index",
        "https://company-site.net/contact",
        "https://trusted-site.org/help",
        "http://bad-login.com/verify",
        "http://fake-bank.com/login",
        "http://account-check.net/update",
        "http://verify-user.org/signin",
        "http://suspicious-site.com/confirm",
        "http://credential-check.net/auth",
    ]

    training_targets = [
        0,
        0,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        1,
        1,
    ]

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(2, 4),
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=500,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(
        training_urls,
        training_targets,
    )

    model_path = tmp_path / "model.joblib"

    joblib.dump(
        model,
        model_path,
    )

    validation = pd.DataFrame(
        {
            "url_model_input": [
                "https://good-four.com/home?a=1&b=2",
                "https://trusted-two.org/about",
                "https://normal-site.net/products",
                "https://safe-domain.com/contact",
                "http://bad-auth.com/login?a=1&b=2",
                "http://fake-login.net/verify",
                "http://credential-alert.org/update",
                "http://account-warning.com/signin",
            ],
            "target": [
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
            ],
        }
    )

    validation_path = tmp_path / "validation.parquet"

    validation.to_parquet(
        validation_path,
        index=False,
    )

    phase2b_metrics_path = tmp_path / "phase2b_metrics.json"

    phase2b_metrics_path.write_text(
        json.dumps(
            {
                "rule_baseline_fpr": 0.50,
            }
        ),
        encoding="utf-8",
    )

    report_dir = tmp_path / "robustness"

    payload = run_robustness_evaluation(
        model_path=model_path,
        validation_path=validation_path,
        phase2b_metrics_path=phase2b_metrics_path,
        report_dir=report_dir,
    )

    assert payload["locked_test_used"] is False

    assert (report_dir / "robustness_summary.csv").exists()

    assert (report_dir / "robustness_metrics.json").exists()

    assert (report_dir / "largest_drift_examples.csv").exists()

    assert (report_dir / "probability_drift.png").exists()

    assert (report_dir / "report.md").exists()
