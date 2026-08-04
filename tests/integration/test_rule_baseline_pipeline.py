from pathlib import Path

import pandas as pd

from phishguard.evaluation.rule_baseline import evaluate_rule_baseline


def test_rule_baseline_pipeline_generates_reports(tmp_path: Path) -> None:
    validation_path = tmp_path / "validation.parquet"
    output_dir = tmp_path / "rule_report"

    frame = pd.DataFrame(
        {
            "url_model_input": [
                "https://example.com/about",
                "https://docs.python.org/3/",
                "http://192.168.1.10/login",
                "http://secure-login.example/verify/account",
                "https://bit.ly/update-account",
                "https://shop.example/products?id=10",
            ],
            "target": [0, 0, 1, 1, 1, 0],
        }
    )
    frame.to_parquet(validation_path, index=False)

    payload = evaluate_rule_baseline(
        validation_path=validation_path,
        output_dir=output_dir,
    )

    assert payload["locked_test_used"] is False
    assert (output_dir / "report.md").exists()
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "threshold_table.csv").exists()
    assert (output_dir / "precision_recall_curve.png").exists()
