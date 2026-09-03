from pathlib import Path

from phishguard.training.calibration import run_calibration_experiment

run_calibration_experiment(
    metrics_path=Path("reports/models/tfidf_logistic_v3_rootcanon/validation_metrics.json"),
    report_dir=Path("reports/models/calibration_v3_rootcanon"),
    artifact_dir=Path("artifacts/models/calibration_v3_rootcanon"),
)
