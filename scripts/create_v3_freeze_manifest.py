"""Create the frozen V3 manifest before locked-test evaluation."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path

import sklearn

MODEL_PATH = Path("artifacts/models/calibration_v3_rootcanon/sigmoid_grouped_cv.joblib")

ROBUSTNESS_PATH = Path("reports/models/robustness_v3_rootcanon/robustness_metrics.json")

EXTERNAL_DATA_PATH = Path("data/external/processed/external_ood_eval_v3_20260903.parquet")

EXTERNAL_METRICS_PATH = Path(
    "reports/models/external_ood_v3_rootcanon_20260903/external_ood_metrics.json"
)

OUTPUT_PATH = Path("reports/models/final_v3_freeze_manifest.json")


def sha256_file(path: Path) -> str:
    """Return SHA-256 for a file."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def git_head() -> str:
    """Return current Git HEAD."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        text=True,
    ).strip()


def main() -> None:
    """Write the frozen V3 model manifest."""
    required_paths = [
        MODEL_PATH,
        ROBUSTNESS_PATH,
        EXTERNAL_DATA_PATH,
        EXTERNAL_METRICS_PATH,
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required freeze artifact missing: {path}")

    robustness = json.loads(ROBUSTNESS_PATH.read_text(encoding="utf-8"))

    external_metrics = json.loads(EXTERNAL_METRICS_PATH.read_text(encoding="utf-8"))

    threshold = float(robustness["operating_threshold"])

    external_threshold = float(external_metrics["decision_threshold"]["value"])

    if threshold != external_threshold:
        raise RuntimeError(
            "External evaluation threshold does not match frozen robustness threshold."
        )

    if external_metrics.get(
        "locked_test_scored",
        False,
    ):
        raise RuntimeError("External report unexpectedly indicates locked-test scoring.")

    payload = {
        "manifest_version": 1,
        "model_version": ("tfidf-logistic-v3-rootcanon"),
        "freeze_status": "frozen_before_locked_test",
        "git_commit": git_head(),
        "model": {
            "path": str(MODEL_PATH),
            "sha256": sha256_file(MODEL_PATH),
            "calibration": ("5-fold registered-domain grouped-CV sigmoid"),
        },
        "decision_threshold": {
            "value": threshold,
            "source": ("Phase 2C-D2 validation threshold matched to rule-baseline FPR"),
            "tuned_on_external_data": False,
            "locked_before_test": True,
        },
        "robustness": {
            "path": str(ROBUSTNESS_PATH),
            "sha256": sha256_file(ROBUSTNESS_PATH),
            "strict_invariance_passed": robustness["strict_invariance_passed"],
        },
        "external_evaluation": {
            "dataset_path": str(EXTERNAL_DATA_PATH),
            "dataset_sha256": sha256_file(EXTERNAL_DATA_PATH),
            "metrics_path": str(EXTERNAL_METRICS_PATH),
            "metrics_sha256": sha256_file(EXTERNAL_METRICS_PATH),
            "snapshot": "2026-09-03",
            "model_retrained_after_results": False,
            "calibration_changed_after_results": False,
            "threshold_retuned_after_results": False,
        },
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
        },
        "locked_test": {
            "registered_domain_used_for_decontamination": True,
            "urls_used_before_final_evaluation": False,
            "labels_used_before_final_evaluation": False,
            "predictions_generated_before_final_evaluation": False,
            "model_scored_before_final_evaluation": False,
        },
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Freeze manifest written to: {OUTPUT_PATH}")

    print(f"Frozen Git commit: {payload['git_commit']}")

    print(f"Frozen threshold: {threshold:.15f}")

    print("Locked test model-scored: False")


if __name__ == "__main__":
    main()
