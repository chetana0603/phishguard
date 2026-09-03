"""One-time final evaluation of the frozen V3 model on the locked test set."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

FREEZE_MANIFEST_PATH = Path("reports/models/final_v3_freeze_manifest.json")

TEST_PATH = Path("data/processed/test.parquet")

OUTPUT_DIR = Path("reports/models/final_locked_test_v3")

ECE_BINS = 15


def _sha256_file(
    path: Path,
) -> str:
    """Return SHA-256 digest for a file."""
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)

    return digest.hexdigest()


def _git_head() -> str:
    """Return current Git HEAD."""
    return subprocess.check_output(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()


def _positive_class_scores(
    estimator: Any,
    urls: pd.Series,
) -> np.ndarray:
    """Return probabilities assigned to phishing class 1."""
    probabilities = np.asarray(
        estimator.predict_proba(urls),
        dtype=float,
    )

    classes = np.asarray(estimator.classes_)

    phishing_indices = np.flatnonzero(classes == 1)

    if len(phishing_indices) != 1:
        raise ValueError("Expected exactly one phishing class labelled 1.")

    return probabilities[
        :,
        int(phishing_indices[0]),
    ].astype(float)


def _expected_calibration_error(
    targets: np.ndarray,
    scores: np.ndarray,
    *,
    n_bins: int = ECE_BINS,
) -> float:
    """Compute equal-width expected calibration error."""
    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    bin_ids = np.digitize(
        scores,
        edges[1:-1],
        right=False,
    )

    error = 0.0

    for bin_index in range(n_bins):
        mask = bin_ids == bin_index

        if not np.any(mask):
            continue

        confidence = float(scores[mask].mean())

        observed = float(targets[mask].mean())

        weight = float(mask.mean())

        error += weight * abs(observed - confidence)

    return float(error)


def _threshold_metrics(
    targets: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float | int]:
    """Calculate binary metrics at the frozen threshold."""
    predictions = (scores >= threshold).astype(np.int8)

    true_positive = int(np.sum((targets == 1) & (predictions == 1)))

    false_negative = int(np.sum((targets == 1) & (predictions == 0)))

    true_negative = int(np.sum((targets == 0) & (predictions == 0)))

    false_positive = int(np.sum((targets == 0) & (predictions == 1)))

    positive_count = true_positive + false_negative

    negative_count = true_negative + false_positive

    predicted_positive = true_positive + false_positive

    precision = true_positive / predicted_positive if predicted_positive else 0.0

    recall = true_positive / positive_count if positive_count else 0.0

    false_positive_rate = false_positive / negative_count if negative_count else 0.0

    specificity = true_negative / negative_count if negative_count else 0.0

    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0

    accuracy = float((predictions == targets).mean())

    return {
        "threshold": float(threshold),
        "true_positive": true_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "accuracy": accuracy,
        "precision": float(precision),
        "recall": float(recall),
        "false_positive_rate": float(false_positive_rate),
        "specificity": float(specificity),
        "f1": float(f1),
    }


def _verify_freeze_manifest(
    manifest: dict[str, object],
) -> tuple[Path, float]:
    """Verify frozen artifacts before the locked test is accessed."""
    if manifest.get("freeze_status") != "frozen_before_locked_test":
        raise RuntimeError("Freeze manifest does not have the expected status.")

    locked_test = manifest.get("locked_test")

    if not isinstance(
        locked_test,
        dict,
    ):
        raise RuntimeError("Freeze manifest has no locked_test section.")

    if locked_test.get("model_scored_before_final_evaluation"):
        raise RuntimeError("Manifest indicates that the locked test was already model-scored.")

    model = manifest.get("model")

    if not isinstance(
        model,
        dict,
    ):
        raise RuntimeError("Freeze manifest has no model section.")

    model_path = Path(str(model["path"]))

    if not model_path.exists():
        raise FileNotFoundError(f"Frozen model not found at {model_path}.")

    expected_model_hash = str(model["sha256"])

    actual_model_hash = _sha256_file(model_path)

    if actual_model_hash != expected_model_hash:
        raise RuntimeError("Frozen model SHA-256 does not match the manifest.")

    robustness = manifest.get("robustness")

    if not isinstance(
        robustness,
        dict,
    ):
        raise RuntimeError("Freeze manifest has no robustness section.")

    if not robustness.get("strict_invariance_passed"):
        raise RuntimeError("Frozen model did not pass strict robustness checks.")

    robustness_path = Path(str(robustness["path"]))

    if _sha256_file(robustness_path) != str(robustness["sha256"]):
        raise RuntimeError("Robustness report SHA-256 does not match the manifest.")

    external = manifest.get("external_evaluation")

    if not isinstance(
        external,
        dict,
    ):
        raise RuntimeError("Freeze manifest has no external_evaluation section.")

    external_metrics_path = Path(str(external["metrics_path"]))

    if _sha256_file(external_metrics_path) != str(external["metrics_sha256"]):
        raise RuntimeError("External metrics SHA-256 does not match the manifest.")

    external_dataset_path = Path(str(external["dataset_path"]))

    if _sha256_file(external_dataset_path) != str(external["dataset_sha256"]):
        raise RuntimeError("External dataset SHA-256 does not match the manifest.")

    decision_threshold = manifest.get("decision_threshold")

    if not isinstance(
        decision_threshold,
        dict,
    ):
        raise RuntimeError("Freeze manifest has no decision_threshold section.")

    threshold = float(decision_threshold["value"])

    if not 0.0 <= threshold <= 1.0:
        raise RuntimeError("Frozen threshold is outside [0, 1].")

    return (
        model_path,
        threshold,
    )


def evaluate_locked_test(
    *,
    manifest_path: Path = FREEZE_MANIFEST_PATH,
    test_path: Path = TEST_PATH,
    output_dir: Path = OUTPUT_DIR,
    dry_run: bool = False,
) -> dict[str, object] | None:
    """Verify freeze state and optionally score the locked test exactly once."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Freeze manifest not found at {manifest_path}.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    model_path, threshold = _verify_freeze_manifest(manifest)

    if not test_path.exists():
        raise FileNotFoundError(f"Locked test set not found at {test_path}.")

    print()
    print("Frozen V3 locked-test preflight")
    print("=" * 74)
    print(f"Frozen model: {model_path}")
    print(f"Frozen threshold: {threshold:.15f}")
    print(f"Freeze commit: {manifest['git_commit']}")
    print("Model hash verified: True")
    print("Robustness hash verified: True")
    print("External evaluation hashes verified: True")

    if dry_run:
        print("Locked test loaded: False")
        print("Locked test scored: False")
        print("Dry run complete.")

        return None

    # ---------------------------------------------------------
    # FINAL LOCKED-TEST ACCESS BEGINS HERE.
    # No model or threshold decisions are allowed after this run.
    # ---------------------------------------------------------

    frame = pd.read_parquet(
        test_path,
        columns=[
            "url_model_input",
            "target",
        ],
    )

    if frame.empty:
        raise ValueError("Locked test set is empty.")

    targets = frame["target"].to_numpy(dtype=np.int8)

    if set(np.unique(targets)) != {
        0,
        1,
    }:
        raise ValueError("Locked test must contain classes 0 and 1.")

    estimator = joblib.load(model_path)

    scores = _positive_class_scores(
        estimator,
        frame["url_model_input"].astype(str),
    )

    threshold_metrics = _threshold_metrics(
        targets,
        scores,
        threshold=threshold,
    )

    probability_metrics = {
        "average_precision": float(
            average_precision_score(
                targets,
                scores,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                targets,
                scores,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                targets,
                scores,
            )
        ),
        "log_loss": float(
            log_loss(
                targets,
                scores,
                labels=[
                    0,
                    1,
                ],
            )
        ),
        "ece": (
            _expected_calibration_error(
                targets,
                scores,
            )
        ),
    }

    payload: dict[
        str,
        object,
    ] = {
        "evaluation": ("final_locked_test_v3"),
        "model_version": manifest["model_version"],
        "freeze_git_commit": manifest["git_commit"],
        "evaluation_git_commit": (_git_head()),
        "freeze_manifest_path": str(manifest_path),
        "freeze_manifest_sha256": (_sha256_file(manifest_path)),
        "test_path": str(test_path),
        "test_rows": int(len(frame)),
        "decision_threshold": (threshold),
        "threshold_metrics": (threshold_metrics),
        "probability_metrics": (probability_metrics),
        "locked_test_scored": True,
        "post_test_policy": {
            "model_retraining_allowed": False,
            "recalibration_allowed": False,
            "threshold_retuning_allowed": False,
            "preprocessing_changes_allowed": False,
        },
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (output_dir / "locked_test_metrics.json").write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    predictions = pd.DataFrame(
        {
            "row_index": np.arange(len(frame)),
            "target": targets,
            "score": scores,
            "prediction": (scores >= threshold).astype(np.int8),
        }
    )

    predictions.to_csv(
        output_dir / "locked_test_predictions_summary.csv",
        index=False,
    )

    print()
    print("FINAL LOCKED-TEST EVALUATION")
    print("=" * 74)
    print(f"Rows: {len(frame):,}")
    print(f"Threshold: {threshold:.15f}")
    print()
    print(f"TP: {threshold_metrics['true_positive']:,}")
    print(f"FN: {threshold_metrics['false_negative']:,}")
    print(f"TN: {threshold_metrics['true_negative']:,}")
    print(f"FP: {threshold_metrics['false_positive']:,}")
    print()
    print(f"Accuracy: {threshold_metrics['accuracy']:.4%}")
    print(f"Precision: {threshold_metrics['precision']:.4%}")
    print(f"Recall / TPR: {threshold_metrics['recall']:.4%}")
    print(f"FPR: {threshold_metrics['false_positive_rate']:.4%}")
    print(f"Specificity: {threshold_metrics['specificity']:.4%}")
    print(f"F1: {threshold_metrics['f1']:.4f}")
    print()
    print(f"ROC-AUC: {probability_metrics['roc_auc']:.4f}")
    print(f"Average precision: {probability_metrics['average_precision']:.4f}")
    print(f"Brier score: {probability_metrics['brier_score']:.6f}")
    print(f"Log loss: {probability_metrics['log_loss']:.6f}")
    print(f"ECE: {probability_metrics['ece']:.6f}")
    print()
    print("Locked test scored: True")
    print("Model / threshold changes after this point: prohibited")

    return payload


def main() -> None:
    """Run locked-test preflight or final evaluation."""
    parser = argparse.ArgumentParser(
        description=("Evaluate the frozen V3 model on the locked internal test set.")
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=("Verify frozen artifacts without loading or scoring the locked test."),
    )

    args = parser.parse_args()

    evaluate_locked_test(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
