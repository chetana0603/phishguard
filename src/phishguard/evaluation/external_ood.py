"""Phase 2C-C3: frozen external/OOD evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
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

DEFAULT_EXTERNAL_PATH = Path("data/external/processed/external_ood_eval_v3_20260903.parquet")

DEFAULT_MODEL_PATH = Path("artifacts/models/calibration_v3_rootcanon/sigmoid_grouped_cv.joblib")

DEFAULT_REPORT_DIR = Path("reports/models/external_ood_v3_rootcanon_20260903")

FROZEN_THRESHOLD = 0.768113160039295

ECE_BINS = 15


def _sha256_file(
    path: Path,
) -> str:
    """Return SHA-256 digest for a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        while True:
            chunk = file_handle.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


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


def expected_calibration_error(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int = ECE_BINS,
) -> float:
    """Calculate binary expected calibration error."""
    targets = np.asarray(
        targets,
        dtype=float,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    if len(targets) != len(probabilities):
        raise ValueError("targets and probabilities must have equal length.")

    if len(targets) == 0:
        raise ValueError("Cannot calculate ECE for an empty dataset.")

    if n_bins < 1:
        raise ValueError("n_bins must be positive.")

    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("Probabilities must lie within [0, 1].")

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1,
    )

    bin_ids = np.digitize(
        probabilities,
        edges[1:-1],
        right=False,
    )

    error = 0.0

    for bin_index in range(n_bins):
        mask = bin_ids == bin_index

        if not np.any(mask):
            continue

        confidence = float(probabilities[mask].mean())

        accuracy = float(targets[mask].mean())

        weight = float(mask.mean())

        error += weight * abs(accuracy - confidence)

    return float(error)


def _threshold_metrics(
    targets: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float | int]:
    """Calculate binary metrics at a frozen threshold."""
    predictions = (scores >= threshold).astype(np.int8)

    targets = targets.astype(np.int8)

    true_positive = int(np.sum((targets == 1) & (predictions == 1)))

    false_negative = int(np.sum((targets == 1) & (predictions == 0)))

    true_negative = int(np.sum((targets == 0) & (predictions == 0)))

    false_positive = int(np.sum((targets == 0) & (predictions == 1)))

    positive_count = true_positive + false_negative

    negative_count = true_negative + false_positive

    predicted_positive = true_positive + false_positive

    recall = true_positive / positive_count if positive_count else 0.0

    false_positive_rate = false_positive / negative_count if negative_count else 0.0

    precision = true_positive / predicted_positive if predicted_positive else 0.0

    specificity = true_negative / negative_count if negative_count else 0.0

    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    accuracy = (true_positive + true_negative) / len(targets) if len(targets) else 0.0

    return {
        "threshold": float(threshold),
        "true_positive": (true_positive),
        "false_negative": (false_negative),
        "true_negative": (true_negative),
        "false_positive": (false_positive),
        "recall_tpr": float(recall),
        "false_negative_rate": float(1.0 - recall),
        "false_positive_rate": float(false_positive_rate),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


def _score_quantiles(
    scores: np.ndarray,
) -> dict[str, float]:
    """Summarize probability distribution."""
    return {
        "min": float(np.min(scores)),
        "p05": float(
            np.quantile(
                scores,
                0.05,
            )
        ),
        "p25": float(
            np.quantile(
                scores,
                0.25,
            )
        ),
        "median": float(np.median(scores)),
        "p75": float(
            np.quantile(
                scores,
                0.75,
            )
        ),
        "p95": float(
            np.quantile(
                scores,
                0.95,
            )
        ),
        "max": float(np.max(scores)),
        "mean": float(np.mean(scores)),
    }


def _root_slash_diagnostic(
    estimator: Any,
    benign_frame: pd.DataFrame,
    baseline_scores: np.ndarray,
    *,
    threshold: float,
) -> dict[str, object]:
    """
    Measure sensitivity of synthesized Tranco roots to adding '/'.

    This is a diagnostic only and does not alter primary OOD metrics.
    """
    if benign_frame.empty:
        raise ValueError("No benign-proxy rows available for root-slash diagnostic.")

    slash_urls = benign_frame["url_model_input"].astype(str).str.rstrip("/") + "/"

    slash_scores = _positive_class_scores(
        estimator,
        slash_urls,
    )

    absolute_drift = np.abs(slash_scores - baseline_scores)

    baseline_predictions = baseline_scores >= threshold

    slash_predictions = slash_scores >= threshold

    flip_mask = baseline_predictions != slash_predictions

    baseline_false_positive_rate = float(baseline_predictions.mean())

    slash_false_positive_rate = float(slash_predictions.mean())

    return {
        "rows": int(len(benign_frame)),
        "mean_absolute_drift": float(absolute_drift.mean()),
        "median_absolute_drift": float(np.median(absolute_drift)),
        "p95_absolute_drift": float(
            np.quantile(
                absolute_drift,
                0.95,
            )
        ),
        "max_absolute_drift": float(absolute_drift.max()),
        "flip_count": int(flip_mask.sum()),
        "flip_rate": float(flip_mask.mean()),
        "baseline_fpr": (baseline_false_positive_rate),
        "with_root_slash_fpr": (slash_false_positive_rate),
        "fpr_change": float(slash_false_positive_rate - baseline_false_positive_rate),
        "interpretation": (
            "Sensitivity diagnostic only. "
            "Tranco supplies domains, not observed URL paths; "
            "adding '/' changes the synthesized representation "
            "and is not used for primary OOD metrics."
        ),
    }


def evaluate_external_ood(
    *,
    external_path: Path = DEFAULT_EXTERNAL_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    report_dir: Path = DEFAULT_REPORT_DIR,
    threshold: float = FROZEN_THRESHOLD,
) -> dict[str, object]:
    """Evaluate the frozen calibrated model on external/OOD data."""
    if not external_path.exists():
        raise FileNotFoundError(f"External OOD evaluation dataset not found at {external_path}.")

    if not model_path.exists():
        raise FileNotFoundError(f"Frozen calibrated model not found at {model_path}.")

    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must lie within [0, 1].")

    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = pd.read_parquet(external_path)

    required_columns = {
        "url_model_input",
        "target",
        "source",
        "registered_domain",
    }

    missing = required_columns.difference(frame.columns)

    if missing:
        raise ValueError(f"External dataset missing required columns: {sorted(missing)}")

    if frame.empty:
        raise ValueError("External dataset is empty.")

    target_values = set(frame["target"].dropna())

    if target_values != {
        0,
        1,
    }:
        raise ValueError("External dataset must contain both target classes 0 and 1.")

    estimator = joblib.load(model_path)

    urls = frame["url_model_input"].astype(str)

    targets = frame["target"].to_numpy(dtype=np.int8)

    scores = _positive_class_scores(
        estimator,
        urls,
    )

    if not np.all(np.isfinite(scores)):
        raise RuntimeError("Model generated non-finite probabilities.")

    threshold_metrics = _threshold_metrics(
        targets,
        scores,
        threshold=threshold,
    )

    ranking_metrics = {
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
    }

    probability_metrics = {
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
        "ece": float(
            expected_calibration_error(
                targets,
                scores,
            )
        ),
    }

    phishing_mask = targets == 1

    benign_mask = targets == 0

    phishing_scores = scores[phishing_mask]

    benign_scores = scores[benign_mask]

    phishing_predictions = phishing_scores >= threshold

    benign_predictions = benign_scores >= threshold

    phishtank_recall = float(phishing_predictions.mean())

    tranco_fpr = float(benign_predictions.mean())

    benign_frame = frame.loc[benign_mask].copy()

    root_slash_diagnostic = _root_slash_diagnostic(
        estimator,
        benign_frame,
        benign_scores,
        threshold=threshold,
    )

    source_summary: dict[
        str,
        object,
    ] = {
        "phishtank": {
            "rows": int(phishing_mask.sum()),
            "recall_tpr": (phishtank_recall),
            "false_negative_rate": float(1.0 - phishtank_recall),
            "score_quantiles": (_score_quantiles(phishing_scores)),
        },
        "tranco_benign_proxy": {
            "rows": int(benign_mask.sum()),
            "false_positive_rate": (tranco_fpr),
            "specificity": float(1.0 - tranco_fpr),
            "score_quantiles": (_score_quantiles(benign_scores)),
        },
    }

    prevalence = float(targets.mean())

    payload: dict[
        str,
        object,
    ] = {
        "phase": "2C-C3",
        "created_at_utc": (datetime.now(UTC).isoformat()),
        "evaluation_type": ("frozen external/OOD evaluation"),
        "model": {
            "path": str(model_path),
            "sha256": (_sha256_file(model_path)),
            "calibration": ("group-aware 5-fold sigmoid"),
            "model_version": ("tfidf-logistic-v3-rootcanon"),
        },
        "decision_threshold": {
            "value": float(threshold),
            "frozen_before_external_evaluation": True,
            "source": ("Phase 2C-D2 V3 validation threshold matched to rule-baseline FPR"),
            "tuned_on_external_data": False,
        },
        "external_dataset": {
            "path": str(external_path),
            "sha256": (_sha256_file(external_path)),
            "rows": int(len(frame)),
            "phishing_rows": int(phishing_mask.sum()),
            "benign_proxy_rows": int(benign_mask.sum()),
            "constructed_phishing_prevalence": (prevalence),
        },
        "ranking_metrics": (ranking_metrics),
        "threshold_metrics": (threshold_metrics),
        "probability_metrics": (probability_metrics),
        "source_specific_metrics": (source_summary),
        "tranco_root_slash_sensitivity": (root_slash_diagnostic),
        "score_quantiles": {
            "overall": (_score_quantiles(scores)),
            "phishing": (_score_quantiles(phishing_scores)),
            "benign_proxy": (_score_quantiles(benign_scores)),
        },
        "locked_test": {
            "loaded_during_c3": False,
            "urls_loaded_during_c3": False,
            "labels_loaded_during_c3": False,
            "predictions_generated_during_c3": False,
            "scored_during_c3": False,
        },
        "external_data_policy": {
            "retraining_after_results": False,
            "recalibration_after_results": False,
            "threshold_retuning_after_results": False,
        },
        "interpretation_notes": [
            (
                "PhishTank recall and Tranco false-positive rate "
                "are the primary source-specific OOD metrics."
            ),
            (
                "Precision and average precision depend on the "
                "constructed external class prevalence and should "
                "not be interpreted as deployment prevalence metrics."
            ),
            (
                "Tranco is a popular-domain benign proxy rather "
                "than definitive benign URL ground truth."
            ),
            ("Tranco URLs were synthesized as https://<domain> without a trailing slash."),
            (
                "Root-slash sensitivity is reported separately "
                "and does not alter primary OOD results."
            ),
            (
                "The external snapshot is evaluation-only. "
                "No model, calibration, or threshold changes are "
                "made after observing these results."
            ),
        ],
    }

    (report_dir / "external_ood_metrics.json").write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    predictions = pd.DataFrame(
        {
            "source": frame["source"].astype(str),
            "target": targets,
            "score": scores,
            "prediction": (scores >= threshold).astype(np.int8),
        }
    )

    predictions.to_csv(
        report_dir / "external_predictions_summary.csv",
        index=False,
    )

    print()
    print("Phase 2C-C3 frozen external/OOD evaluation")

    print("=" * 74)

    print(f"External rows: {len(frame):,}")

    print(f"PhishTank phishing rows: {phishing_mask.sum():,}")

    print(f"Tranco benign-proxy rows: {benign_mask.sum():,}")

    print()

    print(f"Frozen threshold: {threshold:.6f}")

    print()

    print("Primary external metrics")

    print("-" * 74)

    print(f"PhishTank recall / TPR: {phishtank_recall:.4%}")

    print(f"Tranco false-positive rate: {tranco_fpr:.4%}")

    print(f"Precision: {threshold_metrics['precision']:.4%}")

    print(f"F1: {threshold_metrics['f1']:.4f}")

    print(f"ROC-AUC: {ranking_metrics['roc_auc']:.4f}")

    print(f"Average precision: {ranking_metrics['average_precision']:.4f}")

    print()

    print("Probability diagnostics")

    print("-" * 74)

    print(f"Brier score: {probability_metrics['brier_score']:.6f}")

    print(f"Log loss: {probability_metrics['log_loss']:.6f}")

    print(f"ECE: {probability_metrics['ece']:.6f}")

    print()

    print("Tranco root-slash sensitivity")

    print("-" * 74)

    print(f"Baseline FPR: {root_slash_diagnostic['baseline_fpr']:.4%}")

    print(f"With root slash FPR: {root_slash_diagnostic['with_root_slash_fpr']:.4%}")

    print(f"Prediction flip rate: {root_slash_diagnostic['flip_rate']:.4%}")

    print(f"Mean probability drift: {root_slash_diagnostic['mean_absolute_drift']:.6f}")

    print()

    print("Model retrained: False")

    print("Calibration changed: False")

    print("Threshold retuned: False")

    print("Locked test scored: False")

    return payload


def main() -> None:
    """Run Phase 2C-C3."""
    parser = argparse.ArgumentParser(
        description=("Evaluate frozen PhishGuard model on external/OOD data.")
    )

    parser.add_argument(
        "--external-path",
        type=Path,
        default=DEFAULT_EXTERNAL_PATH,
    )

    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
    )

    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=FROZEN_THRESHOLD,
    )

    args = parser.parse_args()

    evaluate_external_ood(
        external_path=(args.external_path),
        model_path=(args.model_path),
        report_dir=(args.report_dir),
        threshold=(args.threshold),
    )


if __name__ == "__main__":
    main()
