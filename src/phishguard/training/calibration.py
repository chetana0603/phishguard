"""Phase 2C-A: probability calibration for the TF-IDF phishing model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

from phishguard.config import (
    RANDOM_STATE,
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
)
from phishguard.models.tfidf_logistic import (
    TfidfLogisticSpec,
    build_pipeline,
)

PHASE2B_METRICS_PATH = Path("reports/models/tfidf_logistic/validation_metrics.json")

CALIBRATION_REPORT_DIR = Path("reports/models/calibration")

CALIBRATION_ARTIFACT_DIR = Path("artifacts/models/calibration")


def expected_calibration_error(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_bins: int = 15,
) -> float:
    """Compute equal-width expected calibration error."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")

    targets = np.asarray(targets, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)

    if targets.shape != probabilities.shape:
        raise ValueError("targets and probabilities must have the same shape.")

    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("probabilities must lie in [0, 1].")

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


def _positive_class_scores(
    estimator: Any,
    urls: pd.Series,
) -> np.ndarray:
    """Return probabilities assigned to phishing class 1."""
    probabilities = estimator.predict_proba(urls)

    classes = np.asarray(estimator.classes_)

    phishing_indices = np.flatnonzero(classes == 1)

    if len(phishing_indices) != 1:
        raise ValueError("Expected exactly one phishing class labelled 1.")

    phishing_index = int(phishing_indices[0])

    return probabilities[:, phishing_index].astype(float)


def _load_selected_spec(
    metrics_path: Path = PHASE2B_METRICS_PATH,
) -> TfidfLogisticSpec:
    """Load the Phase 2B selected model configuration."""
    if not metrics_path.exists():
        raise FileNotFoundError(f"Phase 2B metrics not found at {metrics_path}.")

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))

    spec_payload = payload.get("best_spec")

    if not isinstance(
        spec_payload,
        dict,
    ):
        raise ValueError("Phase 2B metrics do not contain best_spec.")

    spec = TfidfLogisticSpec(**spec_payload)

    if not spec.scheme_neutral:
        raise ValueError("Phase 2C requires the scheme/www-neutral Phase 2B model.")

    return spec


def _validate_required_columns(
    frame: pd.DataFrame,
    *,
    frame_name: str,
) -> None:
    """Ensure a frame contains all columns needed by Phase 2C."""
    required = {
        "url_model_input",
        "target",
        "registered_domain",
    }

    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(f"{frame_name} data missing required columns: {sorted(missing)}")

    if frame.empty:
        raise ValueError(f"{frame_name} data is empty.")

    targets = set(frame["target"].dropna().astype(int))

    if targets != {0, 1}:
        raise ValueError(f"{frame_name} target must contain both classes 0 and 1.")


def _grouped_fit_calibration_split(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create approximately 80/20 model-fit/calibration partitions.

    Registered domains are kept entirely within one side
    of the split.
    """
    _validate_required_columns(
        frame,
        frame_name="Training",
    )

    targets = frame["target"].to_numpy(dtype=np.int8)

    groups = frame["registered_domain"].astype(str).to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fit_indices, calibration_indices = next(
        splitter.split(
            frame["url_model_input"],
            targets,
            groups=groups,
        )
    )

    fit_frame = frame.iloc[fit_indices].copy()

    calibration_frame = frame.iloc[calibration_indices].copy()

    fit_domains = set(fit_frame["registered_domain"].astype(str))

    calibration_domains = set(calibration_frame["registered_domain"].astype(str))

    overlap = fit_domains.intersection(calibration_domains)

    if overlap:
        raise RuntimeError(
            "Domain leakage detected across "
            "model-fit/calibration split: "
            f"{len(overlap)} overlapping domains."
        )

    if fit_frame["target"].nunique() != 2:
        raise RuntimeError("Model-fit partition does not contain both classes.")

    if calibration_frame["target"].nunique() != 2:
        raise RuntimeError("Calibration partition does not contain both classes.")

    return (
        fit_frame,
        calibration_frame,
    )


def _grouped_cv_splits(
    frame: pd.DataFrame,
    *,
    n_splits: int = 5,
) -> list[
    tuple[
        np.ndarray,
        np.ndarray,
    ]
]:
    """
    Build deterministic registered-domain-disjoint CV folds.

    Every registered domain remains in exactly one validation
    fold and cannot appear in the corresponding training fold.
    """
    _validate_required_columns(
        frame,
        frame_name="Training",
    )

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    targets = frame["target"].to_numpy(dtype=np.int8)

    groups = frame["registered_domain"].astype(str).to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    splits = list(
        splitter.split(
            frame["url_model_input"],
            targets,
            groups=groups,
        )
    )

    if len(splits) != n_splits:
        raise RuntimeError("Unexpected number of cross-validation folds.")

    for fold_number, (
        train_indices,
        calibration_indices,
    ) in enumerate(
        splits,
        start=1,
    ):
        train_groups = set(groups[train_indices])

        calibration_groups = set(groups[calibration_indices])

        overlap = train_groups.intersection(calibration_groups)

        if overlap:
            raise RuntimeError(
                f"Registered-domain leakage "
                f"detected in CV fold {fold_number}: "
                f"{len(overlap)} overlapping domains."
            )

        train_targets = set(targets[train_indices])

        calibration_targets = set(targets[calibration_indices])

        if train_targets != {0, 1}:
            raise RuntimeError(
                f"CV fold {fold_number} training partition does not contain both classes."
            )

        if calibration_targets != {0, 1}:
            raise RuntimeError(
                f"CV fold {fold_number} calibration partition does not contain both classes."
            )

    return splits


def _calibration_metrics(
    targets: np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    """Calculate ranking and probability-quality metrics."""
    return {
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
                labels=[0, 1],
            )
        ),
        "ece": expected_calibration_error(
            targets,
            scores,
            n_bins=15,
        ),
    }


def _save_reliability_curve(
    targets: np.ndarray,
    score_sets: dict[
        str,
        np.ndarray,
    ],
    output_path: Path,
    *,
    title: str,
) -> None:
    """Save validation reliability curves."""
    figure, axis = plt.subplots(figsize=(7, 6))

    axis.plot(
        [0.0, 1.0],
        [0.0, 1.0],
        linestyle="--",
        label="Perfect calibration",
    )

    for name, scores in score_sets.items():
        observed, predicted = calibration_curve(
            targets,
            scores,
            n_bins=15,
            strategy="uniform",
        )

        axis.plot(
            predicted,
            observed,
            marker="o",
            label=name,
        )

    axis.set_xlabel("Mean predicted phishing probability")

    axis.set_ylabel("Observed phishing frequency")

    axis.set_title(title)

    axis.legend()

    axis.grid(
        True,
        alpha=0.25,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=160,
    )

    plt.close(figure)


def _render_report(
    payload: dict[str, object],
) -> str:
    """Create a compact Phase 2C-A Markdown report."""
    split = payload["split"]
    comparison = payload["comparison"]
    final_metrics = payload["final_sigmoid_cv_metrics"]

    assert isinstance(
        split,
        dict,
    )

    assert isinstance(
        comparison,
        list,
    )

    assert isinstance(
        final_metrics,
        dict,
    )

    lines = [
        "# Phase 2C-A — Probability Calibration",
        "",
        "## Data policy",
        "",
        f"- Training rows: **{split['training_rows']:,}**",
        (f"- Initial calibration comparison model-fit rows: **{split['model_fit_rows']:,}**"),
        (f"- Initial calibration comparison calibration rows: **{split['calibration_rows']:,}**"),
        f"- Validation rows: **{split['validation_rows']:,}**",
        "- Registered-domain overlap: **0**",
        "- Locked test used: **False**",
        "",
        "## Initial calibration comparison",
        "",
        ("| Method | AP | ROC AUC | Brier | Log loss | ECE |"),
        "|---|---:|---:|---:|---:|---:|",
    ]

    for row in comparison:
        assert isinstance(
            row,
            dict,
        )

        lines.append(
            "| "
            f"{row['method']} | "
            f"{row['average_precision']:.4f} | "
            f"{row['roc_auc']:.4f} | "
            f"{row['brier_score']:.5f} | "
            f"{row['log_loss']:.5f} | "
            f"{row['ece']:.5f} |"
        )

    lines.extend(
        [
            "",
            "## Preferred calibration method",
            "",
            "- Method: **sigmoid**",
            ("- Status: **preferred pending Phase 2C-B robustness testing**"),
            ("- Calibration uses five registered-domain-disjoint folds."),
            (
                "- With `ensemble=False`, the final "
                "underlying classifier is fitted on "
                "the complete training split."
            ),
            "",
            "### Grouped-CV sigmoid validation metrics",
            "",
            (f"- Average precision: **{final_metrics['average_precision']:.4f}**"),
            (f"- ROC AUC: **{final_metrics['roc_auc']:.4f}**"),
            (f"- Brier score: **{final_metrics['brier_score']:.5f}**"),
            (f"- Log loss: **{final_metrics['log_loss']:.5f}**"),
            (f"- ECE: **{final_metrics['ece']:.5f}**"),
            "",
            "## Guardrails",
            "",
            ("- The Phase 2B scheme/www-neutral TF-IDF Logistic Regression model is used."),
            ("- Registered domains cannot cross model-fit/calibration or CV fold boundaries."),
            "- The locked test set remains untouched.",
            ("- Sigmoid remains provisional until Phase 2C-B robustness evaluation."),
            "",
        ]
    )

    return "\n".join(lines)


def run_calibration_experiment(
    *,
    train_path: Path = TRAIN_DATA_PATH,
    validation_path: Path = VALIDATION_DATA_PATH,
    report_dir: Path = CALIBRATION_REPORT_DIR,
    artifact_dir: Path = CALIBRATION_ARTIFACT_DIR,
) -> dict[str, object]:
    """
    Compare uncalibrated, sigmoid, and isotonic calibration.

    Then build a grouped-CV sigmoid candidate using all available
    training rows for the final base estimator.
    """
    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found at {train_path}.")

    if not validation_path.exists():
        raise FileNotFoundError(f"Validation data not found at {validation_path}.")

    train_frame = pd.read_parquet(train_path)

    validation_frame = pd.read_parquet(validation_path)

    _validate_required_columns(
        train_frame,
        frame_name="Training",
    )

    _validate_required_columns(
        validation_frame,
        frame_name="Validation",
    )

    spec = _load_selected_spec()

    # ---------------------------------------------------------
    # Part 1:
    # Clean held-out calibration comparison.
    # ---------------------------------------------------------

    (
        fit_frame,
        calibration_frame,
    ) = _grouped_fit_calibration_split(train_frame)

    fit_urls = fit_frame["url_model_input"].astype(str)

    fit_targets = fit_frame["target"].to_numpy(dtype=np.int8)

    calibration_urls = calibration_frame["url_model_input"].astype(str)

    calibration_targets = calibration_frame["target"].to_numpy(dtype=np.int8)

    validation_urls = validation_frame["url_model_input"].astype(str)

    validation_targets = validation_frame["target"].to_numpy(dtype=np.int8)

    # Fit Phase 2B scheme-neutral base model only
    # on the model-fit partition.
    base_pipeline = build_pipeline(spec)

    base_pipeline.fit(
        fit_urls,
        fit_targets,
    )

    uncalibrated_scores = _positive_class_scores(
        base_pipeline,
        validation_urls,
    )

    estimators: dict[
        str,
        Any,
    ] = {
        "uncalibrated": base_pipeline,
    }

    score_sets: dict[
        str,
        np.ndarray,
    ] = {
        "uncalibrated": (uncalibrated_scores),
    }

    # Both calibrators receive the exact same
    # pre-fitted base classifier and disjoint
    # calibration partition.
    for method in (
        "sigmoid",
        "isotonic",
    ):
        calibrated = CalibratedClassifierCV(
            estimator=FrozenEstimator(base_pipeline),
            method=method,
        )

        calibrated.fit(
            calibration_urls,
            calibration_targets,
        )

        scores = _positive_class_scores(
            calibrated,
            validation_urls,
        )

        estimators[method] = calibrated

        score_sets[method] = scores

    rows: list[dict[str, object]] = []

    for (
        method,
        scores,
    ) in score_sets.items():
        metrics = _calibration_metrics(
            validation_targets,
            scores,
        )

        rows.append(
            {
                "method": method,
                **metrics,
            }
        )

    comparison = pd.DataFrame(rows)

    # Primary calibration metrics are losses:
    # lower Brier/log-loss/ECE is better.
    comparison = comparison.sort_values(
        by=[
            "brier_score",
            "log_loss",
            "ece",
        ],
        ascending=[
            True,
            True,
            True,
        ],
    ).reset_index(drop=True)

    comparison.to_csv(
        report_dir / "calibration_comparison.csv",
        index=False,
    )

    _save_reliability_curve(
        validation_targets,
        score_sets,
        report_dir / "reliability_curve.png",
        title=("Phase 2C-A Validation Calibration Comparison"),
    )

    # Keep these models for auditability.
    joblib.dump(
        estimators["sigmoid"],
        artifact_dir / "sigmoid_calibrated.joblib",
        compress=3,
    )

    joblib.dump(
        estimators["isotonic"],
        artifact_dir / "isotonic_calibrated.joblib",
        compress=3,
    )

    # ---------------------------------------------------------
    # Part 2:
    # Preferred grouped-CV sigmoid candidate.
    #
    # Cross-validated predictions train the calibrator.
    # ensemble=False then gives us one final base classifier
    # trained on the complete training split.
    # ---------------------------------------------------------

    cv_splits = _grouped_cv_splits(
        train_frame,
        n_splits=5,
    )

    final_sigmoid = CalibratedClassifierCV(
        estimator=build_pipeline(spec),
        method="sigmoid",
        cv=cv_splits,
        ensemble=False,
        n_jobs=1,
    )

    final_sigmoid.fit(
        train_frame["url_model_input"].astype(str),
        train_frame["target"].to_numpy(dtype=np.int8),
    )

    final_sigmoid_scores = _positive_class_scores(
        final_sigmoid,
        validation_urls,
    )

    final_sigmoid_metrics = _calibration_metrics(
        validation_targets,
        final_sigmoid_scores,
    )

    _save_reliability_curve(
        validation_targets,
        {
            "grouped_cv_sigmoid": (final_sigmoid_scores),
        },
        report_dir / "grouped_cv_sigmoid_reliability_curve.png",
        title=("Phase 2C-A Grouped-CV Sigmoid Reliability"),
    )

    joblib.dump(
        final_sigmoid,
        artifact_dir / "sigmoid_grouped_cv.joblib",
        compress=3,
    )

    # ---------------------------------------------------------
    # Persist final Phase 2C-A metadata only AFTER
    # grouped-CV calibration has been evaluated.
    # ---------------------------------------------------------

    split_summary: dict[
        str,
        object,
    ] = {
        "training_rows": int(len(train_frame)),
        "model_fit_rows": int(len(fit_frame)),
        "calibration_rows": int(len(calibration_frame)),
        "validation_rows": int(len(validation_frame)),
        "model_fit_domains": int(fit_frame["registered_domain"].nunique()),
        "calibration_domains": int(calibration_frame["registered_domain"].nunique()),
        "domain_overlap": 0,
    }

    payload: dict[
        str,
        object,
    ] = {
        "phase": "2C-A",
        "locked_test_used": False,
        "selected_phase2b_spec": (spec.to_dict()),
        "split": split_summary,
        "comparison": (comparison.to_dict(orient="records")),
        "preferred_method": "sigmoid",
        "selection_status": ("preferred_pending_robustness"),
        "final_sigmoid_cv_metrics": (final_sigmoid_metrics),
        "final_training_rows": int(len(train_frame)),
        "calibration_cv_folds": 5,
        "calibration_cv_group": ("registered_domain"),
        "calibration_cv_domain_leakage": 0,
    }

    (report_dir / "calibration_metrics.json").write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    (artifact_dir / "model_metadata.json").write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    (report_dir / "report.md").write_text(
        _render_report(payload),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Console summary.
    # ---------------------------------------------------------

    print()
    print("Phase 2C-A calibration comparison")
    print("=" * 50)
    print(comparison.to_string(index=False))

    print()

    print(
        f"Model-fit rows: "
        f"{len(fit_frame):,} | "
        f"Calibration rows: "
        f"{len(calibration_frame):,} | "
        f"Validation rows: "
        f"{len(validation_frame):,}"
    )

    print("Registered-domain overlap: 0")

    print("Locked test used: False")

    print()

    print("Grouped-CV sigmoid candidate")

    print("=" * 50)

    for (
        metric,
        value,
    ) in final_sigmoid_metrics.items():
        print(f"{metric}: {value:.6f}")

    print(f"Final base-model training rows: {len(train_frame):,}")

    print("Calibration CV folds: 5")

    print("Group: registered_domain")

    print("Registered-domain leakage: 0")

    print("Preferred method: sigmoid (pending robustness testing)")

    print("Locked test used: False")

    return payload


def main() -> None:
    """Run Phase 2C-A calibration."""
    run_calibration_experiment()


if __name__ == "__main__":
    main()
