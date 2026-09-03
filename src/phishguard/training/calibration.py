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

    targets = np.asarray(
        targets,
        dtype=float,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

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


def _load_selected_spec(
    metrics_path: Path = PHASE2B_METRICS_PATH,
) -> TfidfLogisticSpec:
    """Load the selected TF-IDF model configuration."""
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


def _grouped_fit_calibration_split(
    frame: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """Create an approximately 80/20 grouped calibration split."""
    required = {
        "url_model_input",
        "target",
        "registered_domain",
    }

    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(f"Training data missing required columns: {sorted(missing)}")

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
            f"Domain leakage detected across calibration split: {len(overlap)} overlapping domains."
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
) -> list[
    tuple[
        np.ndarray,
        np.ndarray,
    ]
]:
    """Create deterministic registered-domain-disjoint CV splits."""
    required = {
        "url_model_input",
        "target",
        "registered_domain",
    }

    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(f"Training data missing required columns: {sorted(missing)}")

    targets = frame["target"].to_numpy(dtype=np.int8)

    groups = frame["registered_domain"].astype(str).to_numpy()

    splitter = StratifiedGroupKFold(
        n_splits=5,
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

    for train_indices, calibration_indices in splits:
        train_domains = set(groups[train_indices])

        calibration_domains = set(groups[calibration_indices])

        overlap = train_domains.intersection(calibration_domains)

        if overlap:
            raise RuntimeError(
                "Domain leakage detected across grouped "
                "calibration CV: "
                f"{len(overlap)} overlapping domains."
            )

        train_targets = targets[train_indices]

        calibration_targets = targets[calibration_indices]

        if np.unique(train_targets).size != 2:
            raise RuntimeError("Grouped-CV training fold does not contain both classes.")

        if np.unique(calibration_targets).size != 2:
            raise RuntimeError("Grouped-CV calibration fold does not contain both classes.")

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
                labels=[
                    0,
                    1,
                ],
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
) -> None:
    """Save validation reliability curves."""
    figure, axis = plt.subplots(
        figsize=(
            7,
            6,
        )
    )

    axis.plot(
        [
            0.0,
            1.0,
        ],
        [
            0.0,
            1.0,
        ],
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

    axis.set_title("Phase 2C-A Validation Reliability")

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


def run_calibration_experiment(
    *,
    metrics_path: Path = PHASE2B_METRICS_PATH,
    train_path: Path = TRAIN_DATA_PATH,
    validation_path: Path = VALIDATION_DATA_PATH,
    report_dir: Path = CALIBRATION_REPORT_DIR,
    artifact_dir: Path = CALIBRATION_ARTIFACT_DIR,
) -> dict[str, object]:
    """
    Compare calibration approaches and build the grouped-CV
    sigmoid candidate used for robustness evaluation.
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

    spec = _load_selected_spec(metrics_path)

    fit_frame, calibration_frame = _grouped_fit_calibration_split(train_frame)

    fit_urls = fit_frame["url_model_input"].astype(str)

    fit_targets = fit_frame["target"].to_numpy(dtype=np.int8)

    calibration_urls = calibration_frame["url_model_input"].astype(str)

    calibration_targets = calibration_frame["target"].to_numpy(dtype=np.int8)

    validation_urls = validation_frame["url_model_input"].astype(str)

    validation_targets = validation_frame["target"].to_numpy(dtype=np.int8)

    # ---------------------------------------------------------
    # Stage 1:
    # Diagnostic 80/20 grouped holdout calibration comparison.
    # ---------------------------------------------------------

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

    for method, scores in score_sets.items():
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
    )

    split_summary = {
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
        "comparison": comparison.to_dict(orient="records"),
        "selection_status": ("pending_grouped_cv"),
    }

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

    # ---------------------------------------------------------
    # Stage 2:
    # Final 5-fold grouped-CV sigmoid calibration.
    #
    # Sigmoid is kept as the predetermined method from V2.
    # V3 is not selecting a new calibration method from the
    # external/OOD results.
    # ---------------------------------------------------------

    print()
    print("Training grouped-CV sigmoid candidate...")

    cv_splits = _grouped_cv_splits(train_frame)

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

    payload["preferred_method"] = "sigmoid"

    payload["selection_status"] = "preferred_pending_robustness"

    payload["final_sigmoid_cv_metrics"] = final_sigmoid_metrics

    payload["final_training_rows"] = int(len(train_frame))

    payload["calibration_cv_folds"] = 5

    payload["calibration_cv_group"] = "registered_domain"

    payload["locked_test_used"] = False

    joblib.dump(
        final_sigmoid,
        artifact_dir / "sigmoid_grouped_cv.joblib",
        compress=3,
    )

    (report_dir / "calibration_metrics.json").write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("Grouped-CV sigmoid candidate")
    print("=" * 50)

    for metric, value in final_sigmoid_metrics.items():
        print(f"{metric}: {value:.6f}")

    print(f"Final base-model training rows: {len(train_frame):,}")

    print("Calibration CV folds: 5")

    print("Group: registered_domain")

    print("Locked test used: False")

    return payload


def main() -> None:
    """Run the default calibration experiment."""
    run_calibration_experiment()


if __name__ == "__main__":
    main()
