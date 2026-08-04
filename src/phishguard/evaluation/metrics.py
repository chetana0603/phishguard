"""Shared binary-classification and operational-impact metrics."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class OperationalCosts:
    """Relative costs used only for historical impact simulation."""

    false_negative: float = 25.0
    false_positive: float = 2.0
    true_positive: float = 0.0
    true_negative: float = 0.0


@dataclass(frozen=True)
class BinaryMetrics:
    """Threshold-dependent and ranking metrics for binary scores."""

    threshold: float
    n_samples: int
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    precision: float
    recall: float
    specificity: float
    false_positive_rate: float
    f1: float
    accuracy: float
    predicted_positive_rate: float
    average_precision: float
    roc_auc: float
    brier_score: float
    expected_calibration_error: float
    simulated_cost: float
    simulated_cost_per_10000: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _as_binary_array(values: Iterable[int]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.int8)
    if array.ndim != 1:
        raise ValueError("Expected a one-dimensional target array.")
    if not set(np.unique(array)).issubset({0, 1}):
        raise ValueError("Targets must contain only 0 and 1.")
    return array


def _as_score_array(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1:
        raise ValueError("Expected a one-dimensional score array.")
    if not np.isfinite(array).all():
        raise ValueError("Scores must be finite.")
    if ((array < 0.0) | (array > 1.0)).any():
        raise ValueError("Scores must be between 0 and 1.")
    return array


def expected_calibration_error(
    y_true: Iterable[int],
    y_score: Iterable[float],
    *,
    n_bins: int = 10,
) -> float:
    """Compute equal-width expected calibration error for binary probabilities."""
    targets = _as_binary_array(y_true)
    scores = _as_score_array(y_score)
    if len(targets) != len(scores):
        raise ValueError("Targets and scores must have equal length.")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")
    if len(targets) == 0:
        raise ValueError("Cannot evaluate empty arrays.")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(scores, edges[1:-1], right=True)

    ece = 0.0
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        if not mask.any():
            continue
        confidence = float(scores[mask].mean())
        observed_rate = float(targets[mask].mean())
        ece += float(mask.mean()) * abs(observed_rate - confidence)
    return float(ece)


def evaluate_binary_scores(
    y_true: Iterable[int],
    y_score: Iterable[float],
    *,
    threshold: float,
    costs: OperationalCosts | None = None,
) -> BinaryMetrics:
    """Evaluate continuous scores at one decision threshold."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    targets = _as_binary_array(y_true)
    scores = _as_score_array(y_score)
    if len(targets) != len(scores):
        raise ValueError("Targets and scores must have equal length.")
    if len(targets) == 0:
        raise ValueError("Cannot evaluate empty arrays.")

    predictions = (scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()

    specificity = tn / (tn + fp) if tn + fp else 0.0
    false_positive_rate = fp / (tn + fp) if tn + fp else 0.0

    unique_targets = np.unique(targets)
    if len(unique_targets) == 2:
        average_precision = float(average_precision_score(targets, scores))
        roc_auc = float(roc_auc_score(targets, scores))
    else:
        average_precision = float("nan")
        roc_auc = float("nan")

    applied_costs = costs or OperationalCosts()
    simulated_cost = (
        tp * applied_costs.true_positive
        + fp * applied_costs.false_positive
        + tn * applied_costs.true_negative
        + fn * applied_costs.false_negative
    )

    return BinaryMetrics(
        threshold=float(threshold),
        n_samples=int(len(targets)),
        true_positive=int(tp),
        false_positive=int(fp),
        true_negative=int(tn),
        false_negative=int(fn),
        precision=float(precision_score(targets, predictions, zero_division=0)),
        recall=float(recall_score(targets, predictions, zero_division=0)),
        specificity=float(specificity),
        false_positive_rate=float(false_positive_rate),
        f1=float(f1_score(targets, predictions, zero_division=0)),
        accuracy=float(accuracy_score(targets, predictions)),
        predicted_positive_rate=float(predictions.mean()),
        average_precision=average_precision,
        roc_auc=roc_auc,
        brier_score=float(brier_score_loss(targets, scores)),
        expected_calibration_error=expected_calibration_error(targets, scores),
        simulated_cost=float(simulated_cost),
        simulated_cost_per_10000=float(simulated_cost / len(targets) * 10000),
    )
