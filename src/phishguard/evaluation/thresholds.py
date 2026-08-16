"""Validation-only threshold search for security operating profiles."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from phishguard.evaluation.metrics import OperationalCosts, expected_calibration_error

_COUNT_COLUMNS = {
    "n_samples",
    "true_positive",
    "false_positive",
    "true_negative",
    "false_negative",
}


@dataclass(frozen=True)
class ThresholdProfile:
    """Selected operating threshold and its validation metrics."""

    name: str
    description: str
    threshold: float
    metrics: dict[str, int | float]
    constraints_met: bool
    selection_note: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "threshold": self.threshold,
            "metrics": self.metrics,
            "constraints_met": self.constraints_met,
            "selection_note": self.selection_note,
        }


def _validate_inputs(
    y_true: Iterable[int],
    y_score: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    targets = np.asarray(list(y_true), dtype=np.int8)
    scores = np.asarray(list(y_score), dtype=float)

    if targets.ndim != 1 or scores.ndim != 1:
        raise ValueError("Targets and scores must be one-dimensional.")
    if len(targets) != len(scores):
        raise ValueError("Targets and scores must have equal length.")
    if len(targets) == 0:
        raise ValueError("Cannot evaluate empty arrays.")
    if not set(np.unique(targets)).issubset({0, 1}):
        raise ValueError("Targets must contain only 0 and 1.")
    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite.")
    if ((scores < 0.0) | (scores > 1.0)).any():
        raise ValueError("Scores must be between 0 and 1.")

    return targets, scores


def build_threshold_table(
    y_true: Iterable[int],
    y_score: Iterable[float],
    *,
    costs: OperationalCosts | None = None,
) -> pd.DataFrame:
    """Evaluate every distinct score threshold in O(n log n) time.

    The Phase 2A rule scores had few distinct values. Statistical models produce
    nearly one unique probability per URL, so repeatedly evaluating every sample
    at every threshold would be quadratic. This implementation sorts once and
    derives all confusion matrices cumulatively.
    """
    targets, scores = _validate_inputs(y_true, y_score)
    applied_costs = costs or OperationalCosts()

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_targets = targets[order]

    cumulative_positive = np.cumsum(sorted_targets, dtype=np.int64)
    cumulative_negative = np.cumsum(1 - sorted_targets, dtype=np.int64)

    group_ends = np.flatnonzero(np.r_[sorted_scores[:-1] != sorted_scores[1:], True])
    thresholds = sorted_scores[group_ends]
    true_positive = cumulative_positive[group_ends]
    false_positive = cumulative_negative[group_ends]

    total_positive = int(sorted_targets.sum())
    total_negative = int(len(sorted_targets) - total_positive)
    false_negative = total_positive - true_positive
    true_negative = total_negative - false_positive

    # Include boundary thresholds for explicit all-negative/all-positive states.
    boundary_thresholds = np.array([1.0, 0.0], dtype=float)
    boundary_tp = np.array(
        [int((scores >= 1.0).dot(targets)), total_positive],
        dtype=np.int64,
    )
    boundary_fp = np.array(
        [int((scores >= 1.0).dot(1 - targets)), total_negative],
        dtype=np.int64,
    )

    thresholds = np.concatenate((thresholds, boundary_thresholds))
    true_positive = np.concatenate((true_positive, boundary_tp))
    false_positive = np.concatenate((false_positive, boundary_fp))
    false_negative = total_positive - true_positive
    true_negative = total_negative - false_positive

    frame = pd.DataFrame(
        {
            "threshold": thresholds,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
        }
    ).drop_duplicates(subset=["threshold"], keep="first")

    tp = frame["true_positive"].to_numpy(dtype=float)
    fp = frame["false_positive"].to_numpy(dtype=float)
    tn = frame["true_negative"].to_numpy(dtype=float)
    fn = frame["false_negative"].to_numpy(dtype=float)

    predicted_positive = tp + fp
    actual_positive = tp + fn
    actual_negative = tn + fp

    frame["n_samples"] = len(targets)
    frame["precision"] = np.divide(
        tp,
        predicted_positive,
        out=np.zeros_like(tp),
        where=predicted_positive > 0,
    )
    frame["recall"] = np.divide(
        tp,
        actual_positive,
        out=np.zeros_like(tp),
        where=actual_positive > 0,
    )
    frame["specificity"] = np.divide(
        tn,
        actual_negative,
        out=np.zeros_like(tn),
        where=actual_negative > 0,
    )
    frame["false_positive_rate"] = np.divide(
        fp,
        actual_negative,
        out=np.zeros_like(fp),
        where=actual_negative > 0,
    )
    frame["f1"] = np.divide(
        2 * frame["precision"] * frame["recall"],
        frame["precision"] + frame["recall"],
        out=np.zeros(len(frame), dtype=float),
        where=(frame["precision"] + frame["recall"]) > 0,
    )
    frame["accuracy"] = (tp + tn) / len(targets)
    frame["predicted_positive_rate"] = predicted_positive / len(targets)

    if len(np.unique(targets)) == 2:
        average_precision = float(average_precision_score(targets, scores))
        roc_auc = float(roc_auc_score(targets, scores))
    else:
        average_precision = float("nan")
        roc_auc = float("nan")

    frame["average_precision"] = average_precision
    frame["roc_auc"] = roc_auc
    frame["brier_score"] = float(brier_score_loss(targets, scores))
    frame["expected_calibration_error"] = expected_calibration_error(targets, scores)

    simulated_cost = (
        tp * applied_costs.true_positive
        + fp * applied_costs.false_positive
        + tn * applied_costs.true_negative
        + fn * applied_costs.false_negative
    )
    frame["simulated_cost"] = simulated_cost
    frame["simulated_cost_per_10000"] = simulated_cost / len(targets) * 10000

    for column in _COUNT_COLUMNS:
        frame[column] = frame[column].astype(int)

    return frame.sort_values("threshold").reset_index(drop=True)


def select_threshold_at_max_fpr(
    table: pd.DataFrame,
    *,
    max_fpr: float,
) -> pd.Series:
    """Select maximum recall while keeping FPR at or below a fixed limit."""
    if not 0.0 <= max_fpr <= 1.0:
        raise ValueError("max_fpr must be between 0 and 1.")
    if table.empty:
        raise ValueError("Threshold table cannot be empty.")

    candidates = table[
        (table["predicted_positive_rate"] > 0) & (table["false_positive_rate"] <= max_fpr)
    ]
    if candidates.empty:
        candidates = table[table["predicted_positive_rate"] > 0]
    if candidates.empty:
        candidates = table

    return candidates.sort_values(
        ["recall", "precision", "false_positive_rate", "threshold"],
        ascending=[False, False, True, False],
    ).iloc[0]


def _row_to_profile(
    row: pd.Series,
    *,
    name: str,
    description: str,
    constraints_met: bool,
    selection_note: str,
) -> ThresholdProfile:
    metrics = {
        key: int(value) if key in _COUNT_COLUMNS else float(value)
        for key, value in row.to_dict().items()
        if not key.startswith("_")
    }
    return ThresholdProfile(
        name=name,
        description=description,
        threshold=float(row["threshold"]),
        metrics=metrics,
        constraints_met=constraints_met,
        selection_note=selection_note,
    )


def _constraint_violation(
    table: pd.DataFrame,
    *,
    max_fpr: float | None = None,
    min_precision: float | None = None,
    min_recall: float | None = None,
) -> pd.Series:
    """Calculate how far each threshold is from satisfying constraints."""
    violation = pd.Series(0.0, index=table.index)

    if max_fpr is not None:
        violation += (table["false_positive_rate"] - max_fpr).clip(lower=0.0)
    if min_precision is not None:
        violation += (min_precision - table["precision"]).clip(lower=0.0)
    if min_recall is not None:
        violation += (min_recall - table["recall"]).clip(lower=0.0)

    return violation


def select_threshold_profiles(table: pd.DataFrame) -> dict[str, ThresholdProfile]:
    """Select constrained high-security, balanced, and conservative profiles."""
    if table.empty:
        raise ValueError("Threshold table cannot be empty.")

    usable = table[table["predicted_positive_rate"] > 0].copy()
    if usable.empty:
        usable = table.copy()

    high_candidates = usable[
        (usable["false_positive_rate"] <= 0.05) & (usable["precision"] >= 0.70)
    ]
    high_met = not high_candidates.empty
    if high_met:
        high_row = high_candidates.sort_values(
            ["recall", "false_positive_rate", "precision", "threshold"],
            ascending=[False, True, False, True],
        ).iloc[0]
        high_note = "Selected from thresholds satisfying FPR <= 5% and precision >= 70%."
    else:
        fallback = usable.copy()
        fallback["_constraint_violation"] = _constraint_violation(
            fallback,
            max_fpr=0.05,
            min_precision=0.70,
        )
        high_row = fallback.sort_values(
            ["_constraint_violation", "recall", "false_positive_rate", "precision"],
            ascending=[True, False, True, False],
        ).iloc[0]
        high_note = (
            "No threshold satisfied all constraints; selected the smallest total "
            "constraint violation."
        )

    balanced_candidates = usable[
        (usable["false_positive_rate"] <= 0.05)
        & (usable["precision"] >= 0.70)
        & (usable["recall"] >= 0.50)
    ]
    balanced_met = not balanced_candidates.empty
    if balanced_met:
        balanced_row = balanced_candidates.sort_values(
            ["simulated_cost", "recall", "false_positive_rate", "precision", "threshold"],
            ascending=[True, False, True, False, True],
        ).iloc[0]
        balanced_note = (
            "Selected by minimum simulated cost among thresholds satisfying FPR <= 5%, "
            "precision >= 70%, and recall >= 50%."
        )
    else:
        fallback = usable.copy()
        fallback["_constraint_violation"] = _constraint_violation(
            fallback,
            max_fpr=0.05,
            min_precision=0.70,
            min_recall=0.50,
        )
        balanced_row = fallback.sort_values(
            ["_constraint_violation", "simulated_cost", "recall", "false_positive_rate"],
            ascending=[True, True, False, True],
        ).iloc[0]
        balanced_note = (
            "No threshold satisfied all constraints; selected the smallest total "
            "constraint violation."
        )

    conservative_candidates = usable[
        (usable["false_positive_rate"] <= 0.01) & (usable["recall"] >= 0.10)
    ]
    conservative_met = not conservative_candidates.empty
    if conservative_met:
        conservative_row = conservative_candidates.sort_values(
            ["precision", "recall", "false_positive_rate", "threshold"],
            ascending=[False, False, True, False],
        ).iloc[0]
        conservative_note = "Selected from thresholds satisfying FPR <= 1% and recall >= 10%."
    else:
        fallback = usable.copy()
        fallback["_constraint_violation"] = _constraint_violation(
            fallback,
            max_fpr=0.01,
            min_recall=0.10,
        )
        conservative_row = fallback.sort_values(
            ["_constraint_violation", "precision", "recall", "false_positive_rate"],
            ascending=[True, False, False, True],
        ).iloc[0]
        conservative_note = (
            "No threshold satisfied all constraints; selected the smallest total "
            "constraint violation."
        )

    return {
        "high_security": _row_to_profile(
            high_row,
            name="high_security",
            description=(
                "Maximise phishing recall while keeping FPR at or below 5% and "
                "precision at or above 70%."
            ),
            constraints_met=high_met,
            selection_note=high_note,
        ),
        "balanced": _row_to_profile(
            balanced_row,
            name="balanced",
            description=(
                "Minimise simulated cost while keeping FPR at or below 5%, precision "
                "at or above 70%, and recall at or above 50%."
            ),
            constraints_met=balanced_met,
            selection_note=balanced_note,
        ),
        "conservative": _row_to_profile(
            conservative_row,
            name="conservative",
            description=(
                "Maximise precision while keeping FPR at or below 1% and recall at or above 10%."
            ),
            constraints_met=conservative_met,
            selection_note=conservative_note,
        ),
    }
