"""Validation-only threshold search for security operating profiles."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from phishguard.evaluation.metrics import OperationalCosts, evaluate_binary_scores


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


def _candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    fixed_grid = np.linspace(0.0, 1.0, 101)
    observed = np.unique(np.round(scores, 6))
    return np.unique(np.concatenate((fixed_grid, observed)))


def build_threshold_table(
    y_true: Iterable[int],
    y_score: Iterable[float],
    *,
    costs: OperationalCosts | None = None,
) -> pd.DataFrame:
    """Evaluate all useful validation thresholds."""
    targets = np.asarray(list(y_true), dtype=np.int8)
    scores = np.asarray(list(y_score), dtype=float)
    if len(targets) != len(scores):
        raise ValueError("Targets and scores must have equal length.")

    rows = [
        evaluate_binary_scores(
            targets,
            scores,
            threshold=float(threshold),
            costs=costs,
        ).to_dict()
        for threshold in _candidate_thresholds(scores)
    ]
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def _row_to_profile(
    row: pd.Series,
    *,
    name: str,
    description: str,
    constraints_met: bool,
    selection_note: str,
) -> ThresholdProfile:
    metrics = {
        key: (
            int(value)
            if key
            in {
                "n_samples",
                "true_positive",
                "false_positive",
                "true_negative",
                "false_negative",
            }
            else float(value)
        )
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


def select_threshold_profiles(
    table: pd.DataFrame,
) -> dict[str, ThresholdProfile]:
    """Select constrained security operating profiles."""
    if table.empty:
        raise ValueError("Threshold table cannot be empty.")

    usable = table[table["predicted_positive_rate"] > 0].copy()

    if usable.empty:
        usable = table.copy()

    # High Security:
    # maximise recall while FPR <= 5% and precision >= 70%.
    high_security_candidates = usable[
        (usable["false_positive_rate"] <= 0.05) & (usable["precision"] >= 0.70)
    ].copy()

    high_security_constraints_met = not high_security_candidates.empty

    if high_security_constraints_met:
        high_security_row = high_security_candidates.sort_values(
            [
                "recall",
                "false_positive_rate",
                "precision",
                "threshold",
            ],
            ascending=[False, True, False, True],
        ).iloc[0]

        high_security_note = "Selected from thresholds satisfying FPR <= 5% and precision >= 70%."
    else:
        fallback = usable.copy()
        fallback["_constraint_violation"] = _constraint_violation(
            fallback,
            max_fpr=0.05,
            min_precision=0.70,
        )

        high_security_row = fallback.sort_values(
            [
                "_constraint_violation",
                "recall",
                "false_positive_rate",
                "precision",
            ],
            ascending=[True, False, True, False],
        ).iloc[0]

        high_security_note = (
            "No threshold satisfied all constraints; selected the "
            "threshold with the smallest total constraint violation."
        )

    # Balanced:
    # minimise simulated cost while FPR <= 5%,
    # precision >= 70%, and recall >= 50%.
    balanced_candidates = usable[
        (usable["false_positive_rate"] <= 0.05)
        & (usable["precision"] >= 0.70)
        & (usable["recall"] >= 0.50)
    ].copy()

    balanced_constraints_met = not balanced_candidates.empty

    if balanced_constraints_met:
        balanced_row = balanced_candidates.sort_values(
            [
                "simulated_cost",
                "recall",
                "false_positive_rate",
                "precision",
                "threshold",
            ],
            ascending=[True, False, True, False, True],
        ).iloc[0]

        balanced_note = (
            "Selected by minimum simulated cost among thresholds "
            "satisfying FPR <= 5%, precision >= 70%, and recall >= 50%."
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
            [
                "_constraint_violation",
                "simulated_cost",
                "recall",
                "false_positive_rate",
            ],
            ascending=[True, True, False, True],
        ).iloc[0]

        balanced_note = (
            "No threshold satisfied all constraints; selected the "
            "threshold with the smallest total constraint violation."
        )

    # Conservative:
    # maximise precision while FPR <= 1% and recall >= 10%.
    conservative_candidates = usable[
        (usable["false_positive_rate"] <= 0.01) & (usable["recall"] >= 0.10)
    ].copy()

    conservative_constraints_met = not conservative_candidates.empty

    if conservative_constraints_met:
        conservative_row = conservative_candidates.sort_values(
            [
                "precision",
                "recall",
                "false_positive_rate",
                "threshold",
            ],
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
            [
                "_constraint_violation",
                "precision",
                "recall",
                "false_positive_rate",
            ],
            ascending=[True, False, False, True],
        ).iloc[0]

        conservative_note = (
            "No threshold satisfied all constraints; selected the "
            "threshold with the smallest total constraint violation."
        )

    return {
        "high_security": _row_to_profile(
            high_security_row,
            name="high_security",
            description=(
                "Maximise phishing recall while keeping FPR at or "
                "below 5% and precision at or above 70%."
            ),
            constraints_met=high_security_constraints_met,
            selection_note=high_security_note,
        ),
        "balanced": _row_to_profile(
            balanced_row,
            name="balanced",
            description=(
                "Minimise simulated cost while keeping FPR at or "
                "below 5%, precision at or above 70%, and recall "
                "at or above 50%."
            ),
            constraints_met=balanced_constraints_met,
            selection_note=balanced_note,
        ),
        "conservative": _row_to_profile(
            conservative_row,
            name="conservative",
            description=(
                "Maximise precision while keeping FPR at or below 1% and recall at or above 10%."
            ),
            constraints_met=conservative_constraints_met,
            selection_note=conservative_note,
        ),
    }
