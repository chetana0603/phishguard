import numpy as np

from phishguard.evaluation.thresholds import (
    build_threshold_table,
    select_threshold_at_max_fpr,
)


def test_continuous_threshold_table_uses_distinct_scores() -> None:
    targets = np.array([0, 1, 0, 1, 1, 0], dtype=np.int8)
    scores = np.array([0.05, 0.91, 0.20, 0.80, 0.65, 0.40])

    table = build_threshold_table(targets, scores)

    assert len(table) <= len(np.unique(scores)) + 2
    assert table["threshold"].between(0.0, 1.0).all()
    assert table["true_positive"].between(0, 3).all()
    assert table["false_positive"].between(0, 3).all()


def test_select_threshold_at_max_fpr_respects_limit() -> None:
    targets = [0, 0, 0, 0, 1, 1, 1, 1]
    scores = [0.01, 0.10, 0.20, 0.70, 0.30, 0.60, 0.80, 0.95]

    table = build_threshold_table(targets, scores)
    selected = select_threshold_at_max_fpr(table, max_fpr=0.25)

    assert selected["false_positive_rate"] <= 0.25
    assert selected["recall"] > 0
