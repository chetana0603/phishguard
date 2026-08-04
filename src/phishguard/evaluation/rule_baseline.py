"""Run and report the Phase 2A rule-based validation baseline."""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from phishguard.config import (
    RULE_BASELINE_REPORT_DIR,
    VALIDATION_DATA_PATH,
    ensure_directories,
)
from phishguard.evaluation.metrics import OperationalCosts
from phishguard.evaluation.thresholds import (
    build_threshold_table,
    select_threshold_profiles,
)
from phishguard.models.rule_baseline import RULE_BASELINE_VERSION, score_url

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REQUIRED_COLUMNS = {"url_model_input", "target"}


def _validate_frame(frame: pd.DataFrame) -> None:
    missing = _REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Validation data is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Validation data is empty.")
    if not set(frame["target"].unique()).issubset({0, 1}):
        raise ValueError("Validation target must contain only 0 and 1.")


def _score_validation(frame: pd.DataFrame) -> tuple[np.ndarray, Counter[str], dict[str, float]]:
    scores: list[float] = []
    rule_counts: Counter[str] = Counter()

    start = time.perf_counter()
    for value in frame["url_model_input"]:
        prediction = score_url(value)
        scores.append(prediction.score)
        rule_counts.update(prediction.triggered_rule_names)
    elapsed_seconds = time.perf_counter() - start

    count = len(frame)
    timing = {
        "total_seconds": elapsed_seconds,
        "milliseconds_per_url": elapsed_seconds / count * 1000,
        "urls_per_second": count / elapsed_seconds if elapsed_seconds else float("inf"),
    }
    return np.asarray(scores, dtype=float), rule_counts, timing


def _save_precision_recall_curve(
    targets: np.ndarray,
    scores: np.ndarray,
    output_path: Path,
) -> None:
    precision, recall, _ = precision_recall_curve(targets, scores)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(recall, precision)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Rule baseline precision-recall curve")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _save_score_distribution(
    targets: np.ndarray,
    scores: np.ndarray,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.hist(scores[targets == 0], bins=20, alpha=0.6, label="Legitimate")
    axis.hist(scores[targets == 1], bins=20, alpha=0.6, label="Phishing")
    axis.set_xlabel("Rule risk score")
    axis.set_ylabel("Number of URLs")
    axis.set_title("Validation score distribution")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _save_threshold_tradeoff(table: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(table["threshold"], table["recall"], label="Recall")
    axis.plot(
        table["threshold"],
        table["false_positive_rate"],
        label="False-positive rate",
    )
    axis.set_xlabel("Decision threshold")
    axis.set_ylabel("Rate")
    axis.set_title("Validation threshold trade-off")
    axis.legend()
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def _render_markdown(payload: dict[str, object]) -> str:
    profiles = payload["profiles"]
    assert isinstance(profiles, dict)
    balanced = profiles["balanced"]
    assert isinstance(balanced, dict)
    balanced_metrics = balanced["metrics"]
    assert isinstance(balanced_metrics, dict)

    lines = [
        "# Phase 2A — Rule Baseline Validation Report",
        "",
        f"**Model:** `{payload['model_version']}`",
        "",
        f"**Validation rows:** {payload['validation_rows']:,}",
        "",
        "This is a transparent heuristic comparison baseline, not a production detector.",
        "The locked test set was not used.",
        "",
        "## Operating profiles selected on validation data",
        "",
        "| Profile | Threshold | Precision | Recall | FPR | F1 | Cost / 10,000 | Constraints |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for name in ("high_security", "balanced", "conservative"):
        profile = profiles[name]
        assert isinstance(profile, dict)
        metrics = profile["metrics"]
        assert isinstance(metrics, dict)
        lines.append(
            "| "
            f"{name.replace('_', ' ').title()} | {profile['threshold']:.2f} | "
            f"{_percentage(metrics['precision'])} | {_percentage(metrics['recall'])} | "
            f"{_percentage(metrics['false_positive_rate'])} | "
            f"{metrics['f1']:.4f} | "
            f"{metrics['simulated_cost_per_10000']:.2f} | "
            f"{'Met' if profile['constraints_met'] else 'Fallback'} |"
        )

    lines.extend(
        [
            "",
            "## Ranking and probability diagnostics",
            "",
            f"- Average precision: **{balanced_metrics['average_precision']:.4f}**",
            f"- ROC AUC: **{balanced_metrics['roc_auc']:.4f}**",
            f"- Brier score: **{balanced_metrics['brier_score']:.4f}**",
            "- Expected calibration error: "
            f"**{balanced_metrics['expected_calibration_error']:.4f}**",
            "",
            "The rule score is not a calibrated probability. Calibration is evaluated here only to",
            "establish how much improvement later statistical models require.",
            "",
            "## Balanced-profile confusion matrix",
            "",
            f"- True positives: {balanced_metrics['true_positive']:,}",
            f"- False negatives: {balanced_metrics['false_negative']:,}",
            f"- True negatives: {balanced_metrics['true_negative']:,}",
            f"- False positives: {balanced_metrics['false_positive']:,}",
            "",
            "## Runtime",
            "",
            f"- Total scoring time: {payload['timing']['total_seconds']:.3f} seconds",
            f"- Mean latency: {payload['timing']['milliseconds_per_url']:.4f} milliseconds per URL",
            f"- Throughput: {payload['timing']['urls_per_second']:.2f} URLs per second",
            "",
            "## Most frequently triggered rules",
            "",
            "| Rule | Trigger count |",
            "|---|---:|",
        ]
    )

    for item in payload["top_triggered_rules"]:
        lines.append(f"| `{item['rule']}` | {item['count']:,} |")

    lines.extend(
        [
            "",
            "### Profile-selection notes",
            "",
        ]
    )

    for name in ("high_security", "balanced", "conservative"):
        profile = profiles[name]
        assert isinstance(profile, dict)
        lines.append(f"- **{name.replace('_', ' ').title()}:** {profile['selection_note']}")

    lines.extend(
        [
            "",
            "## Generated figures",
            "",
            "- `precision_recall_curve.png`",
            "- `score_distribution.png`",
            "- `threshold_tradeoff.png`",
            "",
            "## Limitations",
            "",
            "- Rules are manually selected and are expected to miss novel phishing patterns.",
            "- Several suspicious patterns also occur in legitimate URLs.",
            "- The additive score is not an estimated phishing probability.",
            "- Thresholds were selected on validation data only; they are not final test results.",
            "- No URL was visited or downloaded during scoring.",
            "",
        ]
    )
    return "\n".join(lines)


def evaluate_rule_baseline(
    *,
    validation_path: Path = VALIDATION_DATA_PATH,
    output_dir: Path = RULE_BASELINE_REPORT_DIR,
) -> dict[str, object]:
    """Evaluate the rule baseline on validation data and save reproducible reports."""
    ensure_directories()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not validation_path.exists():
        raise FileNotFoundError(
            f"Validation data not found at {validation_path}. Run the Phase 1 split first."
        )

    frame = pd.read_parquet(validation_path)
    _validate_frame(frame)

    targets = frame["target"].to_numpy(dtype=np.int8)
    scores, rule_counts, timing = _score_validation(frame)
    costs = OperationalCosts(
        false_negative=25.0,
        false_positive=2.0,
    )

    threshold_table = build_threshold_table(
        targets,
        scores,
        costs=costs,
    )

    profiles = select_threshold_profiles(threshold_table)

    top_rules = [{"rule": rule, "count": int(count)} for rule, count in rule_counts.most_common(15)]
    payload: dict[str, object] = {
        "model_version": RULE_BASELINE_VERSION,
        "validation_rows": int(len(frame)),
        "target_convention": {"0": "legitimate", "1": "phishing"},
        "locked_test_used": False,
        "cost_assumptions": {
            "false_negative": costs.false_negative,
            "false_positive": costs.false_positive,
            "true_positive": costs.true_positive,
            "true_negative": costs.true_negative,
        },
        "timing": timing,
        "profiles": {name: profile.to_dict() for name, profile in profiles.items()},
        "top_triggered_rules": top_rules,
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    threshold_table.to_csv(output_dir / "threshold_table.csv", index=False)
    pd.DataFrame(top_rules).to_csv(output_dir / "rule_trigger_counts.csv", index=False)
    (output_dir / "report.md").write_text(_render_markdown(payload), encoding="utf-8")

    _save_precision_recall_curve(
        targets,
        scores,
        output_dir / "precision_recall_curve.png",
    )
    _save_score_distribution(
        targets,
        scores,
        output_dir / "score_distribution.png",
    )
    _save_threshold_tradeoff(
        threshold_table,
        output_dir / "threshold_tradeoff.png",
    )

    print(f"Evaluated {len(frame):,} validation URLs with {RULE_BASELINE_VERSION}.")
    print(f"Saved report to {output_dir / 'report.md'}")
    print(f"Balanced threshold: {profiles['balanced'].threshold:.2f}")
    return payload


def main() -> None:
    evaluate_rule_baseline()


if __name__ == "__main__":
    main()
