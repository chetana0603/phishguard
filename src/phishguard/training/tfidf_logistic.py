"""Train, select, evaluate, and persist the Phase 2B URL model."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from phishguard.config import (
    RULE_BASELINE_REPORT_DIR,
    TFIDF_LOGISTIC_ARTIFACT_DIR,
    TFIDF_LOGISTIC_REPORT_DIR,
    TRAIN_DATA_PATH,
    VALIDATION_DATA_PATH,
    ensure_directories,
)
from phishguard.evaluation.metrics import OperationalCosts, evaluate_binary_scores
from phishguard.evaluation.thresholds import (
    build_threshold_table,
    select_threshold_at_max_fpr,
    select_threshold_profiles,
)
from phishguard.models.tfidf_logistic import (
    TFIDF_LOGISTIC_VERSION,
    TfidfLogisticSpec,
    build_classifier,
    build_pipeline,
    build_vectorizer,
    compact_candidate_specs,
    full_candidate_specs,
    phishing_scores,
    top_character_ngrams,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REQUIRED_COLUMNS = {"url_model_input", "target"}
_DEFAULT_RULE_FPR = 0.0163


def _validate_frame(frame: pd.DataFrame, *, split_name: str) -> None:
    missing = _REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"{split_name} data is missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{split_name} data is empty.")
    if frame["url_model_input"].isna().any():
        raise ValueError(f"{split_name} data contains missing model URLs.")
    if not set(frame["target"].unique()).issubset({0, 1}):
        raise ValueError(f"{split_name} target must contain only 0 and 1.")


def _load_rule_baseline_fpr(path: Path) -> tuple[float, dict[str, object] | None]:
    metrics_path = path / "metrics.json"
    if not metrics_path.exists():
        return _DEFAULT_RULE_FPR, None

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    profiles = payload.get("profiles", {})
    balanced = profiles.get("balanced", {})
    metrics = balanced.get("metrics", {})
    fpr = float(metrics.get("false_positive_rate", _DEFAULT_RULE_FPR))
    return fpr, payload


def _protocol_only_scores(urls: pd.Series) -> np.ndarray:
    """Return 1 for plain-HTTP URLs and 0 otherwise."""
    return (
        urls.astype(str).str.strip().str.lower().str.startswith("http://").astype(float).to_numpy()
    )


def _model_comparison_row(
    *,
    spec: TfidfLogisticSpec,
    feature_count: int,
    vectorizer_seconds: float,
    fit_seconds: float,
    scoring_seconds: float,
    targets: np.ndarray,
    scores: np.ndarray,
    matched_fpr: float,
    costs: OperationalCosts,
) -> dict[str, object]:
    default_metrics = evaluate_binary_scores(
        targets,
        scores,
        threshold=0.5,
        costs=costs,
    )
    threshold_table = build_threshold_table(targets, scores, costs=costs)
    matched_row = select_threshold_at_max_fpr(threshold_table, max_fpr=matched_fpr)

    return {
        **spec.to_dict(),
        "feature_count": feature_count,
        "vectorizer_seconds": vectorizer_seconds,
        "fit_seconds": fit_seconds,
        "scoring_seconds": scoring_seconds,
        "average_precision": default_metrics.average_precision,
        "roc_auc": default_metrics.roc_auc,
        "brier_score": default_metrics.brier_score,
        "expected_calibration_error": default_metrics.expected_calibration_error,
        "default_precision": default_metrics.precision,
        "default_recall": default_metrics.recall,
        "default_f1": default_metrics.f1,
        "default_fpr": default_metrics.false_positive_rate,
        "matched_fpr_limit": matched_fpr,
        "matched_threshold": float(matched_row["threshold"]),
        "matched_precision": float(matched_row["precision"]),
        "matched_recall": float(matched_row["recall"]),
        "matched_f1": float(matched_row["f1"]),
        "matched_actual_fpr": float(matched_row["false_positive_rate"]),
    }


def _search_candidates(
    train_urls: pd.Series,
    train_targets: np.ndarray,
    validation_urls: pd.Series,
    validation_targets: np.ndarray,
    specs: Iterable[TfidfLogisticSpec],
    *,
    matched_fpr: float,
    costs: OperationalCosts,
) -> pd.DataFrame:
    grouped: dict[tuple[object, ...], list[TfidfLogisticSpec]] = defaultdict(list)
    for spec in specs:
        grouped[spec.vectorizer_key].append(spec)

    rows: list[dict[str, object]] = []
    for group_specs in grouped.values():
        vectorizer_spec = group_specs[0]
        vectorizer = build_vectorizer(vectorizer_spec)

        vectorizer_start = time.perf_counter()
        train_matrix = vectorizer.fit_transform(train_urls)
        validation_matrix = vectorizer.transform(validation_urls)
        vectorizer_seconds = time.perf_counter() - vectorizer_start

        for spec in group_specs:
            classifier = build_classifier(spec)
            fit_start = time.perf_counter()
            classifier.fit(train_matrix, train_targets)
            fit_seconds = time.perf_counter() - fit_start

            scoring_start = time.perf_counter()
            scores = classifier.predict_proba(validation_matrix)[:, 1]
            scoring_seconds = time.perf_counter() - scoring_start

            rows.append(
                _model_comparison_row(
                    spec=spec,
                    feature_count=int(train_matrix.shape[1]),
                    vectorizer_seconds=vectorizer_seconds,
                    fit_seconds=fit_seconds,
                    scoring_seconds=scoring_seconds,
                    targets=validation_targets,
                    scores=scores,
                    matched_fpr=matched_fpr,
                    costs=costs,
                )
            )

    return (
        pd.DataFrame(rows)
        .sort_values(
            [
                "average_precision",
                "matched_recall",
                "roc_auc",
                "expected_calibration_error",
            ],
            ascending=[False, False, False, True],
        )
        .reset_index(drop=True)
    )


def _spec_from_row(row: pd.Series) -> TfidfLogisticSpec:
    class_weight = row["class_weight"]
    if pd.isna(class_weight):
        class_weight = None
    return TfidfLogisticSpec(
        name=str(row["name"]),
        ngram_min=int(row["ngram_min"]),
        ngram_max=int(row["ngram_max"]),
        c=float(row["c"]),
        class_weight=class_weight,
        min_df=int(row["min_df"]),
        max_features=int(row["max_features"]),
        sublinear_tf=bool(row["sublinear_tf"]),
        lowercase=bool(row["lowercase"]),
        scheme_neutral=bool(row["scheme_neutral"]),
    )


def _candidate_summary(row: pd.Series) -> dict[str, object]:
    """Return JSON-safe comparison metrics for one candidate."""
    return {
        "name": str(row["name"]),
        "scheme_neutral": bool(row["scheme_neutral"]),
        "average_precision": float(row["average_precision"]),
        "roc_auc": float(row["roc_auc"]),
        "matched_threshold": float(row["matched_threshold"]),
        "matched_precision": float(row["matched_precision"]),
        "matched_recall": float(row["matched_recall"]),
        "matched_f1": float(row["matched_f1"]),
        "matched_actual_fpr": float(row["matched_actual_fpr"]),
    }


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
    axis.set_title("TF-IDF logistic-regression precision-recall curve")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _save_roc_curve(
    targets: np.ndarray,
    scores: np.ndarray,
    output_path: Path,
) -> None:
    false_positive_rate, true_positive_rate, _ = roc_curve(targets, scores)
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(false_positive_rate, true_positive_rate)
    axis.plot([0, 1], [0, 1], linestyle="--")
    axis.set_xlabel("False-positive rate")
    axis.set_ylabel("True-positive rate")
    axis.set_title("TF-IDF logistic-regression ROC curve")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _save_calibration_curve(
    targets: np.ndarray,
    scores: np.ndarray,
    output_path: Path,
) -> None:
    observed, predicted = calibration_curve(
        targets,
        scores,
        n_bins=10,
        strategy="quantile",
    )
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.plot(predicted, observed, marker="o", label="Model")
    axis.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    axis.set_xlabel("Mean predicted phishing probability")
    axis.set_ylabel("Observed phishing frequency")
    axis.set_title("Validation calibration curve")
    axis.legend()
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
    axis.hist(scores[targets == 0], bins=40, alpha=0.6, label="Legitimate")
    axis.hist(scores[targets == 1], bins=40, alpha=0.6, label="Phishing")
    axis.set_xlabel("Raw logistic phishing score")
    axis.set_ylabel("Number of URLs")
    axis.set_title("Validation score distribution")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _percentage(value: float) -> str:
    return f"{value * 100:.2f}%"


def _append_robustness_sections(
    lines: list[str],
    payload: dict[str, object],
) -> None:
    """Add shortcut and robustness diagnostics to the report."""
    raw_candidate = payload["best_raw_validation_candidate"]
    neutral_candidate = payload["scheme_neutral_candidate"]
    protocol = payload["protocol_only_diagnostic"]

    assert isinstance(raw_candidate, dict)
    assert isinstance(protocol, dict)

    lines.extend(
        [
            "",
            "## Robustness candidate comparison",
            "",
            "| Candidate | Scheme neutral | AP | ROC AUC | Recall at matched FPR | Actual FPR |",
            "|---|---|---:|---:|---:|---:|",
            (
                f"| `{raw_candidate['name']}` | No | "
                f"{raw_candidate['average_precision']:.4f} | "
                f"{raw_candidate['roc_auc']:.4f} | "
                f"{_percentage(raw_candidate['matched_recall'])} | "
                f"{_percentage(raw_candidate['matched_actual_fpr'])} |"
            ),
        ]
    )

    if isinstance(neutral_candidate, dict):
        lines.append(
            f"| `{neutral_candidate['name']}` | Yes | "
            f"{neutral_candidate['average_precision']:.4f} | "
            f"{neutral_candidate['roc_auc']:.4f} | "
            f"{_percentage(neutral_candidate['matched_recall'])} | "
            f"{_percentage(neutral_candidate['matched_actual_fpr'])} |"
        )

    lines.extend(
        [
            "",
            "The saved pipeline prefers the scheme-and-www-neutral candidate when available.",
            "The raw candidate is retained as an in-dataset performance benchmark.",
            "",
            "## Dataset shortcut diagnostic",
            "",
            "A protocol-only detector that flags plain HTTP URLs as phishing achieved:",
            "",
            f"- Precision: **{_percentage(protocol['precision'])}**",
            f"- Recall: **{_percentage(protocol['recall'])}**",
            f"- False-positive rate: **{_percentage(protocol['false_positive_rate'])}**",
            "",
            "All legitimate URLs in the current dataset use HTTPS. Consequently, protocol and",
            "leading-www patterns are treated as collection-bias indicators rather than reliable",
            "production phishing evidence.",
        ]
    )


def _render_report(payload: dict[str, object]) -> str:
    profiles = payload["profiles"]
    assert isinstance(profiles, dict)
    balanced = profiles["balanced"]
    assert isinstance(balanced, dict)
    balanced_metrics = balanced["metrics"]
    assert isinstance(balanced_metrics, dict)
    matched = payload["matched_rule_fpr_metrics"]
    assert isinstance(matched, dict)
    best_spec = payload["best_spec"]
    assert isinstance(best_spec, dict)

    lines = [
        "# Phase 2B — Character TF-IDF Logistic Regression",
        "",
        f"**Model:** `{payload['model_version']}`",
        "",
        f"**Training rows:** {payload['training_rows']:,}",
        "",
        f"**Validation rows:** {payload['validation_rows']:,}",
        "",
        "The locked test set was not loaded or evaluated.",
        "",
        "## Selected configuration",
        "",
        f"- Candidate: `{best_spec['name']}`",
        f"- Character n-grams: {best_spec['ngram_min']}–{best_spec['ngram_max']}",
        f"- Logistic-regression C: {best_spec['c']}",
        f"- Class weight: `{best_spec['class_weight']}`",
        f"- Maximum TF-IDF features: {best_spec['max_features']:,}",
        f"- Learned features: {payload['feature_count']:,}",
        "",
        "## Validation ranking and probability diagnostics",
        "",
        f"- Average precision: **{balanced_metrics['average_precision']:.4f}**",
        f"- ROC AUC: **{balanced_metrics['roc_auc']:.4f}**",
        f"- Brier score: **{balanced_metrics['brier_score']:.4f}**",
        f"- Expected calibration error: **{balanced_metrics['expected_calibration_error']:.4f}**",
        "",
        "The raw logistic-regression probabilities are evaluated for calibration but are not yet "
        "treated as production-calibrated risk. Formal calibration is a later phase.",
        "",
        "## Comparison at the rule baseline false-positive rate",
        "",
        f"- FPR limit: **{_percentage(payload['rule_baseline_fpr'])}**",
        f"- Selected threshold: **{matched['threshold']:.6f}**",
        f"- Precision: **{_percentage(matched['precision'])}**",
        f"- Recall: **{_percentage(matched['recall'])}**",
        f"- Actual FPR: **{_percentage(matched['false_positive_rate'])}**",
        f"- F1: **{matched['f1']:.4f}**",
        "",
        "## Operating profiles selected on validation data",
        "",
        "| Profile | Threshold | Precision | Recall | FPR | F1 | Constraints |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]

    for name in ("high_security", "balanced", "conservative"):
        profile = profiles[name]
        assert isinstance(profile, dict)
        metrics = profile["metrics"]
        assert isinstance(metrics, dict)
        lines.append(
            "| "
            f"{name.replace('_', ' ').title()} | {profile['threshold']:.6f} | "
            f"{_percentage(metrics['precision'])} | {_percentage(metrics['recall'])} | "
            f"{_percentage(metrics['false_positive_rate'])} | {metrics['f1']:.4f} | "
            f"{'Met' if profile['constraints_met'] else 'Fallback'} |"
        )
    _append_robustness_sections(lines, payload)

    lines.extend(
        [
            "",
            "## Runtime",
            "",
            f"- Final pipeline fit time: {payload['timing']['fit_seconds']:.3f} seconds",
            f"- Validation scoring time: {payload['timing']['scoring_seconds']:.3f} seconds",
            "- Mean validation latency: "
            f"{payload['timing']['milliseconds_per_url']:.4f} milliseconds per URL",
            f"- Throughput: {payload['timing']['urls_per_second']:.2f} URLs per second",
            "",
            "## Saved outputs",
            "",
            "- `model_comparison.csv`",
            "- `validation_metrics.json`",
            "- `threshold_table.csv`",
            "- `top_character_ngrams.csv`",
            "- `precision_recall_curve.png`",
            "- `roc_curve.png`",
            "- `calibration_curve.png`",
            "- `score_distribution.png`",
            "- `artifacts/models/tfidf_logistic/pipeline.joblib`",
            "",
            "## Limitations",
            "",
            "- Model selection and threshold selection both use the validation split.",
            (
                "- The test split remains locked until the final model family "
                "and calibration are fixed."
            ),
            (
                "- Character n-grams can learn dataset-specific URL patterns "
                "and require drift monitoring."
            ),
            "- Global coefficients explain influential substrings but are not causal evidence.",
            "- No URL was visited or downloaded during training or scoring.",
            "",
        ]
    )
    return "\n".join(lines)


def train_tfidf_logistic(
    *,
    train_path: Path = TRAIN_DATA_PATH,
    validation_path: Path = VALIDATION_DATA_PATH,
    report_dir: Path = TFIDF_LOGISTIC_REPORT_DIR,
    artifact_dir: Path = TFIDF_LOGISTIC_ARTIFACT_DIR,
    candidate_specs: list[TfidfLogisticSpec] | None = None,
    full_grid: bool = False,
    max_features: int | None = None,
) -> dict[str, object]:
    """Run validation-only model selection and persist the winning pipeline."""
    ensure_directories()
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found at {train_path}.")
    if not validation_path.exists():
        raise FileNotFoundError(f"Validation data not found at {validation_path}.")

    train_frame = pd.read_parquet(train_path)
    validation_frame = pd.read_parquet(validation_path)
    _validate_frame(train_frame, split_name="Training")
    _validate_frame(validation_frame, split_name="Validation")

    train_urls = train_frame["url_model_input"].astype(str)
    validation_urls = validation_frame["url_model_input"].astype(str)
    train_targets = train_frame["target"].to_numpy(dtype=np.int8)
    validation_targets = validation_frame["target"].to_numpy(dtype=np.int8)

    specs = candidate_specs or (full_candidate_specs() if full_grid else compact_candidate_specs())
    if max_features is not None:
        specs = [replace(spec, max_features=max_features) for spec in specs]

    rule_fpr, rule_payload = _load_rule_baseline_fpr(RULE_BASELINE_REPORT_DIR)
    costs = OperationalCosts(false_negative=25.0, false_positive=2.0)
    protocol_scores = _protocol_only_scores(validation_urls)

    protocol_metrics = evaluate_binary_scores(
        validation_targets,
        protocol_scores,
        threshold=0.5,
        costs=costs,
    )
    comparison = _search_candidates(
        train_urls,
        train_targets,
        validation_urls,
        validation_targets,
        specs,
        matched_fpr=rule_fpr,
        costs=costs,
    )
    comparison.to_csv(
        report_dir / "model_comparison.csv",
        index=False,
    )

    raw_candidates = comparison.loc[~comparison["scheme_neutral"].astype(bool)]

    neutral_candidates = comparison.loc[comparison["scheme_neutral"].astype(bool)]

    best_raw_row = raw_candidates.iloc[0] if not raw_candidates.empty else comparison.iloc[0]

    selected_row = (
        neutral_candidates.iloc[0] if not neutral_candidates.empty else comparison.iloc[0]
    )

    best_spec = _spec_from_row(selected_row)
    pipeline = build_pipeline(best_spec)

    fit_start = time.perf_counter()
    pipeline.fit(train_urls, train_targets)
    fit_seconds = time.perf_counter() - fit_start

    scoring_start = time.perf_counter()
    scores = phishing_scores(pipeline, validation_urls)
    scoring_seconds = time.perf_counter() - scoring_start

    threshold_table = build_threshold_table(
        validation_targets,
        scores,
        costs=costs,
    )
    profiles = select_threshold_profiles(threshold_table)
    matched_row = select_threshold_at_max_fpr(threshold_table, max_fpr=rule_fpr)

    top_ngrams = top_character_ngrams(pipeline, top_n=50)
    top_ngrams.to_csv(report_dir / "top_character_ngrams.csv", index=False)
    threshold_table.to_csv(report_dir / "threshold_table.csv", index=False)

    feature_count = int(len(pipeline.named_steps["tfidf"].get_feature_names_out()))
    milliseconds_per_url = scoring_seconds / len(validation_frame) * 1000
    urls_per_second = len(validation_frame) / scoring_seconds if scoring_seconds else float("inf")

    payload: dict[str, object] = {
        "model_version": TFIDF_LOGISTIC_VERSION,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "training_rows": int(len(train_frame)),
        "validation_rows": int(len(validation_frame)),
        "locked_test_used": False,
        "target_convention": {"0": "legitimate", "1": "phishing"},
        "best_spec": best_spec.to_dict(),
        "selection_policy": (
            "Prefer the scheme-and-www-neutral candidate when available; "
            "retain the best raw candidate as an in-dataset benchmark."
        ),
        "best_raw_validation_candidate": _candidate_summary(best_raw_row),
        "scheme_neutral_candidate": (
            _candidate_summary(neutral_candidates.iloc[0]) if not neutral_candidates.empty else None
        ),
        "protocol_only_diagnostic": protocol_metrics.to_dict(),
        "feature_count": feature_count,
        "rule_baseline_fpr": rule_fpr,
        "rule_baseline_metrics_available": rule_payload is not None,
        "matched_rule_fpr_metrics": {
            key: int(value)
            if key
            in {
                "n_samples",
                "true_positive",
                "false_positive",
                "true_negative",
                "false_negative",
            }
            else float(value)
            for key, value in matched_row.to_dict().items()
        },
        "profiles": {name: profile.to_dict() for name, profile in profiles.items()},
        "cost_assumptions": {
            "false_negative": costs.false_negative,
            "false_positive": costs.false_positive,
            "true_positive": costs.true_positive,
            "true_negative": costs.true_negative,
        },
        "timing": {
            "fit_seconds": fit_seconds,
            "scoring_seconds": scoring_seconds,
            "milliseconds_per_url": milliseconds_per_url,
            "urls_per_second": urls_per_second,
        },
    }

    joblib.dump(pipeline, artifact_dir / "pipeline.joblib", compress=3)
    (artifact_dir / "model_metadata.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "feature_schema.json").write_text(
        json.dumps(
            {
                "input_column": "url_model_input",
                "input_type": "string",
                "target_column": "target",
                "target_convention": {"0": "legitimate", "1": "phishing"},
                "external_network_requests": False,
                "webpage_content_features": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (report_dir / "validation_metrics.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    (report_dir / "report.md").write_text(
        _render_report(payload),
        encoding="utf-8",
    )

    _save_precision_recall_curve(
        validation_targets,
        scores,
        report_dir / "precision_recall_curve.png",
    )
    _save_roc_curve(
        validation_targets,
        scores,
        report_dir / "roc_curve.png",
    )
    _save_calibration_curve(
        validation_targets,
        scores,
        report_dir / "calibration_curve.png",
    )
    _save_score_distribution(
        validation_targets,
        scores,
        report_dir / "score_distribution.png",
    )

    balanced = profiles["balanced"].metrics
    print(f"Selected {best_spec.name} from {len(comparison)} candidates.")
    print(f"Average precision: {balanced['average_precision']:.4f}")
    print(f"ROC AUC: {balanced['roc_auc']:.4f}")
    print(
        "Recall at rule-baseline FPR: "
        f"{float(matched_row['recall']):.4f} at threshold "
        f"{float(matched_row['threshold']):.6f}"
    )
    print(f"Saved report to {report_dir / 'report.md'}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="Run all 18 configurations instead of the compact six-candidate search.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Override the maximum TF-IDF features for every candidate.",
    )
    args = parser.parse_args()
    train_tfidf_logistic(
        full_grid=args.full_grid,
        max_features=args.max_features,
    )


if __name__ == "__main__":
    main()
