"""Phase 2C-D2: internal robustness evaluation for the V3 calibrated model."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import SplitResult, urlsplit, urlunsplit

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phishguard.config import VALIDATION_DATA_PATH
from phishguard.evaluation.thresholds import (
    build_threshold_table,
    select_threshold_at_max_fpr,
)

CALIBRATED_MODEL_PATH = Path("artifacts/models/calibration_v3_rootcanon/sigmoid_grouped_cv.joblib")

# The rule-baseline FPR is the fixed comparison operating point
# established in Phase 2B.
PHASE2B_METRICS_PATH = Path("reports/models/tfidf_logistic/validation_metrics.json")

ROBUSTNESS_REPORT_DIR = Path("reports/models/robustness_v3_rootcanon")

STRICT_INVARIANCE_PERTURBATIONS = {
    "scheme_toggle",
    "www_toggle",
    "host_case",
    "root_slash",
}

INVARIANCE_TOLERANCE = 1e-12


def _sha256_file(
    path: Path,
) -> str:
    """Return SHA-256 digest for an artifact."""
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

    phishing_index = int(phishing_indices[0])

    return probabilities[
        :,
        phishing_index,
    ].astype(float)


def _parse_url(
    value: object,
) -> (
    tuple[
        SplitResult,
        bool,
    ]
    | None
):
    """
    Parse a URL while supporting URLs without an explicit scheme.

    Returns:
        (parsed_url, had_explicit_scheme)
    """
    text = str(value).strip()

    if not text:
        return None

    if any(character.isspace() for character in text):
        return None

    try:
        parsed = urlsplit(text)
    except ValueError:
        return None

    if parsed.scheme and parsed.netloc:
        if not parsed.hostname:
            return None

        return (
            parsed,
            True,
        )

    try:
        fallback = urlsplit(f"//{text}")
    except ValueError:
        return None

    if not fallback.hostname:
        return None

    return (
        fallback,
        False,
    )


def _rebuild_url(
    parsed: SplitResult,
    *,
    had_explicit_scheme: bool,
) -> str:
    """Reconstruct a URL while preserving explicit-scheme status."""
    if had_explicit_scheme:
        return urlunsplit(parsed)

    result = urlunsplit(
        (
            "",
            parsed.netloc,
            parsed.path,
            parsed.query,
            parsed.fragment,
        )
    )

    if result.startswith("//"):
        result = result[2:]

    return result


def _build_netloc(
    parsed: SplitResult,
    hostname: str,
) -> str:
    """Rebuild netloc with a replacement hostname."""
    user_info = ""

    if parsed.username is not None:
        user_info = parsed.username

        if parsed.password is not None:
            user_info += f":{parsed.password}"

        user_info += "@"

    host_part = hostname

    if ":" in hostname and not hostname.startswith("["):
        host_part = f"[{hostname}]"

    try:
        port = parsed.port
    except ValueError:
        port = None

    if port is not None:
        host_part += f":{port}"

    return f"{user_info}{host_part}"


def perturb_scheme(
    value: object,
) -> str | None:
    """Toggle HTTP and HTTPS while preserving the rest of the URL."""
    parsed_result = _parse_url(value)

    if parsed_result is None:
        return None

    parsed, had_scheme = parsed_result

    if had_scheme:
        scheme = parsed.scheme.lower()

        if scheme == "https":
            new_scheme = "http"
        elif scheme == "http":
            new_scheme = "https"
        else:
            return None

        perturbed = parsed._replace(scheme=new_scheme)

        return urlunsplit(perturbed)

    perturbed = parsed._replace(scheme="https")

    return urlunsplit(perturbed)


def perturb_www(
    value: object,
) -> str | None:
    """Add or remove a leading www. hostname prefix."""
    parsed_result = _parse_url(value)

    if parsed_result is None:
        return None

    parsed, had_scheme = parsed_result

    hostname = parsed.hostname

    if not hostname:
        return None

    hostname_lower = hostname.lower()

    new_hostname = hostname[4:] if hostname_lower.startswith("www.") else f"www.{hostname}"

    if not new_hostname:
        return None

    new_netloc = _build_netloc(
        parsed,
        new_hostname,
    )

    perturbed = parsed._replace(netloc=new_netloc)

    return _rebuild_url(
        perturbed,
        had_explicit_scheme=had_scheme,
    )


def perturb_host_case(
    value: object,
) -> str | None:
    """
    Change hostname letter case.

    Hostnames are case-insensitive, making this a strict
    invariance probe for V3.
    """
    parsed_result = _parse_url(value)

    if parsed_result is None:
        return None

    parsed, had_scheme = parsed_result

    hostname = parsed.hostname

    if not hostname:
        return None

    if not any(character.isalpha() for character in hostname):
        return None

    new_hostname = hostname.swapcase()

    if new_hostname == hostname:
        return None

    new_netloc = _build_netloc(
        parsed,
        new_hostname,
    )

    perturbed = parsed._replace(netloc=new_netloc)

    return _rebuild_url(
        perturbed,
        had_explicit_scheme=had_scheme,
    )


def perturb_root_slash(
    value: object,
) -> str | None:
    """
    Toggle only the HTTP(S) root-path representation.

    Examples:
        example.com
        example.com/

    V3 canonicalization explicitly treats these as equivalent.

    Non-root paths are excluded.
    """
    parsed_result = _parse_url(value)

    if parsed_result is None:
        return None

    parsed, had_scheme = parsed_result

    path = parsed.path

    if path == "":
        new_path = "/"
    elif path == "/":
        new_path = ""
    else:
        return None

    perturbed = parsed._replace(path=new_path)

    return _rebuild_url(
        perturbed,
        had_explicit_scheme=had_scheme,
    )


def perturb_non_root_trailing_slash(
    value: object,
) -> str | None:
    """
    Toggle a trailing slash only for non-root paths.

    Examples:
        /login  <-> /login/

    This remains a sensitivity probe because these paths can
    represent different resources.
    """
    parsed_result = _parse_url(value)

    if parsed_result is None:
        return None

    parsed, had_scheme = parsed_result

    path = parsed.path

    if path in {
        "",
        "/",
    }:
        return None

    new_path = path[:-1] if path.endswith("/") else f"{path}/"

    if new_path == path:
        return None

    perturbed = parsed._replace(path=new_path)

    return _rebuild_url(
        perturbed,
        had_explicit_scheme=had_scheme,
    )


def perturb_trailing_slash(
    value: object,
) -> str | None:
    """
    Legacy trailing-slash perturbation.

    Retained for backward compatibility with older unit tests.

    V3 robustness evaluation does NOT use this combined probe.
    """
    parsed_result = _parse_url(value)

    if parsed_result is None:
        return None

    parsed, had_scheme = parsed_result

    path = parsed.path

    new_path = path[:-1] if path.endswith("/") else f"{path}/"

    if new_path == path:
        return None

    perturbed = parsed._replace(path=new_path)

    return _rebuild_url(
        perturbed,
        had_explicit_scheme=had_scheme,
    )


def perturb_query_order(
    value: object,
) -> str | None:
    """
    Reverse query-parameter order when keys are unique.

    Duplicate query keys are skipped because their ordering
    can be semantically meaningful.
    """
    parsed_result = _parse_url(value)

    if parsed_result is None:
        return None

    parsed, had_scheme = parsed_result

    if not parsed.query:
        return None

    parts = parsed.query.split("&")

    if len(parts) < 2:
        return None

    keys = [part.partition("=")[0] for part in parts]

    if len(set(keys)) != len(keys):
        return None

    reordered = list(reversed(parts))

    new_query = "&".join(reordered)

    if new_query == parsed.query:
        return None

    perturbed = parsed._replace(query=new_query)

    return _rebuild_url(
        perturbed,
        had_explicit_scheme=had_scheme,
    )


PERTURBATIONS: dict[
    str,
    Callable[
        [object],
        str | None,
    ],
] = {
    "scheme_toggle": (perturb_scheme),
    "www_toggle": (perturb_www),
    "host_case": (perturb_host_case),
    "root_slash": (perturb_root_slash),
    "non_root_trailing_slash": (perturb_non_root_trailing_slash),
    "query_order": (perturb_query_order),
}


def _load_rule_baseline_fpr(
    metrics_path: Path,
) -> float:
    """Load the fixed Phase 2B rule-baseline FPR."""
    if not metrics_path.exists():
        raise FileNotFoundError(f"Phase 2B metrics not found at {metrics_path}.")

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))

    value = payload.get("rule_baseline_fpr")

    if value is None:
        raise ValueError("Phase 2B metrics do not contain rule_baseline_fpr.")

    fpr = float(value)

    if not (0.0 <= fpr <= 1.0):
        raise ValueError("rule_baseline_fpr must lie in [0, 1].")

    return fpr


def _derive_operating_threshold(
    targets: np.ndarray,
    scores: np.ndarray,
    *,
    max_fpr: float,
) -> tuple[
    float,
    dict[
        str,
        object,
    ],
]:
    """Select the V3 validation operating threshold at the fixed FPR."""
    threshold_table = build_threshold_table(
        targets,
        scores,
    )

    selected = select_threshold_at_max_fpr(
        threshold_table,
        max_fpr=max_fpr,
    )

    if "threshold" not in selected.index:
        raise RuntimeError("Threshold-selection result does not contain a threshold.")

    threshold = float(selected["threshold"])

    selected_metrics: dict[
        str,
        object,
    ] = {}

    for key, value in selected.to_dict().items():
        if isinstance(
            value,
            (
                np.bool_,
                bool,
            ),
        ):
            selected_metrics[str(key)] = bool(value)

        elif isinstance(
            value,
            (
                np.integer,
                int,
            ),
        ):
            selected_metrics[str(key)] = int(value)

        elif isinstance(
            value,
            (
                np.floating,
                float,
            ),
        ):
            selected_metrics[str(key)] = float(value)

        else:
            selected_metrics[str(key)] = str(value)

    return (
        threshold,
        selected_metrics,
    )


def _score_perturbation(
    *,
    name: str,
    estimator: Any,
    frame: pd.DataFrame,
    original_scores: np.ndarray,
    threshold: float,
    transform: Callable[
        [object],
        str | None,
    ],
) -> tuple[
    dict[
        str,
        object,
    ],
    pd.DataFrame,
]:
    """Evaluate probability and decision stability."""
    transformed_urls: list[str] = []

    original_indices: list[int] = []

    urls = frame["url_model_input"].astype(str)

    for position, url in enumerate(urls):
        transformed = transform(url)

        if transformed is None:
            continue

        if transformed == url:
            continue

        transformed_urls.append(transformed)

        original_indices.append(position)

    expected_invariant = name in STRICT_INVARIANCE_PERTURBATIONS

    if not original_indices:
        metrics: dict[
            str,
            object,
        ] = {
            "perturbation": name,
            "expected_invariant": (expected_invariant),
            "invariance_passed": (True if expected_invariant else None),
            "applicable_rows": 0,
            "applicability_rate": 0.0,
            "mean_absolute_drift": 0.0,
            "median_absolute_drift": 0.0,
            "p95_absolute_drift": 0.0,
            "max_absolute_drift": 0.0,
            "mean_signed_drift": 0.0,
            "flip_count": 0,
            "flip_rate": 0.0,
            "predicted_legitimate_to_phishing": 0,
            "predicted_phishing_to_legitimate": 0,
            "legitimate_row_flip_rate": 0.0,
            "phishing_row_flip_rate": 0.0,
        }

        return (
            metrics,
            pd.DataFrame(),
        )

    index_array = np.asarray(
        original_indices,
        dtype=int,
    )

    original_subset_scores = original_scores[index_array]

    perturbed_series = pd.Series(
        transformed_urls,
        dtype=str,
    )

    perturbed_scores = _positive_class_scores(
        estimator,
        perturbed_series,
    )

    targets = frame["target"].to_numpy(dtype=np.int8)[index_array]

    original_predictions = (original_subset_scores >= threshold).astype(np.int8)

    perturbed_predictions = (perturbed_scores >= threshold).astype(np.int8)

    signed_drift = perturbed_scores - original_subset_scores

    absolute_drift = np.abs(signed_drift)

    flips = original_predictions != perturbed_predictions

    legitimate_to_phishing = (original_predictions == 0) & (perturbed_predictions == 1)

    phishing_to_legitimate = (original_predictions == 1) & (perturbed_predictions == 0)

    legitimate_rows = targets == 0

    phishing_rows = targets == 1

    legitimate_flip_rate = float(flips[legitimate_rows].mean()) if np.any(legitimate_rows) else 0.0

    phishing_flip_rate = float(flips[phishing_rows].mean()) if np.any(phishing_rows) else 0.0

    max_absolute_drift = float(absolute_drift.max())

    flip_count = int(flips.sum())

    if expected_invariant:
        invariance_passed: bool | None = (
            max_absolute_drift <= INVARIANCE_TOLERANCE and flip_count == 0
        )
    else:
        invariance_passed = None

    metrics = {
        "perturbation": name,
        "expected_invariant": (expected_invariant),
        "invariance_passed": (invariance_passed),
        "applicable_rows": int(len(index_array)),
        "applicability_rate": float(len(index_array) / len(frame)),
        "mean_absolute_drift": float(absolute_drift.mean()),
        "median_absolute_drift": float(np.median(absolute_drift)),
        "p95_absolute_drift": float(
            np.quantile(
                absolute_drift,
                0.95,
            )
        ),
        "max_absolute_drift": (max_absolute_drift),
        "mean_signed_drift": float(signed_drift.mean()),
        "flip_count": (flip_count),
        "flip_rate": float(flips.mean()),
        "predicted_legitimate_to_phishing": int(legitimate_to_phishing.sum()),
        "predicted_phishing_to_legitimate": int(phishing_to_legitimate.sum()),
        "legitimate_row_flip_rate": (legitimate_flip_rate),
        "phishing_row_flip_rate": (phishing_flip_rate),
    }

    details = pd.DataFrame(
        {
            "row_index": (index_array),
            "target": (targets),
            "original_score": (original_subset_scores),
            "perturbed_score": (perturbed_scores),
            "signed_drift": (signed_drift),
            "absolute_drift": (absolute_drift),
            "original_prediction": (original_predictions),
            "perturbed_prediction": (perturbed_predictions),
            "flipped": (flips),
        }
    )

    return (
        metrics,
        details,
    )


def _safe_url_hash(
    value: object,
) -> str:
    """Hash a URL so reports never expose phishing URLs."""
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def _save_largest_drift_examples(
    *,
    frame: pd.DataFrame,
    detail_frames: dict[
        str,
        pd.DataFrame,
    ],
    output_path: Path,
    top_n: int = 10,
) -> None:
    """Save hashed examples with the largest non-zero drifts."""
    rows: list[
        dict[
            str,
            object,
        ]
    ] = []

    urls = frame["url_model_input"].astype(str)

    for (
        perturbation,
        details,
    ) in detail_frames.items():
        if details.empty:
            continue

        nonzero = details.loc[details["absolute_drift"] > 0.0]

        if nonzero.empty:
            continue

        largest = nonzero.nlargest(
            top_n,
            "absolute_drift",
        )

        for _, row in largest.iterrows():
            index = int(row["row_index"])

            rows.append(
                {
                    "perturbation": (perturbation),
                    "url_sha256": (_safe_url_hash(urls.iloc[index])),
                    "target": int(row["target"]),
                    "original_score": float(row["original_score"]),
                    "perturbed_score": float(row["perturbed_score"]),
                    "signed_drift": float(row["signed_drift"]),
                    "absolute_drift": float(row["absolute_drift"]),
                    "original_prediction": int(row["original_prediction"]),
                    "perturbed_prediction": int(row["perturbed_prediction"]),
                    "flipped": bool(row["flipped"]),
                }
            )

    pd.DataFrame(rows).to_csv(
        output_path,
        index=False,
    )


def _save_drift_plot(
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save mean and 95th-percentile probability drift."""
    if summary.empty:
        return

    positions = np.arange(len(summary))

    width = 0.38

    figure, axis = plt.subplots(
        figsize=(
            10,
            6,
        )
    )

    axis.bar(
        positions - width / 2,
        summary["mean_absolute_drift"],
        width,
        label=("Mean absolute drift"),
    )

    axis.bar(
        positions + width / 2,
        summary["p95_absolute_drift"],
        width,
        label=("95th percentile drift"),
    )

    axis.set_xticks(positions)

    axis.set_xticklabels(
        summary["perturbation"],
        rotation=25,
        ha="right",
    )

    axis.set_ylabel("Absolute probability change")

    axis.set_title("Phase 2C-D2 V3 Internal Robustness")

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=160,
    )

    plt.close(figure)


def _render_report(
    payload: dict[
        str,
        object,
    ],
) -> str:
    """Render the Phase 2C-D2 Markdown report."""
    rows = payload["perturbations"]

    assert isinstance(
        rows,
        list,
    )

    lines = [
        "# Phase 2C-D2 — V3 Internal Robustness",
        "",
        "## Evaluation policy",
        "",
        (f"- Validation rows: **{payload['validation_rows']:,}**"),
        (f"- Matched FPR limit: **{payload['matched_fpr_limit']:.4%}**"),
        (f"- V3 calibrated operating threshold: **{payload['operating_threshold']:.6f}**"),
        ("- Model: **tfidf-logistic-v3 + root-path canonicalization + grouped-CV sigmoid**"),
        ("- Threshold selected using the existing validation split only."),
        "- External/OOD snapshot used: **False**",
        "- Locked test loaded/scored: **False**",
        "",
        "## Perturbation results",
        "",
        ("| Perturbation | Applicable | Mean drift | P95 drift | Max drift | Flip rate | Check |"),
        "|---|---:|---:|---:|---:|---:|---|",
    ]

    for row in rows:
        assert isinstance(
            row,
            dict,
        )

        if row["expected_invariant"]:
            check = "PASS" if row["invariance_passed"] else "FAIL"
        else:
            check = "diagnostic"

        lines.append(
            "| "
            f"{row['perturbation']} | "
            f"{row['applicable_rows']:,} | "
            f"{row['mean_absolute_drift']:.6f} | "
            f"{row['p95_absolute_drift']:.6f} | "
            f"{row['max_absolute_drift']:.6f} | "
            f"{row['flip_rate']:.4%} | "
            f"{check} |"
        )

    lines.extend(
        [
            "",
            "## Strict invariance checks",
            "",
            (
                "- `scheme_toggle`, `www_toggle`, "
                "`host_case`, and `root_slash` are "
                "strict V3 invariance probes."
            ),
            (
                "- `root_slash` tests only equivalent "
                "root representations such as "
                "`example.com` and `example.com/`."
            ),
            (f"- Strict invariance tolerance: `{payload['strict_invariance_tolerance']}`."),
            (
                "- Overall strict invariance status: "
                f"**{'PASS' if payload['strict_invariance_passed'] else 'FAIL'}**"
            ),
            "",
            "## Sensitivity diagnostics",
            "",
            ("- `non_root_trailing_slash` deliberately keeps `/login` and `/login/` distinct."),
            (
                "- `query_order` remains a sensitivity "
                "diagnostic rather than a guaranteed "
                "semantic-preserving transformation."
            ),
            "",
            "## Safety and leakage controls",
            "",
            "- The external/OOD snapshot was not loaded.",
            "- The locked test split was not loaded.",
            (
                "- Raw phishing URLs are not written to "
                "the example report; SHA-256 hashes are "
                "stored instead."
            ),
            "",
            "## Generated artifacts",
            "",
            "- `robustness_summary.csv`",
            "- `robustness_metrics.json`",
            "- `largest_drift_examples.csv`",
            "- `probability_drift.png`",
            "- `report.md`",
            "",
        ]
    )

    return "\n".join(lines)


def run_robustness_evaluation(
    *,
    model_path: Path = CALIBRATED_MODEL_PATH,
    validation_path: Path = VALIDATION_DATA_PATH,
    phase2b_metrics_path: Path = PHASE2B_METRICS_PATH,
    report_dir: Path = ROBUSTNESS_REPORT_DIR,
) -> dict[
    str,
    object,
]:
    """Run Phase 2C-D2 V3 validation robustness evaluation."""
    report_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not model_path.exists():
        raise FileNotFoundError(f"Calibrated model not found at {model_path}.")

    if not validation_path.exists():
        raise FileNotFoundError(f"Validation data not found at {validation_path}.")

    frame = pd.read_parquet(validation_path)

    required = {
        "url_model_input",
        "target",
    }

    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(f"Validation data missing required columns: {sorted(missing)}")

    if frame.empty:
        raise ValueError("Validation data is empty.")

    targets = frame["target"].to_numpy(dtype=np.int8)

    if set(np.unique(targets)) != {
        0,
        1,
    }:
        raise ValueError("Validation target must contain classes 0 and 1.")

    estimator = joblib.load(model_path)

    validation_urls = frame["url_model_input"].astype(str)

    original_scores = _positive_class_scores(
        estimator,
        validation_urls,
    )

    matched_fpr = _load_rule_baseline_fpr(phase2b_metrics_path)

    (
        operating_threshold,
        threshold_metrics,
    ) = _derive_operating_threshold(
        targets,
        original_scores,
        max_fpr=matched_fpr,
    )

    summary_rows: list[
        dict[
            str,
            object,
        ]
    ] = []

    detail_frames: dict[
        str,
        pd.DataFrame,
    ] = {}

    for (
        name,
        transform,
    ) in PERTURBATIONS.items():
        metrics, details = _score_perturbation(
            name=name,
            estimator=estimator,
            frame=frame,
            original_scores=original_scores,
            threshold=operating_threshold,
            transform=transform,
        )

        summary_rows.append(metrics)

        detail_frames[name] = details

    strict_rows = [row for row in summary_rows if row["expected_invariant"]]

    strict_invariance_passed = all(bool(row["invariance_passed"]) for row in strict_rows)

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        report_dir / "robustness_summary.csv",
        index=False,
    )

    _save_largest_drift_examples(
        frame=frame,
        detail_frames=detail_frames,
        output_path=(report_dir / "largest_drift_examples.csv"),
    )

    _save_drift_plot(
        summary,
        report_dir / "probability_drift.png",
    )

    payload: dict[
        str,
        object,
    ] = {
        "phase": "2C-D2",
        "model_version": ("tfidf-logistic-v3-rootcanon"),
        "locked_test_used": False,
        "external_ood_used": False,
        "model_path": str(model_path),
        "model_sha256": (_sha256_file(model_path)),
        "validation_rows": int(len(frame)),
        "matched_fpr_limit": float(matched_fpr),
        "operating_threshold": float(operating_threshold),
        "selected_threshold_metrics": (threshold_metrics),
        "strict_invariance_tolerance": (INVARIANCE_TOLERANCE),
        "strict_invariance_passed": (strict_invariance_passed),
        "perturbations": (summary_rows),
    }

    (report_dir / "robustness_metrics.json").write_text(
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

    print()
    print("Phase 2C-D2 V3 internal robustness")

    print("=" * 88)

    print(f"Validation rows: {len(frame):,}")

    print(f"Matched FPR limit: {matched_fpr:.4%}")

    print(f"Selected V3 calibrated threshold: {operating_threshold:.6f}")

    print("External/OOD data used: False")

    print("Locked test used: False")

    print()

    display = summary[
        [
            "perturbation",
            "applicable_rows",
            "mean_absolute_drift",
            "p95_absolute_drift",
            "max_absolute_drift",
            "flip_rate",
        ]
    ].copy()

    checks: list[str] = []

    for row in summary_rows:
        if row["expected_invariant"]:
            checks.append("PASS" if row["invariance_passed"] else "FAIL")
        else:
            checks.append("diagnostic")

    display["check"] = checks

    print(display.to_string(index=False))

    print()

    print(f"Strict V3 invariance checks: {'PASS' if strict_invariance_passed else 'FAIL'}")

    print()

    return payload


def main() -> None:
    """Run Phase 2C-D2."""
    run_robustness_evaluation()


if __name__ == "__main__":
    main()
