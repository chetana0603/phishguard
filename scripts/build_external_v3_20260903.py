from pathlib import Path

from phishguard.data.external_ood import build_external_candidate

build_external_candidate(
    phishtank_path=Path("data/external/raw/phishtank_online_valid_20260903.csv"),
    tranco_path=Path("data/external/raw/tranco_top1m_20260903.csv"),
    tranco_list_id="manual-20260903",
    source_metadata={
        "acquisition": "manual_browser",
        "snapshot_date": "2026-09-03",
        "reason": ("Python outbound network access previously failed with WinError 10013."),
        "evaluation_role": "fresh_v3_external_snapshot",
        "model_frozen_before_snapshot": True,
        "model_version": "tfidf-logistic-v3-rootcanon",
    },
    output_path=Path("data/external/processed/external_ood_candidate_v3_20260903.parquet"),
    report_dir=Path("reports/data/external_v3_20260903"),
    max_per_class=30_000,
)
