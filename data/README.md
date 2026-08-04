# PhishGuard Data

## Source

PhiUSIIL Phishing URL (Website), UCI Machine Learning Repository, dataset ID 967.

## Target convention

Original dataset:
- `0` = phishing
- `1` = legitimate

Internal PhishGuard convention:
- `1` = phishing
- `0` = legitimate

## Version 1 scope

Version 1 uses only information available in the URL string. It does not use webpage HTML,
page titles, rendered content, or any feature requiring a request to the URL.

## Storage policy

Raw, interim, and processed datasets are generated locally and are not committed to GitHub.
The code required to reproduce each artefact is version-controlled.

## Safety

Dataset URLs must never be automatically opened, requested, rendered as clickable links, or
included as live hyperlinks in generated reports.
