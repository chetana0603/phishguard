## Phase 2C — Calibration, Robustness, and External Evaluation

### Phase 2C-A — Probability Calibration

The selected scheme-neutral character TF-IDF Logistic Regression model
was calibrated using registered-domain-aware splitting.

A grouped 80/20 model-fit/calibration split was first used to compare
uncalibrated, sigmoid, and isotonic probability calibration.

V3 holdout calibration results:

| Method | AP | ROC-AUC | Brier | Log Loss | ECE |
|---|---:|---:|---:|---:|---:|
| Isotonic | 0.9345 | 0.9340 | 0.0871 | 0.2870 | 0.0110 |
| Sigmoid | 0.9384 | 0.9343 | 0.0873 | 0.2899 | 0.0218 |
| Uncalibrated | 0.9384 | 0.9343 | 0.0913 | 0.3060 | 0.0558 |

The final candidate used five-fold StratifiedGroupKFold calibration
grouped by `registered_domain`, with sigmoid calibration and
`ensemble=False`.

Final grouped-CV sigmoid validation results:

- Average Precision: 0.940383
- ROC-AUC: 0.936372
- Brier score: 0.087175
- Log loss: 0.286352
- ECE: 0.030398
- Training rows: 141,090
- Calibration CV folds: 5
- Grouping key: `registered_domain`
- Locked test scored: False

### Phase 2C-B — Initial Robustness Evaluation

The calibrated URL model was stress-tested using controlled URL
perturbations.

The initial robustness probes included:

- HTTP/HTTPS scheme toggling
- leading `www.` toggling
- hostname case changes
- trailing-slash changes
- query-parameter reordering

Scheme and `www.` normalization were stable.

The initial hostname-case test exposed substantial prediction
sensitivity even though DNS hostnames are case-insensitive. This was
treated as a preprocessing defect rather than normal model variation.

Hostname-only lowercasing was therefore introduced while preserving
path, query, and fragment case.

After this fix:

- scheme-toggle drift: 0
- `www.`-toggle drift: 0
- hostname-case drift: 0
- hostname-case prediction flips: 0

Trailing-slash sensitivity remained large and was initially retained
as a diagnostic because `/login` and `/login/` are not guaranteed to
represent the same server resource.

### Phase 2C-C1 — Initial External/OOD Dataset Construction

An external evaluation dataset was constructed using:

- PhishTank verified and online phishing URLs
- Tranco popular domains as a benign proxy

Python-based acquisition failed on Windows with `WinError 10013`, so
the source snapshots were downloaded manually in a browser.

Snapshot date: 2026-08-18.

Development-set registered domains were removed before evaluation.

Initial candidate:

- PhishTank raw rows: 72,299
- PhishTank development overlap removed: 32,042
- Tranco development overlap removed: 35,066
- Tranco/PhishTank domain conflicts removed: 696
- Final phishing rows: 12,200
- Final benign-proxy rows: 12,200
- Total rows: 24,400
- Locked test scored: False

Raw source hashes were recorded for provenance.

### Phase 2C-C2 — Locked-Test Domain Decontamination

Before external evaluation, external examples sharing a
`registered_domain` with the locked internal test split were removed.

Only the locked-test `registered_domain` column was loaded.

The following were not accessed:

- locked-test URLs
- locked-test labels
- locked-test features
- locked-test model predictions
- locked-test evaluation metrics

Initial decontaminated external set:

- External rows before: 24,400
- Locked-test rows inspected for domain identifier only: 47,030
- Unique locked-test domain hashes: 35,104
- Overlapping external rows removed: 577
- External rows after: 23,823
- Remaining phishing rows: 12,058
- Remaining benign-proxy rows: 11,765
- Locked test scored: False

No class rebalancing was performed after decontamination.

### Phase 2C-C3 — V2 External/OOD Evaluation

The frozen V2 calibrated model was evaluated once on the August
external snapshot.

Results:

- PhishTank recall: 92.42%
- Tranco benign-proxy false-positive rate: 18.78%
- Precision: 83.45%
- F1: 0.8771
- ROC-AUC: 0.9465
- Average Precision: 0.9534
- Brier score: 0.1254
- Log loss: 0.4051
- ECE: 0.1383

A root-path diagnostic revealed a severe representation defect.

For synthesized Tranco root URLs:

- baseline FPR without `/`: 18.78%
- FPR after adding `/`: 83.70%
- prediction flip rate: 64.91%
- mean probability drift: 0.5082

Because `https://example.com` and `https://example.com/` identify the
same HTTP root resource, this was treated as a legitimate
preprocessing invariance defect.

The August external snapshot therefore became development evidence and
was no longer considered an unbiased final external evaluation set.

No model hyperparameters, calibration method, or threshold were tuned
using the August results.

### Phase 2C-D — V3 Root-Path Canonicalization

A targeted preprocessing change was introduced:

- empty root path and `/` are canonicalized to the same representation
- non-root paths remain unchanged
- `/login` and `/login/` remain distinct
- hostname lowercasing remains enabled
- path, query, and fragment case remain preserved

Examples:

`example.com` == `example.com/`

`example.com?x=1` == `example.com/?x=1`

`example.com/login` != `example.com/login/`

The model version was advanced to:

`tfidf-logistic-v3-rootcanon`

No classifier hyperparameters were changed.

### Phase 2C-D1 — V3 Internal Training and Calibration

V3 validation ranking performance:

- Average Precision: 0.9404
- ROC-AUC: 0.9364

At the rule-baseline FPR operating point, the uncalibrated V3 model
reached approximately 74.75% recall.

The final V3 model used five-fold registered-domain-grouped sigmoid
calibration.

Final calibrated V3 validation metrics:

- Average Precision: 0.940383
- ROC-AUC: 0.936372
- Brier score: 0.087175
- Log loss: 0.286352
- ECE: 0.030398

The validation-derived operating threshold was frozen at:

`0.768113160039295`

Matched validation FPR limit:

`0.016314423433444566`

### Phase 2C-D2 — V3 Internal Robustness

The robustness evaluation was updated to separate semantically
equivalent root-slash changes from potentially meaningful non-root
trailing-slash changes.

Results:

| Perturbation | Applicable Rows | Mean Drift | P95 Drift | Max Drift | Flip Rate | Status |
|---|---:|---:|---:|---:|---:|---|
| Scheme toggle | 47,030 | 0.000000 | 0.000000 | 0.000000 | 0.0000% | PASS |
| `www.` toggle | 47,030 | 0.000000 | 0.000000 | 0.000000 | 0.0000% | PASS |
| Host case | 46,894 | 0.000000 | 0.000000 | 0.000000 | 0.0000% | PASS |
| Root slash | 41,774 | 0.000000 | 0.000000 | 0.000000 | 0.0000% | PASS |
| Non-root trailing slash | 5,256 | 0.005136 | 0.025323 | 0.397530 | 0.7420% | Diagnostic |
| Query order | 420 | 0.000402 | 0.001123 | 0.028582 | 0.2381% | Diagnostic |

All strict V3 invariance checks passed.

External/OOD data was not used during this stage.

The locked test was not scored.

### Phase 2C-E1 — Fresh External Snapshot

Because the August external snapshot had influenced the V3
preprocessing design, a completely fresh external snapshot was
acquired on 2026-09-03.

Sources:

- fresh PhishTank verified-online feed
- fresh Tranco top-1M ranking

Initial fresh candidate:

- PhishTank raw rows: 74,280
- PhishTank development overlap removed: 32,382
- Tranco development overlap removed: 34,815
- Tranco/PhishTank conflicts removed: 696
- Final phishing rows: 13,077
- Final benign-proxy rows: 13,077
- Total rows: 26,154

Model scoring was not performed during dataset preparation.

### Phase 2C-E2 — Fresh External Test-Domain Decontamination

The September external candidate was again decontaminated against the
locked-test registered domains.

Only the locked-test registered-domain identifiers were loaded and
immediately hashed.

Results:

- External rows before: 26,154
- Locked-test rows inspected for registered domain only: 47,030
- Unique locked-test domain hashes: 35,104
- Overlapping external rows removed: 626
- External rows after: 25,528
- Remaining phishing rows: 12,934
- Remaining benign-proxy rows: 12,594

Locked-test URLs loaded: False

Locked-test labels loaded: False

Locked-test predictions generated: False

Locked test scored: False

### Phase 2C-E3 — Final Fresh External/OOD Evaluation

The frozen V3 model was evaluated once using:

- model: `tfidf-logistic-v3-rootcanon`
- grouped-CV sigmoid calibration
- threshold: `0.768113160039295`
- fresh September external snapshot
- no external-data tuning

Results:

- PhishTank recall / TPR: 83.5859%
- Tranco benign-proxy FPR: 18.4612%
- Precision: 82.3005%
- F1: 0.8294
- ROC-AUC: 0.9012
- Average Precision: 0.9167
- Brier score: 0.158137
- Log loss: 0.510190
- ECE: 0.136937

Root-slash diagnostic:

- baseline Tranco FPR: 18.4612%
- root-slash Tranco FPR: 18.4612%
- prediction flip rate: 0.0000%
- mean probability drift: 0.000000

This confirmed that the root-path representation defect discovered in
V2 was eliminated on fresh unseen external data.

The high Tranco benign-proxy FPR and increased calibration error
indicate meaningful distribution shift.

The Tranco result is not interpreted as deployment FPR because the
dataset contains popular registered domains rather than a natural
distribution of benign browsing URLs.

No model retraining, recalibration, preprocessing change, or threshold
retuning was performed after observing these results.

### Phase 2C-F — Frozen Model Manifest

Before locked-test model scoring, a freeze manifest was created.

Frozen model:

`tfidf-logistic-v3-rootcanon`

Calibration:

`5-fold registered-domain grouped-CV sigmoid`

Frozen threshold:

`0.768113160039295`

Freeze Git commit:

`9198acad4f81016fba47d57ae74f434880a2d58b`

Frozen model SHA-256:

`7df6dd6102d37f4d3358db5a7536b48395f1c1f610414e0d9c7c6fcb3463f058`

The manifest recorded that before final evaluation:

- locked-test registered domains had been used only for external-data decontamination
- locked-test URLs had not been used
- locked-test labels had not been used
- locked-test predictions had not been generated
- the locked test had not been model-scored

### Phase 2C-G — Final Locked-Test Evaluation

The frozen V3 model was evaluated once on the locked internal test set.

No model, preprocessing, calibration, or threshold changes were made
after this evaluation.

Locked-test rows:

47,030

Confusion matrix:

- True positives: 14,746
- False negatives: 5,314
- True negatives: 26,568
- False positives: 402

Final metrics:

| Metric | Result |
|---|---:|
| Accuracy | 87.8461% |
| Precision | 97.3462% |
| Recall / TPR | 73.5095% |
| False-positive rate | 1.4905% |
| Specificity | 98.5095% |
| F1 | 0.8377 |
| ROC-AUC | 0.9394 |
| Average Precision | 0.9424 |
| Brier score | 0.085254 |
| Log loss | 0.282593 |
| ECE | 0.025873 |

The final test FPR of 1.49% remained below the validation operating
limit of 1.63%.

Test ranking and probability-quality metrics were also close to the
validation results, supporting the stability of the domain-grouped
internal evaluation procedure.

The locked test is now considered consumed.

Any future changes to preprocessing, model configuration, calibration,
or threshold must be treated as a new model version rather than a
revision of V3.

---

## Phase 2C Final Decision

`tfidf-logistic-v3-rootcanon` is the final Phase 2 candidate.

The model demonstrated:

- domain-grouped internal generalization
- stable validation-to-test performance
- 97.35% test precision
- 73.51% test phishing recall
- 1.49% test false-positive rate
- zero drift for scheme, `www.`, hostname case, and root-path invariance
- meaningful but degraded performance under fresh external distribution shift

The external evaluation also demonstrates that internal test
performance should not be interpreted as equivalent to deployment
performance.

The model is frozen.

Future model improvements will use a new model version.