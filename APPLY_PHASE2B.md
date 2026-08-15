# Apply Phase 2B

1. Start from a green `main` branch:

```powershell
git switch main
git pull --ff-only
git switch -c phase2/tfidf-logistic-regression
```

2. Copy the contents of this patch folder into the repository root. Allow `config.py` and
`evaluation/thresholds.py` to be replaced. The threshold replacement is required because
continuous ML scores need a scalable O(n log n) threshold table; the Phase 2A implementation
was suitable only for a small number of distinct rule scores.

3. Add the direct model-serialization dependency and refresh the lock file:

```powershell
uv add "joblib>=1.5"
```

4. Add this precise ignore rule to `.gitignore` if it is not already present:

```gitignore
/artifacts/models/**/*.joblib
```

Do not add `models/` or `**/models/`; those patterns hide source code under
`src/phishguard/models` from Git and Linux CI.

5. Apply `README_UPDATE.md`, then format and test:

```powershell
uv run ruff check . --fix --no-cache
uv run ruff format .
uv run ruff check . --no-cache
uv run ruff format --check .
uv run pytest --cov=phishguard --cov-report=term-missing
```

6. Confirm the model source files are tracked before pushing:

```powershell
git check-ignore -v src/phishguard/models/tfidf_logistic.py
git ls-files src/phishguard/models
```

The first command should print nothing. After staging, the second command should include both
`rule_baseline.py` and `tfidf_logistic.py`.

7. Run the compact six-candidate search:

```powershell
uv run python -m phishguard.training.tfidf_logistic
```

The full 18-candidate grid is optional and can take substantially longer:

```powershell
uv run python -m phishguard.training.tfidf_logistic --full-grid
```

8. Inspect:

```powershell
code reports\models\tfidf_logistic\report.md
Import-Csv reports\models\tfidf_logistic\model_comparison.csv |
    Format-Table name, average_precision, roc_auc, matched_recall, matched_actual_fpr
```

9. Before committing, verify that `pipeline.joblib` is ignored:

```powershell
git status --short
git check-ignore -v artifacts/models/tfidf_logistic/pipeline.joblib
```

10. Commit only after the report and checks are correct:

```powershell
git add .
git status
git commit -m "feat: add character TF-IDF logistic regression model"
git push -u origin phase2/tfidf-logistic-regression
```
