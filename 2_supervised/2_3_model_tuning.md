## Model Tuning Workflow

**Position in DS pipeline:**

```
... → Cross-Validation → GridSearchCV / RandomizedSearchCV → Evaluate best model
```

```python
# ── 0. Imports ────────────────────────────────────────────────
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

# ── 1. Inspect Pipeline Parameters ───────────────────────────
# Parameter names follow the pattern: <step_name>__<parameter>
pipeline.get_params()

# ── 2. Define Parameter Grid ──────────────────────────────────
param_grid = {
    "model__C":           [0.01, 0.1, 1, 10, 100],   # LogisticRegression example
    "model__n_neighbors": [3, 5, 7, 10, 15],          # KNN example
    "model__weights":     ["uniform", "distance"],     # KNN example
}

# ── 3a. GridSearchCV (exhaustive) ─────────────────────────────
gs = GridSearchCV(
    pipeline,
    param_grid,
    scoring="accuracy",
    cv=StratifiedKFold(5),
    n_jobs=-1,
    verbose=1,
)
gs.fit(X_train, y_train)

print("Best score: ", round(gs.best_score_, 3))
print("Best params:", gs.best_params_)

# ── 3b. RandomizedSearchCV (faster on large grids) ────────────
rs = RandomizedSearchCV(
    pipeline,
    param_grid,
    n_iter=20,                  # number of random combinations to try
    scoring="accuracy",
    cv=StratifiedKFold(5),
    n_jobs=-1,
    random_state=42,
    verbose=1,
)
rs.fit(X_train, y_train)

print("Best score: ", round(rs.best_score_, 3))
print("Best params:", rs.best_params_)

# ── 4. Evaluate Best Model on Test Set ────────────────────────
best_model = gs.best_estimator_   # or rs.best_estimator_
y_pred = best_model.predict(X_test)

print("Test Accuracy:", accuracy_score(y_test, y_pred))
```