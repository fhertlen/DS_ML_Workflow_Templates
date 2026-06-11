## Linear Regression Workflow

**Structure identical to `2_2_logistic_regression.md` — only deltas listed here.**

```python
# ── 0. Imports ────────────────────────────────────────────────
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# ── 1. Load & Preprocess (see 1_1_fetch_data.md) ─────────────
# ── 2. Data Cleaning     (see 1_2_data_cleaning.md) ──────────
# ── 3. EDA               (see 1_3_EDA.md) ────────────────────
# ── 4. Baseline Model ────────────────────────────────────────
y_baseline = np.full(y_test.shape, y_train.mean())
print("Baseline RMSE:", np.sqrt(mean_squared_error(y_test, y_baseline)))

# ── 5. Build Pipeline ─────────────────────────────────────────
col_scale = X_train.select_dtypes(include="number").columns.tolist()
ct = ColumnTransformer(
    [("scaling", StandardScaler(), col_scale)], remainder="passthrough"
)

pipeline = Pipeline([
    ("preprocessing", ct),
    ("model", LinearRegression()),
])

# ── 6. Cross-Validate ─────────────────────────────────────────
scores = cross_val_score(
    pipeline, X_train, y_train,
    cv=KFold(5, shuffle=True, random_state=42),
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
)
print("CV RMSE: {:.2f} ± {:.2f}".format(-scores.mean(), scores.std()))

# ── 7. Train & Evaluate on Test Set ───────────────────────────
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAE:  ", mean_absolute_error(y_test, y_pred))
print("R²:   ", r2_score(y_test, y_pred))

# ── 8. Hyperparameter Tuning (see 2_x_model_tuning.md) ───────
```