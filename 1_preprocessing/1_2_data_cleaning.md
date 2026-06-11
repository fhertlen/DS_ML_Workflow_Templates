## Data Cleaning Workflow

**Pragmatic order in a DS project:**

```
Load Data
 └─ Pre-Split Data Cleaning
     └─ Train-Test Split
         └─ EDA
             └─ Post-Split Data Cleaning
                 └─ Model Training
```

```python
# ── 0. Imports ────────────────────────────────────────────────
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# ── 1. Load Data (see 1_1_fetch_data.md) ─────────────────────

# ══ PRE-SPLIT: Structural Fixes (no leakage risk) ════════════

# ── 2. Column Selection ───────────────────────────────────────
df = df[["feature_1", "feature_2", "feature_3", "target"]]

# ── 3. Rename Columns ─────────────────────────────────────────
df = df.rename(columns={
    "Feature 1": "feature_1",
    "Identifier": "id",
})

# ── 4. Fix Dtypes ─────────────────────────────────────────────
df["date_feature"]  = pd.to_datetime(df["date_feature"])
df["int_feature"]   = df["int_feature"].astype(int)
df["float_feature"] = df["float_feature"].astype(float)

# Nullable integer (preserves NaN before imputation)
df["int_feature"] = pd.array(df["int_feature"], dtype=pd.Int64Dtype())

# ── 5. Drop & Filter ──────────────────────────────────────────
df = df.drop_duplicates()
df = df.drop(columns=["id", "irrelevant_col"])  # no predictive value

df = df[df["feature"] == "value"]               # filter by value
df = df.dropna(subset=["feature"])              # drop rows with NaN in key column
df = df.reset_index(drop=True)

# ── 6. Train-Test Split ───────────────────────────────────────
X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── → EDA here (see 1_3_EDA.md) ──────────────────────────────

# ══ POST-SPLIT: Fit on Train, Transform Both ══════════════════

# ── 7. Missing Values ─────────────────────────────────────────
X_train.isna().sum()  # inspect

# Numerical: impute with median (robust to outliers)
median_val = X_train["num_feature"].median()
X_train["num_feature"] = X_train["num_feature"].fillna(median_val)
X_test["num_feature"]  = X_test["num_feature"].fillna(median_val)

# Categorical: impute with mode
mode_val = X_train["cat_feature"].mode()[0]
X_train["cat_feature"] = X_train["cat_feature"].fillna(mode_val)
X_test["cat_feature"]  = X_test["cat_feature"].fillna(mode_val)

# ── 8. Encoding ───────────────────────────────────────────────

# Binary categorical
X_train["binary_feature"] = X_train["binary_feature"].map({"class_a": 0, "class_b": 1})
X_test["binary_feature"]  = X_test["binary_feature"].map({"class_a": 0, "class_b": 1})

# Nominal categorical (no ordinal relationship)
X_train = pd.get_dummies(X_train, columns=["nominal_feature"], drop_first=True)
X_test  = pd.get_dummies(X_test,  columns=["nominal_feature"], drop_first=True)

# Align columns in case test set is missing a dummy category
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# ── 9. Scaling (optional — required for distance-based models) ─
col_scale = X_train.select_dtypes(include="number").columns.tolist()

ct = ColumnTransformer(
    [("scaling", StandardScaler(), col_scale)], remainder="passthrough"
)

X_train = ct.fit_transform(X_train)
X_test  = ct.transform(X_test)
```