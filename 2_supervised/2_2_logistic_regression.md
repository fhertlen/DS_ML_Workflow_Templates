## Logistic Regression Workflow

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
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay,
)

# ── 1. Load & Preprocess (see 1_1_fetch_data.md) ─────────────
# ── 2. Data Cleaning     (see 1_2_data_cleaning.md) ──────────
# ── 3. EDA               (see 1_3_EDA.md) ────────────────────

# ── 4. Baseline Model ─────────────────────────────────────────
y_baseline = np.where(simple_condition, 1, 0)
print("Baseline Accuracy:", accuracy_score(y_test, y_baseline))

# ── 5. Build Pipeline ─────────────────────────────────────────
col_scale = X_train.select_dtypes(include="number").columns.tolist()
ct = ColumnTransformer(
    [("scaling", StandardScaler(), col_scale)], remainder="passthrough"
)

pipeline = Pipeline([
    ("preprocessing", ct),
    ("model", LogisticRegression(max_iter=1000)),
])

# ── 6. Cross-Validate ─────────────────────────────────────────
scores = cross_val_score(pipeline, X_train, y_train, cv=StratifiedKFold(5), n_jobs=-1)
print("CV Accuracy: {:.2f} ± {:.2f}".format(scores.mean(), scores.std()))

# ── 7. Train & Evaluate on Test Set ───────────────────────────
pipeline.fit(X_train, y_train)
y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("Train Accuracy:", accuracy_score(y_train, pipeline.predict(X_train)))
print("Test Accuracy: ", accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()

print(classification_report(y_test, y_pred, target_names=["class_0", "class_1"]))

RocCurveDisplay.from_predictions(y_test, y_proba).plot()
print("AUC:", roc_auc_score(y_test, y_proba))

# ── 8. Custom Threshold (optional) ───────────────────────────
y_pred_custom = np.where(y_proba >= 0.4, 1, 0)

# ── 9. Hyperparameter Tuning (see 2_3_model_tuning.md) ───────
```