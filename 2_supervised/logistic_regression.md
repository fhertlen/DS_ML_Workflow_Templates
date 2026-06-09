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
# ── 0. Imports ─────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.preprocessing import StandardScaler

# ── 1. Load Data ─────────────────────────────────────────────
df = pd.read_csv("data.csv")
df.head()

# ── 2. Define Features & Target ──────────────────────────────
X = df[["feature_1", "feature_2", "feature_3"]]
y = df["target"]

# ── 3. Train-Test Split ───────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# ── 4. Data Cleaning (on train only — no leakage!) ───────────

# Check for missing values
X_train.isna().sum()

# Impute numerical columns with median (robust to outliers)
median_val = X_train["num_feature"].median()
X_train["num_feature"] = X_train["num_feature"].fillna(median_val)
X_test["num_feature"]  = X_test["num_feature"].fillna(median_val)

# Drop columns with too many NaNs or no predictive value
X_train = X_train.drop(columns=["irrelevant_col"])
X_test  = X_test.drop(columns=["irrelevant_col"])

# Encode binary categorical feature
X_train["cat_feature"] = X_train["cat_feature"].map({"class_a": 0, "class_b": 1})
X_test["cat_feature"]  = X_test["cat_feature"].map({"class_a": 0, "class_b": 1})

# Encode nominal categorical feature (no ordinal relationship)
X_train = pd.get_dummies(X_train, columns=["nominal_feature"], drop_first=True)
X_test  = pd.get_dummies(X_test,  columns=["nominal_feature"], drop_first=True)

# ── 5. EDA (on training data only) ───────────────────────────
df_train = X_train.copy()
df_train["target"] = y_train

# Overview
df_train.info()                        # dtypes, non-null counts
df_train.describe()                    # summary statistics
y_train.value_counts(normalize=True)   # class balance

# Distributions & separability
sns.pairplot(df_train, hue="target", corner=True)

# Correlations
sns.heatmap(df_train.corr(numeric_only=True), annot=True, cmap="YlGnBu")

# ── 6. Baseline Model ─────────────────────────────────────────
y_baseline = np.where(simple_condition, 1, 0)
print("Baseline Accuracy:", accuracy_score(y_test, y_baseline))

# ── 7. Train Model ────────────────────────────────────────────
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ── 8. Evaluate ───────────────────────────────────────────────
y_pred = model.predict(X_test)
print("Train Accuracy:", accuracy_score(y_train, model.predict(X_train)))
print("Test Accuracy: ", accuracy_score(y_test, y_pred))

# ConfusionMatrixDisplay
print(confusion_matrix(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(cm).plot()

# classification_report
print(classification_report(y_test, y_pred, target_names=["class_0", "class_1"]))

# ROC Curve + AUC
RocCurveDisplay.from_predictions(y_test, y_proba).plot()
print("AUC:", roc_auc_score(y_test, y_proba))


# ── 9. Custom Threshold (optional) ───────────────────────────
y_proba = model.predict_proba(X_test)[:, 1]
y_pred_custom = np.where(y_proba >= 0.4, 1, 0)
```