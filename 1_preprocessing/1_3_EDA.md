## EDA Workflow

**Position in DS pipeline:**

```
... → Train-Test Split → EDA (on X_train only!) → Post-Split DC → Model Training
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── 1. Structural Overview ────────────────────────────────────
df_train = X_train.copy()
df_train["target"] = y_train

df_train.shape          # (rows, cols)
df_train.info()         # dtypes, non-null counts
df_train.describe()     # mean, std, min/max, quartiles

# ── 2. Missing Values ─────────────────────────────────────────
df_train.isna().sum()
df_train.isna().mean().sort_values(ascending=False)  # as fraction

# ── 3. Target Distribution ────────────────────────────────────
# Classification
y_train.value_counts(normalize=True)  # class balance

# Regression
y_train.hist(bins=30)

# ── 4. Univariate Distributions ───────────────────────────────
# Numerical features
df_train.hist(bins=30, figsize=(12, 8))
plt.tight_layout()

# Single feature
sns.histplot(df_train["num_feature"], kde=True)

# Categorical features
df_train["cat_feature"].value_counts().plot(kind="bar")

# ── 5. Outlier Detection ──────────────────────────────────────
sns.boxplot(data=df_train[["num_feature_1", "num_feature_2"]])

# IQR-based flag (inspect, don't drop blindly)
Q1 = df_train["num_feature"].quantile(0.25)
Q3 = df_train["num_feature"].quantile(0.75)
IQR = Q3 - Q1
outliers = df_train[(df_train["num_feature"] < Q1 - 1.5 * IQR) |
                    (df_train["num_feature"] > Q3 + 1.5 * IQR)]

# ── 6. Correlations ───────────────────────────────────────────
sns.heatmap(df_train.corr(numeric_only=True), annot=True,
            fmt=".2f", cmap="coolwarm", center=0)

# Top correlations with target
df_train.corr(numeric_only=True)["target"].sort_values(ascending=False)

# ── 7. Feature vs. Target ─────────────────────────────────────
# Numerical feature vs. classification target
sns.boxplot(x="target", y="num_feature", data=df_train)

# Numerical feature vs. regression target
sns.scatterplot(x="num_feature", y="target", data=df_train)

# Categorical feature vs. target
sns.countplot(x="cat_feature", hue="target", data=df_train)

# Pairplot (separability overview — can be slow on many features)
sns.pairplot(df_train, hue="target", corner=True)
```