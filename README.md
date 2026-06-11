# ds-workflows

Personal collection of generic, reusable Data Science workflow templates.
Designed as working templates and cheat sheets — not tutorials.

## Structure

```
ds-workflows/
├── preprocessing/
│   ├── 1_1_fetch_data.md         # DB query via SQLAlchemy / CSV load
│   ├── 1_2_data_cleaning.md      # Pre- and post-split cleaning pipeline
│   └── 1_3_EDA.md                # Exploratory data analysis on X_train
│
├── supervised/
│   ├── 2_1_linear_regression.md  # Regression pipeline + metrics
│   ├── 2_2_logistic_regression.md # Classification pipeline + metrics
│   └── 2_3_model_tuning.md       # GridSearchCV / RandomizedSearchCV
│
└── unsupervised/                  # coming soon
```

## Usage

Each model template (`supervised/`) references the preprocessing templates
by filename. Run them in order:

```
1_1_fetch_data → 1_2_data_cleaning → 1_3_EDA → 2_x_<model>
```

Preprocessing steps are intentionally kept separate to avoid redundancy
across model templates.