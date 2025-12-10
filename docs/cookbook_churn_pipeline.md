# Documenting a Subscription-Churn-Style Pipeline with NDEL

This guide walks through a synthetic, subscription-churn-like pipeline and shows how to document it with NDEL. The pipeline is generic and neutral—no real data or sensitive domains are involved.

## 1) The pipeline code

See `examples/python_churn_pipeline.py` for a small pandas + sklearn workflow:
- Load `subscriptions.csv` into a DataFrame.
- Filter active users.
- Create features like `sessions_per_month` and `tenure_months`.
- Train a `LogisticRegression` to predict `churned`.
- Compute AUROC via `roc_auc_score`.

## 2) Running NDEL against the pipeline

You can analyze the source string or the callable itself. Example:

```python
from ndel import describe_callable, NdelConfig, PrivacyConfig, DomainConfig
from examples.python_churn_pipeline import train_churn_model

config = NdelConfig(
    privacy=PrivacyConfig(hide_table_names=True, redact_identifiers=["email", "ip"]),
    domain=DomainConfig(pipeline_name="churn_prediction_pipeline"),
)

ndel_text = describe_callable(train_churn_model, config=config)
print(ndel_text)
```

You can also store config in a project `ndel_profile.py` (see the bootstrap prompt in `docs/bootstrap_ndel_config_prompt.md`).

## 3) Example NDEL output (snippet)

See `examples/python_churn_pipeline.ndel.txt` for a fuller rendering. A short snippet:

```
pipeline "churn_prediction_pipeline":
  datasets:
    - name: df
  transformations:
    - name: trans_filter_1
      kind: filter
  models:
    - name: model
      algorithm_family: LogisticRegression
      inputs:
        - sessions_per_month
        - tenure_months
```

## 4) How the semantics map

- **Datasets**: inferred from data loads (`pd.read_csv`).
- **Transformations**: filters, feature engineering (new columns), joins/aggregations when present.
- **Features**: columns derived or selected for training.
- **Models**: estimators like `LogisticRegression`, with inputs traced from features.
- **Metrics**: `roc_auc_score` captured as an evaluation metric.

NDEL is post-facto and static: it analyzes AST/SQL text and does not execute or mutate your code. The goal is a trustworthy, shareable description of what the pipeline does.
