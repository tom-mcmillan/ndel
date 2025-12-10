# NDEL - Narrative Descriptive Expression Language

NDEL is a **post-facto descriptive DSL** for data science and machine learning. It statically analyzes existing Python and SQL (no execution, no code generation) to produce human-readable descriptions that connect datasets, transformations, features, models, and metrics.

## Conceptual Overview

- **Library-first**: import and call APIs; there is no CLI. Main entry points: `describe_python_source`, `describe_sql_source`, `describe_callable`, `describe_sql_and_python`, plus helpers for diffs (`describe_pipeline_diff`) and config validation (`validate_config`).
- **Semantic model**: Pipelines contain datasets (with sources), ordered transformations (filter, aggregation, join, feature_engineering, other), features (with origins), models (with inputs/metadata), and metrics (with higher-is-better hints).
- **Config & privacy**: `NdelConfig` bundles `PrivacyConfig` (hide table names/paths, redact identifiers, truncate literals), `DomainConfig` (aliases and pipeline name), and `AbstractionLevel` (HIGH/MEDIUM/LOW). Validation surfaces unknown aliases and privacy/abstraction hints.
- **Analyzers**: Python analyzer detects pandas IO, filters, assignments, groupbys/aggregations, merges/concats, chained ops, sklearn Pipelines/ColumnTransformers, preprocessing, features in model training, and common metrics. SQL analyzer parses FROM/JOIN/WHERE/GROUP BY/SELECT-derived columns into datasets, transformations, and derived features. Lineage merge aligns SQL-produced datasets with Python pipelines via dataset sources.
- **Rendering & provenance**: `render_pipeline` outputs an indented, privacy-aware DSL with lineage (inputs/outputs/origins). Abstraction controls detail level. Privacy redaction avoids leaking sensitive names.
- **Diffs & CI checks**: `diff_pipelines`/`describe_pipeline_diff` summarize added/removed datasets, transformations, features, models, metrics. Validation returns issues instead of raising, suitable for CI.

## Use Cases

- Document DS/ML pipelines in notebooks or scripts.
- Describe SQL analytics queries for stakeholders.
- Publish shareable docs from private codebases.
- Track semantic diffs across versions and validate configs in CI.

## Quick Start (Library Usage)

```python
from ndel import (
    describe_callable,
    NdelConfig,
    PrivacyConfig,
    DomainConfig,
)

from my_project.pipelines import train_churn_model

config = NdelConfig(
    privacy=PrivacyConfig(
        hide_table_names=True,
        hide_file_paths=True,
        redact_identifiers=["email", "ip"],
    ),
    domain=DomainConfig(
        dataset_aliases={"df_users": "users_activity_30d"},
        model_aliases={"model": "churn_prediction_model"},
        pipeline_name="churn_prediction_pipeline",
    ),
)

ndel_text = describe_callable(train_churn_model, config=config)
print(ndel_text)
```

Config is optional when experimenting; start with `describe_python_source` or `describe_callable` alone. A project-specific `ndel_profile.py` can be generated and versioned later.

## How it Works (Conceptual)

Implementation code → parsed (Python/SQL) → semantic model (datasets, transformations, features, models, metrics, lineage) → rendered NDEL text with privacy/abstraction applied. SQL and Python pipelines can be merged for unified lineage (SQL → Python → model).

## Further Reading

- [Architecture Overview](docs/overview.md)
- [Bootstrap Config Prompt](docs/bootstrap_ndel_config_prompt.md)
- [Cookbook: Churn-Style Pipeline](docs/cookbook_churn_pipeline.md)
- [Cookbook: CI Integration](docs/cookbook_ci_integration.md)
- [Cookbook: Custom Feature Store Detector](docs/cookbook_custom_feature_store_detector.md)

## Status

Early and experimental. Expect rapid iteration and changes.
