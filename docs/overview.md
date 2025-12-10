# NDEL Overview

NDEL is a post-facto descriptive DSL for data science and machine learning code. It statically analyzes existing Python and SQL (no execution, no code generation) to produce human-readable descriptions that connect datasets, transformations, features, models, and metrics.

## Purpose & High-Level Design
- Describe existing DS/ML pipelines without running code.
- Bridge Python and SQL into a unified, human-readable lineage.
- Keep implementation code as the source of truth; NDEL renders descriptive semantics.

## Library-First API Surface
- Primary entry points: `describe_python_source`, `describe_sql_source`, `describe_callable`, `describe_sql_and_python`.
- Helpers: `describe_pipeline_diff` (semantic diffs), `validate_config` (config/pipeline checks).
- Rendering: `render_pipeline` produces an indentation-based DSL.
- Diffing: `diff_pipelines` + `describe_pipeline_diff` summarize changes.
- Validation: `validate_config_against_pipeline` returns issues instead of raising.

Quick example:
```python
from ndel import describe_python_source

code = """
import pandas as pd
df = pd.read_csv('data.csv')
df = df[df["x"] > 0]
"""

print(describe_python_source(code))
```

## Semantic Model
- **Pipeline**: name, datasets, transformations, features, models, metrics, description. Captures ordered lineage.
- **Dataset**: `name`, `source` (table/path identifier), `description`, `source_type`, `notes`.
- **Transformation**: `name`, `description`, `kind` (`filter`, `aggregation`, `join`, `feature_engineering`, `other`), `inputs`, `outputs`.
- **Feature**: `name`, `description`, `origin` (dataset or transformation), `data_type`.
- **Model**: `name`, `task`, `algorithm_family`, `inputs`, `target`, `description`, `hyperparameters`.
- **Metric**: `name`, `description`, `dataset`, `higher_is_better`.

Example rendered snippet:
```
pipeline "churn_prediction_pipeline":
  datasets:
    - name: users
  transformations:
    - name: df_filter_1
      kind: filter
      inputs:
        - users
      outputs:
        - users
  models:
    - name: model
      algorithm_family: LogisticRegression
      inputs:
        - x
```

## Configuration & Validation
- **NdelConfig** bundles:
  - `PrivacyConfig`: hide table names/paths, redact identifiers, truncate literals.
  - `DomainConfig`: dataset/model/feature aliases, pipeline name.
  - `AbstractionLevel`: HIGH/MEDIUM/LOW detail level.
- Validation: `validate_config_against_pipeline` (via `validate_config`) warns about unknown aliases, missing PII redaction, abstraction mismatches.

## Analyzers
- **Python analyzer** (AST-based): detects pandas IO, filters, assignments, groupbys/aggregations, merges/concats, chained ops, sklearn Pipelines/ColumnTransformers, preprocessing, features used in model training, common metrics from `sklearn.metrics`. Tracks provenance to link transformations, features, and model inputs.
- **SQL analyzer**: heuristic parsing of FROM/JOIN/WHERE/GROUP BY/SELECT-derived columns into datasets (with sources), transformations (joins, filters, aggregations, projections), and derived features.
- **Lineage merge**: `merge_pipelines` aligns SQL-produced datasets with Python pipelines via dataset sources for unified lineage (SQL → Python → model).

## Rendering
- `render_pipeline` produces an indented DSL.
- Honors `NdelConfig`: abstraction (HIGH/MEDIUM/LOW) and privacy redaction.
- Shows datasets, ordered transformations (inputs/outputs), features (origins), models (inputs), and metrics.

## Diff & Validation APIs
- `diff_pipelines` / `describe_pipeline_diff`: added/removed datasets, transformations, features, models, metrics for semantic diffs across versions.
- `validate_config_against_pipeline` (and `validate_config` helper): returns issues (warnings/errors) instead of raising, suitable for CI.

## Testing Surface
- Coverage across transformations, features, sklearn pipelines/column transformers, metrics detection, SQL analysis, lineage merge, validation, and diffing to keep behavior consistent.
