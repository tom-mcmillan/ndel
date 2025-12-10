# Integrating NDEL into CI

This guide shows how to run NDEL in CI to keep semantic documentation and privacy checks aligned with your codebase. All examples are generic and CI-provider-agnostic.

## Why run NDEL in CI?
- Ensure pipelines still produce valid NDEL descriptions.
- Keep `NdelConfig` consistent with the codebase (aliases, pipeline naming, abstraction levels).
- Catch obvious PII-like identifiers that are not redacted.

## Example CI step (pseudo-YAML)

```yaml
jobs:
  ndel-check:
    steps:
      - install project deps + ndel
      - run python -m ndel_ci_check.py
      - upload ndel artifacts (optional)
```

## Minimal validation script (conceptual)

```python
import sys
from ndel import describe_callable, validate_config
from ndel.validation import ValidationIssue
from ndel.lineage import merge_pipelines
from ndel.api import describe_sql_source

from ndel_profile import make_ndel_config  # your project config
from my_project.pipelines import train_churn_model

config = make_ndel_config()

pipeline_py = describe_callable(train_churn_model, config=config)  # or analyze_python_source
# Optionally analyze SQL and merge
pipeline_sql = describe_sql_source("SELECT 1")  # placeholder
pipeline = merge_pipelines(pipeline_sql, pipeline_py)

issues = validate_config(config, pipeline)
errors = [i for i in issues if i.kind == "error"]
for issue in issues:
    print(f"{issue.kind}: {issue.code} - {issue.message}")

if errors:
    sys.exit(1)
```

## Generating NDEL artifacts in CI
- Run `describe_python_source` / `describe_callable` / `describe_sql_source` for core pipelines.
- Save the rendered NDEL text (via `render_pipeline`) as build artifacts for docs.
- Optional: diff against previous NDEL outputs using `describe_pipeline_diff` to spot semantic changes.

## Tips
- Keep CI fast: analyze a representative subset of pipelines.
- Fail on critical issues (e.g., unknown aliases marked as errors), warn on others.
- Store `ndel_profile.py` alongside code; regenerate with the bootstrap prompt when schemas or models change.
