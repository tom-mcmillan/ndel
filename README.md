# NDEL - Narrative Descriptive Expression Language

NDEL is a descriptive DSL that restates the intent of existing Python and SQL used in data science and machine learning. It is used post-facto inside DS/ML projects: you import the library and it describes your existing code without executing it or generating new code. The implementation remains the source of truth; NDEL provides a higher-level, human-readable, and shareable explanation of what that code does, especially when the original cannot be shown publicly.

## What is NDEL?

NDEL captures the semantics of DS/ML pipelines and analytics queries in concise prose. It mirrors the behavior of underlying Python and SQL so stakeholders can understand the logic, assumptions, and flow without seeing the private implementation.

## Use Cases

- Documenting DS/ML pipelines built in Python notebooks or scripts
- Describing SQL analytics queries for business stakeholders
- Generating public-facing docs from private or sensitive codebases

## Quick Start (Library Usage)

Import NDEL in your Python project; there is no CLI. Example:

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

Config is optional when experimenting; you can start with `describe_python_source` or `describe_callable` alone. A project-specific config file (e.g., `ndel_profile.py`) can be added and versioned later.

## Configuration & Domain Adaptation

`NdelConfig` bundles:
- `PrivacyConfig` for hiding sensitive details (table names, file paths, redaction keywords, literal length limits).
- `DomainConfig` for mapping code-level names to human-friendly aliases and setting a pipeline name.
- `AbstractionLevel` to control how detailed descriptions should be (high/medium/low).

In many projects, you’ll want a tailored `NdelConfig` that understands your datasets, models, and privacy requirements. We recommend generating an initial `ndel_profile.py` with an LLM that can see your codebase, then reviewing and versioning it like any other code. A bootstrap prompt for this will be provided separately.

## How it works (conceptual)

Implementation code → parsed → NDEL semantic model → rendered as NDEL text. The generated description traces back to the real code, ensuring accuracy while remaining readable and shareable.

## Status

Early and experimental. Expect rapid iteration and changes.
