# NDEL - Non-Deterministic Expression Language

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
- [NDEL Philosophy](docs/philosophy.md)
- [Cookbook: Churn-Style Pipeline](docs/cookbook_churn_pipeline.md)
- [Cookbook: CI Integration](docs/cookbook_ci_integration.md)
- [Cookbook: Custom Feature Store Detector](docs/cookbook_custom_feature_store_detector.md)
- [MCP Server Guide](docs/mcp_server.md)

## MCP Server Quickstart (local)

NDEL ships with an MCP server you can run locally over stdio (Crash-style). After install, start it with the `ndel-mcp` console script.

### Install and run

```bash
pipx install ndel  # or: pip install ndel
ndel-mcp           # starts stdio MCP server
```

### Configure in MCP clients

- **Cursor/Claude Code/VS Code (stdio command):**
  ```json
  {
    "mcpServers": {
      "ndel": { "command": "ndel-mcp" }
    }
  }
  ```

### Configuration knobs (env)

- `NDEL_ABSTRACTION` (`high|medium|low`)
- `NDEL_HIDE_TABLE_NAMES` (`true|false`)
- `NDEL_HIDE_PATHS` (`true|false`)
- `NDEL_REDACT_IDENTIFIERS` (comma list, e.g., `email,ip`)
- `NDEL_MAX_LITERAL_LEN` (int)
- `NDEL_PIPELINE_NAME` (string)
- `NDEL_PRIVACY_SAFE` (`true|false`) — safer defaults (hide tables/paths, redact email/ip)

### Hosting later

MCP spec also allows HTTP/SSE transports. Today NDEL runs stdio by default; for self-hosting you can:
- Run `ndel-mcp` on a VM/container and let clients exec it via SSH.
- Add a thin HTTP bridge (POST/GET with JSON-RPC) in front of the stdio server if your client supports HTTP transport.

See [MCP Server Guide](docs/mcp_server.md) for a fuller walkthrough, tool list, Docker usage, and hosting notes.

## Status

Early and experimental. Expect rapid iteration and changes.

## NDEL Philosophy: Non-Deterministic Semantics

NDEL is a semantic protocol, not a deterministic compiler. The library extracts structural signals from code—datasets, transformations, features, models, metrics—but an LLM ultimately interprets and phrases the NDEL descriptions. Each repo can adopt its own semantic dialect via LLM-generated config, and phrasing may evolve as LLM capabilities improve. Treat NDEL as a linguistic lens over your pipelines, not as a strict AST-to-DSL translator.

**What we are building:** a semantic interface for LLMs to understand DS/ML pipelines. NDEL turns messy Python/SQL/notebooks into a structured Pipeline graph plus a DSL schema. The LLM (user-supplied) writes the final NDEL text non-deterministically within that schema. The goal is controlled semantic generation: structure + privacy + domain hints from NDEL; language and narrative from the LLM.

**Why it matters:**
- LLMs need structure to avoid hallucinations and lineage mistakes; NDEL provides it.
- It is an explanation layer, not an execution layer—focused on meaning, documentation, audits, and cross-team communication.
- Privacy is preserved by redaction and aliases; you can externalize semantics without leaking secrets.

## NDEL and LLMs

- NDEL provides structure (Pipeline + DSL) and constraints; an external LLM writes the NDEL description.
- NDEL never owns API keys or calls LLM providers. You supply a callback.
- Deterministic `render_pipeline` is a fallback/debug tool; primary text generation should use `*_with_llm` APIs.

Example (pseudo-LLM callback):

```python
from ndel import (
    describe_callable_with_llm,
    NdelConfig,
    PrivacyConfig,
    DomainConfig,
)
from my_project.pipelines import train_model

def my_llm_generate(prompt: str) -> str:
    # Call your LLM provider here and return response text
    return "pipeline \"example\":\n  # LLM text"

config = NdelConfig(
    privacy=PrivacyConfig(hide_table_names=True),
    domain=DomainConfig(pipeline_name="subscription_churn_pipeline"),
)

ndel_text = describe_callable_with_llm(train_model, llm_generate=my_llm_generate, config=config)
print(ndel_text)
```
