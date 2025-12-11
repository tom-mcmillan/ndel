# NDEL – Non-Deterministic Expression Language

NDEL is a post-facto descriptive DSL for data science and machine learning. It statically analyzes Python and SQL (no execution) to build a pipeline graph (datasets, transformations, features, models, metrics) and renders human-readable NDEL text with optional privacy redaction and LLM-driven phrasing.

## What it does
- Static analyzers for Python (pandas/sklearn patterns) and SQL to extract pipeline semantics.
- Deterministic renderer or LLM renderer (`describe_*_with_llm`) for flexible wording.
- Privacy and domain controls (hide names/paths, redact identifiers, aliases, abstraction level).
- Diff and validation helpers for CI/workflow checks.
- MCP server (`ndel-mcp`) for stdio clients.

## Install
```bash
pip install ndel          # or: pipx install ndel
```

## Library quickstart
```python
from ndel import describe_python_source, describe_sql_source

py_code = """
import pandas as pd
df = pd.read_csv("s3://bucket/data.csv")
df = df[df["score"] > 0.5]
"""

sql = """
select user_id, avg(score) as avg_score
from events.users
where country = 'US'
group by user_id
"""

print(describe_python_source(py_code))
print(describe_sql_source(sql))
```

## MCP server (stdio)
Run the server:
```bash
ndel-mcp
```
Client snippet (Cursor/Claude Code/VS Code):
```json
{
  "mcpServers": { "ndel": { "command": "ndel-mcp" } }
}
```

## Configuration (env)
- `NDEL_ABSTRACTION` = `high|medium|low`
- `NDEL_HIDE_TABLE_NAMES` = `true|false`
- `NDEL_HIDE_PATHS` = `true|false`
- `NDEL_REDACT_IDENTIFIERS` = comma list (e.g., `email,ip`)
- `NDEL_MAX_LITERAL_LEN` = int
- `NDEL_PIPELINE_NAME` = string
- `NDEL_PRIVACY_SAFE` = `true|false` (safer defaults)
- `NDEL_DEBUG` = `true|false` (bubble up exceptions)

## Examples
See `examples/` for curated samples; older demos live in `examples/legacy/`.

## Development
- Run tests: `pytest`
- Build/publish: standard Python packaging via `pyproject.toml`

License: Apache-2.0
