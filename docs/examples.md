# NDEL Examples (Quick Reference)

## Python analysis

```python
from ndel import describe_python_source

code = """
import pandas as pd
df = pd.read_csv("s3://bucket/data.csv")
df = df[df["score"] > 0.5]
"""

print(describe_python_source(code))
```

## SQL analysis

```python
from ndel import describe_sql_source

sql = """
select user_id, avg(score) as avg_score
from events.users
where country = 'US'
group by user_id
"""

print(describe_sql_source(sql))
```

## MCP usage

Run the server and configure your client:

```bash
ndel-mcp
```

Client snippet (Cursor/Claude Code/VS Code):

```json
{
  "mcpServers": {
    "ndel": { "command": "ndel-mcp" }
  }
}
```
