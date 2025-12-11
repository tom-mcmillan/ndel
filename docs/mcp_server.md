# NDEL MCP Server

Run NDEL as a local MCP server (stdio transport), Crash-style. It exposes tools for Python/SQL analysis, diffing, validation, and prompt building. No external data sources or proprietary services are contacted.

## Install and run locally

```bash
pipx install ndel   # or: pip install ndel
ndel-mcp            # starts stdio MCP server
```

Python 3.10+ required. Logs go to stderr; stdout is reserved for MCP messages.

## MCP client configuration (stdio)

Point your MCP client at the `ndel-mcp` command. Example (Cursor/Claude Code/VS Code):

```json
{
  "mcpServers": {
    "ndel": { "command": "ndel-mcp" }
  }
}
```

## Environment configuration

- `NDEL_ABSTRACTION` (`high|medium|low`)
- `NDEL_HIDE_TABLE_NAMES` (`true|false`)
- `NDEL_HIDE_PATHS` (`true|false`)
- `NDEL_REDACT_IDENTIFIERS` (comma list, e.g., `email,ip`)
- `NDEL_MAX_LITERAL_LEN` (int)
- `NDEL_PIPELINE_NAME` (string)
- `NDEL_PRIVACY_SAFE` (`true|false`) — if true, defaults to hiding tables/paths and redacts `email,ip` unless overridden
- `NDEL_DEBUG` (`true|false`) — bubble up exceptions instead of wrapping errors

## Tools exposed

- `describe_python_text` / `describe_python_json`
- `describe_sql_text` / `describe_sql_json`
- `describe_sql_and_python_text` / `describe_sql_and_python_json`
- `pipeline_diff`
- `validate_config`
- `build_prompt`
- `list_docs` / `get_doc`
- `health`

## Docker

```bash
docker build -t ndel-mcp .
docker run --rm ndel-mcp
```

Override env vars to tune privacy/abstraction.

## Hosting options

- **Recommended default:** stdio transport via local process or remote exec (e.g., SSH to a VM/container and run `ndel-mcp`).
- **HTTP/SSE (not implemented here):** MCP spec allows a streamable HTTP transport. If your client requires HTTP, place a thin HTTP→stdio bridge/proxy in front of `ndel-mcp` and secure it (auth token, TLS, firewall).

## Security notes

- Stdio isolates access to the invoking client; prefer it for local use.
- If exposing over HTTP, require auth and restrict network access.
