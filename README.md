# NDEL — Non-Deterministic Expression Language for DS/ML Pipelines

NDEL is a domain-specific language for describing data science and machine learning pipelines in a human-readable, privacy-safe, and abstract form.  
It acts as a translation layer between Python/SQL code and end users who need to understand what computations were performed on their behalf — without exposing internal implementation details.

NDEL does not execute code.  
NDEL does not show Python or SQL to end users.  
Instead, it statically analyzes DS/ML code to produce a semantic pipeline graph, then renders it as explanatory NDEL text.

NDEL is conceptually similar to CEL (Common Expression Language):  
a structured IR, deterministic semantics, and optional natural-language rendering.  
But while CEL is fully deterministic, NDEL optionally uses an LLM for non-deterministic descriptive output.

---

## Why NDEL Exists

When systems run DS/ML pipelines for users, we need to answer:

• What data was used?  
• What filters, joins, or transforms were applied?  
• How were features engineered?  
• What model was trained?  
• What metrics were computed?

But we must not expose:

• Python code  
• SQL queries  
• Table or column names  
• File paths  
• Internal business logic  

NDEL provides a safe explanatory DSL for communicating pipeline behavior.

---

## What NDEL Does

Given Python and/or SQL code, NDEL performs:

### 1. Static Analysis
Recover pipeline semantics from:
- pandas I/O
- filters, masks, groupbys, merges
- sklearn pipelines and column transformers
- SQL joins, aggregates, derived columns
- model training and evaluation

### 2. Semantic Pipeline Graph
A deterministic intermediate representation (IR) describing:
- datasets and origins
- lineage and transformations
- derived features
- model steps
- metrics

### 3. Privacy and Abstraction
Configurable via environment:
- hide table names
- hide filesystem paths
- redact identifiers
- alias datasets
- choose abstraction level (HIGH, MEDIUM, LOW)
- privacy-safe preset

### 4. Rendering
Two rendering modes:
- deterministic renderer → canonical NDEL text
- optional LLM renderer using your own LLM callback

### 5. Diff, Validation, Serialization
NDEL can compare pipelines, enforce privacy rules, and export JSON representations.

---

## Architecture Overview

Flat but separated modules:

- config.py — privacy/abstraction/domain config and merge helpers  
- schema.py — IR dataclasses, diff/merge, invariants, JSON Schemas, serialization  
- analysis.py — Python AST + sqlglot SQL analyzers → Pipeline  
- formatter.py — deterministic renderer, grammar/validator, privacy helpers, LLM prompt/render  
- types.py — shared type aliases (LLM callbacks, detectors)  
- index.py — MCP server/tools and public entrypoints  

This keeps concerns clear without nested packages.

---

## The NDEL MCP Server

Key tools (stdio via `ndel-mcp`):
- `describe_python_text` / `describe_sql_text` / `describe_sql_and_python_text` → deterministic NDEL text  
- `describe_*_json` → pipeline JSON  
- `pipeline_diff` → structured diff + summary  
- `validate_config_tool` / `validate_pipeline_structural` → config and IR checks  
- `build_prompt` → LLM-ready prompt; `ndel_grammar` / `validate_ndel` helpers  
- `synthesize_ndel` → minimal stub from intent  
- `health`, `list_docs`, `get_doc`

The MCP server reads configuration entirely from environment variables, giving it a stable and predictable personality.

---

## Development Status

NDEL is currently undergoing a full architectural rewrite.  
Legacy tests and examples have been removed while the new IR, config system, analyzers, and renderer are rebuilt from first principles.

---

## Running Locally

- Recommended install:
  - Create venv: `python3 -m venv .venv && source .venv/bin/activate`
  - Upgrade pip: `python -m pip install --upgrade pip`
  - Install: `python -m pip install -e '.[dev]'` (includes `sqlglot`; regex fallback if missing)
- MCP server: `ndel-mcp` (or `python -m ndel.index`) reads `.ndel.yml` or env for privacy/aliases/abstraction
- Tests: `pytest`
- LLM rendering: supply your own callback via `render_pipeline_with_llm`; NDEL never calls an API directly.

## Validation and Schemas

- Pipelines are validated against a JSON Schema and IR invariants (unique names, known references).
- Config payloads are schema-validated; privacy and aliases are honored upstream.
- MCP tools expose deterministic rendering and validation helpers (`validate_config`, `validate_pipeline_structural`, `pipeline_diff`, `build_prompt`).

---

## Roadmap

- New config system (privacy and abstraction)
- Formalized pipeline IR
- Updated Python and SQL analyzers
- Deterministic and LLM rendering pipeline
- Minimal API surface
- crash-style MCP adapter
- New test suite
- Examples and documentation

---

## License

MIT
