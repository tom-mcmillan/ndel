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

NDEL follows a clean, layered design:

src/
  analyzers/     Python and SQL static analysis  
  pipeline/      Semantic IR (datasets, lineage, transforms, diff, validation)  
  rendering/     Deterministic and LLM rendering  
  config/        Privacy and abstraction configuration  
  api/           Public entrypoints (describe, diff, validate, prompt)  
  mcp/           Thin MCP adapter exposing NDEL tools  
  utils/         Environment and logging helpers  

This structure ensures clear module boundaries and allows the MCP server to remain a minimal integration layer.

---

## The NDEL MCP Server

The MCP adapter exposes exactly five tools:

### describe_pipeline
Given Python or SQL code, return:
- deterministic NDEL text  
- optional LLM-rendered explanation  
- pipeline IR (JSON)

### diff_pipelines
Diff two pipelines or two NDEL texts.

### validate_pipeline
Validate privacy rules, alias mappings, and structural consistency.

### build_prompt
Return the deterministic NDEL prompt used for the LLM renderer.

### health
Standard MCP health check.

The MCP server reads configuration entirely from environment variables, giving it a stable and predictable personality.

---

## Development Status

NDEL is currently undergoing a full architectural rewrite.  
Legacy tests and examples have been removed while the new IR, config system, analyzers, and renderer are rebuilt from first principles.

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
