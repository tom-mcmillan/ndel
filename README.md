# NDEL - Narrative Descriptive Expression Language

NDEL is a descriptive DSL that restates the intent of existing Python and SQL used in data science and machine learning. The implementation code remains the source of truth; NDEL provides a higher-level, human-readable, and shareable explanation of what that code does, especially when the original cannot be shown publicly.

## What is NDEL?

NDEL captures the semantics of DS/ML pipelines and analytics queries in concise prose. It mirrors the behavior of underlying Python and SQL so stakeholders can understand the logic, assumptions, and flow without seeing the private implementation.

## Use Cases

- Documenting DS/ML pipelines built in Python notebooks or scripts
- Describing SQL analytics queries for business stakeholders
- Generating public-facing docs from private or sensitive codebases

## How it works (conceptual)

Implementation code → parsed → NDEL semantic model → rendered as NDEL text. The generated description traces back to the real code, ensuring accuracy while remaining readable and shareable.

## Status

Early and experimental. Expect rapid iteration and changes.
