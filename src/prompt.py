"""
Default LLM prompt template for rendering NDEL pipelines.

Use this as a starting point when wiring your own `llm_generate` callback:

    from ndel.prompt import DEFAULT_NDEL_PROMPT
    prompt = DEFAULT_NDEL_PROMPT.format(pipeline_json=json_payload)
    llm_generate(prompt)
"""

DEFAULT_NDEL_PROMPT = """You write NDEL (Non-Deterministic Expression Language) descriptions for data/ML pipelines.
You receive a pipeline JSON with datasets, transformations, features, models, and metrics.
- Preserve structure and lineage; do not invent entities.
- Output indentation-based NDEL text (sections like pipeline, datasets, transformations, features, models, metrics).
- Privacy/abstraction may already be applied; do not re-infer sensitive details.
- Follow the NDEL grammar and allowed sections/kinds provided below.

Pipeline JSON (ground truth):
{pipeline_json}

Grammar and rules:
{ndel_grammar}

Write the NDEL description now, using only the provided structure."""

__all__ = ["DEFAULT_NDEL_PROMPT"]
