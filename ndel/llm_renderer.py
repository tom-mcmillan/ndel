from __future__ import annotations

import json
from typing import Any, Callable, Dict

from ndel.config import NdelConfig
from ndel.semantic_model import Pipeline
from ndel.serialization import pipeline_to_dict

# User-supplied LLM callback; NDEL does not know how this is implemented.
LLMGenerate = Callable[[str], str]


def build_ndel_prompt(pipeline: Pipeline, config: NdelConfig | None = None) -> str:
    """
    Build a text prompt to send to an LLM, instructing it to write an NDEL-style
    description of the given Pipeline.

    The prompt makes clear:
    - The LLM is free to phrase descriptions non-deterministically while preserving semantics.
    - Output should be indentation-based NDEL text (pipeline, datasets, transformations, features, models, metrics).
    - The provided structure is ground truth; do not hallucinate extra entities.
    - Privacy/abstraction may have been applied already; do not re-infer sensitive details.
    """

    data: Dict[str, Any] = pipeline_to_dict(pipeline)
    pipeline_json = json.dumps(data, indent=2, sort_keys=True)

    prompt_parts: list[str] = []
    prompt_parts.append(
        "You are an assistant that writes NDEL (Non-Deterministic Expression Language) "
        "descriptions for data science and machine learning pipelines."
    )
    prompt_parts.append(
        "NDEL is a descriptive, post-facto DSL. Your job is to take a structured "
        "pipeline description and produce a human-readable NDEL text that:\n"
        "- captures datasets, transformations, features, models, and metrics,\n"
        "- preserves the semantics and lineage,\n"
        "- uses indentation-based sections (e.g., pipeline, datasets, transformations, features, models, metrics),\n"
        "- is free to vary phrasing while staying faithful to the structure."
    )
    prompt_parts.append(
        "You have freedom in how you phrase descriptions and explain steps, as long as you remain accurate "
        "and stay within the NDEL DSL style."
    )
    prompt_parts.append(
        "Here is the pipeline structure as JSON. Use it as ground truth; do not hallucinate extra datasets, "
        "models, or metrics that are not present."
    )
    prompt_parts.append(pipeline_json)
    prompt_parts.append(
        "Now, write an NDEL description of this pipeline. Use a clear, indentation-based text layout. "
        "You may choose headings like 'pipeline', 'datasets', 'transformations', 'features', 'models', 'metrics'. "
        "Focus on expressing the relationships and meanings, not on recreating code. The output should be pure NDEL "
        "text, with no additional commentary."
    )

    return "\n\n".join(prompt_parts)


def render_pipeline_with_llm(
    pipeline: Pipeline,
    llm_generate: LLMGenerate,
    config: NdelConfig | None = None,
) -> str:
    """
    Render a Pipeline to NDEL text by building an LLM prompt and delegating
    to a user-supplied LLM callback. NDEL itself has no knowledge of API keys
    or providers; the caller owns LLM access.
    """

    prompt = build_ndel_prompt(pipeline, config=config)
    return llm_generate(prompt)


__all__ = ["LLMGenerate", "build_ndel_prompt", "render_pipeline_with_llm"]
