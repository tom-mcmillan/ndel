from __future__ import annotations

import json
from typing import Any, Callable, Dict

from src.config import NdelConfig
from src.types import Pipeline, pipeline_to_dict
from src.prompt import DEFAULT_NDEL_PROMPT

# User-supplied LLM callback; NDEL does not know how this is implemented.
LLMGenerate = Callable[[str], str]


def build_ndel_prompt(
    pipeline: Pipeline,
    config: NdelConfig | None = None,
    extra_instructions: str | None = None,
) -> str:
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

    base = DEFAULT_NDEL_PROMPT.format(pipeline_json=pipeline_json)
    prompt_parts: list[str] = [base]
    if extra_instructions:
        prompt_parts.append(extra_instructions.strip())
    return "\n\n".join(prompt_parts)


def render_pipeline_with_llm(
    pipeline: Pipeline,
    llm_generate: LLMGenerate,
    config: NdelConfig | None = None,
    extra_instructions: str | None = None,
) -> str:
    """
    Render a Pipeline to NDEL text by building an LLM prompt and delegating
    to a user-supplied LLM callback. NDEL itself has no knowledge of API keys
    or providers; the caller owns LLM access.
    """

    prompt = build_ndel_prompt(pipeline, config=config, extra_instructions=extra_instructions)
    return llm_generate(prompt)


__all__ = ["LLMGenerate", "build_ndel_prompt", "render_pipeline_with_llm"]
