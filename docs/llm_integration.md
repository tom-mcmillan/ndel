# Integrating Your LLM with NDEL

NDEL provides structure (Pipeline + DSL). Your LLM provides the non-deterministic language. NDEL never calls providers or owns API keys—you supply a callback.

## Core callback

```python
from ndel.llm_renderer import LLMGenerate, render_pipeline_with_llm

def my_llm_generate(prompt: str) -> str:
    # Call your LLM provider (OpenAI, Anthropic, local, etc.) and return text
    ...
```

## High-level describe APIs

```python
from ndel import (
    describe_python_source_with_llm,
    describe_sql_source_with_llm,
    describe_sql_and_python_with_llm,
    NdelConfig,
)

source = "df = 1"  # example Python
text = describe_python_source_with_llm(source, llm_generate=my_llm_generate, config=NdelConfig())
```

These functions:
- Analyze code into a structured Pipeline.
- Build an NDEL prompt for the LLM.
- Delegate to your `llm_generate` callback.

## Deterministic renderer (fallback)

`render_pipeline` and non-LLM `describe_*` are deterministic, useful for debugging and tests. The intended flow is structure → LLM → NDEL text.

## Notes
- Keep `llm_generate` provider-agnostic; no HTTP logic lives inside NDEL.
- Privacy/abstraction is applied during analysis/config; the LLM should not re-infer sensitive details.
