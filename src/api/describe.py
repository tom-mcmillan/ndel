from __future__ import annotations

import inspect
import textwrap
from collections.abc import Callable

from src.config import NdelConfig
from src.rendering.llm_renderer import LLMGenerate, render_pipeline_with_llm
from src.pipeline import Pipeline, diff_pipelines, merge_pipelines, pipeline_to_dict, pipeline_to_json, validate_config_against_pipeline
from src.analyzers.python_analyzer import analyze_python_source
from src.rendering.render import render_pipeline
from src.analyzers.sql_analyzer import analyze_sql_source
from src.pipeline import ValidationIssue


def describe_python_source(source: str, config: NdelConfig | None = None) -> str:
    """Analyze Python DS/ML code and render deterministic NDEL text (fallback path)."""

    pipeline = analyze_python_source(source, config=config)
    return render_pipeline(pipeline, config=config)


def describe_callable(func: Callable, config: NdelConfig | None = None) -> str:
    """Analyze a callable's source code and render deterministic NDEL text (fallback)."""

    try:
        source = inspect.getsource(func)
    except OSError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(f"Could not retrieve source for {func!r}") from exc

    return describe_python_source(textwrap.dedent(source), config=config)


def describe_sql_source(sql: str, config: NdelConfig | None = None) -> str:
    """Analyze SQL text and render a deterministic NDEL pipeline description (fallback)."""

    pipeline = analyze_sql_source(sql, config=config)
    return render_pipeline(pipeline, config=config)


__all__ = ["describe_python_source", "describe_callable", "describe_sql_source"]


def describe_sql_and_python(sql: str, py_source: str, config: NdelConfig | None = None) -> str:
    """Analyze SQL then Python, merge pipelines, and render unified NDEL text."""

    p_sql = analyze_sql_source(sql, config=config)
    p_py = analyze_python_source(py_source, config=config)
    merged = merge_pipelines(p_sql, p_py)
    return render_pipeline(merged, config=config)


__all__.append("describe_sql_and_python")


def validate_config(config: NdelConfig, pipeline: Pipeline) -> list[ValidationIssue]:
    return validate_config_against_pipeline(config, pipeline)


__all__.append("validate_config")


def describe_pipeline_diff(old: Pipeline, new: Pipeline) -> str:
    """Return a human-readable summary of differences between two Pipelines."""

    d = diff_pipelines(old, new)
    parts: list[str] = []

    def add(label: str, items: list[str]):
        if items:
            parts.append(f"{label}: {', '.join(items)}")

    add("Datasets added", d.datasets_added)
    add("Datasets removed", d.datasets_removed)
    add("Transformations added", d.transformations_added)
    add("Transformations removed", d.transformations_removed)
    add("Features added", d.features_added)
    add("Features removed", d.features_removed)
    add("Models added", d.models_added)
    add("Models removed", d.models_removed)
    add("Metrics added", d.metrics_added)
    add("Metrics removed", d.metrics_removed)

    return "\n".join(parts) if parts else "No differences"


__all__.append("describe_pipeline_diff")


# Serialization helpers
__all__.append("pipeline_to_dict")
__all__.append("pipeline_to_json")


def describe_python_source_with_llm(
    source: str,
    llm_generate: LLMGenerate,
    config: NdelConfig | None = None,
) -> str:
    """
    Analyze Python DS/ML source code into a Pipeline and render it via an external LLM.

    The caller MUST provide an LLM callback (llm_generate) that takes a prompt string
    and returns an NDEL text response. NDEL does not call any LLM APIs on its own.
    """

    pipeline = analyze_python_source(source, config=config)
    return render_pipeline_with_llm(pipeline, llm_generate=llm_generate, config=config)


def describe_callable_with_llm(
    func: Callable,
    llm_generate: LLMGenerate,
    config: NdelConfig | None = None,
) -> str:
    """Analyze the source of a Python callable and render it via an external LLM."""

    try:
        source = inspect.getsource(func)
    except OSError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(f"Could not retrieve source for {func!r}") from exc

    return describe_python_source_with_llm(textwrap.dedent(source), llm_generate=llm_generate, config=config)


def describe_sql_source_with_llm(
    sql: str,
    llm_generate: LLMGenerate,
    config: NdelConfig | None = None,
) -> str:
    """Analyze SQL text into a Pipeline and render it via an external LLM."""

    pipeline = analyze_sql_source(sql, config=config)
    return render_pipeline_with_llm(pipeline, llm_generate=llm_generate, config=config)


def describe_sql_and_python_with_llm(
    sql: str,
    py_source: str,
    llm_generate: LLMGenerate,
    config: NdelConfig | None = None,
) -> str:
    """Analyze both SQL and Python pipelines, merge them, and render via an external LLM."""

    sql_pipeline = analyze_sql_source(sql, config=config)
    py_pipeline = analyze_python_source(py_source, config=config)
    merged = merge_pipelines(sql_pipeline, py_pipeline)
    return render_pipeline_with_llm(merged, llm_generate=llm_generate, config=config)


__all__ += [
    "describe_python_source_with_llm",
    "describe_callable_with_llm",
    "describe_sql_source_with_llm",
    "describe_sql_and_python_with_llm",
]
