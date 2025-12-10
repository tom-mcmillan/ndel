from __future__ import annotations

import inspect
import textwrap
from collections.abc import Callable

from ndel.config import NdelConfig
from ndel.py_analyzer import analyze_python_source
from ndel.sql_analyzer import analyze_sql_source
from ndel.render import render_pipeline
from ndel.lineage import merge_pipelines
from ndel.diff import diff_pipelines
from ndel.validation import ValidationIssue, validate_config_against_pipeline


def describe_python_source(source: str, config: NdelConfig | None = None) -> str:
    """Analyze Python DS/ML code into NDEL text.

    This is static: code is not executed. The optional config can influence
    naming (aliases) and, in the future, privacy and abstraction behavior.
    """

    pipeline = analyze_python_source(source, config=config)
    return render_pipeline(pipeline, config=config)


def describe_callable(func: Callable, config: NdelConfig | None = None) -> str:
    """Analyze a callable's source code and render NDEL text.

    Uses inspect.getsource under the hood. Raises RuntimeError if the source
    cannot be retrieved (e.g. built-in or dynamically generated).
    """

    try:
        source = inspect.getsource(func)
    except OSError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(f"Could not retrieve source for {func!r}") from exc

    return describe_python_source(textwrap.dedent(source), config=config)


def describe_sql_source(sql: str, config: NdelConfig | None = None) -> str:
    """Analyze SQL text and render an NDEL pipeline description."""

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
