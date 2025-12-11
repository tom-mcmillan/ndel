"""FastMCP server exposing NDEL analysis tools."""

from __future__ import annotations

import os
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from ndel.config import AbstractionLevel, DomainConfig, NdelConfig, PrivacyConfig
from ndel.diff import diff_pipelines
from ndel.lineage import merge_pipelines
from ndel.llm_renderer import build_ndel_prompt
from ndel.py_analyzer import analyze_python_source
from ndel.semantic_model import Dataset, Feature, Metric, Model, Pipeline, Transformation
from ndel.serialization import pipeline_to_dict
from ndel.sql_analyzer import analyze_sql_source
from ndel.validation import validate_config_against_pipeline


mcp = FastMCP(name="ndel")
ROOT_DIR = Path(__file__).resolve().parents[2]

DOC_PATHS: Dict[str, Path] = {
    "readme": ROOT_DIR / "README.md",
    "overview": ROOT_DIR / "docs" / "overview.md",
    "philosophy": ROOT_DIR / "docs" / "philosophy.md",
    "cookbook_ci": ROOT_DIR / "docs" / "cookbook_ci_integration.md",
    "cookbook_churn": ROOT_DIR / "docs" / "cookbook_churn_pipeline.md",
    "cookbook_feature_store": ROOT_DIR / "docs" / "cookbook_custom_feature_store_detector.md",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bool_env(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in {"1", "true", "yes", "on"}


def _list_env(name: str) -> list[str]:
    val = os.getenv(name)
    if not val:
        return []
    return [part.strip() for part in val.split(",") if part.strip()]


def _abstraction_from_env(default: AbstractionLevel = AbstractionLevel.MEDIUM) -> AbstractionLevel:
    raw = os.getenv("NDEL_ABSTRACTION")
    if not raw:
        return default
    raw_lower = raw.lower()
    for level in AbstractionLevel:
        if level.value == raw_lower:
            return level
    return default


def _privacy_from_env() -> PrivacyConfig:
    return PrivacyConfig(
        hide_table_names=_bool_env("NDEL_HIDE_TABLE_NAMES", False),
        hide_file_paths=_bool_env("NDEL_HIDE_PATHS", False),
        redact_identifiers=_list_env("NDEL_REDACT_IDENTIFIERS"),
        max_literal_length=int(os.getenv("NDEL_MAX_LITERAL_LEN", "200")),
    )


def _domain_from_env() -> DomainConfig:
    return DomainConfig(
        pipeline_name=os.getenv("NDEL_PIPELINE_NAME"),
    )


def _build_config(payload: Optional[Dict[str, Any]] = None) -> NdelConfig:
    payload = payload or {}

    # Start with env defaults
    privacy = _privacy_from_env()
    domain = _domain_from_env()
    abstraction = _abstraction_from_env()

    # Overlay payload values if provided
    if "privacy" in payload and payload["privacy"] is not None:
        p = payload["privacy"]
        privacy = PrivacyConfig(
            hide_table_names=p.get("hide_table_names", privacy.hide_table_names),
            hide_file_paths=p.get("hide_file_paths", privacy.hide_file_paths),
            redact_identifiers=p.get("redact_identifiers", privacy.redact_identifiers),
            max_literal_length=p.get("max_literal_length", privacy.max_literal_length),
        )

    if "domain" in payload and payload["domain"] is not None:
        d = payload["domain"]
        domain = DomainConfig(
            dataset_aliases=d.get("dataset_aliases", domain.dataset_aliases),
            model_aliases=d.get("model_aliases", domain.model_aliases),
            feature_aliases=d.get("feature_aliases", domain.feature_aliases),
            pipeline_name=d.get("pipeline_name", domain.pipeline_name),
        )

    if "abstraction" in payload and payload["abstraction"] is not None:
        raw_abs = str(payload["abstraction"]).lower()
        abstraction = next((lvl for lvl in AbstractionLevel if lvl.value == raw_abs), abstraction)

    return NdelConfig(privacy=privacy, domain=domain, abstraction=abstraction)


def _filter_fields(cls, data: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in allowed}


def _dict_to_pipeline(data: Dict[str, Any]) -> Pipeline:
    datasets = [
        Dataset(**_filter_fields(Dataset, ds))
        for ds in data.get("datasets", [])
    ]
    transformations = [
        Transformation(**_filter_fields(Transformation, t))
        for t in data.get("transformations", [])
    ]
    features = [
        Feature(**_filter_fields(Feature, f))
        for f in data.get("features", [])
    ]
    models = [
        Model(**_filter_fields(Model, m))
        for m in data.get("models", [])
    ]
    metrics = [
        Metric(**_filter_fields(Metric, m))
        for m in data.get("metrics", [])
    ]

    return Pipeline(
        name=data.get("name", "pipeline"),
        datasets=datasets,
        transformations=transformations,
        features=features,
        models=models,
        metrics=metrics,
        description=data.get("description"),
    )


def _read_doc(name: str) -> str:
    if name not in DOC_PATHS:
        raise ValueError(f"Unknown doc: {name}")
    path = DOC_PATHS[name]
    if not path.exists():
        raise FileNotFoundError(f"Doc not found: {path}")
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool
async def describe_python_text(source: str, config: Optional[Dict[str, Any]] = None) -> str:
    """Analyze Python DS/ML code and return deterministic NDEL text."""

    ndel_config = _build_config(config)
    pipeline = analyze_python_source(source, config=ndel_config)
    # Render deterministically via render_pipeline (called inside describe)
    from ndel.render import render_pipeline
    return render_pipeline(pipeline, config=ndel_config)


@mcp.tool
async def describe_sql_text(sql: str, config: Optional[Dict[str, Any]] = None) -> str:
    """Analyze SQL text and return deterministic NDEL text."""

    ndel_config = _build_config(config)
    pipeline = analyze_sql_source(sql, config=ndel_config)
    from ndel.render import render_pipeline
    return render_pipeline(pipeline, config=ndel_config)


@mcp.tool
async def describe_sql_and_python_text(
    sql: str,
    py_source: str,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Analyze SQL then Python, merge lineage, and return deterministic NDEL text."""

    ndel_config = _build_config(config)
    p_sql = analyze_sql_source(sql, config=ndel_config)
    p_py = analyze_python_source(py_source, config=ndel_config)
    merged = merge_pipelines(p_sql, p_py)
    from ndel.render import render_pipeline
    return render_pipeline(merged, config=ndel_config)


@mcp.tool
async def describe_python_json(source: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Analyze Python and return pipeline JSON."""

    ndel_config = _build_config(config)
    pipeline = analyze_python_source(source, config=ndel_config)
    return pipeline_to_dict(pipeline)


@mcp.tool
async def describe_sql_json(sql: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Analyze SQL and return pipeline JSON."""

    ndel_config = _build_config(config)
    pipeline = analyze_sql_source(sql, config=ndel_config)
    return pipeline_to_dict(pipeline)


@mcp.tool
async def describe_sql_and_python_json(
    sql: str,
    py_source: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Analyze SQL then Python, merge lineage, and return pipeline JSON."""

    ndel_config = _build_config(config)
    p_sql = analyze_sql_source(sql, config=ndel_config)
    p_py = analyze_python_source(py_source, config=ndel_config)
    merged = merge_pipelines(p_sql, p_py)
    return pipeline_to_dict(merged)


@mcp.tool
async def pipeline_diff(
    old_pipeline: Dict[str, Any],
    new_pipeline: Dict[str, Any],
) -> Dict[str, Any]:
    """Diff two pipelines (dict form) and return summary plus structured diff."""

    old = _dict_to_pipeline(old_pipeline)
    new = _dict_to_pipeline(new_pipeline)
    d = diff_pipelines(old, new)

    summary_parts: List[str] = []
    def add(label: str, items: List[str]) -> None:
        if items:
            summary_parts.append(f"{label}: {', '.join(items)}")

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

    return {
        "summary": "\n".join(summary_parts) if summary_parts else "No differences",
        "diff": {
            "datasets_added": d.datasets_added,
            "datasets_removed": d.datasets_removed,
            "transformations_added": d.transformations_added,
            "transformations_removed": d.transformations_removed,
            "features_added": d.features_added,
            "features_removed": d.features_removed,
            "models_added": d.models_added,
            "models_removed": d.models_removed,
            "metrics_added": d.metrics_added,
            "metrics_removed": d.metrics_removed,
        },
    }


@mcp.tool
async def validate_config(
    config: Dict[str, Any],
    pipeline: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate NdelConfig against a pipeline dict."""

    ndel_config = _build_config(config)
    pipe = _dict_to_pipeline(pipeline)
    issues = validate_config_against_pipeline(ndel_config, pipe)
    return {
        "issues": [
            {"kind": issue.kind, "code": issue.code, "message": issue.message}
            for issue in issues
        ]
    }


@mcp.tool
async def build_prompt(pipeline: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> str:
    """Build an LLM-ready NDEL prompt from a pipeline dict."""

    ndel_config = _build_config(config)
    pipe = _dict_to_pipeline(pipeline)
    return build_ndel_prompt(pipe, config=ndel_config)


@mcp.tool
async def list_docs() -> Dict[str, Any]:
    """List available NDEL documentation resources."""

    available = [
        {"name": name, "path": str(path)}
        for name, path in DOC_PATHS.items()
        if path.exists()
    ]
    return {"docs": available}


@mcp.tool
async def get_doc(name: str) -> Dict[str, str]:
    """Retrieve a documentation file by name."""

    content = _read_doc(name)
    return {"name": name, "content": content}


if __name__ == "__main__":
    mcp.run()
