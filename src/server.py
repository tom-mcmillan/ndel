"""FastMCP server exposing NDEL analysis tools."""

from __future__ import annotations

import os
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP

from src import __version__ as NDEL_VERSION
from src.config import AbstractionLevel, DomainConfig, NdelConfig, PrivacyConfig
from src.pipeline import (
    Dataset,
    Feature,
    Metric,
    Model,
    Pipeline,
    Transformation,
    diff_pipelines,
    merge_pipelines,
    pipeline_to_dict,
    validate_config_against_pipeline,
)
from src.llmrenderer import build_ndel_prompt
from src.analyzer import analyze_python_source, analyze_sql_source
from src.grammar import describe_grammar, validate_ndel_text


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
# NDEL grammar helpers
# ---------------------------------------------------------------------------


@mcp.tool
async def ndel_grammar() -> Dict[str, str]:
    """Return the human-readable NDEL grammar description."""

    return {"grammar": describe_grammar()}


@mcp.tool
async def validate_ndel(text: str) -> Dict[str, Any]:
    """Validate NDEL text against expected shape; returns warnings if any."""

    warnings = validate_ndel_text(text)
    return {"warnings": warnings, "is_valid": len(warnings) == 0}


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
    # Safe mode enables conservative defaults if explicit envs are not provided.
    safe_mode = _bool_env("NDEL_PRIVACY_SAFE", False)
    default_redactions = ["email", "ip"] if safe_mode else []
    return PrivacyConfig(
        hide_table_names=_bool_env("NDEL_HIDE_TABLE_NAMES", safe_mode),
        hide_file_paths=_bool_env("NDEL_HIDE_PATHS", safe_mode),
        redact_identifiers=_list_env("NDEL_REDACT_IDENTIFIERS") or default_redactions,
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


def _safe_execute(func, *args, **kwargs):
    """Run a tool body with optional safety net; honors NDEL_DEBUG to raise."""

    debug = _bool_env("NDEL_DEBUG", False)
    try:
        return func(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - defensive
        if debug:
            raise
        return {"error": str(exc), "type": exc.__class__.__name__}


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

    def _run():
        ndel_config = _build_config(config)
        pipeline = analyze_python_source(source, config=ndel_config)
        from src.render import render_pipeline
        return render_pipeline(pipeline, config=ndel_config)

    return _safe_execute(_run)


@mcp.tool
async def describe_sql_text(sql: str, config: Optional[Dict[str, Any]] = None) -> str:
    """Analyze SQL text and return deterministic NDEL text."""

    def _run():
        ndel_config = _build_config(config)
        pipeline = analyze_sql_source(sql, config=ndel_config)
        from src.render import render_pipeline
        return render_pipeline(pipeline, config=ndel_config)

    return _safe_execute(_run)


@mcp.tool
async def describe_sql_and_python_text(
    sql: str,
    py_source: str,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Analyze SQL then Python, merge lineage, and return deterministic NDEL text."""

    def _run():
        ndel_config = _build_config(config)
        p_sql = analyze_sql_source(sql, config=ndel_config)
        p_py = analyze_python_source(py_source, config=ndel_config)
        merged = merge_pipelines(p_sql, p_py)
        from src.render import render_pipeline
        return render_pipeline(merged, config=ndel_config)

    return _safe_execute(_run)


@mcp.tool
async def describe_python_json(source: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Analyze Python and return pipeline JSON."""

    return _safe_execute(
        lambda: pipeline_to_dict(analyze_python_source(source, config=_build_config(config)))
    )


@mcp.tool
async def describe_sql_json(sql: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Analyze SQL and return pipeline JSON."""

    return _safe_execute(
        lambda: pipeline_to_dict(analyze_sql_source(sql, config=_build_config(config)))
    )


@mcp.tool
async def describe_sql_and_python_json(
    sql: str,
    py_source: str,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Analyze SQL then Python, merge lineage, and return pipeline JSON."""

    def _run():
        ndel_config = _build_config(config)
        p_sql = analyze_sql_source(sql, config=ndel_config)
        p_py = analyze_python_source(py_source, config=ndel_config)
        merged = merge_pipelines(p_sql, p_py)
        return pipeline_to_dict(merged)

    return _safe_execute(_run)


@mcp.tool
async def pipeline_diff(
    old_pipeline: Dict[str, Any],
    new_pipeline: Dict[str, Any],
) -> Dict[str, Any]:
    """Diff two pipelines (dict form) and return summary plus structured diff."""

    def _run():
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

    return _safe_execute(_run)


@mcp.tool
async def validate_config(
    config: Dict[str, Any],
    pipeline: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate NdelConfig against a pipeline dict."""

    def _run():
        ndel_config = _build_config(config)
        pipe = _dict_to_pipeline(pipeline)
        issues = validate_config_against_pipeline(ndel_config, pipe)
        return {
            "issues": [
                {"kind": issue.kind, "code": issue.code, "message": issue.message}
                for issue in issues
            ]
        }

    return _safe_execute(_run)


@mcp.tool
async def build_prompt(pipeline: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> str:
    """Build an LLM-ready NDEL prompt from a pipeline dict."""

    return _safe_execute(
        lambda: build_ndel_prompt(_dict_to_pipeline(pipeline), config=_build_config(config))
    )


@mcp.tool
async def synthesize_ndel(
    intent: str,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Create a minimal NDEL pipeline from high-level intent text.

    This is a bridge for conversational agents: it emits a stub Pipeline with
    the intent as description and a single dataset placeholder, then renders
    deterministically. Callers can combine with ndel_grammar/validate_ndel for
    stricter checks.
    """

    def _run():
        ndel_config = _build_config(config)
        placeholder = Pipeline(
            name=ndel_config.domain.pipeline_name or "intent_pipeline",
            datasets=[Dataset(name="input", description="user intent input")],
            transformations=[],
            features=[],
            models=[],
            metrics=[],
            description=intent,
        )
        from src.render import render_pipeline
        return render_pipeline(placeholder, config=ndel_config)

    return _safe_execute(_run)


@mcp.tool
async def list_docs() -> Dict[str, Any]:
    """List available NDEL documentation resources."""

    return _safe_execute(
        lambda: {
            "docs": [
                {"name": name, "path": str(path)}
                for name, path in DOC_PATHS.items()
                if path.exists()
            ]
        }
    )


@mcp.tool
async def get_doc(name: str) -> Dict[str, str]:
    """Retrieve a documentation file by name."""

    return _safe_execute(lambda: {"name": name, "content": _read_doc(name)})


async def _health_impl() -> Dict[str, Any]:
    """Lightweight health check with version and basic config flags."""

    return {
        "status": "ok",
        "version": NDEL_VERSION,
        "privacy_safe_mode": _bool_env("NDEL_PRIVACY_SAFE", False),
    }


health = mcp.tool(_health_impl)


if __name__ == "__main__":
    mcp.run()


def main() -> None:
    """Console entrypoint for the NDEL MCP server (stdio transport)."""

    mcp.run()


__all__ = ["main"]
