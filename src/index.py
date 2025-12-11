"""FastMCP server and public API surface for NDEL."""

from __future__ import annotations

import os
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastmcp import FastMCP

from src import __version__ as NDEL_VERSION
from src.config import AbstractionLevel, DomainConfig, NdelConfig, PrivacyConfig
from src.schema import (
    CONFIG_SCHEMA,
    PIPELINE_SCHEMA,
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
    validate_pipeline_structure,
    validate_schema,
)
from src.analysis import analyze_python_source, analyze_sql_source
from src.formatter import (
    describe_grammar,
    NDEL_GRAMMAR,
    render_pipeline,
    validate_ndel_text,
    build_ndel_prompt,
    render_pipeline_with_llm,
    DEFAULT_NDEL_PROMPT,
)

mcp = FastMCP(name="ndel")
ROOT_DIR = Path(__file__).resolve().parents[2]

DOC_PATHS: Dict[str, Path] = {
    "readme": ROOT_DIR / "README.md",
}

# Register resources for MCP: README, grammar, prompt.
RESOURCES: Dict[str, Dict[str, str]] = {
    "readme": {"uri": "ndel://docs/readme", "name": "README", "description": "NDEL overview and usage", "path": str(DOC_PATHS["readme"])},
    "grammar": {"uri": "ndel://docs/grammar", "name": "NDEL Grammar", "description": "NDEL grammar text"},
    "prompt": {"uri": "ndel://docs/prompt", "name": "NDEL Prompt Template", "description": "Default NDEL LLM prompt"},
    "tools": {
        "uri": "ndel://docs/tools",
        "name": "NDEL Tools",
        "description": "List of available MCP tools and their purpose",
    },
}


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


def _abstraction_from_str(raw: str | None, default: AbstractionLevel = AbstractionLevel.MEDIUM) -> AbstractionLevel:
    if not raw:
        return default
    raw_lower = raw.lower()
    for level in AbstractionLevel:
        if level.value == raw_lower:
            return level
    return default


def _privacy_from_env() -> PrivacyConfig:
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


def _load_file_config(config_path: Path | None) -> Dict[str, Any]:
    if not config_path:
        return {}
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _discover_config_file(start_dir: Path) -> Path | None:
    for candidate in [start_dir / ".ndel.yml", start_dir / ".ndel.yaml", start_dir / ".ndel.json"]:
        if candidate.exists():
            return candidate
    return None


def _config_schema_errors(payload: Dict[str, Any]) -> List[str]:
    return validate_schema(payload, CONFIG_SCHEMA)


def _schema_errors(payload: Dict[str, Any]) -> List[str]:
    return validate_schema(payload, PIPELINE_SCHEMA)


def _build_config(payload: Optional[Dict[str, Any]] = None, config_path: str | None = None) -> NdelConfig:
    payload = payload or {}
    schema_errs = _config_schema_errors(payload)
    if schema_errs:
        raise ValueError(f"Config payload failed schema validation: {schema_errs}")

    file_config_path = Path(config_path) if config_path else _discover_config_file(Path.cwd())
    file_config = _load_file_config(file_config_path)

    privacy = _privacy_from_env()
    domain = _domain_from_env()
    abstraction = _abstraction_from_str(os.getenv("NDEL_ABSTRACTION"))

    file_privacy = file_config.get("privacy") if isinstance(file_config, dict) else None
    if isinstance(file_privacy, dict):
        privacy = PrivacyConfig(
            hide_table_names=file_privacy.get("hide_table_names", privacy.hide_table_names),
            hide_file_paths=file_privacy.get("hide_file_paths", privacy.hide_file_paths),
            redact_identifiers=file_privacy.get("redact_identifiers", privacy.redact_identifiers),
            max_literal_length=file_privacy.get("max_literal_length", privacy.max_literal_length),
        )

    file_domain = file_config.get("domain") if isinstance(file_config, dict) else None
    if isinstance(file_domain, dict):
        domain = DomainConfig(
            dataset_aliases=file_domain.get("dataset_aliases", domain.dataset_aliases),
            model_aliases=file_domain.get("model_aliases", domain.model_aliases),
            feature_aliases=file_domain.get("feature_aliases", domain.feature_aliases),
            pipeline_name=file_domain.get("pipeline_name", domain.pipeline_name),
        )

    file_abstraction = file_config.get("abstraction") if isinstance(file_config, dict) else None
    abstraction = _abstraction_from_str(file_abstraction, abstraction)

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
        abstraction = _abstraction_from_str(str(payload["abstraction"]), abstraction)

    return NdelConfig(privacy=privacy, domain=domain, abstraction=abstraction)


def _safe_execute(func, *args, **kwargs):
    """Run a tool body with optional safety net; honors NDEL_DEBUG to raise."""

    debug = _bool_env("NDEL_DEBUG", False)
    try:
        return func(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - defensive
        if debug:
            raise
        return {"error": str(exc), "type": exc.__class__.__name__, "code": "SERVER_ERROR"}


def _filter_fields(cls, data: Dict[str, Any]) -> Dict[str, Any]:
    allowed = {f.name for f in fields(cls)}
    return {k: v for k, v in data.items() if k in allowed}


def _dict_to_pipeline(data: Dict[str, Any]) -> Pipeline:
    schema_errors = _schema_errors(data)
    if schema_errors:
        raise ValueError(f"Pipeline payload failed schema validation: {schema_errors}")

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


@mcp.tool
async def ndel_grammar() -> Dict[str, str]:
    """Return the human-readable NDEL grammar description."""
    return {"grammar": describe_grammar()}


@mcp.tool
async def validate_ndel(text: str) -> Dict[str, Any]:
    """Validate NDEL text against expected shape; returns warnings if any."""

    warnings = validate_ndel_text(text)
    return {"warnings": warnings, "is_valid": len(warnings) == 0}


@mcp.tool
async def describe_python_text(source: str, config: Optional[Dict[str, Any]] = None) -> str:
    """Analyze Python DS/ML code and return deterministic NDEL text."""

    def _run():
        ndel_config = _build_config(config)
        pipeline = analyze_python_source(source, config=ndel_config)
        return render_pipeline(pipeline, config=ndel_config)

    return _safe_execute(_run)


@mcp.tool
async def describe_sql_text(sql: str, config: Optional[Dict[str, Any]] = None) -> str:
    """Analyze SQL text and return deterministic NDEL text."""

    def _run():
        ndel_config = _build_config(config)
        pipeline = analyze_sql_source(sql, config=ndel_config)
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
        schema_errs_old = _schema_errors(old_pipeline)
        schema_errs_new = _schema_errors(new_pipeline)
        if schema_errs_old or schema_errs_new:
            return {
                "error": "schema_validation_failed",
                "details": {
                    "old_pipeline_errors": schema_errs_old,
                    "new_pipeline_errors": schema_errs_new,
                },
            }
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
async def validate_config_tool(
    config: Dict[str, Any],
    pipeline: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate NdelConfig against a pipeline dict."""

    def _run():
        schema_errs = _schema_errors(pipeline)
        if schema_errs:
            return {"issues": [{"kind": "error", "code": "SCHEMA_INVALID", "message": "; ".join(schema_errs)}]}
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
async def validate_pipeline_structural(
    pipeline: Dict[str, Any],
) -> Dict[str, Any]:
    """Validate structural integrity of a pipeline (unique names, known references)."""

    def _run():
        schema_errs = _schema_errors(pipeline)
        if schema_errs:
            return {"issues": [{"kind": "error", "code": "SCHEMA_INVALID", "message": "; ".join(schema_errs)}]}
        pipe = _dict_to_pipeline(pipeline)
        issues = validate_pipeline_structure(pipe)
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
    """Create a minimal NDEL pipeline from high-level intent text."""

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
        return render_pipeline(placeholder, config=ndel_config)

    return _safe_execute(_run)


@mcp.tool
async def list_docs() -> Dict[str, Any]:
    """List available NDEL documentation resources."""

    return _safe_execute(
        lambda: {
            "docs": [
                {"name": meta["name"], "uri": meta["uri"], "description": meta.get("description", "")}
                for meta in RESOURCES.values()
            ]
        }
    )


@mcp.tool
async def get_doc(name: str) -> Dict[str, str]:
    """Retrieve a documentation file by name."""

    def _run():
        if name == "grammar":
            return {"name": "grammar", "content": NDEL_GRAMMAR}
        if name == "prompt":
            return {"name": "prompt", "content": DEFAULT_NDEL_PROMPT}
        if name == "tools":
            tools = [
                "describe_python_text",
                "describe_sql_text",
                "describe_sql_and_python_text",
                "describe_python_json",
                "describe_sql_json",
                "describe_sql_and_python_json",
                "pipeline_diff",
                "validate_config_tool",
                "validate_pipeline_structural",
                "build_prompt",
                "ndel_grammar",
                "validate_ndel",
                "ndel_prompt_template",
                "synthesize_ndel",
                "health",
                "list_docs",
                "get_doc",
            ]
            return {"name": "tools", "content": "\n".join(tools)}
        if name not in RESOURCES or "path" not in RESOURCES[name]:
            raise ValueError(f"Unknown doc: {name}")
        path = Path(RESOURCES[name]["path"])
        if not path.exists():
            raise FileNotFoundError(f"Doc not found: {path}")
        return {"name": name, "content": path.read_text(encoding="utf-8")}

    return _safe_execute(_run)


@mcp.tool
async def ndel_prompt_template() -> Dict[str, str]:
    """Return the default deterministic NDEL prompt template (for LLM rendering)."""

    return _safe_execute(lambda: {"prompt": DEFAULT_NDEL_PROMPT, "grammar": NDEL_GRAMMAR})


async def _health_impl() -> Dict[str, Any]:
    """Lightweight health check with version and basic config flags."""

    return {
        "status": "ok",
        "version": NDEL_VERSION,
        "privacy_safe_mode": _bool_env("NDEL_PRIVACY_SAFE", False),
    }


health = mcp.tool(_health_impl)


def main() -> None:
    """Console entrypoint for the NDEL MCP server (stdio transport)."""

    mcp.run()


__all__ = ["main"]
