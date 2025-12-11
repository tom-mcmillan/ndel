"""NDEL - Describe data operations in human-readable form."""

__version__ = "0.2.0"

from src.config import AbstractionLevel, DomainConfig, NdelConfig, PrivacyConfig
from src.schema import (
    Dataset,
    Feature,
    Metric,
    Model,
    Pipeline,
    Transformation,
    ValidationIssue,
    diff_pipelines,
    merge_pipelines,
    pipeline_to_dict,
    pipeline_to_json,
    validate_config_against_pipeline,
    validate_pipeline_structure,
)
from src.analysis import analyze_python_source, analyze_sql_source, PythonAnalyzer, AnalysisContext
from src.formatter import (
    render_pipeline,
    apply_privacy,
    apply_privacy_to_payload,
    describe_grammar,
    validate_ndel_text,
    build_ndel_prompt,
    render_pipeline_with_llm,
    LLMGenerate,
)

__all__ = [
    # Analysis
    "analyze_python_source",
    "analyze_sql_source",
    "PythonAnalyzer",
    "AnalysisContext",

    # Rendering / formatting
    "render_pipeline",
    "apply_privacy",
    "apply_privacy_to_payload",
    "describe_grammar",
    "validate_ndel_text",
    "validate_pipeline_structure",
    "render_pipeline_with_llm",
    "build_ndel_prompt",
    "LLMGenerate",

    # Semantic model
    "Pipeline",
    "Dataset",
    "Transformation",
    "Feature",
    "Model",
    "Metric",
    "diff_pipelines",
    "merge_pipelines",
    "pipeline_to_dict",
    "pipeline_to_json",
    "ValidationIssue",

    # Configuration
    "NdelConfig",
    "PrivacyConfig",
    "DomainConfig",
    "AbstractionLevel",
    "validate_config_against_pipeline",
]
