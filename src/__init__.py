"""NDEL - Describe data operations in human-readable form."""

__version__ = "0.2.0"

from src.describe import (
    describe_python_source,
    describe_callable,
    describe_callable_with_llm,
    describe_sql_and_python,
    describe_sql_and_python_with_llm,
    describe_sql_source,
    describe_sql_source_with_llm,
    describe_pipeline_diff,
    pipeline_to_dict,
    pipeline_to_json,
    validate_config,
)
from src.config import AbstractionLevel, DomainConfig, NdelConfig, PrivacyConfig
from src.types import (
    Dataset,
    Feature,
    Metric,
    Model,
    Pipeline,
    Transformation,
    ValidationIssue,
)
from src.llmrenderer import LLMGenerate, render_pipeline_with_llm

__all__ = [
    # Primary public API
    "describe_python_source",
    "describe_callable",
    "describe_callable_with_llm",
    "describe_sql_and_python",
    "describe_sql_and_python_with_llm",
    "describe_pipeline_diff",
    "pipeline_to_dict",
    "pipeline_to_json",
    "validate_config",
    "describe_sql_source",
    "describe_sql_source_with_llm",
    "describe_python_source_with_llm",
    "render_pipeline_with_llm",
    "LLMGenerate",

    # Semantic model
    "Pipeline",
    "Dataset",
    "Transformation",
    "Feature",
    "Model",
    "Metric",

    # Configuration
    "NdelConfig",
    "PrivacyConfig",
    "DomainConfig",
    "AbstractionLevel",
    "ValidationIssue",
]
