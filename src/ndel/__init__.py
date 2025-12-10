"""NDEL - Describe data operations in human-readable form."""

__version__ = "0.2.0"

from typing import Optional

from ndel.api import (
    describe_callable,
    describe_pipeline_diff,
    describe_python_source,
    describe_sql_source,
    describe_sql_and_python,
    validate_config,
)
from ndel.ast import (
    ASTNode,
    BinaryOp,
    Call,
    Conditional,
    DomainDeclaration,
    FuzzyPredicate,
    Identifier,
    IndexAccess,
    ListLiteral,
    Literal,
    MapLiteral,
    MemberAccess,
    UnaryOp,
)
from ndel.config import AbstractionLevel, DomainConfig, NdelConfig, PrivacyConfig
from ndel.domains import DomainConfig as LegacyDomainConfig
from ndel.domains import apply_domain
from ndel.parser import parse_ndel, print_ast
from ndel.semantic_model import Dataset, Feature, Metric, Model, Pipeline, Transformation
from ndel.validation import ValidationIssue
from ndel.translator import translate

__version__ = "0.2.0"


def describe(python_code: str, domain: Optional[LegacyDomainConfig] = None) -> str:
    """Backward-compatible describe entrypoint using the legacy translator."""

    ndel_text = translate(python_code)
    return apply_domain(ndel_text, domain)


__all__ = [
    # Primary public API
    "describe_python_source",
    "describe_callable",
    "describe_sql_and_python",
    "describe_pipeline_diff",
    "validate_config",
    "describe_sql_source",

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
    "describe_pipeline_diff",

    # Compatibility API
    "describe",
    "translate",
    "apply_domain",
    "parse_ndel",
    "print_ast",
    "ASTNode",
    "Literal",
    "Identifier",
    "BinaryOp",
    "UnaryOp",
    "FuzzyPredicate",
    "MemberAccess",
    "IndexAccess",
    "Call",
    "ListLiteral",
    "MapLiteral",
    "DomainDeclaration",
    "Conditional",
]
