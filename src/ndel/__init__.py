"""NDEL - Describe data operations in human-readable form."""

__version__ = "0.2.0"

from .parser import parse_ndel, print_ast
from .ast import (
    ASTNode, Literal, Identifier, BinaryOp, UnaryOp,
    FuzzyPredicate, MemberAccess, IndexAccess, Call,
    ListLiteral, MapLiteral, DomainDeclaration, Conditional
)
from .translator import translate
from .domains import DomainConfig, apply_domain

from typing import Optional


def describe(python_code: str, domain: Optional[DomainConfig] = None) -> str:
    """
    Convert Python code to human-readable NDEL description.

    Args:
        python_code: Python code containing SQL queries or pandas operations
        domain: Optional vocabulary mappings provided by consuming application

    Returns:
        Human-readable NDEL description

    Example:
        >>> describe('pd.read_sql("SELECT name FROM users", conn)')
        'FIND name FROM users'

        >>> describe(code, domain={"tables": {"users": "customers"}})
        'FIND name FROM customers'
    """
    try:
        ndel_text = translate(python_code)
        return apply_domain(ndel_text, domain)
    except Exception:
        return "Analysis performed"


__all__ = [
    # Main API
    "describe",
    "DomainConfig",

    # Lower-level access
    "translate",
    "apply_domain",
    "parse_ndel",
    "print_ast",
]
