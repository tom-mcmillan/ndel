"""
NDEL AST Node Classes

Defines all Abstract Syntax Tree node types produced by the NDEL parser.
"""

from dataclasses import dataclass, field
from typing import Any, List, Optional, Union, Dict


@dataclass
class ASTNode:
    """Base class for all AST nodes."""
    line: int = 0
    column: int = 0
    confidence: Optional[float] = None


@dataclass
class Literal(ASTNode):
    """Literal value: number, string, boolean, or null."""
    value: Any = None


@dataclass
class Identifier(ASTNode):
    """Variable or field reference."""
    name: str = ""


@dataclass
class BinaryOp(ASTNode):
    """Binary operation: left operator right."""
    left: ASTNode = None
    operator: str = ""
    right: ASTNode = None


@dataclass
class UnaryOp(ASTNode):
    """Unary operation: operator operand."""
    operator: str = ""
    operand: ASTNode = None


@dataclass
class Conditional(ASTNode):
    """Ternary conditional: condition ? true_branch : false_branch."""
    condition: ASTNode = None
    true_branch: ASTNode = None
    false_branch: ASTNode = None


@dataclass
class FuzzyPredicate(ASTNode):
    """
    Fuzzy predicate expression.

    Examples:
        player is "young"
        trend shows "improving"
        value approximately "high"
    """
    subject: ASTNode = None
    operator: str = ""  # 'is', 'shows', 'approximately', 'roughly', 'fuzzy'
    fuzzy_value: str = ""
    resolution: Optional[Dict] = None


@dataclass
class MemberAccess(ASTNode):
    """Member/field access: object.field."""
    object: ASTNode = None
    member: str = ""


@dataclass
class IndexAccess(ASTNode):
    """Index access: object[index]."""
    object: ASTNode = None
    index: ASTNode = None


@dataclass
class Call(ASTNode):
    """Function or method call: func(args) or obj.method(args)."""
    function: Union[str, ASTNode] = None
    arguments: List[ASTNode] = field(default_factory=list)


@dataclass
class ListLiteral(ASTNode):
    """List literal: [elem1, elem2, ...]."""
    elements: List[ASTNode] = field(default_factory=list)


@dataclass
class MapLiteral(ASTNode):
    """Map literal: {key1: value1, key2: value2, ...}."""
    entries: List[tuple] = field(default_factory=list)


@dataclass
class DomainDeclaration(ASTNode):
    """Domain declaration: @domain("name")."""
    domain: str = ""


# Type alias for any AST node
Node = Union[
    Literal,
    Identifier,
    BinaryOp,
    UnaryOp,
    Conditional,
    FuzzyPredicate,
    MemberAccess,
    IndexAccess,
    Call,
    ListLiteral,
    MapLiteral,
    DomainDeclaration,
]

__all__ = [
    "ASTNode",
    "Literal",
    "Identifier",
    "BinaryOp",
    "UnaryOp",
    "Conditional",
    "FuzzyPredicate",
    "MemberAccess",
    "IndexAccess",
    "Call",
    "ListLiteral",
    "MapLiteral",
    "DomainDeclaration",
    "Node",
]
