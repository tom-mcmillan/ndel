"""
Tests for the NDEL parser.
"""

import pytest
from ndel import (
    parse_ndel,
    Literal,
    Identifier,
    BinaryOp,
    FuzzyPredicate,
    MemberAccess,
    ListLiteral,
    DomainDeclaration,
)


class TestLiterals:
    """Test parsing of literal values."""

    def test_integer(self):
        ast = parse_ndel("42")
        assert len(ast) == 1
        assert isinstance(ast[0], Literal)
        assert ast[0].value == 42

    def test_float(self):
        ast = parse_ndel("3.14")
        assert len(ast) == 1
        assert isinstance(ast[0], Literal)
        assert ast[0].value == 3.14

    def test_string(self):
        ast = parse_ndel('"hello"')
        assert len(ast) == 1
        assert isinstance(ast[0], Literal)
        assert ast[0].value == "hello"

    def test_boolean_true(self):
        ast = parse_ndel("true")
        assert len(ast) == 1
        assert isinstance(ast[0], Literal)
        assert ast[0].value is True

    def test_boolean_false(self):
        ast = parse_ndel("false")
        assert len(ast) == 1
        assert isinstance(ast[0], Literal)
        assert ast[0].value is False

    def test_null(self):
        ast = parse_ndel("null")
        assert len(ast) == 1
        assert isinstance(ast[0], Literal)
        assert ast[0].value is None


class TestIdentifiers:
    """Test parsing of identifiers and member access."""

    def test_simple_identifier(self):
        ast = parse_ndel("age")
        assert len(ast) == 1
        assert isinstance(ast[0], Identifier)
        assert ast[0].name == "age"

    def test_member_access(self):
        ast = parse_ndel("player.age")
        assert len(ast) == 1
        assert isinstance(ast[0], MemberAccess)
        assert ast[0].member == "age"
        assert isinstance(ast[0].object, Identifier)
        assert ast[0].object.name == "player"


class TestBinaryOperations:
    """Test parsing of binary operations."""

    def test_comparison(self):
        ast = parse_ndel("age < 25")
        assert len(ast) == 1
        assert isinstance(ast[0], BinaryOp)
        assert ast[0].operator == "<"
        assert isinstance(ast[0].left, Identifier)
        assert isinstance(ast[0].right, Literal)

    def test_logical_and(self):
        ast = parse_ndel("a && b")
        assert len(ast) == 1
        assert isinstance(ast[0], BinaryOp)
        assert ast[0].operator == "&&"

    def test_precedence(self):
        # Multiplication should bind tighter than addition
        ast = parse_ndel("1 + 2 * 3")
        assert isinstance(ast[0], BinaryOp)
        assert ast[0].operator == "+"
        assert isinstance(ast[0].right, BinaryOp)
        assert ast[0].right.operator == "*"


class TestFuzzyPredicates:
    """Test parsing of fuzzy predicates."""

    def test_is_operator(self):
        ast = parse_ndel('player is "promising"')
        assert len(ast) == 1
        assert isinstance(ast[0], FuzzyPredicate)
        assert ast[0].operator == "is"
        assert ast[0].fuzzy_value == "promising"
        assert isinstance(ast[0].subject, Identifier)

    def test_shows_operator(self):
        ast = parse_ndel('trend shows "improving"')
        assert len(ast) == 1
        assert isinstance(ast[0], FuzzyPredicate)
        assert ast[0].operator == "shows"
        assert ast[0].fuzzy_value == "improving"

    def test_mixed_fuzzy_and_concrete(self):
        ast = parse_ndel('age < 25 && potential is "high"')
        assert len(ast) == 1
        assert isinstance(ast[0], BinaryOp)
        assert ast[0].operator == "&&"
        # Left side is concrete comparison
        assert isinstance(ast[0].left, BinaryOp)
        assert ast[0].left.operator == "<"
        # Right side is fuzzy predicate
        assert isinstance(ast[0].right, FuzzyPredicate)
        assert ast[0].right.fuzzy_value == "high"


class TestCollections:
    """Test parsing of collection literals."""

    def test_list_literal(self):
        ast = parse_ndel("[1, 2, 3]")
        assert len(ast) == 1
        assert isinstance(ast[0], ListLiteral)
        assert len(ast[0].elements) == 3

    def test_empty_list(self):
        ast = parse_ndel("[]")
        assert len(ast) == 1
        assert isinstance(ast[0], ListLiteral)
        assert len(ast[0].elements) == 0


class TestDomainDeclaration:
    """Test parsing of domain declarations."""

    def test_domain_declaration(self):
        ast = parse_ndel('@domain("soccer")')
        assert len(ast) == 1
        assert isinstance(ast[0], DomainDeclaration)
        assert ast[0].domain == "soccer"


class TestErrorHandling:
    """Test error handling in parser."""

    def test_unterminated_string(self):
        with pytest.raises(SyntaxError) as exc_info:
            parse_ndel('"unterminated')
        assert "Unterminated string" in str(exc_info.value)

    def test_unbalanced_parens(self):
        with pytest.raises(SyntaxError) as exc_info:
            parse_ndel("((1 + 2)")
        assert "Expected" in str(exc_info.value)

    def test_incomplete_expression(self):
        with pytest.raises(SyntaxError) as exc_info:
            parse_ndel("1 +")
        assert "Unexpected token" in str(exc_info.value)
