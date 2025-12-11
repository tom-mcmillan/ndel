#!/usr/bin/env python3
"""
NDEL Capability Test Script
Tests everything the current NDEL implementation can do.
"""

import sys
sys.path.insert(0, '/Users/thomasmcmillan/projects/ndel/reference/python')

from parser import parse_ndel, print_ast, Lexer, Parser
from parser import (
    Literal, Identifier, BinaryOp, UnaryOp, Conditional,
    FuzzyPredicate, MemberAccess, IndexAccess, Call,
    ListLiteral, MapLiteral, DomainDeclaration
)

def test_section(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def test_parse(expr, description=""):
    """Test parsing an expression and show the AST."""
    print(f"\n--- {description or expr} ---")
    print(f"Input: {expr}")
    try:
        ast = parse_ndel(expr)
        print("AST:")
        for node in ast:
            print_ast(node, indent=1)
        return ast
    except Exception as e:
        print(f"ERROR: {e}")
        return None

def ast_summary(node):
    """Return a one-line summary of an AST node."""
    if isinstance(node, Literal):
        return f"Literal({repr(node.value)})"
    elif isinstance(node, Identifier):
        return f"Identifier({node.name})"
    elif isinstance(node, BinaryOp):
        return f"BinaryOp({node.operator})"
    elif isinstance(node, FuzzyPredicate):
        return f"FuzzyPredicate({node.operator}, '{node.fuzzy_value}')"
    elif isinstance(node, Call):
        func = node.function if isinstance(node.function, str) else ast_summary(node.function)
        return f"Call({func}, {len(node.arguments)} args)"
    elif isinstance(node, MemberAccess):
        return f"MemberAccess(.{node.field})"
    elif isinstance(node, ListLiteral):
        return f"List[{len(node.elements)} items]"
    elif isinstance(node, MapLiteral):
        return f"Map{{{len(node.entries)} entries}}"
    elif isinstance(node, DomainDeclaration):
        return f"@domain('{node.domain}')"
    else:
        return type(node).__name__

# ============================================================================
# PART 1: Lexer Tests
# ============================================================================
test_section("LEXER TESTS")

def test_tokenize(source):
    print(f"\nTokenizing: {source}")
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    for tok in tokens:
        print(f"  {tok.type.name:15} = {repr(tok.value)}")
    return tokens

test_tokenize("42")
test_tokenize('"hello world"')
test_tokenize("age < 25")
test_tokenize('player is "young"')
test_tokenize("@domain")

# ============================================================================
# PART 2: Basic Literal Parsing
# ============================================================================
test_section("LITERAL PARSING")

test_parse("42", "Integer literal")
test_parse("3.14", "Float literal")
test_parse('"hello"', "String literal")
test_parse("true", "Boolean true")
test_parse("false", "Boolean false")
test_parse("null", "Null literal")

# ============================================================================
# PART 3: Identifier and Member Access
# ============================================================================
test_section("IDENTIFIERS & MEMBER ACCESS")

test_parse("age", "Simple identifier")
test_parse("player_name", "Identifier with underscore")
test_parse("player.age", "Member access")
test_parse("team.players.count", "Chained member access")

# ============================================================================
# PART 4: Arithmetic Expressions
# ============================================================================
test_section("ARITHMETIC EXPRESSIONS")

test_parse("1 + 2", "Addition")
test_parse("5 - 3", "Subtraction")
test_parse("4 * 5", "Multiplication")
test_parse("10 / 2", "Division")
test_parse("7 % 3", "Modulo")
test_parse("1 + 2 * 3", "Precedence test")
test_parse("(1 + 2) * 3", "Parentheses")
test_parse("-5", "Unary minus")

# ============================================================================
# PART 5: Comparison Expressions
# ============================================================================
test_section("COMPARISON EXPRESSIONS")

test_parse("age < 25", "Less than")
test_parse("age <= 25", "Less than or equal")
test_parse("age > 25", "Greater than")
test_parse("age >= 25", "Greater than or equal")
test_parse("age == 25", "Equal")
test_parse("age != 25", "Not equal")

# ============================================================================
# PART 6: Logical Expressions
# ============================================================================
test_section("LOGICAL EXPRESSIONS")

test_parse("true && false", "Logical AND")
test_parse("true || false", "Logical OR")
test_parse("!true", "Logical NOT")
test_parse("age > 20 && age < 30", "Compound condition")
test_parse("a && b || c", "AND/OR precedence")

# ============================================================================
# PART 7: Fuzzy Expressions (THE KEY NDEL FEATURE)
# ============================================================================
test_section("FUZZY EXPRESSIONS")

test_parse('player is "young"', "Fuzzy IS operator")
test_parse('trend shows "improving"', "Fuzzy SHOWS operator")
test_parse('value approximately "high"', "Fuzzy APPROXIMATELY operator")
test_parse('age < "young" && potential is "high"', "Mixed fuzzy/concrete")
test_parse('player is "promising young striker"', "Multi-word fuzzy value")

# ============================================================================
# PART 8: Ternary Conditional
# ============================================================================
test_section("CONDITIONAL EXPRESSIONS")

test_parse('age > 30 ? "old" : "young"', "Ternary conditional")
test_parse('a ? b ? c : d : e', "Nested ternary")

# ============================================================================
# PART 9: Function Calls
# ============================================================================
test_section("FUNCTION CALLS")

test_parse("count()", "No-arg function")
test_parse("sum(1, 2, 3)", "Multi-arg function")
test_parse("player.goals(season)", "Method call")
test_parse("confidence()", "Confidence function")
test_parse("with_confidence(0.9, result)", "With confidence function")
test_parse("filter(players, p, p.age < 25)", "Higher-order function")

# ============================================================================
# PART 10: Index Access
# ============================================================================
test_section("INDEX ACCESS")

test_parse("list[0]", "Array index")
test_parse('map["key"]', "Map key access")
test_parse("players[0].name", "Index then member")

# ============================================================================
# PART 11: Collection Literals
# ============================================================================
test_section("COLLECTION LITERALS")

test_parse("[1, 2, 3]", "List literal")
test_parse("[]", "Empty list")
test_parse('{"name": "John", "age": 25}', "Map literal")
test_parse("{}", "Empty map")
test_parse('[1, "two", true]', "Mixed type list")

# ============================================================================
# PART 12: Domain Declarations
# ============================================================================
test_section("DOMAIN DECLARATIONS")

test_parse('@domain("soccer")', "Domain declaration")
test_parse('@domain("finance")', "Different domain")

# ============================================================================
# PART 13: Complex Real-World Expressions
# ============================================================================
test_section("COMPLEX REAL-WORLD EXPRESSIONS")

test_parse(
    'player.age < 25 && player.goals > 10 && player.potential is "high"',
    "Soccer player filter"
)

test_parse(
    'salary < 100000 && (experience > 5 || education is "advanced")',
    "HR filter with OR"
)

test_parse(
    'performance is "excellent" ? bonus * 1.5 : bonus',
    "Conditional bonus calculation"
)

test_parse(
    'players.filter(p, p.age < "young" && p.position is "forward")',
    "List filter with fuzzy"
)

# ============================================================================
# PART 14: Error Handling Tests
# ============================================================================
test_section("ERROR HANDLING")

def test_parse_error(expr, description):
    print(f"\n--- {description} ---")
    print(f"Input: {expr}")
    try:
        ast = parse_ndel(expr)
        print("Unexpectedly succeeded!")
    except SyntaxError as e:
        print(f"SyntaxError (expected): {e}")
    except Exception as e:
        print(f"Other error: {type(e).__name__}: {e}")

test_parse_error("1 +", "Incomplete expression")
test_parse_error("((1 + 2)", "Unbalanced parens")
test_parse_error('"unclosed string', "Unclosed string")

# ============================================================================
# PART 15: What's NOT Implemented
# ============================================================================
test_section("NOT IMPLEMENTED (Expected Failures)")

print("""
These features appear in docs/examples but aren't in the parser:

1. query/where/order by/limit syntax (player_search.ndel uses these)
   Current: Only expression parsing, no query DSL

2. let bindings (let x = expr)
   Current: Not supported

3. for/if statements (for each player in squad:)
   Current: Not supported, expression-only

4. compute keyword (compute value_ratio = ...)
   Current: Not supported

5. Interpreter/Evaluator
   Current: interpreter.py and fuzzy_resolver.py are EMPTY files

6. Fuzzy resolution
   Current: Parser identifies fuzzy predicates but cannot resolve them

7. Confidence propagation
   Current: AST nodes have confidence field but no logic
""")

# ============================================================================
# SUMMARY
# ============================================================================
test_section("CAPABILITY SUMMARY")

print("""
WHAT WORKS:
-----------
✓ Lexer: Full tokenization of NDEL syntax
✓ Parser: Recursive descent parser producing clean AST
✓ Literals: Numbers, strings, booleans, null
✓ Identifiers: Simple and dotted (member access)
✓ Operators: All arithmetic, comparison, logical
✓ Fuzzy operators: is, shows, approximately (parsed, not resolved)
✓ Conditionals: Ternary expressions
✓ Functions: Call syntax, argument lists
✓ Collections: List and map literals
✓ Indexing: Array and map access
✓ Domain declarations: @domain("name")

WHAT'S MISSING:
---------------
✗ Interpreter: Empty file - no evaluation
✗ Fuzzy Resolver: Empty file - no resolution
✗ Query DSL: where/order by/limit not implemented
✗ Statements: Only expressions, no let/for/if statements
✗ Confidence tracking: Field exists but unused
✗ LLM integration: Runtime exists but not connected

BOTTOM LINE:
------------
NDEL currently has a COMPLETE PARSER that produces a clean AST.
It can parse any expression in the grammar.
But it CANNOT EVALUATE anything - the interpreter is empty.
""")
