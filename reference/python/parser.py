"""
NDEL Parser - Reference Implementation
Version: 0.1.0

A basic recursive descent parser for NDEL expressions.
"""

import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union, Dict
from enum import Enum, auto


# ============================================================================
# Token Types
# ============================================================================

class TokenType(Enum):
    # Literals
    NUMBER = auto()
    STRING = auto()
    BOOLEAN = auto()
    NULL = auto()
    
    # Identifiers and Keywords
    IDENTIFIER = auto()
    IS = auto()
    SHOWS = auto()
    APPROXIMATELY = auto()
    HAS = auto()
    IN = auto()
    AND = auto()
    OR = auto()
    NOT = auto()
    IF = auto()
    THEN = auto()
    ELSE = auto()
    CONFIDENCE = auto()
    WITH_CONFIDENCE = auto()
    DOMAIN = auto()
    
    # Operators
    PLUS = auto()
    MINUS = auto()
    MULTIPLY = auto()
    DIVIDE = auto()
    MODULO = auto()
    EQUAL = auto()
    NOT_EQUAL = auto()
    LESS = auto()
    LESS_EQUAL = auto()
    GREATER = auto()
    GREATER_EQUAL = auto()
    LOGICAL_AND = auto()
    LOGICAL_OR = auto()
    LOGICAL_NOT = auto()
    
    # Delimiters
    LPAREN = auto()
    RPAREN = auto()
    LBRACKET = auto()
    RBRACKET = auto()
    LBRACE = auto()
    RBRACE = auto()
    DOT = auto()
    COMMA = auto()
    COLON = auto()
    SEMICOLON = auto()
    QUESTION = auto()
    
    # Special
    EOF = auto()
    FUZZY_STRING = auto()


@dataclass
class Token:
    type: TokenType
    value: Any
    line: int
    column: int


# ============================================================================
# Lexer
# ============================================================================

class Lexer:
    """Tokenizes NDEL source code."""
    
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens = []
        
        # Keywords mapping
        self.keywords = {
            'true': TokenType.BOOLEAN,
            'false': TokenType.BOOLEAN,
            'null': TokenType.NULL,
            'is': TokenType.IS,
            'shows': TokenType.SHOWS,
            'approximately': TokenType.APPROXIMATELY,
            'has': TokenType.HAS,
            'in': TokenType.IN,
            'and': TokenType.AND,
            'or': TokenType.OR,
            'not': TokenType.NOT,
            'if': TokenType.IF,
            'then': TokenType.THEN,
            'else': TokenType.ELSE,
            'confidence': TokenType.CONFIDENCE,
            'with_confidence': TokenType.WITH_CONFIDENCE,
        }
        
    def error(self, msg: str):
        raise SyntaxError(f"Line {self.line}, Column {self.column}: {msg}")
    
    def peek(self, offset: int = 0) -> Optional[str]:
        pos = self.pos + offset
        if pos < len(self.source):
            return self.source[pos]
        return None
    
    def advance(self) -> Optional[str]:
        if self.pos < len(self.source):
            char = self.source[self.pos]
            self.pos += 1
            if char == '\n':
                self.line += 1
                self.column = 1
            else:
                self.column += 1
            return char
        return None
    
    def skip_whitespace(self):
        while self.peek() and self.peek() in ' \t\n\r':
            self.advance()
    
    def skip_comment(self):
        if self.peek() == '/' and self.peek(1) == '/':
            # Single-line comment
            while self.peek() and self.peek() != '\n':
                self.advance()
            self.advance()  # Skip newline
            return True
        elif self.peek() == '/' and self.peek(1) == '*':
            # Multi-line comment
            self.advance()  # Skip /
            self.advance()  # Skip *
            while self.peek() and not (self.peek() == '*' and self.peek(1) == '/'):
                self.advance()
            self.advance()  # Skip *
            self.advance()  # Skip /
            return True
        return False
    
    def read_string(self, quote: str) -> str:
        value = ""
        self.advance()  # Skip opening quote
        
        while self.peek() and self.peek() != quote:
            if self.peek() == '\\':
                self.advance()
                next_char = self.advance()
                if next_char == 'n':
                    value += '\n'
                elif next_char == 't':
                    value += '\t'
                elif next_char == '\\':
                    value += '\\'
                elif next_char == quote:
                    value += quote
                else:
                    value += next_char
            else:
                value += self.advance()
        
        if not self.peek():
            self.error(f"Unterminated string")
        
        self.advance()  # Skip closing quote
        return value
    
    def read_number(self) -> Union[int, float]:
        value = ""
        has_dot = False
        
        while self.peek() and (self.peek().isdigit() or self.peek() == '.'):
            if self.peek() == '.':
                if has_dot:
                    break
                has_dot = True
            value += self.advance()
        
        if self.peek() and self.peek() in 'eE':
            value += self.advance()
            if self.peek() and self.peek() in '+-':
                value += self.advance()
            while self.peek() and self.peek().isdigit():
                value += self.advance()
            has_dot = True
        
        return float(value) if has_dot else int(value)
    
    def read_identifier(self) -> str:
        value = ""
        while self.peek() and (self.peek().isalnum() or self.peek() == '_'):
            value += self.advance()
        return value
    
    def next_token(self) -> Token:
        # Skip whitespace and comments
        while True:
            self.skip_whitespace()
            if not self.skip_comment():
                break
        
        line = self.line
        column = self.column
        
        # Check EOF
        if not self.peek():
            return Token(TokenType.EOF, None, line, column)
        
        char = self.peek()
        
        # String literals
        if char in '"\'':
            value = self.read_string(char)
            # Check if it's a fuzzy string (heuristic: contains natural language)
            is_fuzzy = any(word in value.lower() for word in 
                          ['young', 'high', 'low', 'good', 'poor', 'excellent',
                           'promising', 'veteran', 'clinical', 'reasonable'])
            token_type = TokenType.FUZZY_STRING if is_fuzzy else TokenType.STRING
            return Token(token_type, value, line, column)
        
        # Numbers
        if char.isdigit():
            value = self.read_number()
            return Token(TokenType.NUMBER, value, line, column)
        
        # Identifiers and keywords
        if char.isalpha() or char == '_':
            value = self.read_identifier()
            if value in self.keywords:
                token_type = self.keywords[value]
                if token_type == TokenType.BOOLEAN:
                    value = value == 'true'
                elif token_type == TokenType.NULL:
                    value = None
            else:
                token_type = TokenType.IDENTIFIER
            return Token(token_type, value, line, column)
        
        # Two-character operators
        two_char = self.source[self.pos:self.pos+2] if self.pos + 1 < len(self.source) else ""
        
        two_char_tokens = {
            '==': TokenType.EQUAL,
            '!=': TokenType.NOT_EQUAL,
            '<=': TokenType.LESS_EQUAL,
            '>=': TokenType.GREATER_EQUAL,
            '&&': TokenType.LOGICAL_AND,
            '||': TokenType.LOGICAL_OR,
        }
        
        if two_char in two_char_tokens:
            self.advance()
            self.advance()
            return Token(two_char_tokens[two_char], two_char, line, column)
        
        # Single-character tokens
        single_char_tokens = {
            '+': TokenType.PLUS,
            '-': TokenType.MINUS,
            '*': TokenType.MULTIPLY,
            '/': TokenType.DIVIDE,
            '%': TokenType.MODULO,
            '<': TokenType.LESS,
            '>': TokenType.GREATER,
            '!': TokenType.LOGICAL_NOT,
            '(': TokenType.LPAREN,
            ')': TokenType.RPAREN,
            '[': TokenType.LBRACKET,
            ']': TokenType.RBRACKET,
            '{': TokenType.LBRACE,
            '}': TokenType.RBRACE,
            '.': TokenType.DOT,
            ',': TokenType.COMMA,
            ':': TokenType.COLON,
            ';': TokenType.SEMICOLON,
            '?': TokenType.QUESTION,
        }
        
        if char in single_char_tokens:
            self.advance()
            return Token(single_char_tokens[char], char, line, column)
        
        self.error(f"Unexpected character: {char}")
    
    def tokenize(self) -> List[Token]:
        """Tokenize the entire source."""
        tokens = []
        while True:
            token = self.next_token()
            tokens.append(token)
            if token.type == TokenType.EOF:
                break
        return tokens


# ============================================================================
# AST Nodes
# ============================================================================

@dataclass
class ASTNode:
    """Base class for AST nodes."""
    line: int = 0
    column: int = 0
    confidence: Optional[float] = None


@dataclass
class Literal(ASTNode):
    value: Any
    
    
@dataclass
class Identifier(ASTNode):
    name: str


@dataclass
class BinaryOp(ASTNode):
    left: ASTNode
    operator: str
    right: ASTNode


@dataclass
class UnaryOp(ASTNode):
    operator: str
    operand: ASTNode


@dataclass
class Conditional(ASTNode):
    condition: ASTNode
    true_branch: ASTNode
    false_branch: ASTNode


@dataclass
class FuzzyPredicate(ASTNode):
    subject: ASTNode
    operator: str  # 'is', 'shows', 'approximately'
    fuzzy_value: str
    resolution: Optional[Dict] = None


@dataclass
class MemberAccess(ASTNode):
    object: ASTNode
    field: str


@dataclass
class IndexAccess(ASTNode):
    object: ASTNode
    index: ASTNode


@dataclass
class Call(ASTNode):
    function: Union[str, ASTNode]  # Function name or object for method call
    arguments: List[ASTNode]


@dataclass
class ListLiteral(ASTNode):
    elements: List[ASTNode]


@dataclass
class MapLiteral(ASTNode):
    entries: List[tuple[ASTNode, ASTNode]]


@dataclass
class DomainDeclaration(ASTNode):
    domain: str


# ============================================================================
# Parser
# ============================================================================

class Parser:
    """Recursive descent parser for NDEL."""
    
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.pos = 0
        
    def error(self, msg: str):
        token = self.current_token()
        raise SyntaxError(f"Line {token.line}, Column {token.column}: {msg}")
    
    def current_token(self) -> Token:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return self.tokens[-1]  # Return EOF token
    
    def peek(self, offset: int = 0) -> TokenType:
        pos = self.pos + offset
        if pos < len(self.tokens):
            return self.tokens[pos].type
        return TokenType.EOF
    
    def consume(self, expected: TokenType) -> Token:
        token = self.current_token()
        if token.type != expected:
            self.error(f"Expected {expected}, got {token.type}")
        self.pos += 1
        return token
    
    def match(self, *types: TokenType) -> bool:
        return self.peek() in types
    
    def parse(self) -> List[ASTNode]:
        """Parse the token stream into an AST."""
        statements = []
        
        while not self.match(TokenType.EOF):
            if self.peek() == TokenType.DOMAIN:
                statements.append(self.parse_domain_declaration())
            else:
                statements.append(self.parse_expression())
                # Optional semicolon
                if self.match(TokenType.SEMICOLON):
                    self.pos += 1
        
        return statements
    
    def parse_domain_declaration(self) -> DomainDeclaration:
        """Parse @domain("domain_name")"""
        token = self.consume(TokenType.DOMAIN)
        self.consume(TokenType.LPAREN)
        domain_token = self.consume(TokenType.STRING)
        self.consume(TokenType.RPAREN)
        
        return DomainDeclaration(
            domain=domain_token.value,
            line=token.line,
            column=token.column
        )
    
    def parse_expression(self) -> ASTNode:
        """Parse an expression."""
        return self.parse_conditional()
    
    def parse_conditional(self) -> ASTNode:
        """Parse ternary conditional: expr ? true_branch : false_branch"""
        expr = self.parse_logical_or()
        
        if self.match(TokenType.QUESTION):
            self.pos += 1
            true_branch = self.parse_expression()
            self.consume(TokenType.COLON)
            false_branch = self.parse_expression()
            
            return Conditional(
                condition=expr,
                true_branch=true_branch,
                false_branch=false_branch,
                line=expr.line,
                column=expr.column
            )
        
        return expr
    
    def parse_logical_or(self) -> ASTNode:
        """Parse logical OR expression."""
        left = self.parse_logical_and()
        
        while self.match(TokenType.LOGICAL_OR, TokenType.OR):
            op_token = self.current_token()
            self.pos += 1
            right = self.parse_logical_and()
            left = BinaryOp(
                left=left,
                operator='||',
                right=right,
                line=op_token.line,
                column=op_token.column
            )
        
        return left
    
    def parse_logical_and(self) -> ASTNode:
        """Parse logical AND expression."""
        left = self.parse_equality()
        
        while self.match(TokenType.LOGICAL_AND, TokenType.AND):
            op_token = self.current_token()
            self.pos += 1
            right = self.parse_equality()
            left = BinaryOp(
                left=left,
                operator='&&',
                right=right,
                line=op_token.line,
                column=op_token.column
            )
        
        return left
    
    def parse_equality(self) -> ASTNode:
        """Parse equality operators."""
        left = self.parse_relational()
        
        while self.match(TokenType.EQUAL, TokenType.NOT_EQUAL):
            op_token = self.current_token()
            self.pos += 1
            right = self.parse_relational()
            left = BinaryOp(
                left=left,
                operator=op_token.value,
                right=right,
                line=op_token.line,
                column=op_token.column
            )
        
        return left
    
    def parse_relational(self) -> ASTNode:
        """Parse relational operators."""
        left = self.parse_additive()
        
        while self.match(TokenType.LESS, TokenType.LESS_EQUAL, 
                         TokenType.GREATER, TokenType.GREATER_EQUAL):
            op_token = self.current_token()
            self.pos += 1
            right = self.parse_additive()
            left = BinaryOp(
                left=left,
                operator=op_token.value,
                right=right,
                line=op_token.line,
                column=op_token.column
            )
        
        return left
    
    def parse_additive(self) -> ASTNode:
        """Parse addition and subtraction."""
        left = self.parse_multiplicative()
        
        while self.match(TokenType.PLUS, TokenType.MINUS):
            op_token = self.current_token()
            self.pos += 1
            right = self.parse_multiplicative()
            left = BinaryOp(
                left=left,
                operator=op_token.value,
                right=right,
                line=op_token.line,
                column=op_token.column
            )
        
        return left
    
    def parse_multiplicative(self) -> ASTNode:
        """Parse multiplication, division, and modulo."""
        left = self.parse_unary()
        
        while self.match(TokenType.MULTIPLY, TokenType.DIVIDE, TokenType.MODULO):
            op_token = self.current_token()
            self.pos += 1
            right = self.parse_unary()
            left = BinaryOp(
                left=left,
                operator=op_token.value,
                right=right,
                line=op_token.line,
                column=op_token.column
            )
        
        return left
    
    def parse_unary(self) -> ASTNode:
        """Parse unary operators."""
        if self.match(TokenType.LOGICAL_NOT, TokenType.NOT, 
                      TokenType.MINUS, TokenType.PLUS):
            op_token = self.current_token()
            self.pos += 1
            operand = self.parse_unary()
            return UnaryOp(
                operator=op_token.value if op_token.value else '!',
                operand=operand,
                line=op_token.line,
                column=op_token.column
            )
        
        return self.parse_fuzzy()
    
    def parse_fuzzy(self) -> ASTNode:
        """Parse fuzzy predicates (is, shows, approximately)."""
        left = self.parse_postfix()
        
        if self.match(TokenType.IS, TokenType.SHOWS, TokenType.APPROXIMATELY):
            op_token = self.current_token()
            self.pos += 1
            
            # Expect a fuzzy value (string)
            if not self.match(TokenType.STRING, TokenType.FUZZY_STRING):
                self.error("Expected string value after fuzzy operator")
            
            value_token = self.current_token()
            self.pos += 1
            
            return FuzzyPredicate(
                subject=left,
                operator=op_token.value,
                fuzzy_value=value_token.value,
                line=op_token.line,
                column=op_token.column
            )
        
        return left
    
    def parse_postfix(self) -> ASTNode:
        """Parse postfix operations (member access, indexing, function calls)."""
        expr = self.parse_primary()
        
        while True:
            if self.match(TokenType.DOT):
                self.pos += 1
                field_token = self.consume(TokenType.IDENTIFIER)
                expr = MemberAccess(
                    object=expr,
                    field=field_token.value,
                    line=field_token.line,
                    column=field_token.column
                )
            elif self.match(TokenType.LBRACKET):
                self.pos += 1
                index = self.parse_expression()
                self.consume(TokenType.RBRACKET)
                expr = IndexAccess(
                    object=expr,
                    index=index,
                    line=expr.line,
                    column=expr.column
                )
            elif self.match(TokenType.LPAREN):
                self.pos += 1
                arguments = []
                
                if not self.match(TokenType.RPAREN):
                    arguments.append(self.parse_expression())
                    while self.match(TokenType.COMMA):
                        self.pos += 1
                        arguments.append(self.parse_expression())
                
                self.consume(TokenType.RPAREN)
                expr = Call(
                    function=expr,
                    arguments=arguments,
                    line=expr.line,
                    column=expr.column
                )
            else:
                break
        
        return expr
    
    def parse_primary(self) -> ASTNode:
        """Parse primary expressions."""
        token = self.current_token()
        
        # Literals
        if self.match(TokenType.NUMBER):
            self.pos += 1
            return Literal(value=token.value, line=token.line, column=token.column)
        
        if self.match(TokenType.STRING):
            self.pos += 1
            return Literal(value=token.value, line=token.line, column=token.column)
        
        if self.match(TokenType.FUZZY_STRING):
            self.pos += 1
            # Fuzzy strings as primary expressions get special treatment
            return FuzzyPredicate(
                subject=None,
                operator='fuzzy',
                fuzzy_value=token.value,
                line=token.line,
                column=token.column
            )
        
        if self.match(TokenType.BOOLEAN):
            self.pos += 1
            return Literal(value=token.value, line=token.line, column=token.column)
        
        if self.match(TokenType.NULL):
            self.pos += 1
            return Literal(value=None, line=token.line, column=token.column)
        
        # Identifiers and special functions
        if self.match(TokenType.IDENTIFIER):
            self.pos += 1
            return Identifier(name=token.value, line=token.line, column=token.column)
        
        if self.match(TokenType.CONFIDENCE):
            return self.parse_confidence_function()
        
        if self.match(TokenType.WITH_CONFIDENCE):
            return self.parse_with_confidence_function()
        
        # Parenthesized expression
        if self.match(TokenType.LPAREN):
            self.pos += 1
            expr = self.parse_expression()
            self.consume(TokenType.RPAREN)
            return expr
        
        # List literal
        if self.match(TokenType.LBRACKET):
            return self.parse_list_literal()
        
        # Map literal
        if self.match(TokenType.LBRACE):
            return self.parse_map_literal()
        
        self.error(f"Unexpected token: {token.type}")
    
    def parse_list_literal(self) -> ListLiteral:
        """Parse list literal: [expr, expr, ...]"""
        token = self.consume(TokenType.LBRACKET)
        elements = []
        
        if not self.match(TokenType.RBRACKET):
            elements.append(self.parse_expression())
            while self.match(TokenType.COMMA):
                self.pos += 1
                if self.match(TokenType.RBRACKET):  # Trailing comma
                    break
                elements.append(self.parse_expression())
        
        self.consume(TokenType.RBRACKET)
        return ListLiteral(elements=elements, line=token.line, column=token.column)
    
    def parse_map_literal(self) -> MapLiteral:
        """Parse map literal: {key: value, ...}"""
        token = self.consume(TokenType.LBRACE)
        entries = []
        
        if not self.match(TokenType.RBRACE):
            key = self.parse_expression()
            self.consume(TokenType.COLON)
            value = self.parse_expression()
            entries.append((key, value))
            
            while self.match(TokenType.COMMA):
                self.pos += 1
                if self.match(TokenType.RBRACE):  # Trailing comma
                    break
                key = self.parse_expression()
                self.consume(TokenType.COLON)
                value = self.parse_expression()
                entries.append((key, value))
        
        self.consume(TokenType.RBRACE)
        return MapLiteral(entries=entries, line=token.line, column=token.column)
    
    def parse_confidence_function(self) -> Call:
        """Parse confidence() function."""
        token = self.consume(TokenType.CONFIDENCE)
        self.consume(TokenType.LPAREN)
        self.consume(TokenType.RPAREN)
        
        return Call(
            function='confidence',
            arguments=[],
            line=token.line,
            column=token.column
        )
    
    def parse_with_confidence_function(self) -> Call:
        """Parse with_confidence(score, expr) function."""
        token = self.consume(TokenType.WITH_CONFIDENCE)
        self.consume(TokenType.LPAREN)
        
        score = self.parse_expression()
        self.consume(TokenType.COMMA)
        expr = self.parse_expression()
        
        self.consume(TokenType.RPAREN)
        
        return Call(
            function='with_confidence',
            arguments=[score, expr],
            line=token.line,
            column=token.column
        )


# ============================================================================
# Main Entry Point
# ============================================================================

def parse_ndel(source: str) -> List[ASTNode]:
    """
    Parse NDEL source code into an AST.
    
    Args:
        source: NDEL source code string
        
    Returns:
        List of AST nodes (statements)
    """
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    
    parser = Parser(tokens)
    ast = parser.parse()
    
    return ast


def print_ast(node: ASTNode, indent: int = 0):
    """Pretty print an AST node."""
    prefix = "  " * indent
    
    if isinstance(node, Literal):
        print(f"{prefix}Literal({repr(node.value)})")
    elif isinstance(node, Identifier):
        print(f"{prefix}Identifier({node.name})")
    elif isinstance(node, BinaryOp):
        print(f"{prefix}BinaryOp({node.operator})")
        print_ast(node.left, indent + 1)
        print_ast(node.right, indent + 1)
    elif isinstance(node, UnaryOp):
        print(f"{prefix}UnaryOp({node.operator})")
        print_ast(node.operand, indent + 1)
    elif isinstance(node, FuzzyPredicate):
        print(f"{prefix}FuzzyPredicate({node.operator}, '{node.fuzzy_value}')")
        if node.subject:
            print_ast(node.subject, indent + 1)
    elif isinstance(node, MemberAccess):
        print(f"{prefix}MemberAccess(.{node.field})")
        print_ast(node.object, indent + 1)
    elif isinstance(node, Call):
        print(f"{prefix}Call({node.function})")
        for arg in node.arguments:
            print_ast(arg, indent + 1)
    elif isinstance(node, ListLiteral):
        print(f"{prefix}List[")
        for elem in node.elements:
            print_ast(elem, indent + 1)
        print(f"{prefix}]")
    elif isinstance(node, DomainDeclaration):
        print(f"{prefix}@domain('{node.domain}')")
    else:
        print(f"{prefix}{type(node).__name__}")


if __name__ == "__main__":
    # Test the parser with some examples
    examples = [
        'age < "young"',
        'player is "promising"',
        'salary < 100000 && potential is "high"',
        'confidence() > 0.8 ? "high" : "low"',
        'players.filter(p, p.age < 25 && p.goals > 10)',
        '@domain("soccer")',
        '[1, 2, 3]',
        '{"name": "John", "age": 25}',
    ]
    
    for example in examples:
        print(f"\nParsing: {example}")
        print("-" * 40)
        try:
            ast = parse_ndel(example)
            for node in ast:
                print_ast(node)
        except Exception as e:
            print(f"Error: {e}")
