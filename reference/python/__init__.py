"""
NDEL - Non-Deterministic Expression Language
Version: 0.1.0

A revolutionary expression language that combines deterministic structure 
with non-deterministic value resolution.
"""

from .parser import parse_ndel, Parser, Lexer, ASTNode
from .interpreter import NDELInterpreter, NDELResult
from .fuzzy_resolver import FuzzyResolver, Resolution

__version__ = "0.1.0"
__all__ = [
    "parse_ndel",
    "Parser", 
    "Lexer",
    "ASTNode",
    "NDELInterpreter",
    "NDELResult",
    "FuzzyResolver",
    "Resolution",
]

# Convenience function
def evaluate(expression: str, context: dict = None, domain: str = "general"):
    """
    Evaluate an NDEL expression.
    
    Args:
        expression: NDEL expression string
        context: Variables and data available to the expression
        domain: Domain for fuzzy resolution
        
    Returns:
        NDELResult with value and confidence
    """
    from .interpreter import NDELInterpreter
    
    interpreter = NDELInterpreter(domain=domain)
    return interpreter.evaluate(expression, context or {})
