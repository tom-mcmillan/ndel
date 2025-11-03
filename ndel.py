"""
NDEL - Non-Deterministic Expression Language
A library for parsing and interpreting fuzzy expressions
"""

class NDEL:
    def __init__(self, domain="general"):
        self.domain = domain
        self.fuzzy_mappings = {
            "young": "< 23",
            "old": "> 30",
            "good": "> 7",
            "high": "> 80"
        }
    
    def parse(self, expression: str) -> dict:
        """Parse NDEL expression into AST"""
        return {
            "type": "expression",
            "domain": self.domain,
            "raw": expression,
            "parsed": self._parse_expression(expression)
        }
    
    def translate(self, text: str, to="ndel") -> str:
        """Translate between natural language and NDEL"""
        if to == "ndel":
            return self._to_ndel(text)
        else:
            return self._to_natural(text)
    
    def _to_ndel(self, natural_text: str) -> str:
        """Convert natural language to NDEL"""
        text = natural_text.lower()
        conditions = []
        
        if "young" in text:
            conditions.append('age < "young"')
        if "good" in text or "strong" in text:
            conditions.append('performance is "good"')
        if "high" in text:
            conditions.append('potential is "high"')
            
        if not conditions:
            conditions.append('criteria meets "requirements"')
            
        return f'@domain("{self.domain}")\n\nwhere {" and ".join(conditions)}'
    
    def _to_natural(self, ndel_expr: str) -> str:
        """Convert NDEL to natural language"""
        if "young" in ndel_expr:
            return "Find young entities with the specified criteria"
        return "Find all matching items"
    
    def _parse_expression(self, expr: str) -> dict:
        # Simple parsing logic
        return {"conditions": expr}

# Export main interface
def translate(text: str, to_format: str = "ndel", domain: str = "general") -> str:
    """Main translation function"""
    ndel = NDEL(domain)
    return ndel.translate(text, to=to_format)
