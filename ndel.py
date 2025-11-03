"""NDEL - Non-Deterministic Expression Language"""

def translate(text: str, to_format: str = "ndel", domain: str = "general") -> str:
    """Translate between natural language and NDEL"""
    if to_format == "ndel":
        # Natural to NDEL
        conditions = []
        lower = text.lower()
        
        if "young" in lower:
            conditions.append('age < "young"')
        if "good" in lower or "strong" in lower:
            conditions.append('performance is "good"')
        if "high" in lower:
            conditions.append('potential is "high"')
            
        if not conditions:
            conditions.append('criteria meets "requirements"')
            
        return f'@domain("{domain}")\n\nwhere {" and ".join(conditions)}'
    else:
        # NDEL to Natural
        if "young" in text:
            return "Find young entities with the specified criteria"
        return "Find all matching items"
