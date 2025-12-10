"""Runtime domain configuration support."""

from typing import TypedDict, Optional
import re


class DomainConfig(TypedDict, total=False):
    """Domain config provided by consuming application at runtime."""
    tables: dict[str, str]   # "technical_name" -> "Friendly Name"
    columns: dict[str, str]  # "technical_name" -> "Friendly Name"


def apply_domain(text: str, domain: Optional[DomainConfig]) -> str:
    """Apply domain vocabulary mappings to text."""
    if not domain:
        return text

    result = text

    # Replace table names (after FROM keyword)
    for technical, friendly in domain.get("tables", {}).items():
        result = re.sub(
            rf'\bFROM\s+{re.escape(technical)}\b',
            f'FROM {friendly}',
            result,
            flags=re.IGNORECASE
        )

    # Replace column names (word boundaries)
    for technical, friendly in domain.get("columns", {}).items():
        result = re.sub(rf'\b{re.escape(technical)}\b', friendly, result)

    return result


__all__ = ["DomainConfig", "apply_domain"]
