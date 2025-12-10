"""Translate Python code to NDEL descriptions."""

import re
from typing import Optional, List


def translate(python_code: str) -> str:
    """
    Convert Python code to NDEL description.

    Extracts SQL queries and pandas operations, renders as NDEL.
    Returns technical names - domain mapping applied separately.
    """
    try:
        parts = []

        # Extract SQL queries
        sql = _extract_sql(python_code)
        if sql:
            parts.append(_sql_to_ndel(sql))

        # Extract pandas operations
        pandas_ops = _extract_pandas_ops(python_code)
        for op in pandas_ops:
            parts.append(op)

        if parts:
            return "\n".join(parts)
        else:
            return "Analysis performed"

    except Exception:
        return "Analysis performed"


def _extract_sql(code: str) -> Optional[str]:
    """Extract SQL query string from Python code."""
    # Look for triple-quoted SQL
    pattern = r'"""([\s\S]*?)"""'
    matches = re.findall(pattern, code)
    for match in matches:
        if re.search(r'\bSELECT\b', match, re.IGNORECASE):
            return match.strip()

    # Look for single-quoted SQL in pd.read_sql
    pattern = r'read_sql\s*\(\s*["\']([^"\']+)["\']'
    matches = re.findall(pattern, code)
    for match in matches:
        if re.search(r'\bSELECT\b', match, re.IGNORECASE):
            return match.strip()

    return None


def _sql_to_ndel(sql: str) -> str:
    """Convert SQL query to NDEL syntax."""
    lines = []

    # SELECT -> FIND
    select_match = re.search(
        r'SELECT\s+(.*?)\s+FROM',
        sql,
        re.IGNORECASE | re.DOTALL
    )
    from_match = re.search(
        r'FROM\s+(\w+)',
        sql,
        re.IGNORECASE
    )

    if select_match and from_match:
        columns = select_match.group(1).strip()
        columns = re.sub(r'\s+', ' ', columns)  # Normalize whitespace
        table = from_match.group(1)
        lines.append(f"FIND {columns} FROM {table}")

    # WHERE
    where_match = re.search(
        r'WHERE\s+(.*?)(?:ORDER BY|GROUP BY|LIMIT|$)',
        sql,
        re.IGNORECASE | re.DOTALL
    )
    if where_match:
        conditions = where_match.group(1).strip()
        conditions = re.sub(r'\s+', ' ', conditions)
        # Convert = 'value' to = "value"
        conditions = re.sub(r"=\s*'([^']*)'", r'= "\1"', conditions)
        lines.append(f"WHERE {conditions}")

    # GROUP BY
    group_match = re.search(
        r'GROUP BY\s+(.*?)(?:ORDER BY|HAVING|LIMIT|$)',
        sql,
        re.IGNORECASE | re.DOTALL
    )
    if group_match:
        grouping = group_match.group(1).strip()
        grouping = re.sub(r'\s+', ' ', grouping)
        lines.append(f"GROUP BY {grouping}")

    # ORDER BY -> RANK BY
    order_match = re.search(
        r'ORDER BY\s+(.*?)(?:LIMIT|$)',
        sql,
        re.IGNORECASE | re.DOTALL
    )
    if order_match:
        ordering = order_match.group(1).strip()
        lines.append(f"RANK BY {ordering}")

    # LIMIT
    limit_match = re.search(r'LIMIT\s+(\d+)', sql, re.IGNORECASE)
    if limit_match:
        lines.append(f"LIMIT {limit_match.group(1)}")

    return "\n".join(lines)


def _extract_pandas_ops(code: str) -> List[str]:
    """Extract pandas operations and convert to NDEL."""
    ops = []

    # df['new_col'] = expression -> COMPUTE
    compute_pattern = r"df\['(\w+)'\]\s*=\s*(.+)"
    for match in re.finditer(compute_pattern, code):
        col_name = match.group(1)
        expression = match.group(2).strip()
        # Clean up the expression
        expression = re.sub(r"df\['(\w+)'\]", r"\1", expression)
        ops.append(f"COMPUTE {col_name} = {expression}")

    # df.sort_values('col', ascending=False) -> RANK BY col DESC
    sort_pattern = r"sort_values\s*\(\s*['\"](\w+)['\"].*?ascending\s*=\s*False"
    for match in re.finditer(sort_pattern, code):
        ops.append(f"RANK BY {match.group(1)} DESC")

    # df.sort_values('col') without ascending=False
    sort_pattern_asc = r"sort_values\s*\(\s*['\"](\w+)['\"](?!.*ascending\s*=\s*False)"
    for match in re.finditer(sort_pattern_asc, code):
        if "ascending=False" not in match.group(0):
            ops.append(f"RANK BY {match.group(1)}")

    return ops


__all__ = ["translate"]
