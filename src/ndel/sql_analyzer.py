from __future__ import annotations

import re

from ndel.config import NdelConfig
from ndel.semantic_model import Dataset, Pipeline, Transformation


def analyze_sql_source(sql: str, config: NdelConfig | None = None) -> Pipeline:
    """Minimal SQL analyzer stub.

    Detects simple SELECT ... FROM clauses and WHERE filters to populate
    datasets and transformations. Future work will expand this to full SQL
    parsing and richer semantics.
    """

    datasets: list[Dataset] = []
    transformations: list[Transformation] = []

    from_match = re.search(r"from\s+([\w\.]+)", sql, flags=re.IGNORECASE)
    table_name = from_match.group(1) if from_match else "unknown_table"

    dataset = Dataset(name=table_name, description="dataset referenced in SQL", source_type="table")
    datasets.append(dataset)

    where_match = re.search(r"where\s+(.+)", sql, flags=re.IGNORECASE | re.DOTALL)
    if where_match:
        condition = where_match.group(1).strip()
        transformations.append(
            Transformation(
                name="sql_where_filter",
                description=f"filter rows with condition: {condition}",
                kind="filter",
                inputs=[table_name],
                outputs=[table_name],
            )
        )

    return Pipeline(
        name="sql_pipeline",
        datasets=datasets,
        transformations=transformations,
        features=[],
        models=[],
        metrics=[],
        description=None,
    )


__all__ = ["analyze_sql_source"]
