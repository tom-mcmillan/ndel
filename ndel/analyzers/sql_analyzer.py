from __future__ import annotations

import re
from typing import List

from ndel.config.core import NdelConfig
from ndel.pipeline.semantic_model import Dataset, Feature, Pipeline, Transformation


def analyze_sql_source(sql: str, config: NdelConfig | None = None) -> Pipeline:
    """Lightweight SQL analysis into a Pipeline representation."""

    text = sql
    datasets: list[Dataset] = []
    transformations: list[Transformation] = []
    features: list[Feature] = []

    # Table extraction
    tables: List[str] = []
    tables += re.findall(r"\bfrom\s+([\w\.]+)", text, flags=re.IGNORECASE)
    tables += re.findall(r"\bjoin\s+([\w\.]+)", text, flags=re.IGNORECASE)
    tables = list(dict.fromkeys(tables))  # de-dupe
    if not tables:
        tables = ["unknown_table"]

    for tbl in tables:
        datasets.append(Dataset(name=tbl, source=tbl, description="table referenced in SQL", source_type="table"))

    # Joins
    join_pattern = re.compile(
        r"join\s+([\w\.]+)(?:\s+\w+)?\s+on\s+(.+?)(?=\bjoin\b|\bwhere\b|\bgroup by\b|\bhaving\b|\border by\b|$)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in join_pattern.finditer(text):
        table = match.group(1)
        condition = " ".join(match.group(2).split())
        transformations.append(
            Transformation(
                name=f"join_{table}",
                description=f"join {table} on {condition}",
                kind="join",
                inputs=[tables[0], table] if tables else [table],
                outputs=[tables[0]],
            )
        )

    # WHERE filters
    where_match = re.search(r"where\s+(.+?)(group by|having|order by|$)", text, flags=re.IGNORECASE | re.DOTALL)
    if where_match:
        condition = where_match.group(1).strip()
        transformations.append(
            Transformation(
                name="sql_where_filter",
                description=f"filter rows with condition: {condition}",
                kind="filter",
                inputs=[tables[0]],
                outputs=[tables[0]],
            )
        )

    # GROUP BY / aggregation
    group_match = re.search(r"group by\s+(.+?)(having|order by|$)", text, flags=re.IGNORECASE | re.DOTALL)
    aggregates = re.findall(r"\b(count|sum|avg|mean|min|max)\s*\(", text, flags=re.IGNORECASE)
    if group_match or aggregates:
        group_cols = group_match.group(1).strip() if group_match else ""
        agg_desc = "aggregate"
        if aggregates:
            agg_desc = f"aggregate using {', '.join(set(a.lower() for a in aggregates))}"
        if group_cols:
            agg_desc += f" grouped by {group_cols}"
        transformations.append(
            Transformation(
                name="sql_groupby_agg",
                description=agg_desc,
                kind="aggregation",
                inputs=[tables[0]],
                outputs=[tables[0]],
            )
        )

    # Projection / derived columns
    select_match = re.search(r"select\s+(.+?)\s+from", text, flags=re.IGNORECASE | re.DOTALL)
    if select_match:
        select_clause = select_match.group(1)
        derived_cols = re.findall(r"\bas\s+([\w_]+)", select_clause, flags=re.IGNORECASE)
        if derived_cols:
            for col in derived_cols:
                features.append(
                    Feature(
                        name=col,
                        description="derived column from SELECT",
                        origin=tables[0],
                        data_type=None,
                    )
                )
            transformations.append(
                Transformation(
                    name="sql_projection",
                    description=f"compute derived columns: {', '.join(derived_cols)}",
                    kind="feature_engineering",
                    inputs=[tables[0]],
                    outputs=derived_cols,
                )
            )

    # Apply domain aliases if present
    if config and config.domain:
        alias_map = config.domain.dataset_aliases
        for ds in datasets:
            if ds.name in alias_map:
                ds.name = alias_map[ds.name]

    pipeline = Pipeline(
        name="sql_pipeline",
        datasets=datasets,
        transformations=transformations,
        features=features,
        models=[],
        metrics=[],
        description=None,
    )
    return pipeline


__all__ = ["analyze_sql_source"]
