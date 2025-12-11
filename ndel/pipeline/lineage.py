from __future__ import annotations

from ndel.pipeline.semantic_model import Pipeline


def merge_pipelines(p_sql: Pipeline, p_py: Pipeline) -> Pipeline:
    """Merge SQL and Python pipelines by matching dataset sources.

    If a Python dataset shares the same source as a SQL dataset, they are
    treated as the same logical dataset. Transformations, features, and models
    are concatenated in a simple SQL-then-Python order.
    """

    sql_sources = {ds.source or ds.name: ds for ds in p_sql.datasets}
    merged_datasets = list(p_sql.datasets)

    for ds in p_py.datasets:
        key = ds.source or ds.name
        if key in sql_sources:
            # Use SQL dataset identity but keep python name if different
            sql_ds = sql_sources[key]
            if ds.name != sql_ds.name:
                sql_ds.notes.append(f"Also referenced in Python as {ds.name}")
        else:
            merged_datasets.append(ds)

    merged_transformations = list(p_sql.transformations) + list(p_py.transformations)
    merged_features = list(p_sql.features) + list(p_py.features)
    merged_models = list(p_sql.models) + list(p_py.models)
    merged_metrics = list(p_sql.metrics) + list(p_py.metrics)

    return Pipeline(
        name=p_py.name or p_sql.name or "combined_pipeline",
        datasets=merged_datasets,
        transformations=merged_transformations,
        features=merged_features,
        models=merged_models,
        metrics=merged_metrics,
        description=None,
    )


__all__ = ["merge_pipelines"]
