from __future__ import annotations

import json
from typing import Any, Dict

from ndel.semantic_model import Dataset, Feature, Metric, Model, Pipeline, Transformation


def _dataset_to_dict(dataset: Dataset) -> Dict[str, Any]:
    return {
        "name": dataset.name,
        "source": dataset.source,
        "description": dataset.description,
        "source_type": dataset.source_type,
        "notes": list(dataset.notes),
    }


def _transformation_to_dict(transformation: Transformation) -> Dict[str, Any]:
    return {
        "name": transformation.name,
        "description": transformation.description,
        "kind": transformation.kind,
        "inputs": list(transformation.inputs),
        "outputs": list(transformation.outputs),
    }


def _feature_to_dict(feature: Feature) -> Dict[str, Any]:
    return {
        "name": feature.name,
        "description": feature.description,
        "origin": feature.origin,
        "data_type": feature.data_type,
    }


def _model_to_dict(model: Model) -> Dict[str, Any]:
    return {
        "name": model.name,
        "task": model.task,
        "algorithm_family": model.algorithm_family,
        "inputs": list(model.inputs),
        "target": model.target,
        "description": model.description,
        "hyperparameters": model.hyperparameters if model.hyperparameters is not None else None,
    }


def _metric_to_dict(metric: Metric) -> Dict[str, Any]:
    return {
        "name": metric.name,
        "description": metric.description,
        "dataset": metric.dataset,
        "higher_is_better": metric.higher_is_better,
    }


def pipeline_to_dict(pipeline: Pipeline) -> Dict[str, Any]:
    """
    Serialize a Pipeline into a JSON-serializable dict suitable for consumption by LLMs.

    Exposes semantic structure:
    - pipeline name/description
    - datasets (name, source, description, source_type, notes)
    - transformations (name, description, kind, inputs, outputs)
    - features (name, description, origin, data_type)
    - models (name, task, algorithm_family, inputs, target, description, hyperparameters)
    - metrics (name, description, dataset, higher_is_better)
    """

    return {
        "name": pipeline.name,
        "description": pipeline.description,
        "datasets": [_dataset_to_dict(ds) for ds in pipeline.datasets],
        "transformations": [_transformation_to_dict(t) for t in pipeline.transformations],
        "features": [_feature_to_dict(f) for f in pipeline.features],
        "models": [_model_to_dict(m) for m in pipeline.models],
        "metrics": [_metric_to_dict(m) for m in pipeline.metrics],
    }


def pipeline_to_json(pipeline: Pipeline, **json_kwargs: Any) -> str:
    """Convenience helper that dumps a Pipeline to a JSON string."""

    return json.dumps(pipeline_to_dict(pipeline), **json_kwargs)


__all__ = ["pipeline_to_dict", "pipeline_to_json"]
