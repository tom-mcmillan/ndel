from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict

from ndel.pipeline.semantic_model import Pipeline


def _dataclass_to_dict(obj: Any) -> Any:
    """
    Recursively convert dataclasses and nested structures into plain Python types
    suitable for JSON serialization.
    """

    if is_dataclass(obj):
        return {k: _dataclass_to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(v) for v in obj]
    return obj


def pipeline_to_dict(pipeline: Pipeline) -> Dict[str, Any]:
    """
    Serialize a Pipeline into a JSON-serializable dict suitable for consumption by LLMs.

    The dict exposes:
    - "name": str
    - "description": str | None
    - "datasets": list[dict]
    - "transformations": list[dict]
    - "features": list[dict]
    - "models": list[dict]
    - "metrics": list[dict]
    """

    data = _dataclass_to_dict(pipeline)
    assert isinstance(data, dict)
    return data


def pipeline_to_json(pipeline: Pipeline, **json_kwargs: Any) -> str:
    """Convenience helper: serialize a Pipeline to a JSON string."""

    return json.dumps(pipeline_to_dict(pipeline), **json_kwargs)


__all__ = ["pipeline_to_dict", "pipeline_to_json"]
