from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass
class Dataset:
    name: str
    description: str | None = None
    source_type: Literal["table", "view", "file", "feature_store", "other"] | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        return self.name


@dataclass
class Transformation:
    name: str
    description: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    kind: Literal["filter", "aggregation", "join", "feature_engineering", "other"] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        return self.name


@dataclass
class Feature:
    name: str
    description: str
    origin: str | None = None
    data_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        return self.name


@dataclass
class Model:
    name: str
    task: str
    algorithm_family: str | None = None
    inputs: list[str] = field(default_factory=list)
    target: str | None = None
    description: str | None = None
    hyperparameters: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        return self.name


@dataclass
class Metric:
    name: str
    description: str | None = None
    dataset: str | None = None
    higher_is_better: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        return self.name


@dataclass
class Pipeline:
    name: str
    datasets: list[Dataset] = field(default_factory=list)
    transformations: list[Transformation] = field(default_factory=list)
    features: list[Feature] = field(default_factory=list)
    models: list[Model] = field(default_factory=list)
    metrics: list[Metric] = field(default_factory=list)
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        return self.name


__all__ = [
    "Dataset",
    "Transformation",
    "Feature",
    "Model",
    "Metric",
    "Pipeline",
]
