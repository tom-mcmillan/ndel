from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal

from src.config import AbstractionLevel, NdelConfig


# ---------------------------------------------------------------------------
# Semantic model
# ---------------------------------------------------------------------------


@dataclass
class Dataset:
    name: str
    source: str | None = None
    description: str | None = None
    source_type: Literal["table", "view", "file", "feature_store", "other"] | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:  # pragma: no cover - convenience
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

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.name


@dataclass
class Feature:
    name: str
    description: str
    origin: str | None = None
    data_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:  # pragma: no cover - convenience
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

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.name


@dataclass
class Metric:
    name: str
    description: str | None = None
    dataset: str | None = None
    higher_is_better: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:  # pragma: no cover - convenience
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

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.name


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


@dataclass
class PipelineDiff:
    datasets_added: list[str] = field(default_factory=list)
    datasets_removed: list[str] = field(default_factory=list)
    transformations_added: list[str] = field(default_factory=list)
    transformations_removed: list[str] = field(default_factory=list)
    features_added: list[str] = field(default_factory=list)
    features_removed: list[str] = field(default_factory=list)
    models_added: list[str] = field(default_factory=list)
    models_removed: list[str] = field(default_factory=list)
    metrics_added: list[str] = field(default_factory=list)
    metrics_removed: list[str] = field(default_factory=list)


def diff_pipelines(old: Pipeline, new: Pipeline) -> PipelineDiff:
    def names(items):
        return {getattr(i, "name", str(i)) for i in items}

    old_ds, new_ds = names(old.datasets), names(new.datasets)
    old_tf, new_tf = names(old.transformations), names(new.transformations)
    old_feat, new_feat = names(old.features), names(new.features)
    old_models, new_models = names(old.models), names(new.models)
    old_metrics, new_metrics = names(old.metrics), names(new.metrics)

    return PipelineDiff(
        datasets_added=sorted(new_ds - old_ds),
        datasets_removed=sorted(old_ds - new_ds),
        transformations_added=sorted(new_tf - old_tf),
        transformations_removed=sorted(old_tf - new_tf),
        features_added=sorted(new_feat - old_feat),
        features_removed=sorted(old_feat - new_feat),
        models_added=sorted(new_models - old_models),
        models_removed=sorted(old_models - new_models),
        metrics_added=sorted(new_metrics - old_metrics),
        metrics_removed=sorted(old_metrics - new_metrics),
    )


# ---------------------------------------------------------------------------
# Lineage merge
# ---------------------------------------------------------------------------


def merge_pipelines(p_sql: Pipeline, p_py: Pipeline) -> Pipeline:
    """Merge SQL and Python pipelines by matching dataset sources."""

    sql_sources = {ds.source or ds.name: ds for ds in p_sql.datasets}
    merged_datasets = list(p_sql.datasets)

    for ds in p_py.datasets:
        key = ds.source or ds.name
        if key in sql_sources:
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


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@dataclass
class ValidationIssue:
    kind: str  # e.g., "warning", "error"
    code: str  # e.g., "UNKNOWN_DATASET_ALIAS"
    message: str


def validate_config_against_pipeline(config: NdelConfig, pipeline: Pipeline) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    datasets = {ds.name for ds in pipeline.datasets}
    models = {m.name for m in pipeline.models}
    features = {f.name for f in pipeline.features}

    domain = config.domain
    if domain:
        for key in domain.dataset_aliases:
            if key not in datasets:
                issues.append(
                    ValidationIssue(
                        kind="warning",
                        code="UNKNOWN_DATASET_ALIAS",
                        message=f"Dataset alias '{key}' does not match any dataset in pipeline.",
                    )
                )
        for key in domain.model_aliases:
            if key not in models:
                issues.append(
                    ValidationIssue(
                        kind="warning",
                        code="UNKNOWN_MODEL_ALIAS",
                        message=f"Model alias '{key}' does not match any model in pipeline.",
                    )
                )
        for key in domain.feature_aliases:
            if key not in features:
                issues.append(
                    ValidationIssue(
                        kind="warning",
                        code="UNKNOWN_FEATURE_ALIAS",
                        message=f"Feature alias '{key}' does not match any feature in pipeline.",
                    )
                )

    pii_like = {"email", "ip", "phone", "address", "ssn"}
    names_union = {n.lower() for n in datasets | models | features}
    hits = pii_like & names_union
    if (config.privacy is None or not config.privacy.redact_identifiers) and hits:
        issues.append(
            ValidationIssue(
                kind="warning",
                code="MISSING_PII_REDACTION",
                message=f"Potential PII identifiers detected ({', '.join(sorted(hits))}) but redact_identifiers is empty.",
            )
        )

    if config.abstraction == AbstractionLevel.HIGH:
        if len(pipeline.features) > 5 or len(pipeline.transformations) > 5:
            issues.append(
                ValidationIssue(
                    kind="warning",
                    code="ABSTRACTION_HIGH_WITH_DETAILS",
                    message="AbstractionLevel.HIGH with many features/transformations; consider MEDIUM for more detail.",
                )
            )

    return issues


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _dataclass_to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {k: _dataclass_to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(v) for v in obj]
    return obj


def pipeline_to_dict(pipeline: Pipeline) -> dict[str, Any]:
    data = _dataclass_to_dict(pipeline)
    assert isinstance(data, dict)
    return data


def pipeline_to_json(pipeline: Pipeline, **json_kwargs: Any) -> str:
    return json.dumps(pipeline_to_dict(pipeline), **json_kwargs)


__all__ = [
    "Dataset",
    "Transformation",
    "Feature",
    "Model",
    "Metric",
    "Pipeline",
    "PipelineDiff",
    "ValidationIssue",
    "diff_pipelines",
    "merge_pipelines",
    "validate_config_against_pipeline",
    "pipeline_to_dict",
    "pipeline_to_json",
]
