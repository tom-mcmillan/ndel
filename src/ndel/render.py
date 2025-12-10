from __future__ import annotations

from typing import Any

from .semantic_model import Dataset, Feature, Metric, Model, Pipeline, Transformation


def render_pipeline(pipeline: Pipeline) -> str:
    """Render a Pipeline into human-readable NDEL text."""

    def indent(level: int) -> str:
        return "  " * level

    lines: list[str] = [f'pipeline "{pipeline.name}":']

    if pipeline.description:
        lines.append(f"{indent(1)}description: {pipeline.description}")

    def render_notes(notes: list[str], level: int) -> None:
        if notes:
            lines.append(f"{indent(level)}notes:")
            for note in notes:
                lines.append(f"{indent(level + 1)}- {note}")

    def render_str_list(label: str, items: list[str], level: int) -> None:
        if items:
            lines.append(f"{indent(level)}{label}:")
            for item in sorted(items):
                lines.append(f"{indent(level + 1)}- {item}")

    def render_kv_dict(label: str, values: dict[str, Any], level: int) -> None:
        if values:
            lines.append(f"{indent(level)}{label}:")
            for key in sorted(values):
                lines.append(f"{indent(level + 1)}{key}: {values[key]}")

    def render_dataset(dataset: Dataset, level: int) -> None:
        lines.append(f"{indent(level)}- name: {dataset.name}")
        if dataset.description:
            lines.append(f"{indent(level + 1)}description: {dataset.description}")
        if dataset.source_type:
            lines.append(f"{indent(level + 1)}source_type: {dataset.source_type}")
        render_notes(dataset.notes, level + 1)

    def render_transformation(transformation: Transformation, level: int) -> None:
        lines.append(f"{indent(level)}- name: {transformation.name}")
        lines.append(f"{indent(level + 1)}description: {transformation.description}")
        if transformation.kind:
            lines.append(f"{indent(level + 1)}kind: {transformation.kind}")
        render_str_list("inputs", transformation.inputs, level + 1)
        render_str_list("outputs", transformation.outputs, level + 1)

    def render_feature(feature: Feature, level: int) -> None:
        lines.append(f"{indent(level)}- name: {feature.name}")
        lines.append(f"{indent(level + 1)}description: {feature.description}")
        if feature.origin:
            lines.append(f"{indent(level + 1)}origin: {feature.origin}")
        if feature.data_type:
            lines.append(f"{indent(level + 1)}data_type: {feature.data_type}")

    def render_model(model: Model, level: int) -> None:
        lines.append(f"{indent(level)}- name: {model.name}")
        lines.append(f"{indent(level + 1)}task: {model.task}")
        if model.algorithm_family:
            lines.append(f"{indent(level + 1)}algorithm_family: {model.algorithm_family}")
        render_str_list("inputs", model.inputs, level + 1)
        if model.target:
            lines.append(f"{indent(level + 1)}target: {model.target}")
        if model.description:
            lines.append(f"{indent(level + 1)}description: {model.description}")
        if model.hyperparameters:
            render_kv_dict("hyperparameters", model.hyperparameters, level + 1)

    def render_metric(metric: Metric, level: int) -> None:
        lines.append(f"{indent(level)}- name: {metric.name}")
        if metric.description:
            lines.append(f"{indent(level + 1)}description: {metric.description}")
        if metric.dataset:
            lines.append(f"{indent(level + 1)}dataset: {metric.dataset}")
        if metric.higher_is_better is not None:
            bool_value = "true" if metric.higher_is_better else "false"
            lines.append(f"{indent(level + 1)}higher_is_better: {bool_value}")

    def add_section(title: str, items: list[Any], renderer) -> None:
        lines.append(f"{indent(1)}{title}:")
        for item in sorted(items, key=lambda x: x.name):
            renderer(item, level=2)

    add_section("datasets", pipeline.datasets, render_dataset)
    add_section("transformations", pipeline.transformations, render_transformation)
    add_section("features", pipeline.features, render_feature)
    add_section("models", pipeline.models, render_model)
    add_section("metrics", pipeline.metrics, render_metric)

    return "\n".join(lines) + "\n"


__all__ = ["render_pipeline"]
