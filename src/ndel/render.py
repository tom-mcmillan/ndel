from __future__ import annotations

import re
from typing import Any

from ndel.config import NdelConfig
from .semantic_model import Dataset, Feature, Metric, Model, Pipeline, Transformation


def _apply_privacy(value: str, config: NdelConfig | None) -> str:
    """Apply privacy rules to a string value based on NdelConfig.

    - Redacts table-like names when hide_table_names is enabled.
    - Redacts filesystem-like or URI paths when hide_file_paths is enabled.
    - Masks identifiers listed in redact_identifiers.
    - Truncates literals longer than max_literal_length.
    """

    if config is None or config.privacy is None:
        return value

    privacy = config.privacy
    text = value

    if privacy.hide_table_names:
        table_pattern = re.compile(r"\b[\w-]+(?:\.[\w-]+)+\b")
        text = table_pattern.sub("<redacted-table>", text)

    if privacy.hide_file_paths:
        path_pattern = re.compile(r"(s3://\S+|gs://\S+|\b[A-Za-z]:[\\/][^\s]+|(?<!\w)/[^\s]+)")
        text = path_pattern.sub("<redacted-path>", text)

    if privacy.redact_identifiers:
        for ident in privacy.redact_identifiers:
            if not ident:
                continue
            ident_pattern = re.compile(re.escape(ident), flags=re.IGNORECASE)
            text = ident_pattern.sub("<redacted>", text)

    max_len = privacy.max_literal_length
    if max_len is not None and max_len > 0 and len(text) > max_len:
        text = text[:max_len] + "..."

    return text


def render_pipeline(pipeline: Pipeline, config: NdelConfig | None = None) -> str:
    """Render a Pipeline into human-readable NDEL text."""

    def indent(level: int) -> str:
        return "  " * level

    def sanitize(text: str) -> str:
        return _apply_privacy(text, config)

    pipeline_name_raw = config.domain.pipeline_name if config and config.domain else pipeline.name
    pipeline_name = sanitize(pipeline_name_raw)
    lines: list[str] = [f'pipeline "{pipeline_name}":']

    if pipeline.description:
        lines.append(f"{indent(1)}description: {sanitize(pipeline.description)}")

    def render_notes(notes: list[str], level: int) -> None:
        if notes:
            lines.append(f"{indent(level)}notes:")
            for note in notes:
                lines.append(f"{indent(level + 1)}- {sanitize(note)}")

    def render_str_list(label: str, items: list[str], level: int) -> None:
        if items:
            lines.append(f"{indent(level)}{label}:")
            for item in sorted(items, key=lambda x: sanitize(x)):
                lines.append(f"{indent(level + 1)}- {sanitize(item)}")

    def render_kv_dict(label: str, values: dict[str, Any], level: int) -> None:
        if values:
            lines.append(f"{indent(level)}{label}:")
            for key in sorted(values):
                sanitized_key = sanitize(str(key))
                sanitized_value = sanitize(str(values[key]))
                lines.append(f"{indent(level + 1)}{sanitized_key}: {sanitized_value}")

    def render_dataset(dataset: Dataset, level: int) -> None:
        lines.append(f"{indent(level)}- name: {sanitize(dataset.name)}")
        if dataset.description:
            lines.append(f"{indent(level + 1)}description: {sanitize(dataset.description)}")
        if dataset.source_type:
            lines.append(f"{indent(level + 1)}source_type: {sanitize(dataset.source_type)}")
        render_notes(dataset.notes, level + 1)

    def render_transformation(transformation: Transformation, level: int) -> None:
        lines.append(f"{indent(level)}- name: {sanitize(transformation.name)}")
        lines.append(f"{indent(level + 1)}description: {sanitize(transformation.description)}")
        if transformation.kind:
            lines.append(f"{indent(level + 1)}kind: {sanitize(transformation.kind)}")
        render_str_list("inputs", transformation.inputs, level + 1)
        render_str_list("outputs", transformation.outputs, level + 1)

    def render_feature(feature: Feature, level: int) -> None:
        lines.append(f"{indent(level)}- name: {sanitize(feature.name)}")
        lines.append(f"{indent(level + 1)}description: {sanitize(feature.description)}")
        if feature.origin:
            lines.append(f"{indent(level + 1)}origin: {sanitize(feature.origin)}")
        if feature.data_type:
            lines.append(f"{indent(level + 1)}data_type: {sanitize(feature.data_type)}")

    def render_model(model: Model, level: int) -> None:
        lines.append(f"{indent(level)}- name: {sanitize(model.name)}")
        lines.append(f"{indent(level + 1)}task: {sanitize(model.task)}")
        if model.algorithm_family:
            lines.append(f"{indent(level + 1)}algorithm_family: {sanitize(model.algorithm_family)}")
        render_str_list("inputs", model.inputs, level + 1)
        if model.target:
            lines.append(f"{indent(level + 1)}target: {sanitize(model.target)}")
        if model.description:
            lines.append(f"{indent(level + 1)}description: {sanitize(model.description)}")
        if model.hyperparameters:
            render_kv_dict("hyperparameters", model.hyperparameters, level + 1)

    def render_metric(metric: Metric, level: int) -> None:
        lines.append(f"{indent(level)}- name: {sanitize(metric.name)}")
        if metric.description:
            lines.append(f"{indent(level + 1)}description: {sanitize(metric.description)}")
        if metric.dataset:
            lines.append(f"{indent(level + 1)}dataset: {sanitize(metric.dataset)}")
        if metric.higher_is_better is not None:
            bool_value = "true" if metric.higher_is_better else "false"
            lines.append(f"{indent(level + 1)}higher_is_better: {bool_value}")

    def add_section(title: str, items: list[Any], renderer) -> None:
        lines.append(f"{indent(1)}{title}:")
        for item in sorted(items, key=lambda x: sanitize(x.name)):
            renderer(item, level=2)

    add_section("datasets", pipeline.datasets, render_dataset)
    add_section("transformations", pipeline.transformations, render_transformation)
    add_section("features", pipeline.features, render_feature)
    add_section("models", pipeline.models, render_model)
    add_section("metrics", pipeline.metrics, render_metric)

    return "\n".join(lines) + "\n"


__all__ = ["render_pipeline"]
