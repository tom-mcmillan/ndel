from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List

from src.config import AbstractionLevel, NdelConfig
from src.schema import (
    Dataset,
    Feature,
    Metric,
    Model,
    Pipeline,
    Transformation,
    ValidationIssue,
    pipeline_to_dict,
    validate_pipeline_structure,
)

# ---------------------------------------------------------------------------
# Grammar and primitives
# ---------------------------------------------------------------------------


PRIMITIVES = {
    "sections": ["datasets", "transformations", "features", "models", "metrics"],
    "transformation_kinds": ["filter", "aggregation", "join", "feature_engineering", "other"],
    "source_types": ["table", "view", "file", "feature_store", "other"],
}


NDEL_GRAMMAR = r"""
pipeline "<name>":
  [description: <text>]
  datasets:
    - name: <identifier>
      [description: <text>]
      [source_type: (table|view|file|feature_store|other)]
      [notes:
        - <text>...]
  transformations:
    - name: <identifier>
      description: <text>
      [kind: (filter|aggregation|join|feature_engineering|other)]
      [inputs:
        - <identifier>...]
      [outputs:
        - <identifier>...]
  features:
    - name: <identifier>
      description: <text>
      [origin: <identifier>]
      [data_type: <text>]
  models:
    - name: <identifier>
      task: <text>
      [algorithm_family: <text>]
      [inputs:
        - <identifier>...]
      [target: <identifier>]
      [description: <text>]
      [hyperparameters:
        key: value ...]
  metrics:
    - name: <identifier>
      [description: <text>]
      [dataset: <identifier>]
      [higher_is_better: (true|false)]
""".strip()


def describe_grammar() -> str:
    return NDEL_GRAMMAR


def validate_ndel_text(ndel_text: str) -> List[str]:
    warnings: List[str] = []
    lines = [ln.rstrip("\n") for ln in ndel_text.splitlines() if ln.strip()]
    if not lines:
        return ["empty NDEL text"]

    if not lines[0].lstrip().startswith("pipeline "):
        warnings.append('first line should start with "pipeline "')

    for idx, line in enumerate(lines, start=1):
        if "\t" in line:
            warnings.append(f"line {idx}: tabs found; use spaces")
        indent = len(line) - len(line.lstrip(" "))
        if indent % 2 != 0:
            warnings.append(f"line {idx}: indent not a multiple of 2 spaces")

    section_pattern = re.compile(r"^\s{2}(\w+):\s*$")
    for line in lines:
        m = section_pattern.match(line)
        if m:
            section = m.group(1)
            if section not in PRIMITIVES["sections"] and section != "pipeline":
                warnings.append(f"unknown section '{section}'")

    bullet_pattern = re.compile(r"^\s{4}- ")
    for idx, line in enumerate(lines, start=1):
        if line.strip().startswith("-") and not bullet_pattern.match(line):
            warnings.append(f"line {idx}: list items must be indented by 4 spaces")

    return warnings


# ---------------------------------------------------------------------------
# Privacy helpers
# ---------------------------------------------------------------------------


def apply_privacy(value: str, config: NdelConfig | None) -> str:
    """Apply privacy rules from config to a string."""

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


def apply_privacy_to_payload(value: Any, config: NdelConfig | None) -> Any:
    """Recursively apply privacy rules to strings inside a payload."""

    if isinstance(value, str):
        return apply_privacy(value, config)
    if isinstance(value, dict):
        return {k: apply_privacy_to_payload(v, config) for k, v in value.items()}
    if isinstance(value, list):
        return [apply_privacy_to_payload(v, config) for v in value]
    return value


# ---------------------------------------------------------------------------
# Deterministic rendering
# ---------------------------------------------------------------------------


def render_pipeline(pipeline: Pipeline, config: NdelConfig | None = None) -> str:
    """
    Deterministic renderer for NDEL text.
    """

    def indent(level: int) -> str:
        return "  " * level

    def sanitize(text: str) -> str:
        return apply_privacy(text, config)

    abstraction = config.abstraction if config else AbstractionLevel.MEDIUM

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

    if abstraction in {AbstractionLevel.MEDIUM, AbstractionLevel.LOW}:
        add_section("transformations", pipeline.transformations, render_transformation)

    if abstraction == AbstractionLevel.LOW:
        feature_list = pipeline.features
    elif abstraction == AbstractionLevel.MEDIUM:
        feature_list = [f for f in pipeline.features if f.description and "training" in f.description.lower()]
    else:
        feature_list = []

    if abstraction != AbstractionLevel.HIGH:
        add_section("features", feature_list, render_feature)

    add_section("models", pipeline.models, render_model)
    add_section("metrics", pipeline.metrics, render_metric)

    output = "\n".join(lines) + "\n"

    warnings = validate_ndel_text(output)
    if warnings:
        lines.append(f"# validator_warnings: {'; '.join(warnings)}")
        output = "\n".join(lines) + "\n"

    return output


# ---------------------------------------------------------------------------
# LLM prompt and rendering
# ---------------------------------------------------------------------------


LLMGenerate = Callable[[str], str]

DEFAULT_NDEL_PROMPT = """You write NDEL (Non-Deterministic Expression Language) descriptions for data/ML pipelines.
You receive a pipeline JSON with datasets, transformations, features, models, and metrics.
- Preserve structure and lineage; do not invent entities.
- Output indentation-based NDEL text (sections like pipeline, datasets, transformations, features, models, metrics).
- Privacy/abstraction may already be applied; do not re-infer sensitive details.
- Follow the NDEL grammar and allowed sections/kinds provided below.
- Honor the configuration notes and do not undo redactions or abstractions.

Pipeline JSON (ground truth):
{pipeline_json}

Grammar and rules:
{ndel_grammar}

Config notes:
{config_notes}

Write the NDEL description now, using only the provided structure."""


def build_ndel_prompt(
    pipeline: Pipeline,
    config: NdelConfig | None = None,
    extra_instructions: str | None = None,
) -> str:
    """
    Build a text prompt to send to an LLM, instructing it to write an NDEL-style description.
    """

    data: Dict[str, Any] = pipeline_to_dict(pipeline)
    pipeline_json = json.dumps(data, indent=2, sort_keys=True)

    config_lines: list[str] = []
    if config:
        config_lines.append("Configuration hints (applied upstream; do not re-expand hidden details):")
        config_lines.append(f"- Abstraction level: {config.abstraction.value if isinstance(config.abstraction, AbstractionLevel) else config.abstraction}")
        if config.privacy:
            config_lines.append(
                "- Privacy: "
                f"hide_table_names={config.privacy.hide_table_names}, "
                f"hide_file_paths={config.privacy.hide_file_paths}, "
                f"redact_identifiers={config.privacy.redact_identifiers or '[]'}, "
                f"max_literal_length={config.privacy.max_literal_length}"
            )
        if config.domain:
            config_lines.append(
                f"- Domain aliases already applied: datasets={bool(config.domain.dataset_aliases)}, "
                f"models={bool(config.domain.model_aliases)}, features={bool(config.domain.feature_aliases)}, "
                f"pipeline_name={config.domain.pipeline_name or 'default'}"
            )

    base = DEFAULT_NDEL_PROMPT.format(
        pipeline_json=pipeline_json,
        ndel_grammar=NDEL_GRAMMAR,
        config_notes="\n".join(config_lines) if config_lines else "No additional config constraints.",
    )
    prompt_parts: list[str] = [base]
    if extra_instructions:
        prompt_parts.append(extra_instructions.strip())
    return "\n\n".join(prompt_parts)


def render_pipeline_with_llm(
    pipeline: Pipeline,
    llm_generate: LLMGenerate,
    config: NdelConfig | None = None,
    extra_instructions: str | None = None,
) -> str:
    """
    Render a Pipeline to NDEL text by building an LLM prompt and delegating
    to a user-supplied LLM callback.
    """

    prompt = build_ndel_prompt(pipeline, config=config, extra_instructions=extra_instructions)
    return llm_generate(prompt)


__all__ = [
    "PRIMITIVES",
    "NDEL_GRAMMAR",
    "describe_grammar",
    "validate_ndel_text",
    "apply_privacy",
    "apply_privacy_to_payload",
    "validate_pipeline_structure",
    "render_pipeline",
    "LLMGenerate",
    "DEFAULT_NDEL_PROMPT",
    "build_ndel_prompt",
    "render_pipeline_with_llm",
]
