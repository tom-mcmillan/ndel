from __future__ import annotations

import re
from typing import Any, Dict, List


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
        warnings.append("first line should start with 'pipeline "'")

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


__all__ = ["PRIMITIVES", "NDEL_GRAMMAR", "describe_grammar", "validate_ndel_text"]
