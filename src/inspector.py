from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import yaml

from src.analyzers import analyze_python_source, analyze_sql_source
from src.model import Dataset, Pipeline


def _is_code_file(path: Path) -> bool:
    return path.suffix.lower() in {".py", ".sql"}


def _iter_code_files(root: Path):
    for dirpath, _, filenames in os.walk(root):
        dir_path = Path(dirpath)
        if any(part.startswith(".") for part in dir_path.parts if part != root.name):
            continue
        for name in filenames:
            path = dir_path / name
            if _is_code_file(path):
                yield path


def _collect_from_python(path: Path) -> Pipeline:
    text = path.read_text(encoding="utf-8")
    return analyze_python_source(text)


def _collect_from_sql(path: Path) -> Pipeline:
    text = path.read_text(encoding="utf-8")
    return analyze_sql_source(text)


def _merge_names(p: Pipeline) -> Dict[str, set[str]]:
    return {
        "datasets": {ds.name for ds in p.datasets},
        "models": {m.name for m in p.models},
        "features": {f.name for f in p.features},
    }


def inspect_project(root: str = ".") -> Dict[str, Any]:
    root_path = Path(root).resolve()
    names = {"datasets": set(), "models": set(), "features": set()}
    for path in _iter_code_files(root_path):
        if path.suffix.lower() == ".py":
            p = _collect_from_python(path)
        else:
            p = _collect_from_sql(path)
        merged = _merge_names(p)
        for key in names:
            names[key].update(merged[key])
    pii_candidates = {n for n in names["features"] | names["datasets"] | names["models"] if any(k in n.lower() for k in ["email", "ip", "phone", "ssn", "address"])}
    return {
        "domain": {
            "dataset_aliases": {},
            "model_aliases": {},
            "feature_aliases": {},
        },
        "privacy": {
            "redact_identifiers": sorted(pii_candidates),
        },
        "abstraction": "medium",
        "observed": {
            "datasets": sorted(names["datasets"]),
            "models": sorted(names["models"]),
            "features": sorted(names["features"]),
        },
    }


def write_ndel_config(output_path: str = ".ndel.yml", root: str = ".") -> Path:
    config = inspect_project(root=root)
    out_path = Path(output_path).resolve()
    out_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return out_path


__all__ = ["inspect_project", "write_ndel_config"]
