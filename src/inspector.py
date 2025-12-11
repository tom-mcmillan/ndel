from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from src.analyzers import analyze_python_source, analyze_sql_source
from src.model import Dataset, Pipeline


CODE_SUFFIXES = {".py", ".sql"}


def _is_code_file(path: Path) -> bool:
    return path.suffix.lower() in CODE_SUFFIXES


def _iter_code_files(root: Path) -> Iterable[Path]:
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


def _sources_from_pipeline(p: Pipeline) -> List[str]:
    return [ds.source for ds in p.datasets if ds.source]


def inspect_project(root: str = ".", db_catalog: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Inspect codebase and optional DB catalog to suggest ndel config.

    Returns a dict suitable for writing to .ndel.yml. Existing user-edited
    fields (aliases/privacy) should be preserved by the composer; this function
    only produces fresh observations/suggestions.
    """

    root_path = Path(root).resolve()
    names = {"datasets": set(), "models": set(), "features": set()}
    sources: set[str] = set()
    seen_in: dict[str, set[str]] = {"datasets": set(), "models": set(), "features": set()}

    for path in _iter_code_files(root_path):
        pipeline = _collect_from_python(path) if path.suffix.lower() == ".py" else _collect_from_sql(path)
        merged = _merge_names(pipeline)
        for key in names:
            names[key].update(merged[key])
            for item in merged[key]:
                seen_in[key].add(f"{item} @ {path.relative_to(root_path)}")
        sources.update(_sources_from_pipeline(pipeline))

    if db_catalog:
        datasets = db_catalog.get("datasets") or []
        features = db_catalog.get("features") or []
        names["datasets"].update(datasets if isinstance(datasets, list) else [])
        names["features"].update(features if isinstance(features, list) else [])

    pii_keys = ["email", "ip", "phone", "ssn", "address"]
    pii_candidates = {
        n
        for n in names["features"] | names["datasets"] | names["models"]
        if any(k in n.lower() for k in pii_keys)
    }

    obs = {
        "datasets": sorted(names["datasets"]),
        "models": sorted(names["models"]),
        "features": sorted(names["features"]),
        "sources": sorted(sources),
        "locations": {k: sorted(v) for k, v in seen_in.items() if v},
        "transforms": {},
    }

    suggestion_abstraction = "high" if len(names["features"]) > 20 or len(names["datasets"]) > 10 else "medium"

    return {
        "domain": {
            "dataset_aliases": {},
            "model_aliases": {},
            "feature_aliases": {},
            "pipeline_name": None,
        },
        "privacy": {
            "redact_identifiers": sorted(pii_candidates),
            "hide_file_paths": bool(sources),
        },
        "abstraction": suggestion_abstraction,
        "observed": obs,
    }


def _merge_existing(existing: Dict[str, Any], fresh: Dict[str, Any]) -> Dict[str, Any]:
    """Preserve user edits in domain/privacy; refresh observed/suggestions."""

    merged = fresh
    if not isinstance(existing, dict):
        return merged

    # Preserve domain aliases and pipeline_name if user set them
    for key in ["domain", "privacy", "abstraction"]:
        if key in existing:
            if key == "privacy" and isinstance(existing[key], dict) and isinstance(fresh.get("privacy"), dict):
                merged["privacy"] = {**fresh["privacy"], **existing["privacy"]}
            elif key == "domain" and isinstance(existing[key], dict) and isinstance(fresh.get("domain"), dict):
                merged["domain"] = {**fresh["domain"], **existing["domain"]}
            else:
                merged[key] = existing[key]
    return merged


def write_ndel_config(output_path: str = ".ndel.yml", root: str = ".", db_catalog: Dict[str, Any] | None = None) -> Path:
    fresh = inspect_project(root=root, db_catalog=db_catalog)
    out_path = Path(output_path).resolve()
    existing: Dict[str, Any] = {}
    if out_path.exists():
        existing = yaml.safe_load(out_path.read_text(encoding="utf-8")) or {}
    merged = _merge_existing(existing, fresh)
    out_path.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")
    return out_path


__all__ = ["inspect_project", "write_ndel_config"]
