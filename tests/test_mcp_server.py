import os
from pathlib import Path

import pytest

from ndel import mcp_server


def test_dict_to_pipeline_roundtrip():
    pipeline_dict = {
        "name": "test_pipeline",
        "datasets": [{"name": "ds1", "source": "table1", "description": "test"}],
        "transformations": [{"name": "t1", "description": "desc", "inputs": ["ds1"], "outputs": ["ds1"], "kind": "filter"}],
        "features": [{"name": "f1", "description": "feature", "origin": "ds1"}],
        "models": [{"name": "m1", "task": "classification", "inputs": ["f1"]}],
        "metrics": [{"name": "acc", "description": "accuracy", "dataset": "validation", "higher_is_better": True}],
    }

    pipeline = mcp_server._dict_to_pipeline(pipeline_dict)
    assert pipeline.name == "test_pipeline"
    assert pipeline.datasets[0].name == "ds1"
    assert pipeline.features[0].origin == "ds1"
    assert pipeline.models[0].inputs == ["f1"]
    assert pipeline.metrics[0].higher_is_better is True


def test_build_config_env_overrides(monkeypatch):
    monkeypatch.setenv("NDEL_HIDE_TABLE_NAMES", "true")
    monkeypatch.setenv("NDEL_ABSTRACTION", "high")
    monkeypatch.setenv("NDEL_PIPELINE_NAME", "env_pipeline")
    cfg = mcp_server._build_config({})

    assert cfg.privacy.hide_table_names is True
    assert cfg.abstraction.value == "high"
    assert cfg.domain.pipeline_name == "env_pipeline"

    # Payload override wins
    cfg2 = mcp_server._build_config({"privacy": {"hide_table_names": False}, "abstraction": "medium"})
    assert cfg2.privacy.hide_table_names is False
    assert cfg2.abstraction.value == "medium"


@pytest.mark.asyncio
async def test_list_and_get_docs():
    docs = await mcp_server.list_docs()
    assert isinstance(docs, dict)
    assert "docs" in docs
    available = {item["name"] for item in docs["docs"]}
    assert "readme" in available

    content = await mcp_server.get_doc("readme")
    assert "NDEL" in content["content"]
