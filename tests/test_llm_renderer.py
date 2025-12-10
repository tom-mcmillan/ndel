from ndel.llm_renderer import render_pipeline_with_llm
from ndel.api import describe_python_source_with_llm
from ndel.semantic_model import Dataset, Pipeline, Transformation


def stub_llm(prompt: str) -> str:
    assert "pipeline" in prompt.lower()
    return "pipeline \"stub\":\n  # LLM-generated NDEL goes here\n"


def test_render_pipeline_with_llm_stub() -> None:
    pipeline = Pipeline(
        name="demo",
        datasets=[Dataset(name="users")],
        transformations=[Transformation(name="t1", description="", kind="filter", inputs=["users"], outputs=["users"])],
        features=[],
        models=[],
        metrics=[],
        description=None,
    )

    out = render_pipeline_with_llm(pipeline, llm_generate=stub_llm)
    assert "pipeline \"stub\"" in out


def test_describe_python_source_with_llm_stub() -> None:
    source = """
import pandas as pd
df = pd.read_csv('x.csv')
"""

    out = describe_python_source_with_llm(source, llm_generate=stub_llm)
    assert "LLM-generated" in out
