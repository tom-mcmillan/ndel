from ndel.render import render_pipeline
from ndel.semantic_model import Dataset, Pipeline


def test_render_pipeline_smoke() -> None:
    pipeline = Pipeline(
        name="demo_pipeline",
        description="Demo pipeline for testing",
        datasets=[Dataset(name="users", description="user table")],
    )

    output = render_pipeline(pipeline)

    assert output
    assert "pipeline \"demo_pipeline\"" in output
    assert "users" in output
