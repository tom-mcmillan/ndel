from ndel.diff import diff_pipelines
from ndel.api import describe_pipeline_diff
from ndel.semantic_model import Dataset, Feature, Model, Pipeline


def _pipeline_with(name: str, datasets=None, features=None, models=None) -> Pipeline:
    return Pipeline(
        name=name,
        datasets=datasets or [],
        transformations=[],
        features=features or [],
        models=models or [],
        metrics=[],
        description=None,
    )


def test_diff_pipelines_added_removed() -> None:
    old = _pipeline_with(
        "old",
        datasets=[Dataset(name="ds1")],
        models=[Model(name="m1", task="task")],
    )
    new = _pipeline_with(
        "new",
        datasets=[Dataset(name="ds1"), Dataset(name="ds2")],
        models=[Model(name="m1", task="task"), Model(name="m2", task="task")],
        features=[Feature(name="f1", description="", origin=None, data_type=None)],
    )

    d = diff_pipelines(old, new)

    assert d.datasets_added == ["ds2"]
    assert d.models_added == ["m2"]
    assert d.features_added == ["f1"]
    assert d.datasets_removed == []


def test_describe_pipeline_diff_contains_names() -> None:
    old = _pipeline_with("old", datasets=[Dataset(name="ds1")])
    new = _pipeline_with("new", datasets=[Dataset(name="ds1"), Dataset(name="ds2")])

    text = describe_pipeline_diff(old, new)

    assert "ds2" in text
