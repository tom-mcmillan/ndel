from ndel.serialization import pipeline_to_dict, pipeline_to_json
from ndel.semantic_model import Dataset, Feature, Metric, Model, Pipeline, Transformation


def test_pipeline_to_dict_and_json() -> None:
    pipeline = Pipeline(
        name="demo",
        datasets=[Dataset(name="users", source="users")],
        transformations=[
            Transformation(
                name="t1",
                description="filter active",
                kind="filter",
                inputs=["users"],
                outputs=["users"],
            )
        ],
        features=[Feature(name="age", description="age feature", origin="t1", data_type="int")],
        models=[Model(name="m1", task="cls", algorithm_family="LogisticRegression", inputs=["age"], target="y", description=None, hyperparameters=None)],
        metrics=[Metric(name="accuracy", description="", dataset=None, higher_is_better=True)],
        description=None,
    )

    data = pipeline_to_dict(pipeline)

    for key in ["name", "datasets", "transformations", "features", "models", "metrics"]:
        assert key in data

    # Ensure data is JSON-serializable
    json_str = pipeline_to_json(pipeline)
    assert isinstance(json_str, str)
    assert json_str
