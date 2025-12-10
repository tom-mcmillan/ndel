from ndel.config import AbstractionLevel, NdelConfig
from ndel.semantic_model import Dataset, Feature, Model, Pipeline, Transformation
from ndel.render import render_pipeline


def _sample_pipeline() -> Pipeline:
    return Pipeline(
        name="sample",
        datasets=[Dataset(name="df")],
        transformations=[
            Transformation(
                name="t1",
                description="transform",
                inputs=["df"],
                outputs=["df"],
                kind="feature_engineering",
            )
        ],
        features=[
            Feature(name="train_feat", description="feature used in model training", origin="df", data_type=None),
            Feature(name="extra_feat", description="other feature", origin="df", data_type=None),
        ],
        models=[Model(name="model", task="task")],
        metrics=[],
        description=None,
    )


def test_abstraction_high_hides_transforms_and_features() -> None:
    pipeline = _sample_pipeline()
    config = NdelConfig(abstraction=AbstractionLevel.HIGH)

    output = render_pipeline(pipeline, config=config)

    assert "datasets:" in output
    assert "models:" in output
    assert "transformations:" not in output
    assert "features:" not in output


def test_abstraction_medium_shows_transformations_and_training_features_only() -> None:
    pipeline = _sample_pipeline()
    config = NdelConfig(abstraction=AbstractionLevel.MEDIUM)

    output = render_pipeline(pipeline, config=config)

    assert "transformations:" in output
    assert "features:" in output
    assert "train_feat" in output
    assert "extra_feat" not in output


def test_abstraction_low_shows_all_details() -> None:
    pipeline = _sample_pipeline()
    config = NdelConfig(abstraction=AbstractionLevel.LOW)

    output = render_pipeline(pipeline, config=config)

    assert "transformations:" in output
    assert "features:" in output
    assert "train_feat" in output
    assert "extra_feat" in output
