from src.formatter import validate_pipeline_structure
from src.schema import Dataset, Feature, Pipeline, Transformation


def test_validate_pipeline_structure_detects_duplicates_and_unknowns():
    pipeline = Pipeline(
        name="p",
        datasets=[Dataset(name="ds"), Dataset(name="ds")],
        transformations=[
            Transformation(
                name="t1",
                description="uses missing",
                inputs=["missing_input"],
                outputs=["out1"],
                kind="filter",
            )
        ],
        features=[Feature(name="feat", description="d", origin="missing_output")],
        models=[],
        metrics=[],
    )

    issues = validate_pipeline_structure(pipeline)
    codes = {issue.code for issue in issues}
    assert "DUPLICATE_DATASET" in codes
    assert "UNKNOWN_INPUT" in codes
    assert "UNKNOWN_FEATURE_ORIGIN" in codes
