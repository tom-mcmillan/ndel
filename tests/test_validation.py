from ndel.config import DomainConfig, NdelConfig
from ndel.semantic_model import Dataset, Pipeline
from ndel.validation import validate_config_against_pipeline


def test_validation_unknown_aliases_and_pii_hint() -> None:
    pipeline = Pipeline(
        name="p",
        datasets=[Dataset(name="users", source="users")],
        transformations=[],
        features=[],
        models=[],
        metrics=[],
    )

    config = NdelConfig(
        domain=DomainConfig(dataset_aliases={"missing": "alias"}),
        privacy=None,
    )

    issues = validate_config_against_pipeline(config, pipeline)

    codes = {i.code for i in issues}
    assert "UNKNOWN_DATASET_ALIAS" in codes


def test_validation_pii_warning() -> None:
    pipeline = Pipeline(
        name="p",
        datasets=[Dataset(name="users", source="users")],
        transformations=[],
        features=[],
        models=[],
        metrics=[],
    )

    config = NdelConfig(privacy=None)

    issues = validate_config_against_pipeline(config, pipeline)

    assert issues == []

    pii_pipeline = Pipeline(
        name="p",
        datasets=[Dataset(name="users", source="users")],
        transformations=[],
        features=[],
        models=[],
        metrics=[],
    )
    pii_pipeline.datasets.append(Dataset(name="email", source="email"))

    issues = validate_config_against_pipeline(config, pii_pipeline)
    assert any(i.code == "MISSING_PII_REDACTION" for i in issues)
