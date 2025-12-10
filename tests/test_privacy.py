from ndel import Pipeline, Dataset, Model
from ndel.config import NdelConfig, PrivacyConfig
from ndel.render import render_pipeline


def test_hide_table_names() -> None:
    pipeline = Pipeline(
        name="pipeline",
        datasets=[Dataset(name="analytics.users")],
    )
    config = NdelConfig(privacy=PrivacyConfig(hide_table_names=True))

    output = render_pipeline(pipeline, config=config)

    assert "<redacted-table>" in output
    assert "analytics.users" not in output


def test_hide_file_paths() -> None:
    pipeline = Pipeline(
        name="pipeline",
        datasets=[Dataset(name="df", description="/mnt/data/file.csv")],
    )
    config = NdelConfig(privacy=PrivacyConfig(hide_file_paths=True))

    output = render_pipeline(pipeline, config=config)

    assert "<redacted-path>" in output
    assert "/mnt/data/file.csv" not in output


def test_redact_identifiers() -> None:
    pipeline = Pipeline(
        name="pipeline",
        datasets=[Dataset(name="df", description="contains email field")],
        models=[Model(name="model", task="task")],
    )
    config = NdelConfig(privacy=PrivacyConfig(redact_identifiers=["email"]))

    output = render_pipeline(pipeline, config=config)

    assert "<redacted>" in output
    assert "email" not in output.lower()


def test_truncate_long_literals() -> None:
    long_text = "a" * 50
    pipeline = Pipeline(
        name="pipeline",
        datasets=[Dataset(name="df", description=long_text)],
    )
    config = NdelConfig(privacy=PrivacyConfig(max_literal_length=10))

    output = render_pipeline(pipeline, config=config)

    assert "aaaaaaaaaa..." in output
    assert long_text not in output
