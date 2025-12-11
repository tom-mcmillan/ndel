from src.formatter import render_pipeline, build_ndel_prompt
from src.config import AbstractionLevel, DomainConfig, NdelConfig, PrivacyConfig
from src.schema import Dataset, Pipeline


def test_build_prompt_includes_config_notes():
    pipeline = Pipeline(name="demo")
    config = NdelConfig(
        abstraction=AbstractionLevel.HIGH,
        privacy=PrivacyConfig(
            hide_table_names=True,
            hide_file_paths=True,
            redact_identifiers=["email"],
            max_literal_length=16,
        ),
        domain=DomainConfig(
            dataset_aliases={"raw.users": "users_public"},
            model_aliases={},
            feature_aliases={},
            pipeline_name="friendly_name",
        ),
    )

    prompt = build_ndel_prompt(pipeline, config=config, extra_instructions="keep wording concise")

    assert "Abstraction level: high" in prompt
    assert "hide_table_names=True" in prompt
    assert "hide_file_paths=True" in prompt
    assert "redact_identifiers=['email']" in prompt
    assert "pipeline_name=friendly_name" in prompt
    assert "keep wording concise" in prompt  # extra instructions are appended


def test_render_pipeline_applies_privacy_filters():
    ds = Dataset(name="schema.table", source="/tmp/path/file.csv", description="from /tmp/path/file.csv")
    pipeline = Pipeline(name="p", datasets=[ds])
    config = NdelConfig(
        privacy=PrivacyConfig(
            hide_table_names=True,
            hide_file_paths=True,
            redact_identifiers=[],
        )
    )

    text = render_pipeline(pipeline, config=config)

    assert "<redacted-table>" in text
    assert "<redacted-path>" in text
    assert "schema.table" not in text
    assert "/tmp/path" not in text
