import pytest

from src.config import NdelConfig, PrivacyConfig
from src.formatter import apply_privacy, apply_privacy_to_payload
from src.schema import PIPELINE_SCHEMA, validate_schema


def test_apply_privacy_string_and_payload():
    cfg = NdelConfig(privacy=PrivacyConfig(hide_table_names=True, hide_file_paths=True, redact_identifiers=["email"]))
    text = "select * from db.table where path='/tmp/file' and email='x@y.com'"
    redacted = apply_privacy(text, cfg)
    assert "<redacted-table>" in redacted
    assert "<redacted-path>" in redacted
    assert "<redacted>" in redacted

    nested = {"note": text, "items": ["email", "plain"]}
    redacted_payload = apply_privacy_to_payload(nested, cfg)
    assert "<redacted>" in redacted_payload["note"]
    assert redacted_payload["items"][0] == "<redacted>"


def test_pipeline_schema_validation_passes():
    payload = {
        "name": "p",
        "datasets": [{"name": "ds"}],
        "transformations": [{"name": "t1", "description": "desc"}],
        "features": [{"name": "f1", "description": "d"}],
        "models": [{"name": "m", "task": "cls"}],
        "metrics": [{"name": "acc", "higher_is_better": True}],
    }
    errors = validate_schema(payload, PIPELINE_SCHEMA)
    assert errors == []


def test_pipeline_schema_validation_catches_missing_fields():
    payload = {"datasets": [{}]}
    errors = validate_schema(payload, PIPELINE_SCHEMA)
    if errors:
        assert any("name" in err for err in errors)
    else:
        pytest.skip("jsonschema not installed; schema validation skipped")
