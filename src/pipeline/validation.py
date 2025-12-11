from __future__ import annotations

from dataclasses import dataclass
from typing import List

from src.config.core import AbstractionLevel, NdelConfig
from src.pipeline.semantic_model import Dataset, Feature, Model, Pipeline


@dataclass
class ValidationIssue:
    kind: str  # e.g., "warning", "error"
    code: str  # e.g., "UNKNOWN_DATASET_ALIAS"
    message: str


def validate_config_against_pipeline(config: NdelConfig, pipeline: Pipeline) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []

    datasets = {ds.name for ds in pipeline.datasets}
    models = {m.name for m in pipeline.models}
    features = {f.name for f in pipeline.features}

    domain = config.domain
    if domain:
        for key in domain.dataset_aliases:
            if key not in datasets:
                issues.append(
                    ValidationIssue(
                        kind="warning",
                        code="UNKNOWN_DATASET_ALIAS",
                        message=f"Dataset alias '{key}' does not match any dataset in pipeline.",
                    )
                )
        for key in domain.model_aliases:
            if key not in models:
                issues.append(
                    ValidationIssue(
                        kind="warning",
                        code="UNKNOWN_MODEL_ALIAS",
                        message=f"Model alias '{key}' does not match any model in pipeline.",
                    )
                )
        for key in domain.feature_aliases:
            if key not in features:
                issues.append(
                    ValidationIssue(
                        kind="warning",
                        code="UNKNOWN_FEATURE_ALIAS",
                        message=f"Feature alias '{key}' does not match any feature in pipeline.",
                    )
                )

    # Privacy hints
    pii_like = {"email", "ip", "phone", "address", "ssn"}
    names_union = {n.lower() for n in datasets | models | features}
    hits = pii_like & names_union
    if (config.privacy is None or not config.privacy.redact_identifiers) and hits:
        issues.append(
            ValidationIssue(
                kind="warning",
                code="MISSING_PII_REDACTION",
                message=f"Potential PII identifiers detected ({', '.join(sorted(hits))}) but redact_identifiers is empty.",
            )
        )

    # Abstraction sanity
    if config.abstraction == AbstractionLevel.HIGH:
        if len(pipeline.features) > 5 or len(pipeline.transformations) > 5:
            issues.append(
                ValidationIssue(
                    kind="warning",
                    code="ABSTRACTION_HIGH_WITH_DETAILS",
                    message="AbstractionLevel.HIGH with many features/transformations; consider MEDIUM for more detail.",
                )
            )

    return issues


__all__ = ["ValidationIssue", "validate_config_against_pipeline"]
