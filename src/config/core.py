from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class AbstractionLevel(Enum):
    """Controls how detailed NDEL descriptions are."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class PrivacyConfig:
    """Settings for preventing sensitive details from leaking into NDEL text."""

    hide_table_names: bool = False
    hide_file_paths: bool = False
    redact_identifiers: List[str] = field(default_factory=list)
    max_literal_length: int = 200


@dataclass
class DomainConfig:
    """Mappings from code-level names to human-friendly domain aliases."""

    dataset_aliases: Dict[str, str] = field(default_factory=dict)
    model_aliases: Dict[str, str] = field(default_factory=dict)
    feature_aliases: Dict[str, str] = field(default_factory=dict)
    pipeline_name: Optional[str] = None


@dataclass
class NdelConfig:
    """Bundle of configuration used when analyzing and rendering NDEL text."""

    privacy: Optional[PrivacyConfig] = None
    domain: Optional[DomainConfig] = None
    abstraction: AbstractionLevel = AbstractionLevel.MEDIUM


__all__ = [
    "AbstractionLevel",
    "PrivacyConfig",
    "DomainConfig",
    "NdelConfig",
]
