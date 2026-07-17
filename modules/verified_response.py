"""Typed payload passed from verified data access to deterministic wording."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from modules.semantic_dictionary import FIELDS


@dataclass(frozen=True)
class VerifiedResponsePayload:
    intent: str
    scope: str
    project_code: str | None
    requested_fields: tuple[str, ...]
    verified_values: dict[str, Any]
    verified_sources: dict[str, str]
    confidence: float = 1.0
    requires_clarification: bool = False
    clarification_options: tuple[dict[str, str], ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def project_field_payload(project: dict[str, Any], fields: list[str] | tuple[str, ...]) -> VerifiedResponsePayload:
    canonical = tuple(name for name in fields if name in FIELDS)
    return VerifiedResponsePayload(
        intent="project_field_lookup",
        scope="project",
        project_code=project.get("project_code"),
        requested_fields=canonical,
        verified_values={name: project.get(name) for name in canonical},
        verified_sources={name: FIELDS[name].source_column for name in canonical},
    )
