from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


LANGUAGES = {
    "python",
    "javascript",
    "typescript",
    "c",
    "cpp",
    "java",
    "html",
    "css",
    "sql",
    "go",
    "rust",
}
TASK_TYPES = {
    "code_generation",
    "debugging",
    "code_explanation",
    "unit_test_generation",
}
SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class Provenance:
    source_name: str
    source_uri: str
    license: str
    repository: str = ""
    revision: str = ""


@dataclass(frozen=True)
class Example:
    id: str
    task_type: str
    language: str
    prompt: str
    completion: str
    provenance: Provenance
    tests: dict[str, Any] | None = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Example:
        provenance = value["provenance"]
        return cls(
            id=str(value["id"]),
            task_type=str(value["task_type"]),
            language=str(value["language"]),
            prompt=str(value["prompt"]),
            completion=str(value["completion"]),
            provenance=Provenance(**provenance),
            tests=value.get("tests"),
            schema_version=str(value.get("schema_version", SCHEMA_VERSION)),
        )
