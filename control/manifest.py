"""Manifest v2 loader and validator (proposal.tmp Section 5).

Each tool declares: name, command (repo-relative), category, requires,
produces, modality, calibrated, vram_budget_gb, tier. Future categories
(e.g. preprocessing) plug in with zero code changes.
"""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError


class ToolEntry(BaseModel, frozen=True):
    name: str = Field(min_length=1)
    command: str = Field(min_length=1)
    category: str = Field(min_length=1)
    requires: list[str] = Field(default_factory=list)
    produces: list[str] = Field(default_factory=list)
    modality: str = Field(min_length=1)
    calibrated: bool = False
    vram_budget_gb: float = Field(default=2.0, gt=0)
    tier: str = ""


class Manifest(BaseModel, frozen=True):
    version: int = Field(default=2, ge=2)
    tools: list[ToolEntry]

    def by_name(self, name: str) -> ToolEntry:
        for tool in self.tools:
            if tool.name == name:
                return tool
        raise KeyError(f"Unknown tool: {name}")

    def by_category(self, category: str) -> list[ToolEntry]:
        return [t for t in self.tools if t.category == category]

    def capable_of(self, need: str, have: list[str]) -> list[ToolEntry]:
        """Tools producing `need` whose requirements are subset of `have`."""
        return [
            t for t in self.tools
            if need in t.produces and set(t.requires) <= set(have)
        ]


def load_manifest(path: Path) -> Manifest:
    payload: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        return Manifest(**payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid manifest {path}: {exc}") from exc
