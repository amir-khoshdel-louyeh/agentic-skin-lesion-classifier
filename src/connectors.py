"""Integration point stubs for external systems (EHR / FHIR / literature APIs).

These are lightweight placeholders describing the intended interfaces. Replace
the internals with concrete clients when integrating real systems.
"""

from __future__ import annotations

from typing import Any, Mapping


class FHIRConnector:
    """Stub interface for a FHIR/EHR connector.

    Methods should be implemented to fetch and return patient context, prior
    diagnoses, or exam metadata for improved clinical reasoning.
    """

    def __init__(self, base_url: str, auth: Any | None = None) -> None:
        self.base_url = base_url
        self.auth = auth

    def get_patient_context(self, patient_id: str) -> Mapping[str, Any] | None:
        # TODO: implement FHIR queries (e.g., Observation, Condition, Encounter)
        return None

    def submit_report(self, patient_id: str, report: Mapping[str, Any]) -> bool:
        # TODO: implement submitting diagnostic reports to EHR systems
        return False


class LiteratureSearchConnector:
    """Stub for a biomedical literature search client (PubMed, Europe PMC, etc.)."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key

    def search(self, query: str, max_results: int = 5) -> list[Mapping[str, str]]:
        # TODO: implement real literature queries
        return []


__all__ = ["FHIRConnector", "LiteratureSearchConnector"]
