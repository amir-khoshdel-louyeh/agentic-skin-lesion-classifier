"""Operational pipeline: ingestion, vision extraction, tool augmentation, synthesis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import json
import logging

import numpy as np

from models.vision_language import OllamaVisionLanguageInterface, StructuredClinicalDescription
from src.data_utils import load_hmnist_pixels, reconstruct_image_from_row


logger = logging.getLogger(__name__)


@dataclass
class FinalReport:
    prediction: str
    confidence: float
    reasoning: list[str]
    references: list[str]
    clinical_description: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OperationalPipeline:
    """High-level pipeline implementing the operational sequence described in the plan."""

    def __init__(self, vl_interface: OllamaVisionLanguageInterface | None = None):
        self.vl = vl_interface or OllamaVisionLanguageInterface()

    def ingest(self, *, row_index: int) -> np.ndarray:
        """Ingest HMNIST row and reconstruct RGB image."""
        df = load_hmnist_pixels()
        if row_index < 0 or row_index >= len(df):
            raise IndexError("row_index out of range for HMNIST dataset")
        image = reconstruct_image_from_row(df.iloc[row_index])
        return image

    def visual_feature_extraction(self, image: np.ndarray, call_model: bool = True) -> StructuredClinicalDescription:
        """Run deterministic feature extractor and optionally the vision model."""
        desc = self.vl.describe_image(image.astype(np.uint8), call_model=call_model)
        return desc

    def check_abcd_criteria(self, desc: StructuredClinicalDescription) -> dict[str, Any]:
        """Simple ABCD heuristic checks based on structured features.

        Diameter is not available from HMNIST; this function checks A, B, C, and uses
        brightness as a proxy for diameter-related contrast when possible.
        """
        a = desc.asymmetry_score >= 0.25
        b = desc.border_irregularity_score >= 0.3
        c = desc.color_variation_score >= 0.22
        d = desc.brightness_score < 0.45

        score = sum(bool(x) for x in (a, b, c, d)) / 4.0

        return {
            "A": bool(a),
            "B": bool(b),
            "C": bool(c),
            "D_proxy": bool(d),
            "abcd_score": float(score),
        }

    def trigger_oncology_context(self, abcd: dict[str, Any], desc: StructuredClinicalDescription) -> dict[str, Any]:
        """Decide whether to trigger precision oncology context.

        Uses a conservative heuristic: high ABCD score or explicit risk flags from vision model.
        """
        abcd_score = abcd.get("abcd_score", 0.0)
        risk = abcd_score >= 0.5

        # If vision model returned JSON with risk_flags, try to inspect it.
        model_risk = False
        if desc.model_interpretation:
            try:
                parsed = json.loads(desc.model_interpretation)
                flags = parsed.get("risk_flags") if isinstance(parsed, dict) else None
                if flags:
                    model_risk = True
            except Exception:
                model_risk = False

        trigger = bool(risk or model_risk)
        return {"trigger_precision_oncology": trigger, "reason": "high_abcd_or_model_flag" if trigger else "low_risk"}

    def literature_search(self, desc: StructuredClinicalDescription) -> list[str]:
        """Stubbed literature search: return short references for ambiguous cases.

        In real deployment this would query PubMed or a biomedical index. Here we
        return placeholder references when features are high-risk or unusual.
        """
        refs: list[str] = []
        if desc.color_variation_score > 0.4 or desc.asymmetry_score > 0.4:
            refs.append("PMID:00000001 - Example melanoma review (placeholder)")
        return refs

    def evidence_synthesis(self, desc: StructuredClinicalDescription, abcd: dict[str, Any], onc: dict[str, Any], refs: list[str]) -> FinalReport:
        """Merge outputs into a final structured report."""
        reasoning: list[str] = []
        reasoning.append(desc.visual_summary)
        reasoning.extend(desc.clinical_clues)
        reasoning.append(f"ABCD summary: {abcd}")
        if onc.get("trigger_precision_oncology"):
            reasoning.append("Precision oncology context recommended based on risk signals.")

        # Simple prediction heuristic mapping to HMNIST label names (placeholder)
        # This is a conservative mapping: if many risk signals, predict 'melanoma-likely'
        pred_label = "benign-uncertain"
        confidence = 0.5
        if abcd.get("abcd_score", 0) >= 0.75:
            pred_label = "melanoma-likely"
            confidence = 0.85
        elif abcd.get("abcd_score", 0) >= 0.5:
            pred_label = "suspicious"
            confidence = 0.65
        else:
            pred_label = "likely-benign"
            confidence = 0.75

        report = FinalReport(
            prediction=pred_label,
            confidence=float(confidence),
            reasoning=reasoning,
            references=refs,
            clinical_description=desc.model_dump() if hasattr(desc, "model_dump") else desc.__dict__,
        )

        return report

    def run(self, row_index: int, call_model: bool = True) -> dict[str, Any]:
        """Execute the full pipeline and return a structured dict report."""
        image = self.ingest(row_index=row_index)
        desc = self.visual_feature_extraction(image, call_model=call_model)
        abcd = self.check_abcd_criteria(desc)
        onc = self.trigger_oncology_context(abcd, desc)
        refs = self.literature_search(desc)
        report = self.evidence_synthesis(desc, abcd, onc, refs)

        out = report.to_dict()
        out.update({"abcd": abcd, "oncology_trigger": onc})
        return out


__all__ = ["OperationalPipeline", "FinalReport"]
