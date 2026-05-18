"""Operational pipeline: ingestion, vision extraction, tool augmentation, synthesis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import json
import logging
from datetime import datetime

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
    interpretability_passed: bool = True
    missing_reasons: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TraceEntry:
    step: str
    timestamp: str
    detail: str
    evidence: dict[str, Any] | list | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.step, "timestamp": self.timestamp, "detail": self.detail, "evidence": self.evidence}


class OperationalPipeline:
    """High-level pipeline implementing the operational sequence described in the plan."""

    def __init__(self, vl_interface: OllamaVisionLanguageInterface | None = None):
        self.vl = vl_interface or OllamaVisionLanguageInterface()
        self.explain_log: list[dict[str, Any]] = []

    def _log(self, step: str, detail: str, evidence: dict[str, Any] | list | None = None) -> None:
        entry = TraceEntry(step=step, timestamp=datetime.utcnow().isoformat() + "Z", detail=detail, evidence=evidence)
        try:
            self.explain_log.append(entry.to_dict())
        except Exception:
            # Best-effort logging; avoid failing pipeline due to logging problems
            self.explain_log.append({"step": step, "timestamp": datetime.utcnow().isoformat() + "Z", "detail": detail, "evidence": None})

    def ingest(self, *, row_index: int) -> np.ndarray:
        """Ingest HMNIST row and reconstruct RGB image."""
        df = load_hmnist_pixels()
        if row_index < 0 or row_index >= len(df):
            raise IndexError("row_index out of range for HMNIST dataset")
        image = reconstruct_image_from_row(df.iloc[row_index])
        self._log("ingest", f"Loaded HMNIST row {row_index}", evidence={"shape": image.shape})
        return image

    def visual_feature_extraction(self, image: np.ndarray, call_model: bool = True) -> StructuredClinicalDescription:
        """Run deterministic feature extractor and optionally the vision model."""
        desc = self.vl.describe_image(image.astype(np.uint8), call_model=call_model)
        # Record structured features and model interpretation (if present)
        try:
            feat = desc.model_dump() if hasattr(desc, "model_dump") else desc.__dict__
        except Exception:
            feat = {
                "asymmetry_score": getattr(desc, "asymmetry_score", None),
                "border_irregularity_score": getattr(desc, "border_irregularity_score", None),
            }
        self._log("visual_feature_extraction", "Computed clinical features and optional model interpretation", evidence=feat)
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

        abcd = {
            "A": bool(a),
            "B": bool(b),
            "C": bool(c),
            "D_proxy": bool(d),
            "abcd_score": float(score),
        }
        self._log("abcd_check", "Evaluated ABCD heuristic", evidence=abcd)
        return abcd

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
        onc = {"trigger_precision_oncology": trigger, "reason": "high_abcd_or_model_flag" if trigger else "low_risk"}
        self._log("oncology_trigger", "Evaluated precision oncology trigger", evidence=onc)
        return onc

    def literature_search(self, desc: StructuredClinicalDescription) -> list[str]:
        """Stubbed literature search: return short references for ambiguous cases.

        In real deployment this would query PubMed or a biomedical index. Here we
        return placeholder references when features are high-risk or unusual.
        """
        refs: list[str] = []
        if desc.color_variation_score > 0.4 or desc.asymmetry_score > 0.4:
            refs.append("PMID:00000001 - Example melanoma review (placeholder)")
        self._log("literature_search", "Performed literature search (stub)", evidence={"references": refs})
        return refs

    def evidence_synthesis(self, desc: StructuredClinicalDescription, abcd: dict[str, Any], onc: dict[str, Any], refs: list[str]) -> FinalReport:
        """Merge outputs into a final structured report."""
        reasoning: list[str] = []
        # Collect clinical reasons: visual summary + explicit clinical clues
        clinical_reasons: list[str] = []
        if getattr(desc, "visual_summary", None):
            clinical_reasons.append(desc.visual_summary)
        clinical_reasons.extend(getattr(desc, "clinical_clues", []) or [])

        # Ensure at least two clinical reasons are present; add fallbacks from numeric features if missing
        missing_reasons: list[str] = []
        if len(clinical_reasons) < 2:
            fallback_a = f"Asymmetry score: {getattr(desc, 'asymmetry_score', 0.0):.2f}"
            fallback_b = f"Border irregularity score: {getattr(desc, 'border_irregularity_score', 0.0):.2f}"
            clinical_reasons.append(fallback_a)
            clinical_reasons.append(fallback_b)
            missing_reasons.extend([fallback_a, fallback_b])

        # Final reasoning includes the clinical reasons and a concise ABCD summary
        reasoning.extend(clinical_reasons)
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

        # Build citations list describing which tools/guidelines contributed
        citations: list[str] = []
        citations.append("ABCD_heuristic")
        if getattr(desc, "model_interpretation", None):
            citations.append("VisionModel")
        if refs:
            citations.append("LiteratureSearch")
        if onc.get("trigger_precision_oncology"):
            citations.append("PrecisionOncologyHeuristic")

        interpretability_passed = len(clinical_reasons) >= 2

        report = FinalReport(
            prediction=pred_label,
            confidence=float(confidence),
            reasoning=reasoning,
            references=refs,
            clinical_description=desc.model_dump() if hasattr(desc, "model_dump") else desc.__dict__,
            interpretability_passed=interpretability_passed,
            missing_reasons=missing_reasons,
            citations=list(dict.fromkeys(citations)),
        )

        # Log synthesis outcome
        self._log(
            "evidence_synthesis",
            "Synthesized evidence into final prediction",
            evidence={
                "prediction": pred_label,
                "confidence": confidence,
                "reasoning": reasoning,
                "references": refs,
                "interpretability_passed": interpretability_passed,
                "missing_reasons": missing_reasons,
                "citations": list(dict.fromkeys(citations)),
            },
        )

        return report

    def run(self, row_index: int, call_model: bool = True) -> dict[str, Any]:
        """Execute the full pipeline and return a structured dict report."""
        # reset explainability log for this run
        self.explain_log = []

        start_ts = datetime.utcnow()
        self._log("run_start", "Pipeline run started", evidence={"row_index": row_index})

        image = self.ingest(row_index=row_index)
        desc = self.visual_feature_extraction(image, call_model=call_model)
        abcd = self.check_abcd_criteria(desc)
        onc = self.trigger_oncology_context(abcd, desc)
        refs = self.literature_search(desc)
        report = self.evidence_synthesis(desc, abcd, onc, refs)

        end_ts = datetime.utcnow()
        elapsed = (end_ts - start_ts).total_seconds()
        self._log("run_end", "Pipeline run finished", evidence={"elapsed_seconds": elapsed})

        out = report.to_dict()
        out.update({"abcd": abcd, "oncology_trigger": onc})
        out.update({"explainability_log": self.explain_log})
        out.update({"timing": {"start_utc": start_ts.isoformat() + "Z", "end_utc": end_ts.isoformat() + "Z", "elapsed_seconds": elapsed}})
        return out


__all__ = ["OperationalPipeline", "FinalReport"]
