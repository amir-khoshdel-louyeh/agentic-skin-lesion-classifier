"""Vision-language interface backed by Ollama for lesion feature description."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from ollama import Client
from pydantic import BaseModel, Field

from src.data_utils import (
    HMNIST_RGB_CSV,
    load_hmnist_pixels,
    preprocess_image,
    reconstruct_image_from_row,
)


class StructuredClinicalDescription(BaseModel):
    """Structured lesion description suitable for downstream reasoning tools."""

    asymmetry_score: float = Field(ge=0.0, le=1.0)
    border_irregularity_score: float = Field(ge=0.0, le=1.0)
    color_variation_score: float = Field(ge=0.0, le=1.0)
    mean_rgb: list[float] = Field(description="Mean channel intensity in RGB order, 0-255 scale.")
    brightness_score: float = Field(ge=0.0, le=1.0)
    dominant_tone: str
    clinical_clues: list[str]
    visual_summary: str
    model_interpretation: str | None = None


class OllamaVisionLanguageInterface:
    """Wrapper around Ollama vision models with HMNIST ingestion helpers."""

    def __init__(
        self,
        model_name: str = "llama3.2-vision",
        host: str = "http://127.0.0.1:11434",
        timeout_seconds: int = 120,
    ) -> None:
        self.model_name = model_name
        self.client = Client(host=host, timeout=timeout_seconds)

    def ensure_model_available(self) -> bool:
        """Return True when configured model is present in local Ollama registry."""
        model_list = self.client.list()
        models = model_list.get("models", [])
        return any(self.model_name in (model.get("name") or "") for model in models)

    def validate_prerequisites(self) -> tuple[bool, list[str]]:
        """Check runtime prerequisites for calling the vision model.

        Returns (ok, messages). `ok` is True when all checks pass.
        """
        msgs: list[str] = []
        ok = True

        # Check Pillow
        try:
            import PIL  # noqa: F401
        except Exception:
            ok = False
            msgs.append("Pillow (PIL) is not installed. Install via `pip install pillow`.")

        # Check Ollama client connectivity and model availability
        try:
            model_list = self.client.list()
            models = model_list.get("models", [])
            if not any(self.model_name in (m.get("name") or "") for m in models):
                ok = False
                msgs.append(f"Ollama model '{self.model_name}' is not available locally.")
        except Exception as exc:  # noqa: BLE001
            ok = False
            msgs.append(f"Failed to contact Ollama at {self.client.host}: {exc}")

        return ok, msgs

    def ingest_hmnist_row(
        self,
        row_index: int,
        hmnist_csv: Path | str = HMNIST_RGB_CSV,
    ) -> np.ndarray:
        """Load one HMNIST row and reconstruct it into a uint8 RGB image."""
        frame = load_hmnist_pixels(hmnist_csv=hmnist_csv)
        if row_index < 0 or row_index >= len(frame):
            raise IndexError(f"row_index out of range: {row_index}")

        return reconstruct_image_from_row(frame.iloc[row_index])

    def extract_structured_features(self, image: np.ndarray) -> StructuredClinicalDescription:
        """Compute deterministic image features and build a clinical description."""
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("Expected image shape (H, W, 3).")

        img_u8 = image.astype(np.uint8)
        gray = img_u8.mean(axis=2)

        left_half = gray[:, : gray.shape[1] // 2]
        right_half = np.fliplr(gray[:, gray.shape[1] // 2 :])
        min_width = min(left_half.shape[1], right_half.shape[1])
        asymmetry = float(np.mean(np.abs(left_half[:, :min_width] - right_half[:, :min_width])) / 255.0)

        gradient_x = np.abs(np.diff(gray, axis=1)).mean()
        gradient_y = np.abs(np.diff(gray, axis=0)).mean()
        border_irregularity = float(np.clip((gradient_x + gradient_y) / (2 * 64.0), 0.0, 1.0))

        rgb_std = img_u8.reshape(-1, 3).std(axis=0)
        color_variation = float(np.clip(np.mean(rgb_std) / 128.0, 0.0, 1.0))

        mean_rgb = img_u8.reshape(-1, 3).mean(axis=0)
        brightness = float(np.clip(gray.mean() / 255.0, 0.0, 1.0))

        tone = self._derive_dominant_tone(mean_rgb)
        clues = self._derive_clinical_clues(asymmetry, border_irregularity, color_variation, brightness)

        summary = (
            f"Lesion appears {tone} with asymmetry={asymmetry:.2f}, "
            f"border_irregularity={border_irregularity:.2f}, color_variation={color_variation:.2f}, "
            f"brightness={brightness:.2f}."
        )

        return StructuredClinicalDescription(
            asymmetry_score=asymmetry,
            border_irregularity_score=border_irregularity,
            color_variation_score=color_variation,
            mean_rgb=[float(v) for v in mean_rgb],
            brightness_score=brightness,
            dominant_tone=tone,
            clinical_clues=clues,
            visual_summary=summary,
        )

    def describe_image(self, image: np.ndarray, call_model: bool = True) -> StructuredClinicalDescription:
        """Generate a structured description from a raw image array."""
        baseline = self.extract_structured_features(image)

        if not call_model:
            return baseline

        try:
            model_text = self._run_vision_prompt(image, baseline)
            baseline.model_interpretation = model_text
            return baseline
        except Exception as exc:  # noqa: BLE001
            baseline.model_interpretation = f"Model call failed: {exc}"
            return baseline

    def describe_hmnist_row(
        self,
        row_index: int,
        hmnist_csv: Path | str = HMNIST_RGB_CSV,
        call_model: bool = True,
    ) -> StructuredClinicalDescription:
        """Ingest HMNIST row, preprocess image, and return structured clinical description."""
        image = self.ingest_hmnist_row(row_index=row_index, hmnist_csv=hmnist_csv)
        # Keep model-facing data normalized while preserving original clinical feature scales.
        _ = preprocess_image(image, normalize=True)
        return self.describe_image(image=image, call_model=call_model)

    def _run_vision_prompt(
        self,
        image: np.ndarray,
        baseline: StructuredClinicalDescription,
    ) -> str:
        """Call Ollama vision model and ask for a concise clinical interpretation."""
        image_path = self._write_temp_image(image)
        prompt = self._build_prompt(baseline)

        try:
            response = self.client.chat(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [str(image_path)],
                    }
                ],
                options={"temperature": 0.1},
            )
        finally:
            image_path.unlink(missing_ok=True)

        message = response.get("message", {})
        content = message.get("content", "").strip()
        if not content:
            return "No text returned from vision model."

        parsed = self._extract_json_or_text(content)
        if isinstance(parsed, dict):
            return json.dumps(parsed, ensure_ascii=True)
        return str(parsed)

    def _build_prompt(self, baseline: StructuredClinicalDescription) -> str:
        """Compose model prompt grounded in deterministic image features."""
        baseline_json = baseline.model_dump_json(indent=2)
        return (
            "You are a dermatology vision assistant. Analyze the provided lesion image and "
            "produce a concise clinical visual interpretation.\n"
            "Use the baseline computed features as context, but verify with image evidence.\n"
            "Return JSON with keys: visual_summary, suspected_patterns, risk_flags.\n"
            f"Baseline features:\n{baseline_json}"
        )

    def _write_temp_image(self, image: np.ndarray) -> Path:
        """Persist temporary PNG image for Ollama image input."""
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("Pillow is required for vision image serialization.") from exc

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            path = Path(tmp.name)
        Image.fromarray(image.astype(np.uint8), mode="RGB").save(path)
        return path

    @staticmethod
    def _derive_dominant_tone(mean_rgb: np.ndarray) -> str:
        """Map mean RGB values into coarse lesion color descriptors."""
        r, g, b = [float(v) for v in mean_rgb]
        if r > 170 and g > 140 and b > 120:
            return "light-brown"
        if r > 130 and g > 90 and b > 70:
            return "brown"
        if r < 90 and g < 90 and b < 90:
            return "dark"
        if r > b + 20:
            return "reddish-brown"
        return "mixed"

    @staticmethod
    def _derive_clinical_clues(
        asymmetry: float,
        border_irregularity: float,
        color_variation: float,
        brightness: float,
    ) -> list[str]:
        """Translate numeric image features into human-readable clinical clues."""
        clues: list[str] = []

        clues.append("Asymmetric pattern" if asymmetry >= 0.25 else "Relatively symmetric pattern")
        clues.append("Irregular border transitions" if border_irregularity >= 0.30 else "Mostly smooth border transitions")
        clues.append("Notable color heterogeneity" if color_variation >= 0.22 else "Limited color heterogeneity")
        clues.append("Low overall brightness" if brightness < 0.45 else "Moderate to high brightness")

        return clues

    @staticmethod
    def _extract_json_or_text(content: str) -> dict[str, Any] | str:
        """Attempt to parse JSON responses while tolerating fenced outputs."""
        stripped = content.strip()

        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if len(lines) >= 3:
                stripped = "\n".join(lines[1:-1]).strip()

        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return content
