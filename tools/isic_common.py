"""Shared inference path for the locally fine-tuned ISIC2019 models.

One implementation backs `tools/isic_efficientnet_b4.py`,
`tools/isic_resnet50.py` and `tools/isic_vit.py` so the three wrappers
cannot drift apart. Uniform JSON contract with the other diagnostic
tools; temperature comes from `models/<dir>/calibration.json` when
present (written by tools/calibrate.py --fit-temperature), otherwise
T=1.0 is used and the report shows the uncalibrated numbers.

VRAM discipline: load -> infer -> release in finally (del + empty_cache)
with a before/after budget check, same as the existing diagnostic tools.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import models, transforms

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from control.confidence import env_entropy_threshold  # noqa: E402
from control.confidence import evaluate as entropy_gate  # noqa: E402
from control.schemas import HAM10000_CLASSES  # noqa: E402
from control.vram import check as vram_check  # noqa: E402

CLASSES = list(HAM10000_CLASSES)

ARCHS = {
    "efficientnet_b4": models.efficientnet_b4,
    "resnet50": models.resnet50,
    "vit_b_16": models.vit_b_16,
}


def fail(message: str) -> int:
    print(json.dumps({"status": "error", "message": message}))
    return 1


def build_arch(arch: str, num_classes: int = len(CLASSES)) -> torch.nn.Module:
    """Same head shape tools/train_isic.py saves (num_classes=7)."""
    if arch not in ARCHS:
        raise ValueError(f"Unknown arch: {arch}")
    return ARCHS[arch](weights=None, num_classes=num_classes)


def calibration_for(model_dir: Path) -> tuple[float, float, str]:
    """Fitted (temperature, melanoma logit bias) from calibration.json.

    The bias is a class-prior adjustment fitted for melanoma sensitivity;
    both values default to identity when no calibration file exists.
    """
    path = model_dir / "calibration.json"
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            temperature = float(payload.get("temperature", 1.0))
            bias = float(payload.get("melanoma_logit_bias", 0.0))
            if temperature > 0:
                return temperature, bias, "calibration.json"
        except (ValueError, TypeError, OSError):
            pass
    return 1.0, 0.0, "default"


def probabilities(model_dir: Path, image_path: str,
                  device: torch.device | None = None,
                  temperature: float | None = None,
                  melanoma_bias: float | None = None) -> dict[str, float]:
    """Class distribution in-process (build scripts: temperature fits).

    Same checkpoint, transform, temperature and melanoma bias as the CLI
    path. Pass temperature=1.0 / melanoma_bias=0.0 to get the raw scan
    used for fitting. Used by tools/fit_temperature.py so fitting 210
    cases does not pay a fresh interpreter + weight load per image.
    """
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    if temperature is None or melanoma_bias is None:
        fitted_t, fitted_bias, _ = calibration_for(model_dir)
        temperature = fitted_t if temperature is None else temperature
        melanoma_bias = fitted_bias if melanoma_bias is None else melanoma_bias
    checkpoint = torch.load(str(model_dir / "best.pt"),
                            map_location=device, weights_only=False)
    classes = list(checkpoint.get("classes", CLASSES))
    model = build_arch(checkpoint.get("arch", ""), len(classes))
    try:
        model.load_state_dict(checkpoint["model"])
        model.to(device).eval()
        image = ImageOps.exif_transpose(
            Image.open(image_path)).convert("RGB")
        inputs = eval_transform(int(checkpoint.get("size", 224))) (
            image).unsqueeze(0).to(device)  # noqa: E211 - shared helper
        with torch.no_grad():
            logits = model(inputs)
            logits[0, classes.index("Melanoma")] += melanoma_bias
            probs = torch.softmax(logits / temperature, dim=1)[0]
        return {classes[i]: float(probs[i]) for i in range(len(classes))}
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def eval_transform(size: int):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])


def main(tool_name: str, model_dir_name: str, tier: str,
         vram_gb: float, description: str) -> int:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--image", dest="image_path", required=True)
    parser.add_argument("--metadata", dest="metadata", required=False)
    # Calibration-only overrides: reproduce the uncalibrated scan used to
    # fit T/bias without editing the wrapper or the saved checkpoint.
    parser.add_argument("--temperature", type=float, default=None,
                        help=argparse.SUPPRESS)
    parser.add_argument("--melanoma-bias", type=float, default=None,
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        return fail(f"Image not found: {args.image_path}")
    metadata: dict = {}
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError:
            return fail("Invalid metadata JSON.")

    model_dir = BASE_DIR / "models" / model_dir_name
    if not (model_dir / "best.pt").exists():
        return fail(f"Weights missing at: {model_dir / 'best.pt'}")

    ok, reason = vram_check(vram_gb)
    if not ok:
        return fail(f"VRAM guard: {reason}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vram_before = (torch.cuda.memory_allocated(device)
                   if device.type == "cuda" else 0)
    model = None
    try:
        checkpoint = torch.load(str(model_dir / "best.pt"),
                                map_location=device, weights_only=False)
        arch = checkpoint.get("arch", model_dir_name.replace("isic-", "").replace("-", "_"))
        size = int(checkpoint.get("size", 224))
        model = build_arch(arch, len(checkpoint.get("classes", CLASSES)))
        model.load_state_dict(checkpoint["model"])
        model.to(device).eval()

        temperature, melanoma_bias, temp_source = calibration_for(model_dir)
        if args.temperature is not None:
            temperature, temp_source = args.temperature, "override"
        if args.melanoma_bias is not None:
            melanoma_bias = args.melanoma_bias

        image = ImageOps.exif_transpose(
            Image.open(args.image_path)).convert("RGB")
        inputs = eval_transform(size)(image).unsqueeze(0).to(device)
        classes = list(checkpoint.get("classes", CLASSES))
        with torch.no_grad():
            logits = model(inputs)
            logits[0, classes.index("Melanoma")] += melanoma_bias
            probs = torch.softmax(logits / temperature, dim=1)[0]

        dist = {classes[i]: float(probs[i]) for i in range(len(classes))}
        gate = entropy_gate(dist, entropy_threshold=env_entropy_threshold())
        idx = classes.index(gate["top_class"])
        print(json.dumps({
            "status": "success",
            "tool": tool_name,
            "model_tier": tier,
            "model_executed": f"isic2019_finetune_{arch}",
            "predicted_class_index": idx,
            "disease_name": gate["top_class"],
            "confidence_score": gate["confidence"],
            "class_probabilities": {k: round(v, 6)
                                    for k, v in sorted(dist.items())},
            "entropy": gate["entropy"],
            "entropy_threshold": env_entropy_threshold(),
            "temperature": temperature,
            "melanoma_logit_bias": melanoma_bias,
            "temperature_source": temp_source,
            "uncertainty_flags": list(gate["uncertainty_flags"]),
            "metadata": metadata,
        }, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI must report, not crash
        return fail(f"Execution failed: {exc}")
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
            budget = int(vram_gb * 1024**3)
            if torch.cuda.memory_allocated(device) - vram_before > budget:
                print(json.dumps({"status": "warning",
                                  "message": "VRAM budget exceeded."}),
                      file=sys.stderr)
