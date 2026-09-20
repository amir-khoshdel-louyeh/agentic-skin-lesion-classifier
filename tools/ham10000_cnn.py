"""Uniform CLI wrapper: ready-made HAM10000 CNN triage tier (Phase 2).

Contract: identical JSON output shape across diagnostic tools.
VRAM discipline: load -> infer -> release in finally (del + empty_cache),
with before/after allocation check against the 2 GB tool budget.
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "models" / "derm-cnn"

ISIC_TO_CLASS = {
    "akiec": "Actinic keratoses",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis",
    "df": "Dermatofibroma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
    "mel": "Melanoma",
}

VRAM_BUDGET_BYTES = 2 * 1024**3

# Calibrated by temperature scaling: T=4.7 fitted by NLL grid search on a
# 105-image ISIC2019_full split (15/class, SCC excluded, 2026-09-29).
# Shared weights/arch with tools/skin_lesion_fast.py, hence the same T.
TEMPERATURE = 4.7


def fail(message: str) -> int:
    print(json.dumps({"status": "error", "message": message}))
    return 1


def load_arch():
    spec = importlib.util.spec_from_file_location(
        "derm_cnn_model", WEIGHTS_DIR / "model.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def preprocess(image_path: str, torch, device) -> object:
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("RGB").resize((28, 28))
    pixels = list(image.get_flattened_data())
    tensor = torch.tensor(pixels, dtype=torch.float32).reshape(1, 28, 28, 3)
    tensor = tensor.permute(0, 3, 1, 2) / 255.0
    return tensor.to(device)


def main() -> int:
    parser = argparse.ArgumentParser(description="HAM10000 CNN triage tier.")
    parser.add_argument("--image", dest="image_path", required=True)
    parser.add_argument("--metadata", dest="metadata", required=False)
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        return fail(f"Image not found: {args.image_path}")

    metadata: dict = {}
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError:
            return fail("Invalid metadata JSON.")

    try:
        import torch
    except ImportError:
        return fail("torch is not installed.")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    vram_before = (
        torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    )

    labels = json.loads((WEIGHTS_DIR / "labels.json").read_text())
    model = None
    try:
        arch = load_arch()
        model, _ = arch.load_model(
            str(WEIGHTS_DIR / "model.pth"), device.type
        )
        inputs = preprocess(args.image_path, torch, device)
        with torch.no_grad():
            probs = torch.softmax(model(inputs) / TEMPERATURE, dim=1)[0]
            confidence, class_idx = torch.max(probs, dim=0)
        idx = int(class_idx.item())
        code = labels[str(idx)]
        result = {
            "status": "success",
            "tool": "ham10000-cnn",
            "model_tier": "tier1_fast",
            "model_executed": "derm_cnn_ham10000",
            "predicted_class_index": idx,
            "disease_name": ISIC_TO_CLASS[code],
            "confidence_score": round(float(confidence.item()), 4),
            "metadata": metadata,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI must report, not crash
        return fail(f"Execution failed: {exc}")
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
            vram_after = torch.cuda.memory_allocated(device)
            if vram_after - vram_before > VRAM_BUDGET_BYTES:
                print(
                    json.dumps(
                        {
                            "status": "warning",
                            "message": "VRAM budget exceeded after release.",
                        }
                    ),
                    file=sys.stderr,
                )


if __name__ == "__main__":
    raise SystemExit(main())
