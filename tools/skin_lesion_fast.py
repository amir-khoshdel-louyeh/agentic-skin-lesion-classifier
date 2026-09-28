"""Fast-tier classifier: HAM10000 SkinCNN triage (models/derm-cnn).

Uses the vendored lightweight CNN (model.pth + model.py + labels.json,
28x28 input) — no download needed. Uniform JSON contract.
VRAM discipline: load -> infer -> release in finally.
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
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

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

FT_WEIGHTS = BASE_DIR / "models" / "derm-cnn-ft" / "skin_ft.pt"


def resolve_weights() -> tuple[Path, str, float]:
    """Prefer fine-tuned weights (held-out acc 0.65, mel-recall 0.76,
    T=1.25, 2026-09-29); fall back to vendored model.pth (T=4.7) so
    fresh clones without the git-ignored ft weights still run."""
    if FT_WEIGHTS.exists():
        return FT_WEIGHTS, "derm_cnn_ham10000_ft", 1.25
    return (WEIGHTS_DIR / "model.pth"), "derm_cnn_ham10000", 4.7


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast-Tier SkinCNN triage classifier.")
    parser.add_argument("--image", dest="image_path", required=True)
    parser.add_argument("--metadata", dest="metadata", required=False)
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(json.dumps({"status": "error", "message": f"Image not found: {args.image_path}"}))
        return 1
    weights, model_tag, temperature = resolve_weights()
    if not weights.exists():
        return fail(f"Fast-tier weights missing at: {weights}")

    metadata = {}
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError:
            return fail("Invalid metadata JSON.")

    try:
        import torch
    except ImportError:
        return fail("torch is not installed.")

    from control.vram import check as vram_check  # noqa: E402

    vram_ok, vram_reason = vram_check(1.0)
    if not vram_ok:
        return fail(f"VRAM guard: {vram_reason}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vram_before = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0

    labels = json.loads((WEIGHTS_DIR / "labels.json").read_text())
    model = None
    try:
        arch = load_arch()
        model, _ = arch.load_model(str(weights), device.type)
        image = ImageOps.exif_transpose(Image.open(args.image_path)).convert("RGB").resize((28, 28))
        pixels = list(image.get_flattened_data())
        inputs = torch.tensor(pixels, dtype=torch.float32).reshape(1, 28, 28, 3)
        inputs = inputs.permute(0, 3, 1, 2) / 255.0
        inputs = inputs.to(device)
        with torch.no_grad():
            probs = torch.softmax(model(inputs) / temperature, dim=1)[0]
            confidence, class_idx = torch.max(probs, dim=0)
        idx = int(class_idx.item())
        code = labels[str(idx)]
        result = {
            "status": "success",
            "tool": "skin-lesion-fast",
            "model_tier": "tier1_fast",
            "model_executed": model_tag,
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
            if torch.cuda.memory_allocated(device) - vram_before > VRAM_BUDGET_BYTES:
                print(json.dumps({"status": "warning", "message": "VRAM budget exceeded."}), file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
