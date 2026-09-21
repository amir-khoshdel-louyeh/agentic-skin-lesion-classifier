"""Multimodal mid-tier wrapper: image + metadata fusion (Phase 2).

Cross-attention model (cesaraha, Apache-2.0) with reproduced encoding
stats (see models/multimodal/arch.py). Local records carry age/sex but
no localization, so the location vector is zeros — identical to the
author's own missing-value handling. Uniform JSON contract; VRAM
released in finally with budget check.
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

from PIL import Image, ImageOps
from torchvision import transforms

BASE_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "models" / "multimodal"

IDX_TO_CLASS = {
    0: "Actinic keratoses",
    1: "Basal cell carcinoma",
    2: "Benign keratosis",
    3: "Dermatofibroma",
    4: "Melanoma",
    5: "Melanocytic nevi",
    6: "Vascular lesions",
}

VRAM_BUDGET_BYTES = 1 * 1024**3

FT_WEIGHTS = BASE_DIR / "models" / "multimodal-ft" / "best_ft.pt"


def resolve_weights() -> tuple[Path, str, float]:
    """Prefer fine-tuned weights (held-out acc 0.67 with real metadata,
    T=1.1, 2026-09-29); fall back to vendored best.pt (T=3.65) so fresh
    clones without the git-ignored ft weights still run."""
    if FT_WEIGHTS.exists():
        return FT_WEIGHTS, "cross_attention_fusion_ham10000_ft", 1.1
    return (WEIGHTS_DIR / "best.pt",
            "cross_attention_fusion_ham10000", 3.65)

PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def fail(message: str) -> int:
    print(json.dumps({"status": "error", "message": message}))
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Multimodal fusion tier.")
    parser.add_argument("--image", dest="image_path", required=True)
    parser.add_argument("--metadata", dest="metadata", required=False)
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        return fail(f"Image not found: {args.image_path}")
    weights, model_tag, temperature = resolve_weights()
    if not weights.exists():
        return fail(f"Multimodal weights missing at: {weights}")

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

    spec = importlib.util.spec_from_file_location(
        "mm_arch", WEIGHTS_DIR / "arch.py"
    )
    arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vram_before = (
        torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    )
    model = None
    try:
        model = arch.CrossAttentionFusionModel(meta_dim=19, num_classes=7)
        checkpoint = torch.load(str(weights), map_location=device,
                                weights_only=False)
        state = checkpoint["model"] if "model" in checkpoint else checkpoint
        model.load_state_dict(state)
        model.to(device).eval()

        image = ImageOps.exif_transpose(
            Image.open(args.image_path)).convert("RGB")
        inputs = PREPROCESS(image).unsqueeze(0).to(device)
        meta = arch.encode_metadata(
            age=metadata.get("age"),
            sex=metadata.get("sex"),
            localization=metadata.get("localization"),
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(
                model(inputs, meta) / temperature, dim=1)[0]
            confidence, class_idx = torch.max(probs, dim=0)
        idx = int(class_idx.item())
        result = {
            "status": "success",
            "tool": "multimodal-fusion",
            "model_tier": "tier2_mid",
            "model_executed": model_tag,
            "predicted_class_index": idx,
            "disease_name": IDX_TO_CLASS[idx],
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
                print(
                    json.dumps({"status": "warning",
                                "message": "VRAM budget exceeded."}),
                    file=sys.stderr,
                )


if __name__ == "__main__":
    raise SystemExit(main())
