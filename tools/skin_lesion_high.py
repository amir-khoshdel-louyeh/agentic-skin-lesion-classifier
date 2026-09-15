"""High-tier classifier: cross-attention fusion (models/multimodal).

Uses the vendored HAM10000 multimodal weights (best.pt + arch.py,
224px + age/sex metadata, meta_dim=19) — no download needed.
Adds closed-set entropy gating (control/confidence.py) so uncertain
cases carry uncertainty_flags ["borderline"]. Uniform JSON contract.
VRAM discipline: load -> infer -> release in finally.
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
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from control.confidence import DEFAULT_ENTROPY_THRESHOLD  # noqa: E402
from control.confidence import evaluate as entropy_gate  # noqa: E402

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

VRAM_BUDGET_BYTES = 2 * 1024**3

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
    parser = argparse.ArgumentParser(description="High-Tier multimodal fusion classifier.")
    parser.add_argument("--image", dest="image_path", required=True)
    parser.add_argument("--metadata", dest="metadata", required=False)
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(json.dumps({"status": "error", "message": f"Image not found: {args.image_path}"}))
        return 1
    if not (WEIGHTS_DIR / "best.pt").exists():
        return fail(f"High-tier weights missing at: {WEIGHTS_DIR / 'best.pt'}")
    if not (WEIGHTS_DIR / "arch.py").exists():
        return fail(f"High-tier arch missing at: {WEIGHTS_DIR / 'arch.py'}")

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

    spec = importlib.util.spec_from_file_location("mm_arch_high", WEIGHTS_DIR / "arch.py")
    arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vram_before = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    model = None
    try:
        model = arch.CrossAttentionFusionModel(meta_dim=19, num_classes=7)
        checkpoint = torch.load(str(WEIGHTS_DIR / "best.pt"), map_location=device, weights_only=False)
        state = checkpoint["model"] if "model" in checkpoint else checkpoint
        model.load_state_dict(state)
        model.to(device).eval()

        image = ImageOps.exif_transpose(Image.open(args.image_path)).convert("RGB")
        inputs = PREPROCESS(image).unsqueeze(0).to(device)
        meta = arch.encode_metadata(
            age=metadata.get("age"),
            sex=metadata.get("sex"),
            localization=metadata.get("localization"),
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(inputs, meta)
            probs = torch.softmax(logits, dim=1)[0]
        dist = {IDX_TO_CLASS[i]: round(float(probs[i]), 4) for i in range(len(IDX_TO_CLASS))}
        gate = entropy_gate(dist, entropy_threshold=DEFAULT_ENTROPY_THRESHOLD)
        result = {
            "status": "success",
            "tool": "skin-lesion-high",
            "model_tier": "tier3_high",
            "model_executed": "cross_attention_fusion_ham10000",
            "predicted_class_index": list(IDX_TO_CLASS.values()).index(gate["top_class"]),
            "disease_name": gate["top_class"],
            "confidence_score": gate["confidence"],
            "entropy": gate["entropy"],
            "entropy_threshold": DEFAULT_ENTROPY_THRESHOLD,
            "uncertainty_flags": gate["uncertainty_flags"],
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
