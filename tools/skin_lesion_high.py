"""High-tier classifier: calibrated CNN + multimodal ensemble.

Mean of the triage-CNN (models/derm-cnn) and cross-attention fusion
(models/multimodal/best.pt) softmax vectors, each temperature-scaled
(CNN T=4.7, MM T=3.65, fitted by NLL on a 105-image ISIC2019_full split,
2026-09-29), gated by closed-set entropy (control/confidence.py).
Pure fusion only reached ~0.38 accuracy without real metadata, so the
mean-ensemble of strong ready weights replaces it — no training needed.
Members run SEQUENTIALLY (load -> infer -> release each). A gated or
disagreeing verdict carries uncertainty_flags, forcing
"suspicious — refer" in control/decide.py by design.
Uniform JSON contract; VRAM released in finally with budget check.
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
from tools.ham10000_cnn import (  # noqa: E402
    ISIC_TO_CLASS,
    WEIGHTS_DIR as CNN_DIR,
)
from tools.ham10000_cnn import load_arch as load_cnn_arch
from tools.ham10000_cnn import preprocess as cnn_preprocess

MM_DIR = BASE_DIR / "models" / "multimodal"

MM_IDX_TO_CLASS = {
    0: "Actinic keratoses",
    1: "Basal cell carcinoma",
    2: "Benign keratosis",
    3: "Dermatofibroma",
    4: "Melanoma",
    5: "Melanocytic nevi",
    6: "Vascular lesions",
}

CNN_TEMPERATURE = 4.7
MM_TEMPERATURE = 3.65
VRAM_BUDGET_BYTES = 2 * 1024**3

MM_PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def fail(message: str) -> int:
    print(json.dumps({"status": "error", "message": message}))
    return 1


def load_mm_arch():
    spec = importlib.util.spec_from_file_location(
        "mm_arch_high", MM_DIR / "arch.py")
    arch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arch)
    return arch


def member_cnn(torch, device) -> dict[str, float]:
    """Full 7-class distribution keyed by disease name (calibrated)."""
    import json as _json

    labels = _json.loads((CNN_DIR / "labels.json").read_text())
    arch = load_cnn_arch()
    model, _ = arch.load_model(str(CNN_DIR / "model.pth"), device.type)
    try:
        inputs = cnn_preprocess(IMAGE_PATH, torch, device)
        with torch.no_grad():
            probs = torch.softmax(
                model(inputs) / CNN_TEMPERATURE, dim=1)[0]
        return {ISIC_TO_CLASS[labels[str(i)]]: float(probs[i])
                for i in range(len(labels))}
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def member_multimodal(torch, device, metadata: dict) -> dict[str, float]:
    """Full 7-class distribution keyed by disease name (calibrated)."""
    arch = load_mm_arch()
    model = arch.CrossAttentionFusionModel(meta_dim=19, num_classes=7)
    try:
        checkpoint = torch.load(str(MM_DIR / "best.pt"),
                                map_location=device, weights_only=False)
        state = checkpoint["model"] if "model" in checkpoint else checkpoint
        model.load_state_dict(state)
        model.to(device).eval()
        image = ImageOps.exif_transpose(
            Image.open(IMAGE_PATH)).convert("RGB")
        inputs = MM_PREPROCESS(image).unsqueeze(0).to(device)
        meta = arch.encode_metadata(
            age=metadata.get("age"),
            sex=metadata.get("sex"),
            localization=metadata.get("localization"),
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(
                model(inputs, meta) / MM_TEMPERATURE, dim=1)[0]
        return {MM_IDX_TO_CLASS[i]: float(probs[i])
                for i in range(len(MM_IDX_TO_CLASS))}
    finally:
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


IMAGE_PATH = ""


def main() -> int:
    global IMAGE_PATH
    parser = argparse.ArgumentParser(
        description="Calibrated CNN+multimodal ensemble high tier.")
    parser.add_argument("--image", dest="image_path", required=True)
    parser.add_argument("--metadata", dest="metadata", required=False)
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        return fail(f"Image not found: {args.image_path}")
    if not (MM_DIR / "best.pt").exists():
        return fail(f"Multimodal weights missing at: {MM_DIR}")
    metadata: dict = {}
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError:
            return fail("Invalid metadata JSON.")
    IMAGE_PATH = args.image_path

    try:
        import torch
    except ImportError:
        return fail("torch is not installed.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vram_before = (torch.cuda.memory_allocated(device)
                   if device.type == "cuda" else 0)
    try:
        # Sequential: member A fully released before B loads.
        dist_a = member_cnn(torch, device)
        dist_b = member_multimodal(torch, device, metadata)
        classes = sorted(set(dist_a) | set(dist_b))
        avg = {c: round((dist_a.get(c, 0.0) + dist_b.get(c, 0.0)) / 2.0, 4)
               for c in classes}
        gate = entropy_gate(avg, entropy_threshold=DEFAULT_ENTROPY_THRESHOLD)
        top_a = max(dist_a, key=dist_a.get)
        top_b = max(dist_b, key=dist_b.get)
        members_agree = top_a == top_b
        flags = list(gate["uncertainty_flags"])
        if not members_agree and "disagreement" not in flags:
            flags.append("disagreement")
        print(json.dumps({
            "status": "success",
            "tool": "skin-lesion-high",
            "model_tier": "tier3_high",
            "model_executed": "mean_cnn_multimodal_7class_calibrated",
            "predicted_class_index": classes.index(gate["top_class"]),
            "disease_name": gate["top_class"],
            "confidence_score": gate["confidence"],
            "entropy": gate["entropy"],
            "entropy_threshold": DEFAULT_ENTROPY_THRESHOLD,
            "members": {
                "ham10000-cnn": {"class": top_a,
                                 "confidence": round(dist_a[top_a], 4)},
                "multimodal-fusion": {"class": top_b,
                                      "confidence": round(dist_b[top_b], 4)},
                "agree": members_agree,
            },
            "uncertainty_flags": flags,
            "metadata": metadata,
        }, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI must report, not crash
        return fail(f"Execution failed: {exc}")
    finally:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            if torch.cuda.memory_allocated(device) - vram_before > VRAM_BUDGET_BYTES:
                print(json.dumps({"status": "warning",
                                  "message": "VRAM budget exceeded."}),
                      file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
