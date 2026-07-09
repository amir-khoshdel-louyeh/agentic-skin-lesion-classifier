"""VLM high-tier wrapper: HAM10000 fine-tuned Qwen2-VL 2B (Phase 2).

Generative VLM, not a classifier head: the model names one of the 7
classes in constrained text. Confidence is left null — the orchestrator
supplies calibrated confidence via the closed-set protocol
(control/confidence.py). VRAM: full LLM unload required before this
tool (sequential schedule); release in finally.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "drdiag-vlm"

CLASSES = [
    "Actinic keratoses",
    "Basal cell carcinoma",
    "Benign keratosis",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic nevi",
    "Vascular lesions",
]

PROMPT = (
    "You are a dermatology screening assistant. Look at the skin lesion "
    "image and reply with EXACTLY ONE of these 7 class names, nothing else:\n"
    + "\n".join(f"- {c}" for c in CLASSES)
)


def fail(message: str) -> int:
    print(json.dumps({"status": "error", "message": message}))
    return 1


def extract_class(text: str) -> str | None:
    lowered = text.strip().lower()
    for name in CLASSES:
        if name.lower() in lowered:
            return name
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="DrDiag VLM high tier.")
    parser.add_argument("--image", dest="image_path", required=True)
    parser.add_argument("--metadata", dest="metadata", required=False)
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        return fail(f"Image not found: {args.image_path}")
    if not MODEL_DIR.exists():
        return fail(f"VLM weights missing at: {MODEL_DIR}")

    metadata: dict = {}
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError:
            return fail("Invalid metadata JSON.")

    try:
        import torch
        from PIL import Image
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    except ImportError as exc:
        return fail(f"Missing dependency: {exc}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    vram_before = torch.cuda.memory_allocated() if device == "cuda" else 0
    model = None
    try:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            str(MODEL_DIR), torch_dtype=dtype, local_files_only=True
        ).to(device).eval()
        processor = AutoProcessor.from_pretrained(
            str(MODEL_DIR), local_files_only=True
        )
        image = Image.open(args.image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text], images=[image], return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=32)
        reply = processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0]
        predicted = extract_class(reply)
        if predicted is None:
            return fail(f"Unparseable VLM reply: {reply!r}")
        result = {
            "status": "success",
            "tool": "drdiag-vlm",
            "model_tier": "tier3_high",
            "model_executed": "drdiag_qwen2vl_ham10000",
            "predicted_class_index": CLASSES.index(predicted),
            "disease_name": predicted,
            "confidence_score": None,
            "raw_reply": reply.strip(),
            "metadata": metadata,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI must report, not crash
        return fail(f"Execution failed: {exc}")
    finally:
        del model
        if device == "cuda":
            import torch as _t

            _t.cuda.empty_cache()
            if _t.cuda.memory_allocated() - vram_before > 6 * 1024**3:
                print(
                    json.dumps({"status": "warning",
                                "message": "VLM VRAM residue high."}),
                    file=sys.stderr,
                )


if __name__ == "__main__":
    raise SystemExit(main())
