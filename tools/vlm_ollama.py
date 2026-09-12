"""Ollama VLM wrapper: stronger small VLM candidate (Phase 7).

Sends the lesion image to a vision-capable Ollama model with a
constrained 7-class prompt and JSON mode. Confidence is null by design
(calibrated later via control/confidence.py closed-set protocol) —
same contract as tools/drdiag_vlm.py.

Sequential GPU rule: every request passes keep_alive=0 so the VLM
unloads after inference and never competes with the qwen3:8b agent LLM
for the 8 GB budget. The caller must likewise ensure the LLM is idle.
"""

import argparse
import base64
import json
import os
import urllib.request

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
    "image and classify it as EXACTLY ONE of these 7 classes. Reply with "
    "a JSON object only: {\"class\": \"<one of the 7 names>\", "
    "\"reasoning\": \"<one sentence>\"}. The 7 names:\n"
    + "\n".join(f"- {c}" for c in CLASSES)
)

DEFAULT_MODEL = "qwen3-vl:8b"
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def fail(message: str) -> int:
    print(json.dumps({"status": "error", "message": message}))
    return 1


REPAIR_SUFFIX = ("\nYour last reply was not valid JSON. Reply again with "
                 "ONLY the JSON object, no other text.")

# Substring fallback when the model will not emit JSON at all: scan the
# raw reply for a class name (same approach as tools/drdiag_vlm.py).
def extract_class(text: str) -> str | None:
    lowered = text.strip().lower()
    for name in CLASSES:
        if name.lower() in lowered:
            return name
    return None


def chat(model: str, image_b64: str, timeout: int,
         prompt: str = PROMPT) -> dict:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt,
                      "images": [image_b64]}],
        "format": "json",
        "stream": False,
        "keep_alive": 0,  # unload after inference (sequential GPU)
        # num_predict must cover the thinking trace + answer: 200 starves
        # thinking on some frames and yields empty content. Thinking stays
        # ON (think=false returns nothing on this model).
        "options": {"temperature": 0, "num_predict": 1024},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def classify(model: str, image_b64: str, timeout: int) -> tuple[str, str, int]:
    """Returns (class, reasoning, attempts). Exactly one repair retry,
    then substring fallback — mirroring proposal.tmp Section 7."""
    attempts = 0
    for prompt in (PROMPT, PROMPT + REPAIR_SUFFIX):
        attempts += 1
        reply = chat(model, image_b64, timeout)
        content = reply.get("message", {}).get("content", "")
        try:
            payload = json.loads(content)
        except (ValueError, TypeError):
            payload = None
        if isinstance(payload, dict) and payload.get("class") in CLASSES:
            return (payload["class"],
                    str(payload.get("reasoning", ""))[:500], attempts)
        last_content = content
    rescued = extract_class(last_content)
    if rescued is not None:
        return (rescued, "[rescued from non-JSON reply]", attempts)
    raise ValueError(f"Unparseable VLM reply: {last_content[:200]!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Ollama VLM screening tier.")
    parser.add_argument("--image", dest="image_path", required=True)
    parser.add_argument("--metadata", dest="metadata", required=False)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int, default=600)
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
        with open(args.image_path, "rb") as handle:
            image_b64 = base64.b64encode(handle.read()).decode()
    except OSError as exc:
        return fail(f"Unreadable: {exc}")

    try:
        predicted, reasoning, attempts = classify(
            args.model, image_b64, args.timeout)
        print(json.dumps({
            "status": "success",
            "tool": "vlm-ollama",
            "model_tier": "tier3_high",
            "model_executed": f"ollama_{args.model}",
            "predicted_class_index": CLASSES.index(predicted),
            "disease_name": predicted,
            "confidence_score": None,
            "reasoning": reasoning,
            "attempts": attempts,
            "metadata": metadata,
        }, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI must report, not crash
        return fail(f"Execution failed: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
