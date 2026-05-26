import argparse
import json
from pathlib import Path
from PIL import Image
from core import analyze_skin_lesion

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif"}


def validate_image_path(image_path: str) -> dict:
    path = Path(image_path)
    if not path.exists():
        return {
            "status": "error",
            "message": f"Image not found at: {image_path}"
        }
    if not path.is_file():
        return {
            "status": "error",
            "message": f"Path exists but is not a file: {image_path}"
        }
    if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        return {
            "status": "error",
            "message": f"Unsupported file type: {path.suffix}. Supported types: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}"
        }

    try:
        with Image.open(path) as img:
            img.verify()
    except Exception:
        return {
            "status": "error",
            "message": f"File is not a valid image or is corrupted: {image_path}"
        }

    return {"status": "ok"}


def analyze_command(args: argparse.Namespace) -> int:
    validation = validate_image_path(args.image_path)
    if validation["status"] != "ok":
        print(json.dumps(validation, indent=2, ensure_ascii=False))
        return 1

    result = analyze_skin_lesion(args.image_path, args.model_tier)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result.get("status") == "success" else 1


def escalate_command(args: argparse.Namespace) -> int:
    validation = validate_image_path(args.image_path)
    if validation["status"] != "ok":
        print(json.dumps(validation, indent=2, ensure_ascii=False))
        return 1

    first = analyze_skin_lesion(args.image_path, "tier1_fast")
    output = {"tier1_result": first}

    if first.get("status") != "success":
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return 1

    if first["confidence_score"] < args.threshold:
        second = analyze_skin_lesion(args.image_path, "tier2_deep")
        output["tier2_result"] = second
        output["final_result"] = second if second.get("status") == "success" else first
    else:
        output["final_result"] = first

    print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


def status_command(args: argparse.Namespace) -> int:
    print(json.dumps({
        "sample_image_path": str(Path(args.image_path).resolve()) if args.image_path else None,
        "model_tiers": ["tier1_fast", "tier2_deep"],
        "description": "Local skin lesion analysis toolset available to OpenClaw terminal and skill workflows."
    }, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python tools/skin_tools.py",
        description="Local skin lesion analysis tools for OpenClaw-driven workflows."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser("analyze", help="Run a tiered skin lesion model analysis.")
    analyze_parser.add_argument("--image", dest="image_path", required=True, help="Path to the lesion image.")
    analyze_parser.add_argument("--model-tier", dest="model_tier", choices=["tier1_fast", "tier2_deep"], required=True)
    analyze_parser.set_defaults(func=analyze_command)

    escalate_parser = subparsers.add_parser("escalate", help="Run tier1 analysis and escalate to tier2 if needed.")
    escalate_parser.add_argument("--image", dest="image_path", required=True, help="Path to the lesion image.")
    escalate_parser.add_argument("--threshold", type=float, default=0.85, help="Confidence threshold for escalation.")
    escalate_parser.set_defaults(func=escalate_command)

    status_parser = subparsers.add_parser("status", help="Show the available local skin lesion tools.")
    status_parser.add_argument("--image", dest="image_path", default=None, help="Optional sample image path.")
    status_parser.set_defaults(func=status_command)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
