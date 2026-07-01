---
name: skin-lesion-high
description: Local OpenClaw tool for high-accuracy skin lesion classification using an offline ViT-Large model.
metadata: { "openclaw": { "requires": { "bins": ["python"] } } }
---

# skin-lesion-high

This skill runs the local high-tier skin lesion classification model.

## Command

```bash
python C:/Amir/GitHub/agentic-skin-lesion-classifier/openclaw-skills/skin-lesion-high/tools/skin_lesion_high.py --image <path_to_image> [--metadata '<json_string>']
```

## Behavior

- Uses internal model tier `tier3_high`
- Runs the offline `ViT-Large` HAM10000 classifier
- Automatically uses CUDA when a supported NVIDIA GPU is available; otherwise falls back to CPU
- Applies image preprocessing:
  - EXIF orientation correction
  - RGB conversion
  - Resize to `224×224`
  - Normalization using mean/std of `0.5`
- Accepts optional metadata as a JSON string and includes it in the output
- Intended for high-confidence final classification after lower-tier screening or whenever maximum accuracy is desired

## Output

The tool returns a JSON object with:

- `status`
- `tool`
- `model_tier`
- `model_executed`
- `predicted_class_index`
- `disease_name`
- `confidence_score`
- `metadata`

## Usage in OpenClaw

Use this tool when the agent requires the highest-accuracy local skin lesion classification.

Typical scenarios include:

- Escalation from the fast screening model when confidence is low
- Final diagnostic classification before presenting results
- Offline inference when internet access is unavailable
- Cases where accuracy is prioritized over inference speed