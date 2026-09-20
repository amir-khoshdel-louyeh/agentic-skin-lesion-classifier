---
name: skin-lesion-high
description: Local OpenClaw tool for high-accuracy skin lesion classification using image+metadata fusion.
metadata: { "openclaw": { "requires": { "bins": ["python"] } } }
---

# skin-lesion-high

This skill runs the local high-tier skin lesion classification model.

## Command

```bash
python tools/skin_lesion_high.py --image <path_to_image> [--metadata '<json_string>']
```

## Behavior

- Uses internal model tier `tier3_high`
- Runs the vendored `cross_attention_fusion_ham10000` model (`models/multimodal/best.pt`)
- Fuses the lesion image with age/sex metadata (missing values handled as zeros/unknown)
- Applies image preprocessing:
  - EXIF orientation correction
  - RGB conversion
  - Resize to `224×224`
  - ImageNet normalization (mean `0.485/0.456/0.406`, std `0.229/0.224/0.225`)
- Output adds `entropy`, `entropy_threshold`, and `uncertainty_flags` (uncertain cases carry `"borderline"`)
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
- `entropy`
- `entropy_threshold`
- `uncertainty_flags`
- `metadata`

## Usage in OpenClaw

Use this tool when the agent requires the highest-accuracy local skin lesion classification.

Typical scenarios include:

- Escalation from the fast screening model when confidence is low
- Final diagnostic classification before presenting results
- Offline inference when internet access is unavailable
- Cases where accuracy is prioritized over inference speed