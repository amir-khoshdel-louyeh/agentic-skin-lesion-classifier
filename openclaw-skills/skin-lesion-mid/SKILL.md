---
name: skin-lesion-mid
description: Local OpenClaw tool for balanced skin lesion screening using EfficientNet-B4.
metadata: { "openclaw": { "requires": { "bins": ["python"] } } }
---

# skin-lesion-mid

This skill runs the balanced skin lesion model using EfficientNet-B4.

## Command

```bash
C:/Amir/GitHub/agentic-skin-lesion-classifier/.venv/Scripts/python.exe skills/skin-lesion-mid/tools/skin_lesion_mid.py --image <path_to_image>
```

## Behavior

- Uses internal model tier `tier2_mid`
- Runs `efficientnet_b4` as the mid-tier screening model
- Intended for higher-quality predictions with moderate latency
- Uses a 380x380 input resize for EfficientNet-B4

## Output

The tool returns a JSON object with:

- `status`
- `tool`
- `model_tier`
- `model_executed`
- `predicted_class_index`
- `disease_name`
- `confidence_score`

## Usage in OpenClaw

Use this tool when the agent should escalate from the fast screening pass to a more accurate mid-tier model.
