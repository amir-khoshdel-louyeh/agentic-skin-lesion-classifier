---
name: skin-lesion-mid
description: Local OpenClaw tool for balanced skin lesion screening using EfficientNet-B0.
metadata: { "openclaw": { "requires": { "bins": ["python"] } } }
---

# skin-lesion-mid

This skill runs the balanced skin lesion model using DermAI EfficientNet-B0.

## Command

```bash
python tools/skin_lesion_mid.py --image <path_to_image>
```

## Behavior

- Uses internal model tier `tier2_mid`
- Runs `dermai_efficientnet_b0_ham10000` (`models/dermai-b0`) as the mid-tier screening model
- Intended for higher-quality predictions with moderate latency
- Uses the vendored image processor (224px) from `models/dermai-b0`

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
