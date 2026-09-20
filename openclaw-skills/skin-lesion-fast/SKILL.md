---
name: skin-lesion-fast
description: Local OpenClaw tool for fast skin lesion screening using SkinCNN triage.
metadata: { "openclaw": { "requires": { "bins": ["python"] } } }
---

# skin-lesion-fast

This skill runs the local fast screening skin lesion model.

## Command

```bash
python tools/skin_lesion_fast.py --image <path_to_image>
```

## Behavior

- Uses internal model tier `tier1_fast`
- Runs `derm_cnn_ham10000` (SkinCNN, `models/derm-cnn`, 28×28 input) as the first-pass screening model
- Intended for low-latency, quick detection and early filtering

## Output

The tool returns a JSON object with:

- `status`
- `model_executed`
- `predicted_class_index`
- `disease_name`
- `confidence_score`

## Usage in OpenClaw

Use this tool when the agent should perform the first-pass screening only.
If the confidence score is low, the agent can escalate to the deeper tool.
