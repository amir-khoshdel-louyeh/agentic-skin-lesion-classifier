---
name: skin-lesion-tools
description: Local tools for tiered skin lesion analysis, confidence escalation, and medical reporting.
metadata: { "openclaw": { "requires": { "bins": ["python"] } } }
---

# Skin Lesion Tools

These tools are intended for use by OpenClaw agents that can execute terminal commands on the local host.

## Available commands

```bash
python tools/skin_tools.py analyze --image data/sample_lesion.jpg --model-tier tier1_fast
python tools/skin_tools.py analyze --image data/sample_lesion.jpg --model-tier tier2_deep
python tools/skin_tools.py escalate --image data/sample_lesion.jpg
python tools/skin_tools.py status
```

## Usage guidance

- Use `analyze` to run a single model tier and inspect the JSON output.
- Use `escalate` to run tier1 first and automatically escalate to tier2 when confidence is below 0.85.
- Use `status` to confirm the available local tools and paths.

## Why this skill exists

This skill documents the local CLI surface for skin lesion analysis, giving OpenClaw access to a reproducible toolchain rather than requiring the model to directly load the CNN models.
