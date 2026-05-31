# Local tool command helper

Available local CLI tools:

1. `C:/Amir/GitHub/agentic-skin-lesion-classifier/.venv/Scripts/python.exe skills/skin-lesion-fast/tools/skin_lesion_fast.py --image <path> [--metadata <json>]`
   - Fast first-pass screening
   - Model tier: `tier1_fast`
   - Model executed: `efficientnet_b0`

2. `C:/Amir/GitHub/agentic-skin-lesion-classifier/.venv/Scripts/python.exe skills/skin-lesion-mid/tools/skin_lesion_mid.py --image <path> [--metadata <json>]`
   - Balanced mid-tier screening
   - Model tier: `tier2_mid`
   - Model executed: `efficientnet_b4`

Guidance:
- Choose the most appropriate local tool based on the prompt and metadata.
- Prefer `skin_lesion_fast.py` for lightweight screening.
- Escalate to `skin_lesion_mid.py` if the prompt asks for higher accuracy or if confidence is low.
- Execute the exact command shown above; do not invent or rely on shell aliases like `skin-lesion-fast`.
- Return the final result as concise Markdown and include the exact command executed.
