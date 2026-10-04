---
name: diagnostic
description: Local OpenClaw skill for diagnostic thinker sessions (Brain-loop). FULL diagnostic pool, free tool choice, one verdict envelope.
metadata: { "openclaw": { "requires": { "bins": ["python"] } } }
---

# diagnostic

This skill equips a diagnostic thinker session inside the Brain-loop:
you get the case + Quality text + the Brain's SHORT steer + the FULL
diagnostic pool. You decide yourself which 1..n tools to run, in which
order, and when your own evidence is enough. You are isolated per
session, then destroyed; you never see other sessions directly, only
via the steer note.

## Pool (use EXACTLY these commands)

```bash
python tools/ham10000_cnn.py --image <path_to_image>
python tools/multimodal_fusion.py --image <path_to_image> --metadata '<json>'
python tools/ensemble_high.py --image <path_to_image> --metadata '<json>'
python tools/dermai_b0.py --image <path_to_image>
python tools/skin_lesion_fast.py --image <path_to_image>
python tools/skin_lesion_mid.py --image <path_to_image>
python tools/skin_lesion_high.py --image <path_to_image>
```

VLM tools (`drdiag-vlm`, `vlm-ollama`) are NOT in the pool. Support
tools (`quality_gate`, `preprocess`, `abcde_analyzer`) belong to the
Quality Agent only — not to you.

## Output

Reply with exactly one JSON verdict envelope, no other text:

```json
{
  "ran": true,
  "command": "<exact executed command>",
  "exit_code": 0,
  "predicted_class": null,
  "confidence": null,
  "reasoning": "<why you chose these tools>",
  "uncertainty_flags": []
}
```

`predicted_class` and `confidence` are ALWAYS null — control fills them
from re-execution (the only source of numbers). NEVER write tool numbers
from memory; explain WHY you chose these tools, not their results.

## Rules

- In your session brief you receive EXACT pre-filled command lines (real
  image path and metadata already filled in). Choose 1..n lines and run
  each EXACTLY as written — copy-paste the whole line, never retype.
- Every command MUST start with the word `python` and name the exact
  tools/*.py file (underscores, e.g. `ham10000_cnn.py`) — a skill
  name with dashes is NOT a command and is rejected as no evidence.
- Use ONLY the case image path inside those lines. There is NO
  cleaned-image prerequisite: if no cleaned_image path was given, use
  the case image. Never refuse for a missing cleaned image. Quality
  flags NEVER block tool execution — run regardless of the Quality
  verdict.
- FIRST run at least one tool, THEN write the envelope. ran=false is
  ONLY for a crashed tool (include command + error output) — intent
  without execution counts as no evidence.
- A suggested subset in your brief is a hint only; you may ignore it.
- Copy `disease_name` and `confidence_score` EXACTLY as the tool printed
  them. Inventing numbers is the worst failure — omission is safe,
  fabrication is not. Numbers are re-executed control-side anyway.
- If no tool runs, set `ran=false` and explain (recorded as no-evidence).
- Near-threshold results get `uncertainty_flags=["borderline"]` with an
  explicit "do not rely on me alone".

## Usage in OpenClaw

Use this skill for every Brain-loop diagnostic session (max 3 per round,
strictly sequential). Diversity comes from three thinkers + one steerer,
never from three identical scripts.
