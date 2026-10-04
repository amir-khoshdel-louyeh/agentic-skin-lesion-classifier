---
name: specialist
description: Local OpenClaw skill for disease specialists (chain step 4). Advocates FOR one assigned disease with cites; never averages.
metadata: { "openclaw": { "requires": { "bins": ["python"] } } }
---

# specialist

This skill equips a disease specialist (chain step 4): an advocate FOR
exactly ONE assigned shortlisted disease. Specialists run only on
trigger (≥2 distinct countable classes, top-2 shortlist) and only after
the Brain-loop + critic. They see the same FULL diagnostic pool as the
diagnostic sessions.

## Job

Argue why the evidence fits YOUR assigned disease (the alternative
under debate is named in your brief). Follow the Brain-written angle in
your brief as steering — verify it yourself with a tool.

## Commands

Use any ONE ready diagnostic tool, e.g.:

```bash
python tools/ham10000_cnn.py --image <path_to_image>
python tools/multimodal_fusion.py --image <path_to_image> --metadata '<json>'
python tools/ensemble_high.py --image <path_to_image> --metadata '<json>'
```

## Output

Reply with exactly one JSON verdict envelope (see JSON instruction),
citing the receipt command and numbers.

## Rules

- Every command MUST start with the word `python` and name the exact
  tools/*.py file (underscores) — a skill name with dashes is NOT a command
  and is rejected as no evidence.
- Your `predicted_class` MUST be your assigned disease unless the
  tool's own output says otherwise — report the tool output faithfully,
  never fabricate it.
- A class outside the shortlist is recorded as no-evidence.
- Persistent disagreement is named (`specialist dilemma: A vs B`), never
  averaged away.

## Usage in OpenClaw

Use this skill only for advocate sessions on the triggered shortlist.
Cite receipts per claim; beyond the evidence, stay silent.
