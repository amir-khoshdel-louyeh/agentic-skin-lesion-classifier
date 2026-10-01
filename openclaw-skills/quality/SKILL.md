---
name: quality
description: Local OpenClaw skill for image-quality assessment (chain step 1). The Quality Agent decides pass/cleaned/flagged and never diagnoses.
metadata: { "openclaw": { "requires": { "bins": ["python"] } } }
---

# quality

This skill equips the Quality Agent (chain step 1): assess whether the
case image is good enough, clean it with `preprocess` (0..n times by its
own judgment), or pass it forward with flags. It never diagnoses and
never stops the round.

## Commands

```bash
python tools/quality_gate.py --image <path_to_image>
python tools/preprocess.py --image <path_to_image> --output <cleaned_path> --steps hair,denoise
python tools/abcde_analyzer.py --image <path_to_image>
```

- `quality_gate`: blur / exposure / resolution evidence.
- `preprocess`: hair removal + denoise; writes a cleaned copy for re-check.
- `abcde_analyzer`: read-only context (shape/color heuristic). Its output
  informs quality wording only — never a diagnosis.

## Output

Reply with exactly one JSON object, no other text:

```json
{
  "verdict": "pass|cleaned|flagged",
  "reasons": ["<why, citing tool JSON>"],
  "cites": ["<tool name + value seen>"],
  "cleaned_image": "<path from preprocess>|null"
}
```

## Rules

- `pass`: image good enough as-is.
- `cleaned`: you ran `preprocess`; `cleaned_image` is required.
- `flagged`: usable but with flags — the round flows forward with the flag.
- Never emit disease names or malignancy judgments (never diagnoses).
- Never abort the round (never stops the round): a tool crash is
  `no-evidence` for that check, recorded in `reasons`/`cites`.
- Every reason MUST cite the tool JSON it relies on.

## Usage in OpenClaw

Use this skill only for the Quality Agent at the start of a round. The
verdict text is stored and passed forward to the Brain-loop; downstream
sessions decide diagnosis from their own tools.
