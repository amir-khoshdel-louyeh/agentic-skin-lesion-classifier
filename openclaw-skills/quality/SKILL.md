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

Your session brief carries the same EXACT pre-filled lines the
diagnostic sessions get: the real case image is already in them, and the
`preprocess` line already carries a concrete `--output` path. Run a line
EXACTLY as written — never construct or retype a path yourself, and
never substitute `<path>`-style placeholders.

When you clean, report that line's exact `--output` path as
`cleaned_image` — the path must exist on disk, and downstream sessions
may run their tools on it (never a placeholder).

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

- You have no image capability: you never look at, describe, or search
  the web for the image. The case image is only a path argument inside
  the command lines. Judge solely from the tool JSON and the
  control-side `quality_gate` result in your brief.
- `pass`: image good enough as-is.
- `cleaned`: you ran `preprocess`; `cleaned_image` is required.
- `flagged`: usable but with flags — the round flows forward with the flag.
  Use this whenever anything is uncertain, or whenever you did not run
  the `preprocess` line.
- Never emit disease names or malignancy judgments (never diagnoses).
- Do not call any tool other than the lines above, and do not write
  medical commentary.
- Never abort the round (never stops the round): a tool crash is
  `no-evidence` for that check, recorded in `reasons`/`cites`.
- Every reason MUST cite the tool JSON it relies on.

## Usage in OpenClaw

Use this skill only for the Quality Agent at the start of a round. The
verdict text is stored and passed forward to the Brain-loop; downstream
sessions decide diagnosis from their own tools.
