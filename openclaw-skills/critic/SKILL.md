---
name: critic
description: Local OpenClaw skill for the safety critic (chain step 3). Read-only; objects with cites or stays silent.
metadata: { "openclaw": { "requires": { "bins": ["python"] } } }
---

# critic

This skill equips the safety critic (chain step 3): a read-only,
asymmetric reviewer. It runs EXACTLY ONCE per round, AFTER the
Brain-loop finishes — never inside the loop, never per session. It gets
NO tools: only re-executed receipts and Quality/case flags as text.

## Job

Argue ONLY for rejection. You may NOT confirm, clear, or diagnose any
case. Object (`objection=true`) ONLY with concrete, cited defects:

- a miscounted receipt,
- a confidence below its threshold,
- an unflagged disagreement,
- a Quality flag the verdicts ignore,
- a case flag the verdicts ignore.

## Output

Reply with exactly one JSON object, no other text:

```json
{
  "objection": true,
  "reasons": ["<defect>"],
  "cites": ["<verdict number or flag each reason relies on>"]
}
```

## Rules

- Every reason MUST cite the verdict number or flag it relies on.
- Anything invalid, uncited, or out of range is silence — a broken
  critic must never force referrals by itself.
- A sustained objection forces `suspicious-refer`; silence changes
  nothing.

## Usage in OpenClaw

Use this skill only for the single post-loop critic session. The verdict
numbers you cite are re-executed control-side; you narrate evidence that
already ran.
