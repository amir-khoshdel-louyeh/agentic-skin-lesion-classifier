# Phase 5 Validation
Date: 2026-09-25 · Full 24-record sweep, all-local Ollama qwen3:8b.

## Sweep
- Records covered: 24/24 (indices 0–23), failures: 0.
- Decisions across all logged rounds: mostly suspicious (one-way
  escalation working as designed); benign only on full tier agreement.
- Per-round round_*.md reports are runtime artifacts (git-ignored);
  audit.jsonl holds the machine-readable trail.

## Performance
- Avg ~15.5 s/round (2 sequential agents, GPU-resident 8B model).
- GPU VRAM peak 5773 MiB of 8151 MiB — inside the 8 GB budget;
  sequential schedule holds (no LLM/tool contention observed).

## Debate dry-run
- `main.py --chat --record 0` prints the grounded evidence block
  (round report + audit receipts) then enters the interactive loop.
- Missing round fails fast with a rerun hint (verified).

## Known limitations
- Triage CNN is overconfident (0.76 acc @ 0.99 conf); VLM weights
  rejected (mode collapse); shared AK blind spot 0/3 on both tiers.
- Agent transcription errors occur; control-side verification corrects
  them (disagreement flag → escalation).
- Research use only — not a medical device or diagnosis.
