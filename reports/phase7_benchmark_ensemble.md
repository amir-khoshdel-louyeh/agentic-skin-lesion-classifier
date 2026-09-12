# Phase 7 Benchmark — ensemble-high (tier3_high, ACCEPTED)

Fallback high tier per plan.tmp: both ready-made candidates rejected,
so the high tier is the mean of the triage-CNN and multimodal-fusion
softmax vectors, gated by closed-set entropy (threshold 0.9 nats).
Members run sequentially with release between; uniform JSON contract.

- Images attempted: 24, scored: 21, failures: 0
- Accuracy: 0.7619 (ties triage CNN 0.7619, beats mid 0.5714)
- Avg confidence: 0.7282 (vs CNN 0.9931 — honest, not hot)
- Avg seconds/image: 2.45
- Gated borderline: 8/24, member disagreement: 10/24

## Per-class accuracy

- Actinic keratoses: 0.0 (n=3) — shared blind spot survives
- Basal cell carcinoma: 1.0 (n=3)
- Benign keratosis: 1.0 (n=3)
- Dermatofibroma: 1.0 (n=3)
- Melanocytic nevi: 1.0 (n=3)
- Melanoma: 0.3333 (n=3)
- Vascular lesions: 1.0 (n=3)

## Gate audit (5 errors in scored set)

- 4/5 errors gated: 0000002, 0000074, 0056176 (borderline+disagreement),
  0056166 (disagreement + confidence 0.63 < 0.75 threshold — decide.py
  refers it anyway).
- 1 confident error escapes: 0056302 called Melanoma at 0.85, no flags
  (truth: Actinic keratoses). The gate narrows silent failure, it does
  not eliminate it — documented, not hidden.

## Failures

- none

## Role assignment: ACCEPTED as tier3_high

- Why it earns the slot despite tying CNN accuracy: equal accuracy at
  honestly calibrated confidence (0.73 vs 0.99) plus 4/5 errors
  converted from silent verdicts into referred borderlines. A high
  tier's job is ambiguity, not raw accuracy — this one refuses to
  clear cases it disagrees on.
- Wired as the third agent role in orchestrator.py ("High-tier role:
  use ONLY tools/ensemble_high.py ..."). control/decide.py is
  verdict-count agnostic; gated verdicts force "suspicious — refer"
  through the existing borderline rule with zero decision-code changes.
- Known limits carried forward: AK blind spot (all three tiers 0/3),
  Melanoma recall 1/3, one escaping confident error (0056302).
- Manifest: category diagnostic, tier3_high, status ready, calibrated
  false, 2.0 GB VRAM (sequential members, released between).
