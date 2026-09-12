# Phase 6 Benchmark — abcde-analyzer (support)

Deterministic OpenCV heuristic (asymmetry, border, color, diameter +
0–8 score). No weights, CPU-only, ~0.2 s/image.

- Images attempted: 24, segmented: 24, failures: 0
- Band distribution: high 10 / moderate 9 / low 5
- Avg seconds/image: 0.20
- Scored vs ground truth: 21 (same 3 stems missing from the HAM10000
  training CSV as in Phase 2)

## Malignancy enrichment (Melanoma / BCC / AK = malignant)

- High band: 4 malignant / 9 scored (44%)
- Moderate band: 4 malignant / 8 scored (50%)
- Low band: 1 malignant / 4 scored (25%)
- Base rate: 9 malignant / 21 scored (43%)

No meaningful enrichment: the score separates morphologically busy from
quiet lesions but does NOT separate malignant from benign on dermoscopy
close-ups. First calibration (naive ABCDE mapping) was degenerate
(20/21 "high"); rescaled cutoffs + background-inversion fix documented
in the tool docstring.

## Failures

- none (3 whole-frame Otsu inversions handled inside the tool)

## Role assignment

- Assigned: support flag ONLY (tier `support`, category `analysis`).
  Agents may cite the band as one morphological observation; it MUST
  never trigger escalation alone and never appear in a verdict as a
  diagnosis.
- Reason: zero discriminative power for malignancy on this data
  (high-band malignancy rate ≈ base rate). Honest weak signal, kept
  because it is deterministic, free (0 GB VRAM), and gives the 8B agent
  grounded shape/color vocabulary instead of invented descriptions.
- Manifest: status ready, calibrated false.
