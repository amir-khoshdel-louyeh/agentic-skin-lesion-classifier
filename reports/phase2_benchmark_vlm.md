# Phase 2 Benchmark — drdiag-vlm (tier3_high)

- Images attempted: 24, scored: 21
- Accuracy: 0.1429
- Avg confidence: n/a (VLM confidence is null by design; calibrated via closed-set protocol)
- Avg seconds/image: 5.58

## Per-class accuracy

- Actinic keratoses: 0.0 (n=3)
- Basal cell carcinoma: 0.0 (n=3)
- Benign keratosis: 0.0 (n=3)
- Dermatofibroma: 0.0 (n=3)
- Melanocytic nevi: 0.0 (n=3)
- Melanoma: 1.0 (n=3)
- Vascular lesions: 0.0 (n=3)

## Failures

- none

## Role assignment: REJECTED
The model predicts "Melanoma" for all 24 images (mode collapse).
Accuracy 0.143 equals the melanoma base rate — zero diagnostic value.
The wrapper itself is sound (clean runs, valid JSON, GPU, VRAM
released); the weights are degenerate. This tool MUST NOT be used in
any agent role. Manifest status set to rejected_benchmark.
Mid/high tiers now depend on the multimodal wrapper (pending).
