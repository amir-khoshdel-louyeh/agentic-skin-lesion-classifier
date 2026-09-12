# Phase 7 Benchmark — dermai-b0 (candidate, REJECTED)

Ready-made candidate: sgonzalez2000/dermai-efficientnet-b0 (Apache-2.0),
transformers-native EfficientNet-B0, 224px, HAM10000 7-class ISIC-code
head. 16.3 MB, no training performed.

- Images attempted: 24, scored: 21, failures: 0
- Accuracy: 0.7143 (vs triage CNN 0.7619, multimodal 0.5714)
- Avg confidence: 0.8866 (vs CNN 0.9931 — less overconfident, still hot)
- Avg seconds/image: 4.33 (model reload per CLI call; vs CNN 1.62)

## Per-class accuracy

- Actinic keratoses: 0.0 (n=3) — same blind spot as both existing tiers
- Basal cell carcinoma: 1.0 (n=3)
- Benign keratosis: 1.0 (n=3)
- Dermatofibroma: 1.0 (n=3)
- Melanocytic nevi: 1.0 (n=3)
- Melanoma: 0.0 (n=3, all called nevi) — worse than CNN's 1/3
- Vascular lesions: 1.0 (n=3)

## Failures

- none (one dev-time id2label int/str key bug, fixed before benchmarking)

## Role assignment: REJECTED (dominated, not degenerate)

Not mode collapse — 5/7 classes perfect — but strictly dominated by the
incumbent triage CNN: lower accuracy (0.71 < 0.76), same AK blind spot,
worse melanoma recall (0/3 vs 1/3), 3x latency. It buys nothing any
current tier lacks. This tool MUST NOT be used in any agent role.
Manifest status: rejected_benchmark. Wrapper kept (sound CLI, valid
JSON, VRAM released) in case a future ensemble wants a third vote.
