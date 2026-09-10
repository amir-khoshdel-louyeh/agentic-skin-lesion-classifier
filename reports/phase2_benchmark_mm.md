# Phase 2 Benchmark — multimodal-fusion (tier2_mid)

- Images attempted: 24, scored: 21
- Accuracy: 0.5714
- Avg confidence: 0.6719
- Avg seconds/image: 2.36

## Per-class accuracy

- Actinic keratoses: 0.0 (n=3)
- Basal cell carcinoma: 0.6667 (n=3)
- Benign keratosis: 0.6667 (n=3)
- Dermatofibroma: 0.3333 (n=3)
- Melanocytic nevi: 0.6667 (n=3)
- Melanoma: 0.6667 (n=3)
- Vascular lesions: 1.0 (n=3)

## Failures

- none

## Role assignment
- Assigned: mid tier (tier2_mid), second opinion.
- Reason: accuracy 0.57 BELOW the CNN's 0.76, but avg confidence 0.67 =
  honestly calibrated (vs CNN's 0.99 overconfidence). Disagreement
  between the two tiers is a reliable borderline signal — exactly the
  "disagreement" flag in the verdict schema.
- Known blind spot (both tiers): Actinic keratoses 0/3. AK outputs from
  either tool are auto-borderline.
- Manifest: status ready, calibrated true (only tool so far).
