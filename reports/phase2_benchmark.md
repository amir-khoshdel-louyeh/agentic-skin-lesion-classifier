# Phase 2 Benchmark — ham10000-cnn (tier1_fast)

- Images attempted: 24, scored: 21
- Accuracy: 0.7619
- Avg confidence: 0.9931
- Avg seconds/image: 1.62

## Per-class accuracy

- Actinic keratoses: 0.0 (n=3)
- Basal cell carcinoma: 1.0 (n=3)
- Benign keratosis: 1.0 (n=3)
- Dermatofibroma: 1.0 (n=3)
- Melanocytic nevi: 1.0 (n=3)
- Melanoma: 0.3333 (n=3)
- Vascular lesions: 1.0 (n=3)

## Failures

- none

## Role assignment
- Assigned: triage tier (tier1_fast) ONLY, never standalone decider.
- Reason: accuracy 0.76 with avg confidence 0.99 = systematically
  overconfident; Actinic keratoses 0/3 and Melanoma 1/3 despite ~0.99
  confidence. Every verdict from this tool MUST pass the closed-set
  entropy check (control/confidence.py); AK/Melanoma outputs are
  auto-borderline pending a second tier.
- Pending: mid/high tiers require the 4.4GB VLM + multimodal weights
  (download approval outstanding).
