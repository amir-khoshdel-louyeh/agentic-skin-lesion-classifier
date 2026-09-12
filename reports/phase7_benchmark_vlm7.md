# Phase 7 Benchmark — vlm-ollama / qwen3-vl:8b (candidate, REJECTED)

Stronger small VLM candidate via Ollama (8.8B, Q4_K_M, 6.1 GB):
constrained 7-class prompt, JSON mode, temperature 0, confidence null
by design (closed-set protocol would calibrate later). Sequential GPU:
keep_alive=0 unloads the VLM after every call.

- Images attempted: 24, succeeded: 24, failures: 0 (after wrapper fix)
- Scored vs ground truth: 21, accuracy: 0.3333
- Avg seconds/image: ~8 (Ollama-managed VRAM, no PyTorch budget to check)

## Per-class accuracy

- Actinic keratoses: 1.0 (n=3)
- Basal cell carcinoma: 0.0 (n=3)
- Benign keratosis: 0.0 (n=3)
- Dermatofibroma: 0.0 (n=3)
- Melanocytic nevi: 0.3333 (n=3)
- Melanoma: 0.0 (n=3)
- Vascular lesions: 1.0 (n=3)

## Failures (wrapper, fixed before scoring)

- 6/24 first-pass replies came back EMPTY: num_predict=200 starved the
  thinking trace (content empty, no error). Fixed with num_predict=1024;
  all 6 then classified first-attempt. Second finding: think=false
  returns nothing on this model — thinking must stay ON.
- One repair retry + substring fallback added per proposal.tmp
  Section 7, mirroring the agent runner's own policy.

## Role assignment: REJECTED

Strictly better than the 2B VLM (0.14, total melanoma collapse): this
one spreads predictions over 5 classes and takes AK + vascular cleanly.
But 0.33 is far below the mid tier (0.57), let alone a high tier — and
Melanoma 0/3 plus BCC 0/3 means it misses exactly the malignancies a
high tier must catch. A high tier that is wrong 2/3 of the time is
worse than no high tier: it would launder guesses through the most
trusted slot. This tool MUST NOT be used in any agent role. Manifest
status: rejected_benchmark.
