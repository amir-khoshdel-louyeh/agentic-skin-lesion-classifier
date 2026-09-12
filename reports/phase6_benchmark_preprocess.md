# Phase 6 Benchmark — preprocess (preprocessing slot)

Image-in → image-out activation of the slot reserved in proposal.tmp
Section 5. CPU-only OpenCV, deterministic, no weights.

- Images attempted: 24, converted: 24, failures: 0
- Avg seconds/image: 0.19 (hair) / 0.42 (hair+denoise)

## Quality-gate pass flips (before → after)

- hair only: 0 flips (8/24 frames had real hairs inpainted, 52–1179 px;
  clean frames pass through nearly untouched)
- hair+denoise: 7 pass → fail flips, 0 fail → pass flips

Denoising smooths texture, which collapses the Laplacian-variance blur
metric — it makes the gate grade WORSE, not better. First calibration
(black-hat threshold 10) was degenerate too: it "removed" 3–10% of
pixels on clean frames (lesion texture, not hair). Fixed with threshold
25 + long-thin component filter.

## Failures

- none

## Role assignment / chain rule

- Assigned: preprocessing tier, chained in the agent loop on
  quality-gate failure per plan.tmp: gate flags a frame → agent runs
  `python tools/preprocess.py --image <in> --output <cleaned> [--steps
  ...]` → re-runs quality-gate on the cleaned image → continues the
  round with the cleaned path and cites both receipts.
- Default `--steps hair`: do-no-harm (0 flips on 24 images).
- `denoise` is OPT-IN for visibly grainy inputs only, with the
  documented cost (blur metric drops; expect a `blurry` flag after).
- Never silently replace the record image: the cleaned path is a new
  artifact, and both receipts stay in the verdict envelope.
- Manifest: category `preprocessing`, produces `cleaned_image`,
  discovered via `capable_of` with zero orchestrator/agent code changes.
  Status ready, calibrated false, 0.0 GB VRAM.
