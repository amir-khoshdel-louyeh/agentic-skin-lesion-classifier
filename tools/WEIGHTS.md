# Weight Provenance (Phase 2)
Weights live under `models/` (git-ignored). This file records WHAT was
fetched, WHERE from, and its integrity hash. No training performed.

## models/derm-cnn/ — light CNN triage tier [DOWNLOADED]
- Source: https://huggingface.co/iamhmh/derm-cnn-ham10000
- License: CC BY-NC 4.0 (weights) / MIT (code) — non-commercial only.
- Files: model.pth (5,119,928 bytes), model.py (arch SkinCNN), labels.json
- sha256:
  - model.pth:   c720b1a42a5c99ff88a858e71fff5587d0c9c8c21a6c7d6997c6c048f90c7551
  - model.py:    5934df3560af9f84e233c5873b2df20dd4c490354e6bc309b0cdfbe332b1d853
  - labels.json: f91cc58878b2ebb5cc300e12a24784e25e87b28e41bfcb037cc212fa9e45cdda
- Label map (ISIC codes): 0 akiec, 1 bcc, 2 bkl, 3 df, 4 nv, 5 vasc, 6 mel
- Caveat: 28×28 input, claimed 0.99 accuracy looks optimistic —
  must be measured on local 24 images before role assignment.

## Candidates [PENDING download approval]
- abaryan/DrDiag_qwen2vl_Ham10000 — 2B VLM, ~4.4 GB safetensors.
  https://huggingface.co/abaryan/DrDiag_qwen2vl_Ham10000
- cesaraha/ham10000-multimodal-skin-lesion — needs author repo code
  (architecture not self-contained), metadata required at inference.
  https://huggingface.co/cesaraha/ham10000-multimodal-skin-lesion
