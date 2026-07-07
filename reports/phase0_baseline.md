# Phase 0 Baseline Report
Date: 2026-09-25 · Scope: environment only, no code changes.

## Hardware
- CPU: 24 cores · RAM: 15 GB · Disk free: ~815 GB
- GPU: NVIDIA GeForce RTX 5070 Laptop, 8 GB, sm_120 (12, 0)
- torch 2.14.0+cu130: `cuda=True`, capability (12, 0) ✓

## Runtime
- `.venv` (system-site-packages): Python 3.14.7, yaml 6.0.3, pydantic 2.12.5
  (note: pinned pydantic==2.7.1 cannot build on Python 3.14)
- Ollama: serving on localhost:11434
  - `qwen3:8b` (5.2 GB) — default
  - `qwen2.5-coder:14b` (9.0 GB) — fallback
- OpenClaw 2026.7.1-2: default `ollama/qwen3:8b`, provider registered
  with baseUrl http://localhost:11434, no API key required.
- Skills installed and ready: skin-lesion-fast/mid/high (re-installed
  from cleaned repo).

## Smoke Tests
- `openclaw agent --local` trivial reply: **SMOKE_OK in ~28 s**
  (vs ~176 s for 14B on CPU) — GPU residency confirmed.
- Direct tool call (`tools/skin_lesion_fast.py` on ISIC_0000002.jpg):
  expected error — `models/offline_fast` weights absent.
  Plumbing works; weights are Phase 2 scope.

## Open Items for Phase 1+
- Ready-made HAM10000 weights download + benchmark (Phase 2).
- `prompts/` two-layer structure, verdict schema, path checks (Phase 1).
