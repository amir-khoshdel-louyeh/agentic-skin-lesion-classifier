# Phase 6 Benchmark — notify (notification)

Desktop toast on round completion/failure. Tries `notify-send`,
falls back to console bell + stderr line when headless. Never fails
the caller (exit 0 with `delivered: true/false`).

- Messages attempted: 24 (one per prompt record: 18 info / 3 warning /
  3 error), receipts valid: 24
- Avg seconds/message: 0.04
- Fallback path (no `notify-send` on PATH): receipt ok,
  `backend: console-fallback`, exit 0
- Empty message: clean `error` contract, exit 1

## Failures

- none

## Role assignment / wiring

- Assigned: notification tier, control-side hook (NOT an agent tool —
  agents never call it; it carries no image evidence so it stays out
  of receipt verification by design).
- Wired in this step: `orchestrator.run_round` pings on completion
  (`ping()` in orchestrator.py, all errors swallowed); `main.py`
  sweep pings `error` on per-record failure. A missing desktop can
  never break a round — verified via the fallback path above.
- Manifest: category `notification`, requires `message`, produces
  `notification_receipt`. Status ready, calibrated false, 0.0 GB VRAM.
