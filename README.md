# Agentic Skin Lesion Classifier

A local, agent-orchestrated dermoscopy screening workflow: specialised CNN tools, an LLM supervisor loop, an adversarial critic and a deterministic referee that produces a receipt-backed two-part report.

![Agent execution](assets/agent-demo.png)

`Python` · `PyTorch` · `OpenClaw` · `Ollama` · Agentic AI / medical imaging · Research prototype

## At a glance

| | |
|---|---|
| **Idea** | Let an LLM agent choose and sequence specialised vision tools, while deterministic code owns the numbers |
| **Tools** | EfficientNet-B0 (fast), EfficientNet-B4 (balanced), ViT-Large, locally trained 28×28 triage CNN, DermAI |
| **Orchestration** | Quality → brain loop (up to 7 sessions) → critic → specialists → referee |
| **Outputs** | Structured JSON, confidence, Markdown report with re-executed receipts |
| **Tests** | 18 script-style test files; **not runnable under `pytest`** (they call `SystemExit` at import) |
| **Verified here** | Triage model rebuilt from ISIC 2019 and executes successfully; the LLM round remains inconclusive because the agent did not issue a tool command |
| **Role** | Solo — agent design, tool contracts, orchestration, reporting |

## Overview

Single-model skin-lesion classifiers produce one number and no audit trail. This project treats the classifier as one tool among several and asks a different question: can an agent coordinate fast and slow models, challenge its own conclusion, and leave a report whose every numeric claim can be re-derived?

The design separates three responsibilities. **LLM agents** decide quality, route, tools and when to stop. **Deterministic code** owns thresholds, VRAM limits, receipt re-execution and the final tally. **The referee** writes the machine-decided part of the report without an LLM. The result is an agentic system that can be wrong in its reasoning but cannot silently invent a probability.

## Pipeline

```text
Input image
   │
   ▼
1. Quality agent ── pass / cleaned / flagged (never diagnoses)
   │
   ▼
2. Brain loop ── supervisor steers up to 7 sessions;
   each session sees the full tool pool and must cite receipts
   │
   ▼
3. Critic ── one post-loop objection with citations, or silence
   │
   ▼
4. Specialists ── max 2 advocate briefs when ≥2 classes are plausible
   │
   ▼
5. Referee (plain code, no LLM) ── counts, thresholds, report
   │
   ▼
Receipt re-execution control-side → two-part Markdown report
```

Every tool call returns a receipt. The control side re-executes those calls to confirm the values that appear in the report, so the narrative cannot drift from the measured output.

## Screenshots

| Structured output | Example report |
|---|---|
| ![](assets/json-output.png) | ![](assets/example-report.png) |

A full end-to-end recording is available as [`assets/system-demonstration.gif`](assets/system-demonstration.gif) (10 MB).

## Tool layer

| Tool | Model | Intended role |
|---|---|---|
| Fast screening | EfficientNet-B0 | Low-latency first pass |
| Balanced screening | EfficientNet-B4 | Accuracy/confidence trade-off |
| High-capacity | ViT-Large | Difficult or ambiguous cases |
| Triage CNN | Compact 28×28 PyTorch CNN | Locally trained seven-class baseline |
| DermAI | CNN baseline | Independent second opinion |
| VLM | Qwen2-VL | **Rejected after benchmarking** — retained for reference only |

Calibration profiles for each model live in `calibration/`, with a summary in `calibration/SUMMARY.md`. The agent receives calibrated probabilities rather than raw logits.

## Repository structure

```text
agentic-skin-lesion-classifier/
├── main.py                  # CLI entry point
├── orchestrator.py          # Round orchestration
├── skin_agent.py            # Agent-facing facade
├── agents/runner.py         # LLM session runner
├── control/                 # Routing, quality, critic, referee, receipts
├── tools/                   # CNN/VLM tool implementations and manifest
├── prompts/                 # Locked system prompts and doctor-editable prompts
├── openclaw-skills/         # Installed OpenClaw skills
├── calibration/             # Per-model calibration profiles
├── tests/                   # 18 standalone test scripts
└── assets/                  # Demo and report screenshots
```

## Running

```bash
python3 -m venv .venv --system-site-packages
.venv/bin/pip install -r requirements.txt
ollama pull qwen3:8b
openclaw models set "ollama/qwen3:8b"

openclaw skills install --force ./openclaw-skills/quality
# … brain, diagnostic, critic, specialist, chat

.venv/bin/python main.py --record-index 0      # one round
.venv/bin/python main.py --no-interactive      # all records, sequentially
.venv/bin/python main.py --chat --record 0     # physician debate after a round
```

Reports are written to `report/` (git-ignored per-round artefacts). Design notes are in `proposal.tmp` and `plan.tmp`; the tool contract is `tools/manifest.yaml`.

### Dataset and local triage model

The original development weights were never committed (`models/` is git-ignored), so the following rebuilds the seven-class 28×28 checkpoint from the local ISIC 2019 tree:

```bash
cp -al ../DataSet/. dataset/ISIC2019_full/
.venv/bin/python tools/train_triage_model.py --max-per-class 2500 --epochs 8
```

The trainer writes `models/derm-cnn/model.py`, `labels.json`, `model.pth`, `calibration.json` and `metrics.json`. It excludes SCC because the tool's `ISIC_TO_CLASS` contract has no SCC mapping. The fitted calibration is read by `tools/ham10000_cnn.py`.

## Verification

- `openclaw --version` reports `2026.7.1-2`; the default agent model was switched from `ollama/qwen3:8b` to `ollama/qwen2.5-coder:14b` for the second orchestration attempt.
- `tools/quality_gate.py --image dataset/ISIC2019_test/ISIC_0000000.jpg` returned `status: success` with `blur_variance 58.21`, `mean_brightness 146.01`, `clipped_fraction 0.0044`, and the `blurry` uncertainty flag.
- The first round completed in 4m38s on `ISIC_0000002.jpg` (quality pass: blur variance 484.86, mean brightness 150.6, clipped fraction 0.0109) and wrote `report/round_0_20261113_165727.md` plus `audit.jsonl`. A later round wrote `report/round_0_20261113_171943.md` after five sessions, but the agent issued no tool command, so both rounds returned `inconclusive — no receipted verdicts`.
- A rerun with `qwen2.5-coder:14b` reached the 700-second timeout without producing a new report; that model was too slow for the full seven-session loop in this environment.
- The local triage checkpoint was rebuilt from **9,087 training / 2,272 held-out validation** images (8 epochs, seed 42): **validation accuracy 0.4124**, fitted temperature **1.1**. Per-class recall ranges from 0.286 (benign keratosis) to 0.824 (vascular lesions); SCC was excluded. These are measurements for this compact rebuild, not the original uncommitted weights.
- `tools/ham10000_cnn.py --image dataset/ISIC2019_test/ISIC_0000002.jpg` returned `status: success` with `Dermatofibroma` at `confidence_score 0.5594`, proving the label/checkpoint contract now loads.
- `isic-vit` still has no `models/isic-vit-b16/best.pt`, so only the rebuilt triage CNN is executable; the report does not claim a second-model consensus.

## Tests

The `tests/` directory contains **18 standalone scripts**, not pytest modules. Several execute assertions at import time and call `SystemExit`, so `pytest tests` fails during collection:

```text
INTERNALERROR ... SystemExit: 0
no tests ran
```

Run them individually instead:

```bash
for t in tests/test_*.py; do echo "== $t"; python3 "$t" || break; done
```

Converting them to ordinary pytest functions (or guarding the entry points) is the cleanest next step. The scripts cover routing, receipts, critic behaviour, grounding, safety, single-writer numbers, specialists and two-part report generation.

## Design principles

- **The LLM is not the authority.** Authorisation, thresholds, receipts and the final report are deterministic.
- **Receipts over prose.** Every number in the report is re-executed control-side.
- **Two-part reports.** Part A is machine-decided; Part B is LLM-voiced explanation. Deleting Part B changes nothing factual.
- **Prompts are versioned.** System prompts are locked; physician-editable prompts are constrained and validated (`--doctor-check`).

## Tech stack

`Python` · `PyTorch` / `torchvision` / `transformers` · `OpenClaw` · `Ollama` · `PyYAML` · `pydantic` · Markdown reporting

## Disclaimer

This is not a medical device. It must not be used for diagnosis, triage or treatment decisions. All outputs are research artefacts.

## Author

**Amir Khoshdel Louyeh**
[GitHub](https://github.com/amir-khoshdel-louyeh) · [LinkedIn](https://www.linkedin.com/in/amir-khoshdel-louyeh)
