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
| **Tests** | 49 pytest tests: 47 fast deterministic tests, 1 checkpoint integration test, 1 opt-in live contract test |
| **Verified here** | 49-test suite passes (47 fast); a bounded live `qwen3:8b` round produced two control-side receipts and a `suspicious` decision (borderline Dermatofibroma 0.5594); no clinical validation |
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
├── artifacts/               # SHA-256 model/calibration manifest
├── tests/                   # 47 fast tests + 1 integration + 1 live opt-in
└── assets/                  # Demo and report screenshots
```

## Running

```bash
python3 -m venv .venv --system-site-packages
.venv/bin/pip install -r requirements.txt    # or requirements.lock
ollama pull qwen3:8b
openclaw models set "ollama/qwen3:8b"

for skill in quality brain diagnostic critic specialist chat; do
  openclaw skills install --force "./openclaw-skills/$skill"
done

.venv/bin/python main.py --record-index 0      # one round
.venv/bin/python main.py --no-interactive      # all records, sequentially
.venv/bin/python main.py --chat --record 0     # physician debate after a round
.venv/bin/python main.py --record-index 0 --agent-model ollama/qwen3:8b
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
- A bounded live round on 2026-11-16 with `--agent-model ollama/qwen3:8b` completed in **132 s** (11 LLM sessions). Control re-executed `ham10000_cnn` twice with a stable receipt (`7937135c…`) and filled `Dermatofibroma` **0.5594**; the referee returned **suspicious** because the verdict was below the 0.75 borderline threshold. `isic_vit` failed because its checkpoint is absent. Report: `report/round_0_20261116_100225.md`. The model wrote numbers into its envelope despite the omit instruction; Part A used only receipt-derived values.
- The local triage checkpoint was rebuilt from **9,087 training / 2,272 held-out validation** images (8 epochs, seed 42): **validation accuracy 0.4124**, fitted temperature **1.1**. Per-class recall ranges from 0.286 (benign keratosis) to 0.824 (vascular lesions); SCC was excluded. These are measurements for this compact rebuild, not the original uncommitted weights.
- `tools/ham10000_cnn.py --image dataset/ISIC2019_test/ISIC_0000002.jpg` returned `status: success` with `Dermatofibroma` at `confidence_score 0.5594`, proving the label/checkpoint contract now loads.
- `isic-vit` still has no `models/isic-vit-b16/best.pt`, so only the rebuilt triage CNN is executable; the report does not claim a second-model consensus.
- `.venv/bin/python -m pytest tests -q` runs the fast tier and reports **47 passed** (integration and live tests deselected). `RUN_LIVE_TESTS=1 .venv/bin/python -m pytest -m live -q` runs the opt-in live contract test. The checkpoint-backed receipt test uses the local triage checkpoint; no fast test starts a model.

## Tests

The 18 former standalone check scripts now run as ordinary pytest tests. They no longer execute assertions at import time or raise `SystemExit`. The default command runs the deterministic fast tier:

```bash
.venv/bin/python -m pytest tests -q
```

| Tier | Command | Requirements |
|---|---|---|
| Fast (default) | `.venv/bin/python -m pytest tests -q` | Python plus `pytest`, `pyyaml`, `pydantic`; no weights or services |
| Integration | `.venv/bin/python -m pytest tests -q -m integration` | Local triage checkpoint and model runtime |
| Live | `RUN_LIVE_TESTS=1 .venv/bin/python -m pytest tests -q -m live` | Local checkpoint, OpenClaw/Ollama stack where applicable |
| All local | `.venv/bin/python -m pytest tests -q -m "not live"` | Fast plus checkpoint integration |

CI (`.github/workflows/tests.yml`) runs the fast tier on Python 3.12 with only `pytest`, `pyyaml` and `pydantic`. The integration and live tiers are explicit local runs because they need model weights and, for live agent traffic, a running OpenClaw/Ollama stack.

Current collection is **49 tests**: 47 fast tests (17 of the 18 converted suites, fourteen deterministic receipt-verification tests, eight inference timeout/retry/model-override tests, one hermetic end-to-end round test, three artefact-integrity tests, one dependency-lock test, two adversarial tool-output tests and one observability test), one checkpoint-backed receipt integration test, and one opt-in live contract smoke test.

The receipt tests inject a fake tool executor, so they prove deterministically that an agent's claimed tool becomes evidence only after a successful control-side execution. Failed, invalid, mismatched, non-pool or unexecuted commands return `ran=False` with no class or confidence. Every successful execution also gets a content-addressed ledger entry: `receipt_id`, tool, exit code, duration in milliseconds, and SHA-256 hashes of the exact command, image bytes and stdout. The hash of tool + command + image + stdout + exit code makes the ID stable for identical executions; timing is recorded separately. The report includes the receipt table plus a hash ledger, and `audit.jsonl` carries the same rows under `receipts`.

Control-side tool JSON is parsed through the strict `ToolResult` Pydantic model. Numeric strings, booleans where numbers belong, out-of-range confidence, unknown class names and malformed flag lists are rejected as invalid tool JSON instead of being coerced; a successful diagnostic result must include both a known class and a confidence. The parser also rejects duplicate object keys, `NaN`/`Infinity`, and payloads over one million characters before validation. The same validation is applied to the quality gate, ABCDE/preprocess probes and the careful-path ensemble.

Each round writes `report/events.jsonl`: one JSON object per stage with `schema_version`, `run_id`, `event_id`, timestamp, stage, status, duration and safe categorical/numeric dimensions. Raw steers, prompts and model reasoning stay out of the event stream; the audit and report remain the explanatory record. Event writes are best effort and can never fail a round.

The checkpoint-backed receipt integration test executes the local triage checkpoint; the agent-side guard tests use fake CLI/HTTP backends and do not start Ollama or OpenClaw.

### Reproducible artefacts and dependencies

`artifacts/manifest.json` is a timestamp-free SHA-256 manifest of every calibration file and every present file under `models/`. Every round enforces it before any inference: tampered or missing calibration fails the round, and missing model files are warnings by default. Pass `--strict-models` to `main.py` (or use the verifier below) to make missing model files fatal:

```bash
.venv/bin/python main.py --record-index 0 --strict-models
.venv/bin/python tools/verify_artifacts.py
.venv/bin/python tools/verify_artifacts.py --strict-models
.venv/bin/python tools/verify_artifacts.py --write   # regenerate after a deliberate rebuild
```

`requirements.lock` pins the installed dependency closure of `requirements.txt` (direct plus transitive, evaluated for this platform), without unrelated host packages. Regenerate it after an intentional dependency change:

```bash
.venv/bin/python tools/lock_requirements.py
.venv/bin/pip install -r requirements.lock
```

The committed lock and manifest were generated on Python 3.14.10 / linux-x86_64. A different platform should regenerate the lock; model hashes remain valid across hosts.

### Inference guards

`control/inference_guard.py` enforces finite per-attempt timeouts, attempt caps and total budgets:

| Path | Default timeout | Attempts | Total budget | Environment prefix |
|---|---:|---:|---:|---|
| OpenClaw session | 600 s | 3 | 900 s | `OPENCLAW_INFERENCE_*` |
| Ollama VLM request | 600 s | 3 | 900 s | `OLLAMA_VLM_INFERENCE_*` |

Timeouts are not retried. OpenClaw retries only its explicit transient-error markers; the direct Ollama VLM path retries only HTTP 429/5xx responses. Oversized environment values are clamped (timeout ≤ 900 s, attempts ≤ 4, total ≤ 1,800 s).

## Design principles

- **The LLM is not the authority.** Authorisation, thresholds, receipts and the final report are deterministic.
- **Receipts over prose.** Every number in the report is re-executed control-side and linked to a content-addressed receipt with command, image and stdout hashes.
- **Two-part reports.** Part A is machine-decided; Part B is LLM-voiced explanation. Deleting Part B changes nothing factual.
- **Prompts are versioned.** System prompts are locked; physician-editable prompts are constrained and validated (`--doctor-check`).
- **Artefacts are hash-checked.** Calibration and model files are recorded in `artifacts/manifest.json`; the dependency closure is pinned in `requirements.lock`.
- **Operations are observable.** Every round emits structured `report/events.jsonl` records with a shared run id, stage timings and decision metadata.

## Tech stack

`Python` · `PyTorch` / `torchvision` / `transformers` · `OpenClaw` · `Ollama` · `PyYAML` · `pydantic` · Markdown reporting

## Disclaimer

This is not a medical device. It must not be used for diagnosis, triage or treatment decisions. All outputs are research artefacts.

## Author

**Amir Khoshdel Louyeh**
[GitHub](https://github.com/amir-khoshdel-louyeh) · [LinkedIn](https://www.linkedin.com/in/amir-khoshdel-louyeh)
