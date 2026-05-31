# 🧠 Agentic Skin Lesion Classifier

A hybrid AI system combining:
- 🧩 OpenClaw (agent orchestration)
- 🦙 Ollama (local LLM reasoning & deterministic medical analysis)
- 🧬 CNN models (medical image classification via PyTorch + FastAPI)
- 🔬 Federated-ready architecture (future extension)

This system separates **reasoning** (LLM), **perception** (CNN), and **orchestration** (agent layer).

---

## 🏗️ System Architecture

```
     [ User Image ]
           ↓
[ Python CNN API (FastAPI) ] ──(Inference)──> [ PyTorch Pretrained Models (ISIC/HAM10000) ]
           ↓
      [ Prediction JSON ]
           ↓
[ Ollama Local LLM (Qwen 2.5) ] ──(Deterministic Parameters)──> [ Low-Temperature Medical Analysis ]
           ↓
      [ Final Report ]
```

---

## ⚙️ Prerequisites

### System Requirements
- Windows 10/11 or WSL2
- Python 3.10–3.11
- Node.js & npm (for OpenClaw gateway runtime)
- Ollama installed locally

---

## 1. Install & Pull LLM Models

1. Install Ollama from https://ollama.com/download
2. Pull the primary reasoning model:

```bash
ollama pull qwen2.5:7b
```

Optional vision model for testing:

```bash
ollama pull llama3.2-vision
```

---

## 2. OpenClaw Gateway Setup

Ensure Node.js is installed, then install the OpenClaw core globally:

```bash
npm install -g openclaw
```

---

## 3. Python Environment (CNN Layer)

Create and activate the virtual environment inside the repository:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### requirements.txt

```text
numpy>=1.26,<3.0
pandas>=2.2,<3.0
pillow>=10.3,<11.0
torch>=2.2
torchvision>=0.17
fastapi>=0.110
uvicorn>=0.27
scikit-learn>=1.4
opencv-python>=4.9
albumentations>=1.4
requests>=2.31.0
```

---

## 🚀 Running the System

### Step 1: Start the OpenClaw runtime

Start the OpenClaw gateway or node service:

```powershell
openclaw gateway run --force
```

or, if the node service is installed:

```powershell
openclaw node start
```

### Step 2: Install the local skill

Install the lightweight local tool so OpenClaw can execute it:

```powershell
openclaw --no-color skills install --force ./openclaw-skills/skin-lesion-fast
```

### Step 3: Run the OpenClaw prompt queue

The file `prompt.txt` contains one record per line. Each record is JSON and must include:

- `image_path` — local image file path
- `prompt` — the task prompt for OpenClaw
- `metadata` — optional JSON object with patient or case details

Run the queue processor with:

```bash
python skin_agent.py
```

If you want to send a single record by index:

```bash
python skin_agent.py --record-index 0
```

This script sends records to OpenClaw one at a time. OpenClaw will decide which available tool(s) to use based on the prompt and the current toolset.

#### OpenClaw agent requirement

`skin_agent.py` requires OpenClaw to be installed and available on your PATH. It uses the OpenClaw CLI to run prompts against your configured local agent.

If the command fails, you may need to start the OpenClaw node/gateway first.

Also make sure the local `skin-lesion-fast` skill is installed in OpenClaw:

```powershell
openclaw --no-color skills install ./openclaw-skills/skin-lesion-fast
```

Then run the queue processor:

```bash
python skin_agent.py
```

### Developer note: local CLI commands

The local tool scripts in `tools/` are provided for skill development and validation only. The system should always run via OpenClaw in normal operation.

## 🧠 Deterministic LLM Logic

### Clinical Guideline

- ❌ Incorrect: Letting the LLM predict cancer probabilities directly.
- ✔️ Correct: Let the CNN handle visual perception, then give the LLM numeric results for calibrated analysis.

The pipeline is designed to avoid hallucinations by using deterministic prompt structures and low-temperature settings.

Example payload:

```python
payload = {
    "model": "qwen2.5:7b",
    "prompt": prompt_for_llm,
    "system": "You are an expert dermatopathologist. Provide a highly professional, calm, reassuring, and scientifically accurate response in English. Clearly state that these are AI predictions.",
    "stream": False,
    "options": {
        "temperature": 0.1,
        "top_p": 0.9,
    }
}
```

---

## 🧬 Data & Extensibility

### Dataset Structure

Supported source distributions include ISIC 2019 and HAM10000:

```text
dataset/
 ├── images/
 ├── metadata.csv
 └── labels.csv
```

### Future Upgrades

- Federated infrastructure via Flower or similar frameworks
- Explainability layers with SHAP / GradCAM visualization
- Multi-agent consensus loops combining transformer and CNN perception

---

## 🚨 Debugging Guide

| Issue | Verification | Target | Notes |
|---|---|---|---|
| WinError 10061 | `ollama ps` or verify port `11434` | Ollama Core Engine | Ensure Ollama is running |
| 404 Not Found | Query endpoint layout | OpenClaw / Ollama API | Confirm correct `/api/generate` URL |
| Invalid config | Validate `~/.openclaw/openclaw.json` | OpenClaw gateway settings | Check path and syntax |

---

## 📜 License & Safety

This system is for academic evaluation only. It is not clinically validated, not certified for medical diagnostics, and should never replace examination or biopsy review by a licensed dermatologist.
