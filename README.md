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

### Step 1: Start the Core Infrastructure

Use the provided PowerShell launcher to start Ollama and the OpenClaw gateway safely.

1. Open `run_openclaw.ps1`
2. Update `$OPENCLAW_DIR` to your OpenClaw runtime folder
3. Run:

```powershell
.\run_openclaw.ps1
```

This script manages isolated background execution for Ollama and the OpenClaw Node gateway, and provides a small command shell for graceful shutdown.

### Step 2: Start the Local CNN Service

In the activated `.venv` session, start the FastAPI service:

```bash
python skin_agent.py
```

The server listens at `http://127.0.0.1:8000/analyze`.

### Step 2.5: Use the local CLI tool

You can also invoke the wrapper tool directly for prediction or validation:

```bash
python tools/skin_tools.py analyze --image data/sample_lesion.jpg --model-tier tier1_fast
python tools/skin_tools.py predict --image data/sample_lesion.jpg --model-tier tier1_fast
python tools/skin_tools.py validate --image data/sample_lesion.jpg
python tools/skin_tools.py status
python tools/skin_tools.py info
python tools/skin_tools.py list-tiers
```

`predict` is an alias for `analyze`, and `info` is an alias for `status`.

### Step 3: Execute the Pipeline

Run the top-level pipeline controller:

```bash
python run_pipeline.py
```

---

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
