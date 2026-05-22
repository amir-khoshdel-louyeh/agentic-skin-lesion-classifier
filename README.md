# 🧠 Agentic Skin Lesion Classifier
A hybrid AI system combining:
- 🧩 OpenClaw (agent orchestration)
- 🦙 Ollama (local LLM reasoning)
- 🧬 CNN models (medical image classification)
- 🔬 Federated-ready architecture (future extension)

This system separates:
- reasoning (LLM)
- perception (CNN)
- orchestration (agent layer)

---

# 🏗️ System Architecture

```
User Image
   ↓
OpenClaw Agent (qwen2.5 / llama3.2)
   ↓ tool call
Python CNN API (FastAPI)
   ↓
PyTorch pretrained models (ISIC / HAM10000)
   ↓
Prediction JSON
   ↓
LLM explanation + report
```

---

# ⚙️ Prerequisites

## System requirements
- Windows 10/11 or WSL2 (recommended)
- Python 3.10–3.11
- Node.js (for OpenClaw)
- Ollama installed locally

---

## Install Ollama

https://ollama.com/download

Pull a model:

```bash
ollama pull qwen2.5:7b
```

(Optional vision testing)

```bash
ollama pull llama3.2-vision
```

---

## Install OpenClaw

```bash
npm install -g openclaw
```

Verify:

```bash
openclaw --version
```

---

# 🧠 OpenClaw Setup

Run gateway:

```powershell
openclaw gateway run
```

Open dashboard:

```powershell
openclaw dashboard
```

Approve device if required:

```powershell
openclaw devices list
openclaw devices approve <device-id>
```

---

# 🐍 Python Environment (CNN Layer Only)

Create venv:

```bash
python -m venv .venv
```

Activate:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 📦 requirements.txt

```txt
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
```

---

# 🚀 Running the System

## 1. Start OpenClaw + Ollama

```powershell
run_openclaw.ps1
```

---

## 2. Start CNN API

```bash
python run_cnn.py
```

Runs:

http://127.0.0.1:8000/predict

---

## 3. Test CNN API

```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@test.jpg"
```

---

# 🧠 Agent Behavior

OpenClaw agent:
- decides when to call CNN tool
- does NOT classify directly
- only interprets results

---

# ⚠️ Critical Rule

❌ Wrong: LLM predicts cancer directly

✔ Correct: CNN predicts → LLM explains

---

# 🧬 Dataset

Supported:
- ISIC 2019
- HAM10000

Structure:

```
dataset/
 ├── images/
 ├── metadata.csv
 └── labels.csv
```

---

# 🔬 Extensibility

Future upgrades:
- Federated learning (Flower)
- Multi-agent voting system
- SHAP / GradCAM explainability
- Multi-hospital architecture

---

# 🚨 Debugging

OpenClaw:
```bash
openclaw gateway status
```

Ollama:
```bash
ollama ps
```

CNN API:
```bash
curl http://127.0.0.1:8000/docs
```

---

# 🧠 Recommended Stack

| Layer | Tech |
|------|------|
| Orchestration | OpenClaw |
| LLM | Ollama |
| Vision (optional) | llama3.2-vision |
| CNN inference | PyTorch |
| API layer | FastAPI |

---

# 📜 License & Safety

⚠️ Research use only
⚠️ Not clinically validated
⚠️ Not for medical diagnosis

