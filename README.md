# Agentic Skin Lesion Classifier

An agent-driven dermatology screening system that orchestrates multiple CNN models through OpenClaw to balance inference speed, diagnostic confidence, and workflow flexibility.

> **Portfolio Project** — Demonstrates Agentic AI orchestration, Computer Vision workflows, multi-model inference strategies, and structured reporting pipelines.

---

## System Demonstration

> Replace the placeholders below with actual screenshots or GIFs of the app execution and results.

### System Workflow

```text
Input image
      │
      ▼
Prompt records in prompt.txt
      │
      ▼
OpenClaw agent via skin_agent.py
      │
      ▼
Tool selection:
  - tools/skin_lesion_fast.py
  - tools/skin_lesion_mid.py
      │
      ▼
Model inference
      │
      ▼
Structured JSON output
      │
      ▼
Final Markdown report
```

### Agent Execution Demo

```text
[ INSERT CLI DEMO GIF HERE ]
```

### Example Report

```text
[ INSERT GENERATED REPORT SCREENSHOT HERE ]
```

---

## Highlights

* Agent-driven model orchestration using OpenClaw
* Multi-model inference pipeline with specialized screening tiers
* EfficientNet-B0 for low-latency first-pass analysis
* EfficientNet-B4 for balanced accuracy and confidence
* Structured JSON outputs for downstream automation
* Prompt-driven execution and tool selection
* Modular architecture designed for extensibility

### Built With

Python • PyTorch • timm • OpenClaw • Computer Vision • Agentic AI • Ollama • OpenClaw

---

## Why This Project Matters

Most skin lesion classification projects focus on a single model and a single prediction output.

This project explores a different approach:

Instead of relying on one classifier, multiple diagnostic tools are exposed to an AI agent that can select the most appropriate inference path depending on task requirements.

The goal is not only image classification but also demonstrating how agentic systems can orchestrate specialized AI tools, manage inference workflows, and generate structured outputs suitable for future clinical decision-support pipelines.

This project showcases concepts increasingly relevant to modern AI engineering:

* Agentic AI
* Tool orchestration
* Multi-model systems
* Explainable workflows
* Modular AI architectures

---

# Overview

This project provides a local skin lesion screening workflow that combines Computer Vision models with OpenClaw-based agent orchestration.

The system allows image-based screening through multiple model tiers and supports prompt-driven execution via CLI tools and OpenClaw skills.

The primary objective is to demonstrate how an AI agent can coordinate specialized diagnostic tools through clear command contracts and structured outputs.

---

# Problem Statement

Traditional skin lesion classification workflows often suffer from one or more of the following limitations:

* Dependence on a single model regardless of context
* Lack of clear escalation paths between fast and accurate models
* Tight coupling between inference and orchestration logic
* Limited support for structured downstream processing

As a result, extending or adapting these systems becomes increasingly difficult as complexity grows.

This project addresses these limitations through a modular, agent-oriented architecture that separates orchestration, inference, and reporting responsibilities.

---

# Solution Approach

The solution consists of three primary layers:

### Inference Layer

Provides specialized diagnostic tools:

* Fast Screening Tool (`EfficientNet-B0`)
* Balanced Screening Tool (`EfficientNet-B4`)

### Orchestration Layer

Provides agent-based tool selection and execution:

* OpenClaw Skills
* Prompt Processing
* Command Routing

### Reporting Layer

Provides structured outputs:

* JSON Results
* Confidence Scores
* Markdown Reports

Workflow:

```text
Input Image
      │
      ▼
OpenClaw Agent
      │
      ▼
Tool Selection
      │
      ├── Fast Model (B0)
      │
      └── Mid Model (B4)
      │
      ▼
Model Inference
      │
      ▼
Structured JSON Output
      │
      ▼
Final Report
```

---

# Demo

## Running the Agent

```bash
python skin_agent.py --record-index 0
```

---

## Direct Tool Invocation

Fast model:

```bash
python tools/skin_lesion_fast.py \
  --image path/to/image.jpg \
  --metadata '{"age":45,"sex":"female"}'
```

Balanced model:

```bash
python tools/skin_lesion_mid.py \
  --image path/to/image.jpg \
  --metadata '{"age":62,"sex":"male"}'
```

---

## OpenClaw Skill Installation

```bash
openclaw --no-color skills install --force ./openclaw-skills/skin-lesion-fast

openclaw --no-color skills install --force ./openclaw-skills/skin-lesion-mid
```

---

## Example Output

> Replace with a real output screenshot.

```text
[ INSERT OUTPUT SCREENSHOT HERE ]
```

---

# Features

* Tiered diagnostic workflow
* Agent-based tool selection
* Prompt-driven execution
* Structured JSON inference output
* Confidence scoring
* Metadata-aware processing
* OpenClaw skill integration
* Extensible model architecture
* Local-first deployment
* Modular CLI tooling

---

# Results & Metrics

> Replace all placeholder values below with actual evaluation results.

| Metric         | Fast Model (B0) | Mid Model (B4) |
| -------------- | --------------- | -------------- |
| Accuracy       | XX.X%           | XX.X%          |
| Precision      | XX.X%           | XX.X%          |
| Recall         | XX.X%           | XX.X%          |
| F1 Score       | XX.X            | XX.X           |
| Inference Time | XX ms           | XX ms          |

### Dataset

```text
[ INSERT DATASET INFORMATION ]
```

Example:

```text
HAM10000
7 skin lesion categories
10,015 dermatoscopic images
```

---

# Architecture

## High-Level Architecture

> Replace with an architecture diagram image.

```text
[ INSERT ARCHITECTURE DIAGRAM HERE ]
```

---

## Components

### Fast Screening Tool

Location:

```text
tools/skin_lesion_fast.py
```

Responsibilities:

* Fast initial classification
* Low-latency inference
* Efficient resource usage

---

### Balanced Screening Tool

Location:

```text
tools/skin_lesion_mid.py
```

Responsibilities:

* Higher diagnostic confidence
* Improved feature extraction
* More computationally intensive inference

---

### Agent Layer

Location:

```text
skin_agent.py
```

Responsibilities:

* Prompt handling
* Tool selection
* Command execution
* Output aggregation

---

### Skill Layer

Location:

```text
openclaw-skills/
```

Responsibilities:

* Agent instructions
* Tool contracts
* Execution metadata

---

# Technical Highlights

* Agentic AI workflow design
* Multi-model orchestration
* Modular CLI architecture
* Structured machine-readable outputs
* Prompt-driven execution pipeline
* Extensible skill-based architecture
* Metadata-aware classification
* Reusable tool contracts
* OpenClaw integration
* Local-first AI deployment

---

# Engineering Decisions

## Why Agentic Architecture?

Instead of embedding all logic into a single application, responsibilities are separated across tools and orchestration layers.

Benefits:

* Easier extensibility
* Better maintainability
* Improved tool reuse
* Clear separation of concerns

---

## Why EfficientNet?

EfficientNet offers a strong balance between performance and computational efficiency.

### EfficientNet-B0

Chosen for:

* Fast inference
* Low resource requirements
* Rapid first-pass screening

### EfficientNet-B4

Chosen for:

* Improved representation quality
* Better classification performance
* Higher diagnostic confidence

---

## Why OpenClaw?

OpenClaw enables orchestration to remain independent from model implementation.

Benefits:

* Tool abstraction
* Modular workflows
* Prompt-based routing
* Future scalability

---

# Challenges & Lessons Learned

## Challenge 1: Tiered Model Coordination

Designing meaningful separation between fast and balanced screening paths required clear execution boundaries and tool responsibilities.

### Solution

* Dedicated command contracts
* Independent tool interfaces
* Explicit model roles

---

## Challenge 2: Agent-to-Tool Communication

Reliable orchestration depends on predictable tool behavior and outputs.

### Solution

* Structured JSON responses
* Standardized input formats
* Consistent CLI interfaces

---

## Challenge 3: Metadata Handling

User-provided metadata can vary significantly in structure and completeness.

### Solution

* Validation layers
* Safe parsing logic
* Fallback handling strategies

---

# Lessons Learned

Through this project I strengthened my understanding of:

* Agentic AI systems
* Tool orchestration
* Multi-model architectures
* Computer Vision deployment
* CLI application design
* Structured AI workflows
* Software modularity
* AI system extensibility

---

# Repository Structure

```text
.
├── openclaw-skills/
│   ├── skin-lesion-fast/
│   │   └── SKILL.md
│   └── skin-lesion-mid/
│       └── SKILL.md
│
├── tools/
│   ├── skin_lesion_fast.py
│   └── skin_lesion_mid.py
│
├── skin_agent.py
├── prompt.txt
├── tool_manifest.md
├── plan.md
├── requirements.txt
└── README.md
```

---

# Getting Started

## Clone Repository

```bash
git clone https://github.com/your-username/agentic-skin-lesion-classifier.git

cd agentic-skin-lesion-classifier
```

---

## Create Virtual Environment

Windows:

```bash
py -3.11 -m venv .venv

.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Install OpenClaw Skills

```bash
openclaw --no-color skills install --force ./openclaw-skills/skin-lesion-fast

openclaw --no-color skills install --force ./openclaw-skills/skin-lesion-mid
```

---

## Run Demo

```bash
python skin_agent.py --record-index 0
```

---

# Testing

Currently, this repository does not include an automated test suite.

Manual verification:

```bash
python tools/skin_lesion_fast.py --image path/to/image.jpg

python tools/skin_lesion_mid.py --image path/to/image.jpg

python skin_agent.py --record-index 0
```

### Expected Outcome

* Successful image validation
* Model inference execution
* Structured JSON output
* Generated report

---

# Future Improvements

* Add deep-tier specialist models
* Add ensemble decision-making
* Add automated evaluation pipelines
* Add unit and integration testing
* Add GitHub Actions CI/CD
* Add FastAPI service layer
* Add web-based interface
* Add explainability visualizations (Grad-CAM)
* Add confidence calibration workflows
* Add model monitoring

---

# Author

## Amir Khoshdel Louyeh

Computer Science Student

### Interests

* Artificial Intelligence
* Agentic AI
* Machine Learning
* Software Engineering
* High Performance Computing

### Connect

GitHub:
https://github.com/amir-khoshdel-louyeh

LinkedIn:
[INSERT LINKEDIN URL]

Portfolio:
[INSERT PORTFOLIO URL]

---

## Disclaimer

This project is intended for educational and research purposes only.

It is not a medical device and should not be used for clinical diagnosis or treatment decisions.
