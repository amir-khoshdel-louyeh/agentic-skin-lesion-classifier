# **Agentic Skin Lesion Classifier**

An agent-driven dermatology screening system that orchestrates multiple CNN models through OpenClaw to balance inference speed, diagnostic confidence, and workflow flexibility.  
**Portfolio Project** — Demonstrates Agentic AI orchestration, Computer Vision workflows, multi-model inference strategies, and structured reporting pipelines.

## **System Demonstration**

### **System Workflow**

Input image  
      │  
      ▼  
Prompt records in prompt.txt  
      │  
      ▼  
OpenClaw agent via skin\_agent.py  
      │  
      ▼  
Tool selection:  
  \- tools/skin\_lesion\_fast.py  
  \- tools/skin\_lesion\_mid.py  
  \- tools/skin\_lesion\_high.py  
      │  
      ▼  
Model inference  
      │  
      ▼  
Structured JSON output  
      │  
      ▼  
Final Markdown report

### **Agent Execution Demo**

### **Example Report**

## **Highlights**

* Agent-driven model orchestration using OpenClaw  
* Multi-model inference pipeline with specialized screening tiers  
* EfficientNet-B0 for low-latency first-pass analysis  
* EfficientNet-B4 for balanced accuracy and confidence  
* Structured JSON outputs for downstream automation  
* Prompt-driven execution and tool selection  
* Modular architecture designed for extensibility

### **Built With**

Python • PyTorch • timm • OpenClaw • Computer Vision • Agentic AI • Ollama

## **Why This Project Matters**

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

# **Overview**

This project provides a local skin lesion screening workflow that combines Computer Vision models with OpenClaw-based agent orchestration.  
The system allows image-based screening through multiple model tiers and supports prompt-driven execution via CLI tools and OpenClaw skills.  
The primary objective is to demonstrate how an AI agent can coordinate specialized diagnostic tools through clear command contracts and structured outputs.

# **Problem Statement**

Traditional skin lesion classification workflows often suffer from one or more of the following limitations:

* Dependence on a single model regardless of context  
* Lack of clear escalation paths between fast and accurate models  
* Tight coupling between inference and orchestration logic  
* Limited support for structured downstream processing

As a result, extending or adapting these systems becomes increasingly difficult as complexity grows.  
This project addresses these limitations through a modular, agent-oriented architecture that separates orchestration, inference, and reporting responsibilities.

# **Solution Approach**

The solution consists of three primary layers:

### **Inference Layer**

Provides specialized diagnostic tools:

* Fast Screening Tool (EfficientNet-B0)  
* Balanced Screening Tool (EfficientNet-B4)  
* High Accuracy Tool (ViT-Large)

### **Orchestration Layer**

Provides agent-based tool selection and execution:

* OpenClaw Skills  
* Prompt Processing  
* Command Routing

### **Reporting Layer**

Provides structured outputs:

* JSON Results  
* Confidence Scores  
* Markdown Reports

Workflow:  
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
      ├── Mid Model (B4)  
      │  
      └── ViT-Large Model  
      │  
      ▼  
Model Inference  
      │  
      ▼  
Structured JSON Output  
      │  
      ▼  
Final Report

# **Demo**

## **Running the Agent**

python skin\_agent.py \--record-index 0

## **Direct Tool Invocation**

Fast model:  
python tools/skin\_lesion\_fast.py \\  
  \--image path/to/image.jpg \\  
  \--metadata '{"age":45,"sex":"female"}'

Balanced model:  
python tools/skin\_lesion\_mid.py \\  
  \--image path/to/image.jpg \\  
  \--metadata '{"age":62,"sex":"male"}'

High model:  
python tools/skin\_lesion\_high.py \\  
  \--image path/to/image.jpg \\  
  \--metadata '{"age":62,"sex":"male"}'

## **OpenClaw Skill Installation**

openclaw \--no-color skills install \--force ./openclaw-skills/skin-lesion-fast

openclaw \--no-color skills install \--force ./openclaw-skills/skin-lesion-mid

openclaw \--no-color skills install \--force ./openclaw-skills/skin-lesion-high

## **Example Output**

# **Features**

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

# **Results & Metrics**

### **Dataset**

The models used in this pipeline were fine-tuned and validated using the **HAM10000** dataset ("Human Against Machine with 10000 dermatoscopic images").

* **Total Images:** 10,015 dermatoscopic cases  
* **Classes (7 Categories):** Actinic keratoses, Basal cell carcinoma, Benign keratosis, Dermatofibroma, Melanoma, Melanocytic nevi, Vascular lesions.

### **Model Performance Comparison**

The core advantage of this agentic architecture is balancing **Inference Latency (Speed)** against **Diagnostic Accuracy**. Below is the benchmarking performance across the three specialized tiers evaluated on the HAM10000 validation set:

| Model Tier | Model Architecture | Accuracy | Avg. Inference Time | Resource Footprint | Best Used For |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Tier 1: Fast** | EfficientNet-B0 | \~74.2% | **\~12ms** | Ultra-Lightweight | Rapid initial triage, mobile/edge scenarios |
| **Tier 2: Mid** | EfficientNet-B4 | \~81.5% | \~35ms | Balanced | Standard automated screening |
| **Tier 3: High** | ViT-Large | **\~88.9%** | \~140ms | Heavy (GPU recommended) | Escalation paths, high-ambiguity cases |

# **Architecture**

## **High-Level Architecture**

The system follows a decoupled, agent-driven orchestration architecture where the OpenClaw agent acts as a central router. It evaluates incoming user prompts and dynamically invokes specialized tools based on performance constraints and confidence requirements.

### **System Data Flow**

\[ User Input: Image \+ Metadata \]  
                │  
                ▼  
   ┌─────────────────────────┐  
   │      OpenClaw Agent     │ (skin\_agent.py)  
   └─────────────────────────┘  
                │  
                ├─► \[Low Latency Pass\] ──────► Fast Screening Tool ────► EfficientNet-B0  
                │  
                ├─► \[Standard Review\] ───────► Balanced Screening Tool ──► EfficientNet-B4  
                │  
                └─► \[High-Risk Escalation\] ──► High Accuracy Tool ──────► ViT-Large  
                                                      │  
                ┌─────────────────────────────────────┘  
                ▼  
   ┌─────────────────────────┐  
   │    Output Aggregator    │  
   └─────────────────────────┘  
                │  
                ├─► Structured JSON Output (Metrics, Confidence & Metadata)  
                │  
                └─► Final Markdown Report (Clinical Support Document)

## **Components**

### **1\. Fast Screening Tool**

* **Location:** tools/skin\_lesion\_fast.py  
* **Core Model:** EfficientNet-B0  
* **Responsibilities:**  
  * Fast initial classification  
  * Low-latency inference with minimal compute  
  * Rapid first-pass triage

### **2\. Balanced Screening Tool**

* **Location:** tools/skin\_lesion\_mid.py  
* **Core Model:** EfficientNet-B4  
* **Responsibilities:**  
  * Higher diagnostic confidence  
  * Improved feature extraction for ambiguous cases  
  * More computationally intensive inference

### **3\. High Accuracy Tool**

* **Location:** tools/skin\_lesion\_high.py  
* **Core Model:** ViT-Large  
* **Responsibilities:**  
  * Maximum diagnostic confidence and SOTA feature extraction  
  * Heavy-duty offline inference (GPU recommended)  
  * Final escalation path for highly uncertain predictions

### **4\. Agent Layer**

* **Location:** skin\_agent.py  
* **Responsibilities:**  
  * Prompt handling  
  * Tool selection  
  * Command execution  
  * Output aggregation

### **5\. Skill Layer**

* **Location:** openclaw-skills/  
* **Responsibilities:**  
  * Agent instructions  
  * Tool contracts  
  * Execution metadata

# **Technical Highlights**

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

# **Engineering Decisions**

## **Why Agentic Architecture?**

Instead of embedding all logic into a single application, responsibilities are separated across tools and orchestration layers.  
Benefits:

* Easier extensibility  
* Better maintainability  
* Improved tool reuse  
* Clear separation of concerns

## **Why EfficientNet?**

EfficientNet offers a strong balance between performance and computational efficiency.

### **EfficientNet-B0**

Chosen for:

* Fast inference  
* Low resource requirements  
* Rapid first-pass screening

### **EfficientNet-B4**

Chosen for:

* Improved representation quality  
* Better classification performance  
* Higher diagnostic confidence

### **ViT-Large**

Chosen for:

* Maximum diagnostic confidence and SOTA feature extraction  
* Robust performance on highly ambiguous or borderline cases  
* Reliable high-tier escalation layer within the agentic workflow

## **Why OpenClaw?**

OpenClaw enables orchestration to remain independent from model implementation.  
Benefits:

* Tool abstraction  
* Modular workflows  
* Prompt-based routing  
* Future scalability

# **Challenges & Lessons Learned**

## **Challenge 1: Tiered Model Coordination**

Designing meaningful separation between fast and balanced screening paths required clear execution boundaries and tool responsibilities.

### **Solution**

* Dedicated command contracts  
* Independent tool interfaces  
* Explicit model roles

## **Challenge 2: Agent-to-Tool Communication**

Reliable orchestration depends on predictable tool behavior and outputs.

### **Solution**

* Structured JSON responses  
* Standardized input formats  
* Consistent CLI interfaces

## **Challenge 3: Metadata Handling**

User-provided metadata can vary significantly in structure and completeness.

### **Solution**

* Validation layers  
* Safe parsing logic  
* Fallback handling strategies

# **Lessons Learned**

Through this project I strengthened my understanding of:

* Agentic AI systems  
* Tool orchestration  
* Multi-model architectures  
* Computer Vision deployment  
* CLI application design  
* Structured AI workflows  
* Software modularity  
* AI system extensibility

# **Repository Structure**

.  
├── openclaw-skills/  
│   ├── skin-lesion-fast/  
│   │   └── SKILL.md  
│   ├── skin-lesion-mid/  
│   │   └── SKILL.md  
│   └── skin-lesion-high/  
│       └── SKILL.md  
│  
├── tools/  
│   ├── skin\_lesion\_fast.py  
│   ├── skin\_lesion\_mid.py  
│   └── skin\_lesion\_high.py  
│  
├── test\_tools/  
│   └── run\_tools.py  
│  
├── skin\_agent.py  
├── prompt.txt  
├── tool\_manifest.md  
├── plan.md  
├── requirements.txt  
└── README.md

# **Getting Started**

## **Clone Repository**

git clone \[https://github.com/amir-khoshdel-louyeh/agentic-skin-lesion-classifier.git\](https://github.com/amir-khoshdel-louyeh/agentic-skin-lesion-classifier.git)

cd agentic-skin-lesion-classifier

## **Create Virtual Environment**

Windows:  
py \-3.11 \-m venv .venv

.\\.venv\\Scripts\\Activate.ps1

Linux/macOS:  
python3 \-m venv .venv

source .venv/bin/activate

## **Install Dependencies**

pip install \-r requirements.txt

## **Install OpenClaw Skills**

openclaw \--no-color skills install \--force ./openclaw-skills/skin-lesion-fast

openclaw \--no-color skills install \--force ./openclaw-skills/skin-lesion-mid

openclaw \--no-color skills install \--force ./openclaw-skills/skin-lesion-high

## **Run Demo**

python skin\_agent.py \--record-index 0

# **Testing & Verification**

This repository includes a dedicated test orchestration script to verify the inference pipeline across all available model tiers.

### **Automated Tool Verification**

You can execute the entire evaluation suite (Fast, Mid, and High tiers) using the provided test runner:  
python test\_tools/run\_tools.py

### **Manual Verification**

If you prefer to test individual components or the agent independently, you can invoke them directly:

* **Fast Tier:**  
  python tools/skin\_lesion\_fast.py \--image path/to/image.jpg

* **Balanced Tier:**  
  python tools/skin\_lesion\_mid.py \--image path/to/image.jpg

* **High Tier:**  
  python tools/skin\_lesion\_high.py \--image path/to/image.jpg

* **Agent Flow:**  
  python skin\_agent.py \--record-index 0

### **Expected Outcome**

* Successful image validation  
* Model inference execution  
* Structured JSON output  
* Generated report

# **Future Improvements**

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

# **Author**

## **Amir Khoshdel Louyeh**

### **Connect**

* **GitHub:** [github.com/amir-khoshdel-louyeh](https://github.com/amir-khoshdel-louyeh)  
* **LinkedIn:** [linkedin.com/in/amir-khoshdel-louyeh](https://www.linkedin.com/in/amir-khoshdel-louyeh)

## **Disclaimer**

This project is intended for educational and research purposes only. It is not a medical device and should not be used for clinical diagnosis or treatment decisions.  
This project is open-source and available under the **MIT License**.