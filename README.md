# Agentic Skin Lesion Classifier — Run & Test Guide

This file describes how to set up the environment, run the pipeline, and run included tests/benchmarks.

Prerequisites
- Python 3.10+ (recommended 3.11)
- `git` and a terminal
- (Optional) Ollama runtime running locally if you want to call the vision model

Model configuration
- The Ollama model name is read from `config.yaml` first.
- To change the model, edit `config.yaml` and update `ollama_model`.
- The `OLLAMA_MODEL` environment variable still works and overrides the config file.

Quick setup
1. Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\Activate.ps1  # or Activate.bat
# macOS / Linux
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure dataset files are present under `dataset/HAM1000/`:
- `HAM10000_metadata.csv` (metadata)
- `hmnist_28_28_RGB.csv/hmnist_28_28_RGB.csv` (HMNIST pixel CSV)

Run the pipeline (single example)

Use module invocation (`python -m`) to ensure local `src` imports resolve correctly.

```bash
# Run the pipeline on HMNIST row 0 and print JSON
python -m cli.run_pipeline 0

# Run without calling the Ollama vision model (deterministic features only)
python -m cli.run_pipeline 0 --no-model
```

Example `config.yaml`:

```yaml
ollama_model: llama3:latest
```

Evaluate accuracy (melanoma vs non-melanoma)

Run with `python -m` to avoid ModuleNotFoundError when importing `src`.

```bash
# Stream and evaluate up to 500 rows (conservative melanoma detection)
python -m cli.evaluate dataset/HAM1000/hmnist_28_28_RGB.csv/hmnist_28_28_RGB.csv --rows 500

# Include vision model calls (requires Ollama)
python -m cli.evaluate dataset/HAM1000/hmnist_28_28_RGB.csv/hmnist_28_28_RGB.csv --rows 200 --call-model
```

Benchmark latency

```bash
# Measure wall-clock and internal pipeline timings for 20 samples
python -m cli.benchmark_latency --rows 20

# Include model calls during benchmarking (will be slower)
python -m cli.benchmark_latency --rows 10 --call-model
```

Notes on explainability & interpretability
- The pipeline logs an `explainability_log` per run and includes `interpretability_passed` and `citations` in the final report.
- See `src/pipeline.py` for logging step names and how reasoning is synthesized.

Making the pipeline dataset-agnostic
- The ingestion API is implemented in `src/ingest.py` with `DatasetAdapter` and `HMNISTAdapter`.
- To run on another dataset (ISIC or custom), implement a `DatasetAdapter` and pass it to `OperationalPipeline`.

Troubleshooting
- If vision calls fail, make sure Ollama is running locally and the model name in `config.yaml` matches one from `ollama list`.
- If you still want to bypass model calls, use `--no-model`.
- If CSV loading fails due to large file sync issues in your editor, run scripts from the terminal where files are accessible.
- You can override which local Ollama model is used by editing `config.yaml` or setting `OLLAMA_MODEL`.
Next steps
- Add `ISICAdapter` implementation to `src/ingest.py` to support ISIC dataset exports.
- Replace `src/connectors.py` stubs with real FHIR / literature clients when integrating EHR or PubMed.

License & Safety
- This repository provides research code and heuristics; it is not clinical-grade software. Do not use for diagnosis without clinical validation.
