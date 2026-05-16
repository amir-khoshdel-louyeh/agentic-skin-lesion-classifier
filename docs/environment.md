# Environment and Dependency Definition

## Runtime Decisions

- Local LLM/VLM runtime: Ollama
- Default vision model: `llama3.2-vision`
- Python version: 3.11
- Package manager: `pip` with `venv`

## Why This Stack

- Python 3.11 offers strong ecosystem compatibility and good runtime performance.
- `pip` + `venv` keeps setup simple and portable for local CLI workflows.
- Ollama runs locally to keep image data private and avoid cloud transfer.

## Setup Instructions

1. Install Python 3.11.
2. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4. Install Ollama and pull the vision model:

```powershell
ollama pull llama3.2-vision
```

5. Copy environment variables:

```powershell
Copy-Item .env.example .env
```

## Dependency Coverage

- Vision/data processing: `numpy`, `pandas`, `pillow`
- Ollama integration/runtime calls: `ollama`, `httpx`
- CLI parsing and rich terminal output: `typer`, `rich`
- Structured validation and JSON-ready schemas: `pydantic`
- Logging and observability: `loguru`
- Environment config and progress helpers: `python-dotenv`, `tqdm`

## Notes

- JSON serialization can use Python's built-in `json` module for report output.
- If `llama3.2-vision` is not available on your machine, update `OLLAMA_MODEL` in `.env`.
