"""Project configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config() -> dict[str, Any]:
    """Load project config from `config.yaml` if available.

    Returns an empty dict when the file is missing.
    """
    if not CONFIG_PATH.exists():
        return {}

    try:
        import yaml

        with CONFIG_PATH.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except ModuleNotFoundError:
        data = _load_simple_yaml(CONFIG_PATH)

    if not isinstance(data, dict):
        raise ValueError("config.yaml must contain a YAML mapping at the top level.")
    return data


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    """Very small YAML fallback parser for simple `key: value` config files.

    Supports flat mappings with scalar string/number/bool values.
    """
    result: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            raise ValueError(f"Unsupported config line: {raw_line!r}")
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid config key in line: {raw_line!r}")
        if value.lower() in {"true", "false"}:
            parsed: Any = value.lower() == "true"
        else:
            try:
                parsed = int(value)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value.strip('"\'')
        result[key] = parsed
    return result


def get_ollama_model_name(default: str = "llama3.2-vision") -> str:
    """Resolve Ollama model name from env/config/default precedence."""
    import os

    env_model = os.environ.get("OLLAMA_MODEL")
    if env_model:
        return env_model

    config = load_config()
    config_model = config.get("ollama_model")
    if isinstance(config_model, str) and config_model.strip():
        return config_model.strip()

    return default
