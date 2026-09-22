"""VRAM budget enforcement (§10.5).

Manifest budgets are enforced, not aspirational: before loading a heavy
model, tools call `check()` and fail fast with an actionable message
instead of dying in a cryptic CUDA OOM. CPU-only hosts always pass.
"""

try:
    from torch.cuda import is_available as _cuda_ok
    from torch.cuda import mem_get_info as _mem_info
except ImportError:  # torch missing: the tool itself reports that later
    _cuda_ok = None  # type: ignore[assignment]
    _mem_info = None  # type: ignore[assignment]


def check(required_gb: float) -> tuple[bool, str]:
    """Return (ok, reason). `ok` False means do NOT load the model."""
    if _cuda_ok is None or not _cuda_ok():
        return True, ""
    free, _ = _mem_info()
    if free >= int(required_gb * 1024**3):
        return True, ""
    return False, (
        f"only {free / 1024**3:.1f} GB GPU free, need {required_gb} GB — "
        "idle/unload the resident LLM first, then retry."
    )
