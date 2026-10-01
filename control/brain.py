"""Main Brain supervisor (plan.tmp §2.2 — chain step 2, lightweight steerer).

Text-only: reads case metadata, the Quality Agent text, and re-executed
receipts of previous sessions. Never loads models, never sees images
(hard boundary — no image path is ever put in its prompt) and gets NO
diagnostic tools.

Per step it writes a SHORT steering note, not a script. `suggested_subset`
is a HINT only; the session may ignore it. It decides how many sessions
run (1..3) via `stop`. Output per step:
`{"steer": str, "suggested_subset": [names], "stop": bool, "why": str}`.
Invalid order = stop (fail-safe). Every order + why is audit-logged.
"""

from __future__ import annotations

import json
import re as _re

MAX_SESSIONS = 3

# VLM tools stay OUT of the diagnostic pool until they re-pass the 0.6
# entry bar (plan.tmp §5). The Brain may not even hint at them.
VLM_EXCLUDED: tuple[str, ...] = ("drdiag-vlm", "vlm-ollama")

BRAIN_SCHEMA = (
    '{"steer": string (2-5 lines), "suggested_subset": [tool names], '
    '"stop": bool, "why": string}'
)


def diagnostic_pool(manifest) -> list[str]:
    """Hintable diagnostic tool names: ready + diagnostic, minus VLM."""
    return [t.name for t in manifest.tools
            if t.status == "ready" and t.category == "diagnostic"
            and t.name not in VLM_EXCLUDED]


def build_brain_prompt(record: dict, quality_text: str,
                        history: list[dict], role_text: str,
                        pool_names: list[str]) -> str:
    """Prompt the Brain for the next order. Text only — no image path."""
    metadata = json.dumps(record.get("metadata", {}), ensure_ascii=False)
    if not history:
        step = ("This is Steer 1 (no session has run yet): give a broad "
                "first look + what to focus on.")
    else:
        receipts = "\n".join(
            f"Session {h['n']}: class={h['class']}, "
            f"confidence={h['confidence']}, "
            f"flags={h['flags'] or 'none'}, "
            f"tools={h['tools'] or 'unknown'}, "
            f"reasoning={h['reasoning'][:250]}"
            for h in history
        )
        step = ("Previous sessions (re-executed receipts):\n" + receipts +
                "\nName the weak point, say what NOT to repeat, suggest "
                "what to verify instead — or stop if the evidence is enough.")
    return (
        f"{role_text}\n"
        f"Case metadata (context only): {metadata}\n"
        f"Quality Agent: {quality_text}\n"
        f"{step}\n"
        f"Hintable tools (hint only, sessions may ignore): "
        f"{', '.join(pool_names) or 'none'}\n\n"
        "Reply with exactly one JSON object, no other text: "
        + BRAIN_SCHEMA
    )


def parse_brain_order(raw: str, pool_names: list[str]) -> dict:
    """Strict-parse a Brain order. Fail-safe is always stop=True.

    Invalid JSON, schema violations, empty steer (with stop=false), or a
    missing why all become `{"stop": True, ...}` so a broken Brain halts
    the loop instead of spawning unbounded sessions. Unknown subset names
    are dropped (hint only — never fatal).
    """
    pool = set(pool_names)

    def stopped(*reasons: str) -> dict:
        return {
            "steer": "",
            "suggested_subset": [],
            "stop": True,
            "why": "fail-safe stop: " + "; ".join(
                str(r)[:200] for r in reasons) or "fail-safe stop",
        }

    match = _re.search(r"\{.*\}", (raw or "").strip(), _re.DOTALL)
    if not match:
        return stopped("no JSON object in Brain reply")
    try:
        payload = json.loads(match.group(0))
    except (ValueError, TypeError):
        return stopped("malformed JSON in Brain reply")
    if not isinstance(payload, dict):
        return stopped("Brain reply is not a JSON object")
    stop = payload.get("stop")
    if not isinstance(stop, bool):
        return stopped(f"invalid stop {stop!r}: must be bool")
    why = payload.get("why")
    if not isinstance(why, str) or not why.strip():
        return stopped("Brain order without why")
    steer = payload.get("steer")
    if stop:
        steer_text = steer if isinstance(steer, str) else ""
    else:
        if not isinstance(steer, str) or not steer.strip():
            return stopped("Brain order to continue without a steer")
        steer_text = steer.strip()
    subset = payload.get("suggested_subset", [])
    if subset is None:
        subset = []
    if not isinstance(subset, list) or not all(
            isinstance(n, str) for n in subset):
        return stopped("invalid suggested_subset: must be a list of names")
    hint = [n for n in subset if n in pool]
    return {
        "steer": steer_text[:1500],
        "suggested_subset": hint,
        "stop": stop,
        "why": why.strip()[:500],
    }


def run_brain_step(record: dict, quality_text: str, history: list[dict],
                   pool_names: list[str], agent_id: str = "main") -> dict:
    """Ask the Brain for one order. Never raises: backend failure = stop."""
    from control.prompts import load_role  # lazy: file layout dependency

    try:
        role_text = load_role("brain")
    except Exception as exc:  # noqa: BLE001 - stop, don't crash
        return {"steer": "", "suggested_subset": [], "stop": True,
                "why": f"fail-safe stop: brain role unavailable: "
                       f"{str(exc)[:150]}"}
    from skin_agent import run_openclaw_cli  # lazy: CLI-only dependency

    try:
        response = run_openclaw_cli(
            build_brain_prompt(record, quality_text, history, role_text,
                               pool_names),
            agent_id=agent_id, show_command=False)
    except Exception as exc:  # noqa: BLE001 - supervisor is advisory
        return {"steer": "", "suggested_subset": [], "stop": True,
                "why": f"fail-safe stop: brain backend unavailable: "
                       f"{str(exc)[:150]}"}
    if isinstance(response, dict) and response.get("payloads"):
        raw = "\n".join(item.get("text", "")
                         for item in response["payloads"])
    else:
        raw = json.dumps(response, ensure_ascii=False)
    return parse_brain_order(raw, pool_names)


def session_tools(command: str) -> str:
    """Tool actually run, derived control-side from the verified command.

    Tells the next steer what NOT to repeat. Unverifiable commands yield
    "unknown" — never a guess from agent prose.
    """
    parts = (command or "").split()
    if len(parts) >= 2 and parts[0] == "python" and parts[1].startswith(
            "tools/") and parts[1].endswith(".py"):
        return parts[1][len("tools/"):-len(".py")]
    return "unknown"


def receipt_summary(verdict, n: int, tools: str = "") -> dict:
    """Text-only receipt digest of one re-executed session for the Brain."""
    return {
        "n": n,
        "class": verdict.predicted_class,
        "confidence": verdict.confidence,
        "flags": ",".join(verdict.uncertainty_flags),
        "tools": tools,
        "reasoning": verdict.reasoning or "",
    }
