"""Main Brain supervisor (plan.tmp §2.2 — chain step 2, lightweight steerer).

Text-only: reads case metadata, the Quality Agent text, and re-executed
receipts of previous sessions. Never loads models, never sees images
(hard boundary — no image path is ever put in its prompt) and gets NO
diagnostic tools.

Per step it writes a SHORT steering note, not a script. `suggested_subset`
is a HINT only; the session may ignore it. It decides how many sessions
run (1..MAX_SESSIONS) via `stop`. Output per step:
`{"steer": str, "suggested_subset": [names], "stop": bool, "why": str}`.
Invalid order = stop (fail-safe). Every order + why is audit-logged.
"""

from __future__ import annotations

import json
import re as _re

from control.jsontext import first_json_object

MAX_SESSIONS = 7

# VLM tools stay OUT of the diagnostic pool until they re-pass the 0.6
# entry bar (plan.tmp §5). The Brain may not even hint at them.
VLM_EXCLUDED: tuple[str, ...] = ("drdiag-vlm", "vlm-ollama")

BRAIN_SCHEMA = (
    '{"steer": string (2-5 lines), "suggested_subset": [tool names], '
    '"stop": bool, "why": string}'
)

# Numeric scores are stripped from the steer before it reaches a session:
# the Brain reads re-executed receipts and tends to quote them, which both
# breaches the no-transcription rule and anchors the next "independent"
# session on the previous answer. The Brain keeps the numbers; the session
# sees the instruction only.
_SCORE_RE = _re.compile(
    r"(?<![\w.])\d+\.\d+(?![\w.])|\b\d{1,3}(?:\.\d+)?\s?%"
)

# The report footer belongs to the renderer. Brain steers echoed it into
# the session prompt, where it read like part of the instruction.
_FOOTER_RE = _re.compile(
    r"research use only\s*[—–-]?\s*not a medical diagnosis\.?",
    _re.IGNORECASE,
)


def sanitize_steer(text: str) -> str:
    """Remove numeric confidence scores from a steer note."""
    cleaned = _SCORE_RE.sub("a numeric score", text or "")
    # Collapse the awkward wording a substitution leaves behind
    # ("a numeric score confidence", "a numeric score sure").
    cleaned = _re.sub(
        r"a numeric score\s+(?:confidence|probability|score|surety|sure|"
        r"confident)\b",
        "a numeric score", cleaned, flags=_re.IGNORECASE)
    # Collapse the stutter left by neighbouring scores ("score score").
    cleaned = _re.sub(
        r"(?:a numeric score)(?:[ ,;]*(?:a numeric score))+",
        "a numeric score", cleaned)
    return _FOOTER_RE.sub("", cleaned).strip()



def diagnostic_pool(manifest) -> list[str]:
    """Hintable diagnostic names: ready + diagnostic, minus VLM/failed.

    A tool with a local calibration record that did not clear the 0.60
    accuracy / 0.10 ECE / 0.90 melanoma-recall gate stays OUT of the pool
    even though its CLI still runs. With no calibration artifacts (fresh
    clone) nothing is blocked, so the pool degrades to the manifest alone.
    """
    from control.calibration import blocked_tools  # lazy: no cycle

    blocked = blocked_tools()
    return [t.name for t in manifest.tools
            if t.status == "ready" and t.category == "diagnostic"
            and t.name not in VLM_EXCLUDED and t.name not in blocked]


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
            f"results=[{h.get('results', '')}], "
            f"reasoning={h['reasoning'][:250]}"
            for h in history
        )
        step = ("Previous sessions (re-executed receipts):\n" + receipts +
                "\nName the weak point, say what NOT to repeat, suggest "
                "what to verify instead — or stop if the evidence is enough.")
        # Informational only (the Brain still decides): tell it when the
        # pool is exhausted with one agreeing class, so a third identical
        # dual-model session is a knowing choice rather than an oversight.
        ran_tools = {
            tool.strip().replace("-", "_")
            for h in history
            for tool in str(h.get("tools") or "").split("+")
            if tool.strip()
        }
        pool_tools = {name.replace("-", "_") for name in pool_names}
        classes = {h.get("class") for h in history if h.get("class")}
        if pool_tools and pool_tools <= ran_tools and len(classes) == 1:
            step += ("\nEvery hintable tool has already been receipted this "
                     "round and they agree on one class: re-running the same "
                     "tools adds no new evidence.")
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

    payload = first_json_object(raw)
    if payload is None:
        return stopped("no JSON object in Brain reply")
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
        "steer": sanitize_steer(steer_text)[:1500],
        "suggested_subset": hint,
        "stop": stop,
        "why": _FOOTER_RE.sub("", why).strip()[:500],
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
    try:
        from control.vram import unload_ollama as _unload
        _unload()  # sequential GPU: free LLM VRAM before tool phases
    except Exception:
        pass
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


SPECIALIST_BRIEFS_SCHEMA = (
    '{"briefs": [{"disease": string, "angle": string}]}'
)


def build_specialist_briefs_prompt(diseases: list[str],
                                   history: list[dict],
                                   quality_text: str,
                                   role_text: str) -> str:
    """Ask the Brain for one advocate angle per shortlisted disease."""
    receipts = "\n".join(
        f"Session {h['n']}: class={h['class']}, "
        f"confidence={h['confidence']}, "
        f"flags={h['flags'] or 'none'}, "
        f"reasoning={h['reasoning'][:250]}"
        for h in history
    )
    return (
        f"{role_text}\n"
        f"Two diseases are under debate: "
        f"{' vs '.join(diseases)}.\n"
        f"Quality Agent: {quality_text}\n"
        f"Re-executed receipts:\n{receipts or '(none)'}\n\n"
        "Write ONE short advocate angle per disease (a DIFFERENT angle "
        "per disease: what would speak FOR it, what to stress, what to "
        "verify). Angles steer advocates; they never decide anything.\n\n"
        "Reply with exactly one JSON object, no other text: "
        + SPECIALIST_BRIEFS_SCHEMA
    )


def parse_specialist_briefs(raw: str,
                            diseases: list[str]) -> dict[str, str] | None:
    """Parse Brain-written advocate angles. Returns None on any defect.

    The briefs must cover EXACTLY the shortlisted diseases (one non-empty
    angle each) — a partial or off-shortlist set is unusable. None means
    "fall back to the static template"; the advocate sessions still run.
    """
    payload = first_json_object(raw)
    if payload is None:
        return None
    briefs = payload.get("briefs")
    if not isinstance(briefs, list):
        return None
    angles: dict[str, str] = {}
    for item in briefs:
        if not isinstance(item, dict):
            return None
        disease, angle = item.get("disease"), item.get("angle")
        if not isinstance(disease, str) or not isinstance(angle, str):
            return None
        if disease in angles or not angle.strip():
            return None
        angles[disease] = angle.strip()[:800]
    if set(angles) != set(diseases):
        return None
    return angles


def run_specialist_briefs(diseases: list[str], history: list[dict],
                          quality_text: str,
                          agent_id: str = "main"
                          ) -> tuple[dict[str, str], str]:
    """Brain-written angle per shortlisted disease. Never raises.

    Returns (briefs, source) with source "brain" or "fallback". On invalid
    Brain output or backend failure the static template fills in — the
    advocate sessions still run; the fallback is audit-logged.
    """
    from control.specialists import specialist_brief  # lazy: template owner

    def fallback() -> tuple[dict[str, str], str]:
        return ({d: specialist_brief(d, diseases) for d in diseases},
                "fallback")

    from control.prompts import load_role  # lazy: file layout dependency

    try:
        role_text = load_role("brain")
    except Exception:  # noqa: BLE001 - fallback, don't crash
        return fallback()
    from skin_agent import run_openclaw_cli  # lazy: CLI-only dependency

    try:
        response = run_openclaw_cli(
            build_specialist_briefs_prompt(diseases, history, quality_text,
                                           role_text),
            agent_id=agent_id, show_command=False)
    except Exception:  # noqa: BLE001 - fallback, don't crash
        return fallback()
    if isinstance(response, dict) and response.get("payloads"):
        raw = "\n".join(item.get("text", "")
                         for item in response["payloads"])
    else:
        raw = json.dumps(response, ensure_ascii=False)
    try:
        from control.vram import unload_ollama as _unload2
        _unload2()  # sequential GPU: free LLM VRAM before tool phases
    except Exception:
        pass
    angles = parse_specialist_briefs(raw, diseases)
    if angles is None:
        return fallback()
    return angles, "brain"


def receipt_summary(verdict, n: int, tools: str = "") -> dict:
    """Text-only receipt digest of one re-executed session for the Brain.

    `verdict` may be one envelope or the list of envelopes a session
    earned (one per tool). Every tool is listed so the Brain can see a
    cross-model disagreement instead of one truncated row.
    """
    items = list(verdict) if isinstance(verdict, (list, tuple)) else [verdict]
    primary = max(items, key=lambda v: v.confidence or 0.0)
    results = "; ".join(
        f"{session_tools(v.command)}={v.predicted_class}"
        + (f"({v.confidence})" if v.confidence is not None else "")
        + (f" flags={','.join(v.uncertainty_flags)}"
           if v.uncertainty_flags else "")
        for v in items
    )
    return {
        "n": n,
        "class": primary.predicted_class,
        "confidence": primary.confidence,
        "flags": ",".join(dict.fromkeys(
            f for v in items for f in v.uncertainty_flags)),
        "tools": tools,
        "results": results,
        "reasoning": primary.reasoning or "",
    }
