"""Quality Agent (plan.tmp §2.1 — chain step 1, LLM thinker).

Toolkit (installed skills): `quality-gate`, `preprocess`,
`abcde-analyzer` (read-only context). The agent decides itself whether
the image is good enough (pass), cleans it with preprocess 0..n times
by its own judgment (cleaned), or passes it forward with flags
(flagged). No blur threshold lives in code.

Output: short JSON + text — `pass|cleaned|flagged` + reasons, citing
the tool JSON it saw. Never diagnoses; never stops the round: a
`flagged` image flows forward with its flag, and a tool crash is
`no-evidence` for that check, never an abort. Invalid output fails
safe to `flagged` (forward with flag), never to a stop.

Control-side evidence (`gate_evidence`): the orchestrator already runs the
deterministic `quality-gate` before this step, so its numbers are handed to
the agent instead of leaving the agent to reproduce them. That keeps the
single-writer rule the rest of the pipeline follows — control produces the
numbers, the model only judges — and it removes the step a small local
model gets wrong most often (running the command and quoting the JSON
back). It is still EVIDENCE, never a decision: the agent alone chooses
pass/cleaned/flagged, and no blur threshold lives in code. The gate's flags
reach the critic independently, so an unusable or wrong Quality verdict can
never hide that evidence.
"""

from __future__ import annotations

import json
import re as _re
from pathlib import Path

from control.paths import ROOT_DIR

QUALITY_TOOLKIT: tuple[str, ...] = (
    "quality-gate",
    "preprocess",
    "abcde-analyzer",
)

QUALITY_SCHEMA = (
    '{"verdict": "pass|cleaned|flagged", "reasons": [string], '
    '"cites": [string], "cleaned_image": string|null}'
)

VALID_VERDICTS = ("pass", "cleaned", "flagged")

# Fresh-session repair attempts after the first reply fails to parse, so a
# weak local model gets three chances to emit the JSON verdict without
# unbounded LLM spend. Two keeps the worst case at three sessions.
MAX_REPAIR_RETRIES = 2

# Fail-safe cite written ONLY by parse_quality() on unparseable or
# schema-violating replies. run_quality() treats a verdict carrying
# exactly this cite as "parse failed" and grants one repair retry;
# anything else (including a genuine `flagged`) is final.
FAILSAFE_CITE = "fail-safe: unparseable or schema-violating reply"


def _release_llm() -> None:
    """Best-effort LLM unload after a session (sequential GPU)."""
    try:
        from control.vram import unload_ollama as _unload
        _unload()
    except Exception:
        pass


def build_quality_repair_prompt(raw: str,
                                  reasons: list[str] | None = None
                                  ) -> str:
    defect = ("Specifically: " + "; ".join(str(r)[:200] for r in reasons)
              if reasons else
              "It was not a valid quality verdict.")
    return (
        "Your previous reply was rejected. " + defect + " Reply ONCE more "
        "with exactly one JSON object, no other text: " + QUALITY_SCHEMA +
        "\nPrevious reply:\n" + (raw or "").strip()[:2000]
    )


def build_quality_prompt(record: dict, role_text: str, toolkit_text: str,
                          gate_evidence: dict | None = None) -> str:
    """Prompt the Quality Agent with case + role + toolkit (+ gate numbers).

    `gate_evidence` is the control-side `quality_gate` result the
    orchestrator already collected; see the module docstring for why it is
    handed over. It is evidence, never a decision.
    """
    metadata = json.dumps(record.get("metadata", {}), ensure_ascii=False)
    if gate_evidence:
        evidence_block = (
            "Control-side quality_gate result (authoritative, read-only "
            "evidence — cite it):\n"
            + json.dumps({
                "quality_pass": gate_evidence.get("passed"),
                "flags": gate_evidence.get("flags", []),
                "metrics": gate_evidence.get("metrics", {}),
                "tool_error": gate_evidence.get("error") or None,
            }, ensure_ascii=False)
            + "\n\n"
        )
        source = ("judge only from the control-side result below")
    else:
        evidence_block = (
            "No control-side result was collected for this case: run the "
            "`quality-gate` line below and judge from its JSON.\n\n"
        )
        source = ("judge only from the tool JSON you get back")
    # Design note (verified live 2026-10-16 with qwen3:8b): the previous
    # wording opened with `Case image: <path>` and asked the agent to
    # "assess this image's quality". A local text-only model read that as a
    # vision task, invented a description of the lesion, and answered with
    # dermatological prose or even a web_search call — never the verdict.
    # Stating the no-image boundary up front, presenting the path only as a
    # command argument, and spelling out the three verdict meanings yields
    # valid, evidence-grounded verdicts (flagged on a blurry case, pass on a
    # clean one).
    return (
        f"{role_text}\n"
        "Decide the image-quality verdict for this case.\n\n"
        "You have no image capability. You must never look at the image, "
        f"describe it, or search the web for it — {source}.\n\n"
        f"Case file (a path argument for the commands below — you cannot see "
        f"its contents): {record.get('image_path')}\n"
        f"Case metadata (context only, never for diagnosis): {metadata}\n\n"
        f"{evidence_block}"
        "Optional cleanup — run a line EXACTLY as written only if YOU decide "
        f"cleaning is worthwhile (0..n times):\n{toolkit_text}\n\n"
        "VERDICT MEANINGS (pick exactly one):\n"
        '- "pass": good enough as-is.\n'
        '- "flagged": usable with its flag. Use this whenever anything is '
        "uncertain OR whenever you did NOT run the preprocess line.\n"
        '- "cleaned": ONLY if you actually ran the preprocess line and hold '
        "its exact --output path; report that path.\n"
        "Never name a disease and never give a malignancy judgment. Do NOT "
        "call any tool other than the lines above, and do NOT write medical "
        "commentary.\n\n"
        "You MUST reply with exactly one JSON object and no other text.\n"
        "Schema: " + QUALITY_SCHEMA
    )


def parse_quality(raw: str) -> dict:
    """Strict-parse a Quality Agent reply. Never raises.

    Fail-safe is always `flagged` (flows forward with its flag): malformed
    JSON, schema violations, missing cites, or a `cleaned` claim without
    a `cleaned_image` path all become `flagged` with the defect recorded.
    Nothing here may stop the round.
    """

    def flagged(*reasons: str) -> dict:
        return {
            "verdict": "flagged",
            "reasons": [str(r)[:300] for r in reasons] or ["invalid output"],
            "cites": [FAILSAFE_CITE],
            "cleaned_image": None,
        }

    match = _re.search(r"\{.*\}", (raw or "").strip(), _re.DOTALL)
    if not match:
        return flagged("no JSON object in Quality Agent reply")
    try:
        payload = json.loads(match.group(0))
    except (ValueError, TypeError):
        return flagged("malformed JSON in Quality Agent reply")
    if not isinstance(payload, dict):
        return flagged("Quality Agent reply is not a JSON object")
    verdict = payload.get("verdict")
    if verdict not in VALID_VERDICTS:
        return flagged(f"invalid verdict {verdict!r}: "
                       "must be pass|cleaned|flagged")
    reasons = payload.get("reasons")
    cites = payload.get("cites")
    if not isinstance(reasons, list) or not reasons or not all(
            isinstance(r, str) and r.strip() for r in reasons):
        return flagged("quality verdict without reasons")
    if not isinstance(cites, list) or not cites or not all(
            isinstance(c, str) and c.strip() for c in cites):
        return flagged("quality verdict without cites")
    cleaned_image = payload.get("cleaned_image")
    if cleaned_image is not None and not (
            isinstance(cleaned_image, str) and cleaned_image.strip()):
        return flagged("invalid cleaned_image: must be a path or null")
    if verdict == "cleaned" and cleaned_image is None:
        return flagged("claimed cleaned without a cleaned_image path")
    return {
        "verdict": verdict,
        "reasons": [str(r)[:300] for r in reasons],
        "cites": [str(c)[:300] for c in cites],
        "cleaned_image": cleaned_image,
    }


def _payload_text(response) -> str:
    """Agent reply text from an OpenClaw payload list, else a JSON dump."""
    if isinstance(response, dict) and response.get("payloads"):
        return "\n".join(item.get("text", "")
                         for item in response["payloads"])
    return json.dumps(response, ensure_ascii=False)


def run_quality(record: dict, toolkit_text: str,
                agent_id: str = "main",
                gate_evidence: dict | None = None) -> dict:
    """Run one Quality Agent session. Never raises, never stops the round.

    Backend failure, missing role files, or invalid output all become a
    `flagged` verdict that flows forward — the round continues.

    `gate_evidence` (optional) is the control-side `quality_gate` result
    handed to the agent as read-only evidence; the agent still decides.
    """
    from control.prompts import load_role  # lazy: file layout dependency

    try:
        role_text = load_role("quality")
    except Exception as exc:  # noqa: BLE001 - flag, don't crash
        return {
            "verdict": "flagged",
            "reasons": [f"quality role unavailable: {str(exc)[:150]}"],
            "cites": ["fail-safe: role prompt failed to load"],
            "cleaned_image": None,
        }
    from skin_agent import run_openclaw_cli  # lazy: CLI-only dependency

    try:
        response = run_openclaw_cli(
            build_quality_prompt(record, role_text, toolkit_text,
                                 gate_evidence=gate_evidence),
            agent_id=agent_id, show_command=False)
    except Exception as exc:  # noqa: BLE001 - advisory role only
        return {
            "verdict": "flagged",
            "reasons": [f"quality backend unavailable: {str(exc)[:150]}"],
            "cites": ["fail-safe: agent backend failed"],
            "cleaned_image": None,
        }
    raw = _payload_text(response)
    result = parse_quality(raw)
    # Up to MAX_REPAIR_RETRIES fresh sessions on parse failure; the first
    # valid verdict wins, and whatever the last attempt produced (valid or
    # fail-safe flagged) is final and flows forward.
    for _ in range(MAX_REPAIR_RETRIES):
        if result["cites"] != [FAILSAFE_CITE]:
            break
        try:
            retry_response = run_openclaw_cli(
                build_quality_repair_prompt(raw, result.get("reasons", [])),
                agent_id=agent_id, show_command=False)
        except Exception:  # noqa: BLE001 - keep the last fail-safe
            break
        raw = _payload_text(retry_response)
        result = parse_quality(raw)
    _release_llm()
    return result


def resolve_cleaned_image(record: dict, quality: dict,
                          root: Path = ROOT_DIR) -> Path | None:
    """Path to the cleaned image downstream sessions should use, or None.

    Only a `cleaned` verdict whose `cleaned_image` exists on disk
    qualifies. Anything else — pass/flagged, a missing or fabricated
    path — returns None so the round keeps the original image. Quality
    never blocks, and sessions are never pointed at a file that is not
    there (the plan.tmp §2.1 `cleaned` hand-off).
    """
    if quality.get("verdict") != "cleaned":
        return None
    candidate = quality.get("cleaned_image")
    if not isinstance(candidate, str) or not candidate.strip():
        return None
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = root / path
    return path if path.is_file() else None


def quality_line(quality: dict) -> str:
    """One-line audit/report rendering: verdict + cited reasons."""
    verdict = quality.get("verdict", "flagged")
    reasons = "; ".join(quality.get("reasons", [])[:3])
    cleaned = quality.get("cleaned_image")
    extra = f" → {cleaned}" if verdict == "cleaned" and cleaned else ""
    return f"Quality: {verdict}{extra} — {reasons}"
