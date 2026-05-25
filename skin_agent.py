# skin_agent.py
import os
import textwrap
import openclaw
from cmdop.exceptions import AgentNotRunningError

SYSTEM_INSTRUCTIONS = textwrap.dedent("""
    You are an OpenClaw agent run for a medical skin lesion classification workflow.
    You have access to a local repository-level toolset documented by the
    `skin-lesion-tools` skill. When you need to analyze an image, use the
    local command-line tools in `tools/skin_tools.py` rather than bypassing
    OpenClaw.

    Workflow:
    1. Run tier 1 analysis with `python tools/skin_tools.py analyze --image <path> --model-tier tier1_fast`.
    2. If confidence is below 0.85, run tier 2 analysis with `python tools/skin_tools.py analyze --image <path> --model-tier tier2_deep`.
    3. Produce a final Markdown clinical report that includes:
       - which local tool commands were executed
       - the model tier names used
       - confidence escalation reasoning
       - the final diagnosis and prediction
""")

SAMPLE_IMAGE_PATH = "data/sample_lesion.jpg"


def ensure_sample_image(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        from PIL import Image
        print(f"[System] Generating a placeholder image at {path} for initial setup...")
        Image.new("RGB", (400, 400), color="red").save(path)


def create_openclaw_client() -> openclaw.OpenClaw:
    try:
        client = openclaw.OpenClaw.local()
        if not client.is_connected:
            raise RuntimeError("OpenClaw connected but reported disconnected state.")
        return client
    except AgentNotRunningError as exc:
        raise RuntimeError(
            "OpenClaw local agent not running. Start it with `cmdop agent start` or ensure the OpenClaw gateway and local agent are running."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "Unable to connect to the local OpenClaw agent. Confirm your OpenClaw gateway and local agent are running."
        ) from exc


def build_agent_prompt(image_path: str) -> str:
    return textwrap.dedent(
        f"""
        {SYSTEM_INSTRUCTIONS}

        Patient case: analyze the image at `{image_path}`.
        If the local `skin-lesion-tools` skill is installed, prefer that skill.
        Otherwise, use terminal-based tool commands against `tools/skin_tools.py`.

        Answer with a clean Markdown clinical report.
        """
    )


def list_agent_skills(client: openclaw.OpenClaw) -> None:
    try:
        skills = client.skills.list()
        skill_names = [skill.name for skill in skills[:12]]
        print(f"[OpenClaw] Discovered skills: {skill_names}{'...' if len(skills) > 12 else ''}")
    except Exception:
        print("[OpenClaw] Could not enumerate skills; continuing with agent execution.")


def main() -> None:
    print("[System] Starting OpenClaw-backed skin lesion agent...")
    ensure_sample_image(SAMPLE_IMAGE_PATH)

    try:
        client = create_openclaw_client()
    except RuntimeError as exc:
        print(f"[Error] {exc}")
        return

    try:
        print(f"[OpenClaw] Connected to local agent. Mode={client.mode}")
        list_agent_skills(client)

        prompt = build_agent_prompt(SAMPLE_IMAGE_PATH)
        print("[OpenClaw] Sending prompt to the local agent...")
        result = client.agent.run(prompt)

        print("\n" + "=" * 20 + " FINAL CLINICAL REPORT " + "=" * 20)
        print(result.text)
        print("=" * 63)
    finally:
        client.close()


if __name__ == "__main__":
    main()
