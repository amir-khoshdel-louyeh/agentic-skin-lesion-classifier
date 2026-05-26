# OpenClaw Tool Development Guide

This guide shows how to add a new local tool to this repository and introduce it to OpenClaw.

## 1. Create the local tool script

1. Create a new Python file in `tools/`, for example:
   - `tools/my_tool.py`
2. Keep your tool CLI simple and self-contained.
3. Expose actions with subcommands so OpenClaw can call them from a terminal skill.

Example pattern:

```python
import argparse
import json

from core import analyze_skin_lesion


def run_command(args):
    result = analyze_skin_lesion(args.image_path, args.model_tier)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description="My OpenClaw-accessible tool")
    parser.add_argument("--image", required=True, help="Path to lesion image")
    parser.add_argument("--model-tier", choices=["tier1_fast", "tier2_deep"], required=True)
    args = parser.parse_args()
    run_command(args)


if __name__ == "__main__":
    main()
```

## 2. Add a local OpenClaw skill manifest

1. Create a new skill folder under `openclaw-skills/`, for example:
   - `openclaw-skills/my-tool/`
2. Add a `SKILL.md` file inside it.
3. Document the tool and how OpenClaw agents can use it.

Example `SKILL.md` structure:

```markdown
---
name: my-tool
description: Local tool for skin lesion analysis and OpenClaw terminal automation.
metadata: { "openclaw": { "requires": { "bins": ["python"] } } }
---

# My Tool

This skill documents how to use the local tool from OpenClaw.

## Commands

```bash
python tools/my_tool.py --image data/sample_lesion.jpg --model-tier tier1_fast
```

## Usage

Use this skill when the agent needs to run tiered model analysis from the local terminal.
```

## 3. Install the skill into OpenClaw

From the repository root, run:

```powershell
openclaw --no-color skills install ./openclaw-skills/my-tool
```

If the skill installs successfully, it will be added to the local OpenClaw workspace.

## 4. Confirm the skill is available

List installed skills:

```powershell
openclaw --no-color skills list
```

Search for your skill name in the output.

## 5. Use the new tool in an OpenClaw agent prompt

When you write the agent prompt or skill instructions, mention the exact command the agent should run.

Example:

- `python tools/my_tool.py --image data/sample_lesion.jpg --model-tier tier1_fast`
- `python tools/my_tool.py --image data/sample_lesion.jpg --model-tier tier2_deep`

If you want the agent to use OpenClaw terminal tooling, instruct it to run those commands from the local shell.

Example aliases:

```bash
python tools/skin_tools.py predict --image data/sample_lesion.jpg --model-tier tier1_fast
python tools/skin_tools.py validate --image data/sample_lesion.jpg
python tools/skin_tools.py info
python tools/skin_tools.py list-tiers
```

## 6. Optional: integrate through `openclaw.OpenClaw.local()`

If you want Python code to drive OpenClaw directly, use the installed OpenClaw SDK:

```python
import openclaw
client = openclaw.OpenClaw.local()
result = client.agent.run("Run the local tool and summarize the output.")
```

This is useful for agent orchestration code when your environment already supports the local OpenClaw agent.

## 7. Verify the command works directly first

Before relying on OpenClaw, test the tool from the terminal:

```powershell
python tools/my_tool.py --image data/sample_lesion.jpg --model-tier tier1_fast
```

If this works, then OpenClaw can safely invoke it.

## 8. Keep skill docs in sync

Whenever you add a new tool or command:

- update `openclaw-skills/my-tool/SKILL.md`
- mention the commands agents should use
- install or reinstall the skill if the manifest changes

---

### Recommended workflow

1. Write the CLI tool in `tools/`
2. Add `openclaw-skills/<skill-name>/SKILL.md`
3. Install with `openclaw skills install`
4. Confirm with `openclaw skills list`
5. Use the tool from OpenClaw agent prompts
6. Repeat for additional tools
