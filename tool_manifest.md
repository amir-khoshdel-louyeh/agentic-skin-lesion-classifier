# Local Tool Command Helper & Manifest

You have access to the following local CLI tools for skin lesion analysis. You must invoke them using exact absolute paths and forward slashes as defined below.

## Available Core Tools

### 1. Tier 1: Fast Screening Classifier
* **Command:** `python C:/Amir/GitHub/agentic-skin-lesion-classifier/tools/skin_lesion_fast.py --image <path> [--metadata <json>]`
* **Model:** ConvNeXt-Small (Optimized for speed)
* **Purpose:** Initial first-pass screening.

### 2. Tier 2: Mid Verification Classifier
* **Command:** `python C:/Amir/GitHub/agentic-skin-lesion-classifier/tools/skin_lesion_mid.py --image <path> [--metadata <json>]`
* **Model:** Swin-Tiny (Balanced accuracy)
* **Purpose:** Reliable secondary verification for ambiguous cases.

### 3. Tier 3: High Precision Classifier
* **Command:** `python C:/Amir/GitHub/agentic-skin-lesion-classifier/tools/skin_lesion_high.py --image <path> [--metadata <json>]`
* **Model:** ViT-Large (SOTA accuracy)
* **Purpose:** Critical analysis required only when lower tiers return sub-threshold confidence.

---

## Execution & Escalation Protocol

You must strictly follow this multi-tiered escalation logic based on the tool outputs:

1. **Step 1 (Initial Run):** Always invoke `skin_lesion_fast.py` first.
2. **Step 2 (First Escalation):** Check the `confidence_score` in the JSON output of the Fast tool. If `confidence_score` is **less than 0.75**, you must immediately escalate and execute `skin_lesion_mid.py`.
3. **Step 3 (Critical Escalation):** Check the output of the Mid tool. If the confidence remains low (**less than 0.70**), or if the user prompt explicitly demands maximum clinical rigor, escalate to `skin_lesion_high.py`.

## Strict Output Formatting Rules

* **No Commentary:** Do not output any intermediate planning text, shell execution thoughts, or progress messages (e.g., "Running tool...", "Please wait...").
* **Command Visibility:** Your final response must explicitly start with a shell code block showing the exact command that was executed.
* **Tier Identification:** Clearly state which tier was used to generate the final result (`tier1_fast`, `tier2_mid`, or `tier3_high`).
* **Format:** Present the final conclusion as a concise Markdown clinical report.