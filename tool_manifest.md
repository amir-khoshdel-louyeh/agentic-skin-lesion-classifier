# Local Tool Command Helper & Manifest (LEGACY)
> Canonical source is `tools/manifest.yaml`. This file is kept only for
> backward compatibility with old prompts; new code defaults to the YAML
> manifest. Voting tools are `ham10000_cnn` (triage), `multimodal-fusion`
> (image+metadata), `ensemble-high` (CNN+fusion mean, entropy-gated).

You have access to the following local CLI tools for skin lesion analysis. You must invoke them using exact absolute paths and forward slashes as defined below.

## Available Core Tools

### 1. Tier 1: Fast Screening Classifier
* **Command:** `python tools/skin_lesion_fast.py --image <path> [--metadata <json>]`
* **Model:** SkinCNN triage (`models/derm-cnn`, 28×28 input, `derm_cnn_ham10000`)
* **Purpose:** Initial first-pass screening.

### 2. Tier 2: Mid Verification Classifier
* **Command:** `python tools/skin_lesion_mid.py --image <path> [--metadata <json>]`
* **Model:** DermAI EfficientNet-B0 (`models/dermai-b0`, 224px, `dermai_efficientnet_b0_ham10000`)
* **Purpose:** Reliable secondary verification for ambiguous cases.

### 3. Tier 3: High Precision Classifier
* **Command:** `python tools/skin_lesion_high.py --image <path> [--metadata <json>]`
* **Model:** Cross-attention image+metadata fusion (`models/multimodal/best.pt`, 224px, `cross_attention_fusion_ham10000`)
* **Purpose:** Critical analysis required only when lower tiers return sub-threshold confidence. Output adds `entropy`, `entropy_threshold`, and `uncertainty_flags` (entropy-gated; uncertain cases carry `"borderline"`).
* **Coverage:** 7 HAM10000 classes only. Squamous cell carcinoma (SCC) is out of coverage: SCC entropy is indistinguishable from in-coverage cases (median 1.07 vs 1.08 on 215 local images), so no detector flag exists — uncertain cases are referred, never forced into a near class.

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