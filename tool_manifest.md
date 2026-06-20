### # Local Tool Command Helper (Updated)

**Available local CLI tools:**

1. `python C:/Amir/GitHub/agentic-skin-lesion-classifier/tools/skin_lesion_fast.py --image <path> [--metadata <json>]`
* **Model:** ConvNeXt-Small (Optimized for speed)
* **Purpose:** First-pass screening (High speed)


2. `python C:/Amir/GitHub/agentic-skin-lesion-classifier/tools/skin_lesion_mid.py --image <path> [--metadata <json>]`
* **Model:** Swin-Tiny (Balanced accuracy)
* **Purpose:** Reliable secondary verification


3. `python C:/Amir/GitHub/agentic-skin-lesion-classifier/tools/skin_lesion_high.py --image <path> [--metadata <json>]`
* **Model:** ViT-Large (SOTA accuracy)
* **Purpose:** Critical analysis for low-confidence results



---

### **Execution Guidance:**

* **Strategy:**
* **Level 1 (Fast):** Always start with `skin_lesion_fast.py`. This layer is sufficient for the majority of images and is highly performant.
* **Level 2 (Escalation):** If the `confidence_score` from the Fast tier is below **0.75**, execute `skin_lesion_mid.py` for secondary verification.
* **Level 3 (Critical):** If the Mid tier result remains ambiguous (Confidence < 0.70) or if the user explicitly requests high clinical precision, execute `skin_lesion_high.py`.


* **Rules:**
* Execute the exact commands provided above. Do not invent shell aliases.
* In the final response, clearly identify the model tier used (`tier1_fast`, `tier2_mid`, or `tier3_high`).
* Return the final result as concise Markdown and include the exact command executed.



---
