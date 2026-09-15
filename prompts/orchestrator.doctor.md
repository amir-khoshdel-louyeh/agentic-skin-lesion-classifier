# Orchestrator — DOCTOR (editable by the physician)
# Guidance: free text (role emphasis, report tone, priorities) plus
# thresholds below. Bounds for every threshold: 0.6 to 0.9. Edits may only
# tighten caution: raise borderline_threshold / confidence_threshold, lower
# entropy_threshold. Loosening edits fail loading. Safety rules live in
# orchestrator.system.md (locked) and always win over this file.
# Check your edit any time: main.py --doctor-check (or the Doctor tab).
Report tone: concise clinical Persian-friendly summary with English terms.
Physician priorities: highlight asymmetry and color variation first.

borderline_threshold: 0.75
entropy_threshold: 0.9
