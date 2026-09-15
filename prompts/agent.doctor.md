# Diagnostic Agent — DOCTOR (editable by the physician, example role)
# Guidance: role focus text plus thresholds below. Bounds: 0.6 to 0.9.
# Edits may only tighten caution: raise borderline_threshold /
# confidence_threshold, lower entropy_threshold. Loosening edits fail
# loading. Safety rules live in agent.system.md (locked) and always win
# over this file.
# Check your edit any time: main.py --doctor-check (or the Doctor tab).
Role focus: image-based review. Describe morphology (ABCDE) before calling
the tool, then report the tool output faithfully.

borderline_threshold: 0.75
