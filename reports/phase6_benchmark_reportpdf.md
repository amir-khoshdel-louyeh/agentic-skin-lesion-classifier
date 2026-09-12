# Phase 6 Benchmark — report-pdf (reporting)

Round Markdown → PDF exporter for the physician. Stdlib-only writer
(no new dependencies, fully offline): headings, wrapped paragraphs,
bullets, rules, footer with page numbers + research-use disclaimer.

- Inputs attempted: 7 tracked reports/*.md, converted: 7, failures: 0
- Validity: 7/7 (header `%PDF-1.4`, `%%EOF`, xref offsets + stream
  lengths verified by re-parsing)
- Avg seconds/conversion: 0.03
- Non-Latin-1 glyphs (em-dashes, bullets) sanitized deterministically;
  no crashes on real report text

## Failures

- none (one dev-time encoding bug on em-dash footer text, fixed before
  benchmarking)

## Role assignment

- Assigned: reporting tier, agent-loop terminal step. After the round
  report Markdown is written, the agent runs `python
  tools/report_pdf.py --input <round.md> --output <round.pdf>` and
  cites the receipt; the PDF path goes to the physician alongside the
  Markdown source (source stays canonical, PDF is the hand-off copy).
- Also fixed in this step: `control/manifest.py render_for_agents`
  rendered every tool as `--image <path>`, which is wrong for
  non-image categories. Usage is now per-category (`preprocessing`,
  `reporting`, `notification`); image tools unchanged.
- Manifest: category `reporting`, requires `report`, produces
  `pdf_report`. Status ready, calibrated false, 0.0 GB VRAM.
