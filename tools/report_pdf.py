"""PDF report exporter: round Markdown -> PDF for the physician.

Stdlib-only minimal PDF writer (no new dependencies, fully offline):
headings, paragraphs with word-wrap, bullets, rules, footer with page
numbers + research-use disclaimer. Uniform JSON contract on stdout; the
PDF is the artifact.
"""

import argparse
import json
import os
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

PAGE_W, PAGE_H = 595.0, 842.0  # A4 in points
MARGIN = 56.0
FOOTER_Y = 36.0
DISCLAIMER = ("Research use only — not a medical device. "
              "Physician review required.")


def fail(message: str) -> int:
    print(json.dumps({"status": "error", "message": message}))
    return 1


def sanitize(text: str) -> str:
    """WinAnsi (Latin-1) only in the minimal writer; replace the rest."""
    return text.encode("latin-1", errors="replace").decode("latin-1")


def pdf_escape(text: str) -> str:
    return sanitize(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def parse_markdown(text: str) -> list[tuple[str, str]]:
    """Split markdown into (kind, text) blocks: h1/h2/h3/li/p/rule."""
    blocks: list[tuple[str, str]] = []
    para: list[str] = []
    hr_re = re.compile(r"^\s*(-{3,}|\*{3,}|_{3,})\s*$")
    head_re = re.compile(r"^(#{1,3})\s+(.*)")

    def flush() -> None:
        if para:
            blocks.append(("p", " ".join(para)))
            para.clear()

    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            flush()
            continue
        if hr_re.match(line):
            flush()
            blocks.append(("rule", ""))
            continue
        head = head_re.match(line)
        if head:
            flush()
            blocks.append((f"h{len(head.group(1))}", head.group(2).strip()))
            continue
        stripped = line.strip()
        if stripped.startswith(("- ", "* ")):
            flush()
            blocks.append(("li", stripped[2:].strip()))
            continue
        if re.match(r"^\d+\.\s+", stripped):
            flush()
            blocks.append(("li", re.sub(r"^\d+\.\s+", "", stripped)))
            continue
        # Strip inline markdown noise for the plain renderer.
        clean = re.sub(r"(\*\*|__)(.*?)\1", r"\2", stripped)
        clean = re.sub(r"(`)(.*?)\1", r"\2", clean)
        para.append(clean)
    flush()
    return blocks


class PdfWriter:
    def __init__(self) -> None:
        self.objects: list[bytes] = []
        self.page_count = 0
        self.font_id = 1  # filled in build()

    def add(self, body: bytes) -> int:
        self.objects.append(body)
        return len(self.objects)

    @staticmethod
    def wrap(text: str, font_size: float, max_w: float) -> list[str]:
        # Helvetica ~0.5 * size average glyph width; conservative wrap.
        max_chars = max(20, int(max_w / (font_size * 0.5)))
        words, lines, cur = text.split(), [], ""
        for word in words:
            trial = f"{cur} {word}".strip()
            if len(trial) <= max_chars:
                cur = trial
            else:
                if cur:
                    lines.append(cur)
                while len(word) > max_chars:
                    lines.append(word[:max_chars])
                    word = word[max_chars:]
                cur = word
        if cur:
            lines.append(cur)
        return lines or [""]

    def build(self, title: str, blocks: list[tuple[str, str]]) -> bytes:
        title = sanitize(title)
        font_ref = self.add(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        self.font_id = font_ref
        content_streams: list[bytes] = []
        current: list[str] = []
        usable_w = PAGE_W - 2 * MARGIN

        def new_stream() -> None:
            current.clear()

        def text_line(x: float, y: float, size: float, text: str) -> None:
            current.append(f"BT /F1 {size:.1f} Tf {x:.1f} {y:.1f} Td ({pdf_escape(text)}) Tj ET")

        def rule_line(y: float) -> None:
            current.append(f"{MARGIN:.1f} {y:.1f} m {PAGE_W - MARGIN:.1f} {y:.1f} l S")

        new_stream()
        y = PAGE_H - MARGIN
        page_no = 1

        def page_break() -> None:
            nonlocal y, page_no
            content_streams.append("\n".join(current).encode("latin-1"))
            new_stream()
            page_no += 1
            y = PAGE_H - MARGIN

        def need(lines: int, size: float) -> None:
            nonlocal y
            if y - lines * (size * 1.35) < MARGIN + 20:
                page_break()

        def footer(n: int) -> list[str]:
            text = pdf_escape(f"{DISCLAIMER}  --  p.{n}")
            return [
                f"BT /F1 8.0 Tf {MARGIN:.1f} {FOOTER_Y:.1f} Td "
                f"({text}) Tj ET"
            ]

        for kind, text in blocks:
            if kind == "h1":
                need(2, 18)
                y -= 8
                for ln in self.wrap(text, 18, usable_w):
                    need(1, 18)
                    text_line(MARGIN, y, 18, ln)
                    y -= 24
                y -= 4
            elif kind == "h2":
                need(2, 14)
                y -= 6
                for ln in self.wrap(text, 14, usable_w):
                    need(1, 14)
                    text_line(MARGIN, y, 14, ln)
                    y -= 19
                y -= 2
            elif kind == "h3":
                need(2, 12)
                for ln in self.wrap(text, 12, usable_w):
                    need(1, 12)
                    text_line(MARGIN, y, 12, ln)
                    y -= 16
            elif kind == "li":
                for ln in self.wrap(f"• {text}", 10, usable_w - 12):
                    need(1, 10)
                    text_line(MARGIN + 12, y, 10, ln)
                    y -= 14
                y -= 2
            elif kind == "rule":
                need(1, 10)
                rule_line(y)
                y -= 16
            else:
                for ln in self.wrap(text, 10, usable_w):
                    need(1, 10)
                    text_line(MARGIN, y, 10, ln)
                    y -= 14
                y -= 4
        content_streams.append("\n".join(current).encode("latin-1"))
        self.page_count = len(content_streams)

        # Assemble objects: content streams + pages + catalog.
        stream_ids = [self.add(b"<< /Length %d >>\nstream\n" % len(s) + s + b"\nendstream")
                      for s in content_streams]
        page_ids = []
        for i, sid in enumerate(stream_ids, start=1):
            # Footer baked per page stream via separate small stream.
            foot = "\n".join(footer(i)).encode("latin-1")
            fid = self.add(b"<< /Length %d >>\nstream\n" % len(foot) + foot + b"\nendstream")
            page_ids.append(self.add(
                ("<< /Type /Page /Parent 0 /MediaBox [0 0 %.0f %.0f] "
                 "/Resources << /Font << /F1 %d 0 R >> >> "
                 "/Contents [%d 0 R %d 0 R] >>" % (PAGE_W, PAGE_H, font_ref, sid, fid)).encode()
            ))
        # Patch Parent refs now that the Pages object id is known.
        pages_id = len(self.objects) + 1
        kids = " ".join(f"{pid} 0 R" for pid in page_ids)
        self.objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(page_ids)} >>".encode())
        for pid in page_ids:
            idx = pid - 1
            self.objects[idx] = self.objects[idx].replace(b"/Parent 0", f"/Parent {pages_id}".encode())
        catalog_id = self.add(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode())
        info_id = self.add(f"<< /Title ({pdf_escape(title)}) /Producer (agentic-skin-lesion-classifier) >>".encode())

        out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for i, body in enumerate(self.objects, start=1):
            offsets.append(len(out))
            out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
        xref_at = len(out)
        out += f"xref\n0 {len(self.objects) + 1}\n".encode()
        out += b"0000000000 65535 f \n"
        for off in offsets[1:]:
            out += f"{off:010d} 00000 n \n".encode()
        out += (f"trailer\n<< /Size {len(self.objects) + 1} "
                f"/Root {catalog_id} 0 R /Info {info_id} 0 R >>\n"
                f"startxref\n{xref_at}\n%%EOF").encode()
        return bytes(out)


def convert(markdown_path: str, pdf_path: Path) -> dict:
    text = Path(markdown_path).read_text(encoding="utf-8")
    blocks = parse_markdown(text)
    if not blocks:
        return {"ok": False, "error": "empty markdown"}
    title = Path(markdown_path).stem
    for kind, content in blocks:
        if kind == "h1":
            title = content
            break
    writer = PdfWriter()
    pdf_bytes = writer.build(title, blocks)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(pdf_bytes)
    pages = writer.page_count or pdf_bytes.count(b"/Type /Page ")
    writer.page_count = pages
    return {"ok": True, "pages": pages,
            "blocks": len(blocks), "bytes": len(pdf_bytes)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Round Markdown -> PDF exporter.")
    parser.add_argument("--input", dest="input_path", required=True)
    parser.add_argument("--output", dest="output_path", required=True)
    args = parser.parse_args()
    if not os.path.exists(args.input_path):
        return fail(f"Markdown not found: {args.input_path}")
    out_path = Path(args.output_path).expanduser()
    if not out_path.is_absolute():
        out_path = BASE_DIR / out_path
    try:
        result = convert(args.input_path, out_path)
    except Exception as exc:  # noqa: BLE001 - CLI must report, not crash
        return fail(f"Conversion failed: {exc}")
    if not result.get("ok"):
        return fail(str(result.get("error")))
    rel = (str(out_path.relative_to(BASE_DIR))
           if out_path.is_relative_to(BASE_DIR) else str(out_path))
    print(json.dumps({
        "status": "success",
        "tool": "report-pdf",
        "model_tier": "reporting",
        "model_executed": "stdlib_markdown_to_pdf",
        "input": args.input_path,
        "output": rel,
        "pages": result["pages"],
        "blocks": result["blocks"],
        "bytes": result["bytes"],
        "uncertainty_flags": [],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
