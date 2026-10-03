"""Tkinter desktop app for the Agentic Skin Lesion Classifier.

Cases-only browser: pick a case, see its image, metadata, and whether a
round report exists. Screening rounds run via the CLI (`main.py`);
reports land in `report/`.

Usage: .venv/bin/python gui.py
"""

import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import tkinter as tk
from tkinter import messagebox, ttk

import yaml
from PIL import Image

from control.chat import latest_report

PROMPT_FILE = ROOT_DIR / "prompt.yaml"
THUMB_SIZE = (300, 300)

# ----- palette -----
BG = "#f2f4f8"
CARD = "#ffffff"
TEXT = "#111928"
MUTED = "#6b7280"
TITLE_FONT = ("TkDefaultFont", 13, "bold")
HEAD_FONT = ("TkDefaultFont", 16, "bold")


def load_records() -> list[dict]:
    records = yaml.safe_load(PROMPT_FILE.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("prompt.yaml must hold a list of records.")
    return records


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Skin Lesion Classifier — Cases")
        self.geometry("460x720")
        self.minsize(380, 550)
        self.configure(bg=BG)
        self._style()
        try:
            self.records = load_records()
            for i, rec in enumerate(self.records):
                rec["_index"] = i
        except Exception as exc:  # noqa: BLE001 - show and exit cleanly
            messagebox.showerror("Startup failed", str(exc))
            raise SystemExit(1)
        self.photo: tk.PhotoImage | None = None
        self.tmpdir = Path(tempfile.mkdtemp(prefix="skin_gui_"))
        self.preview_file = self.tmpdir / "preview.ppm"
        self._build()
        self._refresh_cases()

    # ----- chrome -----
    def _style(self) -> None:
        style = ttk.Style(self)
        for theme in ("clam", "alt", "default"):
            if theme in style.theme_names():
                style.theme_use(theme)
                break
        style.configure(".", background=BG, foreground=TEXT)
        style.configure("TFrame", background=BG)
        style.configure("Card.TFrame", background=CARD)
        style.configure("TLabel", background=BG, foreground=TEXT)
        style.configure("Card.TLabel", background=CARD, foreground=TEXT)
        style.configure("Muted.TLabel", background=BG, foreground=MUTED)
        style.configure("CardMuted.TLabel", background=CARD,
                        foreground=MUTED)

    def _build(self) -> None:
        header = ttk.Frame(self)
        header.pack(fill=tk.X, padx=12, pady=(10, 4))
        ttk.Label(header, text="Cases",
                  font=HEAD_FONT).pack(side=tk.LEFT)
        ttk.Label(header, text=f"  {len(self.records)} cases",
                  style="Muted.TLabel").pack(side=tk.LEFT, pady=6)
        ttk.Label(header, text="Research use only",
                  style="Muted.TLabel").pack(side=tk.RIGHT)

        self._case_card().pack(fill=tk.BOTH, expand=True, padx=12)

        self.status = ttk.Label(self, text="Select a case",
                                style="Muted.TLabel", anchor=tk.W)
        self.status.pack(fill=tk.X, padx=12, pady=(4, 8))

    def _case_card(self) -> ttk.Frame:
        card = ttk.Frame(style="Card.TFrame", padding=8)
        self.search_var = tk.StringVar()
        self.search_var.trace_add("write", lambda *_: self._refresh_cases())
        search = ttk.Entry(card, textvariable=self.search_var)
        search.pack(fill=tk.X, pady=(4, 6))
        search.insert(0, "Filter cases…")
        cols = ("case", "info", "status")
        self.cases = ttk.Treeview(card, columns=cols, show="headings",
                                  height=16, selectmode="browse")
        self.cases.heading("case", text="Case")
        self.cases.heading("info", text="Age / Sex")
        self.cases.heading("status", text="Report")
        self.cases.column("case", width=140)
        self.cases.column("info", width=100)
        self.cases.column("status", width=60, anchor=tk.CENTER)
        self.cases.pack(fill=tk.BOTH, expand=True)
        self.cases.bind("<<TreeviewSelect>>", self._on_case)
        self.preview = ttk.Label(card, text="Select a case",
                                 style="Card.TLabel", anchor=tk.CENTER)
        self.preview.pack(pady=6)
        self.meta = ttk.Label(card, text="", wraplength=380,
                              justify=tk.LEFT, style="CardMuted.TLabel")
        self.meta.pack(anchor=tk.W)
        return card

    # ----- cases -----
    def _selected(self) -> dict | None:
        sel = self.cases.selection()
        if not sel:
            return None
        return self.records[int(sel[0])]

    def _filtered(self) -> list[dict]:
        query = self.search_var.get().strip().lower()
        if not query or query == "filter cases…":
            return self.records
        return [r for r in self.records
                if query in Path(r["image_path"]).stem.lower()]

    def _refresh_cases(self) -> None:
        if not hasattr(self, "cases"):
            return  # search box initialised before the treeview
        keep = self._selected()
        self.cases.delete(*self.cases.get_children())
        for rec in self._filtered():
            meta = rec.get("metadata", {})
            done = "●" if latest_report(rec["_index"]) else "○"
            self.cases.insert("", tk.END, iid=str(rec["_index"]), values=(
                Path(rec["image_path"]).stem,
                f"{meta.get('age')}, {meta.get('sex')}", done))
        if keep is not None:
            try:
                self.cases.selection_set(str(keep["_index"]))
            except tk.TclError:
                pass

    def _on_case(self, _event=None) -> None:
        rec = self._selected()
        if rec is None:
            return
        self.meta.configure(text=" · ".join(
            f"{k}: {v}" for k, v in rec.get("metadata", {}).items()))
        self.status.configure(
            text=f"Case {rec['_index']}: {Path(rec['image_path']).stem}")
        try:
            # No ImageTk on this runtime: PPM round-trip, natively readable.
            img = Image.open(ROOT_DIR / rec["image_path"]).convert("RGB")
            img.thumbnail(THUMB_SIZE)
            img.save(self.preview_file, "PPM")
            self.photo = tk.PhotoImage(file=str(self.preview_file))
            self.preview.configure(image=self.photo, text="")
        except Exception as exc:  # noqa: BLE001 - preview is best-effort
            self.preview.configure(text=f"Preview failed: {exc}")


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
