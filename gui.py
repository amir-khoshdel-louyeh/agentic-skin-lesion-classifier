"""Tkinter desktop app for the Agentic Skin Lesion Classifier.

Thin stdlib-only layer (+ Pillow for thumbnails, already a dependency):
the CLI stays canonical; every action calls the same functions and CLIs
(main.py / orchestrator.run_round / control.chat / tools/*.py). Long
runs execute in worker threads with results marshalled back via queue.

Usage: .venv/bin/python gui.py
"""

import json
import queue
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import yaml
from PIL import Image

from control.chat import audit_for_record, build_debate_prompt, latest_report
from control.manifest import load_manifest

PROMPT_FILE = ROOT_DIR / "prompt.yaml"
MANIFEST_FILE = ROOT_DIR / "tools" / "manifest.yaml"
REPORT_DIR = ROOT_DIR / "report"
THUMB_SIZE = (300, 300)

# ----- palette -----
BG = "#f2f4f8"
CARD = "#ffffff"
ACCENT = "#1a56db"
ACCENT_DARK = "#1e429f"
TEXT = "#111928"
MUTED = "#6b7280"
GREEN = "#057a55"
AMBER = "#c27803"
AMBER_BG = "#fdf6b2"
GREEN_BG = "#def7ec"
GRAY_BG = "#e5e7eb"
USER_BUBBLE = "#e1effe"
ASSIST_BUBBLE = "#ffffff"
MONO = ("TkFixedFont", 9)
TITLE_FONT = ("TkDefaultFont", 13, "bold")
HEAD_FONT = ("TkDefaultFont", 16, "bold")

TOOL_BLURBS = {
    "ham10000-cnn": "Fast triage CNN — first pass, seconds per image.",
    "quality-gate": "Blur / exposure / resolution check before diagnosis.",
    "multimodal-fusion": "Image + age/sex second opinion, calibrated.",
    "abcde-analyzer": "Deterministic shape/color heuristic (support flag).",
    "preprocess": "Hair removal + denoise; cleaned copy for re-check.",
    "report-pdf": "Export the latest round report to PDF.",
    "notify": "Send a desktop toast (completion / failure).",
    "ensemble-high": "High tier: CNN + multimodal mean with entropy gate.",
}

DECISION_STYLE = {
    "suspicious": ("REQUIRES PHYSICIAN REVIEW", AMBER_BG, AMBER),
    "benign": ("NO MALIGNANCY SIGNAL", GREEN_BG, GREEN),
    "inconclusive": ("NO EVIDENCE — RERUN", GRAY_BG, MUTED),
}


def load_records() -> list[dict]:
    records = yaml.safe_load(PROMPT_FILE.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("prompt.yaml must hold a list of records.")
    return records


def parse_decision(report_text: str) -> tuple[str, str]:
    """Extract (decision, reason) from a round report; unknown if absent."""
    match = re.search(r"- Decision:\s*\*\*(\w+)\*\*\s*[—-]\s*(.+)",
                      report_text)
    if match:
        return match.group(1).strip().lower(), match.group(2).strip()
    return "unknown", ""


def summarize_tool(tool_name: str, payload: dict) -> tuple[str, float | None,
                                                           list[tuple[str, str]]]:
    """Readable (headline, confidence 0..1|None, detail rows) for tool JSON."""
    rows: list[tuple[str, str]] = []
    conf = payload.get("confidence_score")
    conf = float(conf) if isinstance(conf, (int, float)) else None
    headline = str(payload.get("disease_name") or payload.get("status",
                                                              "")).upper()
    if tool_name == "quality-gate":
        headline = "PASS — IMAGE USABLE" if payload.get(
            "quality_pass") else "FAIL — DO NOT DIAGNOSE"
        metrics = payload.get("metrics", {})
        rows = [(k, str(v)) for k, v in metrics.items()]
    elif tool_name == "abcde-analyzer":
        headline = (f"ABCDE {payload.get('abcde_score')} — "
                    f"{str(payload.get('risk_band', '')).upper()}")
        sub = (payload.get("abcde") or {}).get("subscores", {})
        rows = [(f"Subscore {k}", str(v)) for k, v in sub.items()]
    elif tool_name == "preprocess":
        headline = "CLEANED IMAGE READY"
        rows = [("Steps", ", ".join(payload.get("steps_applied", []))),
                ("Hair px inpainted",
                 str(payload.get("hair_pixels_inpainted"))),
                ("Quality before",
                 ", ".join(payload.get("quality_before") or ["pass"])),
                ("Quality after",
                 ", ".join(payload.get("quality_after") or ["pass"])),
                ("Output", str(payload.get("output")))]
    elif tool_name == "report-pdf":
        headline = "PDF EXPORTED"
        rows = [("Pages", str(payload.get("pages"))),
                ("Size", f"{payload.get('bytes', 0) / 1024:.1f} KB"),
                ("Output", str(payload.get("output")))]
    elif tool_name == "notify":
        headline = ("DELIVERED" if payload.get("delivered")
                    else "QUEUED (headless fallback)")
        rows = [("Backend", str(payload.get("backend")))]
    elif tool_name == "ensemble-high":
        members = payload.get("members", {})
        rows = [("Entropy",
                 f"{payload.get('entropy')} "
                 f"(gate at {payload.get('entropy_threshold')})"),
                ("CNN vote",
                 f"{members.get('ham10000-cnn', {}).get('class')} "
                 f"({members.get('ham10000-cnn', {}).get('confidence')})"),
                ("Fusion vote",
                 f"{members.get('multimodal-fusion', {}).get('class')} "
                 f"({members.get('multimodal-fusion', {}).get('confidence')})")]
    else:
        rows = [(k, str(v)) for k, v in payload.items()
                if k not in ("status", "tool", "metadata")
                and not isinstance(v, (dict, list))]
    flags = payload.get("uncertainty_flags") or []
    if flags:
        rows.append(("Flags", ", ".join(flags)))
    return headline, conf, rows


def tool_command(tool, record: dict, extra: dict) -> list[str]:
    """Build the CLI argv for a manifest tool on the selected record."""
    cmd = [sys.executable, str(ROOT_DIR / tool.command)]
    image = str(ROOT_DIR / record["image_path"])
    if tool.category == "reporting":
        md = extra.get("markdown") or str(latest_report(record["_index"]))
        cmd += ["--input", md, "--output", str(Path(md).with_suffix(".pdf"))]
    elif tool.category == "notification":
        cmd += ["--message", extra.get("message", "GUI ping"),
                "--level", extra.get("level", "info")]
    else:
        cmd += ["--image", image]
        if tool.category == "preprocessing":
            out = Path(extra.get("output_dir")) / (
                Path(image).stem + "_cleaned.jpg")
            cmd += ["--output", str(out), "--steps",
                    extra.get("steps", "hair,denoise")]
        if "metadata" in tool.requires:
            cmd += ["--metadata", json.dumps(record.get("metadata", {}))]
    return cmd


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Skin Lesion Classifier — Physician Console")
        self.geometry("1180x800")
        self.minsize(1000, 650)
        self.configure(bg=BG)
        self._style()
        try:
            self.records = load_records()
            for i, rec in enumerate(self.records):
                rec["_index"] = i
            self.manifest = load_manifest(MANIFEST_FILE)
        except Exception as exc:  # noqa: BLE001 - show and exit cleanly
            messagebox.showerror("Startup failed", str(exc))
            raise SystemExit(1)
        self.results: queue.Queue = queue.Queue()
        self.photo: tk.PhotoImage | None = None
        self.tmpdir = Path(tempfile.mkdtemp(prefix="skin_gui_"))
        self.preview_file = self.tmpdir / "preview.ppm"
        self._build()
        self._refresh_cases()
        self._refresh_reports()
        self.after(200, self._poll)

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
        style.configure("Accent.TButton", background=ACCENT,
                        foreground="white", padding=8)
        style.map("Accent.TButton", background=[("active", ACCENT_DARK)])
        style.configure("TButton", padding=6)
        style.configure("TNotebook", background=BG)
        style.configure("TNotebook.Tab", padding=(14, 6))
        style.configure("Horizontal.TProgressbar")

    def _build(self) -> None:
        self._menu()
        header = ttk.Frame(self)
        header.pack(fill=tk.X, padx=12, pady=(10, 4))
        title = ttk.Label(header, text="Skin Lesion Classifier",
                          font=HEAD_FONT)
        title.pack(side=tk.LEFT)
        ready = sum(1 for t in self.manifest.tools
                    if t.status == "ready")
        ttk.Label(header,
                  text=f"  {len(self.records)} cases · {ready} tools ready · "
                       "local only",
                  style="Muted.TLabel").pack(side=tk.LEFT, pady=6)
        ttk.Label(header, text="Research use only — not a medical device",
                  style="Muted.TLabel").pack(side=tk.RIGHT)

        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=12)
        paned.add(self._case_card(), weight=1)
        right = ttk.Frame(self)
        paned.add(right, weight=4)
        self.tabs = ttk.Notebook(right)
        self.tabs.pack(fill=tk.BOTH, expand=True)
        self._build_round_tab()
        self._build_tools_tab()
        self._build_reports_tab()
        self._build_chat_tab()

        self.status = ttk.Label(self, text="Ready", style="Muted.TLabel",
                                anchor=tk.W)
        self.status.pack(fill=tk.X, padx=12, pady=(4, 8))

    def _menu(self) -> None:
        bar = tk.Menu(self)
        file_menu = tk.Menu(bar, tearoff=False)
        file_menu.add_command(label="Export round PDF",
                              command=self._export_pdf)
        file_menu.add_separator()
        file_menu.add_command(label="Quit", command=self.destroy)
        bar.add_cascade(label="File", menu=file_menu)
        help_menu = tk.Menu(bar, tearoff=False)
        help_menu.add_command(label="About", command=self._about)
        bar.add_cascade(label="Help", menu=help_menu)
        self.configure(menu=bar)

    def _about(self) -> None:
        messagebox.showinfo(
            "About",
            "Agentic Skin Lesion Classifier — physician console.\n\n"
            "Orchestrates local CNN + multimodal tools through verified "
            "receipts. Every verdict is re-executed control-side.\n\n"
            "Research use only. Not a medical device.")

    def _case_card(self) -> ttk.Frame:
        card = ttk.Frame(style="Card.TFrame", padding=8)
        ttk.Label(card, text="Cases", font=TITLE_FONT,
                  style="Card.TLabel").pack(anchor=tk.W)
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
        self.cases.column("case", width=110)
        self.cases.column("info", width=80)
        self.cases.column("status", width=60, anchor=tk.CENTER)
        self.cases.pack(fill=tk.BOTH, expand=True)
        self.cases.bind("<<TreeviewSelect>>", self._on_case)
        self.preview = ttk.Label(card, text="Select a case",
                                 style="Card.TLabel", anchor=tk.CENTER)
        self.preview.pack(pady=6)
        self.meta = ttk.Label(card, text="", wraplength=250,
                              justify=tk.LEFT, style="CardMuted.TLabel")
        self.meta.pack(anchor=tk.W)
        return card

    # ----- round tab -----
    def _build_round_tab(self) -> None:
        tab = ttk.Frame(self)
        self.tabs.add(tab, text="  Round  ")
        bar = ttk.Frame(tab)
        bar.pack(fill=tk.X, padx=10, pady=8)
        self.run_btn = ttk.Button(bar, text="▶  Run screening round",
                                  style="Accent.TButton",
                                  command=self._run_round)
        self.run_btn.pack(side=tk.LEFT)
        ttk.Button(bar, text="Export PDF",
                   command=self._export_pdf).pack(side=tk.LEFT, padx=8)
        self.progress = ttk.Progressbar(bar, mode="indeterminate",
                                        length=180)
        self.banner = tk.Label(tab, text="No round yet — pick a case and "
                               "press Run.", font=TITLE_FONT, bg=GRAY_BG,
                               fg=MUTED, padx=12, pady=10, anchor=tk.W)
        self.banner.pack(fill=tk.X, padx=10, pady=(0, 6))
        hint = ttk.Label(tab, text="Runs triage → fusion → high-tier agents "
                         "sequentially (minutes). The decision banner above "
                         "is the only thing that matters first.",
                         style="Muted.TLabel", wraplength=700,
                         justify=tk.LEFT)
        hint.pack(anchor=tk.W, padx=10)
        self.report_view = tk.Text(tab, wrap=tk.WORD, font=MONO, bg=CARD,
                                   relief=tk.FLAT, padx=8, pady=8)
        self.report_view.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

    def _set_banner(self, decision: str, reason: str) -> None:
        label, bg, fg = DECISION_STYLE.get(
            decision, ("UNKNOWN", GRAY_BG, MUTED))
        text = f"{label}" + (f"  ·  {reason}" if reason else "")
        self.banner.configure(text=text, bg=bg, fg=fg)

    # ----- tools tab -----
    def _build_tools_tab(self) -> None:
        tab = ttk.Frame(self)
        self.tabs.add(tab, text="  Tools  ")
        bar = ttk.Frame(tab)
        bar.pack(fill=tk.X, padx=10, pady=8)
        ready = [t for t in self.manifest.tools if t.status == "ready"]
        self.tool_names = [t.name for t in ready]
        self.tool_by_name = {t.name: t for t in ready}
        self.tool_var = tk.StringVar(value=self.tool_names[0])
        menu = ttk.OptionMenu(bar, self.tool_var, self.tool_names[0],
                              *self.tool_names, command=self._on_tool_pick)
        menu.pack(side=tk.LEFT)
        ttk.Button(bar, text="Run on selected case", style="Accent.TButton",
                   command=self._run_tool).pack(side=tk.LEFT, padx=8)
        self.tool_blurb = ttk.Label(bar, text="", style="Muted.TLabel")
        self.tool_blurb.pack(side=tk.LEFT)
        self._on_tool_pick(self.tool_names[0])

        self.tool_head = tk.Label(tab, text="", font=TITLE_FONT, bg=BG,
                                  fg=TEXT, padx=12, pady=8, anchor=tk.W)
        self.tool_head.pack(fill=tk.X, padx=10)
        conf_row = ttk.Frame(tab)
        conf_row.pack(fill=tk.X, padx=10)
        ttk.Label(conf_row, text="Confidence").pack(side=tk.LEFT)
        self.conf_bar = ttk.Progressbar(conf_row, length=300, maximum=100)
        self.conf_bar.pack(side=tk.LEFT, padx=8)
        self.conf_val = ttk.Label(conf_row, text="—")
        self.conf_val.pack(side=tk.LEFT)
        self.tool_rows = ttk.Frame(tab, style="Card.TFrame", padding=8)
        self.tool_rows.pack(fill=tk.X, padx=10, pady=8)
        raw_bar = ttk.Frame(tab)
        raw_bar.pack(fill=tk.X, padx=10)
        self.raw_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(raw_bar, text="Show raw JSON",
                        variable=self.raw_var,
                        command=self._toggle_raw).pack(anchor=tk.W)
        self.tool_raw = tk.Text(tab, wrap=tk.WORD, font=MONO, height=10,
                                bg=CARD, relief=tk.FLAT)
        self._raw_payload = ""

    def _on_tool_pick(self, name: str) -> None:
        self.tool_blurb.configure(text=TOOL_BLURBS.get(name, ""))

    def _toggle_raw(self) -> None:
        if self.raw_var.get():
            self.tool_raw.pack(fill=tk.BOTH, expand=True, padx=10,
                               pady=(0, 8))
            self.tool_raw.delete("1.0", tk.END)
            self.tool_raw.insert("1.0", self._raw_payload)
        else:
            self.tool_raw.pack_forget()

    def _render_tool_result(self, tool_name: str, payload: dict,
                            ok: bool) -> None:
        for child in self.tool_rows.winfo_children():
            child.destroy()
        if not ok:
            self.tool_head.configure(
                text="TOOL FAILED", bg="#fbd5d5", fg="#9b1c1c")
            self.conf_bar.configure(value=0)
            self.conf_val.configure(text="—")
            ttk.Label(self.tool_rows, text=payload,
                      style="Card.TLabel", wraplength=700,
                      justify=tk.LEFT).pack(anchor=tk.W)
            return
        headline, conf, rows = summarize_tool(tool_name, payload)
        flagged = bool(payload.get("uncertainty_flags"))
        self.tool_head.configure(
            text=headline,
            bg=AMBER_BG if flagged else GREEN_BG,
            fg=AMBER if flagged else GREEN)
        if conf is None:
            self.conf_bar.configure(value=0)
            self.conf_val.configure(text="n/a (calibrated later)")
        else:
            self.conf_bar.configure(value=conf * 100)
            self.conf_val.configure(text=f"{conf:.2f}")
        for label, value in rows:
            row = ttk.Frame(self.tool_rows, style="Card.TFrame")
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=label, style="CardMuted.TLabel",
                      width=20).pack(side=tk.LEFT)
            ttk.Label(row, text=value, style="Card.TLabel",
                      wraplength=620, justify=tk.LEFT).pack(side=tk.LEFT)
        try:
            self._raw_payload = json.dumps(payload, ensure_ascii=False,
                                           indent=2)
        except (TypeError, ValueError):
            self._raw_payload = str(payload)
        if self.raw_var.get():
            self._toggle_raw()
            self._toggle_raw()

    # ----- reports tab -----
    def _build_reports_tab(self) -> None:
        tab = ttk.Frame(self)
        self.tabs.add(tab, text="  Reports  ")
        split = ttk.PanedWindow(tab, orient=tk.VERTICAL)
        split.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)
        top = ttk.Frame(split)
        split.add(top, weight=3)
        ttk.Label(top, text="Round reports & benchmarks — click to preview",
                  style="Muted.TLabel").pack(anchor=tk.W)
        self.report_list = tk.Listbox(top, height=8, font=MONO,
                                      relief=tk.FLAT, bg=CARD)
        self.report_list.pack(fill=tk.BOTH, expand=True, pady=4)
        self.report_list.bind("<<ListboxSelect>>", self._on_report)
        bottom = ttk.Frame(split)
        split.add(bottom, weight=2)
        ttk.Label(bottom, text="Audit receipts (selected round)",
                  style="Muted.TLabel").pack(anchor=tk.W)
        self.audit_view = tk.Text(bottom, wrap=tk.WORD, font=MONO, bg=CARD,
                                  relief=tk.FLAT, padx=8, pady=8)
        self.audit_view.pack(fill=tk.BOTH, expand=True, pady=4)

    # ----- chat tab -----
    def _build_chat_tab(self) -> None:
        tab = ttk.Frame(self)
        self.tabs.add(tab, text="  Physician chat  ")
        ttk.Label(tab, text="Grounded in the round report + receipts — the "
                  "assistant cannot go beyond the evidence.",
                  style="Muted.TLabel", wraplength=700).pack(
                      anchor=tk.W, padx=10, pady=(8, 0))
        self.chat_hist = tk.Text(tab, wrap=tk.WORD, bg=CARD, relief=tk.FLAT,
                                 state=tk.DISABLED, padx=10, pady=10)
        self.chat_hist.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)
        for tag, bg in (("user", USER_BUBBLE), ("assist", ASSIST_BUBBLE),
                        ("sys", GRAY_BG)):
            self.chat_hist.tag_configure(
                tag, background=bg, lmargin1=10, lmargin2=10, rmargin=10,
                spacing1=4, spacing3=4)
        entry_bar = ttk.Frame(tab)
        entry_bar.pack(fill=tk.X, padx=10, pady=(0, 8))
        self.chat_entry = ttk.Entry(entry_bar)
        self.chat_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.chat_entry.bind("<Return>", lambda _e: self._send_chat())
        ttk.Button(entry_bar, text="Send", style="Accent.TButton",
                   command=self._send_chat).pack(side=tk.LEFT, padx=(8, 0))
        try:
            from skin_agent import find_openclaw_executable  # noqa: E402

            find_openclaw_executable()
            self.chat_openclaw = True
        except Exception:  # noqa: BLE001 - chat needs the OpenClaw CLI
            self.chat_openclaw = False
            self._chat_line("System: OpenClaw CLI not found — chat "
                            "disabled. Rounds still work; or use "
                            "main.py --chat.", "sys")

    # ----- infra -----
    def _set_busy(self, busy: bool, text: str) -> None:
        self.status.configure(text=text)
        if busy:
            self.progress.start(12)
            self.progress.pack(side=tk.LEFT, padx=8)
        else:
            self.progress.stop()
            self.progress.pack_forget()

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

    def _run_thread(self, target, *args) -> None:
        threading.Thread(target=target, args=args, daemon=True).start()

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.results.get_nowait()
                {"round_done": self._on_round_done,
                 "tool_done": self._on_tool_done,
                 "chat_done": self._on_chat_done}[kind](payload)
        except queue.Empty:
            pass
        self.after(200, self._poll)

    # ----- cases -----
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
        try:
            # No ImageTk on this runtime: PPM round-trip, natively readable.
            img = Image.open(ROOT_DIR / rec["image_path"]).convert("RGB")
            img.thumbnail(THUMB_SIZE)
            img.save(self.preview_file, "PPM")
            self.photo = tk.PhotoImage(file=str(self.preview_file))
            self.preview.configure(image=self.photo, text="")
        except Exception as exc:  # noqa: BLE001 - preview is best-effort
            self.preview.configure(text=f"Preview failed: {exc}")

    # ----- round -----
    def _run_round(self) -> None:
        rec = self._selected()
        if rec is None:
            messagebox.showinfo("No case", "Select a case first.")
            return
        self.run_btn.configure(state=tk.DISABLED)
        self._set_busy(True, f"Running round {rec['_index']} — "
                       "agents work sequentially, this takes minutes…")
        self._run_thread(self._round_worker, rec["_index"])

    @staticmethod
    def _round_worker(index: int) -> None:
        from orchestrator import run_round  # noqa: E402

        App._queue().put(("round_done", _safe_run(index, run_round)))

    @staticmethod
    def _queue() -> queue.Queue:
        return _APP_QUEUE[0]

    def _on_round_done(self, payload: dict) -> None:
        self.run_btn.configure(state=tk.NORMAL)
        self._set_busy(False, "Ready")
        if payload.get("error"):
            self._set_banner("inconclusive", "")
            self.banner.configure(text="ROUND FAILED — see log",
                                  bg="#fbd5d5", fg="#9b1c1c")
            self.status.configure(text="Round failed")
            return
        path = Path(payload["report"])
        text = path.read_text(encoding="utf-8")
        decision, reason = parse_decision(text)
        self._set_banner(decision, reason)
        self.status.configure(text=f"Done: {path.name}")
        self.report_view.delete("1.0", tk.END)
        self.report_view.insert("1.0", text)
        self._refresh_cases()
        self._refresh_reports()

    def _export_pdf(self) -> None:
        rec = self._selected()
        if rec is None:
            messagebox.showinfo("No case", "Select a case first.")
            return
        md = latest_report(rec["_index"])
        if md is None:
            messagebox.showinfo("No report", "Run the round first, then "
                                "export.")
            return
        pdf = md.with_suffix(".pdf")
        proc = subprocess.run(
            [sys.executable, str(ROOT_DIR / "tools" / "report_pdf.py"),
             "--input", str(md), "--output", str(pdf)],
            capture_output=True, text=True, timeout=120)
        if proc.returncode == 0:
            self.status.configure(text=f"PDF exported: {pdf.name}")
            if messagebox.askyesno("PDF exported",
                                   f"{pdf.name} written. Open it now?"):
                _open_file(pdf)
        else:
            messagebox.showerror("PDF export failed", proc.stdout[-500:])

    # ----- tools -----
    def _run_tool(self) -> None:
        rec = self._selected()
        if rec is None:
            messagebox.showinfo("No case", "Select a case first.")
            return
        tool = self.tool_by_name[self.tool_var.get()]
        if tool.category == "reporting" and latest_report(
                rec["_index"]) is None:
            messagebox.showinfo("No report", "Run the round first for a "
                                "PDF source.")
            return
        if tool.category == "notification":
            message = _ask_string(self, "Notification",
                                  "Toast message:",
                                  f"Case {rec['_index']} reviewed.")
            if message is None:
                return
            self._notify_message = message
        self._set_busy(True, f"Running {tool.name}…")
        self._run_thread(self._tool_worker, tool.name, rec["_index"],
                         getattr(self, "_notify_message", "GUI ping"))

    @staticmethod
    def _tool_worker(tool_name: str, index: int, message: str) -> None:
        from control.manifest import load_manifest as _lm  # noqa: E402

        manifest = _lm(MANIFEST_FILE)
        tool = manifest.by_name(tool_name)
        rec = load_records()[index]
        rec["_index"] = index
        extra: dict = {"message": message}
        if tool.category == "preprocessing":
            extra["output_dir"] = str(App._tmpdir())
        try:
            proc = subprocess.run(
                tool_command(tool, rec, extra),
                capture_output=True, text=True, timeout=900,
                cwd=str(ROOT_DIR))
            try:
                payload = json.loads(proc.stdout)
            except (ValueError, TypeError):
                payload = proc.stdout[-6000:] or proc.stderr[-2000:]
            App._queue().put(("tool_done",
                              {"ok": proc.returncode == 0,
                               "payload": payload, "tool": tool_name}))
        except Exception as exc:  # noqa: BLE001 - report to the UI
            App._queue().put(("tool_done",
                              {"ok": False, "payload": str(exc),
                               "tool": tool_name}))

    @staticmethod
    def _tmpdir() -> Path:
        return _APP_TMPDIR[0]

    def _on_tool_done(self, payload: dict) -> None:
        self._set_busy(False, f"Tool {payload['tool']}: "
                       f"{'done' if payload['ok'] else 'failed'}")
        data = payload["payload"]
        if not isinstance(data, dict):
            data = {"disease_name": "No JSON output", "raw": str(data)}
        self._render_tool_result(payload["tool"], data, payload["ok"])

    # ----- reports -----
    def _refresh_reports(self) -> None:
        self.report_list.delete(0, tk.END)
        self._report_files = sorted(REPORT_DIR.glob("*.md"))
        for path in self._report_files:
            tag = "round" if path.name.startswith("round_") else "doc"
            self.report_list.insert(tk.END, f"[{tag}] {path.name}")

    def _on_report(self, _event=None) -> None:
        sel = self.report_list.curselection()
        if not sel:
            return
        name = self.report_list.get(sel[0]).split("] ", 1)[1]
        text = (REPORT_DIR / name).read_text(encoding="utf-8")
        self.report_view.delete("1.0", tk.END)
        self.report_view.insert("1.0", text)
        decision, reason = parse_decision(text)
        if decision != "unknown":
            self._set_banner(decision, reason)
        self.tabs.select(0)
        try:
            index = int(name.split("_")[1])
        except (IndexError, ValueError):
            self.audit_view.delete("1.0", tk.END)
            return
        lines = [json.dumps(e, ensure_ascii=False)
                 for e in audit_for_record(index)]
        self.audit_view.delete("1.0", tk.END)
        self.audit_view.insert("1.0", "\n".join(lines[-20:]))

    # ----- chat -----
    def _chat_line(self, text: str, tag: str) -> None:
        self.chat_hist.configure(state=tk.NORMAL)
        self.chat_hist.insert(tk.END, text + "\n", tag)
        self.chat_hist.see(tk.END)
        self.chat_hist.configure(state=tk.DISABLED)

    def _send_chat(self) -> None:
        if not self.chat_openclaw:
            return
        rec = self._selected()
        if rec is None:
            messagebox.showinfo("No case", "Select a case first.")
            return
        question = self.chat_entry.get().strip()
        if not question:
            return
        self.chat_entry.delete(0, tk.END)
        try:
            context = build_debate_prompt(rec["_index"])
        except FileNotFoundError:
            self._chat_line("No round report for this case yet — run the "
                            "round first.", "sys")
            return
        self._chat_line(f"You: {question}", "user")
        self._chat_line("Assistant is thinking…", "sys")
        self._thinking = True
        self._set_busy(True, "Physician chat: waiting for agent…")
        self._run_thread(self._chat_worker, context, question)

    @staticmethod
    def _chat_worker(context: str, question: str) -> None:
        from skin_agent import run_openclaw_cli  # noqa: E402

        try:
            response = run_openclaw_cli(context + "\nFollow-up request: "
                                        + question, show_command=False)
            if isinstance(response, dict) and response.get("payloads"):
                text = "\n".join(i.get("text", "")
                                 for i in response["payloads"])
            else:
                text = json.dumps(response, ensure_ascii=False)[:4000]
            App._queue().put(("chat_done", {"ok": True, "text": text}))
        except Exception as exc:  # noqa: BLE001 - report to the UI
            App._queue().put(("chat_done", {"ok": False, "text": str(exc)}))

    def _on_chat_done(self, payload: dict) -> None:
        self._set_busy(False, "Ready")
        if getattr(self, "_thinking", False):
            # Remove the "thinking…" line (always last sys line).
            self.chat_hist.configure(state=tk.NORMAL)
            self.chat_hist.delete("end-3l", "end-2l")
            self.chat_hist.configure(state=tk.DISABLED)
            self._thinking = False
        self._chat_line(("Assistant: " if payload["ok"] else "Error: ")
                        + payload["text"],
                        "assist" if payload["ok"] else "sys")


def _ask_string(parent: tk.Tk, title: str, prompt: str,
                default: str) -> str | None:
    dialog = tk.Toplevel(parent)
    dialog.title(title)
    dialog.transient(parent)
    dialog.grab_set()
    ttk.Label(dialog, text=prompt).pack(padx=12, pady=(12, 4))
    var = tk.StringVar(value=default)
    ttk.Entry(dialog, textvariable=var, width=50).pack(padx=12)
    result: list[str | None] = [None]

    def ok() -> None:
        result[0] = var.get()
        dialog.destroy()

    ttk.Button(dialog, text="Send toast", command=ok).pack(pady=12)
    dialog.bind("<Return>", lambda _e: ok())
    parent.wait_window(dialog)
    return result[0]


def _open_file(path: Path) -> None:
    try:
        subprocess.run(["xdg-open", str(path)], check=False,
                       capture_output=True, timeout=10)
    except Exception:  # noqa: BLE001 - opening is a convenience only
        pass


_APP_QUEUE: list[queue.Queue] = []
_APP_TMPDIR: list[Path] = []


def _safe_run(index: int, run_round):
    try:
        return {"report": str(run_round(index))}
    except Exception as exc:  # noqa: BLE001 - marshalled to the UI
        return {"error": str(exc)[:500]}


def main() -> None:
    app = App()
    _APP_QUEUE.append(app.results)
    _APP_TMPDIR.append(app.tmpdir)
    app.mainloop()


if __name__ == "__main__":
    main()
