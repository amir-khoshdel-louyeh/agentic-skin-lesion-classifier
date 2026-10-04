"""Tkinter desktop app for the Agentic Skin Lesion Classifier.

Cases-only browser: pick a case, see its image, metadata, and whether a
round report exists. Screening rounds run via the CLI (`main.py`);
reports land in `report/`.

Usage: .venv/bin/python gui.py
"""

import queue
import sys
import tempfile
import threading
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
GREEN = "#057a55"
GREEN_BG = "#def7ec"
GRAY_AVATAR = "#c3c9d4"
GREEN_AVATAR = "#12a56b"
AMBER_BG = "#fdf6b2"
TITLE_FONT = ("TkDefaultFont", 13, "bold")
HEAD_FONT = ("TkDefaultFont", 16, "bold")

# Chain stages in order: avatar key → label.
STAGES = (
    ("quality", "Quality"),
    ("brain", "Brain"),
    ("s1", "Session 1"),
    ("s2", "Session 2"),
    ("s3", "Session 3"),
    ("critic", "Critic"),
    ("spec", "Specialists"),
    ("referee", "Referee"),
)


def load_records() -> list[dict]:
    records = yaml.safe_load(PROMPT_FILE.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("prompt.yaml must hold a list of records.")
    return records


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Skin Lesion Classifier — Cases")
        self.minsize(900, 600)
        try:
            self.state("zoomed")  # open maximized
        except tk.TclError:
            self.geometry("1180x800")
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
        self.results: queue.Queue = queue.Queue()
        self._run_stem: str = ""
        self._stage_state: dict[str, dict] = {}
        self._reset_stages()
        self._build()
        self._refresh_cases()
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

    def _build(self) -> None:
        header = ttk.Frame(self)
        header.pack(fill=tk.X, padx=12, pady=(10, 4))
        ttk.Label(header, text="Cases",
                  font=HEAD_FONT).pack(side=tk.LEFT)
        ttk.Label(header, text=f"  {len(self.records)} cases",
                  style="Muted.TLabel").pack(side=tk.LEFT, pady=6)
        ttk.Label(header, text="Research use only",
                  style="Muted.TLabel").pack(side=tk.RIGHT)

        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=12)
        paned.add(self._case_card(), weight=1)
        right = ttk.Frame(self)
        paned.add(right, weight=3)
        self.tabs = ttk.Notebook(right)
        self.tabs.pack(fill=tk.BOTH, expand=True)
        self._build_main_tab()

        self.status = ttk.Label(self, text="Select a case",
                                style="Muted.TLabel", anchor=tk.W)
        self.status.pack(fill=tk.X, padx=12, pady=(4, 8))

    def _build_main_tab(self) -> None:
        tab = ttk.Frame(self)
        self.tabs.add(tab, text="  Main  ")
        bar = ttk.Frame(tab)
        bar.pack(fill=tk.X, padx=10, pady=8)
        self.main_case_var = tk.StringVar(value="No case selected")
        ttk.Label(bar, textvariable=self.main_case_var,
                  font=TITLE_FONT).pack(side=tk.LEFT)
        self.start_btn = ttk.Button(bar, text="▶  Start",
                                    command=self._on_start)
        self.start_btn.pack(side=tk.LEFT, padx=12)
        self.main_status_var = tk.StringVar(value="Idle.")
        ttk.Label(bar, textvariable=self.main_status_var,
                  style="Muted.TLabel").pack(side=tk.LEFT)
        self.chain = tk.Canvas(tab, bg=CARD, highlightthickness=0,
                               height=190)
        self.chain.pack(fill=tk.X, padx=10, pady=(0, 6))
        self.chain.bind("<Configure>", lambda _e: self._draw_chain())
        self.main_body = ttk.Frame(tab)
        self.main_body.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)
        self._build_chat_box()
        self._draw_chain()

    # ----- main chat box -----
    def _build_chat_box(self) -> None:
        box_frame = ttk.Frame(self.main_body)
        box_frame.pack(fill=tk.BOTH, expand=True)
        self.chat_box = tk.Text(box_frame, wrap=tk.WORD, bg=CARD,
                                relief=tk.FLAT, state=tk.DISABLED,
                                padx=8, pady=8, height=12,
                                font=("TkDefaultFont", 9))
        scroll = ttk.Scrollbar(box_frame, command=self.chat_box.yview)
        self.chat_box.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.chat_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        for tag, fg in (("stage", "#1a56db"), ("user", "#111928"),
                        ("assist", "#057a55"), ("sys", "#6b7280"),
                        ("error", "#9b1c1c")):
            self.chat_box.tag_configure(tag, foreground=fg)
        entry_bar = ttk.Frame(self.main_body)
        entry_bar.pack(fill=tk.X, pady=(6, 0))
        self.chat_entry = ttk.Entry(entry_bar)
        self.chat_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.chat_entry.bind("<Return>", lambda _e: self._on_ask())
        ttk.Button(entry_bar, text="Send",
                   command=self._on_ask).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(entry_bar, text="■ Stop",
                   command=self._on_stop).pack(side=tk.LEFT, padx=(8, 0))

    def _chat_line(self, text: str, tag: str = "sys") -> None:
        self.chat_box.configure(state=tk.NORMAL)
        self.chat_box.insert(tk.END, text + "\n", tag)
        self.chat_box.see(tk.END)
        self.chat_box.configure(state=tk.DISABLED)

    def _log_stage(self, stage: str, payload: dict) -> None:
        # Image stem snapshotted at Start: the user may click other cases
        # mid-round, and the log must name the RUNNING case, not the
        # currently selected one.
        img = self._run_stem or "?"
        if stage == "quality":
            reasons = "; ".join(payload.get("reasons", [])[:2])
            self._chat_line(
                f"[Quality] in: image {img} → out: "
                f"{payload.get('verdict', '?')} ({reasons})", "stage")
        elif stage == "brain":
            self._chat_line(
                f"[Brain steer {payload.get('session', '?')}] in: Quality + "
                f"receipts → out: {payload.get('steer', '')[:160]} "
                f"(why: {payload.get('why', '')[:160]})", "stage")
        elif stage == "session":
            self._chat_line(
                f"[Session {payload.get('n', '?')}] in: steer → out: "
                f"tool={payload.get('tools_run', '?')}, "
                f"class={payload.get('class', '?')}, "
                f"conf={payload.get('confidence', '?')}", "stage")
        elif stage == "critic":
            if payload.get("sustained"):
                reasons = "; ".join(payload.get("reasons", [])[:2])
                self._chat_line(
                    f"[Critic] in: receipts + flags → OBJECTION: {reasons}",
                    "stage")
            else:
                self._chat_line(
                    "[Critic] in: receipts + flags → silence.", "stage")
        elif stage == "specialists":
            if payload.get("triggered"):
                self._chat_line(
                    "[Specialists] trigger: "
                    f"{' vs '.join(payload.get('shortlist', []))}", "stage")
            else:
                self._chat_line(
                    "[Specialists] skipped (no dilemma).", "stage")
        elif stage == "specialist":
            self._chat_line(
                f"[Advocate {payload.get('disease', '?')}] out: "
                f"class={payload.get('class', '?')}", "stage")
        elif stage == "referee":
            self._chat_line(
                f"[Referee] decision: **{payload.get('decision', '?')}** — "
                f"{payload.get('reason', '')} "
                f"(rule={payload.get('rule', '')})", "stage")

    def _on_ask(self) -> None:
        question = self.chat_entry.get().strip()
        if not question:
            return
        rec = self._selected()
        if rec is None:
            self._chat_line("Select a case first.", "error")
            return
        self.chat_entry.delete(0, tk.END)
        self._chat_line(f"You: {question}", "user")
        self._run_thread(self._ask_worker, rec["_index"], question)

    @staticmethod
    def _ask_worker(index: int, question: str) -> None:
        from control.controller import ask, load_case  # noqa: E402

        try:
            state = load_case(index)
        except FileNotFoundError as exc:
            App._queue().put(("chat_done", {"ok": False,
                                            "text": str(exc)}))
            return
        try:
            App._queue().put(("chat_done", {"ok": True,
                                            "text": ask(state, question)}))
        except Exception as exc:  # noqa: BLE001 - report to the UI
            App._queue().put(("chat_done", {"ok": False, "text": str(exc)}))

    def _on_chat_done(self, payload: dict) -> None:
        prefix = "Assistant: " if payload["ok"] else "Error: "
        self._chat_line(prefix + payload["text"],
                        "assist" if payload["ok"] else "error")

    def _on_stop(self) -> None:
        """Shut the system down."""
        self.destroy()

    # ----- chain avatars -----
    def _reset_stages(self) -> None:
        self._stage_state = {key: {"active": False, "sub": ""}
                             for key, _ in STAGES}

    def _draw_chain(self) -> None:
        canvas = self.chain
        canvas.delete("all")
        width = max(canvas.winfo_width(), 200)
        height = max(canvas.winfo_height(), 120)
        n = len(STAGES)
        radius = 30
        margin = 60
        step = (width - 2 * margin) / (n - 1) if n > 1 else 0
        y = height / 2 - 10
        points = []
        for i, (key, _label) in enumerate(STAGES):
            points.append((margin + i * step, y))
        for (x1, y1), (x2, y2) in zip(points, points[1:]):
            canvas.create_line(x1, y1, x2, y2, fill=GRAY_AVATAR, width=3)
        for (key, label), (x, y) in zip(STAGES, points):
            state = self._stage_state.get(key, {})
            fill = GREEN_AVATAR if state.get("active") else GRAY_AVATAR
            canvas.create_oval(x - radius, y - radius, x + radius,
                               y + radius, fill=fill, outline="")
            canvas.create_text(x, y, text=label[0], fill="white",
                               font=("TkDefaultFont", 14, "bold"))
            canvas.create_text(x, y + radius + 12, text=label, fill=TEXT,
                               font=("TkDefaultFont", 9, "bold"))
            if state.get("sub"):
                canvas.create_text(x, y + radius + 26, text=state["sub"],
                                   fill=MUTED, font=("TkDefaultFont", 8))

    def _activate(self, key: str, sub: str = "") -> None:
        if key in self._stage_state:
            self._stage_state[key] = {"active": True, "sub": sub[:40]}
            self._draw_chain()

    def _apply_stage(self, stage: str, payload: dict) -> None:
        self._log_stage(stage, payload)
        if stage == "quality":
            self._activate("quality",
                           str(payload.get("verdict", "")))
            self.main_status_var.set("Quality done — Brain steering…")
        elif stage == "brain":
            self._activate("brain",
                           f"steer {payload.get('session', '?')}")
            self.main_status_var.set(
                f"Brain steer {payload.get('session', '?')} — "
                "session running…")
        elif stage == "session":
            n = payload.get("n", 0)
            self._activate(f"s{n}", str(payload.get("class", "") or ""))
            self.main_status_var.set(
                f"Session {n} done ({payload.get('class', '?')}) — "
                "Brain steering…")
        elif stage == "critic":
            self._activate("critic", "objected" if payload.get(
                "sustained") else "silent")
            self.main_status_var.set("Critic done — specialists…")
        elif stage == "specialists":
            if not payload.get("triggered"):
                self._stage_state["spec"] = {"active": False,
                                             "sub": "skipped"}
                self._draw_chain()
                self.main_status_var.set("No dilemma — referee counting…")
            else:
                self.main_status_var.set(
                    f"Dilemma: {' vs '.join(payload.get('shortlist', []))}")
        elif stage == "specialist":
            self._activate("spec", str(payload.get("disease", "")))
            self.main_status_var.set(
                f"Advocate for {payload.get('disease', '?')} done…")
        elif stage == "referee":
            self._activate("referee", str(payload.get("decision", "")))
            self.main_status_var.set(
                f"Decision: {payload.get('decision', '?')} — "
                f"{payload.get('rule', '')}")

    # ----- round run -----
    def _run_thread(self, target, *args) -> None:
        threading.Thread(target=target, args=args, daemon=True).start()

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.results.get_nowait()
                if kind == "stage":
                    stage, data = payload
                    self._apply_stage(stage, data)
                elif kind == "round_done":
                    self._on_round_done(payload)
                elif kind == "chat_done":
                    self._on_chat_done(payload)
        except queue.Empty:
            pass
        self.after(200, self._poll)

    def _on_start(self) -> None:
        rec = self._selected()
        if rec is None:
            messagebox.showinfo("No case", "Select a case first.")
            return
        self.main_case_var.set(
            f"Case {rec['_index']}: {Path(rec['image_path']).stem}")
        self._run_stem = Path(rec["image_path"]).stem
        self._reset_stages()
        self._draw_chain()
        self.start_btn.configure(state=tk.DISABLED)
        self.main_status_var.set("Round started — Quality running…")
        self.status.configure(text=f"Running round {rec['_index']}…")
        self._chat_line(
            f"── Round started on Case {rec['_index']}: "
            f"{Path(rec['image_path']).stem} ──", "sys")
        self._run_thread(self._round_worker, rec["_index"])

    @staticmethod
    def _round_worker(index: int) -> None:
        from orchestrator import run_round  # noqa: E402

        App._queue().put(("round_done", _safe_run(index)))

    @staticmethod
    def _queue() -> queue.Queue:
        return _APP_QUEUE[0]

    def _on_round_done(self, payload: dict) -> None:
        self.start_btn.configure(state=tk.NORMAL)
        if payload.get("error"):
            self.main_status_var.set("Round FAILED — see terminal log.")
            self.status.configure(text="Round failed")
            self._chat_line("Round FAILED — see terminal log.", "error")
            return
        self.main_status_var.set(f"Done: {Path(payload['report']).name}")
        self.status.configure(text=f"Done: {Path(payload['report']).name}")
        self._chat_line("── Round complete. Ask a question below, "
                        "or press Stop to shut down. ──", "sys")
        self._refresh_cases()

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
        if getattr(self, "main_case_var", None) is not None:
            self.main_case_var.set(
                f"Case {rec['_index']}: {Path(rec['image_path']).stem}")
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


_APP_QUEUE: list[queue.Queue] = []


def _safe_run(index: int):
    try:
        from orchestrator import run_round  # noqa: E402

        def _emit(stage: str, payload: dict) -> None:
            App._queue().put(("stage", (stage, payload)))

        return {"report": str(run_round(index, on_stage=_emit))}
    except Exception as exc:  # noqa: BLE001 - marshalled to the UI
        return {"error": str(exc)[:500]}


def main() -> None:
    app = App()
    _APP_QUEUE.append(app.results)
    app.mainloop()


if __name__ == "__main__":
    main()
