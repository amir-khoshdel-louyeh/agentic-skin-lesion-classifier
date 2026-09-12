"""Tkinter GUI for the Agentic Skin Lesion Classifier (extra, out-of-plan).

Thin stdlib-only layer (+ Pillow for thumbnails, already a dependency):
the CLI stays canonical; every button calls the same functions and CLIs
(main.py / orchestrator.run_round / control.chat / tools/*.py). Long
runs (rounds, VLM-adjacent tools) execute in worker threads with results
marshalled back via queue — the UI never blocks.

Usage: .venv/bin/python gui.py
"""

import json
import queue
import subprocess
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

from control.chat import audit_for_record, build_debate_prompt, latest_report
from control.manifest import load_manifest

PROMPT_FILE = ROOT_DIR / "prompt.yaml"
MANIFEST_FILE = ROOT_DIR / "tools" / "manifest.yaml"
REPORTS_DIR = ROOT_DIR / "reports"
THUMB_SIZE = (300, 300)


def load_records() -> list[dict]:
    records = yaml.safe_load(PROMPT_FILE.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("prompt.yaml must hold a list of records.")
    return records


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
        self.title("Agentic Skin Lesion Classifier")
        self.geometry("1100x750")
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

    # ----- layout -----
    def _build(self) -> None:
        paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)
        left = ttk.Frame(paned, width=280)
        paned.add(left, weight=1)
        ttk.Label(left, text="Cases (prompt.yaml)").pack(
            anchor=tk.W, padx=6, pady=(6, 0))
        self.case_list = tk.Listbox(left, height=22, exportselection=False)
        self.case_list.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        self.case_list.bind("<<ListboxSelect>>", self._on_case)
        self.preview = ttk.Label(left, text="No image")
        self.preview.pack(padx=6, pady=4)
        self.meta = ttk.Label(left, text="", wraplength=260,
                              justify=tk.LEFT)
        self.meta.pack(padx=6, pady=4, anchor=tk.W)

        right = ttk.Frame(paned)
        paned.add(right, weight=4)
        self.tabs = ttk.Notebook(right)
        self.tabs.pack(fill=tk.BOTH, expand=True)
        self._build_round_tab()
        self._build_tools_tab()
        self._build_reports_tab()
        self._build_chat_tab()

        ttk.Label(self, text="Log").pack(anchor=tk.W, padx=6)
        self.log = tk.Text(self, height=7, state=tk.DISABLED)
        self.log.pack(fill=tk.X, padx=6, pady=(0, 6))

    def _build_round_tab(self) -> None:
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="Round")
        bar = ttk.Frame(tab)
        bar.pack(fill=tk.X, padx=6, pady=6)
        self.run_btn = ttk.Button(bar, text="Run round (3 agents)",
                                  command=self._run_round)
        self.run_btn.pack(side=tk.LEFT)
        ttk.Button(bar, text="Export PDF", command=self._export_pdf).pack(
            side=tk.LEFT, padx=6)
        self.round_status = ttk.Label(bar, text="Idle")
        self.round_status.pack(side=tk.LEFT, padx=6)
        self.report_view = tk.Text(tab, wrap=tk.WORD)
        self.report_view.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

    def _build_tools_tab(self) -> None:
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="Tools")
        bar = ttk.Frame(tab)
        bar.pack(fill=tk.X, padx=6, pady=6)
        ready = [t for t in self.manifest.tools if t.status == "ready"]
        self.tool_names = [f"{t.name} [{t.tier}]" for t in ready]
        self.tool_by_label = {label: t for label, t in zip(
            self.tool_names, ready)}
        self.tool_var = tk.StringVar(value=self.tool_names[0])
        ttk.OptionMenu(bar, self.tool_var, self.tool_names[0],
                       *self.tool_names).pack(side=tk.LEFT)
        ttk.Button(bar, text="Run on selected case",
                   command=self._run_tool).pack(side=tk.LEFT, padx=6)
        self.tool_out = tk.Text(tab, wrap=tk.WORD)
        self.tool_out.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

    def _build_reports_tab(self) -> None:
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="Reports")
        bar = ttk.Frame(tab)
        bar.pack(fill=tk.X, padx=6, pady=6)
        self.report_list = tk.Listbox(bar, height=6, exportselection=False)
        self.report_list.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.report_list.bind("<<ListboxSelect>>", self._on_report)
        ttk.Button(bar, text="Refresh",
                   command=self._refresh_reports).pack(side=tk.LEFT, padx=6)
        self.audit_view = tk.Text(tab, wrap=tk.WORD, height=8)
        self.audit_view.pack(fill=tk.X, padx=6, pady=6)
        ttk.Label(tab, text="audit.jsonl (selected record)").pack(
            anchor=tk.W, padx=6)

    def _build_chat_tab(self) -> None:
        tab = ttk.Frame(self.tabs)
        self.tabs.add(tab, text="Physician chat")
        self.chat_hist = tk.Text(tab, wrap=tk.WORD, state=tk.DISABLED)
        self.chat_hist.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        entry_bar = ttk.Frame(tab)
        entry_bar.pack(fill=tk.X, padx=6, pady=6)
        self.chat_entry = ttk.Entry(entry_bar)
        self.chat_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.chat_entry.bind("<Return>", lambda _e: self._send_chat())
        ttk.Button(entry_bar, text="Send",
                   command=self._send_chat).pack(side=tk.LEFT, padx=6)
        try:
            from skin_agent import find_openclaw_executable  # noqa: E402

            find_openclaw_executable()
            self.chat_openclaw = True
        except Exception:  # noqa: BLE001 - chat needs the OpenClaw CLI
            self.chat_openclaw = False
            self._chat_line("System: OpenClaw CLI not found — chat disabled. "
                            "Run rounds from the Round tab; the CLI remains "
                            "available via main.py --chat.")

    # ----- helpers -----
    def _log(self, text: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, text + "\n")
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)

    def _selected(self) -> dict | None:
        sel = self.case_list.curselection()
        return self.records[sel[0]] if sel else None

    def _run_thread(self, target, *args) -> None:
        threading.Thread(target=target, args=args, daemon=True).start()

    def _poll(self) -> None:
        try:
            while True:
                kind, payload = self.results.get_nowait()
                {"log": self._log,
                 "round_done": self._on_round_done,
                 "tool_done": self._on_tool_done,
                 "chat_done": self._on_chat_done}[kind](payload)
        except queue.Empty:
            pass
        self.after(200, self._poll)

    # ----- cases -----
    def _refresh_cases(self) -> None:
        self.case_list.delete(0, tk.END)
        for rec in self.records:
            meta = rec.get("metadata", {})
            self.case_list.insert(
                tk.END,
                f"{rec['_index']}: {Path(rec['image_path']).stem} "
                f"({meta.get('age')}/{meta.get('sex')})")
        self.case_list.selection_set(0)

    def _on_case(self, _event=None) -> None:
        rec = self._selected()
        if rec is None:
            return
        self.meta.configure(text=json.dumps(rec.get("metadata", {}),
                                            ensure_ascii=False))
        try:
            # No ImageTk on this runtime (system Pillow lacks Tk support):
            # round-trip through PPM, which tk.PhotoImage reads natively.
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
            return
        self.run_btn.configure(state=tk.DISABLED)
        self.round_status.configure(text=f"Running round {rec['_index']}…")
        self.results.put(("log", f"Round {rec['_index']} started."))
        self._run_thread(self._round_worker, rec["_index"])

    @staticmethod
    def _round_worker(index: int) -> None:
        # Imported here: worker-thread only, keeps startup light.
        from orchestrator import run_round  # noqa: E402

        App._queue().put(("round_done", _safe_run(index, run_round)))

    @staticmethod
    def _queue() -> queue.Queue:
        return _APP_QUEUE[0]

    def _on_round_done(self, payload: dict) -> None:
        self.run_btn.configure(state=tk.NORMAL)
        if payload.get("error"):
            self.round_status.configure(text="Round failed")
            self.results.put(("log", f"Round failed: {payload['error']}"))
            return
        path = payload["report"]
        self.round_status.configure(text=f"Done: {Path(path).name}")
        self.results.put(("log", f"Round report: {path}"))
        self.report_view.delete("1.0", tk.END)
        self.report_view.insert("1.0", Path(path).read_text(
            encoding="utf-8"))
        self._refresh_reports()

    def _export_pdf(self) -> None:
        rec = self._selected()
        if rec is None:
            return
        md = latest_report(rec["_index"])
        if md is None:
            messagebox.showinfo("No report",
                                "Run the round first, then export.")
            return
        pdf = md.with_suffix(".pdf")
        proc = subprocess.run(
            [sys.executable, str(ROOT_DIR / "tools" / "report_pdf.py"),
             "--input", str(md), "--output", str(pdf)],
            capture_output=True, text=True, timeout=120)
        if proc.returncode == 0:
            self._log(f"PDF exported: {pdf.name}")
        else:
            messagebox.showerror("PDF export failed", proc.stdout[-500:])

    # ----- tools -----
    def _run_tool(self) -> None:
        rec = self._selected()
        if rec is None:
            return
        tool = self.tool_by_label[self.tool_var.get()]
        if tool.category == "reporting" and latest_report(
                rec["_index"]) is None:
            messagebox.showinfo("No report",
                                "Run the round first for a PDF source.")
            return
        self._log(f"$ python {tool.command} …")
        self._run_thread(self._tool_worker, tool.name, rec["_index"])

    @staticmethod
    def _tool_worker(tool_name: str, index: int) -> None:
        from control.manifest import load_manifest as _lm  # noqa: E402

        manifest = _lm(MANIFEST_FILE)
        tool = manifest.by_name(tool_name)
        rec = load_records()[index]
        rec["_index"] = index
        extra: dict = {}
        if tool.category == "preprocessing":
            extra["output_dir"] = str(App._tmpdir())
        try:
            proc = subprocess.run(
                tool_command(tool, rec, extra),
                capture_output=True, text=True, timeout=900,
                cwd=str(ROOT_DIR))
            App._queue().put(("tool_done",
                              {"ok": proc.returncode == 0,
                               "out": (proc.stdout or proc.stderr)[-6000:],
                               "tool": tool_name}))
        except Exception as exc:  # noqa: BLE001 - report to the UI
            App._queue().put(("tool_done",
                              {"ok": False, "out": str(exc),
                               "tool": tool_name}))

    @staticmethod
    def _tmpdir() -> Path:
        return _APP_TMPDIR[0]

    def _on_tool_done(self, payload: dict) -> None:
        self.tool_out.delete("1.0", tk.END)
        self.tool_out.insert("1.0",
                             f"# {payload['tool']} "
                             f"({'ok' if payload['ok'] else 'FAILED'})\n\n"
                             + payload["out"])
        self._log(f"Tool {payload['tool']}: "
                  f"{'ok' if payload['ok'] else 'FAILED'}")

    # ----- reports -----
    def _refresh_reports(self) -> None:
        self.report_list.delete(0, tk.END)
        self._report_files = sorted(REPORTS_DIR.glob("*.md"))
        for path in self._report_files:
            self.report_list.insert(tk.END, path.name)

    def _on_report(self, _event=None) -> None:
        sel = self.report_list.curselection()
        if not sel:
            return
        name = self.report_list.get(sel[0])
        self.report_view.delete("1.0", tk.END)
        self.report_view.insert(
            "1.0", (REPORTS_DIR / name).read_text(encoding="utf-8"))
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
    def _chat_line(self, text: str) -> None:
        self.chat_hist.configure(state=tk.NORMAL)
        self.chat_hist.insert(tk.END, text + "\n\n")
        self.chat_hist.see(tk.END)
        self.chat_hist.configure(state=tk.DISABLED)

    def _send_chat(self) -> None:
        if not self.chat_openclaw:
            return
        rec = self._selected()
        if rec is None:
            return
        question = self.chat_entry.get().strip()
        if not question:
            return
        self.chat_entry.delete(0, tk.END)
        try:
            context = build_debate_prompt(rec["_index"])
        except FileNotFoundError:
            self._chat_line("System: no round report for this case yet — "
                            "run the round first.")
            return
        self._chat_line(f"You: {question}")
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
        self._chat_line(("Assistant: " if payload["ok"] else "Error: ")
                        + payload["text"])


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
