# node_editor/nodes/point_tracking_node.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from node_editor.base_node import BaseNode
from node_editor.nodes.point_predictor_node import (
    _LastKnownPredictor,
    _LinearPredictor,
    _PolyPredictor,
    _EMAPredictor,
    _KalmanPredictor,
)
from node_editor.pin_types import PinSchema, PinDef, PinType
from node_editor.execution import ExecutionMode
from node_editor.project_context import get_project_directory


# ---------------------------------------------------------------------------
# Help text
# ---------------------------------------------------------------------------

_HELP_TEXT = """\
Point Tracking Node
===================

PURPOSE
-------
A self-contained batch tracking application node.
No input pins — everything is configured in the inspector.

The node reads an image sequence from disk, tracks a set of
2D points across all frames, and pushes one result row per
frame to the output pins.  Each output row is a flat 1D array
suitable for direct connection to an ArrayAccumulatorNode.

OUTPUT PINS
-----------
row_data    ARRAY  (1D, float64)
    One row per tracked frame, containing:
      [tracked_pts (N*2), status (N), error (N),
       prev_index_out (1), frame_index_out (1)]
    Total length = N*2 + N + N + 1 + 1 = 4*N + 2.
    Lost points (status=0) have NaN in their tracked_pts.

row_done    TRIGGER
    Fires after each frame row is pushed.  Connect to
    BatchRunnerNode.advance if you need to synchronise
    downstream processing with the tracking loop.

all_done    TRIGGER
    Fires once when the full sequence has been processed
    or when the user clicks Stop.

INSPECTOR PANELS
----------------
Image sequence
    Enter one absolute file path per line in the text box,
    or use Browse to select files via a file dialog.
    Relative paths are resolved against the project directory.

Initial tracking points
    Enter N points, one per line, as "x y" or "x,y".
    These are the pixel coordinates in image 1 (1-based).
    For matchTemplate or ECC (future): add columns w h for
    per-point template size: "x y w h".

Tracking method
    Optical flow (cv2.calcOpticalFlowPyrLK)  [default]
      winSize      half-window size (px), default 15
      maxLevel     pyramid levels, default 3
      maxIter      max iterations, default 30
      epsilon      convergence threshold, default 0.01
      flags        cv2 LK flags integer, default 0
      minEigThresh minEigThreshold, default 1e-4

    Template match (cv2.matchTemplate)  [future]
    ECC (cv2.findTransformECC)          [future]

Update template period
    Controls which frame is used as the reference (prevImg /
    template) for each tracked frame.

    period = 0  →  always use frame 1 (cumulative / fixed ref)
    period = 1  →  use the immediately preceding frame
    period = k  →  use the closest frame of index
                   1, 1+k, 1+2k, 1+3k, ...

    Formula:
      if period == 0:
          prev_index = 1
      else:
          prev_index = 1 + ((index - 2) // period) * period

    The node caches the reference image and its tracked points
    automatically.  When the reference frame changes, the cache
    is updated from the accumulated tracking history.

SubPixel refinement (cv2.cornerSubPix)
    Applied to initPts before tracking starts.
    Window size and termination criteria are configurable.

Lost point handling
    Points whose LK status = 0 are marked as lost.
    Their tracked_pts values are set to NaN.
    They remain NaN for all subsequent frames.

Progress and control
    Progress bar shows frames completed / total.
    Log panel shows per-frame status messages.
    [Start]  begin tracking from frame 1.
    [Pause]  suspend after the current frame finishes.
    [Resume] continue from the paused frame.
    [Stop]   abort; partial results already pushed remain.

KEYBOARD SHORTCUT
-----------------
Ctrl-H / Ctrl-h  —  show this help window.
"""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _parse_points(text: str) -> np.ndarray | None:
    """Parse an N×2 float array from text (csv or space separated)."""
    rows = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) != 2:
            return None
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            return None
        rows.append(vals)
    if not rows:
        return None
    return np.array(rows, dtype=np.float64)


def _calc_prev_index(index: int,
                      period: int) -> int:
    """
    Return 1-based reference frame index for a given
    current frame index and update_template_period.

    period=0: always return 1 (fixed reference).
    period>0: 1 + floor((index-2) / period) * period
    """
    if period == 0 or index <= 1:
        return 1
    return 1 + ((index - 2) // period) * period


def _load_gray(path: str) -> np.ndarray | None:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


def _load_rgb(path: str) -> np.ndarray | None:
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# PointTrackingNode
# ---------------------------------------------------------------------------

class PointTrackingNode(BaseNode):
    """
    Self-contained batch point-tracking application node.

    No input pins.  The inspector provides a complete UI for:
      - Image sequence file list
      - Initial tracking point coordinates
      - Tracking method and parameters
      - Update-template period
      - SubPixel refinement
      - Progress monitoring and Start/Pause/Stop controls

    Tracking runs in a background thread (non-blocking).
    One output row is pushed per tracked frame via push_output().
    """

    EXECUTION_MODE = ExecutionMode.STREAMING
    NODE_TYPE      = "point_tracking"
    DISPLAY_NAME   = "Point Tracking"
    CATEGORY       = "process"
    NODE_WIDTH     = 220
    NODE_HEIGHT    = 130

    HELP_TEXT = _HELP_TEXT

    _BODY_BG   = "#fff8f0"
    _OUTLINE   = "#cc7700"
    _TITLE_FG  = "#663300"
    _STATUS_FG = "#885500"

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[],
            outputs=[
                PinDef("row_data", PinType.ARRAY,
                       "row"),
                PinDef("row_done", PinType.TRIGGER,
                       "rowDone"),
                PinDef("all_done", PinType.TRIGGER,
                       "allDone"),
            ]
        )

    def get_help_text(self) -> str:
        return _HELP_TEXT

    # ── state ─────────────────────────────────────────────────────

    def _init_state(self) -> None:
        if hasattr(self, "_files_text_var"):
            return

        # image sequence
        self._files_text: str = ""
        self._files_text_var = tk.StringVar(
            value="")
        self._file_count_var = tk.StringVar(
            value="0 files")

        # initial points
        self._initpts_text: str = ""
        self._point_count_var = tk.StringVar(
            value="0 points")

        # predictor settings
        self._pred_algo_var = tk.StringVar(
            value="Linear extrapolation (order 1)")
        self._pred_poly_deg_var = tk.IntVar(value=2)
        self._pred_ema_alpha_var = tk.DoubleVar(value=0.5)
        self._pred_kf_qpos_var = tk.DoubleVar(value=1e-2)
        self._pred_kf_qvel_var = tk.DoubleVar(value=1e-3)
        self._pred_kf_r_var = tk.DoubleVar(value=1e-1)
        self._pred_max_hist_var = tk.IntVar(value=10)

        # tracking method
        self._method_var = tk.StringVar(
            value="Optical Flow (LK)")

        # LK parameters
        self._lk_winsize_var  = tk.IntVar(value=15)
        self._lk_maxlevel_var = tk.IntVar(value=3)
        self._lk_maxiter_var  = tk.IntVar(value=30)
        self._lk_eps_var      = tk.DoubleVar(
            value=0.01)
        self._lk_flags_var    = tk.IntVar(value=0)
        self._lk_mineig_var   = tk.DoubleVar(
            value=1e-4)

        # template update period
        self._period_var = tk.IntVar(value=0)

        # subpixel refinement
        self._subpix_var     = tk.BooleanVar(
            value=False)
        self._subpix_win_var = tk.IntVar(value=11)
        self._subpix_iter_var = tk.IntVar(value=30)
        self._subpix_eps_var  = tk.DoubleVar(
            value=0.001)

        # status / progress
        self._status_var   = tk.StringVar(
            value="idle")
        self._progress_var = tk.DoubleVar(value=0.0)

        # thread control
        self._thread:       threading.Thread | None = None
        self._stop_event    = threading.Event()
        self._pause_event   = threading.Event()
        self._pause_event.set()   # not paused initially
        # own thread-active flag — deliberately NOT named _is_running:
        # BaseNode._is_running is engine-managed and forced True on every
        # STREAMING node immediately at creation (see node_editor_app.py),
        # which would make _on_start()'s guard below no-op forever.
        self._tracking_active = False
        self._done_counter  = 0
        self._all_done_counter = 0

        # worker -> UI thread-safe handoff.  The worker thread must
        # never touch a Tk widget/variable directly; it only ever
        # puts events on this queue.  A poller running on the main
        # thread (scheduled via canvas.after, started in _on_start)
        # drains it and applies the updates.
        self._ui_queue: queue.Queue = queue.Queue()
        self._ui_poll_active = False

        # inspector widget refs
        self._files_text_widget:  tk.Text | None = None
        self._initpts_widget:     tk.Text | None = None
        self._log_widget:         tk.Text | None = None
        self._progress_bar:       ttk.Progressbar | None = None
        self._start_btn:          tk.Button | None = None
        self._pause_btn:          tk.Button | None = None
        self._stop_btn:           tk.Button | None = None
        self._help_popup:         tk.Toplevel | None = None

    # ── build_body ────────────────────────────────────────────────

    def build_body(self) -> None:
        self._init_state()
        x, y, w, h = self.x, self.y, self.width, self.height

        self._body_rect = self.canvas.create_rectangle(
            x, y, x+w, y+h,
            fill=self._BODY_BG,
            outline=self._OUTLINE, width=2,
            tags=(self.node_id, "node_body"))
        self._title_item = self.canvas.create_text(
            x+w/2, y+13,
            text=self.DISPLAY_NAME,
            font=("Arial", 9, "bold"),
            fill=self._TITLE_FG,
            tags=(self.node_id,))

        # progress bar on node body
        self._body_progress = ttk.Progressbar(
            self.canvas,
            variable=self._progress_var,
            maximum=100.0,
            mode="determinate",
            length=w-20)
        self.canvas.create_window(
            x+w/2, y+h//2+4,
            window=self._body_progress,
            tags=(self.node_id,))

        status_lbl = tk.Label(
            self.canvas,
            textvariable=self._status_var,
            font=("Arial", 7),
            bg=self._BODY_BG,
            fg=self._STATUS_FG,
            wraplength=w-12, justify="center")
        self.canvas.create_window(
            x+w/2, y+h-12,
            window=status_lbl,
            tags=(self.node_id,))

        self._canvas_items += [
            self._body_rect, self._title_item]

    # ── build_inspector ───────────────────────────────────────────

    def build_inspector(self, parent: tk.Frame) -> None:
        self._init_state()

        win = self._inspector_win
        if win is not None:
            for seq in ("<Control-h>",
                        "<Control-H>"):
                win.bind(
                    seq,
                    lambda e: self._open_help())

        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True)

        tab_seq      = tk.Frame(nb)
        tab_method   = tk.Frame(nb)
        tab_predict  = tk.Frame(nb)
        tab_run      = tk.Frame(nb)
        nb.add(tab_seq,     text="Image Sequence")
        nb.add(tab_method,  text="Method & Params")
        nb.add(tab_predict, text="Predictor")
        nb.add(tab_run,     text="Run & Progress")

        self._build_tab_sequence(tab_seq)
        self._build_tab_method(tab_method)
        self._build_tab_predictor(tab_predict)
        self._build_tab_run(tab_run)

        self._refresh_count_labels()

        if win is not None:
            win.update_idletasks()
            win.minsize(820, 580)
            win.geometry("860x620")

    # ── Tab 1: Image Sequence ─────────────────────────────────────

    def _build_tab_sequence(self,
                             parent: tk.Frame) -> None:
        pad = {"padx": 8, "pady": 4}

        # file list
        files_frame = tk.LabelFrame(
            parent,
            text="Image file list"
                 "  (one absolute path per line)",
            font=("Arial", 9), **pad)
        files_frame.pack(fill="both",
                          expand=True, **pad)

        btn_row = tk.Frame(files_frame)
        btn_row.pack(fill="x", pady=(0, 4))
        tk.Button(
            btn_row, text="Browse files...",
            font=("Arial", 9),
            command=self._on_browse_files).pack(
            side="left")
        tk.Button(
            btn_row, text="Clear",
            font=("Arial", 9),
            command=self._on_clear_files).pack(
            side="left", padx=4)
        tk.Label(
            btn_row,
            text="Drag-and-drop also supported"
                 " in text box",
            font=("Arial", 8),
            fg="#888888").pack(
            side="left", padx=8)

        self._files_text_widget = tk.Text(
            files_frame,
            font=("Courier", 8),
            wrap=tk.NONE,
            height=12)
        self._files_text_widget.bind(
            "<KeyRelease>",
            lambda _e: self._refresh_count_labels())
        vsb = ttk.Scrollbar(
            files_frame, orient="vertical",
            command=self._files_text_widget.yview)
        hsb = ttk.Scrollbar(
            files_frame, orient="horizontal",
            command=self._files_text_widget.xview)
        self._files_text_widget.configure(
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set)
        hsb.pack(side="bottom", fill="x")
        vsb.pack(side="right", fill="y")
        self._files_text_widget.pack(
            fill="both", expand=True)

        counts_row = tk.Frame(files_frame)
        counts_row.pack(fill="x", pady=(4, 0))
        tk.Label(
            counts_row,
            textvariable=self._file_count_var,
            font=("Arial", 8), fg="#3355aa").pack(side="left")

        if self._files_text:
            self._files_text_widget.insert(
                "1.0", self._files_text)

        # initial points
        pts_frame = tk.LabelFrame(
            parent,
            text="Initial tracking points"
                 "  (one point per line:  x y  "
                 "or  x,y  —  pixel coords in "
                 "frame 1)",
            font=("Arial", 9), **pad)
        pts_frame.pack(fill="x", **pad)

        self._initpts_widget = tk.Text(
            pts_frame,
            font=("Courier", 9),
            wrap=tk.NONE,
            height=6)
        self._initpts_widget.bind(
            "<KeyRelease>",
            lambda _e: self._refresh_count_labels())
        pts_vsb = ttk.Scrollbar(
            pts_frame, orient="vertical",
            command=self._initpts_widget.yview)
        pts_vsb.pack(side="right", fill="y")
        self._initpts_widget.pack(
            fill="x", expand=True)

        point_count_row = tk.Frame(pts_frame)
        point_count_row.pack(fill="x", pady=(4, 0))
        tk.Label(
            point_count_row,
            textvariable=self._point_count_var,
            font=("Arial", 8), fg="#3355aa").pack(side="left")

        if self._initpts_text:
            self._initpts_widget.insert(
                "1.0", self._initpts_text)

    def _on_browse_files(self) -> None:
        base = get_project_directory()
        initial = str(base) if base else "."
        paths = filedialog.askopenfilenames(
            title="Select image files",
            initialdir=initial,
            filetypes=[
                ("Image files",
                 "*.jpg *.jpeg *.png *.bmp"
                 " *.tif *.tiff"),
                ("All files", "*.*"),
            ])
        if not paths:
            return
        if self._files_text_widget is None:
            return
        existing = self._files_text_widget.get(
            "1.0", tk.END).strip()
        new_text = ("\n".join(paths)
                    if not existing
                    else existing + "\n"
                    + "\n".join(paths))
        self._files_text_widget.delete(
            "1.0", tk.END)
        self._files_text_widget.insert(
            "1.0", new_text)

    def _on_clear_files(self) -> None:
        if self._files_text_widget is not None:
            self._files_text_widget.delete(
                "1.0", tk.END)
        self._refresh_count_labels()

    def _refresh_count_labels(self) -> None:
        file_text = (self._files_text_widget.get("1.0", tk.END)
                     if self._files_text_widget is not None and self._files_text_widget.winfo_exists()
                     else self._files_text)
        point_text = (self._initpts_widget.get("1.0", tk.END)
                      if self._initpts_widget is not None and self._initpts_widget.winfo_exists()
                      else self._initpts_text)

        file_count = sum(1 for line in file_text.splitlines() if line.strip())
        parsed_points = _parse_points(point_text)
        point_count = 0 if parsed_points is None else int(parsed_points.shape[0])

        self._file_count_var.set(f"{file_count} file{'s' if file_count != 1 else ''}")
        self._point_count_var.set(f"{point_count} point{'s' if point_count != 1 else ''}")

        if self._initpts_widget is not None and self._initpts_widget.winfo_exists():
            if parsed_points is None:
                self._initpts_widget.configure(bg="#fff0f0", fg="#aa0000")
            else:
                self._initpts_widget.configure(bg="#ffffff", fg="#222222")

    # ── Tab 2: Method & Params ────────────────────────────────────

    def _build_tab_method(self,
                           parent: tk.Frame) -> None:
        pad = {"padx": 8, "pady": 4}

        # method selection
        meth_frame = tk.LabelFrame(
            parent, text="Tracking method",
            font=("Arial", 9), **pad)
        meth_frame.pack(fill="x", **pad)

        for name in [
            "Optical Flow (LK)",
            "Template Match (future)",
            "ECC (future)",
        ]:
            state = ("normal"
                     if "future" not in name
                     else "disabled")
            tk.Radiobutton(
                meth_frame, text=name,
                variable=self._method_var,
                value=name,
                font=("Arial", 9),
                state=state,
                command=self._on_method_change,
                anchor="w").pack(
                fill="x", pady=1)

        # LK parameters
        self._lk_frame = tk.LabelFrame(
            parent,
            text="Optical Flow LK parameters",
            font=("Arial", 9), **pad)
        self._lk_frame.pack(fill="x", **pad)

        def _lk_row(lbl, var, from_=0.0,
                     to=1000.0, inc=1.0,
                     fmt="%.4g"):
            r = tk.Frame(self._lk_frame)
            r.pack(fill="x", pady=2)
            tk.Label(
                r, text=lbl,
                font=("Arial", 9),
                width=24, anchor="w").pack(
                side="left")
            tk.Spinbox(
                r, from_=from_, to=to,
                increment=inc,
                textvariable=var,
                width=10, font=("Arial", 9),
                format=fmt).pack(side="left")

        _lk_row("winSize (half px):",
                self._lk_winsize_var,
                from_=1, to=100, inc=2,
                fmt="%.0f")
        _lk_row("maxLevel (pyramid):",
                self._lk_maxlevel_var,
                from_=0, to=8, inc=1,
                fmt="%.0f")
        _lk_row("maxIter:",
                self._lk_maxiter_var,
                from_=1, to=200, inc=1,
                fmt="%.0f")
        _lk_row("epsilon:",
                self._lk_eps_var,
                from_=1e-6, to=1.0,
                inc=0.001, fmt="%.4f")
        _lk_row("flags (int):",
                self._lk_flags_var,
                from_=0, to=65535, inc=1,
                fmt="%.0f")
        _lk_row("minEigThreshold:",
                self._lk_mineig_var,
                from_=1e-8, to=0.1,
                inc=1e-4, fmt="%.8f")

        # update template period
        period_frame = tk.LabelFrame(
            parent,
            text="Update template period",
            font=("Arial", 9), **pad)
        period_frame.pack(fill="x", **pad)

        pr = tk.Frame(period_frame)
        pr.pack(fill="x", pady=2)
        tk.Label(
            pr, text="Period (0=fixed ref):",
            font=("Arial", 9),
            width=24, anchor="w").pack(
            side="left")
        tk.Spinbox(
            pr, from_=0, to=9999, increment=1,
            textvariable=self._period_var,
            width=8, font=("Arial", 9),
            format="%.0f").pack(side="left")

        tk.Label(
            period_frame,
            text="0 = always use frame 1\n"
                 "1 = use previous frame\n"
                 "k = use frame 1,1+k,1+2k,...",
            font=("Arial", 8), fg="#666666",
            justify="left").pack(anchor="w")

        # subpixel refinement
        subpix_frame = tk.LabelFrame(
            parent,
            text="SubPixel refinement"
                 " (cv2.cornerSubPix, on initPts)",
            font=("Arial", 9), **pad)
        subpix_frame.pack(fill="x", **pad)

        tk.Checkbutton(
            subpix_frame,
            text="Enable cornerSubPix",
            variable=self._subpix_var,
            font=("Arial", 9)).pack(anchor="w")

        def _sp_row(lbl, var, from_=0.0,
                     to=999.0, inc=1.0,
                     fmt="%.4g"):
            r = tk.Frame(subpix_frame)
            r.pack(fill="x", pady=1)
            tk.Label(
                r, text=lbl,
                font=("Arial", 9),
                width=24, anchor="w").pack(
                side="left")
            tk.Spinbox(
                r, from_=from_, to=to,
                increment=inc,
                textvariable=var,
                width=10, font=("Arial", 9),
                format=fmt).pack(side="left")

        _sp_row("Window half size:",
                self._subpix_win_var,
                from_=1, to=50, inc=1,
                fmt="%.0f")
        _sp_row("Max iterations:",
                self._subpix_iter_var,
                from_=1, to=200, inc=1,
                fmt="%.0f")
        _sp_row("Epsilon:",
                self._subpix_eps_var,
                from_=1e-6, to=1.0,
                inc=0.0001, fmt="%.4f")

        tk.Button(
            parent,
            text="Help  (Ctrl-H)",
            font=("Arial", 8),
            command=self._open_help).pack(
            anchor="e", **pad)

    def _build_tab_predictor(self,
                             parent: tk.Frame) -> None:
        pad = {"padx": 8, "pady": 4}

        algo_frame = tk.LabelFrame(
            parent, text="Prediction algorithm",
            font=("Arial", 9), **pad)
        algo_frame.pack(fill="x", **pad)

        names = [
            "Last known (order 0)",
            "Linear extrapolation (order 1)",
            "Polynomial extrapolation",
            "Exponential smoothing (EMA)",
            "Kalman filter (constant velocity)",
        ]
        for name in names:
            tk.Radiobutton(
                algo_frame, text=name,
                variable=self._pred_algo_var,
                value=name,
                font=("Arial", 9),
                anchor="w").pack(fill="x", pady=1)

        param_frame = tk.LabelFrame(
            parent, text="Algorithm parameters",
            font=("Arial", 9), **pad)
        param_frame.pack(fill="x", **pad)

        def _pred_row(lbl, var, from_=0.0, to=10.0, inc=1.0, fmt="%.4g"):
            r = tk.Frame(param_frame)
            r.pack(fill="x", pady=2)
            tk.Label(r, text=lbl, font=("Arial", 9), width=24, anchor="w").pack(side="left")
            tk.Spinbox(r, from_=from_, to=to, increment=inc, textvariable=var,
                       width=10, font=("Arial", 9), format=fmt).pack(side="left")

        _pred_row("Poly degree (1–5):",
                  self._pred_poly_deg_var,
                  from_=1, to=5, inc=1, fmt="%.0f")
        _pred_row("EMA alpha (0–1):",
                  self._pred_ema_alpha_var,
                  from_=0.0, to=1.0, inc=0.05, fmt="%.2f")
        _pred_row("Kalman Q_pos:",
                  self._pred_kf_qpos_var,
                  from_=1e-6, to=1.0, inc=1e-3, fmt="%.6f")
        _pred_row("Kalman Q_vel:",
                  self._pred_kf_qvel_var,
                  from_=1e-6, to=1.0, inc=1e-4, fmt="%.6f")
        _pred_row("Kalman R:",
                  self._pred_kf_r_var,
                  from_=1e-6, to=10.0, inc=0.01, fmt="%.6f")
        _pred_row("Max history:",
                  self._pred_max_hist_var,
                  from_=1, to=50, inc=1, fmt="%.0f")

        tk.Label(
            parent,
            text="This predictor provides the initial guess used by cv2.calcOpticalFlowPyrLK(nextPts).",
            font=("Arial", 8), fg="#666666",
            justify="left", wraplength=560).pack(anchor="w", padx=8, pady=(4, 0))

    def _make_predictor(self, cfg: dict):
        algo = cfg["pred_algo"]
        max_hist = cfg["pred_max_hist"]
        if algo == "Last known (order 0)":
            return _LastKnownPredictor()
        if algo == "Linear extrapolation (order 1)":
            return _LinearPredictor(max_history=max_hist)
        if algo == "Polynomial extrapolation":
            return _PolyPredictor(max_history=max_hist, degree=cfg["pred_poly_deg"])
        if algo == "Exponential smoothing (EMA)":
            return _EMAPredictor(max_history=max_hist, alpha=cfg["pred_ema_alpha"])
        if algo == "Kalman filter (constant velocity)":
            return _KalmanPredictor(
                q_pos=cfg["pred_kf_qpos"],
                q_vel=cfg["pred_kf_qvel"],
                r=cfg["pred_kf_r"],
            )
        return _LinearPredictor(max_history=max_hist)

    def _on_method_change(self) -> None:
        pass

    # ── Tab 3: Run & Progress ─────────────────────────────────────

    def _build_tab_run(self,
                        parent: tk.Frame) -> None:
        pad = {"padx": 8, "pady": 4}

        # control buttons
        btn_frame = tk.Frame(parent)
        btn_frame.pack(fill="x", **pad)

        btn_cfg = dict(
            font=("Arial", 10, "bold"),
            relief=tk.FLAT,
            padx=12, pady=4)

        self._start_btn = tk.Button(
            btn_frame, text="▶  Start tracking",
            bg="#e0e0e0", fg="#222222",
            activebackground="#d0d0d0",
            command=self._on_start,
            **btn_cfg)
        self._start_btn.pack(
            side="left", padx=(0, 6))

        self._pause_btn = tk.Button(
            btn_frame, text="⏸  Pause",
            bg="#e0e0e0", fg="#222222",
            activebackground="#d0d0d0",
            command=self._on_pause,
            state="disabled",
            **btn_cfg)
        self._pause_btn.pack(
            side="left", padx=(0, 6))

        self._stop_btn = tk.Button(
            btn_frame, text="■  Stop",
            bg="#e0e0e0", fg="#222222",
            activebackground="#d0d0d0",
            command=self._on_stop,
            state="disabled",
            **btn_cfg)
        self._stop_btn.pack(side="left")

        tk.Button(
            btn_frame,
            text="Help  (Ctrl-H)",
            font=("Arial", 8),
            command=self._open_help).pack(
            side="right")

        # progress bar
        prog_frame = tk.LabelFrame(
            parent, text="Progress",
            font=("Arial", 9), **pad)
        prog_frame.pack(fill="x", **pad)

        self._progress_bar = ttk.Progressbar(
            prog_frame,
            variable=self._progress_var,
            maximum=100.0,
            mode="determinate",
            length=600)
        self._progress_bar.pack(
            fill="x", padx=4, pady=4)

        tk.Label(
            prog_frame,
            textvariable=self._status_var,
            font=("Arial", 9),
            fg="#333333", anchor="w",
            justify="left").pack(
            fill="x", padx=4)

        # log
        log_frame = tk.LabelFrame(
            parent,
            text="Tracking log",
            font=("Arial", 9), **pad)
        log_frame.pack(
            fill="both", expand=True, **pad)

        self._log_widget = tk.Text(
            log_frame,
            font=("Courier", 8),
            wrap=tk.NONE,
            state="disabled",
            bg="#f8f8f8",
            height=16)
        log_vsb = ttk.Scrollbar(
            log_frame, orient="vertical",
            command=self._log_widget.yview)
        log_hsb = ttk.Scrollbar(
            log_frame, orient="horizontal",
            command=self._log_widget.xview)
        self._log_widget.configure(
            yscrollcommand=log_vsb.set,
            xscrollcommand=log_hsb.set)
        log_hsb.pack(side="bottom", fill="x")
        log_vsb.pack(side="right", fill="y")
        self._log_widget.pack(
            fill="both", expand=True)

    # ── inspector helpers ─────────────────────────────────────────

    def _log(self, msg: str) -> None:
        """Queue a log line.  Safe to call from any thread — only the
        main-thread poller (see _poll_ui_queue) ever touches the widget."""
        self._ui_queue.put(("log", msg))

    def _append_log_line(self, msg: str) -> None:
        if (self._log_widget is None
                or not self._log_widget.winfo_exists()):
            return
        self._log_widget.configure(state="normal")
        self._log_widget.insert(tk.END, msg + "\n")
        self._log_widget.see(tk.END)
        self._log_widget.configure(state="disabled")

    def _log_image_event(self, event: str, frame_index: int, path: str, img: np.ndarray | None, prev_index: int | None = None) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        base = Path(path).name
        if img is None:
            self._log(f"[{ts}] {event} frame={frame_index} file={base} imread=FAILED img_size=NA prev={prev_index}")
            return
        h, w = img.shape[:2]
        if prev_index is not None:
            self._log(f"[{ts}] {event} frame={frame_index} file={base} imread=OK img_size={w}x{h} prev={prev_index} next={frame_index}")
        else:
            self._log(f"[{ts}] {event} frame={frame_index} file={base} imread=OK img_size={w}x{h}")

    def _set_status(self, msg: str) -> None:
        self._ui_queue.put(("status", msg))

    def _set_progress(self, frac: float) -> None:
        pct = float(np.clip(frac * 100, 0, 100))
        self._ui_queue.put(("progress", pct))

    def _set_buttons(self, running: bool,
                      paused: bool = False) -> None:
        """Only ever called from the main thread (button handlers or
        the UI poller), so it applies directly without queueing."""
        if self._start_btn is None:
            return
        if running and not paused:
            self._start_btn.configure(state="disabled")
            self._pause_btn.configure(state="normal", text="⏸  Pause")
            self._stop_btn.configure(state="normal")
        elif running and paused:
            self._start_btn.configure(state="disabled")
            self._pause_btn.configure(state="normal", text="▶  Resume")
            self._stop_btn.configure(state="normal")
        else:
            self._start_btn.configure(state="normal")
            self._pause_btn.configure(state="disabled", text="⏸  Pause")
            self._stop_btn.configure(state="disabled")

    # ── worker -> UI queue poller (main thread only) ────────────────

    def _start_ui_poll(self) -> None:
        if self._ui_poll_active:
            return
        self._ui_poll_active = True
        self._poll_ui_queue()

    def _poll_ui_queue(self) -> None:
        if not self.canvas.winfo_exists():
            self._ui_poll_active = False
            return
        try:
            while True:
                item = self._ui_queue.get_nowait()
                kind = item[0]
                if kind == "log":
                    self._append_log_line(item[1])
                elif kind == "status":
                    self._status_var.set(item[1])
                elif kind == "progress":
                    self._progress_var.set(item[1])
                elif kind == "output":
                    if self._on_output_ready:
                        self._on_output_ready(
                            self.node_id, item[1])
                elif kind == "finish":
                    self._set_buttons(False)
                    self._status_var.set(item[1])
                    self._append_log_line(
                        "=== tracking finished ===")
        except queue.Empty:
            pass

        if self._tracking_active or not self._ui_queue.empty():
            self.canvas.after(30, self._poll_ui_queue)
        else:
            self._ui_poll_active = False

    # ── control callbacks ─────────────────────────────────────────

    def _on_start(self) -> None:
        if self._tracking_active:
            return
        # sync text from widgets (safe even before inspector is built)
        self._sync_text_widgets()
        files = self._get_file_list()
        pts   = _parse_points(
            self._initpts_text)

        if not files:
            messagebox.showwarning(
                "No files",
                "Please enter image file paths.")
            return
        if pts is None or pts.shape[0] < 1:
            messagebox.showwarning(
                "No points",
                "Please enter initial "
                "tracking points.")
            return

        # snapshot UI settings before starting the background thread
        cfg = {
            "method": self._method_var.get(),
            "period": self._period_var.get(),
            "subpix": self._subpix_var.get(),
            "subpix_win": self._subpix_win_var.get(),
            "subpix_iter": self._subpix_iter_var.get(),
            "subpix_eps": self._subpix_eps_var.get(),
            "lk_winsize": self._lk_winsize_var.get(),
            "lk_maxlevel": self._lk_maxlevel_var.get(),
            "lk_maxiter": self._lk_maxiter_var.get(),
            "lk_eps": self._lk_eps_var.get(),
            "lk_flags": self._lk_flags_var.get(),
            "lk_mineig": self._lk_mineig_var.get(),
            "pred_algo": self._pred_algo_var.get(),
            "pred_poly_deg": self._pred_poly_deg_var.get(),
            "pred_ema_alpha": self._pred_ema_alpha_var.get(),
            "pred_kf_qpos": self._pred_kf_qpos_var.get(),
            "pred_kf_qvel": self._pred_kf_qvel_var.get(),
            "pred_kf_r": self._pred_kf_r_var.get(),
            "pred_max_hist": self._pred_max_hist_var.get(),
        }

        # clear log
        if self._log_widget is not None:
            self._log_widget.configure(
                state="normal")
            self._log_widget.delete(
                "1.0", tk.END)
            self._log_widget.configure(
                state="disabled")

        self._stop_event.clear()
        self._pause_event.set()
        self._tracking_active = True
        self._set_buttons(True, False)
        self._progress_var.set(0.0)
        self._status_var.set("running")
        self._start_ui_poll()

        self._thread = threading.Thread(
            target=self._tracking_worker,
            args=(files, pts, cfg),
            daemon=True)
        self._thread.start()

    def _on_pause(self) -> None:
        if not self._tracking_active:
            return
        if self._pause_event.is_set():
            # currently running → pause
            self._pause_event.clear()
            self._set_buttons(True, True)
            self._set_status("paused")
            self._log("--- paused ---")
        else:
            # currently paused → resume
            self._pause_event.set()
            self._set_buttons(True, False)
            self._log("--- resumed ---")

    def _on_stop(self) -> None:
        self._stop_event.set()
        self._pause_event.set()  # unblock pause
        self._set_status("stopping...")

    # ── main tracking worker ──────────────────────────────────────

    def _tracking_worker(
        self,
        files: list[str],
        init_pts: np.ndarray,
        cfg: dict,
    ) -> None:
        """
        Runs in a background thread.
        Reads images, tracks points frame by frame,
        pushes one output row per frame.
        """
        n_files = len(files)
        N = init_pts.shape[0]

        self._log(
            f"Starting: {n_files} frames,"
            f" {N} points")
        self._log(
            f"Method: "
            f"{cfg['method']}")
        self._log(
            f"Period: "
            f"{cfg['period']}")

        # ----- load frame 1 -----
        img1_gray = _load_gray(files[0])
        self._log_image_event("imread", 1, files[0], img1_gray)
        if img1_gray is None:
            self._log(
                f"ERROR: cannot read {files[0]}")
            self._finish_worker()
            return

        # subpixel refinement on initPts
        cur_pts = init_pts.copy().astype(
            np.float32)
        if cfg["subpix"]:
            criteria = (
                cv2.TERM_CRITERIA_EPS
                | cv2.TERM_CRITERIA_MAX_ITER,
                cfg["subpix_iter"],
                cfg["subpix_eps"])
            cur_pts_sp = cv2.cornerSubPix(
                img1_gray,
                cur_pts.reshape(-1, 1, 2),
                (cfg["subpix_win"],
                 cfg["subpix_win"]),
                (-1, -1), criteria)
            cur_pts = cur_pts_sp.reshape(-1, 2)
            self._log(
                "SubPix applied to initPts")

        # tracking history:
        # tracked_pts_history[i] = N*2 float (NaN=lost)
        # index is 0-based (frame 0 = image 1)
        tracked_history: list[np.ndarray] = [
            cur_pts.copy().astype(np.float64)]
        image_history: list[np.ndarray] = [
            img1_gray]
        predictor = self._make_predictor(cfg)
        predictor_n = N

        # lost mask: True = lost forever
        lost_mask = np.zeros(N, dtype=bool)

        # push frame 1 result (identity, confidence 1)
        self._push_row(
            pts=cur_pts.astype(np.float64),
            status=np.ones(N, dtype=np.uint8),
            error=np.zeros(N, dtype=np.float32),
            prev_idx=1,
            frame_idx=1,
            lost_mask=lost_mask,
            n_total=n_files)

        # ----- loop over frames 2..n -----
        for frame_idx in range(2, n_files + 1):
            # check stop
            if self._stop_event.is_set():
                self._log(
                    f"Stopped at frame "
                    f"{frame_idx}")
                break

            # check pause (blocking wait)
            self._pause_event.wait()
            if self._stop_event.is_set():
                break

            file_path = files[frame_idx - 1]
            img_gray  = _load_gray(file_path)
            period    = cfg["period"]
            prev_idx  = _calc_prev_index(
                frame_idx, period)
            self._log_image_event("imread", frame_idx, file_path, img_gray, prev_index=prev_idx)
            if img_gray is None:
                self._log(
                    f"frame {frame_idx}: "
                    f"cannot read, skipping")
                # push NaN row
                nan_pts = np.full(
                    (N, 2), np.nan,
                    dtype=np.float64)
                self._push_row(
                    pts=nan_pts,
                    status=np.zeros(
                        N, dtype=np.uint8),
                    error=np.full(
                        N, np.nan,
                        dtype=np.float32),
                    prev_idx=-1,
                    frame_idx=frame_idx,
                    lost_mask=np.ones(
                        N, dtype=bool),
                    n_total=n_files)
                continue

            hist_idx  = prev_idx - 1   # 0-based

            # get prevImg and prevPts
            prev_img  = image_history[hist_idx]
            prev_pts  = tracked_history[
                hist_idx].copy().astype(
                np.float32)

            # set lost points to NaN before tracking
            out_pts   = prev_pts.copy().astype(
                np.float64)
            out_pts[lost_mask] = np.nan

            # track non-lost points
            active = ~lost_mask
            n_active = int(active.sum())

            if n_active == 0:
                self._log(
                    f"frame {frame_idx}: "
                    f"all points lost")
                tracked_history.append(
                    out_pts.copy())
                image_history.append(img_gray)
                status_arr = np.zeros(
                    N, dtype=np.uint8)
                error_arr  = np.full(
                    N, np.nan,
                    dtype=np.float32)
                self._push_row(
                    pts=out_pts,
                    status=status_arr,
                    error=error_arr,
                    prev_idx=prev_idx,
                    frame_idx=frame_idx,
                    lost_mask=lost_mask,
                    n_total=n_files)
                continue

            prev_pts_active = prev_pts[
                active].reshape(-1, 2)
            if n_active != predictor_n:
                # active-point count shrank (a point was lost) — the
                # predictor's history buffer has a fixed shape, so it
                # must restart rather than crash on the new size.
                predictor = self._make_predictor(cfg)
                predictor_n = n_active
                self._log(
                    f"frame {frame_idx}: predictor reset "
                    f"(active points now {n_active}/{N})")
            pred_guess, _ = predictor.update_and_predict(prev_pts_active)
            pred_guess = np.asarray(pred_guess, dtype=np.float32)

            # --- optical flow ---
            ws  = cfg["lk_winsize"]
            ml  = cfg["lk_maxlevel"]
            mi  = cfg["lk_maxiter"]
            eps = cfg["lk_eps"]
            flg = cfg["lk_flags"]
            me  = cfg["lk_mineig"]

            criteria = (
                cv2.TERM_CRITERIA_EPS
                | cv2.TERM_CRITERIA_MAX_ITER,
                mi, eps)

            self._log(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"calcOpticalFlowPyrLK(prev={prev_idx}, next={frame_idx}) "
                f"prev_img={Path(files[prev_idx - 1]).name} next_img={Path(file_path).name} "
                f"N={len(prev_pts_active)} winSize=({ws},{ws}) maxLevel={ml} flags={flg} minEigThreshold={me}")
            next_pts_lk, status_lk, err_lk = \
                cv2.calcOpticalFlowPyrLK(
                    prev_img, img_gray,
                    prev_pts_active.astype(np.float32).reshape(-1, 1, 2),
                    pred_guess.reshape(-1, 1, 2),
                    winSize=(ws, ws),
                    maxLevel=ml,
                    criteria=criteria,
                    flags=flg,
                    minEigThreshold=me)

            next_pts_lk = next_pts_lk.reshape(
                -1, 2).astype(np.float64)
            status_lk   = status_lk.ravel()
            err_lk      = err_lk.ravel()

            # full output arrays
            status_full = np.zeros(
                N, dtype=np.uint8)
            error_full  = np.full(
                N, np.nan, dtype=np.float32)
            active_idx  = np.where(active)[0]

            for local_i, global_i in enumerate(
                    active_idx):
                s = int(status_lk[local_i])
                status_full[global_i] = s
                if s == 1:
                    out_pts[global_i] = \
                        next_pts_lk[local_i]
                    error_full[global_i] = \
                        float(err_lk[local_i])
                else:
                    out_pts[global_i] = np.nan
                    lost_mask[global_i] = True

            n_lost  = int(lost_mask.sum())
            n_good  = N - n_lost
            self._log(
                f"frame {frame_idx:4d}"
                f"  prev={prev_idx:4d}"
                f"  tracked={n_good}/{N}"
                f"  lost={n_lost}"
                f"  {Path(file_path).name}")

            tracked_history.append(
                out_pts.copy())
            image_history.append(img_gray)

            self._push_row(
                pts=out_pts,
                status=status_full,
                error=error_full,
                prev_idx=prev_idx,
                frame_idx=frame_idx,
                lost_mask=lost_mask,
                n_total=n_files)

        self._finish_worker()

    # ── output helpers ────────────────────────────────────────────

    def _push_row(
        self,
        pts:       np.ndarray,
        status:    np.ndarray,
        error:     np.ndarray,
        prev_idx:  int,
        frame_idx: int,
        lost_mask: np.ndarray,
        n_total:   int,
    ) -> None:
        """
        Build a flat 1D row and push to output pins.
        Row layout: [pts_flat (N*2),
                     status (N),
                     error (N),
                     prev_idx (1),
                     frame_idx (1)]
        """
        pts_flat = pts.ravel().copy()
        pts_flat[np.isnan(pts_flat)] = np.nan

        row = np.concatenate([
            pts_flat,
            status.astype(np.float64),
            error.astype(np.float64),
            np.array([float(prev_idx)]),
            np.array([float(frame_idx)]),
        ])

        self._done_counter += 1
        outputs = {
            "row_data": row,
            "row_done": self._done_counter,
        }
        self.push_output(outputs)

        # update progress
        frac = frame_idx / max(n_total, 1)
        self._set_progress(frac)
        self._set_status(
            f"frame {frame_idx}/{n_total}"
            f"  ({frac*100:.1f}%)")

    def _finish_worker(self) -> None:
        self._tracking_active = False
        self._all_done_counter += 1

        # fire all_done trigger
        self.push_output({
            "all_done": self._all_done_counter})

        status_msg = ("done" if not
                      self._stop_event.is_set()
                      else "stopped")
        self._ui_queue.put(("finish", status_msg))

    # ── STREAMING interface ───────────────────────────────────────

    def start_stream(self) -> None:
        pass   # user-controlled via Start button

    def stop_stream(self) -> None:
        self._stop_event.set()
        self._pause_event.set()
        if (self._thread is not None
                and self._thread.is_alive()):
            self._thread.join(timeout=3.0)

    def push_output(self,
                     outputs: dict) -> None:
        # worker thread must not call into Tk-driven callbacks directly
        if threading.current_thread() is threading.main_thread():
            if self._on_output_ready:
                self._on_output_ready(self.node_id, outputs)
        else:
            self._ui_queue.put(("output", outputs))

    def compute(self,
                inputs: dict) -> dict:
        return {}

    # ── help popup ────────────────────────────────────────────────

    def _open_help(self) -> None:
        if (self._help_popup is not None
                and self._help_popup.winfo_exists()):
            self._help_popup.lift()
            return

        popup = tk.Toplevel()
        popup.title(
            "Point Tracking Node — Help")
        popup.geometry("720x600")
        popup.resizable(True, True)
        try:
            px, py = (
                self.canvas.winfo_pointerxy())
            popup.geometry(
                f"+{px+16}+{py+16}")
        except Exception:
            pass

        body = tk.Frame(
            popup, bg="#f8f8f8",
            padx=10, pady=8)
        body.pack(fill="both", expand=True)

        txt = tk.Text(
            body, font=("Courier", 9),
            bg="#f8f8f8", fg="#222222",
            wrap=tk.WORD, relief=tk.FLAT)
        vsb = ttk.Scrollbar(
            body, orient="vertical",
            command=txt.yview)
        txt.configure(
            yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", _HELP_TEXT)
        txt.configure(state="disabled")

        tk.Button(
            body, text="Close",
            font=("Arial", 9),
            command=popup.destroy).pack(
            anchor="e", pady=(8, 0))

        popup.bind(
            "<Escape>",
            lambda _e: popup.destroy())
        popup.protocol(
            "WM_DELETE_WINDOW",
            popup.destroy)
        self._help_popup = popup

    # ── serialization ─────────────────────────────────────────────

    def _sync_text_widgets(self) -> None:
        files_widget = getattr(self, "_files_text_widget", None)
        if files_widget is not None and files_widget.winfo_exists():
            self._files_text = files_widget.get("1.0", tk.END).strip()

        pts_widget = getattr(self, "_initpts_widget", None)
        if pts_widget is not None and pts_widget.winfo_exists():
            self._initpts_text = pts_widget.get("1.0", tk.END).strip()

        try:
            self._refresh_count_labels()
        except Exception:
            pass

    def _get_file_list(self) -> list[str]:
        base = get_project_directory()
        files = []
        for line in self._files_text\
                .splitlines():
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            if not p.is_absolute() \
                    and base is not None:
                p = base / p
            files.append(str(p))
        return files

    def get_params(self) -> dict:
        self._sync_text_widgets()
        return {
            "files_text":   self._files_text,
            "initpts_text": self._initpts_text,
            "method":
                self._method_var.get(),
            "pred_algo": self._pred_algo_var.get(),
            "pred_poly_deg": self._pred_poly_deg_var.get(),
            "pred_ema_alpha": self._pred_ema_alpha_var.get(),
            "pred_kf_qpos": self._pred_kf_qpos_var.get(),
            "pred_kf_qvel": self._pred_kf_qvel_var.get(),
            "pred_kf_r": self._pred_kf_r_var.get(),
            "pred_max_hist": self._pred_max_hist_var.get(),
            "lk_winsize":
                self._lk_winsize_var.get(),
            "lk_maxlevel":
                self._lk_maxlevel_var.get(),
            "lk_maxiter":
                self._lk_maxiter_var.get(),
            "lk_eps":
                self._lk_eps_var.get(),
            "lk_flags":
                self._lk_flags_var.get(),
            "lk_mineig":
                self._lk_mineig_var.get(),
            "period":
                self._period_var.get(),
            "subpix":
                self._subpix_var.get(),
            "subpix_win":
                self._subpix_win_var.get(),
            "subpix_iter":
                self._subpix_iter_var.get(),
            "subpix_eps":
                self._subpix_eps_var.get(),
        }

    def set_params(self, params: dict) -> None:
        self._init_state()
        self._files_text = str(
            params.get("files_text", ""))
        self._initpts_text = str(
            params.get("initpts_text", ""))
        self._method_var.set(
            params.get("method",
                       "Optical Flow (LK)"))
        self._pred_algo_var.set(
            params.get("pred_algo",
                       "Linear extrapolation (order 1)"))
        self._pred_poly_deg_var.set(
            int(params.get("pred_poly_deg", 2)))
        self._pred_ema_alpha_var.set(
            float(params.get("pred_ema_alpha", 0.5)))
        self._pred_kf_qpos_var.set(
            float(params.get("pred_kf_qpos", 1e-2)))
        self._pred_kf_qvel_var.set(
            float(params.get("pred_kf_qvel", 1e-3)))
        self._pred_kf_r_var.set(
            float(params.get("pred_kf_r", 1e-1)))
        self._pred_max_hist_var.set(
            int(params.get("pred_max_hist", 10)))
        self._lk_winsize_var.set(
            int(params.get("lk_winsize", 15)))
        self._lk_maxlevel_var.set(
            int(params.get("lk_maxlevel", 3)))
        self._lk_maxiter_var.set(
            int(params.get("lk_maxiter", 30)))
        self._lk_eps_var.set(
            float(params.get("lk_eps", 0.01)))
        self._lk_flags_var.set(
            int(params.get("lk_flags", 0)))
        self._lk_mineig_var.set(
            float(params.get("lk_mineig",
                              1e-4)))
        self._period_var.set(
            int(params.get("period", 0)))
        self._subpix_var.set(
            bool(params.get("subpix", False)))
        self._subpix_win_var.set(
            int(params.get("subpix_win", 11)))
        self._subpix_iter_var.set(
            int(params.get("subpix_iter", 30)))
        self._subpix_eps_var.set(
            float(params.get("subpix_eps",
                              0.001)))

        # restore text into widgets if open
        if (self._files_text_widget is not None
                and self._files_text_widget
                .winfo_exists()):
            self._files_text_widget.delete(
                "1.0", tk.END)
            self._files_text_widget.insert(
                "1.0", self._files_text)
        if (self._initpts_widget is not None
                and self._initpts_widget
                .winfo_exists()):
            self._initpts_widget.delete(
                "1.0", tk.END)
            self._initpts_widget.insert(
                "1.0", self._initpts_text)

    def close_inspector(self) -> None:
        self._sync_text_widgets()
        super().close_inspector()
        self._files_text_widget  = None
        self._initpts_widget     = None
        self._log_widget         = None
        self._progress_bar       = None
        self._start_btn          = None
        self._pause_btn          = None
        self._stop_btn           = None

    def on_destroy(self) -> None:
        self._stop_event.set()
        self._pause_event.set()
        if (self._help_popup is not None
                and self._help_popup.winfo_exists()):
            self._help_popup.destroy()
        super().on_destroy()
