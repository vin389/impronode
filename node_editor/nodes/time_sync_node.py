# node_editor/nodes/time_sync_node.py
"""
Time-series synchronization node.

Aligns a second time series (t2, x2) onto a first one (t1, x1) by an
adjustable time shift. The inspector lets the user preview the shift
interactively (trial state); only pressing "Apply" commits it to the
output pin (applied state). t1, x1 and x2 are read-only reference
inputs and are never re-exposed on an output pin -- only the
synchronized t2 is.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import colorchooser, messagebox, ttk

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
from scipy.signal import correlate, correlation_lags

from node_editor.base_node import BaseNode
from node_editor.pin_types import PinDef, PinSchema, PinType


class TimeSyncNode(BaseNode):
    """Shift a second time series in time to align it with a first one.

    Inputs:
        t1  (ARRAY, optional) -- time vector for the reference series.
        x1  (ARRAY)           -- reference series values.
        t2  (ARRAY, optional) -- time vector for the series to be shifted.
        x2  (ARRAY)           -- series values to be shifted.
        dt  (SCALAR, optional)-- uniform sample interval, used to build a
                                  time vector for whichever of t1 / t2 is
                                  not connected.

    Output:
        t2_sync (ARRAY) -- t2 shifted by the applied time offset. This is
                            the only thing that changes downstream; t1,
                            x1 and x2 are not re-exposed.
    """

    NODE_TYPE = "time_sync"
    DISPLAY_NAME = "Time Sync"
    CATEGORY = "process"
    SEARCH_KEYWORDS = ("synchronization", "time shift", "align", "cross-correlation", "signal", "time series")

    NODE_WIDTH = 150
    NODE_HEIGHT = 90

    # Marker choices exposed in the style controls. "None" (as a string)
    # means "no marker" -- matplotlib itself uses the string "None" or the
    # Python None object interchangeably, but we keep an explicit string
    # here so it can be stored in a combobox and in the params dict.
    MARKER_CHOICES = ("None", "o", "s", "^", "v", "D", "x", "+", ".")
    INTERP_CHOICES = ("linear", "cubic")
    DIFF_CHOICES = ("0", "1", "2")
    METRIC_CHOICES = ("rmse", "correlation")

    # ══ Construction ════════════════════════════════════════════════

    def __init__(self, node_id: str, canvas: tk.Canvas):
        super().__init__(node_id, canvas)

        # --- Applied state: the ONLY thing that affects the output pin ---
        self._t_shift_applied: float = 0.0

        # --- Trial state: freely editable in the inspector, has no
        # effect on the output pin until "Apply" is pressed ---
        self._t_shift_trial: float = 0.0
        self._slider_min: float = -1.0
        self._slider_max: float = 1.0
        self._interp_method: str = "linear"      # "linear" | "cubic"
        self._diff_order: int = 0                # 0, 1, or 2
        self._error_metric: str = "rmse"         # "rmse" | "correlation"
        self._normalize_error: bool = True
        self._show_normalized_plot: bool = False

        # Per-series plot style. "series1" is (t1, x1); "series2" is the
        # trial-shifted (t2_trial, x2).
        self._style: dict = {
            "series1": {
                "color": "#1f77b4", "line_width": 1.5,
                "marker": "None", "marker_size": 4, "marker_color": "#1f77b4",
            },
            "series2": {
                "color": "#d62728", "line_width": 1.5,
                "marker": "None", "marker_size": 4, "marker_color": "#d62728",
            },
            "legend_on": True,
        }

        # Cached inputs from the last compute() call, kept so the
        # inspector can redraw without forcing a graph recompute.
        self._t1: np.ndarray | None = None
        self._x1: np.ndarray | None = None
        self._t2: np.ndarray | None = None
        self._x2: np.ndarray | None = None

        # Matplotlib objects, created lazily in build_inspector().
        self._fig: Figure | None = None
        self._ax = None
        self._canvas_widget: FigureCanvasTkAgg | None = None
        self._pan_state: dict = {
            "active": False, "x0": None, "y0": None, "xlim0": None, "ylim0": None,
        }

        # Extra canvas item for the on-node status line (created in build_body).
        self._status_item: int | None = None

    # ══ Pin schema ══════════════════════════════════════════════════

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[
                PinDef(name="t1", type=PinType.ARRAY, label="t1", optional=True),
                PinDef(name="x1", type=PinType.ARRAY, label="x1"),
                PinDef(name="t2", type=PinType.ARRAY, label="t2", optional=True),
                PinDef(name="x2", type=PinType.ARRAY, label="x2"),
                PinDef(name="dt", type=PinType.SCALAR, label="dt", optional=True),
            ],
            outputs=[
                PinDef(name="t2_sync", type=PinType.ARRAY, label="t2_sync"),
            ],
        )

    # ══ Compute ═════════════════════════════════════════════════════

    @staticmethod
    def _resolve_time_vector(t: np.ndarray | None, x: np.ndarray | None,
                              dt: float | None) -> np.ndarray | None:
        """Return an explicit time vector for one series.

        Falls back to a uniform grid (dt * arange(n)) when the time
        vector itself is not connected but a sample interval is.
        """
        if t is not None:
            return np.asarray(t, dtype=float)
        if x is not None and dt is not None and dt > 0:
            return np.arange(len(x), dtype=float) * float(dt)
        return None

    def compute(self, inputs: dict) -> dict:
        x1 = inputs.get("x1")
        x2 = inputs.get("x2")
        dt = inputs.get("dt")

        # Cache everything so the inspector can plot without recomputing.
        self._x1 = np.asarray(x1, dtype=float) if x1 is not None else None
        self._x2 = np.asarray(x2, dtype=float) if x2 is not None else None
        self._t1 = self._resolve_time_vector(inputs.get("t1"), self._x1, dt)
        self._t2 = self._resolve_time_vector(inputs.get("t2"), self._x2, dt)

        if self._t2 is None:
            # Nothing to synchronize -- no time base available for series 2.
            return {"t2_sync": None}

        # Only the APPLIED shift ever reaches the output pin.
        return {"t2_sync": self._t2 + self._t_shift_applied}

    # ══ Serialization ═══════════════════════════════════════════════

    def get_params(self) -> dict:
        return {
            "t_shift_applied": self._t_shift_applied,
            "slider_min": self._slider_min,
            "slider_max": self._slider_max,
            "interp_method": self._interp_method,
            "diff_order": self._diff_order,
            "error_metric": self._error_metric,
            "normalize_error": self._normalize_error,
            "show_normalized_plot": self._show_normalized_plot,
            "style": self._style,
        }

    def set_params(self, params: dict) -> None:
        self._t_shift_applied = float(params.get("t_shift_applied", 0.0))
        self._t_shift_trial = self._t_shift_applied
        self._slider_min = float(params.get("slider_min", -1.0))
        self._slider_max = float(params.get("slider_max", 1.0))
        self._interp_method = params.get("interp_method", "linear")
        self._diff_order = int(params.get("diff_order", 0))
        self._error_metric = params.get("error_metric", "rmse")
        self._normalize_error = bool(params.get("normalize_error", True))
        self._show_normalized_plot = bool(params.get("show_normalized_plot", False))

        saved_style = params.get("style")
        if isinstance(saved_style, dict):
            # Merge rather than replace, so a partially-saved dict from an
            # older version of this node does not wipe out defaults.
            for key in ("series1", "series2"):
                if key in saved_style and isinstance(saved_style[key], dict):
                    self._style[key].update(saved_style[key])
            if "legend_on" in saved_style:
                self._style["legend_on"] = bool(saved_style["legend_on"])

        if getattr(self, "_status_item", None) is not None:
            self.canvas.itemconfig(self._status_item, text=self._status_text())

    def deserialize(self, data: dict) -> None:
        super().deserialize(data)
        if getattr(self, "_status_item", None) is not None:
            self.canvas.itemconfig(self._status_item, text=self._status_text())

    # ══ Canvas body (small node face; the real UI lives in the inspector) ══

    def _status_text(self) -> str:
        return f"applied shift: {self._t_shift_applied:.4g}"

    def build_body(self) -> None:
        x, y, w, h = self.x, self.y, self.width, self.height

        self._body_rect = self.canvas.create_rectangle(
            x, y, x + w, y + h,
            fill=self.BODY_COLOR, outline="#666666", width=1,
            tags=(self.node_id,),
        )
        self._canvas_items.append(self._body_rect)

        self._title_item = self.canvas.create_text(
            x + w / 2, y + 13,
            text=self.get_canvas_title(),
            fill=self.TITLE_COLOR, font=("Arial", 9, "bold"),
            tags=(self.node_id,),
        )
        self._canvas_items.append(self._title_item)

        self._status_item = self.canvas.create_text(
            x + w / 2, y + h - 12,
            text=self._status_text(),
            fill="#555555", font=("Arial", 7),
            tags=(self.node_id,),
        )
        self._canvas_items.append(self._status_item)

    def on_resize(self, old_width: int, old_height: int,
                  new_width: int, new_height: int) -> None:
        # The editor already repositions _body_rect and _title_item
        # generically; keep the status line pinned to the bottom edge.
        if getattr(self, "_status_item", None) is not None:
            self.canvas.coords(self._status_item, self.x + new_width / 2, self.y + new_height - 12)

    # ══ Sync-error metric ═══════════════════════════════════════════

    def _compute_sync_error(self, t_shift: float) -> float | None:
        """Sync error between (t1, x1) and (t2 + t_shift, x2) over their
        time overlap. Lower is better for both supported metrics.
        Returns None when the inputs are missing or there is no overlap.
        """
        t1, x1, t2, x2 = self._t1, self._x1, self._t2, self._x2
        if t1 is None or x1 is None or t2 is None or x2 is None:
            return None

        t2_trial = t2 + t_shift
        t_lo = max(t1.min(), t2_trial.min())
        t_hi = min(t1.max(), t2_trial.max())
        mask = (t1 >= t_lo) & (t1 <= t_hi)
        if mask.sum() < 3:
            return None

        t_eval = t1[mask]
        a = x1[mask].copy()

        if self._interp_method == "cubic":
            # extrapolate=False -> NaN outside the shifted series' own
            # domain, filtered out below instead of trusting a spline
            # extrapolation.
            b = CubicSpline(t2_trial, x2, extrapolate=False)(t_eval)
        else:
            b = np.interp(t_eval, t2_trial, x2)

        valid = ~np.isnan(b)
        a, b = a[valid], b[valid]
        if len(a) < 3:
            return None

        for _ in range(self._diff_order):
            a, b = np.diff(a), np.diff(b)
            if len(a) < 2:
                return None

        if self._normalize_error:
            a = (a - a.mean()) / (a.std() + 1e-12)
            b = (b - b.mean()) / (b.std() + 1e-12)

        if self._error_metric == "correlation":
            r = float(np.corrcoef(a, b)[0, 1])
            return 1.0 - r
        return float(np.sqrt(np.mean((a - b) ** 2)))  # rmse

    def _auto_align_cross_correlation(self) -> float | None:
        """Return the t_shift (added to t2) that best aligns x2 onto x1
        via FFT cross-correlation on a common uniform resample.
        """
        t1, x1, t2, x2 = self._t1, self._x1, self._t2, self._x2
        if t1 is None or x1 is None or t2 is None or x2 is None:
            return None

        dt = float(np.median(np.diff(t1)))
        if dt <= 0:
            return None
        t_lo = max(t1.min(), t2.min())
        t_hi = min(t1.max(), t2.max())
        if t_hi <= t_lo:
            return None
        t_grid = np.arange(t_lo, t_hi, dt)
        if len(t_grid) < 4:
            return None

        a = np.interp(t_grid, t1, x1)
        b = np.interp(t_grid, t2, x2)
        for _ in range(self._diff_order):
            a, b = np.diff(a), np.diff(b)
        if len(a) < 2:
            return None

        a = (a - a.mean()) / (a.std() + 1e-12)
        b = (b - b.mean()) / (b.std() + 1e-12)

        corr = correlate(a, b, mode="full", method="fft")
        lags = correlation_lags(len(a), len(b)) * dt
        best_lag = float(lags[int(np.argmax(corr))])

        return float(np.clip(best_lag, self._slider_min, self._slider_max))

    def _refine_shift(self, start_shift: float) -> float | None:
        """Locally minimize the current sync-error metric, bounded by the
        slider range, starting from start_shift. Gives sub-sample
        precision that the lag-quantized cross-correlation search cannot.
        """
        if self._t1 is None or self._x1 is None or self._t2 is None or self._x2 is None:
            return None

        def _cost(shift: float) -> float:
            err = self._compute_sync_error(shift)
            return err if err is not None else float("inf")

        result = minimize_scalar(
            _cost, bounds=(self._slider_min, self._slider_max), method="bounded",
        )
        if not result.success:
            return None
        return float(result.x)

    # ══ Inspector UI ════════════════════════════════════════════════

    def build_inspector(self, parent: tk.Frame) -> None:
        # -- Plot --
        plot_frame = tk.Frame(parent)
        plot_frame.pack(fill="both", expand=True)
        self._build_plot(plot_frame)

        # -- Sync error readout --
        self._insp_error_var = tk.StringVar(value="")
        tk.Label(parent, textvariable=self._insp_error_var, anchor="w",
                 font=("Arial", 9)).pack(fill="x", pady=(4, 0))

        # -- Time-shift slider with adjustable range --
        self._build_shift_controls(parent)

        # -- Interpolation / differencing / metric controls --
        self._build_analysis_controls(parent)

        # -- Plot style controls --
        self._build_style_controls(parent)

        # -- Overlay / legend toggles --
        self._build_overlay_controls(parent)

        # -- Apply --
        apply_row = tk.Frame(parent)
        apply_row.pack(fill="x", pady=(8, 0))
        tk.Button(apply_row, text="Apply", width=12,
                  command=self._on_apply_clicked).pack(side="right")

        # Initial draw.
        self._redraw_plot()

    # -- Plot embedding, pan (drag) and wheel zoom -----------------------

    def _build_plot(self, parent: tk.Frame) -> None:
        self._fig = Figure(figsize=(6, 4), dpi=100)
        self._ax = self._fig.add_subplot(111)
        self._canvas_widget = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas_widget.get_tk_widget().pack(fill="both", expand=True)

        # Wheel zoom, centered on the cursor's data coordinates.
        self._canvas_widget.mpl_connect("scroll_event", self._on_plot_scroll)
        # Left-drag pan.
        self._canvas_widget.mpl_connect("button_press_event", self._on_plot_press)
        self._canvas_widget.mpl_connect("motion_notify_event", self._on_plot_drag)
        self._canvas_widget.mpl_connect("button_release_event", self._on_plot_release)

    def _on_plot_scroll(self, event) -> None:
        if event.xdata is None or event.ydata is None or self._ax is None:
            return

        factor = 0.9 if event.button == "up" else 1.1  # scroll up = zoom in
        xlim, ylim = self._ax.get_xlim(), self._ax.get_ylim()
        key = getattr(event, "key", None)

        # Ctrl+wheel: zoom the x-axis (time) only.
        # Shift+wheel: zoom the y-axis (value) only.
        # No modifier: zoom both axes together.
        if key in ("control", "ctrl"):
            new_xlim = [event.xdata - (event.xdata - v) * factor for v in xlim]
            self._ax.set_xlim(new_xlim)
        elif key == "shift":
            new_ylim = [event.ydata - (event.ydata - v) * factor for v in ylim]
            self._ax.set_ylim(new_ylim)
        else:
            new_xlim = [event.xdata - (event.xdata - v) * factor for v in xlim]
            new_ylim = [event.ydata - (event.ydata - v) * factor for v in ylim]
            self._ax.set_xlim(new_xlim)
            self._ax.set_ylim(new_ylim)
        self._canvas_widget.draw_idle()

    def _on_plot_press(self, event) -> None:
        if event.button != 1 or event.xdata is None or self._ax is None:
            return
        self._pan_state.update(
            active=True, x0=event.xdata, y0=event.ydata,
            xlim0=self._ax.get_xlim(), ylim0=self._ax.get_ylim(),
        )

    def _on_plot_drag(self, event) -> None:
        if not self._pan_state["active"] or event.xdata is None or self._ax is None:
            return
        dx = event.xdata - self._pan_state["x0"]
        dy = event.ydata - self._pan_state["y0"]
        x0, x1 = self._pan_state["xlim0"]
        y0, y1 = self._pan_state["ylim0"]
        self._ax.set_xlim(x0 - dx, x1 - dx)
        self._ax.set_ylim(y0 - dy, y1 - dy)
        self._canvas_widget.draw_idle()

    def _on_plot_release(self, _event) -> None:
        self._pan_state["active"] = False

    # -- Sync error label -------------------------------------------------

    def _update_sync_error_label(self) -> None:
        err = self._compute_sync_error(self._t_shift_trial)
        if err is None:
            self._insp_error_var.set("Sync error: n/a (no overlap or missing input)")
            return
        self._insp_error_var.set(
            f"Sync error: {err:.6g}  "
            f"({self._error_metric}, {self._interp_method}, diff={self._diff_order})"
        )

    # -- Full plot redraw (called on release / style changes, not on drag) --

    def _redraw_plot(self) -> None:
        ax = self._ax
        had_data = bool(ax.lines)
        xlim, ylim = (ax.get_xlim(), ax.get_ylim()) if had_data else (None, None)
        ax.clear()

        t2_trial = (self._t2 + self._t_shift_trial) if self._t2 is not None else None
        x1, x2 = self._x1, self._x2

        if self._show_normalized_plot:
            if x1 is not None:
                x1 = (x1 - x1.mean()) / (x1.std() + 1e-12)
            if x2 is not None:
                x2 = (x2 - x2.mean()) / (x2.std() + 1e-12)

        s1, s2 = self._style["series1"], self._style["series2"]
        if self._t1 is not None and x1 is not None:
            ax.plot(
                self._t1, x1, label="series 1 (t1, x1)",
                color=s1["color"], linewidth=s1["line_width"],
                marker=None if s1["marker"] == "None" else s1["marker"],
                markersize=s1["marker_size"], markerfacecolor=s1["marker_color"],
            )
        if t2_trial is not None and x2 is not None:
            ax.plot(
                t2_trial, x2, label="series 2 (t2_trial, x2)",
                color=s2["color"], linewidth=s2["line_width"],
                marker=None if s2["marker"] == "None" else s2["marker"],
                markersize=s2["marker_size"], markerfacecolor=s2["marker_color"],
            )

        if self._style["legend_on"]:
            ax.legend(loc="upper right", fontsize=8)
        ax.set_xlabel("time")
        ax.set_ylabel("normalized value" if self._show_normalized_plot else "value")

        if had_data:  # preserve the user's pan/zoom across redraws
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        self._canvas_widget.draw_idle()
        self._update_sync_error_label()

    # -- Time-shift slider (live sync-error, deferred full redraw) --------

    def _build_shift_controls(self, parent: tk.Frame) -> None:
        row = tk.Frame(parent)
        row.pack(fill="x", pady=(8, 0))

        tk.Label(row, text="t_shift range:").pack(side="left")

        self._insp_min_var = tk.StringVar(value=str(self._slider_min))
        min_entry = tk.Entry(row, textvariable=self._insp_min_var, width=8)
        min_entry.pack(side="left", padx=(4, 0))

        self._insp_shift_var = tk.DoubleVar(value=self._t_shift_trial)
        self._insp_shift_scale = tk.Scale(
            row, variable=self._insp_shift_var, orient=tk.HORIZONTAL,
            from_=self._slider_min, to=self._slider_max,
            resolution=0.001, showvalue=True, length=280,
            # Fires continuously while dragging: cheap sync-error-only
            # recompute here. The full plot redraw is deferred to the
            # mouse-release / key-release handlers below.
            command=lambda _v: self._on_shift_dragging(),
        )
        self._insp_shift_scale.pack(side="left", fill="x", expand=True, padx=6)

        self._insp_max_var = tk.StringVar(value=str(self._slider_max))
        max_entry = tk.Entry(row, textvariable=self._insp_max_var, width=8)
        max_entry.pack(side="left")

        min_entry.bind("<Return>", lambda _e: self._apply_slider_range())
        max_entry.bind("<Return>", lambda _e: self._apply_slider_range())

        # Full redraw once dragging stops. <ButtonRelease-1> covers mouse
        # drags and trough clicks; <KeyRelease> covers arrow-key nudges,
        # which never fire a ButtonRelease event.
        self._insp_shift_scale.bind("<ButtonRelease-1>", lambda _e: self._on_shift_released())
        self._insp_shift_scale.bind("<KeyRelease>", lambda _e: self._on_shift_released())

    def _apply_slider_range(self) -> None:
        try:
            lo = float(self._insp_min_var.get())
            hi = float(self._insp_max_var.get())
        except ValueError:
            return  # leave the previous range in place on invalid input
        if hi <= lo:
            return
        self._slider_min, self._slider_max = lo, hi
        self._insp_shift_scale.configure(from_=lo, to=hi)
        clamped = min(max(self._t_shift_trial, lo), hi)
        self._insp_shift_var.set(clamped)
        self._t_shift_trial = clamped
        self._redraw_plot()

    def _on_shift_dragging(self) -> None:
        """Cheap path: called on every value change while the slider is
        being dragged. Updates the trial shift and the sync-error number
        only -- no matplotlib redraw, so large arrays stay responsive.
        """
        self._t_shift_trial = self._insp_shift_var.get()
        self._update_sync_error_label()

    def _on_shift_released(self) -> None:
        """Expensive path: called once dragging stops. Does the full
        plot redraw with the final trial shift.
        """
        self._t_shift_trial = self._insp_shift_var.get()
        self._redraw_plot()

    # -- Interpolation / differencing / metric / auto-align ---------------

    def _build_analysis_controls(self, parent: tk.Frame) -> None:
        row1 = tk.Frame(parent)
        row1.pack(fill="x", pady=(8, 0))

        tk.Label(row1, text="Interp:").pack(side="left")
        self._insp_interp_var = tk.StringVar(value=self._interp_method)
        interp_box = ttk.Combobox(
            row1, textvariable=self._insp_interp_var, values=self.INTERP_CHOICES,
            state="readonly", width=8,
        )
        interp_box.pack(side="left", padx=(4, 12))
        interp_box.bind("<<ComboboxSelected>>", lambda _e: self._on_analysis_setting_changed())

        tk.Label(row1, text="Diff order:").pack(side="left")
        self._insp_diff_var = tk.StringVar(value=str(self._diff_order))
        diff_box = ttk.Combobox(
            row1, textvariable=self._insp_diff_var, values=self.DIFF_CHOICES,
            state="readonly", width=4,
        )
        diff_box.pack(side="left", padx=(4, 12))
        diff_box.bind("<<ComboboxSelected>>", lambda _e: self._on_analysis_setting_changed())

        tk.Label(row1, text="Metric:").pack(side="left")
        self._insp_metric_var = tk.StringVar(value=self._error_metric)
        metric_box = ttk.Combobox(
            row1, textvariable=self._insp_metric_var, values=self.METRIC_CHOICES,
            state="readonly", width=10,
        )
        metric_box.pack(side="left", padx=(4, 12))
        metric_box.bind("<<ComboboxSelected>>", lambda _e: self._on_analysis_setting_changed())

        self._insp_normalize_var = tk.BooleanVar(value=self._normalize_error)
        tk.Checkbutton(
            row1, text="Normalize error", variable=self._insp_normalize_var,
            command=self._on_analysis_setting_changed,
        ).pack(side="left")

        row2 = tk.Frame(parent)
        row2.pack(fill="x", pady=(4, 0))
        tk.Button(row2, text="Auto Align (cross-correlation)",
                  command=self._on_auto_align_clicked).pack(side="left")
        tk.Button(row2, text="Refine (minimize error)",
                  command=self._on_refine_clicked).pack(side="left", padx=(8, 0))

    def _on_analysis_setting_changed(self) -> None:
        self._interp_method = self._insp_interp_var.get()
        self._diff_order = int(self._insp_diff_var.get())
        self._error_metric = self._insp_metric_var.get()
        self._normalize_error = bool(self._insp_normalize_var.get())
        self._update_sync_error_label()

    def _on_auto_align_clicked(self) -> None:
        shift = self._auto_align_cross_correlation()
        if shift is None:
            messagebox.showwarning(
                "Auto Align", "Could not estimate a shift (missing input or no overlap).",
                parent=self._inspector_win,
            )
            return
        self._t_shift_trial = shift
        self._insp_shift_var.set(shift)
        self._redraw_plot()

    def _on_refine_clicked(self) -> None:
        shift = self._refine_shift(self._t_shift_trial)
        if shift is None:
            messagebox.showwarning(
                "Refine", "Could not refine the shift (missing input or no overlap).",
                parent=self._inspector_win,
            )
            return
        self._t_shift_trial = shift
        self._insp_shift_var.set(shift)
        self._redraw_plot()

    # -- Plot style controls ----------------------------------------------

    def _build_style_controls(self, parent: tk.Frame) -> None:
        row = tk.Frame(parent)
        row.pack(fill="x", pady=(8, 0))

        tk.Label(row, text="Series:").pack(side="left")
        self._insp_series_var = tk.StringVar(value="series1")
        series_box = ttk.Combobox(
            row, textvariable=self._insp_series_var,
            values=("series1", "series2"), state="readonly", width=8,
        )
        series_box.pack(side="left", padx=(4, 12))
        # Refresh the style widgets below whenever the selected series changes.
        series_box.bind("<<ComboboxSelected>>", lambda _e: self._refresh_style_widgets())

        self._insp_color_btn = tk.Button(row, width=3, command=lambda: self._pick_color("color"))
        self._insp_color_btn.pack(side="left")
        tk.Label(row, text="width:").pack(side="left", padx=(8, 0))
        self._insp_width_var = tk.DoubleVar()
        width_spin = tk.Spinbox(
            row, from_=0.5, to=10.0, increment=0.5, textvariable=self._insp_width_var,
            width=5, command=self._on_style_widget_changed,
        )
        width_spin.pack(side="left")
        width_spin.bind("<Return>", lambda _e: self._on_style_widget_changed())

        tk.Label(row, text="marker:").pack(side="left", padx=(8, 0))
        self._insp_marker_var = tk.StringVar()
        marker_box = ttk.Combobox(
            row, textvariable=self._insp_marker_var, values=self.MARKER_CHOICES,
            state="readonly", width=6,
        )
        marker_box.pack(side="left")
        marker_box.bind("<<ComboboxSelected>>", lambda _e: self._on_style_widget_changed())

        tk.Label(row, text="size:").pack(side="left", padx=(8, 0))
        self._insp_marker_size_var = tk.IntVar()
        size_spin = tk.Spinbox(
            row, from_=1, to=20, textvariable=self._insp_marker_size_var,
            width=4, command=self._on_style_widget_changed,
        )
        size_spin.pack(side="left")
        size_spin.bind("<Return>", lambda _e: self._on_style_widget_changed())

        self._insp_marker_color_btn = tk.Button(
            row, width=3, command=lambda: self._pick_color("marker_color"),
        )
        self._insp_marker_color_btn.pack(side="left", padx=(8, 0))

        self._refresh_style_widgets()

    def _refresh_style_widgets(self) -> None:
        """Load the currently-selected series' style into the widgets."""
        style = self._style[self._insp_series_var.get()]
        self._insp_color_btn.configure(bg=style["color"])
        self._insp_width_var.set(style["line_width"])
        self._insp_marker_var.set(style["marker"])
        self._insp_marker_size_var.set(style["marker_size"])
        self._insp_marker_color_btn.configure(bg=style["marker_color"])

    def _pick_color(self, style_key: str) -> None:
        series = self._insp_series_var.get()
        current = self._style[series][style_key]
        _rgb, hex_color = colorchooser.askcolor(
            color=current, parent=self._inspector_win, title="Choose color",
        )
        if hex_color is None:
            return
        self._style[series][style_key] = hex_color
        self._refresh_style_widgets()
        self._redraw_plot()

    def _on_style_widget_changed(self) -> None:
        series = self._insp_series_var.get()
        style = self._style[series]
        try:
            style["line_width"] = float(self._insp_width_var.get())
            style["marker_size"] = int(self._insp_marker_size_var.get())
        except (tk.TclError, ValueError):
            return  # ignore transient invalid states while typing
        style["marker"] = self._insp_marker_var.get()
        self._redraw_plot()

    # -- Overlay / legend toggles -------------------------------------------

    def _build_overlay_controls(self, parent: tk.Frame) -> None:
        row = tk.Frame(parent)
        row.pack(fill="x", pady=(8, 0))

        self._insp_show_norm_var = tk.BooleanVar(value=self._show_normalized_plot)
        tk.Checkbutton(
            row, text="Show normalized (mean-centered) overlay",
            variable=self._insp_show_norm_var, command=self._on_overlay_setting_changed,
        ).pack(side="left")

        self._insp_legend_var = tk.BooleanVar(value=self._style["legend_on"])
        tk.Checkbutton(
            row, text="Legend", variable=self._insp_legend_var,
            command=self._on_overlay_setting_changed,
        ).pack(side="left", padx=(12, 0))

    def _on_overlay_setting_changed(self) -> None:
        self._show_normalized_plot = bool(self._insp_show_norm_var.get())
        self._style["legend_on"] = bool(self._insp_legend_var.get())
        self._redraw_plot()

    # -- Apply --------------------------------------------------------------

    def _on_apply_clicked(self) -> None:
        if self._t2 is None:
            messagebox.showwarning(
                "Apply", "No time base for series 2 (connect t2, or x2 + dt).",
                parent=self._inspector_win,
            )
            return

        self._t_shift_applied = self._t_shift_trial
        if getattr(self, "_status_item", None) is not None:
            self.canvas.itemconfig(self._status_item, text=self._status_text())

        # Ask the engine to recompute this node's downstream subgraph now
        # that the applied output has changed.
        if self._request_downstream is not None:
            self._request_downstream(self.node_id)