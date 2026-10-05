# node_editor/nodes/fft_1d_node.py
"""
1D Fast Fourier Transform node.

Computes the single-sided amplitude spectrum and phase of a 1D signal.
The time base comes from an optional t array (preferred) or an optional
dt scalar (ignored when t is provided); if neither is given, a unit
sample spacing (dt = 1) is assumed and the frequency axis is reported in
cycles/sample instead of Hz. The inspector shows the time-domain signal
and its frequency-domain spectrum side by side, both pannable and
zoomable, matching the interaction model used in signal_generator_node.py.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from node_editor.base_node import BaseNode
from node_editor.pin_types import PinDef, PinSchema, PinType


class Fft1DNode(BaseNode):
    """Compute the 1D FFT (single-sided magnitude + phase spectrum) of a
    signal, with an optional window function and mean-removal.
    """

    NODE_TYPE = "fft_1d"
    DISPLAY_NAME = "FFT (1D)"
    CATEGORY = "process"
    SEARCH_KEYWORDS = ("fourier", "dft", "spectrum", "frequency")

    NODE_WIDTH = 150
    NODE_HEIGHT = 90

    WINDOW_CHOICES = ("none", "hann", "hamming", "blackman")

    # Relative deviation from a perfectly uniform dt (as a fraction of dt)
    # above which the inspector shows a non-uniform-sampling warning.
    UNIFORMITY_WARNING_TOL = 0.01

    # ══ Construction ════════════════════════════════════════════════

    def __init__(self, node_id: str, canvas: tk.Canvas):
        super().__init__(node_id, canvas)

        # --- Preprocessing settings. These affect the computed outputs
        # directly and immediately (no separate trial/apply state --
        # unlike Time Sync or Signal Generator, there is no interactive
        # search or multi-parameter form here that benefits from a
        # preview-before-commit step). ---
        self._window: str = "none"          # "none" | "hann" | "hamming" | "blackman"
        self._detrend: bool = False         # subtract the mean before transforming
        self._log_magnitude: bool = False   # display-only: log scale on the magnitude axis

        # Cached results from the last compute(), kept so the inspector
        # can redraw without forcing a graph recompute.
        self._t: np.ndarray | None = None
        self._signal: np.ndarray | None = None
        self._dt: float | None = None
        self._freq: np.ndarray | None = None
        self._magnitude: np.ndarray | None = None
        self._phase: np.ndarray | None = None
        self._time_base_source: str = "none"     # "t" | "dt" | "none" -- for the status line
        self._uniformity_warning: str | None = None

        # Matplotlib objects, created lazily in build_inspector(). Two
        # stacked subplots: time-domain signal on top, spectrum below.
        self._fig: Figure | None = None
        self._ax_time = None
        self._ax_freq = None
        self._canvas_widget: FigureCanvasTkAgg | None = None

        # Pan state is tracked per-axis, since each subplot pans/zooms
        # independently. Keyed by the matplotlib Axes object itself.
        self._pan_state: dict = {}

        self._status_item: int | None = None

    # ══ Pin schema ══════════════════════════════════════════════════

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[
                PinDef(name="signal", type=PinType.ARRAY, label="signal"),
                PinDef(name="t", type=PinType.ARRAY, label="t", optional=True),
                PinDef(name="dt", type=PinType.SCALAR, label="dt", optional=True),
            ],
            outputs=[
                PinDef(name="freq", type=PinType.ARRAY, label="freq"),
                PinDef(name="magnitude", type=PinType.ARRAY, label="magnitude"),
                PinDef(name="phase", type=PinType.ARRAY, label="phase"),
            ],
        )

    # ══ Compute ═════════════════════════════════════════════════════

    def _resolve_time_base(self, signal: np.ndarray, t: np.ndarray | None,
                            dt: float | None) -> tuple[np.ndarray, float, str]:
        """Return (t, dt, source_label) for the given signal.

        t takes priority over dt (dt is ignored when t is provided, as
        specified). If neither is available, falls back to a unit sample
        spacing so the node still produces output, with the frequency
        axis later reported in cycles/sample rather than Hz.
        """
        n = len(signal)
        if t is not None:
            t = np.asarray(t, dtype=float)
            dt_eff = float(np.median(np.diff(t))) if n > 1 else 1.0
            return t, dt_eff, "t"
        if dt is not None:
            dt_eff = float(dt)
            return np.arange(n, dtype=float) * dt_eff, dt_eff, "dt"
        return np.arange(n, dtype=float), 1.0, "none"

    def _check_uniform_sampling(self, t: np.ndarray, dt: float) -> str | None:
        """Return a warning string if t deviates from uniform spacing by
        more than UNIFORMITY_WARNING_TOL (relative to dt), else None.
        FFT assumes uniform sampling; a jittery sensor clock can silently
        produce a misleading spectrum otherwise.
        """
        if len(t) < 3 or dt <= 0:
            return None
        diffs = np.diff(t)
        max_dev = float(np.max(np.abs(diffs - dt)))
        rel_dev = max_dev / dt
        if rel_dev > self.UNIFORMITY_WARNING_TOL:
            return (f"warning: t is non-uniform (max deviation {rel_dev:.1%} of dt); "
                     "FFT assumes uniform sampling")
        return None

    def _get_window(self, n: int) -> np.ndarray:
        if self._window == "hann":
            return np.hanning(n)
        if self._window == "hamming":
            return np.hamming(n)
        if self._window == "blackman":
            return np.blackman(n)
        return np.ones(n)  # "none"

    def compute(self, inputs: dict) -> dict:
        signal = inputs.get("signal")
        if signal is None:
            self._t = self._signal = self._freq = self._magnitude = self._phase = None
            self._time_base_source = "none"
            self._uniformity_warning = None
            return {"freq": None, "magnitude": None, "phase": None}

        signal = np.asarray(signal, dtype=float)
        n = len(signal)

        t, dt, source = self._resolve_time_base(signal, inputs.get("t"), inputs.get("dt"))
        self._uniformity_warning = (
            self._check_uniform_sampling(t, dt) if source == "t" else None
        )

        # Preprocessing: optional mean removal, then optional window.
        x = signal - np.mean(signal) if self._detrend else signal.copy()
        x = x * self._get_window(n)

        spectrum = np.fft.rfft(x)
        freq = np.fft.rfftfreq(n, d=dt)

        # Single-sided amplitude spectrum: divide by n, then double every
        # bin except DC (and Nyquist, when n is even and a true Nyquist
        # bin exists) which must not be doubled -- doubling those would
        # overcount energy that has no mirrored negative-frequency twin.
        magnitude = np.abs(spectrum) / n
        if n % 2 == 0:
            magnitude[1:-1] *= 2.0
        else:
            magnitude[1:] *= 2.0
        phase = np.angle(spectrum)

        # Cache for the inspector.
        self._t, self._signal, self._dt = t, signal, dt
        self._freq, self._magnitude, self._phase = freq, magnitude, phase
        self._time_base_source = source

        if getattr(self, "_status_item", None) is not None:
            self.canvas.itemconfig(self._status_item, text=self._status_text())

        return {"freq": freq, "magnitude": magnitude, "phase": phase}

    # ══ Serialization ═══════════════════════════════════════════════

    def get_params(self) -> dict:
        return {
            "window": self._window,
            "detrend": self._detrend,
            "log_magnitude": self._log_magnitude,
        }

    def set_params(self, params: dict) -> None:
        self._window = params.get("window", "none")
        self._detrend = bool(params.get("detrend", False))
        self._log_magnitude = bool(params.get("log_magnitude", False))

    # ══ Canvas body ═════════════════════════════════════════════════

    def _status_text(self) -> str:
        if self._time_base_source == "none":
            return "no time base (dt=1 assumed)"
        if self._dt is None:
            return "no signal"
        return f"dt={self._dt:.4g} (from {self._time_base_source})"

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
        if getattr(self, "_status_item", None) is not None:
            self.canvas.coords(self._status_item, self.x + new_width / 2, self.y + new_height - 12)

    # ══ Inspector UI ════════════════════════════════════════════════

    def build_inspector(self, parent: tk.Frame) -> None:
        plot_frame = tk.Frame(parent)
        plot_frame.pack(fill="both", expand=True)
        self._build_plots(plot_frame)

        self._insp_warning_var = tk.StringVar(value="")
        tk.Label(parent, textvariable=self._insp_warning_var, anchor="w",
                 fg="#aa5500", font=("Arial", 8)).pack(fill="x", pady=(2, 0))

        self._build_settings_controls(parent)
        self._redraw_plots()

    # -- Plot embedding: two stacked, independently pannable/zoomable axes --

    def _build_plots(self, parent: tk.Frame) -> None:
        self._fig = Figure(figsize=(6, 5), dpi=100)
        self._ax_time = self._fig.add_subplot(211)
        self._ax_freq = self._fig.add_subplot(212)
        self._fig.tight_layout(pad=2.0)

        self._canvas_widget = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas_widget.get_tk_widget().pack(fill="both", expand=True)

        for ax in (self._ax_time, self._ax_freq):
            self._pan_state[ax] = {
                "active": False, "x0": None, "y0": None, "xlim0": None, "ylim0": None,
            }

        self._canvas_widget.mpl_connect("scroll_event", self._on_plot_scroll)
        self._canvas_widget.mpl_connect("button_press_event", self._on_plot_press)
        self._canvas_widget.mpl_connect("motion_notify_event", self._on_plot_drag)
        self._canvas_widget.mpl_connect("button_release_event", self._on_plot_release)

    def _on_plot_scroll(self, event) -> None:
        # event.inaxes tells us which of the two subplots the cursor is
        # over; each subplot zooms independently of the other.
        ax = event.inaxes
        if ax is None or event.xdata is None or event.ydata is None:
            return

        factor = 0.9 if event.button == "up" else 1.1
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        key = getattr(event, "key", None)

        # Ctrl+wheel: zoom the x-axis (time or frequency) only.
        # Shift+wheel: zoom the y-axis only.
        # No modifier: zoom both axes together.
        if key in ("control", "ctrl"):
            new_xlim = [event.xdata - (event.xdata - v) * factor for v in xlim]
            ax.set_xlim(new_xlim)
        elif key == "shift":
            new_ylim = [event.ydata - (event.ydata - v) * factor for v in ylim]
            ax.set_ylim(new_ylim)
        else:
            new_xlim = [event.xdata - (event.xdata - v) * factor for v in xlim]
            new_ylim = [event.ydata - (event.ydata - v) * factor for v in ylim]
            ax.set_xlim(new_xlim)
            ax.set_ylim(new_ylim)

        self._canvas_widget.draw_idle()

    def _on_plot_press(self, event) -> None:
        ax = event.inaxes
        if ax is None or event.button != 1 or event.xdata is None:
            return
        self._pan_state[ax].update(
            active=True, x0=event.xdata, y0=event.ydata,
            xlim0=ax.get_xlim(), ylim0=ax.get_ylim(),
        )

    def _on_plot_drag(self, event) -> None:
        # Only the axis where the drag STARTED keeps panning, even if the
        # cursor briefly leaves it mid-drag (event.inaxes would go None).
        for ax, state in self._pan_state.items():
            if not state["active"]:
                continue
            if event.xdata is None or event.inaxes is not ax:
                continue
            dx = event.xdata - state["x0"]
            dy = event.ydata - state["y0"]
            x0, x1 = state["xlim0"]
            y0, y1 = state["ylim0"]
            new_xlim = (x0 - dx, x1 - dx)
            new_ylim = (y0 - dy, y1 - dy)
            ax.set_xlim(new_xlim)
            ax.set_ylim(new_ylim)
            # state["x0"]/state["y0"] are the DATA-SPACE point grabbed at
            # button-press; a data coordinate means the same absolute
            # position no matter what the current view window is, so they
            # must stay fixed for the whole drag -- they are NOT refreshed
            # here, on purpose.
            #
            # state["xlim0"]/state["ylim0"], however, describe the CURRENT
            # view window, which changes every time set_xlim/set_ylim is
            # called above -- so they DO need to be refreshed on every
            # motion event. That refresh was missing: each event's dx/dy
            # was still computed correctly (against the fixed grab point),
            # but was then applied on top of the ORIGINAL button-press
            # window instead of the current one, so the pan under-shot the
            # cursor by a different, growing amount every event -- exactly
            # the "vibrates" + "lags behind the cursor" symptom.
            state["xlim0"] = new_xlim
            state["ylim0"] = new_ylim
            self._canvas_widget.draw_idle()

    def _on_plot_release(self, _event) -> None:
        for state in self._pan_state.values():
            state["active"] = False

    def _redraw_plots(self) -> None:
        ax_t, ax_f = self._ax_time, self._ax_freq

        had_time_data = bool(ax_t.lines)
        had_freq_data = bool(ax_f.lines)
        t_xlim, t_ylim = (ax_t.get_xlim(), ax_t.get_ylim()) if had_time_data else (None, None)
        f_xlim, f_ylim = (ax_f.get_xlim(), ax_f.get_ylim()) if had_freq_data else (None, None)

        ax_t.clear()
        ax_f.clear()

        if self._t is not None and self._signal is not None:
            ax_t.plot(self._t, self._signal, color="#1f77b4", linewidth=1.0)
        ax_t.set_xlabel("time" if self._time_base_source != "none" else "sample index")
        ax_t.set_ylabel("value")
        ax_t.set_title("time domain", fontsize=9)

        if self._freq is not None and self._magnitude is not None:
            ax_f.plot(self._freq, self._magnitude, color="#d62728", linewidth=1.0)
        freq_unit = "Hz" if self._time_base_source != "none" else "cycles/sample"
        ax_f.set_xlabel(f"frequency ({freq_unit})")
        ax_f.set_ylabel("magnitude")
        if self._log_magnitude:
            ax_f.set_yscale("log")
        ax_f.set_title("frequency domain", fontsize=9)

        if had_time_data:
            ax_t.set_xlim(t_xlim)
            ax_t.set_ylim(t_ylim)
        if had_freq_data:
            ax_f.set_xlim(f_xlim)
            ax_f.set_ylim(f_ylim)

        self._fig.tight_layout(pad=2.0)
        self._canvas_widget.draw_idle()

        self._insp_warning_var.set(self._uniformity_warning or "")

    # -- Settings controls (window, detrend, log scale) ----------------------

    def _build_settings_controls(self, parent: tk.Frame) -> None:
        row = tk.Frame(parent)
        row.pack(fill="x", pady=(6, 0))

        tk.Label(row, text="Window:").pack(side="left")
        self._insp_window_var = tk.StringVar(value=self._window)
        window_box = ttk.Combobox(
            row, textvariable=self._insp_window_var, values=self.WINDOW_CHOICES,
            state="readonly", width=10,
        )
        window_box.pack(side="left", padx=(4, 12))
        window_box.bind("<<ComboboxSelected>>", lambda _e: self._on_settings_changed())

        self._insp_detrend_var = tk.BooleanVar(value=self._detrend)
        tk.Checkbutton(
            row, text="Remove mean (detrend)", variable=self._insp_detrend_var,
            command=self._on_settings_changed,
        ).pack(side="left", padx=(0, 12))

        self._insp_log_var = tk.BooleanVar(value=self._log_magnitude)
        tk.Checkbutton(
            row, text="Log magnitude", variable=self._insp_log_var,
            command=self._on_display_setting_changed,
        ).pack(side="left")

    def _on_settings_changed(self) -> None:
        # Window/detrend change the actual computed output, so recompute
        # via the engine immediately -- there is no separate trial/apply
        # step for this node.
        self._window = self._insp_window_var.get()
        self._detrend = bool(self._insp_detrend_var.get())
        if self._request_downstream is not None:
            self._request_downstream(self.node_id)
        # The redraw itself happens once compute() re-populates the
        # cached arrays and the engine calls back into this node; a
        # direct redraw here would still show the pre-recompute data.
        # If your engine calls open inspectors' redraw hooks after
        # compute completes, wire that here instead. Otherwise, this
        # local redraw keeps the preview responsive even before that
        # round-trip completes, using the same cached t/signal:
        self._redraw_plots()

    def _on_display_setting_changed(self) -> None:
        # Log-scale toggle only affects how the cached spectrum is drawn,
        # not the computed values -- no need to touch the engine at all.
        self._log_magnitude = bool(self._insp_log_var.get())
        self._redraw_plots()