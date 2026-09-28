# node_editor/nodes/cwt_1d_node.py
"""
1D Continuous Wavelet Transform (CWT) node.

Computes a scalogram (time-scale/frequency magnitude and phase map) from
a 1D signal using PyWavelets. The time base comes from an optional t
array (preferred) or an optional dt scalar (ignored when t is provided);
if neither is given, dt = 1 is assumed.

Unlike fft_1d_node.py, changing the wavelet family, its sub-parameters,
or the scale range does NOT trigger an immediate recompute -- a CWT's
cost scales with num_scales * n, so edits are staged as "trial" settings
and only committed by pressing "Recompute". A status label shows whether
the currently displayed scalogram matches the staged settings
("Computed") or is stale ("Needs recompute"). Purely cosmetic settings
(view mode, y-axis choice, color scale, colormap) redraw immediately
without needing Recompute, since they only affect how the cached result
is drawn, not what was computed.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np
import pywt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from node_editor.base_node import BaseNode
from node_editor.pin_types import PinDef, PinSchema, PinType


# Continuous wavelet families exposed in the UI. cmor/shan/fbsp need
# extra sub-parameters baked into their pywt name string; the others
# (morl, mexh, gausN) are used as-is.
WAVELET_FAMILY_CHOICES = (
    "morl", "mexh", "cmor", "shan", "fbsp",
    "gaus1", "gaus2", "gaus3", "gaus4",
    "gaus5", "gaus6", "gaus7", "gaus8",
)
FAMILIES_WITH_BANDWIDTH_CENTER = ("cmor", "shan")
FAMILIES_WITH_ORDER = ("fbsp",)

SCALE_SPACING_CHOICES = ("log", "linear")
VIEW_MODE_CHOICES = ("magnitude", "phase")
Y_AXIS_CHOICES = ("frequency", "scale")
COLORMAP_CHOICES = ("viridis", "magma", "plasma", "jet")


def _build_wavelet_name(family: str, params: dict) -> str:
    """Build the pywt wavelet name string, folding in sub-parameters for
    families that need them (cmor, shan, fbsp).
    """
    if family in FAMILIES_WITH_BANDWIDTH_CENTER:
        return f"{family}{params['bandwidth']}-{params['center_freq']}"
    if family in FAMILIES_WITH_ORDER:
        return f"{family}{int(params['order'])}-{params['bandwidth']}-{params['center_freq']}"
    return family  # morl, mexh, gausN take no suffix


def _build_scales(scale_min: float, scale_max: float, num_scales: int, spacing: str) -> np.ndarray:
    num_scales = max(2, int(num_scales))
    if spacing == "log":
        return np.geomspace(scale_min, scale_max, num_scales)
    return np.linspace(scale_min, scale_max, num_scales)


def _default_wavelet_params() -> dict:
    # bandwidth=1.5, center_freq=1.0 are pywt's own commonly-used
    # defaults (as in the canonical "cmor1.5-1.0" example); order=2 is a
    # reasonable default for fbsp.
    return {"bandwidth": 1.5, "center_freq": 1.0, "order": 2}


def _default_scale_settings() -> dict:
    return {"scale_min": 1.0, "scale_max": 64.0, "num_scales": 40, "spacing": "log"}


class Cwt1DNode(BaseNode):
    """Compute a scalogram (CWT magnitude + phase over time and scale) of
    a 1D signal. Wavelet/scale settings are staged and only take effect
    on Recompute; display settings (view, y-axis, color) redraw the
    cached result immediately.
    """

    NODE_TYPE = "cwt_1d"
    DISPLAY_NAME = "CWT (1D)"
    CATEGORY = "process"
    SEARCH_KEYWORDS = ("wavelet", "continuous wavelet transform", "scalogram", "time-frequency")

    NODE_WIDTH = 150
    NODE_HEIGHT = 90

    # ══ Construction ════════════════════════════════════════════════

    def __init__(self, node_id: str, canvas: tk.Canvas):
        super().__init__(node_id, canvas)

        # --- Trial settings: freely edited in the inspector, staged
        # until "Recompute" is pressed. Editing these sets self._dirty. ---
        self._wavelet_family: str = "morl"
        self._wavelet_params: dict = _default_wavelet_params()
        self._scale_settings: dict = _default_scale_settings()

        # --- Applied settings: the ones that actually produced the
        # cached result below and feed compute(). Only Recompute updates
        # these from the trial settings. ---
        self._applied_wavelet_family: str = self._wavelet_family
        self._applied_wavelet_params: dict = dict(self._wavelet_params)
        self._applied_scale_settings: dict = dict(self._scale_settings)

        # Whether the trial settings differ from the applied ones (i.e.
        # the displayed scalogram is stale relative to the staged edits).
        self._dirty: bool = False

        # --- Display-only settings: redraw immediately, never affect
        # compute() or the dirty flag. ---
        self._view_mode: str = "magnitude"     # "magnitude" | "phase"
        self._y_axis_mode: str = "frequency"   # "frequency" | "scale"
        self._log_color: bool = False
        self._y_log_scale: bool = True
        self._colormap: str = "viridis"

        # Cached results from the last compute(), kept so the inspector
        # can redraw without forcing a graph recompute.
        self._t: np.ndarray | None = None
        self._signal: np.ndarray | None = None
        self._dt: float | None = None
        self._scale: np.ndarray | None = None
        self._freq: np.ndarray | None = None
        self._magnitude: np.ndarray | None = None   # shape (num_scales, n)
        self._phase: np.ndarray | None = None        # shape (num_scales, n)
        self._time_base_source: str = "none"          # "t" | "dt" | "none"
        self._compute_error: str | None = None

        # Matplotlib objects, created lazily in build_inspector(). Two
        # stacked subplots: time-domain signal on top, scalogram below.
        self._fig: Figure | None = None
        self._ax_time = None
        self._ax_map = None
        self._colorbar = None   # must be explicitly removed before re-creating
        self._canvas_widget: FigureCanvasTkAgg | None = None

        self._pan_state: dict = {}
        self._status_item: int | None = None

        # Frame that holds the family-specific sub-parameter widgets;
        # rebuilt whenever the selected family changes.
        self._wavelet_param_frame: tk.Frame | None = None

    # ══ Pin schema ══════════════════════════════════════════════════

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[
                PinDef(name="signal", type=PinType.ARRAY, label="signal"),
                PinDef(name="t", type=PinType.ARRAY, label="t", optional=True),
                PinDef(name="dt", type=PinType.SCALAR, label="dt", optional=True),
            ],
            outputs=[
                PinDef(name="t", type=PinType.ARRAY, label="t"),
                PinDef(name="scale", type=PinType.ARRAY, label="scale"),
                PinDef(name="freq", type=PinType.ARRAY, label="freq"),
                PinDef(name="magnitude", type=PinType.ARRAY, label="magnitude"),
                PinDef(name="phase", type=PinType.ARRAY, label="phase"),
            ],
        )

    # ══ Compute ═════════════════════════════════════════════════════

    def _resolve_time_base(self, signal: np.ndarray, t: np.ndarray | None,
                            dt: float | None) -> tuple[np.ndarray, float, str]:
        """Same convention as fft_1d_node.py: t takes priority over dt;
        falls back to dt = 1 if neither is available.
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

    def compute(self, inputs: dict) -> dict:
        signal = inputs.get("signal")
        if signal is None:
            self._reset_cache()
            return self._empty_outputs()

        signal = np.asarray(signal, dtype=float)
        t, dt, source = self._resolve_time_base(signal, inputs.get("t"), inputs.get("dt"))

        scales = _build_scales(**self._applied_scale_settings)
        wavelet_name = _build_wavelet_name(self._applied_wavelet_family, self._applied_wavelet_params)

        try:
            coeffs, freq = pywt.cwt(signal, scales, wavelet_name, sampling_period=dt)
        except Exception as e:  # invalid wavelet name/params, scale out of range, etc.
            self._compute_error = str(e)
            self._reset_cache()
            return self._empty_outputs()

        self._compute_error = None
        magnitude = np.abs(coeffs)
        # Real-valued wavelets (morl, mexh, gausN) produce real coefficients
        # -- phase is meaningless for them, so report zeros rather than a
        # spurious +-pi from sign flips.
        phase = np.angle(coeffs) if np.iscomplexobj(coeffs) else np.zeros_like(coeffs)

        self._t, self._signal, self._dt = t, signal, dt
        self._scale, self._freq = scales, freq
        self._magnitude, self._phase = magnitude, phase
        self._time_base_source = source

        if getattr(self, "_status_item", None) is not None:
            self.canvas.itemconfig(self._status_item, text=self._status_text())

        return {"t": t, "scale": scales, "freq": freq, "magnitude": magnitude, "phase": phase}

    def _empty_outputs(self) -> dict:
        return {"t": None, "scale": None, "freq": None, "magnitude": None, "phase": None}

    def _reset_cache(self) -> None:
        self._t = self._signal = self._dt = None
        self._scale = self._freq = self._magnitude = self._phase = None
        self._time_base_source = "none"
        if getattr(self, "_status_item", None) is not None:
            self.canvas.itemconfig(self._status_item, text=self._status_text())

    # ══ Serialization ═══════════════════════════════════════════════

    def get_params(self) -> dict:
        # Serialize the APPLIED settings -- those are what actually
        # produced the current outputs. Trial edits not yet Recomputed
        # are intentionally NOT persisted, so a reload never silently
        # resumes with unapplied, half-typed settings.
        return {
            "wavelet_family": self._applied_wavelet_family,
            "wavelet_params": self._applied_wavelet_params,
            "scale_settings": self._applied_scale_settings,
            "view_mode": self._view_mode,
            "y_axis_mode": self._y_axis_mode,
            "log_color": self._log_color,
            "y_log_scale": self._y_log_scale,
            "colormap": self._colormap,
        }

    def set_params(self, params: dict) -> None:
        self._applied_wavelet_family = params.get("wavelet_family", "morl")
        saved_params = params.get("wavelet_params")
        if isinstance(saved_params, dict):
            self._applied_wavelet_params = {**_default_wavelet_params(), **saved_params}
        saved_scale = params.get("scale_settings")
        if isinstance(saved_scale, dict):
            self._applied_scale_settings = {**_default_scale_settings(), **saved_scale}

        self._view_mode = params.get("view_mode", "magnitude")
        self._y_axis_mode = params.get("y_axis_mode", "frequency")
        self._log_color = bool(params.get("log_color", False))
        self._y_log_scale = bool(params.get("y_log_scale", True))
        self._colormap = params.get("colormap", "viridis")

        # Trial settings start out matching the applied ones on load, so
        # the inspector opens in a clean, non-dirty state.
        self._wavelet_family = self._applied_wavelet_family
        self._wavelet_params = dict(self._applied_wavelet_params)
        self._scale_settings = dict(self._applied_scale_settings)
        self._dirty = False

    # ══ Canvas body ═════════════════════════════════════════════════

    def _status_text(self) -> str:
        if self._compute_error:
            return f"error: {self._compute_error[:40]}"
        if self._scale is None:
            return "no result yet"
        wavelet_name = _build_wavelet_name(self._applied_wavelet_family, self._applied_wavelet_params)
        return f"{wavelet_name}, {len(self._scale)} scales"

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

        self._build_wavelet_controls(parent)
        self._build_scale_controls(parent)

        # -- Recompute + dirty-state status label --
        action_row = tk.Frame(parent)
        action_row.pack(fill="x", pady=(6, 0))
        tk.Button(action_row, text="Recompute", width=12,
                  command=self._on_recompute_clicked).pack(side="left")
        self._insp_status_var = tk.StringVar()
        self._insp_status_label = tk.Label(
            action_row, textvariable=self._insp_status_var, font=("Arial", 9, "bold"),
        )
        self._insp_status_label.pack(side="left", padx=(10, 0))

        self._build_display_controls(parent)

        self._insp_error_var = tk.StringVar(value="")
        tk.Label(parent, textvariable=self._insp_error_var, anchor="w",
                 fg="#aa0000", font=("Arial", 8)).pack(fill="x", pady=(2, 0))

        self._update_dirty_label()
        self._redraw_plots()

    # -- Plot embedding: time-domain line + scalogram heatmap ---------------

    def _build_plots(self, parent: tk.Frame) -> None:
        self._fig = Figure(figsize=(6, 5.5), dpi=100)
        self._ax_time = self._fig.add_subplot(211)
        self._ax_map = self._fig.add_subplot(212)
        self._fig.tight_layout(pad=2.0)

        self._canvas_widget = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas_widget.get_tk_widget().pack(fill="both", expand=True)

        for ax in (self._ax_time, self._ax_map):
            self._pan_state[ax] = {
                "active": False, "x0": None, "y0": None, "xlim0": None, "ylim0": None,
            }

        self._canvas_widget.mpl_connect("scroll_event", self._on_plot_scroll)
        self._canvas_widget.mpl_connect("button_press_event", self._on_plot_press)
        self._canvas_widget.mpl_connect("motion_notify_event", self._on_plot_drag)
        self._canvas_widget.mpl_connect("button_release_event", self._on_plot_release)

    def _on_plot_scroll(self, event) -> None:
        # Same convention as fft_1d/ifft_1d: no modifier zooms both axes,
        # Ctrl zooms x only, Shift zooms y only. Works identically on the
        # scalogram's log-scale y-axis, since matplotlib's set_ylim
        # accepts values in data space regardless of the axis scale.
        ax = event.inaxes
        if ax is None or event.xdata is None or event.ydata is None:
            return

        factor = 0.9 if event.button == "up" else 1.1
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        key = getattr(event, "key", None)

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
        for ax, state in self._pan_state.items():
            if not state["active"]:
                continue
            if event.xdata is None or event.inaxes is not ax:
                continue
            dx = event.xdata - state["x0"]
            dy = event.ydata - state["y0"]
            x0, x1 = state["xlim0"]
            y0, y1 = state["ylim0"]
            ax.set_xlim(x0 - dx, x1 - dx)
            ax.set_ylim(y0 - dy, y1 - dy)
            self._canvas_widget.draw_idle()

    def _on_plot_release(self, _event) -> None:
        for state in self._pan_state.values():
            state["active"] = False

    def _redraw_plots(self) -> None:
        ax_t, ax_m = self._ax_time, self._ax_map

        had_time_data = bool(ax_t.lines)
        had_map_data = self._colorbar is not None
        t_xlim, t_ylim = (ax_t.get_xlim(), ax_t.get_ylim()) if had_time_data else (None, None)
        m_xlim, m_ylim = (ax_m.get_xlim(), ax_m.get_ylim()) if had_map_data else (None, None)

        ax_t.clear()
        ax_m.clear()
        # pcolormesh's colorbar is a separate Axes attached to the figure;
        # ax_m.clear() does NOT remove it, so a stale one would silently
        # accumulate on every redraw if not explicitly removed first.
        if self._colorbar is not None:
            self._colorbar.remove()
            self._colorbar = None

        if self._t is not None and self._signal is not None:
            ax_t.plot(self._t, self._signal, color="#1f77b4", linewidth=1.0)
        ax_t.set_xlabel("time" if self._time_base_source != "none" else "sample index")
        ax_t.set_ylabel("value")
        ax_t.set_title("time domain", fontsize=9)

        if self._t is not None and self._scale is not None:
            y_values = self._freq if self._y_axis_mode == "frequency" else self._scale
            data = self._magnitude if self._view_mode == "magnitude" else self._phase

            if self._view_mode == "magnitude" and self._log_color:
                # +eps avoids log(0) for exactly-zero coefficients.
                plotted = np.log10(data + 1e-12)
                colorbar_label = "log10(magnitude)"
            else:
                plotted = data
                colorbar_label = "magnitude" if self._view_mode == "magnitude" else "phase (rad)"

            mesh = ax_m.pcolormesh(self._t, y_values, plotted, shading="auto", cmap=self._colormap)
            self._colorbar = self._fig.colorbar(mesh, ax=ax_m, label=colorbar_label)

            if self._y_log_scale:
                ax_m.set_yscale("log")
            ax_m.set_xlabel("time" if self._time_base_source != "none" else "sample index")
            ax_m.set_ylabel("frequency (Hz)" if self._y_axis_mode == "frequency" else "scale")
        ax_m.set_title("scalogram", fontsize=9)

        if had_time_data:
            ax_t.set_xlim(t_xlim)
            ax_t.set_ylim(t_ylim)
        if had_map_data and self._colorbar is not None:
            ax_m.set_xlim(m_xlim)
            ax_m.set_ylim(m_ylim)

        self._fig.tight_layout(pad=2.0)
        self._canvas_widget.draw_idle()

        self._insp_error_var.set(f"Compute error: {self._compute_error}" if self._compute_error else "")

    # -- Wavelet family + sub-parameters (compute-affecting -> dirty) -------

    def _build_wavelet_controls(self, parent: tk.Frame) -> None:
        row = tk.Frame(parent)
        row.pack(fill="x", pady=(6, 0))

        tk.Label(row, text="Wavelet:").pack(side="left")
        self._insp_family_var = tk.StringVar(value=self._wavelet_family)
        family_box = ttk.Combobox(
            row, textvariable=self._insp_family_var, values=WAVELET_FAMILY_CHOICES,
            state="readonly", width=8,
        )
        family_box.pack(side="left", padx=(4, 12))
        family_box.bind("<<ComboboxSelected>>", lambda _e: self._on_family_changed())

        self._wavelet_param_frame = tk.Frame(row)
        self._wavelet_param_frame.pack(side="left")
        self._refresh_wavelet_param_widgets()

    def _refresh_wavelet_param_widgets(self) -> None:
        """Rebuild the sub-parameter entry widgets for the currently
        selected family. Only cmor/shan/fbsp need extra fields; morl,
        mexh and gausN take none.
        """
        for child in self._wavelet_param_frame.winfo_children():
            child.destroy()

        family = self._insp_family_var.get()
        frame = self._wavelet_param_frame

        if family in FAMILIES_WITH_ORDER:
            tk.Label(frame, text="order:").pack(side="left")
            self._insp_order_var = tk.IntVar(value=self._wavelet_params["order"])
            entry = tk.Entry(frame, textvariable=self._insp_order_var, width=4)
            entry.pack(side="left", padx=(2, 8))
            entry.bind("<KeyRelease>", lambda _e: self._on_wavelet_param_changed())

        if family in FAMILIES_WITH_BANDWIDTH_CENTER or family in FAMILIES_WITH_ORDER:
            tk.Label(frame, text="bandwidth:").pack(side="left")
            self._insp_bandwidth_var = tk.DoubleVar(value=self._wavelet_params["bandwidth"])
            bw_entry = tk.Entry(frame, textvariable=self._insp_bandwidth_var, width=6)
            bw_entry.pack(side="left", padx=(2, 8))
            bw_entry.bind("<KeyRelease>", lambda _e: self._on_wavelet_param_changed())

            tk.Label(frame, text="center freq:").pack(side="left")
            self._insp_center_var = tk.DoubleVar(value=self._wavelet_params["center_freq"])
            cf_entry = tk.Entry(frame, textvariable=self._insp_center_var, width=6)
            cf_entry.pack(side="left")
            cf_entry.bind("<KeyRelease>", lambda _e: self._on_wavelet_param_changed())

    def _on_family_changed(self) -> None:
        self._wavelet_family = self._insp_family_var.get()
        self._refresh_wavelet_param_widgets()
        self._mark_dirty()

    def _on_wavelet_param_changed(self) -> None:
        try:
            if hasattr(self, "_insp_order_var"):
                self._wavelet_params["order"] = int(self._insp_order_var.get())
            if hasattr(self, "_insp_bandwidth_var"):
                self._wavelet_params["bandwidth"] = float(self._insp_bandwidth_var.get())
            if hasattr(self, "_insp_center_var"):
                self._wavelet_params["center_freq"] = float(self._insp_center_var.get())
        except (tk.TclError, ValueError):
            return  # ignore transient invalid states while typing
        self._mark_dirty()

    # -- Scale range (compute-affecting -> dirty) ----------------------------

    def _build_scale_controls(self, parent: tk.Frame) -> None:
        row = tk.Frame(parent)
        row.pack(fill="x", pady=(6, 0))

        tk.Label(row, text="Scale:").pack(side="left")
        tk.Label(row, text="min").pack(side="left", padx=(6, 0))
        self._insp_scale_min_var = tk.DoubleVar(value=self._scale_settings["scale_min"])
        min_entry = tk.Entry(row, textvariable=self._insp_scale_min_var, width=6)
        min_entry.pack(side="left", padx=(2, 8))

        tk.Label(row, text="max").pack(side="left")
        self._insp_scale_max_var = tk.DoubleVar(value=self._scale_settings["scale_max"])
        max_entry = tk.Entry(row, textvariable=self._insp_scale_max_var, width=6)
        max_entry.pack(side="left", padx=(2, 8))

        tk.Label(row, text="count").pack(side="left")
        self._insp_num_scales_var = tk.IntVar(value=self._scale_settings["num_scales"])
        count_entry = tk.Entry(row, textvariable=self._insp_num_scales_var, width=5)
        count_entry.pack(side="left", padx=(2, 8))

        tk.Label(row, text="spacing").pack(side="left")
        self._insp_spacing_var = tk.StringVar(value=self._scale_settings["spacing"])
        spacing_box = ttk.Combobox(
            row, textvariable=self._insp_spacing_var, values=SCALE_SPACING_CHOICES,
            state="readonly", width=6,
        )
        spacing_box.pack(side="left", padx=(2, 0))

        for entry in (min_entry, max_entry, count_entry):
            entry.bind("<KeyRelease>", lambda _e: self._on_scale_settings_changed())
        spacing_box.bind("<<ComboboxSelected>>", lambda _e: self._on_scale_settings_changed())

    def _on_scale_settings_changed(self) -> None:
        try:
            scale_min = float(self._insp_scale_min_var.get())
            scale_max = float(self._insp_scale_max_var.get())
            num_scales = int(self._insp_num_scales_var.get())
        except (tk.TclError, ValueError):
            return
        if scale_min <= 0 or scale_max <= scale_min or num_scales < 2:
            return
        self._scale_settings = {
            "scale_min": scale_min, "scale_max": scale_max,
            "num_scales": num_scales, "spacing": self._insp_spacing_var.get(),
        }
        self._mark_dirty()

    # -- Dirty-state tracking -------------------------------------------------

    def _mark_dirty(self) -> None:
        self._dirty = True
        self._update_dirty_label()

    def _update_dirty_label(self) -> None:
        if self._dirty:
            self._insp_status_var.set("Needs recompute")
            self._insp_status_label.configure(fg="#b36b00")   # amber: pending
        else:
            self._insp_status_var.set("Computed")
            self._insp_status_label.configure(fg="#1a7a1a")   # green: up to date

    def _on_recompute_clicked(self) -> None:
        self._applied_wavelet_family = self._wavelet_family
        self._applied_wavelet_params = dict(self._wavelet_params)
        self._applied_scale_settings = dict(self._scale_settings)
        self._dirty = False
        self._update_dirty_label()

        if getattr(self, "_status_item", None) is not None:
            self.canvas.itemconfig(self._status_item, text=self._status_text())

        if self._request_downstream is not None:
            self._request_downstream(self.node_id)
        # No local recompute-and-redraw fallback here (unlike fft_1d's
        # settings handler): the CWT itself only ever runs inside
        # compute(), driven by the engine, since it needs the current
        # signal input which the inspector does not independently hold.
        # The plot refreshes once compute() re-populates the cache and
        # the engine's completion callback (or the next redraw trigger)
        # reaches this node.

    # -- Display-only controls (redraw immediately, never dirty) ------------

    def _build_display_controls(self, parent: tk.Frame) -> None:
        row = tk.Frame(parent)
        row.pack(fill="x", pady=(6, 0))

        tk.Label(row, text="View:").pack(side="left")
        self._insp_view_var = tk.StringVar(value=self._view_mode)
        view_box = ttk.Combobox(
            row, textvariable=self._insp_view_var, values=VIEW_MODE_CHOICES,
            state="readonly", width=9,
        )
        view_box.pack(side="left", padx=(4, 12))
        view_box.bind("<<ComboboxSelected>>", lambda _e: self._on_display_setting_changed())

        tk.Label(row, text="Y axis:").pack(side="left")
        self._insp_yaxis_var = tk.StringVar(value=self._y_axis_mode)
        yaxis_box = ttk.Combobox(
            row, textvariable=self._insp_yaxis_var, values=Y_AXIS_CHOICES,
            state="readonly", width=9,
        )
        yaxis_box.pack(side="left", padx=(4, 12))
        yaxis_box.bind("<<ComboboxSelected>>", lambda _e: self._on_display_setting_changed())

        tk.Label(row, text="Colormap:").pack(side="left")
        self._insp_cmap_var = tk.StringVar(value=self._colormap)
        cmap_box = ttk.Combobox(
            row, textvariable=self._insp_cmap_var, values=COLORMAP_CHOICES,
            state="readonly", width=8,
        )
        cmap_box.pack(side="left", padx=(4, 12))
        cmap_box.bind("<<ComboboxSelected>>", lambda _e: self._on_display_setting_changed())

        self._insp_logcolor_var = tk.BooleanVar(value=self._log_color)
        tk.Checkbutton(
            row, text="Log color", variable=self._insp_logcolor_var,
            command=self._on_display_setting_changed,
        ).pack(side="left", padx=(0, 12))

        self._insp_ylog_var = tk.BooleanVar(value=self._y_log_scale)
        tk.Checkbutton(
            row, text="Log Y axis", variable=self._insp_ylog_var,
            command=self._on_display_setting_changed,
        ).pack(side="left")

    def _on_display_setting_changed(self) -> None:
        self._view_mode = self._insp_view_var.get()
        self._y_axis_mode = self._insp_yaxis_var.get()
        self._colormap = self._insp_cmap_var.get()
        self._log_color = bool(self._insp_logcolor_var.get())
        self._y_log_scale = bool(self._insp_ylog_var.get())
        self._redraw_plots()