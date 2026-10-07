# node_editor/nodes/bandpass_filter_node.py
"""
General-purpose frequency-domain filter node (low-pass / high-pass /
band-pass / band-stop).

Takes a signal (1-D, or 2-D as ns x nt for multiple signals sharing one
time axis) plus an optional time vector / sample interval, and outputs the
filtered signal on the same time axis. The inspector shows an immediately
updating preview (time-domain and frequency-domain plots, original vs.
filtered); only pressing "Apply" writes the filtered signal to the output
pins.
"""

from __future__ import annotations

import copy
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from node_editor.base_node import BaseNode
from node_editor.pin_types import PinDef, PinSchema, PinType


_HELP_TEXT = """
Band-Pass Filter Node
======================
A general frequency-domain filter: choose Low-pass, High-pass, Band-pass,
or Band-stop, with adjustable cutoff frequency(ies), order (steepness),
and response shape.

INPUTS
------
dt       (optional) scalar sample interval, used only when t is NOT
         connected. Ignored if t is connected.
t        (optional) 1-D timestamp array, length nt. If given, dt is
         estimated as the median sample spacing (assumes roughly uniform
         sampling -- this is an FFT-based filter, which is only exact for
         uniformly sampled data).
signal   1-D (length nt) or 2-D (ns x nt, ns independent signals sharing
         one time axis) array to filter. Required.

OUTPUTS
-------
dt       Sample interval actually used.
t        Time vector actually used (passed through, or generated from dt
         if t was not connected).
signal   The filtered signal, same shape as the input signal.

HOW FILTERING WORKS
--------------------
This is a zero-phase FFT-domain filter: the signal's spectrum is
multiplied by a magnitude response H(f) and inverse-transformed back --
there is no phase distortion, and no recursive (IIR) filter state, so it
avoids the classic pitfalls of hand-rolled IIR coefficient design. To
reduce edge/wrap-around artifacts from the implicit circular convolution,
the signal is reflect-padded before filtering (see "Edge padding" below)
and cropped back afterward.

Response "Shape" options:
  Butterworth  -- maximally flat passband, standard smooth roll-off
                  (most common general-purpose choice).
  Gaussian     -- smooth, no ripple, faster roll-off in the middle.
  Ideal        -- brick-wall cutoff (sharpest possible, but can ring /
                  introduce artifacts, especially on short signals).
Note: Chebyshev / Elliptic designs (which need extra passband/stopband
ripple parameters) are intentionally not offered here, to keep this
filter dependency-free (no scipy) and easy to reason about.

"Remove mean (DC) before filtering" subtracts each signal's own mean
before filtering and adds it back afterward for Low-pass / Band-stop
(which are expected to preserve the baseline level); it is not added
back for High-pass / Band-pass (which are expected to remove it).

"Edge padding" reflect-pads each signal by this fraction of its length
at both ends before filtering, then crops back to the original length.
Larger values reduce edge artifacts on short or non-periodic signals, at
some extra computation cost.

MULTIPLE SIGNALS (2-D input)
-----------------------------
If `signal` is 2-D (ns x nt), all ns signals are filtered independently
(each against the shared t / dt), and the "Signal index" control in the
inspector selects which one is shown in the preview plots. The output
`signal` keeps the same 2-D shape.

PLOTS
-----
Time-series plot: original vs. filtered signal over time.
Frequency plot: one-sided amplitude spectrum of original vs. filtered
signal, plus the filter's magnitude response H(f) (right-hand axis,
0 to 1) as a thin dashed reference curve.

Mouse (both plots): drag to pan; wheel to zoom (both axes); Ctrl+wheel
to zoom the X axis only; Shift+wheel to zoom the Y axis only;
double-click to reset the view.

Apply commits the current trial parameters to the node's output pins
(and pushes the update downstream). Editing parameters or dragging the
plots never changes the output by itself.
"""

_MODE_OPTIONS = [
    ("lowpass", "Low-pass"),
    ("highpass", "High-pass"),
    ("bandpass", "Band-pass"),
    ("bandstop", "Band-stop"),
]
_SHAPE_OPTIONS = [
    ("butterworth", "Butterworth"),
    ("gaussian", "Gaussian"),
    ("ideal", "Ideal (brick-wall)"),
]


def _default_params() -> dict:
    return {
        "mode": "bandpass",
        "f_low": 0.5,
        "f_high": 5.0,
        "order": 4.0,
        "shape": "butterworth",
        "remove_mean": True,
        "pad_frac": 0.1,
    }


# ---------------------------------------------------------------------------
# Pure numpy math (module-level, independent of Tk; no scipy dependency,
# matching the rest of this project).
# ---------------------------------------------------------------------------

def _one_sided_spectrum(x: np.ndarray, dt: float):
    """One-sided amplitude spectrum. x: (..., nt). Returns (freqs, amp)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[-1]
    freqs = np.fft.rfftfreq(n, d=dt)
    spectrum = np.fft.rfft(x, axis=-1)
    amp = np.abs(spectrum) * (2.0 / n)
    amp[..., 0] /= 2.0            # DC bin must not be doubled
    if n % 2 == 0:
        amp[..., -1] /= 2.0       # Nyquist bin must not be doubled (even n)
    return freqs, amp


def _design_magnitude_response(freqs: np.ndarray, mode: str, f_low: float,
                                f_high: float, order: float, shape: str) -> np.ndarray:
    """
    Closed-form magnitude response H(f) in [0, 1], applied directly to the
    FFT spectrum (see _run_filter). f_low / f_high <= 0 are treated as
    "no cutoff on this side" (an all-pass on that side), so e.g. band-pass
    with f_low=0 degenerates gracefully to a plain low-pass.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    eps = 1e-12
    order = max(float(order), 1e-6)

    def lowpass(fc: float) -> np.ndarray:
        if fc <= 0:
            return np.zeros_like(freqs)
        if shape == "ideal":
            return (freqs <= fc).astype(np.float64)
        if shape == "gaussian":
            return np.exp(-0.5 * (freqs / max(fc, eps)) ** (2 * order))
        return 1.0 / np.sqrt(1.0 + (freqs / max(fc, eps)) ** (2 * order))  # butterworth

    def highpass(fc: float) -> np.ndarray:
        if fc <= 0:
            return np.ones_like(freqs)
        if shape == "ideal":
            return (freqs >= fc).astype(np.float64)
        if shape == "gaussian":
            return 1.0 - np.exp(-0.5 * (freqs / max(fc, eps)) ** (2 * order))
        ratio = np.divide(fc, freqs, out=np.full_like(freqs, np.inf), where=freqs > eps)
        return 1.0 / np.sqrt(1.0 + ratio ** (2 * order))  # butterworth

    # A steep order combined with a frequency far past the cutoff can
    # transiently overflow the intermediate power term (harmlessly -- it
    # just yields 0.0 / inf, which is the mathematically correct limit);
    # suppress the resulting numpy warning noise rather than the value.
    with np.errstate(over="ignore", invalid="ignore"):
        if mode == "lowpass":
            return lowpass(f_high)
        if mode == "highpass":
            return highpass(f_low)
        if mode == "bandpass":
            return lowpass(f_high) * highpass(f_low)
        if mode == "bandstop":
            return 1.0 - lowpass(f_high) * highpass(f_low)
        return np.ones_like(freqs)


def _run_filter(signal: np.ndarray, dt: float, params: dict):
    """
    Apply the configured zero-phase FFT-domain filter to `signal`
    (1-D (nt,) or 2-D (ns, nt)). Returns (filtered, freqs, H) where
    filtered has the SAME shape/ndim as the input, freqs/H describe the
    magnitude response actually used (for plotting).
    """
    x0 = np.asarray(signal, dtype=np.float64)
    was_1d = (x0.ndim == 1)
    x = np.atleast_2d(x0)
    nt = x.shape[-1]

    means = x.mean(axis=-1, keepdims=True)
    remove_mean = bool(params.get("remove_mean", True))
    xw = (x - means) if remove_mean else x

    pad_frac = float(params.get("pad_frac", 0.0))
    pad = int(round(pad_frac * nt)) if pad_frac > 0 else 0
    pad = min(pad, max(nt - 1, 0))
    xp = np.pad(xw, ((0, 0), (pad, pad)), mode="reflect") if pad > 0 else xw

    n_fft = xp.shape[-1]
    freqs = np.fft.rfftfreq(n_fft, d=dt)
    H = _design_magnitude_response(freqs, params["mode"], params["f_low"],
                                    params["f_high"], params["order"], params["shape"])

    spectrum = np.fft.rfft(xp, axis=-1)
    filtered_spectrum = spectrum * H[np.newaxis, :]
    yp = np.fft.irfft(filtered_spectrum, n=n_fft, axis=-1)
    y = yp[:, pad:pad + nt] if pad > 0 else yp

    if remove_mean and params["mode"] in ("lowpass", "bandstop"):
        y = y + means

    return (y[0] if was_1d else y), freqs, H


RESPONSE_YLIM = (-0.05, 1.05)   # default y-range of the H(f) response axis (frequency plot)


class BandpassFilterNode(BaseNode):
    """General low/high/band-pass/band-stop FFT-domain filter. Outputs
    (dt, t, signal) update only on Apply."""

    NODE_TYPE = "bandpass_filter"
    DISPLAY_NAME = "Band-Pass Filter"
    CATEGORY = "process"
    SEARCH_KEYWORDS = ("filter", "bandpass", "band-pass", "lowpass", "low-pass",
                        "highpass", "high-pass", "bandstop", "band-stop",
                        "frequency", "fft", "butterworth", "spectrum")

    NODE_WIDTH = 160
    NODE_HEIGHT = 90

    HELP_TEXT = _HELP_TEXT

    # ══ Construction ════════════════════════════════════════════════

    def __init__(self, node_id: str, canvas: tk.Canvas):
        super().__init__(node_id, canvas)

        # --- Trial state: freely editable in the inspector, previewed
        # immediately, but does NOT affect the output pins until Apply ---
        self._params: dict = _default_params()

        # --- Applied state: the ONLY thing that affects the output pins.
        self._applied_params: dict = copy.deepcopy(self._params)

        # Last input actually seen from upstream (cached so "Apply" alone,
        # with no new upstream data, can still re-filter and push a fresh
        # output using the currently-applied parameters).
        self._last_signal: np.ndarray | None = None
        self._last_t: np.ndarray | None = None
        self._last_dt: float | None = None
        self._has_input: bool = False

        # Display-only choices (not gated by Apply -- they only affect
        # what the preview plots show, never the filter's own output).
        self._signal_index: int = 0
        self._show_db: bool = False

        # Matplotlib objects, created lazily in build_inspector().
        self._fig_t: Figure | None = None
        self._ax_t = None
        self._canvas_t: FigureCanvasTkAgg | None = None
        self._pan_state_t: dict = {"active": False, "x0": None, "y0": None, "xlim0": None, "ylim0": None}
        self._extent_t = None

        self._fig_f: Figure | None = None
        self._ax_f = None
        self._ax_f2 = None   # twin y-axis for the H(f) response curve
        self._canvas_f: FigureCanvasTkAgg | None = None
        self._pan_state_f: dict = {"active": False, "x0": None, "y0": None, "xlim0": None, "ylim0": None}
        self._extent_f = None

        # Per-field Tk variable bundles, populated in build_inspector().
        self._widget_vars: dict = {}

        self._status_item: int | None = None

    # ══ Pin schema ══════════════════════════════════════════════════

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            # Pin order: sampling step, time axis, then the data (same as the outputs).
            inputs=[
                PinDef(name="dt", type=PinType.SCALAR, label="dt", optional=True),
                PinDef(name="t", type=PinType.ARRAY, label="t", optional=True),
                PinDef(name="signal", type=PinType.ARRAY, label="signal", optional=False),
            ],
            outputs=[
                PinDef(name="dt", type=PinType.SCALAR, label="dt"),
                PinDef(name="t", type=PinType.ARRAY, label="t"),
                PinDef(name="signal", type=PinType.ARRAY, label="signal"),
            ],
        )

    # ══ Compute ═════════════════════════════════════════════════════

    def compute(self, inputs: dict) -> dict:
        signal_in = inputs.get("signal")
        if signal_in is None:
            self._has_input = False
            self._set_canvas_status("waiting for signal input")
            return {}

        signal = np.asarray(signal_in, dtype=np.float64)
        if signal.ndim not in (1, 2):
            self._has_input = False
            self._set_canvas_status("signal must be 1-D or 2-D (ns x nt)")
            return {}

        nt = signal.shape[-1]
        t_in = inputs.get("t")
        dt_in = inputs.get("dt")

        if t_in is not None:
            t = np.asarray(t_in, dtype=np.float64).ravel()
            if t.size != nt:
                self._has_input = False
                self._set_canvas_status("t length does not match signal's last axis")
                return {}
            dt = float(np.median(np.diff(t))) if t.size > 1 else 1.0
            if not np.isfinite(dt) or dt <= 0:
                dt = 1.0
        else:
            dt = float(dt_in) if (dt_in is not None and np.isfinite(float(dt_in)) and float(dt_in) > 0) else 1.0
            t = np.arange(nt, dtype=np.float64) * dt

        self._last_signal = signal
        self._last_t = t
        self._last_dt = dt
        self._has_input = True

        n_sig = 1 if signal.ndim == 1 else signal.shape[0]
        if self._signal_index >= n_sig:
            self._signal_index = 0

        filtered, _freqs, _H = _run_filter(signal, dt, self._applied_params)

        self._set_canvas_status(self._status_text())
        if self.is_inspector_open():
            self._redraw_plots()

        return {"dt": dt, "t": t, "signal": filtered}

    # ══ Serialization ═══════════════════════════════════════════════

    def get_params(self) -> dict:
        # Serialize the APPLIED filter state (what actually produced the
        # current output) plus the display-only preview choices.
        return {
            "filter": self._applied_params,
            "signal_index": self._signal_index,
            "show_db": self._show_db,
        }

    def set_params(self, params: dict) -> None:
        filt = params.get("filter")
        if isinstance(filt, dict):
            self._applied_params.update(filt)
        self._signal_index = int(params.get("signal_index", 0))
        self._show_db = bool(params.get("show_db", False))

        # On load, the inspector should show the settings that are
        # actually in effect, not stale in-progress edits.
        self._params = copy.deepcopy(self._applied_params)

        self._set_canvas_status(self._status_text())

    # ══ Canvas body ═════════════════════════════════════════════════

    def _status_text(self) -> str:
        p = self._applied_params
        mode_label = dict(_MODE_OPTIONS).get(p["mode"], p["mode"])
        if p["mode"] == "lowpass":
            band = f"< {p['f_high']:g} Hz"
        elif p["mode"] == "highpass":
            band = f"> {p['f_low']:g} Hz"
        else:
            band = f"{p['f_low']:g}-{p['f_high']:g} Hz"
        if not self._has_input:
            return f"{mode_label}  {band}  (no input)"
        return f"{mode_label}  {band}"

    def _set_canvas_status(self, text: str) -> None:
        if self._status_item is not None:
            self.canvas.itemconfig(self._status_item, text=text)

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
        if self._status_item is not None:
            self.canvas.coords(self._status_item, self.x + new_width / 2, self.y + new_height - 12)

    # ══ Inspector UI ════════════════════════════════════════════════

    def build_inspector(self, parent: tk.Frame) -> None:
        panes, left, right = self._make_paned(parent)
        left_inner = self._make_scrollable(left)

        self._build_input_info(left_inner)
        self._build_filter_controls(left_inner)
        self._build_display_controls(left_inner)

        apply_row = tk.Frame(left_inner)
        apply_row.pack(fill="x", pady=(10, 4), padx=6)
        tk.Button(apply_row, text="Apply", font=("Arial", 9, "bold"), width=12,
                  command=self._on_apply_clicked).pack(side="left")

        win = self._inspector_win
        if win is not None:
            for seq in ("<Control-h>", "<Control-H>"):
                win.bind(seq, lambda e: self._open_help())

        # The two plots share the right side in a vertical PanedWindow, so the
        # separator between them can be dragged to give either plot more room.
        plots = tk.PanedWindow(right, orient=tk.VERTICAL, sashwidth=6,
                               sashrelief=tk.RAISED, showhandle=True)
        plots.pack(fill="both", expand=True)

        t_frame = tk.LabelFrame(plots, text="Time series  (drag=pan, wheel=zoom, "
                                             "ctrl=zoom X, shift=zoom Y, dbl-click=reset)",
                                 font=("Arial", 9), padx=4, pady=4)
        plots.add(t_frame, minsize=120, stretch="always")
        self._build_time_plot(t_frame)

        f_frame = tk.LabelFrame(plots, text="Frequency spectrum  (same controls)",
                                 font=("Arial", 9), padx=4, pady=4)
        plots.add(f_frame, minsize=120, stretch="always")
        self._build_freq_plot(f_frame)

        # Start with an even split once the pane has its real height.
        def _split_evenly(_event=None):
            h = plots.winfo_height()
            if h > 1:
                plots.sash_place(0, 0, h // 2)
                plots.unbind("<Map>")
        plots.bind("<Map>", lambda e: plots.after_idle(_split_evenly))

        self._redraw_plots()

    # -- Read-only input summary --------------------------------------------

    def _build_input_info(self, parent: tk.Frame) -> None:
        frame = tk.LabelFrame(parent, text="Input", font=("Arial", 9), padx=6, pady=4)
        frame.pack(fill="x", padx=6, pady=(6, 4))
        self._input_info_var = tk.StringVar(value=self._input_info_text())
        tk.Label(frame, textvariable=self._input_info_var, font=("Arial", 8),
                 fg="#333333", justify="left", wraplength=250).pack(anchor="w")

    def _input_info_text(self) -> str:
        if not self._has_input or self._last_signal is None:
            return "Waiting for signal input."
        sig = self._last_signal
        n_sig = 1 if sig.ndim == 1 else sig.shape[0]
        nt = sig.shape[-1]
        dt = self._last_dt or 1.0
        fs = 1.0 / dt if dt > 0 else float("nan")
        return (f"n_signals = {n_sig}, nt = {nt}\n"
                f"dt = {dt:.6g} s  (fs = {fs:.6g} Hz, Nyquist = {fs / 2:.6g} Hz)")

    # -- Filter parameter controls -------------------------------------------

    def _build_filter_controls(self, parent: tk.Frame) -> None:
        frame = tk.LabelFrame(parent, text="Filter Parameters", font=("Arial", 9), padx=6, pady=4)
        frame.pack(fill="x", padx=6, pady=4)
        v = self._widget_vars

        row = tk.Frame(frame)
        row.pack(fill="x", pady=2)
        tk.Label(row, text="Mode:", width=12, anchor="w", font=("Arial", 9)).pack(side="left")
        v["mode_label"] = tk.StringVar()
        mode_combo = ttk.Combobox(row, textvariable=v["mode_label"], state="readonly",
                                   width=14, font=("Arial", 8),
                                   values=[label for _key, label in _MODE_OPTIONS])
        mode_combo.pack(side="left")
        mode_combo.bind("<<ComboboxSelected>>", lambda e: self._on_mode_changed(mode_combo))
        self._mode_combo = mode_combo
        self._sync_mode_label()

        row = tk.Frame(frame)
        row.pack(fill="x", pady=2)
        tk.Label(row, text="Low cutoff (Hz):", width=15, anchor="w", font=("Arial", 9)).pack(side="left")
        v["f_low"] = tk.DoubleVar(value=self._params["f_low"])
        self._f_low_entry = tk.Entry(row, textvariable=v["f_low"], width=10)
        self._f_low_entry.pack(side="left")
        self._f_low_entry.bind("<Return>", lambda _e: self._on_field_changed())
        self._f_low_entry.bind("<FocusOut>", lambda _e: self._on_field_changed())

        row = tk.Frame(frame)
        row.pack(fill="x", pady=2)
        tk.Label(row, text="High cutoff (Hz):", width=15, anchor="w", font=("Arial", 9)).pack(side="left")
        v["f_high"] = tk.DoubleVar(value=self._params["f_high"])
        self._f_high_entry = tk.Entry(row, textvariable=v["f_high"], width=10)
        self._f_high_entry.pack(side="left")
        self._f_high_entry.bind("<Return>", lambda _e: self._on_field_changed())
        self._f_high_entry.bind("<FocusOut>", lambda _e: self._on_field_changed())

        row = tk.Frame(frame)
        row.pack(fill="x", pady=2)
        tk.Label(row, text="Order:", width=15, anchor="w", font=("Arial", 9)).pack(side="left")
        v["order"] = tk.DoubleVar(value=self._params["order"])
        order_entry = tk.Entry(row, textvariable=v["order"], width=10)
        order_entry.pack(side="left")
        order_entry.bind("<Return>", lambda _e: self._on_field_changed())
        order_entry.bind("<FocusOut>", lambda _e: self._on_field_changed())
        tk.Label(row, text="(steepness, typ. 2-8)", font=("Arial", 7), fg="#888888").pack(side="left", padx=(4, 0))

        row = tk.Frame(frame)
        row.pack(fill="x", pady=2)
        tk.Label(row, text="Shape:", width=15, anchor="w", font=("Arial", 9)).pack(side="left")
        v["shape_label"] = tk.StringVar()
        shape_combo = ttk.Combobox(row, textvariable=v["shape_label"], state="readonly",
                                    width=18, font=("Arial", 8),
                                    values=[label for _key, label in _SHAPE_OPTIONS])
        shape_combo.pack(side="left")
        shape_combo.bind("<<ComboboxSelected>>", lambda e: self._on_shape_changed(shape_combo))
        self._shape_combo = shape_combo
        self._sync_shape_label()

        row = tk.Frame(frame)
        row.pack(fill="x", pady=2)
        v["remove_mean"] = tk.BooleanVar(value=self._params["remove_mean"])
        tk.Checkbutton(row, text="Remove mean (DC) before filtering",
                       variable=v["remove_mean"], font=("Arial", 9),
                       command=self._on_field_changed).pack(anchor="w")

        row = tk.Frame(frame)
        row.pack(fill="x", pady=2)
        tk.Label(row, text="Edge padding (frac.):", width=15, anchor="w", font=("Arial", 9)).pack(side="left")
        v["pad_frac"] = tk.DoubleVar(value=self._params["pad_frac"])
        pad_entry = tk.Entry(row, textvariable=v["pad_frac"], width=10)
        pad_entry.pack(side="left")
        pad_entry.bind("<Return>", lambda _e: self._on_field_changed())
        pad_entry.bind("<FocusOut>", lambda _e: self._on_field_changed())
        tk.Label(row, text="(reflect-pad, reduces edge artifacts)",
                 font=("Arial", 7), fg="#888888").pack(side="left", padx=(4, 0))

        self._update_cutoff_enabled_state()

    def _sync_mode_label(self) -> None:
        label_by_key = dict(_MODE_OPTIONS)
        self._widget_vars["mode_label"].set(label_by_key.get(self._params["mode"], label_by_key["bandpass"]))

    def _sync_shape_label(self) -> None:
        label_by_key = dict(_SHAPE_OPTIONS)
        self._widget_vars["shape_label"].set(label_by_key.get(self._params["shape"], label_by_key["butterworth"]))

    def _on_mode_changed(self, combo: ttk.Combobox) -> None:
        key_by_label = {label: key for key, label in _MODE_OPTIONS}
        self._params["mode"] = key_by_label.get(combo.get(), "bandpass")
        self._update_cutoff_enabled_state()
        self._redraw_plots()

    def _on_shape_changed(self, combo: ttk.Combobox) -> None:
        key_by_label = {label: key for key, label in _SHAPE_OPTIONS}
        self._params["shape"] = key_by_label.get(combo.get(), "butterworth")
        self._redraw_plots()

    def _update_cutoff_enabled_state(self) -> None:
        mode = self._params["mode"]
        low_state = "disabled" if mode == "lowpass" else "normal"
        high_state = "disabled" if mode == "highpass" else "normal"
        self._f_low_entry.configure(state=low_state)
        self._f_high_entry.configure(state=high_state)

    def _on_field_changed(self) -> None:
        v = self._widget_vars
        try:
            f_low = float(v["f_low"].get())
            f_high = float(v["f_high"].get())
            order = float(v["order"].get())
            pad_frac = float(v["pad_frac"].get())
        except (tk.TclError, ValueError):
            return  # ignore transient invalid states while typing
        self._params["f_low"] = f_low
        self._params["f_high"] = f_high
        self._params["order"] = order
        self._params["pad_frac"] = max(0.0, pad_frac)
        self._params["remove_mean"] = bool(v["remove_mean"].get())
        self._redraw_plots()

    # -- Display (preview-only) controls -------------------------------------

    def _build_display_controls(self, parent: tk.Frame) -> None:
        frame = tk.LabelFrame(parent, text="Display", font=("Arial", 9), padx=6, pady=4)
        frame.pack(fill="x", padx=6, pady=4)

        row = tk.Frame(frame)
        row.pack(fill="x", pady=2)
        tk.Label(row, text="Signal index:", width=15, anchor="w", font=("Arial", 9)).pack(side="left")
        n_sig = self._n_signals()
        self._signal_index_var = tk.IntVar(value=self._signal_index)
        self._signal_index_spin = tk.Spinbox(
            row, from_=0, to=max(0, n_sig - 1), textvariable=self._signal_index_var,
            width=6, command=self._on_signal_index_changed)
        self._signal_index_spin.pack(side="left")
        self._signal_index_spin.bind("<Return>", lambda _e: self._on_signal_index_changed())
        self._signal_index_spin.bind("<FocusOut>", lambda _e: self._on_signal_index_changed())
        tk.Label(row, text=f"(0 .. {max(0, n_sig - 1)})", font=("Arial", 7), fg="#888888").pack(side="left", padx=(4, 0))

        row = tk.Frame(frame)
        row.pack(fill="x", pady=2)
        self._show_db_var = tk.BooleanVar(value=self._show_db)
        tk.Checkbutton(row, text="Show spectrum in dB", variable=self._show_db_var,
                       font=("Arial", 9), command=self._on_show_db_changed).pack(anchor="w")

    def _n_signals(self) -> int:
        if not self._has_input or self._last_signal is None:
            return 1
        sig = self._last_signal
        return 1 if sig.ndim == 1 else sig.shape[0]

    def _on_signal_index_changed(self) -> None:
        try:
            idx = int(self._signal_index_var.get())
        except (tk.TclError, ValueError):
            return
        n_sig = self._n_signals()
        idx = min(max(idx, 0), max(0, n_sig - 1))
        self._signal_index_var.set(idx)
        self._signal_index = idx
        self._redraw_plots()

    def _on_show_db_changed(self) -> None:
        self._show_db = bool(self._show_db_var.get())
        self._redraw_plots()

    # -- Apply ---------------------------------------------------------------

    def _on_apply_clicked(self) -> None:
        self._applied_params = copy.deepcopy(self._params)
        self._set_canvas_status(self._status_text())
        if self._request_downstream is not None:
            self._request_downstream(self.node_id)

    # -- Help popup -----------------------------------------------------------

    def _open_help(self) -> None:
        popup = getattr(self, "_help_popup", None)
        if popup is not None and popup.winfo_exists():
            popup.lift()
            return
        popup = tk.Toplevel(self.canvas.winfo_toplevel())
        popup.title(f"{self.DISPLAY_NAME} - Help")
        popup.geometry("720x640")
        body = tk.Frame(popup, bg="#f8f8f8", padx=10, pady=8)
        body.pack(fill="both", expand=True)
        txt = tk.Text(body, font=("Courier", 9), bg="#f8f8f8", fg="#222222",
                       wrap=tk.WORD, relief=tk.FLAT)
        vsb = ttk.Scrollbar(body, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", _HELP_TEXT)
        txt.configure(state="disabled")
        tk.Button(body, text="Close", font=("Arial", 9), command=popup.destroy
                  ).pack(anchor="e", pady=(8, 0))
        popup.bind("<Escape>", lambda _e: popup.destroy())
        self._help_popup = popup

    # -- Generic layout helpers -----------------------------------------------
    # (Same helpers used elsewhere in this project's nodes.)

    def _make_scrollable(self, parent: tk.Frame) -> tk.Frame:
        outer = tk.Frame(parent)
        outer.pack(fill="both", expand=True)
        canvas = tk.Canvas(outer, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(canvas)
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win_id, width=e.width))
        return inner

    def _make_paned(self, parent: tk.Frame, left_width: int = 290,
                     left_minsize: int = 220, right_minsize: int = 380):
        panes = tk.PanedWindow(parent, orient=tk.HORIZONTAL, sashwidth=6,
                                sashrelief=tk.RAISED, showhandle=True)
        panes.pack(fill="both", expand=True)
        left = tk.Frame(panes, bg="#f7f7f7")
        right = tk.Frame(panes, bg="#ffffff")
        panes.add(left, width=left_width, minsize=left_minsize)
        panes.add(right, minsize=right_minsize)
        return panes, left, right

    # -- Plot embedding, pan (drag) and wheel zoom -----------------------------
    # (Ctrl = X-only zoom, Shift = Y-only zoom, else both -- same interaction
    # model as the Signal Generator node's inspector, for a consistent feel
    # across the toolbox. Generalized here to drive two independent plots.)

    def _build_time_plot(self, parent: tk.Frame) -> None:
        self._fig_t = Figure(figsize=(6, 3.2), dpi=100)
        self._ax_t = self._fig_t.add_subplot(111)
        self._canvas_t = FigureCanvasTkAgg(self._fig_t, master=parent)
        self._canvas_t.get_tk_widget().pack(fill="both", expand=True)
        self._canvas_t.mpl_connect("scroll_event",
                                    lambda e: self._on_plot_scroll(e, self._ax_t, self._canvas_t))
        self._canvas_t.mpl_connect("button_press_event",
                                    lambda e: self._on_plot_press(e, self._ax_t, self._pan_state_t, self._extent_t, self._canvas_t))
        self._canvas_t.mpl_connect("motion_notify_event",
                                    lambda e: self._on_plot_drag(e, self._ax_t, self._pan_state_t, self._canvas_t))
        self._canvas_t.mpl_connect("button_release_event",
                                    lambda e: self._on_plot_release(e, self._pan_state_t))

    def _build_freq_plot(self, parent: tk.Frame) -> None:
        self._fig_f = Figure(figsize=(6, 3.2), dpi=100)
        self._ax_f = self._fig_f.add_subplot(111)
        self._ax_f2 = self._ax_f.twinx()
        self._canvas_f = FigureCanvasTkAgg(self._fig_f, master=parent)
        self._canvas_f.get_tk_widget().pack(fill="both", expand=True)
        self._canvas_f.mpl_connect("scroll_event",
                                    lambda e: self._on_plot_scroll(e, self._ax_f, self._canvas_f))
        self._canvas_f.mpl_connect("button_press_event",
                                    lambda e: self._on_plot_press(e, self._ax_f, self._pan_state_f, self._extent_f, self._canvas_f))
        self._canvas_f.mpl_connect("motion_notify_event",
                                    lambda e: self._on_plot_drag(e, self._ax_f, self._pan_state_f, self._canvas_f))
        self._canvas_f.mpl_connect("button_release_event",
                                    lambda e: self._on_plot_release(e, self._pan_state_f))

    @staticmethod
    def _event_data_coords(event, ax):
        """
        Convert the event's pixel position into `ax`'s own data
        coordinates, via ax.transData directly, instead of trusting
        event.xdata/event.ydata.

        This matters here because the frequency plot has a twin (twinx)
        axes overlaid on it for the H(f) response curve: matplotlib
        associates each mouse event with whichever axes it hit-tests as
        "on top" (event.inaxes) -- for an overlapping twin axes that is
        often the twin (0..1 response range), not the amplitude axis --
        so event.xdata/event.ydata can silently be in the WRONG axes'
        coordinate system. Going through the pixel position (event.x,
        event.y, which are unambiguous display coordinates) and this
        axes' own transform instead is correct regardless of which axes
        matplotlib happened to associate the event with.
        """
        if event.x is None or event.y is None or not ax.bbox.contains(event.x, event.y):
            return None, None
        try:
            xdata, ydata = ax.transData.inverted().transform((event.x, event.y))
        except Exception:
            return None, None
        return float(xdata), float(ydata)

    @staticmethod
    def _wheel_zoom_mode(event) -> str:
        """'x' (Ctrl held), 'y' (Shift held) or 'both'.

        Read from event.modifiers -- the Ctrl/Shift state carried by the
        wheel event itself -- rather than event.key. matplotlib fills
        event.key from key presses that the plot canvas received, which
        needs keyboard focus; with two plot canvases only the focused one
        saw Ctrl/Shift, so the modifiers silently did nothing on the other
        (the frequency plot)."""
        mods = getattr(event, "modifiers", None) or frozenset()
        key = getattr(event, "key", None) or ""
        if "ctrl" in mods or "control" in mods or key in ("control", "ctrl"):
            return "x"
        if "shift" in mods or key == "shift":
            return "y"
        return "both"

    def _on_plot_scroll(self, event, ax, canvas) -> None:
        if ax is None:
            return
        xdata, ydata = self._event_data_coords(event, ax)
        if xdata is None or ydata is None:
            return
        factor = 0.9 if event.button == "up" else 1.1
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        mode = self._wheel_zoom_mode(event)

        if mode == "x":
            new_xlim = [xdata - (xdata - v) * factor for v in xlim]
            ax.set_xlim(new_xlim)
        elif mode == "y":
            new_ylim = [ydata - (ydata - v) * factor for v in ylim]
            ax.set_ylim(new_ylim)
        else:
            new_xlim = [xdata - (xdata - v) * factor for v in xlim]
            new_ylim = [ydata - (ydata - v) * factor for v in ylim]
            ax.set_xlim(new_xlim)
            ax.set_ylim(new_ylim)
        if mode != "x":
            # Zoom the twin y-axis (H(f) response) about the same screen point,
            # so the response curve stays registered with the amplitude curves.
            for twin in self._twin_y_axes(ax):
                ty = twin.transData.inverted().transform((event.x, event.y))[1]
                twin.set_ylim([ty - (ty - v) * factor for v in twin.get_ylim()])

        canvas.draw_idle()

    def _on_plot_press(self, event, ax, state, extent, canvas) -> None:
        if ax is None:
            return
        xdata, ydata = self._event_data_coords(event, ax)
        if xdata is None:
            return
        if event.dblclick and event.button == 1:
            if extent is not None:
                (x0, x1), (y0, y1) = extent
                ax.set_xlim(x0, x1)
                ax.set_ylim(y0, y1)
                for twin in self._twin_y_axes(ax):
                    twin.set_ylim(*RESPONSE_YLIM)
                canvas.draw_idle()
            return
        if event.button != 1:
            return
        # Pan is computed in the data coordinates of the view at press time
        # (fixed inverse transform, as in scatter_plot_node). Converting each
        # drag position with the CURRENT transform -- which this handler keeps
        # changing by moving the limits -- made the plot vibrate.
        twins = []
        for twin in self._twin_y_axes(ax):
            tinv = twin.transData.inverted()
            twins.append((twin, tinv, tinv.transform((event.x, event.y))[1], twin.get_ylim()))
        state.update(active=True, inv=ax.transData.inverted(), x0=xdata, y0=ydata,
                      xlim0=ax.get_xlim(), ylim0=ax.get_ylim(), twins=twins)

    def _on_plot_drag(self, event, ax, state, canvas) -> None:
        if not state["active"] or ax is None or event.x is None or event.y is None:
            return
        # The frozen transform also keeps the pan going when the cursor
        # leaves the axes during the drag.
        xdata, ydata = state["inv"].transform((event.x, event.y))
        dx = xdata - state["x0"]
        dy = ydata - state["y0"]
        x0, x1 = state["xlim0"]
        y0, y1 = state["ylim0"]
        ax.set_xlim(x0 - dx, x1 - dx)
        ax.set_ylim(y0 - dy, y1 - dy)
        # The twin y-axis (H(f) response) shares x with `ax`, so it already
        # follows horizontally; move it vertically by the same screen distance.
        for twin, tinv, ty0, (t0, t1) in state.get("twins") or ():
            tdy = tinv.transform((event.x, event.y))[1] - ty0
            twin.set_ylim(t0 - tdy, t1 - tdy)
        canvas.draw_idle()

    def _twin_y_axes(self, ax) -> list:
        """Axes overlaid on `ax` with their own y-scale (the frequency plot's
        H(f) response axis), which pan / zoom / reset must move with `ax`."""
        return [self._ax_f2] if ax is self._ax_f and self._ax_f2 is not None else []

    def _on_plot_release(self, _event, state) -> None:
        state["active"] = False

    # -- Redraw ---------------------------------------------------------------

    def _current_signal_row(self, signal: np.ndarray) -> np.ndarray:
        if signal.ndim == 1:
            return signal
        idx = min(max(self._signal_index, 0), signal.shape[0] - 1)
        return signal[idx]

    def _redraw_plots(self) -> None:
        if self._ax_t is None or self._ax_f is None:
            return

        self._input_info_var.set(self._input_info_text()) if hasattr(self, "_input_info_var") else None
        if hasattr(self, "_signal_index_spin"):
            n_sig = self._n_signals()
            self._signal_index_spin.configure(to=max(0, n_sig - 1))

        ax_t = self._ax_t
        had_t = bool(ax_t.lines)
        xlim_t, ylim_t = (ax_t.get_xlim(), ax_t.get_ylim()) if had_t else (None, None)
        ax_t.clear()

        ax_f = self._ax_f
        ax_f2 = self._ax_f2
        had_f = bool(ax_f.lines)
        xlim_f, ylim_f = (ax_f.get_xlim(), ax_f.get_ylim()) if had_f else (None, None)
        ax_f.clear()
        ax_f2.clear()

        if not self._has_input or self._last_signal is None:
            ax_t.text(0.5, 0.5, "Waiting for signal input...", ha="center", va="center",
                       transform=ax_t.transAxes, color="#888888")
            ax_f.text(0.5, 0.5, "Waiting for signal input...", ha="center", va="center",
                       transform=ax_f.transAxes, color="#888888")
            self._canvas_t.draw_idle()
            self._canvas_f.draw_idle()
            return

        t = self._last_t
        dt = self._last_dt
        sig_row = self._current_signal_row(self._last_signal)
        filtered_full, freqs, H = _run_filter(self._last_signal, dt, self._params)
        filt_row = self._current_signal_row(filtered_full)

        # -- time-domain plot --
        ax_t.plot(t, sig_row, color="#1f77b4", linewidth=1.1, label="original")
        ax_t.plot(t, filt_row, color="#d62728", linewidth=1.1, label="filtered")
        ax_t.set_xlabel("time")
        ax_t.set_ylabel("value")
        ax_t.grid(True, linestyle="--", color="#dddddd")
        ax_t.legend(loc="best", fontsize=8)
        n_sig = self._n_signals()
        idx_note = f"  (signal {self._signal_index}/{max(0, n_sig - 1)})" if n_sig > 1 else ""
        ax_t.set_title(f"{self._params['mode']}{idx_note}", fontsize=9)
        self._extent_t = ((float(np.nanmin(t)), float(np.nanmax(t))),
                           self._padded_ylim([sig_row, filt_row]))
        if had_t and xlim_t is not None:
            ax_t.set_xlim(xlim_t)
            ax_t.set_ylim(ylim_t)
        else:
            ax_t.set_xlim(*self._extent_t[0])
            ax_t.set_ylim(*self._extent_t[1])

        # -- frequency-domain plot --
        _freqs_orig, amp_orig_full = _one_sided_spectrum(self._last_signal, dt)
        amp_orig = self._current_signal_row(amp_orig_full)
        amp_filt_full = _one_sided_spectrum(filtered_full, dt)[1]
        amp_filt = self._current_signal_row(amp_filt_full)
        freq_axis = _freqs_orig

        if self._show_db:
            floor = 1e-12
            y_orig = 20.0 * np.log10(np.maximum(amp_orig, floor))
            y_filt = 20.0 * np.log10(np.maximum(amp_filt, floor))
            y_label = "amplitude (dB)"
        else:
            y_orig, y_filt = amp_orig, amp_filt
            y_label = "amplitude"

        ax_f.plot(freq_axis, y_orig, color="#1f77b4", linewidth=1.1, label="original")
        ax_f.plot(freq_axis, y_filt, color="#d62728", linewidth=1.1, label="filtered")
        ax_f2.plot(freqs, H, color="#2ca02c", linewidth=1.0, linestyle="--", alpha=0.7,
                   label="response H(f)")
        ax_f2.set_ylim(*RESPONSE_YLIM)
        ax_f2.set_ylabel("response", color="#2ca02c")
        ax_f2.tick_params(axis="y", colors="#2ca02c")

        ax_f.set_xlabel("frequency (Hz)")
        ax_f.set_ylabel(y_label)
        ax_f.grid(True, linestyle="--", color="#dddddd")
        lines_a, labels_a = ax_f.get_legend_handles_labels()
        lines_b, labels_b = ax_f2.get_legend_handles_labels()
        ax_f.legend(lines_a + lines_b, labels_a + labels_b, loc="best", fontsize=8)

        f_xmax = float(freq_axis[-1]) if freq_axis.size else 1.0
        f_ylim = self._padded_ylim([y_orig, y_filt], allow_negative=self._show_db)
        self._extent_f = ((0.0, f_xmax), f_ylim)
        if had_f and xlim_f is not None:
            ax_f.set_xlim(xlim_f)
            ax_f.set_ylim(ylim_f)
        else:
            ax_f.set_xlim(*self._extent_f[0])
            ax_f.set_ylim(*self._extent_f[1])

        self._canvas_t.draw_idle()
        self._canvas_f.draw_idle()

    @staticmethod
    def _padded_ylim(arrays, allow_negative: bool = True):
        finite_vals = [a[np.isfinite(a)] for a in arrays]
        finite_vals = [a for a in finite_vals if a.size]
        if not finite_vals:
            return (-1.0, 1.0)
        ymin = float(min(np.min(a) for a in finite_vals))
        ymax = float(max(np.max(a) for a in finite_vals))
        if not allow_negative:
            ymin = min(ymin, 0.0)
        pad = 0.08 * max(ymax - ymin, 1e-9)
        return (ymin - pad, ymax + pad)