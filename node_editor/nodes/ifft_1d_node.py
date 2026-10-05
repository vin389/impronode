# node_editor/nodes/ifft_1d_node.py
"""
1D Inverse Fast Fourier Transform node.

Reconstructs a real-valued time-domain signal from a single-sided
amplitude spectrum (magnitude, phase), inverting the normalization used
by fft_1d_node.py. The time base for the reconstructed signal comes from
an optional freq array (preferred -- its spacing gives dt directly) or
an optional dt scalar (ignored when freq is provided); if neither is
given, a unit sample spacing (dt = 1) is assumed.

Caveat: if the upstream fft_1d node had a window or detrend option
enabled, this reconstructs the windowed/detrended signal, not the
original raw one -- windowing and mean removal are lossy and cannot be
undone from the spectrum alone.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from node_editor.base_node import BaseNode
from node_editor.pin_types import PinDef, PinSchema, PinType


class Ifft1DNode(BaseNode):
    """Reconstruct a real-valued 1D signal from a single-sided magnitude
    and phase spectrum (the inverse of Fft1DNode's output convention).
    """

    NODE_TYPE = "ifft_1d"
    DISPLAY_NAME = "IFFT (1D)"
    CATEGORY = "process"
    SEARCH_KEYWORDS = ("inverse fourier", "idft", "synthesis", "reconstruct")

    NODE_WIDTH = 150
    NODE_HEIGHT = 90

    # How to recover the original signal length n from the spectrum
    # length m: an even-length signal gives m = n//2 + 1 with a true
    # Nyquist bin at the end; an odd-length signal gives m = (n+1)//2
    # with no exact Nyquist bin. m alone cannot disambiguate the two.
    PARITY_CHOICES = ("even", "odd")

    # ══ Construction ════════════════════════════════════════════════

    def __init__(self, node_id: str, canvas: tk.Canvas):
        super().__init__(node_id, canvas)

        # --- Settings. These affect the computed output directly and
        # immediately -- no separate trial/apply state, same as fft_1d,
        # since there is no interactive search here to preview first. ---
        self._length_parity: str = "even"   # "even" | "odd"
        self._show_phase: bool = False      # display-only: overlay phase on the spectrum plot

        # Cached results from the last compute(), kept so the inspector
        # can redraw without forcing a graph recompute.
        self._freq: np.ndarray | None = None
        self._magnitude: np.ndarray | None = None
        self._phase: np.ndarray | None = None
        self._t: np.ndarray | None = None
        self._signal: np.ndarray | None = None
        self._dt: float | None = None
        self._n: int | None = None
        self._time_base_source: str = "none"   # "freq" | "dt" | "none" -- for the status line
        self._length_warning: str | None = None

        # Matplotlib objects, created lazily in build_inspector(). Two
        # stacked subplots: spectrum on top, reconstructed signal below.
        self._fig: Figure | None = None
        self._ax_freq = None
        self._ax_phase = None   # twin axis of _ax_freq, only created if _show_phase
        self._ax_time = None
        self._canvas_widget: FigureCanvasTkAgg | None = None

        # Pan state is tracked per-axis, since each subplot pans/zooms
        # independently.
        self._pan_state: dict = {}

        self._status_item: int | None = None

    # ══ Pin schema ══════════════════════════════════════════════════

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[
                PinDef(name="magnitude", type=PinType.ARRAY, label="magnitude"),
                PinDef(name="phase", type=PinType.ARRAY, label="phase"),
                PinDef(name="freq", type=PinType.ARRAY, label="freq", optional=True),
                PinDef(name="dt", type=PinType.SCALAR, label="dt", optional=True),
            ],
            outputs=[
                PinDef(name="t", type=PinType.ARRAY, label="t"),
                PinDef(name="signal", type=PinType.ARRAY, label="signal"),
                PinDef(name="dt", type=PinType.SCALAR, label="dt"),
            ],
        )

    # ══ Compute ═════════════════════════════════════════════════════

    def _resolve_n(self, m: int) -> int:
        """Recover the original signal length n from the spectrum length m,
        using the user-selected parity assumption.
        """
        if self._length_parity == "odd":
            return 2 * m - 1
        return 2 * (m - 1)  # "even"

    def _resolve_dt(self, n: int, freq: np.ndarray | None, dt: float | None) -> tuple[float, str]:
        """Return (dt, source_label) for the reconstructed time axis.

        freq takes priority over dt (dt is ignored when freq is
        provided, as specified). Falls back to dt = 1 if neither is
        available.
        """
        if freq is not None and len(freq) > 1:
            df = float(np.median(np.diff(freq)))
            if df > 0 and n > 0:
                return 1.0 / (n * df), "freq"
        if dt is not None:
            return float(dt), "dt"
        return 1.0, "none"

    def _reconstruct_signal(self, magnitude: np.ndarray, phase: np.ndarray,
                             n: int) -> np.ndarray:
        """Undo fft_1d's single-sided amplitude normalization and invert
        the transform.

        fft_1d computed magnitude = |X| / n, then doubled every bin
        except DC (and Nyquist, when n is even) since those bins have no
        mirrored negative-frequency twin. This reverses exactly that
        scaling before calling irfft.
        """
        amplitude = magnitude * n
        if n % 2 == 0:
            amplitude[1:-1] /= 2.0   # bins 1 .. second-to-last were doubled
        else:
            amplitude[1:] /= 2.0     # all bins except DC were doubled

        spectrum = amplitude * np.exp(1j * phase)
        return np.fft.irfft(spectrum, n=n)

    def compute(self, inputs: dict) -> dict:
        magnitude = inputs.get("magnitude")
        phase = inputs.get("phase")

        if magnitude is None or phase is None:
            self._reset_cache()
            return {"t": None, "signal": None, "dt": None}

        magnitude = np.asarray(magnitude, dtype=float)
        phase = np.asarray(phase, dtype=float)

        if len(magnitude) != len(phase):
            self._length_warning = (
                f"magnitude (len {len(magnitude)}) and phase (len {len(phase)}) "
                "must be the same length"
            )
            self._reset_cache()
            return {"t": None, "signal": None, "dt": None}

        m = len(magnitude)
        n = self._resolve_n(m)

        freq_in = inputs.get("freq")
        freq_in = np.asarray(freq_in, dtype=float) if freq_in is not None else None
        if freq_in is not None and len(freq_in) != m:
            self._length_warning = (
                f"freq (len {len(freq_in)}) does not match magnitude/phase (len {m})"
            )
        else:
            self._length_warning = None

        dt, source = self._resolve_dt(n, freq_in, inputs.get("dt"))
        signal = self._reconstruct_signal(magnitude.copy(), phase, n)
        t = np.arange(n, dtype=float) * dt

        # freq axis for the spectrum plot: use the connected freq if
        # given (and consistent), otherwise reconstruct it from n and dt.
        freq_for_plot = freq_in if (freq_in is not None and len(freq_in) == m) else np.fft.rfftfreq(n, d=dt)

        # Cache for the inspector.
        self._freq, self._magnitude, self._phase = freq_for_plot, magnitude, phase
        self._t, self._signal, self._dt, self._n = t, signal, dt, n
        self._time_base_source = source

        if getattr(self, "_status_item", None) is not None:
            self.canvas.itemconfig(self._status_item, text=self._status_text())

        return {"t": t, "signal": signal, "dt": dt}

    def _reset_cache(self) -> None:
        self._freq = self._magnitude = self._phase = None
        self._t = self._signal = self._dt = self._n = None
        self._time_base_source = "none"
        if getattr(self, "_status_item", None) is not None:
            self.canvas.itemconfig(self._status_item, text=self._status_text())

    # ══ Serialization ═══════════════════════════════════════════════

    def get_params(self) -> dict:
        return {
            "length_parity": self._length_parity,
            "show_phase": self._show_phase,
        }

    def set_params(self, params: dict) -> None:
        self._length_parity = params.get("length_parity", "even")
        self._show_phase = bool(params.get("show_phase", False))

    # ══ Canvas body ═════════════════════════════════════════════════

    def _status_text(self) -> str:
        if self._n is None:
            return "no spectrum" if not self._length_warning else "length mismatch"
        source_label = {"freq": "from freq", "dt": "from dt", "none": "dt=1 assumed"}[self._time_base_source]
        return f"n={self._n} ({self._length_parity}), dt={self._dt:.4g} ({source_label})"

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
        self._ax_freq = self._fig.add_subplot(211)
        self._ax_time = self._fig.add_subplot(212)
        self._fig.tight_layout(pad=2.0)

        self._canvas_widget = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas_widget.get_tk_widget().pack(fill="both", expand=True)

        # for ax in (self._ax_freq, self._ax_time):
        #     self._pan_state[ax] = {
        #         "active": False, "x0": None, "y0": None, "xlim0": None, "ylim0": None,
        #     }
        for ax in (self._ax_freq, self._ax_time):
            self._pan_state[ax] = {
                "active": False, "x_px": None, "y_px": None,
                "xlim0": None, "ylim0": None, "x_scale": None, "y_scale": None,
            }        

        self._canvas_widget.mpl_connect("scroll_event", self._on_plot_scroll)
        self._canvas_widget.mpl_connect("button_press_event", self._on_plot_press)
        self._canvas_widget.mpl_connect("motion_notify_event", self._on_plot_drag)
        self._canvas_widget.mpl_connect("button_release_event", self._on_plot_release)

    def _on_plot_scroll(self, event) -> None:
        # event.inaxes tells us which of the two subplots the cursor is
        # over; each subplot zooms independently. The phase twin-axis
        # (when shown) shares the same x-axis as _ax_freq, so scrolling
        # over it is treated the same as scrolling over _ax_freq.
        ax = event.inaxes
        if ax is self._ax_phase:
            ax = self._ax_freq
        if ax is None or event.xdata is None or event.ydata is None:
            return

        factor = 0.9 if event.button == "up" else 1.1
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        key = getattr(event, "key", None)

        # Ctrl+wheel: zoom the x-axis only. Shift+wheel: zoom the y-axis
        # only. No modifier: zoom both axes together.
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

    # def _on_plot_press(self, event) -> None:
    #     ax = event.inaxes
    #     if ax is self._ax_phase:
    #         ax = self._ax_freq
    #     if ax is None or event.button != 1 or event.xdata is None:
    #         return
    #     self._pan_state[ax].update(
    #         active=True, x0=event.xdata, y0=event.ydata,
    #         xlim0=ax.get_xlim(), ylim0=ax.get_ylim(),
    #     )
    def _on_plot_press(self, event) -> None:
        ax = event.inaxes
        if ax is self._ax_phase:
            ax = self._ax_freq
        if ax is None or event.button != 1 or event.x is None:
            return
        xlim0, ylim0 = ax.get_xlim(), ax.get_ylim()
        bbox = ax.bbox
        if bbox.width <= 0 or bbox.height <= 0:
            return
        self._pan_state[ax].update(
            active=True, x_px=event.x, y_px=event.y, xlim0=xlim0, ylim0=ylim0,
            x_scale=(xlim0[1] - xlim0[0]) / bbox.width,
            y_scale=(ylim0[1] - ylim0[0]) / bbox.height,
        )

    # def _on_plot_drag(self, event) -> None:
    #     event_ax = self._ax_freq if event.inaxes is self._ax_phase else event.inaxes
    #     for ax, state in self._pan_state.items():
    #         if not state["active"]:
    #             continue
    #         if event.xdata is None or event_ax is not ax:
    #             continue
    #         dx = event.xdata - state["x0"]
    #         dy = event.ydata - state["y0"]
    #         x0, x1 = state["xlim0"]
    #         y0, y1 = state["ylim0"]
    #         ax.set_xlim(x0 - dx, x1 - dx)
    #         ax.set_ylim(y0 - dy, y1 - dy)
    #         self._canvas_widget.draw_idle()
    def _on_plot_drag(self, event) -> None:
        """Pan by PIXEL displacement, converted to data units with the
        scale fixed at button-press time -- see time_sync_node.py's
        _on_plot_drag for the full rationale (using a transform that
        changes between calls produces jitter and lag).

        Unlike the previous version, this no longer needs to remap the
        phase twin-axis to _ax_freq here: pixel coordinates are the same
        across the whole figure regardless of which axes matplotlib
        currently associates the event with, so simply updating whatever
        axis's state is active is enough -- that axis was already
        resolved once, correctly, in _on_plot_press.
        """
        if event.x is None or event.y is None:
            return
        for ax, state in self._pan_state.items():
            if not state["active"]:
                continue
            dx = (event.x - state["x_px"]) * state["x_scale"]
            dy = (event.y - state["y_px"]) * state["y_scale"]
            x0, x1 = state["xlim0"]
            y0, y1 = state["ylim0"]
            ax.set_xlim(x0 - dx, x1 - dx)
            ax.set_ylim(y0 - dy, y1 - dy)
            self._canvas_widget.draw_idle()
            
    def _on_plot_release(self, _event) -> None:
        for state in self._pan_state.values():
            state["active"] = False

    def _redraw_plots(self) -> None:
        ax_f, ax_t = self._ax_freq, self._ax_time

        had_freq_data = bool(ax_f.lines)
        had_time_data = bool(ax_t.lines)
        f_xlim, f_ylim = (ax_f.get_xlim(), ax_f.get_ylim()) if had_freq_data else (None, None)
        t_xlim, t_ylim = (ax_t.get_xlim(), ax_t.get_ylim()) if had_time_data else (None, None)

        ax_f.clear()
        if self._ax_phase is not None:
            self._ax_phase.remove()
            self._ax_phase = None
        ax_t.clear()

        if self._freq is not None and self._magnitude is not None:
            ax_f.plot(self._freq, self._magnitude, color="#d62728", linewidth=1.0, label="magnitude")
        freq_unit = "Hz" if self._time_base_source != "none" else "cycles/sample"
        ax_f.set_xlabel(f"frequency ({freq_unit})")
        ax_f.set_ylabel("magnitude", color="#d62728")
        ax_f.set_title("input spectrum", fontsize=9)

        if self._show_phase and self._freq is not None and self._phase is not None:
            self._ax_phase = ax_f.twinx()
            self._ax_phase.plot(self._freq, self._phase, color="#1f77b4", linewidth=0.8,
                                 linestyle="--", label="phase")
            self._ax_phase.set_ylabel("phase (rad)", color="#1f77b4")
            # self._pan_state.setdefault(self._ax_phase, {
            #     "active": False, "x0": None, "y0": None, "xlim0": None, "ylim0": None,
            # })
            self._pan_state.setdefault(self._ax_phase, {
                "active": False, "x_px": None, "y_px": None,
                "xlim0": None, "ylim0": None, "x_scale": None, "y_scale": None,
            })

        if self._t is not None and self._signal is not None:
            ax_t.plot(self._t, self._signal, color="#1f77b4", linewidth=1.0)
        ax_t.set_xlabel("time" if self._time_base_source != "none" else "sample index")
        ax_t.set_ylabel("value")
        ax_t.set_title("reconstructed signal", fontsize=9)

        if had_freq_data:
            ax_f.set_xlim(f_xlim)
            ax_f.set_ylim(f_ylim)
        if had_time_data:
            ax_t.set_xlim(t_xlim)
            ax_t.set_ylim(t_ylim)

        self._fig.tight_layout(pad=2.0)
        self._canvas_widget.draw_idle()

        self._insp_warning_var.set(self._length_warning or "")

    # -- Settings controls (length parity, phase overlay) ---------------------

    def _build_settings_controls(self, parent: tk.Frame) -> None:
        row = tk.Frame(parent)
        row.pack(fill="x", pady=(6, 0))

        tk.Label(row, text="Signal length parity:").pack(side="left")
        self._insp_parity_var = tk.StringVar(value=self._length_parity)
        parity_box = ttk.Combobox(
            row, textvariable=self._insp_parity_var, values=self.PARITY_CHOICES,
            state="readonly", width=8,
        )
        parity_box.pack(side="left", padx=(4, 12))
        parity_box.bind("<<ComboboxSelected>>", lambda _e: self._on_parity_changed())

        self._insp_show_phase_var = tk.BooleanVar(value=self._show_phase)
        tk.Checkbutton(
            row, text="Show phase overlay", variable=self._insp_show_phase_var,
            command=self._on_display_setting_changed,
        ).pack(side="left")

    def _on_parity_changed(self) -> None:
        # Changing parity changes the recovered n, and therefore the
        # reconstructed signal itself -- recompute via the engine.
        self._length_parity = self._insp_parity_var.get()
        if self._request_downstream is not None:
            self._request_downstream(self.node_id)
        # Responsiveness fallback using the previously cached arrays,
        # same caveat as fft_1d_node.py: if the engine recomputes
        # synchronously this is immediately superseded by fresh data;
        # if asynchronous, this redraw briefly shows stale data.
        self._redraw_plots()

    def _on_display_setting_changed(self) -> None:
        # Phase-overlay toggle only affects how the cached spectrum is
        # drawn, not the computed values -- no need to touch the engine.
        self._show_phase = bool(self._insp_show_phase_var.get())
        self._redraw_plots()