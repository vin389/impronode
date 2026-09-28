# node_editor/nodes/signal_generator_node.py
"""
Time-series / signal generator node.

A source node (no input pins) that synthesizes a signal as the sum of up
to six damped sine components plus one noise component, sampled over a
common duration and sample interval. The inspector shows an immediately
updating preview; only pressing "Apply" writes the generated dt and
signal to the output pins.
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


# Number of damped-sine channels (channel 7, the noise channel, is separate).
NUM_SINE_CHANNELS = 6


def _default_sine_channel(index: int) -> dict:
    """Default parameters for one damped-sine channel. Only channel 0 is
    enabled by default so a freshly-created node still outputs something
    sensible before the user has touched the inspector.
    """
    return {
        "enabled": index == 0,
        "amplitude": 1.0,
        "frequency": 1.0 + index,   # spread defaults out: 1, 2, 3, ... Hz
        "decay": 0.0,
        "time_shift": 0.0,
    }


def _default_noise_channel() -> dict:
    return {
        "enabled": False,
        "distribution": "gaussian",  # "gaussian" | "uniform"
        "amplitude": 0.05,           # std for gaussian, half-range for uniform
        "seed": 0,
    }


def _generate_signal(common: dict, sine_channels: list[dict],
                      noise_channel: dict) -> tuple[np.ndarray, float, np.ndarray]:
    """Build the time vector and summed signal from a full parameter set.

    Returns (t, dt, signal). Pure function of its arguments (given the
    same seed, the noise channel is reproducible), so it is safe to call
    both for the live preview (trial params) and inside compute()
    (applied params).
    """
    dt = float(common["dt"])
    duration = float(common["duration"])
    n = max(1, int(round(duration / dt))) if dt > 0 else 1
    t = np.arange(n, dtype=float) * dt

    signal = np.zeros(n, dtype=float)

    for ch in sine_channels:
        if not ch["enabled"]:
            continue
        # Decay runs over the global record time t (as literally specified);
        # time_shift only shifts the sine's phase argument, which is what
        # lets a single real-valued shift stand in for a cosine term.
        envelope = np.exp(-ch["decay"] * t)
        phase_arg = 2.0 * np.pi * ch["frequency"] * (t - ch["time_shift"])
        signal += ch["amplitude"] * envelope * np.sin(phase_arg)

    if noise_channel["enabled"]:
        rng = np.random.default_rng(int(noise_channel["seed"]))
        amp = float(noise_channel["amplitude"])
        if noise_channel["distribution"] == "uniform":
            noise = rng.uniform(-amp, amp, size=n)
        else:  # "gaussian"
            noise = rng.normal(0.0, amp, size=n)
        signal += noise

    return t, dt, signal


class SignalGeneratorNode(BaseNode):
    """Synthesize a signal from up to 6 damped sine channels plus 1 noise
    channel. No input pins. Outputs (dt, t, signal) update only on Apply.
    """

    NODE_TYPE = "signal_generator"
    DISPLAY_NAME = "Signal Generator"
    CATEGORY = "source"
    SEARCH_KEYWORDS = ("sine", "signal", "synthetic", "noise", "waveform")

    NODE_WIDTH = 150
    NODE_HEIGHT = 90

    NOISE_DISTRIBUTION_CHOICES = ("gaussian", "uniform")

    # ══ Construction ════════════════════════════════════════════════

    def __init__(self, node_id: str, canvas: tk.Canvas):
        super().__init__(node_id, canvas)

        # --- Trial state: freely editable in the inspector, previewed
        # immediately, but does NOT affect the output pins until Apply ---
        self._common: dict = {"duration": 10.0, "dt": 0.01}
        self._sine_channels: list[dict] = [
            _default_sine_channel(i) for i in range(NUM_SINE_CHANNELS)
        ]
        self._noise_channel: dict = _default_noise_channel()

        # --- Applied state: the ONLY thing that affects the output pins.
        # Generated eagerly at construction time so the node has a
        # sensible output even if the inspector is never opened. ---
        self._applied_common: dict = copy.deepcopy(self._common)
        self._applied_sine_channels: list[dict] = copy.deepcopy(self._sine_channels)
        self._applied_noise_channel: dict = copy.deepcopy(self._noise_channel)

        # Matplotlib objects, created lazily in build_inspector().
        self._fig: Figure | None = None
        self._ax = None
        self._canvas_widget: FigureCanvasTkAgg | None = None
        self._pan_state: dict = {
            "active": False, "x0": None, "y0": None, "xlim0": None, "ylim0": None,
        }

        # Per-channel Tk variable bundles, populated in build_inspector().
        self._sine_widget_vars: list[dict] = []
        self._noise_widget_vars: dict = {}

        self._status_item: int | None = None

    # ══ Pin schema ══════════════════════════════════════════════════

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[],  # no input pins in this version
            outputs=[
                PinDef(name="dt", type=PinType.SCALAR, label="dt"),
                PinDef(name="t", type=PinType.ARRAY, label="t"),
                PinDef(name="signal", type=PinType.ARRAY, label="signal"),
            ],
        )

    # ══ Compute ═════════════════════════════════════════════════════

    def compute(self, inputs: dict) -> dict:
        # No inputs to read; only the APPLIED parameter snapshot feeds
        # the output pins, regenerated deterministically each time.
        t, dt, signal = _generate_signal(
            self._applied_common, self._applied_sine_channels, self._applied_noise_channel,
        )
        return {"dt": dt, "t": t, "signal": signal}

    # ══ Serialization ═══════════════════════════════════════════════

    def get_params(self) -> dict:
        # Serialize the APPLIED state -- that is what actually produced
        # the current outputs, so it is what a reload should reproduce.
        return {
            "common": self._applied_common,
            "sine_channels": self._applied_sine_channels,
            "noise_channel": self._applied_noise_channel,
        }

    def set_params(self, params: dict) -> None:
        common = params.get("common")
        if isinstance(common, dict):
            self._applied_common.update(common)

        sine_channels = params.get("sine_channels")
        if isinstance(sine_channels, list):
            for i, ch in enumerate(sine_channels[:NUM_SINE_CHANNELS]):
                if isinstance(ch, dict):
                    self._applied_sine_channels[i].update(ch)

        noise_channel = params.get("noise_channel")
        if isinstance(noise_channel, dict):
            self._applied_noise_channel.update(noise_channel)

        # On load, the inspector should show the settings that are
        # actually in effect, not stale in-progress edits.
        self._common = copy.deepcopy(self._applied_common)
        self._sine_channels = copy.deepcopy(self._applied_sine_channels)
        self._noise_channel = copy.deepcopy(self._applied_noise_channel)

    # ══ Canvas body ═════════════════════════════════════════════════

    def _status_text(self) -> str:
        n_active = sum(ch["enabled"] for ch in self._applied_sine_channels)
        noise_on = self._applied_noise_channel["enabled"]
        return f"{n_active} sine ch. + {'noise' if noise_on else 'no noise'}"

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
        # -- Plot --
        plot_frame = tk.Frame(parent)
        plot_frame.pack(fill="both", expand=True)
        self._build_plot(plot_frame)

        # -- Common parameters (duration, dt) --
        self._build_common_controls(parent)

        # -- 6 sine channels + 1 noise channel --
        self._build_channel_table(parent)

        # -- Apply --
        apply_row = tk.Frame(parent)
        apply_row.pack(fill="x", pady=(8, 0))
        tk.Button(apply_row, text="Apply", width=12,
                  command=self._on_apply_clicked).pack(side="right")

        self._redraw_plot()

    # -- Plot embedding, pan (drag) and wheel zoom -------------------------
    # (Same interaction model as the Time Sync node's inspector, for a
    # consistent feel across the toolbox.)

    def _build_plot(self, parent: tk.Frame) -> None:
        self._fig = Figure(figsize=(6, 3.5), dpi=100)
        self._ax = self._fig.add_subplot(111)
        self._canvas_widget = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas_widget.get_tk_widget().pack(fill="both", expand=True)

        self._canvas_widget.mpl_connect("scroll_event", self._on_plot_scroll)
        self._canvas_widget.mpl_connect("button_press_event", self._on_plot_press)
        self._canvas_widget.mpl_connect("motion_notify_event", self._on_plot_drag)
        self._canvas_widget.mpl_connect("button_release_event", self._on_plot_release)

    def _on_plot_scroll(self, event) -> None:
        if event.xdata is None or event.ydata is None or self._ax is None:
            return

        factor = 0.9 if event.button == "up" else 1.1
        xlim, ylim = self._ax.get_xlim(), self._ax.get_ylim()
        modifiers = set(getattr(event, "modifiers", ()) or ())
        modifiers.update((getattr(event, "key", None) or "").split("+"))

        # Shift constrains zoom vertically; Ctrl constrains it horizontally.
        # Shift takes precedence when both modifiers are held.
        if "shift" in modifiers:
            new_ylim = [event.ydata - (event.ydata - v) * factor for v in ylim]
            self._ax.set_ylim(new_ylim)
        elif modifiers.intersection({"control", "ctrl"}):
            new_xlim = [event.xdata - (event.xdata - v) * factor for v in xlim]
            self._ax.set_xlim(new_xlim)
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

    def _redraw_plot(self) -> None:
        """Regenerate the preview signal from the current TRIAL parameters
        and redraw immediately. Cheap (pure numpy), so no live/deferred
        split is needed here unlike the heavier sync-error computation
        in the Time Sync node.
        """
        ax = self._ax
        had_data = bool(ax.lines)
        xlim, ylim = (ax.get_xlim(), ax.get_ylim()) if had_data else (None, None)
        ax.clear()

        t, _dt, signal = _generate_signal(self._common, self._sine_channels, self._noise_channel)
        ax.plot(t, signal, color="#1f77b4", linewidth=1.2)
        ax.set_xlabel("time")
        ax.set_ylabel("value")

        if had_data:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        self._canvas_widget.draw_idle()

    # -- Common parameters (duration, dt) -----------------------------------

    def _set_entry_validity(self, entry: tk.Entry | None, valid: bool) -> None:
        if entry is not None and entry.winfo_exists():
            entry.configure(fg="#000000" if valid else "#cc0000")

    def _validate_common_entries(self) -> None:
        entries = {
            "duration": getattr(self, "_duration_entry", None),
            "dt": getattr(self, "_dt_entry", None),
        }

        for name, entry in entries.items():
            try:
                value = float(self._duration_var.get() if name == "duration" else self._dt_var.get())
                valid = bool(np.isfinite(value) and value > 0)
            except (tk.TclError, ValueError, TypeError):
                valid = False
            self._set_entry_validity(entry, valid)

    def _build_common_controls(self, parent: tk.Frame) -> None:
        row = tk.Frame(parent)
        row.pack(fill="x", pady=(8, 0))

        tk.Label(row, text="Duration:").pack(side="left")
        self._duration_var = tk.DoubleVar(value=self._common["duration"])
        self._duration_entry = tk.Entry(row, textvariable=self._duration_var, width=8)
        self._duration_entry.pack(side="left", padx=(4, 12))

        tk.Label(row, text="dt:").pack(side="left")
        self._dt_var = tk.DoubleVar(value=self._common["dt"])
        self._dt_entry = tk.Entry(row, textvariable=self._dt_var, width=8)
        self._dt_entry.pack(side="left", padx=(4, 0))

        for entry, callback in (
            (self._duration_entry, lambda _e: self._on_common_changed()),
            (self._dt_entry, lambda _e: self._on_common_changed()),
        ):
            entry.bind("<Return>", callback)
            entry.bind("<FocusOut>", callback)
            entry.bind("<KeyRelease>", lambda _e, e=entry: self._validate_common_entries())

        self._validate_common_entries()

    def _on_common_changed(self, *, redraw: bool = True) -> None:
        try:
            duration = float(self._duration_var.get())
            dt = float(self._dt_var.get())
        except (tk.TclError, ValueError):
            self._validate_common_entries()
            return  # ignore transient invalid states while typing
        if not np.isfinite(duration) or not np.isfinite(dt) or duration <= 0 or dt <= 0:
            self._validate_common_entries()
            return
        self._common["duration"] = duration
        self._common["dt"] = dt
        self._validate_common_entries()
        if redraw:
            self._redraw_plot()

    # -- Channel table: 6 sine rows + 1 noise row ---------------------------

    def _validate_sine_entry_state(self, index: int) -> None:
        vars_ = self._sine_widget_vars[index]
        for key, entry in (
            ("amplitude", vars_.get("entries", {}).get("amplitude")),
            ("frequency", vars_.get("entries", {}).get("frequency")),
            ("decay", vars_.get("entries", {}).get("decay")),
            ("time_shift", vars_.get("entries", {}).get("time_shift")),
        ):
            if entry is None:
                continue
            try:
                value = float(vars_[key].get())
                if key == "amplitude":
                    valid = np.isfinite(value) and value >= 0.0
                elif key == "frequency":
                    valid = np.isfinite(value) and value > 0.0
                elif key == "decay":
                    valid = np.isfinite(value) and value >= 0.0
                else:
                    valid = np.isfinite(value)
            except (tk.TclError, TypeError, ValueError):
                valid = False
            self._set_entry_validity(entry, valid)

    def _validate_noise_entry_state(self) -> None:
        vars_ = self._noise_widget_vars
        for key, entry in (
            ("amplitude", vars_.get("entries", {}).get("amplitude")),
            ("seed", vars_.get("entries", {}).get("seed")),
        ):
            if entry is None:
                continue
            try:
                if key == "amplitude":
                    value = float(vars_[key].get())
                    valid = np.isfinite(value) and value >= 0.0
                else:
                    value = int(vars_[key].get())
                    valid = value >= 0
            except (tk.TclError, TypeError, ValueError):
                valid = False
            self._set_entry_validity(entry, valid)

    def _build_channel_table(self, parent: tk.Frame) -> None:
        table = tk.Frame(parent)
        table.pack(fill="x", pady=(8, 0))

        headers = ("----", "on", "amplitude", "frequency", "decay", "time_shift")
        for col, text in enumerate(headers):
            tk.Label(table, text=text, font=("Arial", 8, "bold")).grid(row=0, column=col, padx=4)

        self._sine_widget_vars = []
        for i, ch in enumerate(self._sine_channels):
            row = i + 1
            tk.Label(table, text=f"sine {i + 1}", font=("Arial", 8)).grid(row=row, column=0, sticky="w")

            enabled_var = tk.BooleanVar(value=ch["enabled"])
            tk.Checkbutton(table, variable=enabled_var,
                           command=lambda idx=i: self._on_sine_widget_changed(idx)).grid(row=row, column=1)

            amp_var = tk.DoubleVar(value=ch["amplitude"])
            freq_var = tk.DoubleVar(value=ch["frequency"])
            decay_var = tk.DoubleVar(value=ch["decay"])
            shift_var = tk.DoubleVar(value=ch["time_shift"])

            entry_map = {}
            for col, (var, key) in enumerate(
                ((amp_var, "amplitude"), (freq_var, "frequency"),
                 (decay_var, "decay"), (shift_var, "time_shift")), start=2,
            ):
                entry = tk.Entry(table, textvariable=var, width=8)
                entry.grid(row=row, column=col, padx=2)
                entry_map[key] = entry
                entry.bind("<Return>", lambda _e, idx=i: self._on_sine_widget_changed(idx))
                entry.bind("<FocusOut>", lambda _e, idx=i: self._on_sine_widget_changed(idx))
                entry.bind("<KeyRelease>", lambda _e, idx=i: self._validate_sine_entry_state(idx))

            self._sine_widget_vars.append({
                "enabled": enabled_var, "amplitude": amp_var,
                "frequency": freq_var, "decay": decay_var, "time_shift": shift_var,
                "entries": entry_map,
            })
            self._validate_sine_entry_state(i)

        # -- Channel 7: noise, with a different parameter set --
        noise_row = NUM_SINE_CHANNELS + 1
        tk.Label(table, text="noise (ch. 7)", font=("Arial", 8, "bold")).grid(row=noise_row, column=0, sticky="w")

        noise_enabled_var = tk.BooleanVar(value=self._noise_channel["enabled"])
        tk.Checkbutton(table, variable=noise_enabled_var,
                       command=self._on_noise_widget_changed).grid(row=noise_row, column=1)

        dist_var = tk.StringVar(value=self._noise_channel["distribution"])
        dist_box = ttk.Combobox(table, textvariable=dist_var, values=self.NOISE_DISTRIBUTION_CHOICES,
                                 state="readonly", width=8)
        dist_box.grid(row=noise_row, column=2, padx=2)
        dist_box.bind("<<ComboboxSelected>>", lambda _e: self._on_noise_widget_changed())

        amp_var = tk.DoubleVar(value=self._noise_channel["amplitude"])
        amp_entry = tk.Entry(table, textvariable=amp_var, width=8)
        amp_entry.grid(row=noise_row, column=3, padx=2)
        amp_entry.bind("<Return>", lambda _e: self._on_noise_widget_changed())
        amp_entry.bind("<FocusOut>", lambda _e: self._on_noise_widget_changed())
        amp_entry.bind("<KeyRelease>", lambda _e: self._validate_noise_entry_state())

        tk.Label(table, text="seed:", font=("Arial", 8)).grid(row=noise_row, column=4, sticky="e")
        seed_var = tk.IntVar(value=self._noise_channel["seed"])
        seed_entry = tk.Entry(table, textvariable=seed_var, width=6)
        seed_entry.grid(row=noise_row, column=5, padx=2)
        seed_entry.bind("<Return>", lambda _e: self._on_noise_widget_changed())
        seed_entry.bind("<FocusOut>", lambda _e: self._on_noise_widget_changed())
        seed_entry.bind("<KeyRelease>", lambda _e: self._validate_noise_entry_state())

        self._noise_widget_vars = {
            "enabled": noise_enabled_var, "distribution": dist_var,
            "amplitude": amp_var, "seed": seed_var,
            "entries": {"amplitude": amp_entry, "seed": seed_entry},
        }
        self._validate_noise_entry_state()

    def _on_sine_widget_changed(self, index: int, *, redraw: bool = True) -> None:
        vars_ = self._sine_widget_vars[index]
        ch = self._sine_channels[index]
        try:
            amplitude = float(vars_["amplitude"].get())
            frequency = float(vars_["frequency"].get())
            decay = float(vars_["decay"].get())
            time_shift = float(vars_["time_shift"].get())
        except (tk.TclError, ValueError):
            self._validate_sine_entry_state(index)
            return  # ignore transient invalid states while typing
        if amplitude < 0 or frequency <= 0 or decay < 0 or not np.all(np.isfinite([amplitude, frequency, decay, time_shift])):
            self._validate_sine_entry_state(index)
            return
        ch["amplitude"] = amplitude
        ch["frequency"] = frequency
        ch["decay"] = decay
        ch["time_shift"] = time_shift
        ch["enabled"] = bool(vars_["enabled"].get())
        self._validate_sine_entry_state(index)
        if redraw:
            self._redraw_plot()

    def _on_noise_widget_changed(self, *, redraw: bool = True) -> None:
        vars_ = self._noise_widget_vars
        try:
            amplitude = float(vars_["amplitude"].get())
            seed = int(vars_["seed"].get())
        except (tk.TclError, ValueError):
            self._validate_noise_entry_state()
            return
        if amplitude < 0 or not np.isfinite(amplitude) or seed < 0:
            self._validate_noise_entry_state()
            return
        self._noise_channel["amplitude"] = amplitude
        self._noise_channel["seed"] = seed
        self._noise_channel["distribution"] = vars_["distribution"].get()
        self._noise_channel["enabled"] = bool(vars_["enabled"].get())
        self._validate_noise_entry_state()
        if redraw:
            self._redraw_plot()

    # -- Apply ---------------------------------------------------------------

    def _on_apply_clicked(self) -> None:
        # Clicking a button need not move focus out of the edited Entry.
        # Commit valid pending edits, then redraw only once.
        if hasattr(self, "_duration_var"):
            self._on_common_changed(redraw=False)
        for index in range(len(self._sine_widget_vars)):
            self._on_sine_widget_changed(index, redraw=False)
        if self._noise_widget_vars:
            self._on_noise_widget_changed(redraw=False)

        self._applied_common = copy.deepcopy(self._common)
        self._applied_sine_channels = copy.deepcopy(self._sine_channels)
        self._applied_noise_channel = copy.deepcopy(self._noise_channel)

        if self._ax is not None and self._canvas_widget is not None:
            self._redraw_plot()

        if getattr(self, "_status_item", None) is not None:
            self.canvas.itemconfig(self._status_item, text=self._status_text())

        if self._request_downstream is not None:
            self._request_downstream(self.node_id)