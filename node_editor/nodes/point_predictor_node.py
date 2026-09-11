# node_editor/nodes/point_predictor_node.py

import tkinter as tk
from tkinter import ttk
import numpy as np

from node_editor.base_node import BaseNode
from node_editor.pin_types import PinSchema, PinDef, PinType
from node_editor.execution import ExecutionMode


# ---------------------------------------------------------------------------
# Help text
# ---------------------------------------------------------------------------

_HELP_TEXT = """\
Point Predictor Node
====================

PURPOSE
-------
Maintains a rolling history of N×2 point observations and at
each call outputs a prediction of where those points will be
at the NEXT time step.

Temporal contract:
  call t:  receive xi(t)  →  output xi_guess(t+1)
  call 0:  no history     →  xi_guess(1) = xi(0)   (fallback)
  call 1:  1 step history →  xi_guess(2) = linear if possible
  call t≥2: full history  →  uses selected algorithm

Intended use: feed xi_guess(t+1) into a FeedbackBuffer whose
output drives the prevPts pin of an OpticalFlowLK node.  This
gives the tracker a better starting search position, reducing
drift for fast-moving or noisy point tracks.

INPUT PINS
----------
points   ARRAY  (required)
         Shape (N, 2) float64.  N tracked 2D image points.
         If N changes between calls, history is cleared and
         the node restarts from the new N.

reset    TRIGGER  (optional)
         Any new trigger value clears the history buffer and
         resets all predictor state.  Connect to the LoopNode
         advance pin or a TriggerSourceNode to reset between
         batch runs.

OUTPUT PINS
-----------
predicted   ARRAY
         Shape (N, 2) float64.  Predicted positions for the
         NEXT time step.

confidence  ARRAY
         Shape (N,) float64, values in [0, 1].
         Indicates how reliable each point's prediction is.
           0.0  no history, output is just the input
           0.5  one prior step, linear fallback used
           1.0  sufficient history for the chosen algorithm
         Downstream nodes can use this to filter unreliable
         predictions before passing to OpticalFlowLK.

PREDICTION ALGORITHMS
---------------------
All algorithms degrade gracefully when history is short:
  0 prior steps  →  xi_guess = xi(t)          (confidence 0.0)
  1 prior step   →  xi_guess = linear          (confidence 0.5)
  k prior steps  →  uses selected algorithm    (confidence 1.0)

Last known (order 0)
  xi_guess(t+1) = xi(t)
  Always confidence 0.5 or below.  Useful as a sanity baseline.

Linear extrapolation (order 1)  [default]
  xi_guess(t+1) = 2·xi(t) - xi(t-1)
  Assumes constant velocity.  Best for smooth, slow motion.

Polynomial extrapolation (degree d)
  Fits a degree-d polynomial to the last (d+1) observations
  and evaluates it one step ahead.  d=1 is identical to
  linear.  d=2 adds constant acceleration.  d=3 adds jerk.
  Caution: higher degrees amplify noise.  Rarely better than
  linear for noisy tracking data.

Exponential velocity smoothing
  Smooths the per-frame velocity with an IIR filter:
    v̂(t) = α·(xi(t)-xi(t-1)) + (1-α)·v̂(t-1)
    xi_guess(t+1) = xi(t) + v̂(t)
  α=1.0 reduces to linear.  α→0 freezes velocity at the
  first estimated value.  Typical: 0.3 – 0.7.

Kalman filter  (constant-velocity model, pure NumPy)
  State: [x, y, vx, vy].  Predicts one step with a constant-
  velocity dynamic model.  Measurement model: observes [x, y].
  Parameters:
    Process noise Q_pos  — position process noise variance
    Process noise Q_vel  — velocity process noise variance
    Measurement noise R  — observation noise variance
  One filter per point; state resets when history is cleared.
  Future enhancement: allow per-point R estimated from the
  OF status/error array rather than a fixed scalar.
  Future enhancement: use cv2.KalmanFilter for GPU-accelerated
  large-N tracking (e.g. thousands of points).

INSPECTOR SETTINGS
------------------
Algorithm         Select from the five methods above.
Poly degree       Degree for polynomial extrapolation (1–5).
EMA alpha         Decay for exponential smoothing (0–1).
Kalman Q_pos      Position process noise (default 1e-2).
Kalman Q_vel      Velocity process noise (default 1e-3).
Kalman R          Measurement noise (default 1e-1).
Max history       Maximum number of past steps to retain.
                  Older steps beyond this are discarded.
Preview point     Index of the point to show in the
                  time-series plot (0-based).
Reset button      Clears all history immediately.

KEYBOARD SHORTCUT
-----------------
Ctrl-H / Ctrl-h  — show this help window.
"""


# ---------------------------------------------------------------------------
# Predictor implementations
# ---------------------------------------------------------------------------

class _PredictorBase:
    """Abstract interface for one-step-ahead predictors."""

    def reset(self) -> None: ...
    def update_and_predict(
        self, xt: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        Receive current observation xt (N,2) and return
        (prediction (N,2), confidence float 0..1).
        """
        ...


class _LastKnownPredictor(_PredictorBase):
    """Order-0: repeat last observation."""

    def __init__(self) -> None:
        self._last: np.ndarray | None = None

    def reset(self) -> None:
        self._last = None

    def update_and_predict(
        self, xt: np.ndarray
    ) -> tuple[np.ndarray, float]:
        pred = xt.copy()
        self._last = xt.copy()
        return pred, 0.0


class _HistoryPredictor(_PredictorBase):
    """
    Base for predictors that keep a rolling history buffer.
    History shape: (T, N, 2), most recent last.
    """

    def __init__(self, max_history: int) -> None:
        self._max_history = max_history
        self._history: list[np.ndarray] = []

    def reset(self) -> None:
        self._history.clear()

    def _push(self, xt: np.ndarray) -> None:
        self._history.append(xt.copy())
        if len(self._history) > self._max_history:
            self._history.pop(0)

    def _t(self, idx: int) -> np.ndarray:
        """history[-1] is most recent."""
        return self._history[idx]

    @property
    def _n(self) -> int:
        return len(self._history)


class _LinearPredictor(_HistoryPredictor):
    """xi_guess(t+1) = 2·xi(t) - xi(t-1)."""

    def update_and_predict(
        self, xt: np.ndarray
    ) -> tuple[np.ndarray, float]:
        self._push(xt)
        if self._n < 2:
            return xt.copy(), 0.5
        pred = 2.0 * self._t(-1) - self._t(-2)
        return pred, 1.0


class _PolyPredictor(_HistoryPredictor):
    """
    Polynomial extrapolation of degree d.
    Falls back to lower degree when history is short.
    """

    def __init__(self, max_history: int,
                 degree: int) -> None:
        super().__init__(max_history)
        self._degree = degree

    def update_and_predict(
        self, xt: np.ndarray
    ) -> tuple[np.ndarray, float]:
        self._push(xt)
        n_steps = self._n
        if n_steps < 2:
            return xt.copy(), 0.0

        # effective degree capped by available history
        d = min(self._degree, n_steps - 1)
        T = n_steps
        ts = np.arange(T, dtype=np.float64)
        t_pred = float(T)

        hist = np.stack(self._history, axis=0)  # (T, N, 2)
        N = hist.shape[1]
        pred = np.empty((N, 2), dtype=np.float64)

        for dim in range(2):
            for pt in range(N):
                vals = hist[:, pt, dim]
                coeffs = np.polyfit(ts, vals, d)
                pred[pt, dim] = np.polyval(
                    coeffs, t_pred)

        conf = 1.0 if d == self._degree else 0.5
        return pred, conf


class _EMAPredictor(_HistoryPredictor):
    """Exponential moving average of velocity."""

    def __init__(self, max_history: int,
                 alpha: float) -> None:
        super().__init__(max_history)
        self._alpha = float(alpha)
        self._v_smooth: np.ndarray | None = None

    def reset(self) -> None:
        super().reset()
        self._v_smooth = None

    def update_and_predict(
        self, xt: np.ndarray
    ) -> tuple[np.ndarray, float]:
        self._push(xt)
        if self._n < 2:
            self._v_smooth = None
            return xt.copy(), 0.0

        v_raw = self._t(-1) - self._t(-2)
        if self._v_smooth is None:
            self._v_smooth = v_raw.copy()
        else:
            self._v_smooth = (
                self._alpha * v_raw
                + (1.0 - self._alpha) * self._v_smooth)

        pred = self._t(-1) + self._v_smooth
        return pred, 1.0


class _KalmanPredictor(_PredictorBase):
    """
    Constant-velocity Kalman filter, pure NumPy.
    State per point: [x, y, vx, vy].
    Observation: [x, y].

    Transition:  F = [[1,0,1,0],
                       [0,1,0,1],
                       [0,0,1,0],
                       [0,0,0,1]]
    Observation: H = [[1,0,0,0],
                       [0,1,0,0]]

    Future enhancement: replace with cv2.KalmanFilter for
    large N, or add per-point measurement noise estimated
    from the OF error array.
    """

    def __init__(self, q_pos: float,
                 q_vel: float, r: float) -> None:
        self._q_pos = q_pos
        self._q_vel = q_vel
        self._r     = r

        # per-point state (initialised on first observation)
        self._x: np.ndarray | None = None   # (N, 4)
        self._P: np.ndarray | None = None   # (N, 4, 4)

        # constant matrices
        self._F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)
        self._H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)
        self._call_count = 0

    def _build_q(self) -> np.ndarray:
        q = np.zeros((4, 4), dtype=np.float64)
        q[0, 0] = self._q_pos
        q[1, 1] = self._q_pos
        q[2, 2] = self._q_vel
        q[3, 3] = self._q_vel
        return q

    def _build_r(self) -> np.ndarray:
        return np.eye(2, dtype=np.float64) * self._r

    def reset(self) -> None:
        self._x = None
        self._P = None
        self._call_count = 0

    def update_and_predict(
        self, xt: np.ndarray
    ) -> tuple[np.ndarray, float]:
        N = xt.shape[0]
        transition_matrix = self._F
        observation_matrix = self._H
        process_noise = self._build_q()
        measurement_noise = self._build_r()

        # initialise state on first call or after reset
        if self._x is None or self._x.shape[0] != N:
            self._x = np.zeros(
                (N, 4), dtype=np.float64)
            self._x[:, :2] = xt
            self._P = np.stack(
                [np.eye(4, dtype=np.float64)] * N)
            self._call_count = 0

        self._call_count += 1

        # --- update step (incorporate observation xt) ---
        z = xt   # (N, 2)
        hx = (observation_matrix @ self._x.T).T   # (N, 2)
        innovation = z - hx                       # innovation

        # s = h p h^T + r  (same for all points, p differs)
        # k = p h^T s^-1
        s_inv_list = []
        k_list = []
        for i in range(N):
            s = observation_matrix @ self._P[i] @ observation_matrix.T + measurement_noise
            k = self._P[i] @ observation_matrix.T @ np.linalg.inv(s)
            k_list.append(k)
            s_inv_list.append(np.linalg.inv(s))

        k_arr = np.stack(k_list)   # (N,4,2)
        self._x = self._x + (
            k_arr @ innovation[:, :, np.newaxis]
        )[:, :, 0]
        identity_4 = np.eye(4, dtype=np.float64)
        self._P = np.stack([
            (identity_4 - k_arr[i] @ observation_matrix) @ self._P[i]
            for i in range(N)])

        # --- predict step (estimate state at t+1) ---
        x_pred = (transition_matrix @ self._x.T).T           # (N,4)
        p_pred = np.stack([
            transition_matrix @ self._P[i] @ transition_matrix.T + process_noise
            for i in range(N)])

        self._x = x_pred
        self._P = p_pred

        pred = x_pred[:, :2]
        conf = min(1.0,
                   self._call_count / 5.0)
        return pred, conf


# ---------------------------------------------------------------------------
# PointPredictorNode
# ---------------------------------------------------------------------------

class PointPredictorNode(BaseNode):
    """
    Sequential point predictor: given N×2 observations over time,
    outputs a prediction for the next time step.

    See _HELP_TEXT for full documentation.
    """

    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE      = "point_predictor"
    DISPLAY_NAME   = "Point Predictor"
    CATEGORY       = "process"
    NODE_WIDTH     = 210
    NODE_HEIGHT    = 120

    HELP_TEXT = _HELP_TEXT

    _BODY_BG   = "#f0f4ff"
    _OUTLINE   = "#5566bb"
    _TITLE_FG  = "#223388"
    _STATUS_FG = "#445599"

    _ALGO_NAMES = [
        "Last known (order 0)",
        "Linear extrapolation (order 1)",
        "Polynomial extrapolation",
        "Exponential smoothing (EMA)",
        "Kalman filter (constant velocity)",
    ]

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[
                PinDef("points", PinType.ARRAY,
                       "pts (N×2)", optional=False),
                PinDef("reset",  PinType.TRIGGER,
                       "reset",   optional=True),
            ],
            outputs=[
                PinDef("predicted",  PinType.ARRAY,
                       "pred (N×2)"),
                PinDef("confidence", PinType.ARRAY,
                       "conf (N,)"),
            ]
        )

    def get_help_text(self) -> str:
        return _HELP_TEXT

    # ── state ─────────────────────────────────────────────────────

    def _init_state(self) -> None:
        if hasattr(self, "_algo_var"):
            return

        self._algo_var      = tk.StringVar(
            value=self._ALGO_NAMES[1])
        self._poly_deg_var  = tk.IntVar(value=2)
        self._ema_alpha_var = tk.DoubleVar(value=0.5)
        self._kf_qpos_var   = tk.DoubleVar(value=1e-2)
        self._kf_qvel_var   = tk.DoubleVar(value=1e-3)
        self._kf_r_var      = tk.DoubleVar(value=1e-1)
        self._max_hist_var  = tk.IntVar(value=10)
        self._preview_pt_var = tk.IntVar(value=0)

        self._status_var  = tk.StringVar(
            value="waiting")
        self._info_var    = tk.StringVar(value="")

        self._last_trigger = None
        self._predictor: _PredictorBase | None = None
        self._last_n: int = -1

        # history for mini-plot (raw observations + predictions)
        self._plot_obs:  list[float] = []   # observed  coord of preview point
        self._plot_pred: list[float] = []   # predicted coord (for next step)
        self._plot_dim   = 0                # 0=x, 1=y

        # inspector refs
        self._plot_canvas: tk.Canvas | None = None
        self._help_popup:  tk.Toplevel | None = None

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

        info_lbl = tk.Label(
            self.canvas,
            textvariable=self._info_var,
            font=("Arial", 8),
            bg=self._BODY_BG, fg=self._TITLE_FG)
        self.canvas.create_window(
            x+w/2, y+h//2+4,
            window=info_lbl,
            tags=(self.node_id,))

        status_lbl = tk.Label(
            self.canvas,
            textvariable=self._status_var,
            font=("Arial", 7),
            bg=self._BODY_BG, fg=self._STATUS_FG,
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
            for seq in ("<Control-h>", "<Control-H>"):
                win.bind(seq,
                         lambda e: self._open_help())

        pad = {"padx": 8, "pady": 4}

        panes = tk.PanedWindow(
            parent, orient=tk.HORIZONTAL,
            sashwidth=5, showhandle=True)
        panes.pack(fill="both", expand=True)

        left  = tk.Frame(panes, width=340)
        right = tk.Frame(panes, width=420,
                          bg="#ffffff")
        left.pack_propagate(False)
        right.pack_propagate(False)
        panes.add(left,  stretch="always")
        panes.add(right, stretch="always")

        # ── left panel: settings ──────────────────────────────────

        # algorithm selection
        algo_frame = tk.LabelFrame(
            left, text="Prediction algorithm",
            font=("Arial", 9), **pad)
        algo_frame.pack(fill="x", **pad)

        for name in self._ALGO_NAMES:
            tk.Radiobutton(
                algo_frame, text=name,
                variable=self._algo_var,
                value=name,
                font=("Arial", 9),
                command=self._on_algo_change,
                anchor="w").pack(
                fill="x", pady=1)

        # algorithm parameters
        param_frame = tk.LabelFrame(
            left, text="Algorithm parameters",
            font=("Arial", 9), **pad)
        param_frame.pack(fill="x", **pad)

        def _param_row(lbl, var, from_=0.0,
                        to=10.0, inc=1.0,
                        fmt="%.4g",
                        frame=param_frame):
            r = tk.Frame(frame)
            r.pack(fill="x", pady=2)
            tk.Label(
                r, text=lbl, font=("Arial", 9),
                width=22, anchor="w").pack(
                side="left")
            sb = tk.Spinbox(
                r, from_=from_, to=to,
                increment=inc,
                textvariable=var,
                width=10, font=("Arial", 9),
                format=fmt)
            sb.pack(side="left")
            return sb

        self._sb_poly = _param_row(
            "Poly degree (1–5):",
            self._poly_deg_var,
            from_=1, to=5, inc=1, fmt="%.0f")
        self._sb_alpha = _param_row(
            "EMA alpha (0–1):",
            self._ema_alpha_var,
            from_=0.0, to=1.0, inc=0.05,
            fmt="%.2f")
        self._sb_qpos = _param_row(
            "Kalman Q_pos:",
            self._kf_qpos_var,
            from_=1e-6, to=1.0, inc=1e-3,
            fmt="%.6f")
        self._sb_qvel = _param_row(
            "Kalman Q_vel:",
            self._kf_qvel_var,
            from_=1e-6, to=1.0, inc=1e-4,
            fmt="%.6f")
        self._sb_r = _param_row(
            "Kalman R (meas. noise):",
            self._kf_r_var,
            from_=1e-6, to=10.0, inc=0.01,
            fmt="%.6f")

        # general settings
        gen_frame = tk.LabelFrame(
            left, text="General settings",
            font=("Arial", 9), **pad)
        gen_frame.pack(fill="x", **pad)

        _param_row(
            "Max history steps:",
            self._max_hist_var,
            from_=2, to=1000, inc=1, fmt="%.0f",
            frame=gen_frame)

        # reset
        rst_frame = tk.Frame(left)
        rst_frame.pack(fill="x", **pad)

        tk.Button(
            rst_frame, text="Reset history",
            font=("Arial", 9),
            bg="#882222", fg="white",
            activebackground="#aa3333",
            relief=tk.FLAT, padx=8, pady=3,
            command=self._on_reset_btn).pack(
            side="left")

        tk.Button(
            rst_frame,
            text="Help  (Ctrl-H)",
            font=("Arial", 8),
            command=self._open_help).pack(
            side="right")

        # status
        tk.Label(
            left,
            textvariable=self._status_var,
            font=("Arial", 9), fg="#445599",
            anchor="w", justify="left").pack(
            fill="x", **pad)

        # ── right panel: time-series preview ──────────────────────

        plot_frame = tk.LabelFrame(
            right, text="Time-series preview",
            font=("Arial", 9), **pad)
        plot_frame.pack(
            fill="both", expand=True, **pad)

        ctrl_row = tk.Frame(plot_frame)
        ctrl_row.pack(fill="x", pady=(0, 4))

        tk.Label(
            ctrl_row,
            text="Preview point index:",
            font=("Arial", 9)).pack(side="left")
        tk.Spinbox(
            ctrl_row,
            from_=0, to=9999,
            textvariable=self._preview_pt_var,
            width=6, font=("Arial", 9),
            command=self._refresh_plot).pack(
            side="left", padx=4)

        dim_frame = tk.Frame(ctrl_row)
        dim_frame.pack(side="left", padx=8)
        self._dim_var = tk.IntVar(value=0)
        tk.Radiobutton(
            dim_frame, text="x",
            variable=self._dim_var, value=0,
            font=("Arial", 9),
            command=self._refresh_plot).pack(
            side="left")
        tk.Radiobutton(
            dim_frame, text="y",
            variable=self._dim_var, value=1,
            font=("Arial", 9),
            command=self._refresh_plot).pack(
            side="left")

        legend_row = tk.Frame(plot_frame)
        legend_row.pack(fill="x", pady=(0, 2))
        for color, label in [
            ("#1155cc", "● observed"),
            ("#cc4400", "● predicted"),
        ]:
            tk.Label(
                legend_row, text=label,
                font=("Arial", 8), fg=color).pack(
                side="left", padx=6)

        self._plot_canvas = tk.Canvas(
            plot_frame, bg="#ffffff",
            highlightthickness=1,
            highlightbackground="#cccccc")
        self._plot_canvas.pack(
            fill="both", expand=True)
        self._plot_canvas.bind(
            "<Configure>",
            lambda _e: self._refresh_plot())

        if win is not None:
            win.update_idletasks()
            win.minsize(800, 520)
            win.geometry("820x540")

        self._on_algo_change()
        self._refresh_plot()

    # ── inspector helpers ─────────────────────────────────────────

    def _on_algo_change(self) -> None:
        """Show/hide parameter widgets for the selected algo."""
        algo = self._algo_var.get()
        poly = "Polynomial" in algo
        ema  = "EMA" in algo or "Exponential" in algo
        kalm = "Kalman" in algo

        for sb, show in [
            (self._sb_poly,  poly),
            (self._sb_alpha, ema),
            (self._sb_qpos,  kalm),
            (self._sb_qvel,  kalm),
            (self._sb_r,     kalm),
        ]:
            if not hasattr(self, "_sb_poly"):
                break
            try:
                sb.master.pack_configure(
                    expand=show)
                if show:
                    sb.master.pack(
                        fill="x", pady=2)
                else:
                    sb.master.pack_forget()
            except Exception:
                pass

    def _on_reset_btn(self) -> None:
        self._do_reset()
        if self._request_downstream:
            self._request_downstream(
                self.node_id)

    def _do_reset(self) -> None:
        if self._predictor is not None:
            self._predictor.reset()
        self._last_N  = -1
        self._plot_obs.clear()
        self._plot_pred.clear()
        self._status_var.set("reset")
        self._info_var.set("")
        self._refresh_plot()

    # ── time-series plot ──────────────────────────────────────────

    def _refresh_plot(self) -> None:
        c = self._plot_canvas
        if c is None or not c.winfo_exists():
            return
        c.delete("all")

        cw = c.winfo_width()
        ch = c.winfo_height()
        if cw < 10 or ch < 10:
            return

        obs  = self._plot_obs
        pred = self._plot_pred
        if not obs:
            c.create_text(
                cw/2, ch/2,
                text="No data yet — run the pipeline",
                fill="#888888", font=("Arial", 9))
            return

        dim   = self._dim_var.get() \
            if hasattr(self, "_dim_var") else 0
        pt_idx = self._preview_pt_var.get()
        dim_label = "x" if dim == 0 else "y"

        # obs and pred are lists of (N,2) arrays
        obs_vals  = []
        pred_vals = []
        for arr in obs:
            if (isinstance(arr, np.ndarray)
                    and arr.ndim == 2
                    and pt_idx < arr.shape[0]):
                obs_vals.append(
                    float(arr[pt_idx, dim]))
        for arr in pred:
            if (isinstance(arr, np.ndarray)
                    and arr.ndim == 2
                    and pt_idx < arr.shape[0]):
                pred_vals.append(
                    float(arr[pt_idx, dim]))

        if not obs_vals:
            c.create_text(
                cw/2, ch/2,
                text=f"Point {pt_idx} not in data",
                fill="#888888", font=("Arial", 9))
            return

        all_vals = obs_vals + pred_vals
        vmin = min(all_vals)
        vmax = max(all_vals)
        vrange = vmax - vmin
        if vrange < 1e-9:
            vrange = 1.0

        pad_x, pad_y = 40, 20
        w_plot = cw - 2 * pad_x
        h_plot = ch - 2 * pad_y

        def _to_screen(t, v):
            n_total = max(
                len(obs_vals),
                len(pred_vals)) + 1
            sx = pad_x + t / max(n_total - 1, 1) \
                 * w_plot
            sy = pad_y + (
                1.0 - (v - vmin) / vrange
            ) * h_plot
            return sx, sy

        # draw axes
        c.create_line(
            pad_x, pad_y,
            pad_x, ch - pad_y,
            fill="#cccccc", width=1)
        c.create_line(
            pad_x, ch - pad_y,
            cw - pad_x, ch - pad_y,
            fill="#cccccc", width=1)

        # y-axis labels
        for frac in [0.0, 0.5, 1.0]:
            v = vmin + frac * vrange
            sy = pad_y + (1.0 - frac) * h_plot
            c.create_text(
                pad_x - 4, sy,
                text=f"{v:.1f}",
                anchor="e",
                font=("Arial", 7),
                fill="#666666")

        # x-axis title
        c.create_text(
            cw/2, ch - 4,
            text=f"step  (coord: {dim_label},"
                 f"  point {pt_idx})",
            font=("Arial", 8), fill="#666666")

        # observed series — blue
        if len(obs_vals) >= 2:
            pts_screen = [
                _to_screen(t, v)
                for t, v in enumerate(obs_vals)]
            flat = [
                coord
                for p in pts_screen
                for coord in p]
            c.create_line(
                *flat, fill="#1155cc",
                width=1.5, smooth=False)
        for t, v in enumerate(obs_vals):
            sx, sy = _to_screen(t, v)
            r = 3
            c.create_oval(
                sx-r, sy-r, sx+r, sy+r,
                fill="#1155cc", outline="")

        # predicted series — red/orange, shifted +1
        if len(pred_vals) >= 2:
            pts_screen = [
                _to_screen(t + 1, v)
                for t, v in enumerate(pred_vals)]
            flat = [
                coord
                for p in pts_screen
                for coord in p]
            c.create_line(
                *flat, fill="#cc4400",
                width=1.5, dash=(4, 3),
                smooth=False)
        for t, v in enumerate(pred_vals):
            sx, sy = _to_screen(t + 1, v)
            r = 3
            c.create_oval(
                sx-r, sy-r, sx+r, sy+r,
                fill="#cc4400", outline="")

    # ── help popup ────────────────────────────────────────────────

    def _open_help(self) -> None:
        if (self._help_popup is not None
                and self._help_popup.winfo_exists()):
            self._help_popup.lift()
            return

        popup = tk.Toplevel()
        popup.title("Point Predictor — Help")
        popup.geometry("700x580")
        popup.resizable(True, True)
        try:
            px, py = self.canvas.winfo_pointerxy()
            popup.geometry(f"+{px+16}+{py+16}")
        except Exception:
            pass

        body = tk.Frame(
            popup, bg="#f8f8f8", padx=10, pady=8)
        body.pack(fill="both", expand=True)

        txt = tk.Text(
            body, font=("Courier", 9),
            bg="#f8f8f8", fg="#222222",
            wrap=tk.WORD, relief=tk.FLAT)
        vsb = ttk.Scrollbar(
            body, orient="vertical",
            command=txt.yview)
        txt.configure(yscrollcommand=vsb.set)
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
            "WM_DELETE_WINDOW", popup.destroy)
        self._help_popup = popup

    # ── predictor factory ─────────────────────────────────────────

    def _build_predictor(self) -> _PredictorBase:
        algo = self._algo_var.get()
        mh   = max(2, self._max_hist_var.get())

        if "Last known" in algo:
            return _LastKnownPredictor()
        if "Linear" in algo:
            return _LinearPredictor(mh)
        if "Polynomial" in algo:
            return _PolyPredictor(
                mh, self._poly_deg_var.get())
        if "EMA" in algo or "Exponential" in algo:
            return _EMAPredictor(
                mh, self._ema_alpha_var.get())
        if "Kalman" in algo:
            return _KalmanPredictor(
                self._kf_qpos_var.get(),
                self._kf_qvel_var.get(),
                self._kf_r_var.get())
        return _LinearPredictor(mh)

    # ── compute ───────────────────────────────────────────────────

    def compute(self, inputs: dict) -> dict:
        # handle reset trigger
        reset_val = inputs.get("reset")
        if (reset_val is not None
                and reset_val != self._last_trigger):
            self._last_trigger = reset_val
            self._do_reset()

        pts = inputs.get("points")
        if pts is None or not isinstance(
                pts, np.ndarray):
            self._status_var.set("no input")
            return {}

        pts = np.asarray(
            pts, dtype=np.float64)

        # handle non-standard shapes
        if pts.ndim == 3:
            pts = pts.reshape(-1, 2)
        if pts.ndim != 2 or pts.shape[1] != 2:
            self._status_var.set(
                f"bad shape {pts.shape}"
                f"  — expected (N,2)")
            return {}

        N = pts.shape[0]

        # reset on N change
        if N != self._last_n:
            if self._last_n != -1:
                self._do_reset()
                self._status_var.set(
                    f"N changed "
                    f"{self._last_n}→{N}, reset")
            self._last_n = N
            self._predictor = \
                self._build_predictor()

        # rebuild predictor if algorithm params changed
        # (simple approach: rebuild only when predictor
        #  type is None; user can press Reset to rebuild)
        if self._predictor is None:
            self._predictor = \
                self._build_predictor()

        pred, conf_scalar = \
            self._predictor.update_and_predict(pts)

        # per-point confidence array
        conf_arr = np.full(
            N, conf_scalar, dtype=np.float64)

        # update plot history
        self._plot_obs.append(pts.copy())
        self._plot_pred.append(pred.copy())
        max_hist = max(
            2, self._max_hist_var.get())
        if len(self._plot_obs) > max_hist + 5:
            self._plot_obs  = \
                self._plot_obs[-(max_hist+5):]
            self._plot_pred = \
                self._plot_pred[-(max_hist+5):]

        # update displays
        algo_short = (
            self._algo_var.get()
            .split("(")[0].strip()[:18])
        self._info_var.set(
            f"N={N}  conf={conf_scalar:.2f}")
        self._status_var.set(
            f"{algo_short}  "
            f"hist={len(self._plot_obs)}")

        if self.is_inspector_open():
            self._refresh_plot()

        return {
            "predicted":  pred,
            "confidence": conf_arr,
        }

    # ── serialization ─────────────────────────────────────────────

    def get_params(self) -> dict:
        return {
            "algo":       self._algo_var.get(),
            "poly_deg":   self._poly_deg_var.get(),
            "ema_alpha":  self._ema_alpha_var.get(),
            "kf_qpos":    self._kf_qpos_var.get(),
            "kf_qvel":    self._kf_qvel_var.get(),
            "kf_r":       self._kf_r_var.get(),
            "max_hist":   self._max_hist_var.get(),
            "preview_pt": self._preview_pt_var.get(),
        }

    def set_params(self, params: dict) -> None:
        self._init_state()
        self._algo_var.set(
            params.get("algo",
                       self._ALGO_NAMES[1]))
        self._poly_deg_var.set(
            int(params.get("poly_deg", 2)))
        self._ema_alpha_var.set(
            float(params.get("ema_alpha", 0.5)))
        self._kf_qpos_var.set(
            float(params.get("kf_qpos", 1e-2)))
        self._kf_qvel_var.set(
            float(params.get("kf_qvel", 1e-3)))
        self._kf_r_var.set(
            float(params.get("kf_r", 1e-1)))
        self._max_hist_var.set(
            int(params.get("max_hist", 10)))
        self._preview_pt_var.set(
            int(params.get("preview_pt", 0)))

    def close_inspector(self) -> None:
        super().close_inspector()
        self._plot_canvas = None

    def on_destroy(self) -> None:
        if (self._help_popup is not None
                and self._help_popup.winfo_exists()):
            self._help_popup.destroy()
        super().on_destroy()