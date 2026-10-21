# node_editor/nodes/demo_hermite_interpolation.py
"""
Demo: cubic Hermite interpolation between two points picked with the mouse.

    H(t) = h00(t) P0 + h10(t) M0 + h01(t) P1 + h11(t) M1,   t in [0, 1]
    h00 = 2t^3 - 3t^2 + 1    h10 = t^3 - 2t^2 + t
    h01 = -2t^3 + 3t^2       h11 = t^3 - t^2

P0, P1 are the end points, M0, M1 the tangents (direction and speed of the
curve at each end). A tangent is set by an angle and a STRENGTH relative to
the distance |P1 - P0| (strength 1 = M has the length |P1 - P0|), so moving
an end point keeps the curve's shape. On the canvas each tangent is shown as
a handle at P0 + M0 / 3 and P1 - M1 / 3 -- exactly where the equivalent
Bezier control points are, so dragging a handle feels like a drawing program.
"""

import math
import tkinter as tk

import numpy as np

from node_editor.base_node import BaseNode
from node_editor.execution import ExecutionMode
from node_editor.pin_types import PinDef, PinSchema, PinType
from node_editor.tooltip import add_tooltip

P_COLORS = ("#d9480f", "#1971c2")          # P0 orange, P1 blue
CURVE_COLOR = "#2b8a3e"
GRID_STEPS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)


def hermite(p0, m0, p1, m1, n: int):
    """(t (n,), curve (n, 2)) of the cubic Hermite curve."""
    t = np.linspace(0.0, 1.0, max(2, int(n)))
    t2, t3 = t * t, t * t * t
    h00, h10 = 2 * t3 - 3 * t2 + 1, t3 - 2 * t2 + t
    h01, h11 = -2 * t3 + 3 * t2, t3 - t2
    p0, m0, p1, m1 = (np.asarray(v, dtype=np.float64) for v in (p0, m0, p1, m1))
    curve = (h00[:, None] * p0 + h10[:, None] * m0 + h01[:, None] * p1 + h11[:, None] * m1)
    return t, curve


class DemoHermiteInterpolationNode(BaseNode):
    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE = "demo_hermite_interpolation"
    DISPLAY_NAME = "Demo Hermite Interpolation"
    CATEGORY = "source"
    SEARCH_KEYWORDS = ("hermite", "interpolation", "spline", "curve", "tangent", "demo", "cubic")
    NODE_WIDTH = 200
    NODE_HEIGHT = 90

    HELP_TEXT = (
        "Demo Hermite Interpolation\n\n"
        "1. Click twice on the plane: the first click places P0 (start, orange),\n"
        "   the second P1 (end, blue). Drag a point to move it.\n"
        "2. Shape the curve: drag the small square handles, or use the sliders.\n"
        "   Each end has an ANGLE (direction the curve leaves P0 / arrives at P1)\n"
        "   and a STRENGTH (how far that direction pulls the curve; relative to\n"
        "   the distance P0-P1, 1 = normal, 0 = no pull, larger = wider bend).\n"
        "3. Choose how many points to output.\n\n"
        "Outputs: curve (N, 2) x, y of the interpolated points (P0 first, P1\n"
        "last) and t (N,) their parameter 0 ... 1.\n\n"
        "Mouse: wheel = zoom, right-drag = pan, 'Fit view' = show both points.\n\n"
        "Formula: H(t) = h00 P0 + h10 M0 + h01 P1 + h11 M1 with\n"
        "h00 = 2t^3-3t^2+1, h10 = t^3-2t^2+t, h01 = -2t^3+3t^2, h11 = t^3-t^2.\n"
    )

    def __init__(self, node_id: str, canvas: tk.Canvas):
        super().__init__(node_id, canvas)
        self._pts: list = [None, None]                        # P0, P1 as [x, y] or None
        self._angle_vars = [tk.DoubleVar(value=0.0), tk.DoubleVar(value=0.0)]       # degrees
        self._strength_vars = [tk.DoubleVar(value=1.0), tk.DoubleVar(value=1.0)]
        self._n_var = tk.IntVar(value=50)
        self._show_samples_var = tk.BooleanVar(value=True)
        self._show_handles_var = tk.BooleanVar(value=True)
        self._coord_vars = [[tk.StringVar(), tk.StringVar()] for _ in range(2)]
        self._mouse_var = tk.StringVar(value="")
        self._face_var = tk.StringVar(value="click 2 points\n(double-click to open)")
        self._view = {"cx": 0.0, "cy": 0.0, "scale": None}    # world centre, pixels per unit
        self._drag = None
        self._c = None
        self._syncing = False
        for var in self._angle_vars + self._strength_vars:
            var.trace_add("write", lambda *_a: self._on_shape_changed())
        self._n_var.trace_add("write", lambda *_a: self._on_shape_changed())

    # ── pins / output ──────────────────────────────────────────────────
    def get_pin_schema(self) -> PinSchema:
        return PinSchema(inputs=[], outputs=[
            PinDef("curve", PinType.ARRAY, "curve", shape=(-1, 2), dtype="float64"),
            PinDef("t", PinType.ARRAY, "t", shape=(-1,), dtype="float64"),
        ])

    def _tangents(self):
        """M0, M1 from the angle / strength controls (strength x |P1 - P0|)."""
        p0, p1 = self._pts
        d = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        out = []
        for k in range(2):
            try:
                a, s = math.radians(float(self._angle_vars[k].get())), float(self._strength_vars[k].get())
            except (tk.TclError, ValueError):
                a, s = 0.0, 1.0
            out.append((s * d * math.cos(a), s * d * math.sin(a)))
        return out

    def _result(self):
        if self._pts[0] is None or self._pts[1] is None:
            return None
        m0, m1 = self._tangents()
        try:
            n = int(self._n_var.get())
        except (tk.TclError, ValueError):
            n = 50
        return hermite(self._pts[0], m0, self._pts[1], m1, min(max(n, 2), 100000))

    def compute(self, inputs: dict) -> dict:
        res = self._result()
        if res is None:
            self._face_var.set("click 2 points\n(double-click to open)")
            return {}
        t, curve = res
        self._face_var.set(f"curve: {len(t)} points\nlength {np.sum(np.hypot(*np.diff(curve, axis=0).T)):.3g}")
        return {"curve": curve, "t": t}

    def _push(self) -> None:
        if self._request_downstream is not None:
            self._request_downstream(self.node_id)
        else:
            self.compute({})

    # ── node face ──────────────────────────────────────────────────────
    def build_body(self) -> None:
        x, y, w, h = self.x, self.y, self.width, self.height
        self._body_rect = self.canvas.create_rectangle(
            x, y, x + w, y + h, fill="#eef7ee", outline="#2b8a3e", width=2, tags=(self.node_id, "node_body"))
        self._title_item = self.canvas.create_text(
            x + w / 2, y + 12, text=self.DISPLAY_NAME, font=("Arial", 9, "bold"), fill="#1e5e2b",
            tags=(self.node_id,))
        lbl = tk.Label(self.canvas, textvariable=self._face_var, font=("Arial", 8), bg="#eef7ee", fg="#1e5e2b")
        self.canvas.create_window(x + w / 2 - 12, y + h / 2 + 8, window=lbl, tags=(self.node_id,))
        self._canvas_items += [self._body_rect, self._title_item]

    # ── inspector ──────────────────────────────────────────────────────
    def build_inspector(self, parent: tk.Frame) -> None:
        win = self._inspector_win
        if win is not None:
            win.geometry("900x640")
            win.minsize(640, 460)

        left = tk.Frame(parent)
        left.pack(side="left", fill="both", expand=True)
        right = tk.Frame(parent, padx=8)
        right.pack(side="right", fill="y")

        # Canvas with a one-line instruction that tells what to do next.
        self._hint_var = tk.StringVar()
        tk.Label(left, textvariable=self._hint_var, font=("Arial", 10, "bold"), fg="#1e5e2b",
                 anchor="w").pack(fill="x")
        c = self._c = tk.Canvas(left, bg="#ffffff", highlightthickness=1, highlightbackground="#c8cdd8",
                                cursor="crosshair")
        c.pack(fill="both", expand=True)
        bottom = tk.Frame(left)
        bottom.pack(fill="x")
        tk.Label(bottom, textvariable=self._mouse_var, font=("Arial", 8), fg="#555555").pack(side="left")
        tk.Label(bottom, text="wheel = zoom, right-drag = pan", font=("Arial", 8), fg="#888888").pack(side="right")
        c.bind("<Configure>", lambda _e: self._draw())
        c.bind("<ButtonPress-1>", self._on_press)
        c.bind("<B1-Motion>", self._on_drag)
        c.bind("<ButtonRelease-1>", self._on_release)
        c.bind("<ButtonPress-3>", lambda e: setattr(self, "_pan", (e.x, e.y, self._view["cx"], self._view["cy"])))
        c.bind("<B3-Motion>", self._on_pan)
        c.bind("<Motion>", self._on_motion)
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            c.bind(seq, self._on_wheel)
        add_tooltip(c, "Click to place P0 (start), then P1 (end). Drag a point to move it. Drag a "
                       "small square handle to change the direction / strength of the curve at that "
                       "end. Wheel = zoom, right-drag = pan.")

        # End points (exact values).
        box = tk.LabelFrame(right, text="End points", font=("Arial", 9, "bold"), padx=6, pady=4)
        box.pack(fill="x", pady=(0, 6))
        for k, name in enumerate(("P0 (start)", "P1 (end)")):
            tk.Label(box, text=name, fg=P_COLORS[k], font=("Arial", 9, "bold")).grid(row=k, column=0, sticky="w")
            for j, axis in enumerate("xy"):
                tk.Label(box, text=f"{axis}:", font=("Arial", 8)).grid(row=k, column=1 + 2 * j, sticky="e")
                e = tk.Entry(box, textvariable=self._coord_vars[k][j], width=8, justify="right")
                e.grid(row=k, column=2 + 2 * j, padx=(0, 4), pady=1)
                e.bind("<Return>", lambda _e, kk=k: self._on_coord_entry(kk))
                e.bind("<FocusOut>", lambda _e, kk=k: self._on_coord_entry(kk))
                add_tooltip(e, f"Exact {axis} of {name.split()[0]}; type a number and press Enter. "
                               "Or drag the point on the plane.")
        clear = tk.Button(box, text="Clear points", font=("Arial", 8), command=self._on_clear)
        clear.grid(row=2, column=0, columnspan=5, sticky="w", pady=(4, 0))
        add_tooltip(clear, "Remove both points and start again by clicking on the plane.")

        # Tangents.
        tbox = tk.LabelFrame(right, text="Curve shape (tangents)", font=("Arial", 9, "bold"), padx=6, pady=4)
        tbox.pack(fill="x", pady=(0, 6))
        for k, (name, verb) in enumerate((("At P0", "leaves P0"), ("At P1", "arrives at P1"))):
            tk.Label(tbox, text=name, fg=P_COLORS[k], font=("Arial", 9, "bold")).pack(anchor="w")
            row = tk.Frame(tbox)
            row.pack(fill="x")
            lbl = tk.Label(row, text="angle", width=7, anchor="w", font=("Arial", 8))
            lbl.pack(side="left")
            sc = tk.Scale(row, from_=-180, to=180, resolution=1, orient=tk.HORIZONTAL, length=170,
                          variable=self._angle_vars[k], showvalue=True)
            sc.pack(side="left")
            tip = (f"Direction in which the curve {verb}, in degrees: 0 = right, 90 = up, 180 / -180 = "
                   "left, -90 = down. Same as dragging the handle around the point.")
            add_tooltip(lbl, tip)
            add_tooltip(sc, tip)
            row = tk.Frame(tbox)
            row.pack(fill="x")
            lbl = tk.Label(row, text="strength", width=7, anchor="w", font=("Arial", 8))
            lbl.pack(side="left")
            sc = tk.Scale(row, from_=0.0, to=4.0, resolution=0.05, orient=tk.HORIZONTAL, length=170,
                          variable=self._strength_vars[k], showvalue=True)
            sc.pack(side="left")
            tip = ("How strongly that direction pulls the curve, relative to the distance P0-P1: "
                   "0 = no pull (straight towards the other point), 1 = normal, larger = wider "
                   "bend / overshoot. Same as dragging the handle closer or farther.")
            add_tooltip(lbl, tip)
            add_tooltip(sc, tip)
        presets = tk.Frame(tbox)
        presets.pack(fill="x", pady=(4, 0))
        straight = tk.Button(presets, text="Straight line", font=("Arial", 8), command=self._preset_straight)
        straight.pack(side="left")
        s_curve = tk.Button(presets, text="S-curve", font=("Arial", 8), command=self._preset_s)
        s_curve.pack(side="left", padx=(4, 0))
        arc = tk.Button(presets, text="Arc", font=("Arial", 8), command=self._preset_arc)
        arc.pack(side="left", padx=(4, 0))
        add_tooltip(straight, "Both tangents point from P0 to P1 with strength 1: the curve is the "
                              "straight segment, evenly sampled.")
        add_tooltip(s_curve, "Tangents turned to opposite sides of the line P0-P1: an S shape.")
        add_tooltip(arc, "Tangents turned to the same side: a single bend like an arc.")

        # Output.
        obox = tk.LabelFrame(right, text="Output", font=("Arial", 9, "bold"), padx=6, pady=4)
        obox.pack(fill="x", pady=(0, 6))
        row = tk.Frame(obox)
        row.pack(fill="x")
        lbl = tk.Label(row, text="points", font=("Arial", 8))
        lbl.pack(side="left")
        sp = tk.Spinbox(row, from_=2, to=100000, increment=10, textvariable=self._n_var, width=8)
        sp.pack(side="left", padx=(4, 0))
        tip = ("How many points the 'curve' output has, evenly spaced in the parameter t from 0 (P0) "
               "to 1 (P1). Note: even steps in t are not always even distances along the curve.")
        add_tooltip(lbl, tip)
        add_tooltip(sp, tip)
        cb1 = tk.Checkbutton(obox, text="Show output points", variable=self._show_samples_var, command=self._draw)
        cb1.pack(anchor="w")
        add_tooltip(cb1, "Draw the N output points on the curve as small dots.")
        cb2 = tk.Checkbutton(obox, text="Show handles", variable=self._show_handles_var, command=self._draw)
        cb2.pack(anchor="w")
        add_tooltip(cb2, "Draw the tangent handles (square = drag to reshape, dashed line = direction).")
        fit = tk.Button(right, text="Fit view", font=("Arial", 8), command=self._fit_view)
        fit.pack(anchor="w")
        add_tooltip(fit, "Zoom / pan so that both points and the whole curve are visible.")
        self._info_var = tk.StringVar()
        tk.Label(right, textvariable=self._info_var, font=("Courier", 8), justify="left", anchor="w",
                 fg="#334455").pack(fill="x", pady=(8, 0))
        self._sync_coord_entries()
        self._draw()

    def close_inspector(self) -> None:
        self._c = None
        super().close_inspector()

    # ── coordinates ────────────────────────────────────────────────────
    def _scale(self) -> float:
        if self._view["scale"] is None and self._c is not None:
            self._view["scale"] = max(4.0, min(self._c.winfo_width(), self._c.winfo_height()) / 24.0)
        return self._view["scale"] or 20.0

    def _to_screen(self, X, Y):
        c, s = self._c, self._scale()
        return c.winfo_width() / 2 + (X - self._view["cx"]) * s, c.winfo_height() / 2 - (Y - self._view["cy"]) * s

    def _to_world(self, x, y):
        c, s = self._c, self._scale()
        return self._view["cx"] + (x - c.winfo_width() / 2) / s, self._view["cy"] - (y - c.winfo_height() / 2) / s

    def _handles(self):
        """Screen-independent handle positions: P0 + M0/3 and P1 - M1/3."""
        m0, m1 = self._tangents()
        p0, p1 = self._pts
        return (p0[0] + m0[0] / 3, p0[1] + m0[1] / 3), (p1[0] - m1[0] / 3, p1[1] - m1[1] / 3)

    # ── drawing ────────────────────────────────────────────────────────
    def _draw(self) -> None:
        c = self._c
        if c is None:
            return
        w, h = c.winfo_width(), c.winfo_height()
        if w < 20 or h < 20:
            return
        c.delete("all")
        s = self._scale()
        step = next((g for g in GRID_STEPS if g * s >= 40), GRID_STEPS[-1])
        x0, y1 = self._to_world(0, 0)
        x1, y0 = self._to_world(w, h)
        for gx in np.arange(math.floor(x0 / step) * step, x1 + step, step):
            X, _ = self._to_screen(gx, 0)
            axis = abs(gx) < step / 2
            c.create_line(X, 0, X, h, fill="#9aa3b5" if axis else "#eef0f4", width=1)
            if not axis:
                c.create_text(X + 2, h - 2, text=f"{gx:g}", anchor="sw", fill="#a0a6b2", font=("Arial", 7))
        for gy in np.arange(math.floor(y0 / step) * step, y1 + step, step):
            _, Y = self._to_screen(0, gy)
            axis = abs(gy) < step / 2
            c.create_line(0, Y, w, Y, fill="#9aa3b5" if axis else "#eef0f4", width=1)
            if not axis:
                c.create_text(2, Y - 2, text=f"{gy:g}", anchor="sw", fill="#a0a6b2", font=("Arial", 7))

        p0, p1 = self._pts
        res = self._result()
        if res is not None:
            t, curve = res
            m0, m1 = self._tangents()
            dense = hermite(p0, m0, p1, m1, 400)[1]               # smooth drawing, independent of N
            c.create_line(*[v for q in dense for v in self._to_screen(*q)], fill=CURVE_COLOR, width=3,
                          smooth=False, capstyle=tk.ROUND)
            if self._show_samples_var.get() and len(curve) <= 2000:
                for q in curve:
                    X, Y = self._to_screen(*q)
                    c.create_oval(X - 2, Y - 2, X + 2, Y + 2, fill="#ffffff", outline=CURVE_COLOR)
            if self._show_handles_var.get():
                for k, (hx, hy) in enumerate(self._handles()):
                    PX, PY = self._to_screen(*self._pts[k])
                    HX, HY = self._to_screen(hx, hy)
                    c.create_line(PX, PY, HX, HY, fill=P_COLORS[k], dash=(4, 3), width=1)
                    c.create_rectangle(HX - 5, HY - 5, HX + 5, HY + 5, fill="#ffffff", outline=P_COLORS[k], width=2)
            length = float(np.sum(np.hypot(*np.diff(curve, axis=0).T)))
            self._info_var.set(f"output: {len(t)} points\ncurve length {length:.4g}\n"
                               f"P0 = ({p0[0]:.4g}, {p0[1]:.4g})\nP1 = ({p1[0]:.4g}, {p1[1]:.4g})")
        else:
            self._info_var.set("")
        for k in range(2):
            if self._pts[k] is not None:
                X, Y = self._to_screen(*self._pts[k])
                c.create_oval(X - 7, Y - 7, X + 7, Y + 7, fill=P_COLORS[k], outline="#ffffff", width=2)
                c.create_text(X + 10, Y - 10, text=f"P{k}", anchor="sw", fill=P_COLORS[k], font=("Arial", 10, "bold"))
        self._hint_var.set(
            "Step 1: click on the plane to place P0 (start)" if p0 is None else
            "Step 2: click to place P1 (end)" if p1 is None else
            "Drag points or the square handles to shape the curve")

    # ── mouse ──────────────────────────────────────────────────────────
    def _hit(self, x, y):
        """What is under the mouse: ('pt', k), ('handle', k) or None."""
        if self._pts[0] is not None and self._pts[1] is not None and self._show_handles_var.get():
            for k, hpos in enumerate(self._handles()):
                HX, HY = self._to_screen(*hpos)
                if (HX - x) ** 2 + (HY - y) ** 2 <= 8 ** 2:
                    return ("handle", k)
        for k in range(2):
            if self._pts[k] is not None:
                X, Y = self._to_screen(*self._pts[k])
                if (X - x) ** 2 + (Y - y) ** 2 <= 10 ** 2:
                    return ("pt", k)
        return None

    def _on_press(self, e) -> None:
        hit = self._hit(e.x, e.y)
        if hit is not None:
            self._drag = hit
            return
        for k in range(2):
            if self._pts[k] is None:
                self._pts[k] = list(self._to_world(e.x, e.y))
                if k == 1:
                    self._preset_straight(push=False)
                self._drag = ("pt", k)
                self._after_change()
                return

    def _on_drag(self, e) -> None:
        if self._drag is None:
            return
        kind, k = self._drag
        X, Y = self._to_world(e.x, e.y)
        if kind == "pt":
            self._pts[k] = [X, Y]
        else:
            p = self._pts[k]
            d = math.hypot(self._pts[1][0] - self._pts[0][0], self._pts[1][1] - self._pts[0][1])
            # handle = P0 + M0/3 or P1 - M1/3  ->  M = +-3 (handle - P)
            mx, my = (3 * (X - p[0]), 3 * (Y - p[1])) if k == 0 else (3 * (p[0] - X), 3 * (p[1] - Y))
            self._syncing = True
            try:
                self._angle_vars[k].set(round(math.degrees(math.atan2(my, mx)), 1))
                self._strength_vars[k].set(round(math.hypot(mx, my) / d, 3) if d > 1e-12 else 0.0)
            finally:
                self._syncing = False
        self._sync_coord_entries()
        self._draw()

    def _on_release(self, _e) -> None:
        if self._drag is not None:
            self._drag = None
            self._push()                               # send the result once, on release

    def _on_motion(self, e) -> None:
        if self._c is not None:
            X, Y = self._to_world(e.x, e.y)
            self._mouse_var.set(f"mouse: ({X:.3f}, {Y:.3f})")
            hit = self._hit(e.x, e.y)
            self._c.configure(cursor="fleur" if hit else "crosshair")

    def _on_pan(self, e) -> None:
        p = getattr(self, "_pan", None)
        if p is None:
            return
        s = self._scale()
        self._view["cx"] = p[2] - (e.x - p[0]) / s
        self._view["cy"] = p[3] + (e.y - p[1]) / s
        self._draw()

    def _on_wheel(self, e) -> str:
        steps = 1 if getattr(e, "num", None) == 4 else -1 if getattr(e, "num", None) == 5 else \
            (int(e.delta / 120) if e.delta else 0)
        if steps:
            X, Y = self._to_world(e.x, e.y)
            self._view["scale"] = min(max(self._scale() * 1.2 ** steps, 0.05), 1e5)
            nX, nY = self._to_world(e.x, e.y)               # keep the point under the cursor
            self._view["cx"] += X - nX
            self._view["cy"] += Y - nY
            self._draw()
        return "break"

    def _fit_view(self) -> None:
        if self._c is None:
            return
        pts = [p for p in self._pts if p is not None]
        res = self._result()
        if res is not None:
            pts = list(res[1]) + [np.array(h) for h in self._handles()]
        if not pts:
            self._view.update(cx=0.0, cy=0.0, scale=None)
        else:
            a = np.asarray(pts, dtype=float)
            lo, hi = a.min(axis=0), a.max(axis=0)
            span = max(float(np.max(hi - lo)), 1e-6)
            self._view.update(cx=float((lo[0] + hi[0]) / 2), cy=float((lo[1] + hi[1]) / 2),
                              scale=0.8 * min(self._c.winfo_width(), self._c.winfo_height()) / span)
        self._draw()

    # ── controls ───────────────────────────────────────────────────────
    def _sync_coord_entries(self) -> None:
        for k in range(2):
            for j in range(2):
                self._coord_vars[k][j].set("" if self._pts[k] is None else f"{self._pts[k][j]:.4f}")

    def _on_coord_entry(self, k: int) -> None:
        try:
            x, y = (float(v.get()) for v in self._coord_vars[k])
        except ValueError:
            self._sync_coord_entries()                # restore the valid values
            return
        if self._pts[k] is None or (x, y) != tuple(self._pts[k]):
            first_curve = None in self._pts
            self._pts[k] = [x, y]
            if first_curve and None not in self._pts:
                self._preset_straight(push=False)     # start from the straight line, as with clicks
            self._after_change()

    def _on_shape_changed(self) -> None:
        if self._syncing or self._drag is not None:
            return                                    # handle drags push on release
        self._draw()
        if self._pts[0] is not None and self._pts[1] is not None:
            self._push()

    def _after_change(self) -> None:
        self._sync_coord_entries()
        self._draw()
        self._push()

    def _on_clear(self) -> None:
        self._pts = [None, None]
        self._after_change()

    def _base_angle(self) -> float:
        p0, p1 = self._pts
        return math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))

    def _set_tangents(self, a0, a1, s0=1.0, s1=1.0, push=True) -> None:
        wrap = lambda a: (a + 180.0) % 360.0 - 180.0   # noqa: E731
        self._syncing = True
        try:
            self._angle_vars[0].set(round(wrap(a0), 1))
            self._angle_vars[1].set(round(wrap(a1), 1))
            self._strength_vars[0].set(s0)
            self._strength_vars[1].set(s1)
        finally:
            self._syncing = False
        self._draw()
        if push:
            self._push()

    def _preset_straight(self, push: bool = True) -> None:
        if self._pts[0] is not None and self._pts[1] is not None:
            b = self._base_angle()
            self._set_tangents(b, b, push=push)

    def _preset_s(self) -> None:
        if self._pts[0] is not None and self._pts[1] is not None:
            b = self._base_angle()
            self._set_tangents(b + 60, b + 60, 1.5, 1.5)

    def _preset_arc(self) -> None:
        if self._pts[0] is not None and self._pts[1] is not None:
            b = self._base_angle()
            self._set_tangents(b + 60, b - 60, 1.5, 1.5)

    # ── project ────────────────────────────────────────────────────────
    def get_help_text(self) -> str:
        return self.HELP_TEXT

    def get_params(self) -> dict:
        return {"p0": self._pts[0], "p1": self._pts[1],
                "angles": [float(v.get()) for v in self._angle_vars],
                "strengths": [float(v.get()) for v in self._strength_vars],
                "n": int(self._n_var.get()),
                "show_samples": bool(self._show_samples_var.get()),
                "show_handles": bool(self._show_handles_var.get()),
                "view": dict(self._view)}

    def set_params(self, params: dict) -> None:
        self._syncing = True
        try:
            for k, key in enumerate(("p0", "p1")):
                p = params.get(key)
                self._pts[k] = [float(p[0]), float(p[1])] if p else None
            for k in range(2):
                self._angle_vars[k].set(float(params.get("angles", [0, 0])[k]))
                self._strength_vars[k].set(float(params.get("strengths", [1, 1])[k]))
            self._n_var.set(int(params.get("n", 50)))
            self._show_samples_var.set(bool(params.get("show_samples", True)))
            self._show_handles_var.set(bool(params.get("show_handles", True)))
            view = params.get("view")
            if isinstance(view, dict):
                self._view.update({k: view.get(k, self._view[k]) for k in ("cx", "cy", "scale")})
        finally:
            self._syncing = False
        self.compute({})
