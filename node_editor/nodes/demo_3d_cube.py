# node_editor/nodes/demo_3d_cube.py
"""
Demo 3D Cube: a small demo / display node with no computation.

Shows a cube tumbling in 3D in its inspector window: a wireframe with hidden
lines removed (or optionally shown dashed), optional rainbow-coloured faces,
no lighting, no pins. Number keys 1 - 9 set the rotation speed, 0 pauses.

The 3D view follows the same conventions as the other 3D previews (Camera
Layout Viewer, Image/Data Simulator): world Z is up; left-drag rotates the
view, right-drag pans, the mouse wheel zooms, "Reset view" restores it; a
Perspective intensity slider goes from 0 (orthographic) to 1 (strong
perspective); a lower-left icon shows the global X / Y / Z axes with a
small view-aligned cube at its origin.

Hidden-line removal: a cube is convex, so an edge is visible exactly when
at least one of its two faces faces the eye (back-face test against the
actual eye position, i.e. it accounts for perspective) -- exact, no depth
sorting needed. For the same reason the front-facing faces never overlap
in the projection, so the rainbow faces need no sorting either.
"""

import math
import time
import tkinter as tk

import numpy as np

from node_editor.base_node import BaseNode
from node_editor.execution import ExecutionMode
from node_editor.pin_types import PinSchema
from node_editor.tooltip import add_tooltip

# Unit cube: 8 vertices, 6 faces listed counter-clockwise seen from OUTSIDE
# (so the right-hand normal points outward), 12 edges derived from the faces.
_VERTICES = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
_FACES = (
    (0, 1, 3, 2),   # x = -1
    (4, 6, 7, 5),   # x = +1
    (0, 4, 5, 1),   # y = -1
    (2, 3, 7, 6),   # y = +1
    (0, 2, 6, 4),   # z = -1
    (1, 5, 7, 3),   # z = +1
)
_RAINBOW = ("#e53935", "#fb8c00", "#fdd835", "#43a047", "#1e88e5", "#8e24aa")   # one hue per face
_EDGE_FACES: dict[tuple[int, int], list[int]] = {}
for _f, _quad in enumerate(_FACES):
    for _k in range(4):
        _a, _b = _quad[_k], _quad[(_k + 1) % 4]
        _EDGE_FACES.setdefault((min(_a, _b), max(_a, _b)), []).append(_f)

SCENE_RADIUS = math.sqrt(3.0)  # the cube's half-diagonal: fits the default view
DEG_PER_S_PER_LEVEL = 20.0     # speed level n -> n * 20 degrees per second
FRAME_MS = 16                  # ~60 frames per second
DEFAULT_VIEW = {"yaw": -0.9, "pitch": 0.7, "zoom": 1.0, "target": (0.0, 0.0, 0.0)}


def tumble(angle: float) -> np.ndarray:
    """The cube's own rotation (world frame): about Z, X and Y at different
    rates, so every face comes into view."""
    a, b, c = angle, 0.7 * angle + 0.4, 0.3 * angle
    rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]])
    ry = np.array([[math.cos(c), 0, math.sin(c)], [0, 1, 0], [-math.sin(c), 0, math.cos(c)]])
    return rz @ rx @ ry


def view_rotation(yaw: float, pitch: float) -> np.ndarray:
    """World -> view rotation, as in the other 3D previews: view x = right,
    view y = depth (away from the eye), view z = up on the screen."""
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    return np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]) @ np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])


def eye_distance(persp: float) -> float | None:
    """Eye distance in front of the view target (None = orthographic)."""
    return None if persp <= 1e-6 else SCENE_RADIUS * 3.0 / persp


def project(rot: np.ndarray, persp: float) -> np.ndarray:
    """View-space points (N, 3) -> 2D (N, 2), screen y up, before scaling."""
    d0 = eye_distance(persp)
    f = np.ones(len(rot)) if d0 is None else d0 / np.maximum(d0 + rot[:, 1], d0 * 1e-3)
    return np.column_stack([rot[:, 0] * f, rot[:, 2] * f])


def cube_scene(angle: float, R: np.ndarray, target, persp: float):
    """Everything needed to draw the cube: (2D points (8, 2), facing flags
    per face, visible edges, hidden edges)."""
    world = _VERTICES @ tumble(angle).T
    rot = (world - np.asarray(target, dtype=float)) @ R.T
    d0 = eye_distance(persp)
    eye = None if d0 is None else np.array([0.0, -d0, 0.0])
    facing = []
    for quad in _FACES:
        q = rot[list(quad)]
        normal = np.cross(q[1] - q[0], q[2] - q[0])             # outward
        view_dir = np.array([0.0, 1.0, 0.0]) if eye is None else q.mean(axis=0) - eye
        facing.append(float(np.dot(normal, view_dir)) < 0.0)
    visible, hidden = [], []
    for edge, faces in _EDGE_FACES.items():
        (visible if any(facing[f] for f in faces) else hidden).append(edge)
    return project(rot, persp), facing, visible, hidden


class Demo3DCubeNode(BaseNode):
    """Cube rotating in 3D (hidden lines removed, optional rainbow faces);
    keys 1-9 set the speed, 0 pauses; mouse: rotate / pan / zoom the view."""

    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE = "demo_3d_cube"
    DISPLAY_NAME = "Demo 3D Cube"
    CATEGORY = "visualize"
    SEARCH_KEYWORDS = ("cube", "3d", "rotate", "demo", "wireframe", "hidden line")
    NODE_WIDTH = 170
    NODE_HEIGHT = 80

    HELP_TEXT = (
        "Demo 3D Cube\n\n"
        "Shows a cube rotating in 3D in the inspector window.\n"
        "No input / output pins and no computation -- for demonstrations.\n\n"
        "Keys (inspector window):\n"
        "  1 ... 9   rotation speed (1 = slow, 9 = fast; n x 20 degrees/s)\n"
        "  0         pause\n\n"
        "Mouse (as in the other 3D previews):\n"
        "  left-drag  rotate the view    right-drag  pan    wheel  zoom\n"
        "  'Reset view' restores the default view.\n\n"
        "Perspective intensity: 0 = orthographic (parallel projection),\n"
        "1 = strong perspective (near / far foreshortening).\n"
        "The lower-left icon shows the global X (red), Y (green), Z (blue)\n"
        "axes with a small cube; Z is up.\n\n"
        "Hidden lines (edges behind the cube) are removed; tick\n"
        "'Show hidden lines (dashed)' to see them dashed.\n"
        "'Rainbow surfaces' fills the six faces with rainbow colours\n"
        "(no lighting).\n"
    )

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(inputs=[], outputs=[])

    def _init_state(self) -> None:
        if hasattr(self, "_speed_var"):
            return
        self._speed_var = tk.IntVar(value=3)                 # 0 = paused, 1..9
        self._show_hidden_var = tk.BooleanVar(value=False)
        self._surfaces_var = tk.BooleanVar(value=False)
        self._persp_var = tk.DoubleVar(value=0.3)
        self._status_var = tk.StringVar(value="")
        self._angle = 0.0
        self._last_t = None
        self._after_id = None
        self._view = None
        self._v3 = dict(DEFAULT_VIEW, dragging=False, button=None, last_x=0, last_y=0, last_scale=1.0)
        self._speed_var.trace_add("write", lambda *_a: self._update_status())
        self._update_status()

    # ── node face ──────────────────────────────────────────────────────
    def build_body(self) -> None:
        self._init_state()
        x, y, w, h = self.x, self.y, self.width, self.height
        self._body_rect = self.canvas.create_rectangle(
            x, y, x + w, y + h, fill="#eef2ff", outline="#4d5ea8", width=2,
            tags=(self.node_id, "node_body"))
        self._title_item = self.canvas.create_text(
            x + w / 2, y + 13, text=self.DISPLAY_NAME, font=("Arial", 9, "bold"),
            fill="#2c3a78", tags=(self.node_id,))
        status = tk.Label(self.canvas, textvariable=self._status_var, font=("Arial", 8),
                          bg="#eef2ff", fg="#2c3a78")
        self.canvas.create_window(x + w / 2, y + h / 2 + 8, window=status, tags=(self.node_id,))
        self._canvas_items += [self._body_rect, self._title_item]

    def _update_status(self) -> None:
        level = int(self._speed_var.get())
        self._status_var.set("paused (keys 1-9 run)" if level == 0 else
                             f"speed {level}/9  ({level * DEG_PER_S_PER_LEVEL:.0f} deg/s)")

    # ── inspector ──────────────────────────────────────────────────────
    def build_inspector(self, parent: tk.Frame) -> None:
        self._init_state()
        win = self._inspector_win
        if win is not None:
            win.geometry("600x640")
            win.minsize(320, 360)
            win.bind("<Key>", self._on_key)                  # 1..9 / 0, wherever focus is
            win.after(50, win.focus_force)

        row1 = tk.Frame(parent)
        row1.pack(fill="x")
        tk.Label(row1, textvariable=self._status_var, font=("Arial", 10, "bold"), fg="#2c3a78").pack(side="left")
        tk.Button(row1, text="Reset view", font=("Arial", 8), command=self._reset_view).pack(side="right")

        row2 = tk.Frame(parent)
        row2.pack(fill="x", pady=(2, 0))
        persp_lbl = tk.Label(row2, text="Perspective intensity", font=("Arial", 8))
        persp_lbl.pack(side="left")
        persp = tk.Scale(row2, from_=0.0, to=1.0, resolution=0.02, orient=tk.HORIZONTAL, length=150,
                         variable=self._persp_var, showvalue=True, command=lambda _v: self._draw())
        persp.pack(side="left", padx=(4, 12))
        persp_tip = ("0 = orthographic (parallel projection, sizes do not change with distance); "
                     "1 = strong perspective (near / far foreshortening).")
        add_tooltip(persp_lbl, persp_tip)
        add_tooltip(persp, persp_tip)
        surf_cb = tk.Checkbutton(row2, text="Rainbow surfaces", variable=self._surfaces_var,
                                 command=self._draw)
        surf_cb.pack(side="left")
        add_tooltip(surf_cb, "Fill the six faces with rainbow colours (flat, no lighting). Only "
                             "the faces turned towards you are drawn, so they never overlap.")
        hid_cb = tk.Checkbutton(row2, text="Show hidden lines (dashed)", variable=self._show_hidden_var,
                                command=self._draw)
        hid_cb.pack(side="left", padx=(8, 0))
        add_tooltip(hid_cb, "Hidden lines are the cube edges behind the cube. They are removed by "
                            "default; tick to draw them dashed instead (on top of the surfaces).")

        tk.Label(parent, text="Keys 1 - 9: rotation speed (1 slow ... 9 fast),  0: pause.   "
                              "Mouse: left-drag rotate, right-drag pan, wheel zoom.",
                 font=("Arial", 8), fg="#555555").pack(anchor="w")
        c = self._view = tk.Canvas(parent, bg="#ffffff", highlightthickness=1, highlightbackground="#c8cde0")
        c.pack(fill="both", expand=True, pady=(4, 0))
        c.bind("<Configure>", lambda _e: self._draw())
        c.bind("<ButtonPress-1>", lambda e: self._press(e, 1))
        c.bind("<ButtonPress-3>", lambda e: self._press(e, 3))
        c.bind("<B1-Motion>", self._on_rotate)
        c.bind("<B3-Motion>", self._on_pan)
        for seq in ("<ButtonRelease-1>", "<ButtonRelease-3>"):
            c.bind(seq, lambda _e: self._v3.update(dragging=False, button=None))
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            c.bind(seq, self._on_zoom)
        self._last_t = None
        self._tick()

    def _on_key(self, event) -> str | None:
        if event.char and event.char in "0123456789":
            self._speed_var.set(int(event.char))
            return "break"
        return None

    # ── mouse: same conventions as the other 3D previews ────────────────
    def _press(self, e, button: int) -> None:
        self._view.focus_set()
        self._v3.update(dragging=True, button=button, last_x=e.x, last_y=e.y)

    def _on_rotate(self, e) -> None:
        if self._v3["dragging"] and self._v3["button"] == 1:
            self._v3["yaw"] += (e.x - self._v3["last_x"]) * 0.01
            self._v3["pitch"] += (e.y - self._v3["last_y"]) * 0.01
            self._v3["last_x"], self._v3["last_y"] = e.x, e.y
            self._draw()

    def _on_pan(self, e) -> None:
        if not (self._v3["dragging"] and self._v3["button"] == 3):
            return
        dx, dy, sc = e.x - self._v3["last_x"], e.y - self._v3["last_y"], self._v3["last_scale"]
        if sc > 1e-12:
            rt = view_rotation(self._v3["yaw"], self._v3["pitch"]).T
            self._v3["target"] = tuple(np.asarray(self._v3["target"]) - (rt[:, 0] * dx - rt[:, 2] * dy) / sc)
        self._v3["last_x"], self._v3["last_y"] = e.x, e.y
        self._draw()

    @staticmethod
    def _wheel_steps(e) -> int:
        if getattr(e, "num", None) == 4:
            return 1
        if getattr(e, "num", None) == 5:
            return -1
        delta = getattr(e, "delta", 0)
        return int(delta / 120) if delta else 0

    def _on_zoom(self, e) -> str:
        steps = self._wheel_steps(e)
        if steps:
            self._v3["zoom"] = max(0.05, self._v3["zoom"] * (1.12 ** steps))
            self._draw()
        return "break"

    def _reset_view(self) -> None:
        self._v3.update(DEFAULT_VIEW)
        self._draw()

    # ── animation / drawing ────────────────────────────────────────────
    def _tick(self) -> None:
        """Animation step: advance by speed x elapsed time (smooth whatever the
        frame rate), redraw, and schedule the next step."""
        self._after_id = None
        view = self._view
        if view is None or self._destroyed_view():
            return
        now = time.perf_counter()
        if self._last_t is not None:
            speed = math.radians(int(self._speed_var.get()) * DEG_PER_S_PER_LEVEL)
            self._angle = (self._angle + speed * (now - self._last_t)) % (2.0 * math.pi * 1000.0)
        self._last_t = now
        self._draw()
        try:
            self._after_id = view.after(FRAME_MS, self._tick)
        except tk.TclError:
            self._after_id = None

    def _destroyed_view(self) -> bool:
        try:
            return not self._view.winfo_exists()
        except tk.TclError:
            return True

    def _draw(self) -> None:
        c = self._view
        if c is None or self._destroyed_view():
            return
        w, h = c.winfo_width(), c.winfo_height()
        if w < 40 or h < 40:
            return
        R = view_rotation(self._v3["yaw"], self._v3["pitch"])
        persp = float(self._persp_var.get())
        pts, facing, visible, hidden = cube_scene(self._angle, R, self._v3["target"], persp)
        scale = min(w, h) * 0.36 * self._v3["zoom"] / SCENE_RADIUS
        self._v3["last_scale"] = scale
        xy = [(w / 2 + scale * px, h / 2 - scale * py) for px, py in pts]
        c.delete("all")
        if self._surfaces_var.get():
            for f, quad in enumerate(_FACES):
                if facing[f]:
                    c.create_polygon(*[v for i in quad for v in xy[i]], fill=_RAINBOW[f], outline="")
        if self._show_hidden_var.get():
            for a, b in hidden:
                c.create_line(*xy[a], *xy[b], fill="#9aa3c0", width=1, dash=(4, 3))
        for a, b in visible:
            c.create_line(*xy[a], *xy[b], fill="#1f2d6b", width=2, capstyle=tk.ROUND)
        self._draw_axes_icon(c, h, R)

    def _draw_axes_icon(self, c: tk.Canvas, ch: float, R: np.ndarray) -> None:
        """Lower-left global X/Y/Z axes icon with a small view-aligned cube at
        its origin (same style as the other 3D previews)."""
        corner = np.array([32.0, float(ch) - 32.0])
        axis_len = 22.0
        for ax_v, col, lab in ((np.array([1.0, 0.0, 0.0]), "#cc0000", "X"),
                               (np.array([0.0, 1.0, 0.0]), "#00aa00", "Y"),
                               (np.array([0.0, 0.0, 1.0]), "#0000cc", "Z")):
            rv = R @ ax_v
            e = corner + np.array([rv[0], -rv[2]]) * axis_len
            c.create_line(corner[0], corner[1], e[0], e[1], fill=col, width=2)
            c.create_text(e[0] + (e[0] - corner[0]) * 0.25, e[1] + (e[1] - corner[1]) * 0.25,
                          text=lab, fill=col, font=("Arial", 8, "bold"))
        # Small cube: only the faces turned towards the viewer (view y < 0).
        colors = ("#ffcccc", "#ff9999", "#cceecc", "#99dd99", "#ccccff", "#9999ff")   # -x +x -y +y -z +z
        verts = (_VERTICES * (axis_len * 0.32)) @ R.T
        for f, quad in enumerate(_FACES):
            q = verts[list(quad)]
            if float(np.cross(q[1] - q[0], q[2] - q[0])[1]) < 0.0:
                c.create_polygon([(corner[0] + verts[i][0], corner[1] - verts[i][2]) for i in quad],
                                 fill=colors[f], outline="#333333", width=1.3)

    def _stop_animation(self) -> None:
        if self._after_id is not None and self._view is not None:
            try:
                self._view.after_cancel(self._after_id)
            except tk.TclError:
                pass
        self._after_id = None

    def close_inspector(self) -> None:
        self._stop_animation()
        self._view = None
        super().close_inspector()

    def on_destroy(self) -> None:
        self._stop_animation()
        self._view = None
        super().on_destroy()

    # ── engine / project ───────────────────────────────────────────────
    def compute(self, inputs: dict) -> dict:
        return {}

    def get_help_text(self) -> str:
        return self.HELP_TEXT

    def get_params(self) -> dict:
        self._init_state()
        v = self._v3
        return {"speed": int(self._speed_var.get()), "show_hidden": bool(self._show_hidden_var.get()),
                "surfaces": bool(self._surfaces_var.get()), "perspective": float(self._persp_var.get()),
                "view": {"yaw": v["yaw"], "pitch": v["pitch"], "zoom": v["zoom"],
                         "target": [float(t) for t in v["target"]]}}

    def set_params(self, params: dict) -> None:
        self._init_state()
        try:
            self._speed_var.set(min(9, max(0, int(params.get("speed", 3)))))
        except (TypeError, ValueError):
            self._speed_var.set(3)
        self._show_hidden_var.set(bool(params.get("show_hidden", False)))
        self._surfaces_var.set(bool(params.get("surfaces", False)))
        try:
            self._persp_var.set(min(1.0, max(0.0, float(params.get("perspective", 0.3)))))
        except (TypeError, ValueError):
            self._persp_var.set(0.3)
        view = params.get("view")
        if isinstance(view, dict):
            try:
                self._v3.update(yaw=float(view["yaw"]), pitch=float(view["pitch"]),
                                zoom=max(0.05, float(view["zoom"])),
                                target=tuple(float(t) for t in view["target"][:3]))
            except (KeyError, TypeError, ValueError):
                pass
