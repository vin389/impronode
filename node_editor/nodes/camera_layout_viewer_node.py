# node_editor/nodes/camera_layout_viewer_node.py

import math
import tkinter as tk
from tkinter import ttk

import numpy as np
import cv2

from node_editor.base_node import BaseNode
from node_editor.pin_types import PinSchema, PinDef, PinType
from node_editor.execution import ExecutionMode


_HELP_TEXT = """\
Camera Layout Viewer
====================

PURPOSE
-------
Shows where calibrated cameras sit in 3D world coordinates, relative to
one or more user-defined reference grids. Useful for sanity-checking a
stereo/multi-camera rig: are the cameras where you think they are, are
they pointing at the specimen, is the vergence angle reasonable, does
the working volume actually fall inside both fields of view.

Supports up to 2 cameras.

INPUT PINS (per camera N = 1, 2 -- all optional)
--------------------------------------------------
imgsizeN   ARRAY  [width, height] in pixels. Needed to know how wide
                  the frustum is; without it that camera is skipped.
cmatN      ARRAY  3x3 intrinsic matrix. fx, fy set the FOV; cx, cy
                  shift the frustum for an off-centre principal point.
dvecN      ARRAY  Distortion coefficients. ACCEPTED BUT UNUSED for
                  plotting -- the frustum is drawn from the ideal
                  pinhole model. Connect it or don't; it changes
                  nothing here. (It is listed so this node can be
                  wired straight from a calibration node's outputs
                  without leaving a stray pin.)
rvecN      ARRAY  Rodrigues rotation vector (3,) or (3,1).
tvecN      ARRAY  Translation vector (3,) or (3,1).

rvecN/tvecN are the standard OpenCV extrinsics mapping WORLD to CAMERA:
    X_cam = R @ X_world + t,   R = Rodrigues(rvec)
so the homogeneous form is  T_cam<-world = [[R, t], [0,0,0,1]].
This node inverts that to place the camera in the world:
    T_world<-cam = [[R.T, -R.T @ t], [0,0,0,1]]
    camera centre in world:  C = -R.T @ t
    camera axes in world:    columns of R.T
                             (R.T[:,0] = image right / +u
                              R.T[:,1] = image down  / +v
                              R.T[:,2] = optical axis, forward)

OUTPUT PINS
-----------
cam1_pos, cam2_pos      ARRAY (3,)   camera centre in world coords
cam1_axes, cam2_axes    ARRAY (3,3)  camera axes in world coords,
                                      as COLUMNS [right | down | fwd]
                                      (this is exactly R.T)
grid_points             ARRAY (N,3)  every grid vertex, all grids
                                      concatenated
n_cameras               SCALAR       how many cameras were drawable

FRUSTUM GEOMETRY
----------------
The pyramid apex is the camera centre. The base is the four image
corners back-projected to a depth d:
    x_n = (u - cx) / fx,  y_n = (v - cy) / fy
    P_cam = (x_n*d, y_n*d, d)  ->  P_world = R.T @ P_cam + C
for (u,v) in {(0,0), (w,0), (w,h), (0,h)}. Using the real corners
rather than a symmetric half-FOV means an off-centre principal point
produces a correspondingly skewed (and therefore correct) frustum.

A small notch is drawn on the frustum's top edge to indicate which way
is image-UP, so a rolled camera is immediately visible.

The depth d is chosen automatically from the scene size so the cameras
read clearly against the grids, then multiplied by "Pyramid size scale"
from the inspector. Scale is a multiplier on that automatic value, not
an absolute length -- 1.0 is the automatic default.

INSPECTOR -- GRIDS
------------------
grid_xs / grid_ys / grid_zs define one or more axis-aligned lattices.
Each accepts a semicolon-separated list of sequences; sequence syntax
matches the calibration nodes:
    "0:10:30"            -> 0 10 20 30      (start:step:end, inclusive)
    "0:3"                -> 0 1 2 3         (step defaults to 1)
    "0 10 20 30"         -> explicit values (space or comma separated)
Semicolons separate GRIDS:
    grid_xs = "0:10:30; 60:10:90"
    grid_ys = "0:10:20; 0:10:20"
    grid_zs = "0;        0"
defines two separate 4x3x1 lattices. All three fields must contain the
SAME number of semicolon-separated segments, or nothing is drawn and
the status line reports the mismatch.

A single-value axis (e.g. grid_zs = "0") gives a planar grid, which is
the usual case for a calibration board or a specimen surface.

INSPECTOR -- APPEARANCE
-----------------------
Pyramid size scale   multiplier on the auto-computed frustum depth
Pyramid colour       frustum line colour
Pyramid line width
Grid colour          grid line colour
Grid line width
Perspective intensity  0 = orthographic (parallel lines stay parallel,
                       matches the other 3D previews in this project).
                       Increasing it foreshortens distant geometry;
                       1.0 is a strong wide-angle look. Purely a
                       VIEWING control -- it does not touch the camera
                       parameters or any output.

MOUSE CONTROLS (3D preview)
----------------------------
Left-drag    rotate (orbits the current pan target)
Right-drag   pan
Wheel        zoom
A small axes icon at the lower left shows world X/Y/Z orientation.

KEYBOARD SHORTCUT
-----------------
Ctrl-H / Ctrl-h -- show this help window.
"""

_COMMON_COLORS = [
    "red", "green", "blue", "black", "orange",
    "purple", "brown", "magenta", "cyan", "gray",
]

_CAM_DEFAULT_COLORS = ["#cc2222", "#2255cc"]


# ---------------------------------------------------------------------------
# Sequence / grid parsing
# ---------------------------------------------------------------------------

def _parse_sequence(text: str) -> np.ndarray | None:
    """
    Parse one 1-D sequence. Supports MATLAB-style start:step:end and
    start:end (end inclusive), or an explicit space/comma separated list.
    Returns None if the text is empty or unparseable.
    """
    text = (text or "").strip()
    if not text:
        return None

    if ":" in text:
        parts = text.split(":")
        try:
            if len(parts) == 2:
                start = float(parts[0]); end = float(parts[1]); step = 1.0
            elif len(parts) == 3:
                start = float(parts[0]); step = float(parts[1]); end = float(parts[2])
            else:
                return None
            if step == 0:
                return None
            eps = abs(step) * 1e-9
            arr = np.arange(start, end + np.sign(step) * eps, step, dtype=np.float64)
            return arr if arr.size else None
        except ValueError:
            return None

    try:
        vals = [float(t) for t in text.replace(",", " ").split() if t.strip()]
        return np.array(vals, dtype=np.float64) if vals else None
    except ValueError:
        return None


def _parse_multi_sequence(text: str) -> list[np.ndarray | None]:
    """
    Split on ';' and parse each segment as a sequence. An empty overall
    string yields an empty list (no grids). A segment that fails to
    parse yields None in that slot, so the caller can report WHICH grid
    is bad rather than silently dropping it.
    """
    text = (text or "").strip()
    if not text:
        return []
    return [_parse_sequence(seg) for seg in text.split(";")]


# ---------------------------------------------------------------------------
# Camera geometry
# ---------------------------------------------------------------------------

def _camera_pose_in_world(rvec: np.ndarray, tvec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert OpenCV extrinsics (world -> camera) into the camera's pose
    IN world coordinates.

    Given X_cam = R @ X_world + t, the homogeneous transform is
        T_cam<-world = [[R, t], [0,0,0,1]]
    whose inverse is
        T_world<-cam = [[R.T, -R.T @ t], [0,0,0,1]]

    Returns (centre (3,), axes (3,3)) where `axes` columns are the
    camera's right/down/forward unit vectors expressed in world coords,
    i.e. axes == R.T.
    """
    r_mat, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    t = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    axes = r_mat.T                      # world <- camera rotation
    centre = (-axes @ t).ravel()        # -R.T @ t
    return centre, axes


def _frustum_corners_world(centre: np.ndarray, axes: np.ndarray,
                           cmat: np.ndarray, img_w: float, img_h: float,
                           depth: float) -> np.ndarray:
    """
    Back-project the four image corners to `depth` along the optical
    axis and express them in world coordinates. Returns (4,3) ordered
    [top-left, top-right, bottom-right, bottom-left] in image terms.

    Uses the actual corner pixels rather than a symmetric half-FOV so
    an off-centre principal point yields a correctly skewed frustum.
    """
    fx = float(cmat[0, 0])
    fy = float(cmat[1, 1])
    cx = float(cmat[0, 2])
    cy = float(cmat[1, 2])
    if abs(fx) < 1e-12 or abs(fy) < 1e-12:
        raise ValueError("cmat has a zero focal length")

    corners_px = [(0.0, 0.0), (img_w, 0.0), (img_w, img_h), (0.0, img_h)]
    out = np.zeros((4, 3), dtype=np.float64)
    for k, (u, v) in enumerate(corners_px):
        x_n = (u - cx) / fx
        y_n = (v - cy) / fy
        p_cam = np.array([x_n * depth, y_n * depth, depth], dtype=np.float64)
        out[k] = axes @ p_cam + centre
    return out


def _fov_degrees(cmat: np.ndarray, img_w: float, img_h: float) -> tuple[float, float]:
    """Horizontal and vertical field of view, in degrees, for display."""
    fx = float(cmat[0, 0])
    fy = float(cmat[1, 1])
    if abs(fx) < 1e-12 or abs(fy) < 1e-12:
        return float("nan"), float("nan")
    fov_h = 2.0 * math.degrees(math.atan(img_w / (2.0 * fx)))
    fov_v = 2.0 * math.degrees(math.atan(img_h / (2.0 * fy)))
    return fov_h, fov_v


# ---------------------------------------------------------------------------
# CameraLayoutViewerNode
# ---------------------------------------------------------------------------

class CameraLayoutViewerNode(BaseNode):
    """
    Plots up to 2 calibrated cameras as frustum pyramids alongside
    user-defined reference grids, in an interactive 3D preview with an
    adjustable perspective intensity.

    See _HELP_TEXT (Ctrl-H) for the full reference.
    """

    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE = "camera_layout_viewer"
    DISPLAY_NAME = "Camera Layout Viewer"
    CATEGORY = "visualize"
    SEARCH_KEYWORDS = ("camera", "extrinsics", "frustum", "pose", "rig",
                       "stereo", "layout", "3d preview", "rvec", "tvec",
                       "perspective")
    NODE_WIDTH = 220
    NODE_HEIGHT = 130

    HELP_TEXT = _HELP_TEXT

    MAX_CAMERAS = 2

    _BODY_BG = "#f2f6ff"
    _OUTLINE = "#3a5f9a"
    _TITLE_FG = "#1c3055"
    _STATUS_FG = "#456"

    _CUBE_VERTS = np.array([
        [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
    ], dtype=np.float64)

    _CUBE_FACES = [
        {"normal": np.array([1.0, 0.0, 0.0]),  "verts": [1, 2, 6, 5], "color": "#ff9999"},
        {"normal": np.array([-1.0, 0.0, 0.0]), "verts": [0, 3, 7, 4], "color": "#ffcccc"},
        {"normal": np.array([0.0, 1.0, 0.0]),  "verts": [3, 2, 6, 7], "color": "#99dd99"},
        {"normal": np.array([0.0, -1.0, 0.0]), "verts": [0, 1, 5, 4], "color": "#cceecc"},
        {"normal": np.array([0.0, 0.0, 1.0]),  "verts": [4, 5, 6, 7], "color": "#9999ff"},
        {"normal": np.array([0.0, 0.0, -1.0]), "verts": [0, 1, 2, 3], "color": "#ccccff"},
    ]

    # ── schema ────────────────────────────────────────────────────

    def get_pin_schema(self) -> PinSchema:
        inputs = []
        for n in (1, 2):
            inputs.extend([
                PinDef(f"imgsize{n}", PinType.ARRAY, f"size{n}", optional=True),
                PinDef(f"cmat{n}", PinType.ARRAY, f"cmat{n}", optional=True),
                PinDef(f"dvec{n}", PinType.ARRAY, f"dvec{n}", optional=True),
                PinDef(f"rvec{n}", PinType.ARRAY, f"rvec{n}", optional=True),
                PinDef(f"tvec{n}", PinType.ARRAY, f"tvec{n}", optional=True),
            ])
        return PinSchema(
            inputs=inputs,
            outputs=[
                PinDef("cam1_pos", PinType.ARRAY, "c1pos"),
                PinDef("cam1_axes", PinType.ARRAY, "c1axes"),
                PinDef("cam2_pos", PinType.ARRAY, "c2pos"),
                PinDef("cam2_axes", PinType.ARRAY, "c2axes"),
                PinDef("grid_points", PinType.ARRAY, "gridPts"),
                PinDef("n_cameras", PinType.SCALAR, "nCams"),
            ],
        )

    def get_help_text(self) -> str:
        return _HELP_TEXT

    # ── state ─────────────────────────────────────────────────────

    def _init_state(self) -> None:
        if hasattr(self, "_grid_xs_var"):
            return

        self._grid_xs_var = tk.StringVar(value="0:10:100")
        self._grid_ys_var = tk.StringVar(value="0:10:60")
        self._grid_zs_var = tk.StringVar(value="0")

        self._pyr_scale_var = tk.DoubleVar(value=1.0)
        self._pyr_width_var = tk.DoubleVar(value=1.5)
        self._grid_color_var = tk.StringVar(value="gray")
        self._grid_width_var = tk.DoubleVar(value=1.0)
        self._persp_var = tk.DoubleVar(value=0.0)
        self._show_labels_var = tk.BooleanVar(value=True)

        self._cam_color_vars = {
            1: tk.StringVar(value=_CAM_DEFAULT_COLORS[0]),
            2: tk.StringVar(value=_CAM_DEFAULT_COLORS[1]),
        }

        self._status_var = tk.StringVar(value="waiting for camera inputs")
        self._grid_status_var = tk.StringVar(value="")
        self._cam_status_var = tk.StringVar(value="")

        # resolved scene data, rebuilt every compute()
        self._cameras: dict[int, dict] = {}     # n -> {centre, axes, cmat, w, h, fov}
        self._grids: list[dict] = []            # [{xs, ys, zs}]
        self._grid_points = np.empty((0, 3), dtype=np.float64)

        self._preview_canvas: tk.Canvas | None = None
        self._help_popup: tk.Toplevel | None = None

        # orbit camera (same convention as the other 3D previews in this
        # project: rotation pivots about camera_target, panning moves
        # camera_target in world space)
        self._cam_state = {
            "yaw": -0.9, "pitch": 0.7, "zoom": 1.0,
            "camera_target": [0.0, 0.0, 0.0],
            "dragging": False, "last_x": 0, "last_y": 0, "button": None,
        }
        self._auto_centred = False

    # ── body ──────────────────────────────────────────────────────

    def build_body(self) -> None:
        self._init_state()
        x, y, w, h = self.x, self.y, self.width, self.height

        self._body_rect = self.canvas.create_rectangle(
            x, y, x + w, y + h, fill=self._BODY_BG,
            outline=self._OUTLINE, width=2,
            tags=(self.node_id, "node_body"))
        self._title_item = self.canvas.create_text(
            x + w / 2, y + 13, text=self.DISPLAY_NAME,
            font=("Arial", 9, "bold"), fill=self._TITLE_FG,
            tags=(self.node_id,))

        cam_lbl = tk.Label(self.canvas, textvariable=self._cam_status_var,
                           font=("Arial", 8), bg=self._BODY_BG, fg=self._TITLE_FG,
                           wraplength=w - 12, justify="center")
        self.canvas.create_window(x + w / 2, y + h * 0.42, window=cam_lbl,
                                  tags=(self.node_id,))

        grid_lbl = tk.Label(self.canvas, textvariable=self._grid_status_var,
                            font=("Arial", 8), bg=self._BODY_BG, fg=self._TITLE_FG,
                            wraplength=w - 12, justify="center")
        self.canvas.create_window(x + w / 2, y + h * 0.62, window=grid_lbl,
                                  tags=(self.node_id,))

        status_lbl = tk.Label(self.canvas, textvariable=self._status_var,
                              font=("Arial", 7), bg=self._BODY_BG, fg=self._STATUS_FG,
                              wraplength=w - 12, justify="center")
        self.canvas.create_window(x + w / 2, y + h - 12, window=status_lbl,
                                  tags=(self.node_id,))

        self._canvas_items += [self._body_rect, self._title_item]
        self.canvas.tag_bind(self.node_id, "<Double-Button-1>",
                             lambda e: self.open_inspector())

    # ── inspector ─────────────────────────────────────────────────

    def build_inspector(self, parent: tk.Frame) -> None:
        self._init_state()
        for child in parent.winfo_children():
            child.destroy()

        win = self._inspector_win
        if win is not None:
            for seq in ("<Control-h>", "<Control-H>"):
                win.bind(seq, lambda e: self._open_help())

        panes, left, right = self._make_paned(parent, left_width=300, left_minsize=240)
        pad = {"padx": 6, "pady": 4}
        inner = self._make_scrollable(left)

        # ── grids ─────────────────────────────────────────────────
        grid_frame = tk.LabelFrame(inner, text="Reference grids", font=("Arial", 9), **pad)
        grid_frame.pack(fill="x", **pad)
        tk.Label(grid_frame,
                 text='"0:10:30" = 0 10 20 30.  Use ";" to separate\n'
                      'multiple grids; xs/ys/zs need the same count.',
                 font=("Arial", 7), fg="#888888", justify="left").pack(anchor="w", pady=(0, 4))
        for label, var in (("grid_xs:", self._grid_xs_var),
                           ("grid_ys:", self._grid_ys_var),
                           ("grid_zs:", self._grid_zs_var)):
            row = tk.Frame(grid_frame)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label, font=("Arial", 9), width=8, anchor="w").pack(side="left")
            ent = tk.Entry(row, textvariable=var, font=("Courier", 9))
            ent.pack(side="left", fill="x", expand=True)
            ent.bind("<Return>", lambda e: self._on_settings_changed())
            ent.bind("<FocusOut>", lambda e: self._on_settings_changed())

        tk.Label(grid_frame, textvariable=self._grid_status_var, font=("Arial", 8),
                 fg="#556", anchor="w", justify="left", wraplength=260).pack(fill="x", pady=(4, 0))

        style_row = tk.Frame(grid_frame)
        style_row.pack(fill="x", pady=(4, 0))
        tk.Label(style_row, text="Colour:", font=("Arial", 8)).pack(side="left")
        gc = ttk.Combobox(style_row, textvariable=self._grid_color_var,
                          values=_COMMON_COLORS, width=9)
        gc.pack(side="left", padx=(2, 8))
        gc.bind("<<ComboboxSelected>>", lambda e: self._refresh_preview())
        gc.bind("<FocusOut>", lambda e: self._refresh_preview())
        tk.Label(style_row, text="Width:", font=("Arial", 8)).pack(side="left")
        tk.Spinbox(style_row, from_=0.5, to=6.0, increment=0.5,
                   textvariable=self._grid_width_var, width=5, font=("Arial", 8),
                   command=self._refresh_preview).pack(side="left", padx=(2, 0))

        # ── cameras ───────────────────────────────────────────────
        cam_frame = tk.LabelFrame(inner, text="Cameras", font=("Arial", 9), **pad)
        cam_frame.pack(fill="x", **pad)
        tk.Label(cam_frame, textvariable=self._cam_status_var, font=("Arial", 8),
                 fg="#556", anchor="w", justify="left", wraplength=260).pack(fill="x")

        for n in (1, 2):
            row = tk.Frame(cam_frame)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=f"Cam {n} colour:", font=("Arial", 8), width=12,
                     anchor="w").pack(side="left")
            cc = ttk.Combobox(row, textvariable=self._cam_color_vars[n],
                              values=_COMMON_COLORS + _CAM_DEFAULT_COLORS, width=10)
            cc.pack(side="left")
            cc.bind("<<ComboboxSelected>>", lambda e: self._refresh_preview())
            cc.bind("<FocusOut>", lambda e: self._refresh_preview())

        pyr_row = tk.Frame(cam_frame)
        pyr_row.pack(fill="x", pady=(4, 0))
        tk.Label(pyr_row, text="Pyramid scale:", font=("Arial", 8), width=12,
                 anchor="w").pack(side="left")
        tk.Spinbox(pyr_row, from_=0.05, to=20.0, increment=0.1,
                   textvariable=self._pyr_scale_var, width=7, font=("Arial", 8),
                   command=self._refresh_preview).pack(side="left")

        pw_row = tk.Frame(cam_frame)
        pw_row.pack(fill="x", pady=2)
        tk.Label(pw_row, text="Pyramid width:", font=("Arial", 8), width=12,
                 anchor="w").pack(side="left")
        tk.Spinbox(pw_row, from_=0.5, to=6.0, increment=0.5,
                   textvariable=self._pyr_width_var, width=7, font=("Arial", 8),
                   command=self._refresh_preview).pack(side="left")

        tk.Label(cam_frame, text="Scale multiplies an automatic size based on\n"
                                 "the scene extent; 1.0 is the auto default.",
                 font=("Arial", 7), fg="#888888", justify="left").pack(anchor="w", pady=(2, 0))

        # ── view ──────────────────────────────────────────────────
        view_frame = tk.LabelFrame(inner, text="View", font=("Arial", 9), **pad)
        view_frame.pack(fill="x", **pad)

        tk.Label(view_frame, text="Perspective intensity:", font=("Arial", 9)).pack(anchor="w")
        tk.Scale(view_frame, from_=0.0, to=1.0, resolution=0.02, orient=tk.HORIZONTAL,
                 variable=self._persp_var,
                 command=lambda _v: self._refresh_preview()).pack(fill="x")
        tk.Label(view_frame,
                 text="0 = orthographic (parallel projection).\n"
                      "Higher values foreshorten distant geometry.\n"
                      "Viewing only -- does not affect outputs.",
                 font=("Arial", 7), fg="#888888", justify="left").pack(anchor="w")

        tk.Checkbutton(view_frame, text="Show camera labels", variable=self._show_labels_var,
                       font=("Arial", 9), command=self._refresh_preview).pack(anchor="w", pady=(4, 0))

        btn_row = tk.Frame(inner)
        btn_row.pack(fill="x", **pad)
        tk.Button(btn_row, text="Apply / Refresh", font=("Arial", 9, "bold"),
                  bg="#33558a", fg="white", activebackground="#44669b",
                  relief=tk.FLAT, padx=10, pady=3,
                  command=self._on_settings_changed).pack(side="left")
        tk.Button(btn_row, text="Reset view", font=("Arial", 8),
                  command=self._reset_view).pack(side="left", padx=6)
        tk.Button(btn_row, text="Help (Ctrl-H)", font=("Arial", 8),
                  command=self._open_help).pack(side="right")

        tk.Label(inner, textvariable=self._status_var, font=("Arial", 9),
                 fg="#334477", anchor="w", justify="left", wraplength=260).pack(fill="x", **pad)

        # ── 3D preview ────────────────────────────────────────────
        prev_frame = tk.LabelFrame(right, text="3D preview  (L-drag rotate, R-drag pan, wheel zoom)",
                                   font=("Arial", 9), padx=4, pady=4)
        prev_frame.pack(fill="both", expand=True)
        self._preview_canvas = tk.Canvas(prev_frame, bg="#ffffff", highlightthickness=0)
        self._preview_canvas.pack(fill="both", expand=True)
        c = self._preview_canvas
        c.bind("<ButtonPress-1>", self._on_drag_start)
        c.bind("<B1-Motion>", self._on_drag_motion)
        c.bind("<ButtonRelease-1>", self._on_drag_release)
        c.bind("<ButtonPress-3>", self._on_pan_start)
        c.bind("<B3-Motion>", self._on_pan_motion)
        c.bind("<ButtonRelease-3>", self._on_pan_release)
        c.bind("<MouseWheel>", self._on_wheel)
        c.bind("<Configure>", lambda e: self._refresh_preview())

        if win is not None:
            win.update_idletasks()
            win.minsize(900, 600)
            win.geometry("980x660")

        self._rebuild_grids()
        self._refresh_preview()

    def _make_paned(self, parent: tk.Frame, left_width: int = 280,
                    left_minsize: int = 200, right_minsize: int = 320):
        panes = tk.PanedWindow(parent, orient=tk.HORIZONTAL, sashwidth=6,
                               sashrelief=tk.RAISED, showhandle=True)
        panes.pack(fill="both", expand=True)
        left = tk.Frame(panes, bg="#f7f7f7")
        right = tk.Frame(panes, bg="#ffffff")
        panes.add(left, width=left_width, minsize=left_minsize)
        panes.add(right, minsize=right_minsize)

        def _apply(total_width: int) -> None:
            if total_width < 10:
                return
            try:
                sash_x, _ = panes.sash_coord(0)
                sash_w = int(panes.cget("sashwidth"))
            except Exception:
                return
            try:
                panes.paneconfigure(right, width=max(100, total_width - sash_x - sash_w - 2))
            except Exception:
                pass

        panes.bind("<Configure>",
                   lambda e: _apply(e.width) if e.widget is panes else None)
        panes.after(60, lambda: _apply(panes.winfo_width()))
        return panes, left, right

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

    def _on_settings_changed(self) -> None:
        self._rebuild_grids()
        self._auto_centred = False      # re-frame the view for the new grids
        self._refresh_preview()

    def _reset_view(self) -> None:
        self._cam_state.update(yaw=-0.9, pitch=0.7, zoom=1.0)
        self._auto_centred = False
        self._refresh_preview()

    # ── help ──────────────────────────────────────────────────────

    def _open_help(self) -> None:
        if self._help_popup is not None and self._help_popup.winfo_exists():
            self._help_popup.lift()
            return
        popup = tk.Toplevel()
        popup.title("Camera Layout Viewer - Help")
        popup.geometry("760x640")
        try:
            px, py = self.canvas.winfo_pointerxy()
            popup.geometry(f"+{px + 16}+{py + 16}")
        except Exception:
            pass
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
        tk.Button(body, text="Close", font=("Arial", 9),
                  command=popup.destroy).pack(anchor="e", pady=(8, 0))
        popup.bind("<Escape>", lambda _e: popup.destroy())
        popup.protocol("WM_DELETE_WINDOW", popup.destroy)
        self._help_popup = popup

    # ── grid construction ─────────────────────────────────────────

    def _rebuild_grids(self) -> None:
        xs_list = _parse_multi_sequence(self._grid_xs_var.get())
        ys_list = _parse_multi_sequence(self._grid_ys_var.get())
        zs_list = _parse_multi_sequence(self._grid_zs_var.get())

        self._grids = []
        self._grid_points = np.empty((0, 3), dtype=np.float64)

        if not (xs_list or ys_list or zs_list):
            self._grid_status_var.set("no grids")
            return

        counts = {len(xs_list), len(ys_list), len(zs_list)}
        if len(counts) != 1:
            self._grid_status_var.set(
                f"grid count mismatch: xs={len(xs_list)}, ys={len(ys_list)}, "
                f"zs={len(zs_list)} -- use the same number of ';' segments")
            return

        pts_accum = []
        bad = []
        for k, (xs, ys, zs) in enumerate(zip(xs_list, ys_list, zs_list)):
            if xs is None or ys is None or zs is None:
                bad.append(k + 1)
                continue
            self._grids.append({"xs": xs, "ys": ys, "zs": zs})
            gx, gy, gz = np.meshgrid(xs, ys, zs, indexing="ij")
            pts_accum.append(np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()]))

        if pts_accum:
            self._grid_points = np.concatenate(pts_accum, axis=0)

        msg = f"{len(self._grids)} grid(s), {self._grid_points.shape[0]} vertices"
        if bad:
            msg += f"   (grid {', '.join(str(b) for b in bad)} unparseable)"
        self._grid_status_var.set(msg)

    # ── projection pipeline ───────────────────────────────────────

    def _scene_points(self) -> np.ndarray:
        """Every point that should be framed by the auto-fit: grid
        vertices plus each camera's centre and frustum base."""
        parts = []
        if self._grid_points.size:
            parts.append(self._grid_points)
        for cam in self._cameras.values():
            parts.append(cam["centre"].reshape(1, 3))
            if cam.get("corners") is not None:
                parts.append(cam["corners"])
        if not parts:
            return np.empty((0, 3), dtype=np.float64)
        return np.concatenate(parts, axis=0)

    def _scene_radius_and_centre(self) -> tuple[float, np.ndarray]:
        pts = self._scene_points()
        if pts.shape[0] == 0:
            return 1.0, np.zeros(3)
        finite = pts[np.isfinite(pts).all(axis=1)]
        if finite.shape[0] == 0:
            return 1.0, np.zeros(3)
        centre = finite.mean(axis=0)
        radius = float(np.max(np.linalg.norm(finite - centre, axis=1)))
        return max(radius, 1e-6), centre

    def _rotation_matrix(self) -> np.ndarray:
        yaw, pitch = self._cam_state["yaw"], self._cam_state["pitch"]
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        ryaw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
        rpitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float64)
        return rpitch @ ryaw

    def _view_scale(self, radius: float) -> float:
        canvas_min = 200.0
        c = self._preview_canvas
        if c is not None and c.winfo_exists():
            canvas_min = min(float(c.winfo_width()), float(c.winfo_height()))
        return canvas_min * 0.36 * self._cam_state["zoom"] / max(radius, 1e-6)

    def _project(self, pts: np.ndarray, scale: float, radius: float) -> np.ndarray:
        """
        World -> screen. Orthographic when perspective intensity is 0,
        otherwise a depth-dependent shrink is applied.

        The rotated frame follows the same convention as the other 3D
        previews here: rot[:,0] is screen-right, rot[:,2] is screen-up
        (negated below), and rot[:,1] is depth INTO the screen, so
        larger rot[:,1] means farther away.
        """
        c = self._preview_canvas
        if c is None or not c.winfo_exists():
            return np.zeros((len(pts), 2))
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)

        target = np.asarray(self._cam_state["camera_target"], dtype=np.float64)
        rot = (self._rotation_matrix() @ (pts - target).T).T

        factor = self._perspective_factor(rot, radius)
        sx = rot[:, 0] * scale * factor
        sy = -rot[:, 2] * scale * factor

        cw, ch = float(c.winfo_width()), float(c.winfo_height())
        return np.column_stack([sx + cw / 2.0, sy + ch / 2.0])

    def _perspective_factor(self, rot: np.ndarray, radius: float) -> np.ndarray:
        """
        Per-point shrink factor implementing adjustable perspective.

        p = 0            -> factor == 1 everywhere (true orthographic,
                            identical to this project's other previews)
        p -> 1           -> eye distance approaches ~3x the scene radius,
                            giving pronounced wide-angle foreshortening

        The eye sits at distance d0 in front of the pivot, so a point at
        depth rot[:,1] is scaled by d0 / (d0 + depth). Points at or
        behind the eye are clamped to a small positive denominator so
        they blow up off-screen rather than flipping sign, which would
        otherwise render geometry mirrored through the origin.
        """
        p = float(self._persp_var.get())
        if p <= 1e-6:
            return np.ones(rot.shape[0], dtype=np.float64)
        d0 = max(radius, 1e-6) * (3.0 / max(p, 1e-6))
        denom = d0 + rot[:, 1]
        denom = np.where(denom < d0 * 1e-3, d0 * 1e-3, denom)
        return d0 / denom

    # ── mouse ─────────────────────────────────────────────────────

    def _on_drag_start(self, e) -> None:
        self._cam_state.update(dragging=True, last_x=e.x, last_y=e.y, button=1)

    def _on_drag_motion(self, e) -> None:
        if not self._cam_state["dragging"] or self._cam_state["button"] != 1:
            return
        self._cam_state["yaw"] += (e.x - self._cam_state["last_x"]) * 0.01
        self._cam_state["pitch"] += (e.y - self._cam_state["last_y"]) * 0.01
        self._cam_state["last_x"], self._cam_state["last_y"] = e.x, e.y
        self._refresh_preview()

    def _on_drag_release(self, _e) -> None:
        self._cam_state.update(dragging=False, button=None)

    def _on_pan_start(self, e) -> None:
        self._cam_state.update(dragging=True, last_x=e.x, last_y=e.y, button=3)

    def _on_pan_motion(self, e) -> None:
        if not self._cam_state["dragging"] or self._cam_state["button"] != 3:
            return
        dx = e.x - self._cam_state["last_x"]
        dy = e.y - self._cam_state["last_y"]
        radius, _ = self._scene_radius_and_centre()
        scale = self._view_scale(radius)
        if scale > 1e-12:
            rt = self._rotation_matrix().T
            delta = (rt[:, 0] * dx - rt[:, 2] * dy) / scale
            target = np.asarray(self._cam_state["camera_target"], dtype=np.float64)
            self._cam_state["camera_target"] = (target - delta).tolist()
        self._cam_state["last_x"], self._cam_state["last_y"] = e.x, e.y
        self._refresh_preview()

    def _on_pan_release(self, _e) -> None:
        self._cam_state.update(dragging=False, button=None)

    def _on_wheel(self, e) -> str:
        delta = int(e.delta / 120)
        if delta:
            self._cam_state["zoom"] = max(0.05, self._cam_state["zoom"] * (1.12 ** delta))
            self._refresh_preview()
        return "break"

    # ── drawing ───────────────────────────────────────────────────

    def _draw_grids(self, c: tk.Canvas, scale: float, radius: float) -> None:
        color = self._grid_color_var.get() or "gray"
        try:
            width = max(0.5, float(self._grid_width_var.get()))
        except (tk.TclError, ValueError):
            width = 1.0

        for g in self._grids:
            xs, ys, zs = g["xs"], g["ys"], g["zs"]
            # Lines along each axis: hold the other two fixed and sweep.
            for ys_i in ys:
                for zs_i in zs:
                    seg = np.column_stack([xs, np.full_like(xs, ys_i), np.full_like(xs, zs_i)])
                    self._draw_polyline(c, seg, scale, radius, color, width)
            for xs_i in xs:
                for zs_i in zs:
                    seg = np.column_stack([np.full_like(ys, xs_i), ys, np.full_like(ys, zs_i)])
                    self._draw_polyline(c, seg, scale, radius, color, width)
            if zs.size > 1:
                for xs_i in xs:
                    for ys_i in ys:
                        seg = np.column_stack([np.full_like(zs, xs_i), np.full_like(zs, ys_i), zs])
                        self._draw_polyline(c, seg, scale, radius, color, width)

    def _draw_polyline(self, c: tk.Canvas, pts3: np.ndarray, scale: float,
                       radius: float, color: str, width: float) -> None:
        if pts3.shape[0] < 2:
            return
        proj = self._project(pts3, scale, radius)
        flat = [v for xy in proj for v in xy]
        c.create_line(*flat, fill=color, width=width)

    def _draw_cameras(self, c: tk.Canvas, scale: float, radius: float) -> None:
        try:
            width = max(0.5, float(self._pyr_width_var.get()))
        except (tk.TclError, ValueError):
            width = 1.5

        for n, cam in sorted(self._cameras.items()):
            corners = cam.get("corners")
            if corners is None:
                continue
            color = self._cam_color_vars[n].get() or _CAM_DEFAULT_COLORS[n - 1]
            centre = cam["centre"]

            apex2 = self._project(centre.reshape(1, 3), scale, radius)[0]
            base2 = self._project(corners, scale, radius)

            # apex -> each base corner
            for k in range(4):
                c.create_line(apex2[0], apex2[1], base2[k][0], base2[k][1],
                              fill=color, width=width)
            # base loop
            for k in range(4):
                nxt = (k + 1) % 4
                c.create_line(base2[k][0], base2[k][1], base2[nxt][0], base2[nxt][1],
                              fill=color, width=width)

            # image-up notch: a small triangle above the top edge
            # (corners[0]=top-left, corners[1]=top-right), so a rolled
            # camera is visually obvious.
            top_mid = (corners[0] + corners[1]) / 2.0
            up_dir = corners[0] - corners[3]           # bottom-left -> top-left
            norm = np.linalg.norm(up_dir)
            if norm > 1e-9:
                tip = top_mid + (up_dir / norm) * (np.linalg.norm(corners[1] - corners[0]) * 0.22)
                tri = self._project(np.vstack([corners[0], corners[1], tip]), scale, radius)
                c.create_polygon([v for xy in tri for v in xy],
                                 fill="", outline=color, width=width)

            c.create_oval(apex2[0] - 3, apex2[1] - 3, apex2[0] + 3, apex2[1] + 3,
                          fill=color, outline="")

            if self._show_labels_var.get():
                c.create_text(apex2[0], apex2[1] - 12,
                              text=f"Cam {n}", fill=color, font=("Arial", 8, "bold"))

    def _draw_axis_triad(self, c: tk.Canvas) -> None:
        """World X/Y/Z orientation icon, lower-left, matching the style
        used by the other 3D previews in this project."""
        if not c.winfo_exists() or c.winfo_width() < 2:
            return
        r = self._rotation_matrix()
        corner = np.array([32.0, float(c.winfo_height()) - 32.0])
        axis_len = 22.0

        for axis, color, label in (
            (np.array([1.0, 0.0, 0.0]), "#cc0000", "X"),
            (np.array([0.0, 1.0, 0.0]), "#00aa00", "Y"),
            (np.array([0.0, 0.0, 1.0]), "#0000cc", "Z"),
        ):
            rot = r @ axis
            end = corner + np.array([rot[0], -rot[2]]) * axis_len
            c.create_line(corner[0], corner[1], end[0], end[1], fill=color, width=2)
            c.create_text(end[0] + (end[0] - corner[0]) * 0.2,
                          end[1] + (end[1] - corner[1]) * 0.2,
                          text=label, fill=color, font=("Arial", 8, "bold"))

        verts = self._CUBE_VERTS * (axis_len * 0.32)
        for face in sorted(self._CUBE_FACES, key=lambda f: float((r @ f["normal"])[1]))[:3]:
            pts = []
            for vi in face["verts"]:
                rv = r @ verts[vi]
                pts.append((corner[0] + rv[0], corner[1] - rv[2]))
            c.create_polygon(pts, fill=face["color"], outline="#333333", width=1.3)

    def _refresh_preview(self) -> None:
        c = self._preview_canvas
        if c is None or not c.winfo_exists():
            return
        c.delete("all")

        if not self._grids and not self._cameras:
            c.create_text(c.winfo_width() / 2, c.winfo_height() / 2,
                          text="Connect camera parameters, or define grids in the inspector",
                          fill="#999999", width=max(120, c.winfo_width() - 40))
            return

        radius, centre = self._scene_radius_and_centre()
        if not self._auto_centred:
            self._cam_state["camera_target"] = centre.tolist()
            self._auto_centred = True
        scale = self._view_scale(radius)

        self._draw_grids(c, scale, radius)
        self._draw_cameras(c, scale, radius)
        self._draw_axis_triad(c)

        p = float(self._persp_var.get())
        c.create_text(8, 10, anchor="nw", font=("Arial", 7), fill="#777777",
                      text=("projection: orthographic" if p <= 1e-6
                            else f"projection: perspective ({p:.2f})"))

    # ── compute ───────────────────────────────────────────────────

    @staticmethod
    def _coerce_imgsize(value) -> tuple[float, float] | None:
        try:
            a = np.asarray(value, dtype=np.float64).ravel()
            if a.size < 2:
                return None
            w, h = float(a[0]), float(a[1])
            return (w, h) if w > 0 and h > 0 else None
        except Exception:
            return None

    @staticmethod
    def _coerce_cmat(value) -> np.ndarray | None:
        try:
            return np.asarray(value, dtype=np.float64).reshape(3, 3)
        except Exception:
            return None

    @staticmethod
    def _coerce_vec3(value) -> np.ndarray | None:
        try:
            a = np.asarray(value, dtype=np.float64).ravel()
            return a[:3].reshape(3, 1) if a.size >= 3 else None
        except Exception:
            return None

    def _build_camera(self, n: int, inputs: dict, depth: float) -> dict | None:
        img = self._coerce_imgsize(inputs.get(f"imgsize{n}"))
        cmat = self._coerce_cmat(inputs.get(f"cmat{n}"))
        rvec = self._coerce_vec3(inputs.get(f"rvec{n}"))
        tvec = self._coerce_vec3(inputs.get(f"tvec{n}"))
        # dvec is deliberately read but unused -- the frustum is drawn
        # from the ideal pinhole model. Accepted so the pin can be wired
        # straight from a calibration node without dangling.
        _dvec = inputs.get(f"dvec{n}")

        if img is None or cmat is None or rvec is None or tvec is None:
            return None

        try:
            centre, axes = _camera_pose_in_world(rvec, tvec)
            w, h = img
            fov_h, fov_v = _fov_degrees(cmat, w, h)
            corners = _frustum_corners_world(centre, axes, cmat, w, h, depth)
        except Exception:
            return None

        return {"centre": centre, "axes": axes, "cmat": cmat,
                "w": w, "h": h, "fov": (fov_h, fov_v), "corners": corners}

    def compute(self, inputs: dict) -> dict:
        self._init_state()
        self._rebuild_grids()

        # Pass 1: poses only, with a provisional depth, so we can size
        # the frustums against the whole scene (grids + baseline) rather
        # than against an arbitrary constant.
        self._cameras = {}
        provisional = {}
        for n in (1, 2):
            cam = self._build_camera(n, inputs, depth=1.0)
            if cam is not None:
                provisional[n] = cam

        if provisional or self._grid_points.size:
            extent_pts = []
            if self._grid_points.size:
                extent_pts.append(self._grid_points)
            for cam in provisional.values():
                extent_pts.append(cam["centre"].reshape(1, 3))
            all_pts = np.concatenate(extent_pts, axis=0)
            finite = all_pts[np.isfinite(all_pts).all(axis=1)]
            if finite.shape[0] >= 2:
                span = float(np.max(np.linalg.norm(finite - finite.mean(axis=0), axis=1)))
            else:
                span = 1.0
            base_depth = max(span * 0.25, 1e-6)
        else:
            base_depth = 1.0

        try:
            user_scale = max(1e-3, float(self._pyr_scale_var.get()))
        except (tk.TclError, ValueError):
            user_scale = 1.0
        depth = base_depth * user_scale

        # Pass 2: rebuild with the real depth.
        for n in (1, 2):
            cam = self._build_camera(n, inputs, depth=depth)
            if cam is not None:
                self._cameras[n] = cam

        parts = []
        for n in (1, 2):
            cam = self._cameras.get(n)
            if cam is None:
                parts.append(f"cam{n}: --")
            else:
                fh, fv = cam["fov"]
                parts.append(f"cam{n}: {int(cam['w'])}x{int(cam['h'])}, "
                             f"FOV {fh:.1f}°x{fv:.1f}°")
        self._cam_status_var.set("\n".join(parts))

        n_cams = len(self._cameras)
        if n_cams == 0 and not self._grids:
            self._status_var.set("waiting for camera inputs / grids")
        else:
            extra = ""
            if n_cams == 2:
                baseline = float(np.linalg.norm(
                    self._cameras[1]["centre"] - self._cameras[2]["centre"]))
                extra = f"   baseline {baseline:.4g}"
            self._status_var.set(
                f"ok  {n_cams} camera(s), {len(self._grids)} grid(s){extra}")
            self.set_status("ok", "#33558a")

        self._auto_centred = False
        self._refresh_preview()

        out: dict = {
            "grid_points": self._grid_points,
            "n_cameras": float(n_cams),
        }
        for n in (1, 2):
            cam = self._cameras.get(n)
            if cam is not None:
                out[f"cam{n}_pos"] = cam["centre"].copy()
                out[f"cam{n}_axes"] = cam["axes"].copy()
        return out

    # ── serialization ─────────────────────────────────────────────

    def get_params(self) -> dict:
        return {
            "grid_xs": self._grid_xs_var.get(),
            "grid_ys": self._grid_ys_var.get(),
            "grid_zs": self._grid_zs_var.get(),
            "pyr_scale": self._pyr_scale_var.get(),
            "pyr_width": self._pyr_width_var.get(),
            "grid_color": self._grid_color_var.get(),
            "grid_width": self._grid_width_var.get(),
            "perspective": self._persp_var.get(),
            "show_labels": bool(self._show_labels_var.get()),
            "cam1_color": self._cam_color_vars[1].get(),
            "cam2_color": self._cam_color_vars[2].get(),
        }

    def set_params(self, params: dict) -> None:
        self._init_state()
        self._grid_xs_var.set(str(params.get("grid_xs", "0:10:100")))
        self._grid_ys_var.set(str(params.get("grid_ys", "0:10:60")))
        self._grid_zs_var.set(str(params.get("grid_zs", "0")))
        self._pyr_scale_var.set(float(params.get("pyr_scale", 1.0)))
        self._pyr_width_var.set(float(params.get("pyr_width", 1.5)))
        self._grid_color_var.set(str(params.get("grid_color", "gray")))
        self._grid_width_var.set(float(params.get("grid_width", 1.0)))
        self._persp_var.set(float(params.get("perspective", 0.0)))
        self._show_labels_var.set(bool(params.get("show_labels", True)))
        self._cam_color_vars[1].set(str(params.get("cam1_color", _CAM_DEFAULT_COLORS[0])))
        self._cam_color_vars[2].set(str(params.get("cam2_color", _CAM_DEFAULT_COLORS[1])))

    def close_inspector(self) -> None:
        super().close_inspector()
        self._preview_canvas = None

    def on_destroy(self) -> None:
        if self._help_popup is not None and self._help_popup.winfo_exists():
            self._help_popup.destroy()
        super().on_destroy()