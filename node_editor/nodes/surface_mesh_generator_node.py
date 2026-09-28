# node_editor/nodes/surface_mesh_generator_node.py

import math
import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

from node_editor.base_node import BaseNode
from node_editor.pin_types import PinSchema, PinDef, PinType
from node_editor.execution import ExecutionMode


_HELP_TEXT = """\
Surface Mesh Generator
======================

PURPOSE
-------
Fits an analytic surface (plane or cylinder) to a small set of measured
3D points, then generates a dense structured mesh of points lying
exactly on that fitted surface. Reports how far each given point sits
from the fitted surface, so you can judge the fit quality.

Typical use: you have a handful of surveyed/targeted points on a
specimen (a wall, a column, a pipe), and you want a dense regular grid
over that surface to seed tracking, interpolation, or strain analysis.

INPUT PINS
----------
points   ARRAY (optional)
    (np, 3) array of measured 3D world points. Minimum 3 for a plane,
    5 for a cylinder (see MINIMUM POINTS below). If this pin is not
    connected, the inspector's text box is used instead.

OUTPUT PINS
-----------
mesh_points      ARRAY (M, N, 3)  the dense generated surface mesh
a1_array         ARRAY (M,)       a1 samples (cylinder: RADIANS, unchanged)
a2_array         ARRAY (N,)       the a2 parameter samples
proj_errors      ARRAY (np,)      per-point distance to the fitted
                                  surface, in world units
rms_error        SCALAR           RMS of proj_errors
surface_origin   ARRAY (3,)       xo of the fitted, parameterized surface
surface_axes     ARRAY (3, 3)     [vx | vy | vz] as COLUMNS
radius           SCALAR           cylinder radius (0.0 for a plane)

SURFACE PARAMETERIZATION
------------------------
Plane:     P(a1, a2) = xo + a1*vx + a2*vy
           a1, a2 are in LENGTH units. vz = cross(vx, vy) is the
           plane normal.

Cylinder:  P(a1, a2) = xo + R*cos(a1)*vx + R*sin(a1)*vy + a2*vz
           a1 is an ANGLE IN RADIANS internally and on the output pin;
           the inspector uses DEGREES. a2 is height along the
           axis, in length units. vz = cross(vx, vy) is the cylinder
           axis. Note the R*sin(a1) term is multiplied by vy -- a
           common transcription slip is to drop that factor, which
           would add a scalar to a vector.

The axes are orthonormal and normally right-handed. Reverse theta flips
vy only, making the cylinder triad left-handed while preserving +height.

MINIMUM POINTS
--------------
Plane:    3 points (3 DOF: a point on the plane + 2 for the normal).
Cylinder: 5 points. A general cylinder has 5 DOF -- axis direction (2),
          axis position perpendicular to itself (2), radius (1) -- and
          each point gives exactly one constraint (its distance to the
          axis must equal R). With 4 points the fit is underdetermined
          and admits a one-parameter family of solutions. Fits with
          fewer than ~8 well-spread points are often poorly conditioned
          even when nominally solvable, so treat a suspiciously small
          RMS from few points with caution.

FITTING METHOD
--------------
Plane:    Total-least-squares via SVD of the mean-centred points. The
          normal is the singular vector with the smallest singular
          value. Errors are perpendicular distances to the plane.

Cylinder: Nonlinear, solved without any extra dependencies as a search
          over axis DIRECTION only. For any candidate direction d, the
          points are projected onto the plane perpendicular to d and an
          algebraic (Kasa) 2D circle fit gives the axis position and
          radius in closed form; the residual of that circle fit is the
          cost. The direction is refined locally from the approximate
          height vector using shrinking angular steps. This selects a
          nearby solution rather than scanning all possible cylinder axes.
          Errors are |distance-to-axis - R|. Sparse/degenerate data can
          still have ambiguous fits: the hint is not a uniqueness guarantee.

PARAMETER ORIGIN AND ORIENTATION
---------------------------------
The fit alone does not pin down WHERE a1=0, a2=0 sits or which way the
axes point, so you choose that from the given points (indices are
0-BASED into the input array):

Plane
  Origin point index   this point, projected onto the fitted plane,
                       becomes xo (i.e. a1=0, a2=0).
  X-axis point index   vx points from xo toward this point's
                       projection.
  +Y side point index  optional (-1 = ignore). If given, the normal is
                       flipped if needed so this point lies on the +a2
                       side. Without it, the normal's sign comes from
                       the SVD and is arbitrary but deterministic.

Cylinder
  Origin point index   sets BOTH a1=0 (its angular position around the
                       axis) and a2=0 (its height along the axis).
    Approximate +height vector (X Y Z)
                                             nonzero world-space local Z direction, default
                                             0 0 1. Need not be unit length or exact. The fit
                                             starts here and refines the direction locally.
    Reverse theta        reverses the angular direction without reversing
                                             the height direction supplied by the vector.

MESH RANGE
----------
Set start, end and count for a1 (M values) and a2 (N values); the mesh
is M x N. Cylinder theta uses degrees, default start 0 and end 90.
Output a1_array remains radians for compatibility. Counts are positive
integers. A count of 1 produces only the start sample (interval is 0).

Interval is signed surface spacing in world length units:
    Plane a1/a2 and cylinder a2: (end-start)/(count-1)
    Cylinder a1: R*(end-start)*pi/180/(M-1) (arc length, not chord)
Editing start/end/count updates interval immediately. Editing interval
updates end while keeping start/count fixed. Fit a cylinder first so its
radius is available; until then its arc-length interval is blank.
Negative intervals generate descending ranges.

"Auto-fit ranges" sets each range to span the given points
(with a small margin), which is usually a good starting point --
especially for a cylinder, where guessing an angular range by hand is
awkward.

MEMORY
------
The output is M*N*3 floats. 8000x8000 float64 is about 1.5 GB, which
will fail or thrash on many machines. The inspector shows a live
estimate; float32 halves it. Generation is refused above the memory
cap unless you tick the override, and the mesh is cached so repeated
compute() calls with unchanged settings do not regenerate it.

3D PREVIEW
----------
Draws the given points plus a COARSE M_draw x N_draw resample of the
surface (default 20x20) -- never the full dense mesh, which would be
millions of canvas items. Changing any range/count redraws it.
  Left-drag    rotate (orbits the current pan target)
  Right-drag   pan
  Wheel        zoom
Perspective Intensity (0 to 1, default 0.35) matches Structural Mesh
Viewer: 0 = orthographic, 1 = strong near/far foreshortening. Changes
redraw immediately and affect only the preview, not mesh coordinates.
A world X/Y/Z axes icon sits at the lower left.

KEYBOARD SHORTCUT
-----------------
Ctrl-H / Ctrl-h -- show this help window.
"""

_COMMON_COLORS = [
    "red", "green", "blue", "black", "orange",
    "purple", "brown", "magenta", "cyan", "gray",
]
_MARKERS = ["o", "+", "*", "x", "s", "D", "^", "v"]

# Refuse to allocate more than this unless the user overrides.
_MEMORY_CAP_BYTES = 2.0 * (1024 ** 3)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_points_block(text: str) -> tuple[np.ndarray | None, int]:
    """One point per line, 'x y z' or 'x,y,z'. Blank and '#' lines are
    skipped; malformed lines are counted and skipped individually."""
    rows, skipped = [], 0
    for line in (text or "").strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            skipped += 1
            continue
        if len(vals) != 3:
            skipped += 1
            continue
        rows.append(vals)
    if not rows:
        return None, skipped
    return np.array(rows, dtype=np.float64), skipped


def _coerce_points(arr) -> np.ndarray:
    """Coerce to (np,3) float64, accepting (np,3), (np,1,3) or a flat
    array whose length is a multiple of 3. Drops non-finite rows."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 3 and a.shape[1:] == (1, 3):
        a = a.reshape(-1, 3)
    elif a.ndim == 1:
        if a.size == 0 or a.size % 3 != 0:
            raise ValueError(f"flat array of size {a.size} is not a multiple of 3")
        a = a.reshape(-1, 3)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"expected shape (np,3), got {a.shape}")
    return a[np.isfinite(a).all(axis=1)]


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        raise ValueError("cannot normalize a zero-length vector")
    return np.asarray(v, dtype=np.float64) / n


def _perp_basis(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal vectors spanning the plane perpendicular to d."""
    d = _unit(d)
    # Pick the world axis least aligned with d, so the cross product is
    # well conditioned no matter which way d happens to point.
    seed = np.zeros(3)
    seed[int(np.argmin(np.abs(d)))] = 1.0
    e1 = _unit(np.cross(d, seed))
    e2 = _unit(np.cross(d, e1))
    return e1, e2


def _fibonacci_hemisphere(n: int) -> np.ndarray:
    """n roughly-uniform unit directions on a hemisphere. A cylinder
    axis and its negation describe the same cylinder, so only half the
    sphere needs scanning."""
    i = np.arange(n, dtype=np.float64) + 0.5
    z = i / n                       # (0, 1] -> upper hemisphere
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = np.pi * (1.0 + 5.0 ** 0.5) * i
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _sph_to_dir(theta: float, phi: float) -> np.ndarray:
    st = math.sin(theta)
    return np.array([st * math.cos(phi), st * math.sin(phi), math.cos(theta)])


def _dir_to_sph(d: np.ndarray) -> tuple[float, float]:
    d = _unit(d)
    return math.acos(float(np.clip(d[2], -1.0, 1.0))), math.atan2(float(d[1]), float(d[0]))


# ---------------------------------------------------------------------------
# Plane fit
# ---------------------------------------------------------------------------

def _fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Total-least-squares plane fit via SVD of the mean-centred points.
    Returns (centroid, normal, per-point perpendicular distances).
    """
    if points.shape[0] < 3:
        raise ValueError(f"a plane fit needs at least 3 points, got {points.shape[0]}")
    centroid = points.mean(axis=0)
    centred = points - centroid
    # Smallest right-singular vector = direction of least variance = normal.
    _u, _s, vt = np.linalg.svd(centred, full_matrices=False)
    normal = _unit(vt[-1])
    errors = np.abs(centred @ normal)
    return centroid, normal, errors


# ---------------------------------------------------------------------------
# Cylinder fit
# ---------------------------------------------------------------------------

def _fit_circle_2d(u: np.ndarray, v: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """
    Kasa algebraic circle fit: u^2+v^2 = 2*uc*u + 2*vc*v + (R^2-uc^2-vc^2)
    is LINEAR in the three unknowns, so this is a plain least-squares
    solve with no iteration. Returns (uc, vc, R, radial residuals).
    """
    a = np.column_stack([u, v, np.ones_like(u)])
    b = u * u + v * v
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    uc = 0.5 * float(sol[0])
    vc = 0.5 * float(sol[1])
    r = math.sqrt(max(float(sol[2]) + uc * uc + vc * vc, 0.0))
    resid = np.abs(np.hypot(u - uc, v - vc) - r)
    return uc, vc, r, resid


def _cylinder_cost(direction: np.ndarray, points: np.ndarray):
    """Cost of a candidate axis direction: project perpendicular to it,
    fit a circle, return the summed squared radial residual."""
    e1, e2 = _perp_basis(direction)
    u = points @ e1
    v = points @ e2
    uc, vc, r, resid = _fit_circle_2d(u, v)
    return float(np.sum(resid ** 2)), (uc, vc, r, e1, e2, resid)


def _fit_cylinder(points: np.ndarray,
                  coarse_n: int = 400,
                  refine_iters: int = 120,
                  axis_hint: np.ndarray | None = None):
    """
    Fit a cylinder with no dependency beyond numpy.

    Only the axis DIRECTION is searched (2 DOF); for each candidate the
    axis position and radius follow in closed form from a 2D circle fit
    in the perpendicular plane. With axis_hint, refine locally from that
    nonzero vector using angular steps in the tangent plane. Without a
    hint, retain the legacy global hemisphere scan and angular refinement.

    Returns (axis_point, axis_dir, radius, per-point |dist-to-axis - R|).
    """
    n_pts = points.shape[0]
    if n_pts < 5:
        raise ValueError(
            f"a cylinder fit needs at least 5 points (5 DOF: axis direction 2, "
            f"axis position 2, radius 1); got {n_pts}")

    centroid = points.mean(axis=0)
    centred = points - centroid

    if axis_hint is not None:
        hint = np.asarray(axis_hint, dtype=np.float64)
        if hint.shape != (3,) or not np.isfinite(hint).all():
            raise ValueError("approximate height vector must contain three finite values")
        direction = _unit(hint)
        best_cost, best_extra = _cylinder_cost(direction, centred)
        # Stay in the seed's local basin rather than picking a different
        # cylinder from a global scan. Tangent steps work even at the poles.
        step = math.radians(5.0)
        for _ in range(refine_iters):
            e1, e2 = _perp_basis(direction)
            candidates = []
            for angle in np.arange(8) * (math.pi / 4):
                tangent = math.cos(angle) * e1 + math.sin(angle) * e2
                candidate = _unit(math.cos(step) * direction + math.sin(step) * tangent)
                # Preserve the positive direction specified by the user.
                if np.dot(candidate, hint) <= 0:
                    continue
                cost, extra = _cylinder_cost(candidate, centred)
                candidates.append((cost, candidate, extra))
            cost, candidate, extra = min(candidates, key=lambda item: item[0])
            if cost < best_cost:
                best_cost, direction, best_extra = cost, candidate, extra
            else:
                step *= 0.5
                if step < 1e-9:
                    break
        uc, vc, radius, e1, e2, resid = best_extra
        return centroid + uc * e1 + vc * e2, direction, float(radius), resid

    best_cost, best_dir, best_extra = np.inf, None, None
    for d in _fibonacci_hemisphere(coarse_n):
        try:
            cost, extra = _cylinder_cost(d, centred)
        except ValueError:
            continue
        if cost < best_cost:
            best_cost, best_dir, best_extra = cost, d, extra

    if best_dir is None:
        raise ValueError("cylinder axis search failed to find any valid direction")

    theta, phi = _dir_to_sph(best_dir)
    step = math.pi / max(4, int(math.sqrt(coarse_n)))
    for _ in range(refine_iters):
        improved = False
        for dth, dph in ((step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step)):
            cand = _sph_to_dir(theta + dth, phi + dph)
            try:
                cost, extra = _cylinder_cost(cand, centred)
            except ValueError:
                continue
            if cost < best_cost:
                best_cost, best_extra = cost, extra
                theta, phi = theta + dth, phi + dph
                improved = True
                break
        if not improved:
            step *= 0.5
            if step < 1e-9:
                break

    uc, vc, radius, e1, e2, resid = best_extra
    axis_dir = _unit(_sph_to_dir(theta, phi))
    # Circle centre lives in the perpendicular plane; lift it back out.
    axis_point = centroid + uc * e1 + vc * e2
    return axis_point, axis_dir, float(radius), resid


# ---------------------------------------------------------------------------
# SurfaceMeshGeneratorNode
# ---------------------------------------------------------------------------

class SurfaceMeshGeneratorNode(BaseNode):
    """
    Fits a plane or cylinder to a few measured 3D points and generates a
    dense structured mesh over the fitted surface.

    See _HELP_TEXT (Ctrl-H) for the full reference.
    """

    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE = "surface_mesh_generator"
    DISPLAY_NAME = "Surface Mesh Generator"
    CATEGORY = "process"
    SEARCH_KEYWORDS = ("surface", "plane", "cylinder", "fit", "mesh",
                       "generate", "parametric", "least squares", "grid")
    NODE_WIDTH = 220
    NODE_HEIGHT = 130

    HELP_TEXT = _HELP_TEXT

    _BODY_BG = "#f3fbf3"
    _OUTLINE = "#3a8a4a"
    _TITLE_FG = "#1c5528"
    _STATUS_FG = "#466"

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
        return PinSchema(
            inputs=[
                PinDef("points", PinType.ARRAY, "pts", optional=True),
            ],
            outputs=[
                PinDef("mesh_points", PinType.ARRAY, "mesh"),
                PinDef("a1_array", PinType.ARRAY, "a1"),
                PinDef("a2_array", PinType.ARRAY, "a2"),
                PinDef("proj_errors", PinType.ARRAY, "errs"),
                PinDef("rms_error", PinType.SCALAR, "rms"),
                PinDef("surface_origin", PinType.ARRAY, "xo"),
                PinDef("surface_axes", PinType.ARRAY, "axes"),
                PinDef("radius", PinType.SCALAR, "R"),
            ],
        )

    def get_help_text(self) -> str:
        return _HELP_TEXT

    # ── state ─────────────────────────────────────────────────────

    def _init_state(self) -> None:
        if hasattr(self, "_surface_type_var"):
            return

        self._surface_type_var = tk.StringVar(value="plane")

        # parameterization choices (0-based indices into the point array)
        self._origin_idx_var = tk.IntVar(value=0)
        self._xaxis_idx_var = tk.IntVar(value=1)
        self._yside_idx_var = tk.IntVar(value=-1)
        self._axis_hint_vars = [tk.StringVar(value=v) for v in ("0", "0", "1")]
        self._legacy_height_idx = None
        for var in self._axis_hint_vars:
            var.trace_add("write", lambda *_args: setattr(self, "_legacy_height_idx", None))
        self._reverse_theta_var = tk.BooleanVar(value=False)

        # mesh ranges
        self._a1_start_var = tk.DoubleVar(value=0.0)
        self._a1_end_var = tk.DoubleVar(value=1.0)
        self._a1_count_var = tk.IntVar(value=2000)
        self._a2_start_var = tk.DoubleVar(value=0.0)
        self._a2_end_var = tk.DoubleVar(value=1.0)
        self._a2_count_var = tk.IntVar(value=2000)
        self._a1_interval_var = tk.StringVar(value="")
        self._a2_interval_var = tk.StringVar(value="")
        self._range_syncing = False
        self._range_surface_type = "plane"
        self._dtype_var = tk.StringVar(value="float64")
        self._mem_override_var = tk.BooleanVar(value=False)

        # preview resampling
        self._mdraw_var = tk.IntVar(value=20)
        self._ndraw_var = tk.IntVar(value=20)
        self._perspective_var = tk.DoubleVar(value=0.35)
        self._cam_last_radius = 1.0

        # appearance
        self._given_marker_var = tk.StringVar(value="o")
        self._given_size_var = tk.DoubleVar(value=5.0)
        self._given_color_var = tk.StringVar(value="red")
        self._surf_marker_var = tk.StringVar(value="+")
        self._surf_size_var = tk.DoubleVar(value=2.5)
        self._surf_color_var = tk.StringVar(value="blue")
        self._surf_lines_var = tk.BooleanVar(value=True)
        self._surf_line_color_var = tk.StringVar(value="#88aacc")

        self._points_text = ""
        self._points_widget: tk.Text | None = None
        self._points_from_pin = False

        self._status_var = tk.StringVar(value="waiting for points")
        self._fit_var = tk.StringVar(value="")
        self._err_var = tk.StringVar(value="")
        self._mem_var = tk.StringVar(value="")

        # resolved data
        self._points = np.empty((0, 3), dtype=np.float64)
        self._xo = None
        self._vx = None
        self._vy = None
        self._vz = None
        self._radius = 0.0
        self._errors = np.empty((0,), dtype=np.float64)
        self._preview_grid = np.empty((0, 0, 3), dtype=np.float64)

        # dense-mesh cache: regenerating 8000x8000 on every propagation
        # would be ruinous, so key it on everything that affects it.
        self._mesh_cache = None
        self._mesh_cache_key = None

        self._preview_canvas: tk.Canvas | None = None
        self._help_popup: tk.Toplevel | None = None
        self._cam = {
            "yaw": -0.9, "pitch": 0.7, "zoom": 1.0,
            "target": [0.0, 0.0, 0.0],
            "dragging": False, "last_x": 0, "last_y": 0, "button": None,
        }
        self._auto_centred = False

        for axis in (1, 2):
            for field in ("start", "end", "count"):
                getattr(self, f"_a{axis}_{field}_var").trace_add(
                    "write", lambda *_args, a=axis: self._range_changed(a))
            getattr(self, f"_a{axis}_interval_var").trace_add(
                "write", lambda *_args, a=axis: self._range_changed(a, from_interval=True))
        self._update_intervals()

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

        for frac, var in ((0.40, self._fit_var), (0.60, self._err_var)):
            lbl = tk.Label(self.canvas, textvariable=var, font=("Arial", 8),
                           bg=self._BODY_BG, fg=self._TITLE_FG,
                           wraplength=w - 12, justify="center")
            self.canvas.create_window(x + w / 2, y + h * frac, window=lbl,
                                      tags=(self.node_id,))

        status_lbl = tk.Label(self.canvas, textvariable=self._status_var,
                              font=("Arial", 7), bg=self._BODY_BG,
                              fg=self._STATUS_FG, wraplength=w - 12,
                              justify="center")
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

        _panes, left, right = self._make_paned(parent, left_width=340, left_minsize=280)
        pad = {"padx": 6, "pady": 4}
        inner = self._make_scrollable(left)

        # ── surface type ──────────────────────────────────────────
        tf = tk.LabelFrame(inner, text="Surface type", font=("Arial", 9), **pad)
        tf.pack(fill="x", **pad)
        for val, label in (("plane", "Plane   (a1, a2 in length units)"),
                           ("cylinder", "Cylinder   (a1 = theta in DEGREES, a2 = height)")):
            tk.Radiobutton(tf, text=label, variable=self._surface_type_var, value=val,
                           font=("Arial", 9),
                           command=self._on_type_changed).pack(anchor="w")

        # ── given points ──────────────────────────────────────────
        pf = tk.LabelFrame(inner, text="Given points (pin overrides text box)",
                           font=("Arial", 9), **pad)
        pf.pack(fill="x", **pad)
        tk.Label(pf, text="One point per line:  x y z   or   x,y,z",
                 font=("Arial", 7), fg="#888888").pack(anchor="w")
        self._points_widget = tk.Text(pf, height=6, font=("Courier", 8), wrap=tk.NONE)
        self._points_widget.pack(fill="x")
        self._points_widget.insert("1.0", self._points_text)
        if self._points_from_pin:
            self._points_widget.configure(state="disabled")

        # ── parameterization ──────────────────────────────────────
        self._param_frame = tk.LabelFrame(inner, text="Parameterization (point indices are 0-based)",
                                          font=("Arial", 9), **pad)
        self._param_frame.pack(fill="x", **pad)
        self._build_param_rows(self._param_frame)

        # ── ranges ────────────────────────────────────────────────
        rf = tk.LabelFrame(inner, text="Mesh range", font=("Arial", 9), **pad)
        rf.pack(fill="x", **pad)

        self._a1_label = tk.Label(rf, text="a1:", font=("Arial", 9, "bold"), anchor="w")
        self._a1_label.pack(anchor="w")
        self._range_row(rf, self._a1_start_var, self._a1_end_var, self._a1_count_var, "M", self._a1_interval_var)
        self._a2_label = tk.Label(rf, text="a2:", font=("Arial", 9, "bold"), anchor="w")
        self._a2_label.pack(anchor="w", pady=(4, 0))
        self._range_row(rf, self._a2_start_var, self._a2_end_var, self._a2_count_var, "N", self._a2_interval_var)

        dt_row = tk.Frame(rf)
        dt_row.pack(fill="x", pady=(6, 0))
        tk.Label(dt_row, text="dtype:", font=("Arial", 8)).pack(side="left")
        dt = ttk.Combobox(dt_row, textvariable=self._dtype_var,
                          values=["float64", "float32"], state="readonly", width=9)
        dt.pack(side="left", padx=(2, 8))
        dt.bind("<<ComboboxSelected>>", lambda e: self._update_memory_estimate())
        tk.Button(dt_row, text="Auto-fit ranges", font=("Arial", 8),
                  command=self._auto_fit_ranges).pack(side="left")

        tk.Label(rf, textvariable=self._mem_var, font=("Arial", 8),
                 fg="#885500", anchor="w", justify="left",
                 wraplength=290).pack(fill="x", pady=(4, 0))
        tk.Checkbutton(rf, text=f"Allow >{_MEMORY_CAP_BYTES / (1024**3):.1f} GB allocation",
                       variable=self._mem_override_var, font=("Arial", 8),
                       command=self._update_memory_estimate).pack(anchor="w")

        # ── preview resampling ────────────────────────────────────
        df = tk.LabelFrame(inner, text="Preview resampling", font=("Arial", 9), **pad)
        df.pack(fill="x", **pad)
        dr = tk.Frame(df)
        dr.pack(fill="x")
        tk.Label(dr, text="M_draw:", font=("Arial", 8)).pack(side="left")
        tk.Spinbox(dr, from_=2, to=200, textvariable=self._mdraw_var, width=6,
                   font=("Arial", 8), command=self._refresh_preview).pack(side="left", padx=(2, 8))
        tk.Label(dr, text="N_draw:", font=("Arial", 8)).pack(side="left")
        tk.Spinbox(dr, from_=2, to=200, textvariable=self._ndraw_var, width=6,
                   font=("Arial", 8), command=self._refresh_preview).pack(side="left", padx=(2, 0))
        tk.Label(df, text="The preview never draws the full dense mesh.",
                 font=("Arial", 7), fg="#888888").pack(anchor="w")

        cam_frame = tk.LabelFrame(inner, text="Camera", font=("Arial", 9), **pad)
        cam_frame.pack(fill="x", **pad)
        tk.Label(cam_frame, text="Perspective Intensity:", font=("Arial", 9)).pack(anchor="w")
        tk.Scale(cam_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
             variable=self._perspective_var,
             command=lambda _v: self._refresh_preview()).pack(fill="x")
        tk.Label(cam_frame,
             text="0 = orthographic (parallel projection)\n1 = strong perspective (near/far foreshortening)",
             font=("Arial", 7), fg="#888888", justify="left").pack(anchor="w")

        # ── appearance ────────────────────────────────────────────
        af = tk.LabelFrame(inner, text="Appearance", font=("Arial", 9), **pad)
        af.pack(fill="x", **pad)
        self._style_row(af, "Given pts:", self._given_marker_var,
                        self._given_size_var, self._given_color_var)
        self._style_row(af, "Surface:", self._surf_marker_var,
                        self._surf_size_var, self._surf_color_var)
        lr = tk.Frame(af)
        lr.pack(fill="x", pady=(2, 0))
        tk.Checkbutton(lr, text="Grid lines", variable=self._surf_lines_var,
                       font=("Arial", 8), command=self._refresh_preview).pack(side="left")
        lc = ttk.Combobox(lr, textvariable=self._surf_line_color_var,
                          values=_COMMON_COLORS + ["#88aacc"], width=9)
        lc.pack(side="left", padx=(4, 0))
        lc.bind("<<ComboboxSelected>>", lambda e: self._refresh_preview())
        lc.bind("<FocusOut>", lambda e: self._refresh_preview())

        # ── actions / status ──────────────────────────────────────
        br = tk.Frame(inner)
        br.pack(fill="x", **pad)
        tk.Button(br, text="Apply", font=("Arial", 9, "bold"),
                  bg="#3a8a4a", fg="white", activebackground="#4a9a5a",
                  relief=tk.FLAT, padx=12, pady=3,
                  command=self._on_apply).pack(side="left")
        tk.Button(br, text="Reset view", font=("Arial", 8),
                  command=self._reset_view).pack(side="left", padx=6)
        tk.Button(br, text="Help (Ctrl-H)", font=("Arial", 8),
                  command=self._open_help).pack(side="right")

        sf = tk.LabelFrame(inner, text="Fit result", font=("Arial", 9), **pad)
        sf.pack(fill="x", **pad)
        for var, color in ((self._fit_var, "#1c5528"),
                           (self._err_var, "#446"),
                           (self._status_var, "#334")):
            tk.Label(sf, textvariable=var, font=("Arial", 9), fg=color,
                     anchor="w", justify="left", wraplength=290).pack(fill="x")

        # ── 3D preview ────────────────────────────────────────────
        prev = tk.LabelFrame(right, text="3D preview  (L-drag rotate, R-drag pan, wheel zoom)",
                             font=("Arial", 9), padx=4, pady=4)
        prev.pack(fill="both", expand=True)
        self._preview_canvas = tk.Canvas(prev, bg="#ffffff", highlightthickness=0)
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
            win.minsize(940, 640)
            win.geometry("1020x700")

        self._on_type_changed(redraw=False)
        self._update_memory_estimate()
        self._refresh_preview()

    def _range_row(self, parent, start_var, end_var, count_var, count_label, interval_var) -> None:
        row = tk.Frame(parent)
        row.pack(fill="x", pady=1)
        for label, var, width in (("start:", start_var, 9), ("end:", end_var, 9),
                                  (f"{count_label}:", count_var, 7)):
            tk.Label(row, text=label, font=("Arial", 8)).pack(side="left")
            ent = tk.Entry(row, textvariable=var, width=width, font=("Arial", 8))
            if var is count_var:
                ent.configure(validate="key", validatecommand=(
                    ent.register(lambda text: text.isdecimal() and int(text) >= 1), "%P"))
            ent.pack(side="left", padx=(2, 6))
            ent.bind("<Return>", lambda e: self._on_apply())
            ent.bind("<FocusOut>", lambda e: self._update_memory_estimate())

        interval_row = tk.Frame(parent)
        interval_row.pack(fill="x", pady=1)
        tk.Label(interval_row, text="interval:", font=("Arial", 8)).pack(side="left")
        ent = tk.Entry(interval_row, textvariable=interval_var, width=14, font=("Arial", 8))
        ent.pack(side="left", padx=(2, 6))
        ent.bind("<Return>", lambda e: self._on_apply())
        tk.Label(interval_row, text="surface length units", font=("Arial", 7)).pack(side="left")

    def _interval_factor(self, axis: int) -> float:
        if axis == 1 and self._surface_type_var.get() == "cylinder":
            if self._radius <= 0:
                raise ValueError("fit the cylinder before editing its arc-length interval")
            return self._radius * math.pi / 180.0
        return 1.0

    def _range_changed(self, axis: int, from_interval: bool = False) -> None:
        if self._range_syncing:
            return
        self._range_syncing = True
        interval_var = getattr(self, f"_a{axis}_interval_var")
        try:
            start = float(getattr(self, f"_a{axis}_start_var").get())
            end_var = getattr(self, f"_a{axis}_end_var")
            count = int(getattr(self, f"_a{axis}_count_var").get())
            if count < 1 or not math.isfinite(start):
                raise ValueError("count must be a positive integer")
            factor = self._interval_factor(axis)
            if from_interval:
                interval = float(interval_var.get())
                if not math.isfinite(interval):
                    raise ValueError("interval must be finite")
                end_var.set(start + interval * (count - 1) / factor)
                if count == 1:
                    interval_var.set("0")
            else:
                end = float(end_var.get())
                if not math.isfinite(end):
                    raise ValueError("end must be finite")
                interval_var.set(format((end - start) * factor / (count - 1) if count > 1 else 0.0, ".12g"))
        except (tk.TclError, ValueError):
            # Partial numeric input is normal while typing. Never rewrite it.
            if not from_interval:
                interval_var.set("")
        finally:
            self._range_syncing = False

    def _update_intervals(self) -> None:
        for axis in (1, 2):
            self._range_changed(axis)

    def _style_row(self, parent, label, marker_var, size_var, color_var) -> None:
        row = tk.Frame(parent)
        row.pack(fill="x", pady=1)
        tk.Label(row, text=label, font=("Arial", 8), width=10, anchor="w").pack(side="left")
        mk = ttk.Combobox(row, textvariable=marker_var, values=_MARKERS,
                          state="readonly", width=4)
        mk.pack(side="left", padx=(0, 4))
        mk.bind("<<ComboboxSelected>>", lambda e: self._refresh_preview())
        tk.Spinbox(row, from_=1, to=20, increment=0.5, textvariable=size_var,
                   width=5, font=("Arial", 8),
                   command=self._refresh_preview).pack(side="left", padx=(0, 4))
        cc = ttk.Combobox(row, textvariable=color_var, values=_COMMON_COLORS, width=9)
        cc.pack(side="left")
        cc.bind("<<ComboboxSelected>>", lambda e: self._refresh_preview())
        cc.bind("<FocusOut>", lambda e: self._refresh_preview())

    def _build_param_rows(self, parent: tk.Frame) -> None:
        for child in parent.winfo_children():
            child.destroy()

        is_plane = self._surface_type_var.get() == "plane"

        def _idx_row(text, var, hint):
            row = tk.Frame(parent)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=text, font=("Arial", 8), width=16, anchor="w").pack(side="left")
            sb = tk.Spinbox(row, from_=-1, to=9999, textvariable=var, width=5,
                            font=("Arial", 8))
            sb.pack(side="left")
            tk.Label(row, text=hint, font=("Arial", 7), fg="#888888").pack(side="left", padx=(4, 0))

        if is_plane:
            _idx_row("Origin point:", self._origin_idx_var, "-> a1=0, a2=0")
            _idx_row("X-axis point:", self._xaxis_idx_var, "-> +vx direction")
            _idx_row("+Y side point:", self._yside_idx_var, "-1 = ignore")
        else:
            _idx_row("Origin point:", self._origin_idx_var, "-> a1=0, a2=0")
            tk.Label(parent, text="Approximate +height vector (local Z):", font=("Arial", 8)).pack(anchor="w")
            row = tk.Frame(parent)
            row.pack(fill="x")
            for label, var in zip(("X", "Y", "Z"), self._axis_hint_vars):
                tk.Label(row, text=label, font=("Arial", 8)).pack(side="left")
                tk.Entry(row, textvariable=var, width=7).pack(side="left", padx=(2, 5))
            tk.Label(parent, text="Nonzero vector; local fitting refines this direction.", font=("Arial", 7)).pack(anchor="w")
            tk.Checkbutton(parent, text="Reverse theta direction",
                           variable=self._reverse_theta_var, font=("Arial", 8),
                           command=self._on_apply).pack(anchor="w")

    def _on_type_changed(self, redraw: bool = True) -> None:
        is_plane = self._surface_type_var.get() == "plane"
        surface_type = self._surface_type_var.get()
        if surface_type != self._range_surface_type:
            self._a1_start_var.set(0.0)
            self._a1_end_var.set(1.0 if is_plane else 90.0)
            self._range_surface_type = surface_type
            self._radius = 0.0
        self._update_intervals()
        if hasattr(self, "_param_frame") and self._param_frame.winfo_exists():
            self._build_param_rows(self._param_frame)
        if hasattr(self, "_a1_label") and self._a1_label.winfo_exists():
            self._a1_label.configure(
                text="a1:  (length units)" if is_plane else "a1:  theta, DEGREES")
            self._a2_label.configure(
                text="a2:  (length units)" if is_plane else "a2:  height, length units")
        if redraw:
            self._on_apply()

    def _make_paned(self, parent, left_width=300, left_minsize=220, right_minsize=320):
        panes = tk.PanedWindow(parent, orient=tk.HORIZONTAL, sashwidth=6,
                               sashrelief=tk.RAISED, showhandle=True)
        panes.pack(fill="both", expand=True)
        left = tk.Frame(panes, bg="#f7f7f7")
        right = tk.Frame(panes, bg="#ffffff")
        panes.add(left, width=left_width, minsize=left_minsize)
        panes.add(right, minsize=right_minsize)

        def _apply(total):
            if total < 10:
                return
            try:
                sx, _ = panes.sash_coord(0)
                sw = int(panes.cget("sashwidth"))
            except Exception:
                return
            try:
                panes.paneconfigure(right, width=max(100, total - sx - sw - 2))
            except Exception:
                pass

        panes.bind("<Configure>", lambda e: _apply(e.width) if e.widget is panes else None)
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
        wid = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(wid, width=e.width))
        return inner

    # ── inspector helpers ─────────────────────────────────────────

    def _sync_points_text(self) -> None:
        w = self._points_widget
        if w is not None and w.winfo_exists() and not self._points_from_pin:
            self._points_text = w.get("1.0", tk.END)

    def _on_apply(self) -> None:
        self._sync_points_text()
        self._mesh_cache = None
        self._mesh_cache_key = None
        self._auto_centred = False
        if self._request_downstream:
            self._request_downstream(self.node_id)
        else:
            self._rebuild(None)
            self._refresh_preview()

    def _reset_view(self) -> None:
        self._cam.update(yaw=-0.9, pitch=0.7, zoom=1.0)
        self._auto_centred = False
        self._refresh_preview()

    def _mesh_counts(self) -> tuple[int, int]:
        try:
            m = max(1, int(self._a1_count_var.get()))
            n = max(1, int(self._a2_count_var.get()))
        except (tk.TclError, ValueError):
            m, n = 2, 2
        return m, n

    def _estimated_bytes(self) -> int:
        m, n = self._mesh_counts()
        itemsize = 4 if self._dtype_var.get() == "float32" else 8
        return m * n * 3 * itemsize

    def _update_memory_estimate(self) -> None:
        nbytes = self._estimated_bytes()
        m, n = self._mesh_counts()
        gb = nbytes / (1024 ** 3)
        text = f"mesh {m} x {n} x 3  ->  {gb:.3f} GB ({self._dtype_var.get()})"
        if nbytes > _MEMORY_CAP_BYTES and not self._mem_override_var.get():
            text += "\nEXCEEDS CAP -- tick the override or reduce M/N."
        self._mem_var.set(text)

    def _auto_fit_ranges(self) -> None:
        """Set a1/a2 ranges to span the given points, with a margin.
        Especially useful for a cylinder, where the angular range for a
        given arc is not obvious by inspection."""
        if self._xo is None or self._points.shape[0] == 0:
            self._status_var.set("fit the surface first (press Apply)")
            return
        a1, a2 = self._point_params(self._points)
        if self._surface_type_var.get() == "cylinder":
            a1 = np.rad2deg(a1)
        for arr, s_var, e_var in ((a1, self._a1_start_var, self._a1_end_var),
                                  (a2, self._a2_start_var, self._a2_end_var)):
            lo, hi = float(np.min(arr)), float(np.max(arr))
            span = hi - lo
            margin = span * 0.05 if span > 1e-12 else 1.0
            s_var.set(round(lo - margin, 6))
            e_var.set(round(hi + margin, 6))
        self._update_memory_estimate()
        self._on_apply()

    def _open_help(self) -> None:
        if self._help_popup is not None and self._help_popup.winfo_exists():
            self._help_popup.lift()
            return
        popup = tk.Toplevel()
        popup.title("Surface Mesh Generator - Help")
        popup.geometry("780x660")
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

    # ── fitting + parameterization ────────────────────────────────

    def _resolve_points(self, inputs: dict | None) -> np.ndarray:
        raw = inputs.get("points") if inputs else None
        if raw is not None and isinstance(raw, np.ndarray) and raw.size:
            self._points_from_pin = True
            return _coerce_points(raw)
        self._points_from_pin = False
        self._sync_points_text()
        parsed, _skipped = _parse_points_block(self._points_text)
        return parsed if parsed is not None else np.empty((0, 3), dtype=np.float64)

    def _safe_idx(self, var: tk.IntVar, n: int, default: int) -> int:
        try:
            i = int(var.get())
        except (tk.TclError, ValueError):
            return default
        return i if 0 <= i < n else default

    def _rebuild(self, inputs: dict | None) -> bool:
        """Resolve points, fit the surface, and build the parameterized
        triad. Returns True on success."""
        try:
            pts = self._resolve_points(inputs)
        except Exception as e:
            self._status_var.set(f"points error: {e}")
            self._points = np.empty((0, 3), dtype=np.float64)
            self._xo = None
            return False

        self._points = pts
        n = pts.shape[0]
        is_plane = self._surface_type_var.get() == "plane"
        need = 3 if is_plane else 5

        if n < need:
            kind = "plane" if is_plane else "cylinder"
            self._status_var.set(f"need >= {need} points for a {kind}, have {n}")
            self._fit_var.set("")
            self._err_var.set("")
            self._xo = None
            return False

        try:
            if is_plane:
                self._fit_plane_param(pts)
            else:
                self._fit_cylinder_param(pts)
        except Exception as e:
            self._status_var.set(f"fit error: {e}")
            self._xo = None
            return False

        rms = float(np.sqrt(np.mean(self._errors ** 2))) if self._errors.size else 0.0
        self._err_var.set(
            f"RMS {rms:.6g}   max {float(np.max(self._errors)):.6g}"
            if self._errors.size else "")
        warn = ""
        if not is_plane and n < 8:
            warn = "   (few points: fit may be ill-conditioned)"
        self._status_var.set(f"ok  {n} points{warn}")
        self.set_status("ok", "#3a8a4a")
        self._update_intervals()
        return True

    def _fit_plane_param(self, pts: np.ndarray) -> None:
        centroid, normal, errors = _fit_plane(pts)
        n = pts.shape[0]

        o_i = self._safe_idx(self._origin_idx_var, n, 0)
        x_i = self._safe_idx(self._xaxis_idx_var, n, 1 if n > 1 else 0)
        if x_i == o_i:
            x_i = (o_i + 1) % n

        def _project(p):
            return p - np.dot(p - centroid, normal) * normal

        xo = _project(pts[o_i])
        px = _project(pts[x_i])
        d = px - xo
        if np.linalg.norm(d) < 1e-12:
            raise ValueError("origin and X-axis points project to the same location")
        vx = _unit(d)

        # Optional +Y reference flips the normal so that point lands on
        # the +a2 side; without it the SVD's sign is arbitrary.
        y_i = -1
        try:
            y_i = int(self._yside_idx_var.get())
        except (tk.TclError, ValueError):
            pass
        vy = _unit(np.cross(normal, vx))
        if 0 <= y_i < n:
            if np.dot(pts[y_i] - xo, vy) < 0:
                normal = -normal
                vy = _unit(np.cross(normal, vx))

        self._xo, self._vx, self._vy, self._vz = xo, vx, vy, _unit(np.cross(vx, vy))
        self._radius = 0.0
        self._errors = errors
        self._fit_var.set(
            f"plane  n=({normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f})")

    def _fit_cylinder_param(self, pts: np.ndarray) -> None:
        n = pts.shape[0]

        o_i = self._safe_idx(self._origin_idx_var, n, 0)
        if self._legacy_height_idx is not None:
            h_i = self._legacy_height_idx
            if 0 <= h_i < n and np.linalg.norm(pts[h_i] - pts[o_i]) > 1e-12:
                for var, value in zip(self._axis_hint_vars, pts[h_i] - pts[o_i]):
                    var.set(str(value))
            self._legacy_height_idx = None
        hint = np.array([float(var.get()) for var in self._axis_hint_vars])
        axis_pt, axis_dir, radius, errors = _fit_cylinder(pts, axis_hint=hint)

        vz = axis_dir

        # Origin sits on the axis at the origin point's height, so that
        # the origin point itself maps to a1=0, a2=0.
        rel = pts[o_i] - axis_pt
        xo = axis_pt + np.dot(rel, vz) * vz

        radial = rel - np.dot(rel, vz) * vz
        if np.linalg.norm(radial) < 1e-12:
            raise ValueError("origin point lies on the cylinder axis; pick another")
        vx = _unit(radial)
        vy = _unit(np.cross(vz, vx))
        if self._reverse_theta_var.get():
            vy = -vy

        self._xo, self._vx, self._vy, self._vz = xo, vx, vy, vz
        self._radius = float(radius)
        self._errors = errors
        self._fit_var.set(
            f"cylinder  R={radius:.6g}  axis=({vz[0]:.4f}, {vz[1]:.4f}, {vz[2]:.4f})")

    def _point_params(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Inverse map: world points -> (a1, a2). Used by Auto-fit."""
        rel = pts - self._xo
        if self._surface_type_var.get() == "plane":
            return rel @ self._vx, rel @ self._vy
        a2 = rel @ self._vz
        a1 = np.arctan2(rel @ self._vy, rel @ self._vx)
        return a1, a2

    # ── mesh generation ───────────────────────────────────────────

    def _surface_points(self, a1: np.ndarray, a2: np.ndarray, dtype) -> np.ndarray:
        """
        Evaluate the surface on the outer product of a1 and a2.

        Built one XYZ component at a time: the temporaries are then (M,N)
        rather than (M,N,3), cutting peak memory to about a third of the
        naive broadcast, which matters a lot at 8000x8000.
        """
        m, n = a1.size, a2.size
        out = np.empty((m, n, 3), dtype=dtype)
        xo, vx, vy, vz = self._xo, self._vx, self._vy, self._vz

        if self._surface_type_var.get() == "plane":
            c1 = a1.astype(dtype, copy=False)[:, None]
            c2 = a2.astype(dtype, copy=False)[None, :]
            for k in range(3):
                out[:, :, k] = xo[k] + c1 * vx[k] + c2 * vy[k]
        else:
            r = self._radius
            cos1 = (r * np.cos(a1)).astype(dtype, copy=False)[:, None]
            sin1 = (r * np.sin(a1)).astype(dtype, copy=False)[:, None]
            c2 = a2.astype(dtype, copy=False)[None, :]
            for k in range(3):
                out[:, :, k] = xo[k] + cos1 * vx[k] + sin1 * vy[k] + c2 * vz[k]
        return out

    def _linspaces(self) -> tuple[np.ndarray, np.ndarray]:
        m, n = self._mesh_counts()
        a1 = np.linspace(float(self._a1_start_var.get()),
                         float(self._a1_end_var.get()), m, dtype=np.float64)
        a2 = np.linspace(float(self._a2_start_var.get()),
                         float(self._a2_end_var.get()), n, dtype=np.float64)
        if self._surface_type_var.get() == "cylinder":
            a1 = np.deg2rad(a1)
        return a1, a2

    def _cache_key(self) -> tuple:
        m, n = self._mesh_counts()
        return (
            self._surface_type_var.get(), m, n,
            float(self._a1_start_var.get()), float(self._a1_end_var.get()),
            float(self._a2_start_var.get()), float(self._a2_end_var.get()),
            self._dtype_var.get(), self._radius,
            tuple(np.round(self._xo, 12)), tuple(np.round(self._vx, 12)),
            tuple(np.round(self._vy, 12)), tuple(np.round(self._vz, 12)),
        )

    def _generate_dense(self):
        """Returns (mesh, a1, a2) using the cache when nothing changed."""
        key = self._cache_key()
        if self._mesh_cache is not None and self._mesh_cache_key == key:
            return self._mesh_cache

        nbytes = self._estimated_bytes()
        if nbytes > _MEMORY_CAP_BYTES and not self._mem_override_var.get():
            raise MemoryError(
                f"mesh would need {nbytes / (1024**3):.2f} GB; reduce M/N, "
                f"use float32, or tick the override")

        a1, a2 = self._linspaces()
        dtype = np.float32 if self._dtype_var.get() == "float32" else np.float64
        mesh = self._surface_points(a1, a2, dtype)
        self._mesh_cache = (mesh, a1, a2)
        self._mesh_cache_key = key
        return self._mesh_cache

    def _build_preview_grid(self) -> None:
        if self._xo is None:
            self._preview_grid = np.empty((0, 0, 3), dtype=np.float64)
            return
        try:
            md = max(2, int(self._mdraw_var.get()))
            nd = max(2, int(self._ndraw_var.get()))
            a1 = np.linspace(float(self._a1_start_var.get()),
                             float(self._a1_end_var.get()), md)
            a2 = np.linspace(float(self._a2_start_var.get()),
                             float(self._a2_end_var.get()), nd)
            if self._surface_type_var.get() == "cylinder":
                a1 = np.deg2rad(a1)
            self._preview_grid = self._surface_points(a1, a2, np.float64)
        except Exception:
            self._preview_grid = np.empty((0, 0, 3), dtype=np.float64)

    # ── 3D preview ────────────────────────────────────────────────

    def _scene_points(self) -> np.ndarray:
        parts = []
        if self._points.size:
            parts.append(self._points)
        if self._preview_grid.size:
            parts.append(self._preview_grid.reshape(-1, 3))
        if not parts:
            return np.empty((0, 3), dtype=np.float64)
        return np.concatenate(parts, axis=0)

    def _radius_centre(self) -> tuple[float, np.ndarray]:
        pts = self._scene_points()
        if pts.shape[0] == 0:
            return 1.0, np.zeros(3)
        finite = pts[np.isfinite(pts).all(axis=1)]
        if finite.shape[0] == 0:
            return 1.0, np.zeros(3)
        centre = finite.mean(axis=0)
        return max(float(np.max(np.linalg.norm(finite - centre, axis=1))), 1e-6), centre

    def _rot(self) -> np.ndarray:
        yaw, pitch = self._cam["yaw"], self._cam["pitch"]
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        return (np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
                @ np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]))

    def _scale(self, radius: float) -> float:
        self._cam_last_radius = max(radius, 1e-6)
        cm = 200.0
        c = self._preview_canvas
        if c is not None and c.winfo_exists():
            cm = min(float(c.winfo_width()), float(c.winfo_height()))
        return cm * 0.36 * self._cam["zoom"] / max(radius, 1e-6)

    def _project(self, pts: np.ndarray, scale: float) -> np.ndarray:
        c = self._preview_canvas
        if c is None or not c.winfo_exists():
            return np.zeros((len(pts), 2))
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        rot = (self._rot() @ (pts - np.asarray(self._cam["target"])).T).T
        # Match Structural Mesh Viewer's camera-space +Y depth convention
        # and focal-length mapping. Cache the scene radius once in _scale(),
        # not per grid line/point batch.
        intensity = min(max(float(self._perspective_var.get()), 0.0), 1.0)
        if intensity > 1e-6:
            focal = self._cam_last_radius * (4.0 / intensity - 3.0)
            # Limit magnification near/behind the camera plane.
            factor = focal / np.maximum(focal + rot[:, 1], focal * 0.1)
        else:
            factor = np.ones(len(pts))
        cw, ch = float(c.winfo_width()), float(c.winfo_height())
        return np.column_stack([rot[:, 0] * factor * scale + cw / 2.0,
                                -rot[:, 2] * factor * scale + ch / 2.0])

    def _on_drag_start(self, e):
        self._cam.update(dragging=True, last_x=e.x, last_y=e.y, button=1)

    def _on_drag_motion(self, e):
        if not self._cam["dragging"] or self._cam["button"] != 1:
            return
        self._cam["yaw"] += (e.x - self._cam["last_x"]) * 0.01
        self._cam["pitch"] += (e.y - self._cam["last_y"]) * 0.01
        self._cam["last_x"], self._cam["last_y"] = e.x, e.y
        self._refresh_preview()

    def _on_drag_release(self, _e):
        self._cam.update(dragging=False, button=None)

    def _on_pan_start(self, e):
        self._cam.update(dragging=True, last_x=e.x, last_y=e.y, button=3)

    def _on_pan_motion(self, e):
        if not self._cam["dragging"] or self._cam["button"] != 3:
            return
        dx, dy = e.x - self._cam["last_x"], e.y - self._cam["last_y"]
        radius, _ = self._radius_centre()
        scale = self._scale(radius)
        if scale > 1e-12:
            rt = self._rot().T
            delta = (rt[:, 0] * dx - rt[:, 2] * dy) / scale
            self._cam["target"] = (np.asarray(self._cam["target"]) - delta).tolist()
        self._cam["last_x"], self._cam["last_y"] = e.x, e.y
        self._refresh_preview()

    def _on_pan_release(self, _e):
        self._cam.update(dragging=False, button=None)

    def _on_wheel(self, e) -> str:
        d = int(e.delta / 120)
        if d:
            self._cam["zoom"] = max(0.05, self._cam["zoom"] * (1.12 ** d))
            self._refresh_preview()
        return "break"

    @staticmethod
    def _draw_marker(c, x, y, marker, size, color, filled=True):
        fill = color if filled else ""
        if marker == "o":
            c.create_oval(x - size, y - size, x + size, y + size, fill=fill, outline=color)
        elif marker == "s":
            c.create_rectangle(x - size, y - size, x + size, y + size, fill=fill, outline=color)
        elif marker == "D":
            c.create_polygon(x, y - size, x + size, y, x, y + size, x - size, y,
                             fill=fill, outline=color)
        elif marker == "^":
            c.create_polygon(x, y - size, x + size, y + size, x - size, y + size,
                             fill=fill, outline=color)
        elif marker == "v":
            c.create_polygon(x, y + size, x + size, y - size, x - size, y - size,
                             fill=fill, outline=color)
        elif marker in ("+", "*", "x"):
            if marker in ("+", "*"):
                c.create_line(x - size, y, x + size, y, fill=color)
                c.create_line(x, y - size, x, y + size, fill=color)
            if marker in ("x", "*"):
                c.create_line(x - size, y - size, x + size, y + size, fill=color)
                c.create_line(x - size, y + size, x + size, y - size, fill=color)
        else:
            c.create_oval(x - size, y - size, x + size, y + size, fill=fill, outline=color)

    def _draw_axis_triad(self, c: tk.Canvas) -> None:
        if not c.winfo_exists() or c.winfo_width() < 2:
            return
        r = self._rot()
        corner = np.array([32.0, float(c.winfo_height()) - 32.0])
        for axis, color, label in ((np.array([1.0, 0, 0]), "#cc0000", "X"),
                                   (np.array([0, 1.0, 0]), "#00aa00", "Y"),
                                   (np.array([0, 0, 1.0]), "#0000cc", "Z")):
            rot = r @ axis
            end = corner + np.array([rot[0], -rot[2]]) * 22.0
            c.create_line(corner[0], corner[1], end[0], end[1], fill=color, width=2)
            c.create_text(end[0] + (end[0] - corner[0]) * 0.2,
                          end[1] + (end[1] - corner[1]) * 0.2,
                          text=label, fill=color, font=("Arial", 8, "bold"))
        verts = self._CUBE_VERTS * (22.0 * 0.32)
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

        self._build_preview_grid()
        if self._points.size == 0 and self._preview_grid.size == 0:
            c.create_text(c.winfo_width() / 2, c.winfo_height() / 2,
                          text="Enter or connect at least 3 points, then press Apply",
                          fill="#999999", width=max(120, c.winfo_width() - 40))
            return

        radius, centre = self._radius_centre()
        if not self._auto_centred:
            self._cam["target"] = centre.tolist()
            self._auto_centred = True
        scale = self._scale(radius)

        grid = self._preview_grid
        if grid.size:
            md, nd = grid.shape[0], grid.shape[1]
            if self._surf_lines_var.get():
                lc = self._surf_line_color_var.get() or "#88aacc"
                for i in range(md):
                    proj = self._project(grid[i, :, :], scale)
                    c.create_line(*[v for xy in proj for v in xy], fill=lc)
                for j in range(nd):
                    proj = self._project(grid[:, j, :], scale)
                    c.create_line(*[v for xy in proj for v in xy], fill=lc)
            try:
                ssz = float(self._surf_size_var.get())
            except (tk.TclError, ValueError):
                ssz = 2.5
            scol = self._surf_color_var.get() or "blue"
            smk = self._surf_marker_var.get() or "+"
            for (px, py) in self._project(grid.reshape(-1, 3), scale):
                self._draw_marker(c, px, py, smk, ssz, scol, filled=False)

        if self._points.size:
            try:
                gsz = float(self._given_size_var.get())
            except (tk.TclError, ValueError):
                gsz = 5.0
            gcol = self._given_color_var.get() or "red"
            gmk = self._given_marker_var.get() or "o"
            proj = self._project(self._points, scale)
            for k, (px, py) in enumerate(proj):
                self._draw_marker(c, px, py, gmk, gsz, gcol, filled=True)
                c.create_text(px + gsz + 6, py, text=str(k), anchor="w",
                              font=("Arial", 7), fill=gcol)

        self._draw_axis_triad(c)
        if self._preview_grid.size:
            m, n = self._mesh_counts()
            c.create_text(8, 10, anchor="nw", font=("Arial", 7), fill="#777777",
                          text=f"preview {self._preview_grid.shape[0]}x"
                               f"{self._preview_grid.shape[1]}   (dense mesh {m}x{n})")

    # ── compute ───────────────────────────────────────────────────

    def compute(self, inputs: dict) -> dict:
        self._init_state()

        if not self._rebuild(inputs):
            self._refresh_preview()
            return {}

        self._update_memory_estimate()

        try:
            mesh, a1, a2 = self._generate_dense()
        except MemoryError as e:
            self._status_var.set(str(e))
            self.set_status("error", "#cc0000")
            self._refresh_preview()
            return {
                "proj_errors": self._errors,
                "rms_error": float(np.sqrt(np.mean(self._errors ** 2)))
                if self._errors.size else 0.0,
                "surface_origin": self._xo.copy(),
                "surface_axes": np.column_stack([self._vx, self._vy, self._vz]),
                "radius": float(self._radius),
            }
        except Exception as e:
            self._status_var.set(f"mesh error: {e}")
            self.set_status("error", "#cc0000")
            self._refresh_preview()
            return {}

        self._refresh_preview()
        rms = float(np.sqrt(np.mean(self._errors ** 2))) if self._errors.size else 0.0
        return {
            "mesh_points": mesh,
            "a1_array": a1,
            "a2_array": a2,
            "proj_errors": self._errors,
            "rms_error": rms,
            "surface_origin": self._xo.copy(),
            "surface_axes": np.column_stack([self._vx, self._vy, self._vz]),
            "radius": float(self._radius),
        }

    # ── serialization ─────────────────────────────────────────────

    def get_params(self) -> dict:
        self._sync_points_text()
        return {
            "surface_type": self._surface_type_var.get(),
            "points_text": self._points_text,
            "origin_idx": self._origin_idx_var.get(),
            "xaxis_idx": self._xaxis_idx_var.get(),
            "yside_idx": self._yside_idx_var.get(),
            "axis_hint": [var.get() for var in self._axis_hint_vars],
            "legacy_height_idx": self._legacy_height_idx,
            "theta_unit": "degrees",
            "reverse_theta": bool(self._reverse_theta_var.get()),
            "a1_start": self._a1_start_var.get(), "a1_end": self._a1_end_var.get(),
            "a1_count": self._a1_count_var.get(),
            "a2_start": self._a2_start_var.get(), "a2_end": self._a2_end_var.get(),
            "a2_count": self._a2_count_var.get(),
            "dtype": self._dtype_var.get(),
            "mem_override": bool(self._mem_override_var.get()),
            "mdraw": self._mdraw_var.get(), "ndraw": self._ndraw_var.get(),
            "perspective_intensity": self._perspective_var.get(),
            "given_marker": self._given_marker_var.get(),
            "given_size": self._given_size_var.get(),
            "given_color": self._given_color_var.get(),
            "surf_marker": self._surf_marker_var.get(),
            "surf_size": self._surf_size_var.get(),
            "surf_color": self._surf_color_var.get(),
            "surf_lines": bool(self._surf_lines_var.get()),
            "surf_line_color": self._surf_line_color_var.get(),
        }

    def set_params(self, params: dict) -> None:
        self._init_state()
        st = str(params.get("surface_type", "plane"))
        self._surface_type_var.set(st if st in ("plane", "cylinder") else "plane")
        self._points_text = str(params.get("points_text", ""))
        self._origin_idx_var.set(int(params.get("origin_idx", 0)))
        self._xaxis_idx_var.set(int(params.get("xaxis_idx", 1)))
        self._yside_idx_var.set(int(params.get("yside_idx", -1)))
        hint = params.get("axis_hint", (0, 0, 1))
        for var, value in zip(self._axis_hint_vars, hint):
            var.set(str(value))
        self._legacy_height_idx = (int(params.get("height_idx", 1))
                       if st == "cylinder" and "axis_hint" not in params
                       else params.get("legacy_height_idx"))
        self._range_surface_type = self._surface_type_var.get()
        self._reverse_theta_var.set(bool(params.get("reverse_theta", False)))
        angle_factor = (180.0 / math.pi if st == "cylinder" and params.get("theta_unit") != "degrees" else 1.0)
        self._a1_start_var.set(float(params.get("a1_start", 0.0)) * angle_factor)
        self._a1_end_var.set(float(params["a1_end"]) * angle_factor if "a1_end" in params else (90.0 if st == "cylinder" else 1.0))
        self._a1_count_var.set(max(1, int(params.get("a1_count", 2000))))
        self._a2_start_var.set(float(params.get("a2_start", 0.0)))
        self._a2_end_var.set(float(params.get("a2_end", 1.0)))
        self._a2_count_var.set(max(1, int(params.get("a2_count", 2000))))
        self._dtype_var.set(str(params.get("dtype", "float64")))
        self._mem_override_var.set(bool(params.get("mem_override", False)))
        self._mdraw_var.set(int(params.get("mdraw", 20)))
        self._ndraw_var.set(int(params.get("ndraw", 20)))
        self._perspective_var.set(min(max(float(params.get("perspective_intensity", 0.35)), 0.0), 1.0))
        self._given_marker_var.set(str(params.get("given_marker", "o")))
        self._given_size_var.set(float(params.get("given_size", 5.0)))
        self._given_color_var.set(str(params.get("given_color", "red")))
        self._surf_marker_var.set(str(params.get("surf_marker", "+")))
        self._surf_size_var.set(float(params.get("surf_size", 2.5)))
        self._surf_color_var.set(str(params.get("surf_color", "blue")))
        self._surf_lines_var.set(bool(params.get("surf_lines", True)))
        self._surf_line_color_var.set(str(params.get("surf_line_color", "#88aacc")))
        self._mesh_cache = None
        self._mesh_cache_key = None

    def close_inspector(self) -> None:
        self._sync_points_text()
        super().close_inspector()
        self._points_widget = None
        self._preview_canvas = None

    def on_destroy(self) -> None:
        if self._help_popup is not None and self._help_popup.winfo_exists():
            self._help_popup.destroy()
        self._mesh_cache = None
        super().on_destroy()