# node_editor/nodes/structural_mesh_viewer_node.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
import cv2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from node_editor.base_node import BaseNode
from node_editor.pin_types import PinSchema, PinDef, PinType
from node_editor.execution import ExecutionMode


# ---------------------------------------------------------------------------
# Help text
# ---------------------------------------------------------------------------

_HELP_TEXT = """\
Structural Mesh Viewer
=======================

PURPOSE
-------
Analyzes the dynamic shape of a structured surface mesh (mp x nq points,
tracked over nt time steps) of a structure (beam, column, wall, etc.).
Decomposes each frame's motion into:
  (a) 6-DOF rigid body motion (translation + rotation, exact for large
      rotation, via the Kabsch algorithm anchored to the t=0 reference)
  (b) local (rigid-body-removed) deformation, expressed in a local
      x'-y'-z' frame fixed once at t=0 (x' = wide in-plane direction,
      y' = other in-plane direction, z' = out-of-plane normal)
and estimates Q4 membrane strain fields (eps_xx, eps_yy, gamma_xy, von
Mises equivalent strain) from the local in-plane displacement field,
using the t=0 configuration as the zero-strain reference.

INPUT PINS
----------
points        ARRAY (required)
    Shape (nt, mp, nq, 3): nt time steps, mp x nq structured mesh
    points, [x,y,z] per point (GLOBAL/world coordinates). NaN points
    (lost tracking) are supported and propagate to NaN downstream.

x_hint        ARRAY (optional, shape (3,))
    Approximate x' direction, breaks the PCA sign ambiguity only.

normal_hint   ARRAY (optional, shape (3,))
    Approximate outward surface normal, same role for z'.

img_files     list[str] (optional)
    A list of image file paths (photos of the object), one per time
    step -- img_files[0] is used for frame t=0, img_files[1] for t=1,
    and so on. Used as the background for the "Field Pcolor on Image"
    tab. Each file is read from disk on demand (RGB, uint8, 2D
    grayscale or 3D).

cmat          ARRAY (optional, 3x3)
    Camera intrinsic matrix, used only by "Field Pcolor on Image".

dvec          ARRAY (optional)
    Camera distortion coefficients, used only by "Field Pcolor on Image".

rvec, tvec    ARRAY (optional, shape (3,) or (3,1))
    Camera extrinsic rotation (Rodrigues) and translation, describing
    the camera's pose in the SAME world/global coordinate system as
    the `points` pin. Used only by "Field Pcolor on Image".

OUTPUT PINS
-----------
local_coords     ARRAY (nt, mp, nq, 3)  -- local x',y',z' per point/frame
rbm_translation  ARRAY (nt, 3)          -- centroid position per frame;
                                            this IS the rigid-body-motion
                                            translation time series.
rbm_rotation     ARRAY (nt, 3)          -- Rodrigues rotation vector per
                                            frame (t=0 -> t); this IS the
                                            rigid-body-motion rotation
                                            time series.
strain_fields    ARRAY (nt, mp-1, nq-1, 4, 4)
                                        -- per cell, per Gauss point (2x2
                                           rule), [eps_xx, eps_yy,
                                           gamma_xy, von_mises]; this IS
                                           the strain time series at
                                           Gauss points (index the first
                                           axis for a given frame's slice).
ready            SCALAR                -- 1.0 once decomposition has
                                           been computed successfully.

METHOD NOTES (read before trusting numbers)
--------------------------------------------
- t=0 local frame: PCA about the t=0 centroid (uniform point weighting).
  Smallest eigenvalue axis -> z'. Larger of the remaining two -> x'.
  Sign ambiguity resolved by x_hint/normal_hint if given.
- Per-frame rigid rotation: Kabsch algorithm (SVD orthogonal Procrustes),
  fit between t=0 and frame t using only points valid at BOTH times,
  anchored to the FIXED t=0 reference -- exact for large rotations, no
  frame-to-frame drift.
- Frames with fewer than 3 jointly-valid points are left as NaN.
- Strain: Q4 isoparametric membrane B-matrix computed ONCE from the t=0
  local geometry (co-rotational assumption). Von Mises equivalent strain
  uses the Poisson-independent DIC/GOM convention:
      eps_vm = (2/sqrt(3)) * sqrt(e1^2 + e1*e2 + e2^2)
  No stress is computed (no material properties assumed).
- Recompute runs synchronously; large nt*mp*nq may briefly freeze the UI.

INSPECTOR TABS
--------------
Data & Decomposition:  hints, Recompute button, status/diagnostics, and
                        a TIME-SERIES PLOT below Status: up to 6 series
                        chosen from 6-DOF rigid body motion (translation
                        X/Y/Z, rotation X/Y/Z), or Global/Local X/Y/Z at
                        a chosen mesh point (i, j) -- each independently
                        set to Coordinate mode (raw value) or
                        Displacement mode (value(t) - value(0)). Each
                        series has its own color, marker, marker size,
                        and line width; Y-axis is auto or manually
                        min/max; a dotted vertical playhead line tracks
                        the current time-control frame. Pan/zoom as in
                        Section Plot / Field Pcolor.
3D Animation:           orbit-viewable animated shape. Global/Local view,
                        per-axis amplification (Local view only), an
                        optional colored surface (element-averaged field,
                        colormap, color limits) with approximate hidden-
                        surface removal, a lower-left GLOBAL axes icon,
                        and mesh index edge labels.
Section Plot:           a Lagrangian (material-fixed) cut along one mesh
                        index direction, animated 2D curve, amplification.
Field Pcolor:           animated colormap of a displacement or strain
                        field over the flat local x'-y' mesh, element-
                        averaged or per-Gauss-point.
Field Pcolor on Image:  the same field, colored and warped onto the
                        ACTUAL camera photo, using cv2.projectPoints
                        with cmat/dvec/rvec/tvec to project the GLOBAL
                        3D mesh points (at the current frame) into image
                        pixel coordinates, so the colored mesh appears
                        to sit on the real object in the photo. Includes
                        a Transparency slider for the overlay. Per-Gauss
                        resolution here uses true bilinearly-interpolated
                        3D positions at each cell's edge-midpoints/center
                        (not an abstract index grid), so the projected
                        sub-quads are geometrically correct.

The vertical divider between each tab's left settings panel and its
right-hand view can be dragged to resize both panels, and the right
panel automatically fills any extra space when you resize the window.

MOUSE CONTROLS (Time Series / Section Plot / Field Pcolor / Field on Image)
-----------------------------------------------------------------------------
Left-drag    pan
Mouse wheel  zoom, centered on the cursor
Double-click reset to the full data extent

MOUSE CONTROLS (3D Animation)
-------------------------------
Left-drag    rotate (orbits around the current pan target)
Right-drag   pan
Mouse wheel  zoom

TIME CONTROL BAR
-----------------
|<  Prev  Play  Play & Save...  Stop  Next  >|   plus a scrubber.

"Play & Save..." first opens a Video Recording Settings dialog (Quality
0-100, Number of parallel encoding stripes -1 = automatic, and a Start
frame / End frame range -- defaults to the full sequence, 0 through
nt-1; sensible defaults are pre-filled so you can just click OK), then
a save-file dialog, then records frames Start through End inclusive
(e.g. Start=100, End=200 saves 101 frames) by screenshotting the entire
inspector window at every time step and encoding at exactly the FPS
box's value, regardless of how long capture/encoding actually takes in
real time. Requires Pillow's ImageGrab.

KEYBOARD SHORTCUT
-----------------
Ctrl-H / Ctrl-h -- show this help window.
"""

_CMAP_LIST = [
    "jet", "jet_r", "gray", "gray_r",
    "viridis", "viridis_r", "plasma", "plasma_r",
    "inferno", "inferno_r", "magma", "magma_r",
    "coolwarm", "coolwarm_r", "bwr", "bwr_r", "seismic", "seismic_r",
    "turbo", "turbo_r", "rainbow", "rainbow_r",
    "hot", "hot_r", "cool", "cool_r",
]

_FIELD_OPTIONS = [
    ("ux", "Displacement U_x' (nodal)"),
    ("uy", "Displacement U_y' (nodal)"),
    ("uz", "Displacement U_z' (nodal)"),
    ("exx", "Strain eps_xx"),
    ("eyy", "Strain eps_yy"),
    ("gxy", "Strain gamma_xy"),
    ("vm", "Von Mises equivalent strain"),
]
_DISPLACEMENT_FIELD_KEYS = {"ux", "uy", "uz"}

_SECTION_VALUE_OPTIONS = [
    ("x", "Local X'"),
    ("y", "Local Y'"),
    ("z", "Local Z'"),
]

_COMMON_COLORS = [
    "red", "blue", "green", "orange", "purple", "brown",
    "black", "gray", "magenta", "cyan",
]
_MARKERS = ["o", "+", "*", "x", "s", "D", "^", "v", "<", ">"]

_TS_SOURCE_OPTIONS = [
    "RBM Translation X", "RBM Translation Y", "RBM Translation Z",
    "RBM Rotation X", "RBM Rotation Y", "RBM Rotation Z",
    "Global X", "Global Y", "Global Z",
    "Local X'", "Local Y'", "Local Z'",
]
_TS_POINT_SOURCES = {
    "Global X", "Global Y", "Global Z",
    "Local X'", "Local Y'", "Local Z'",
}
_TS_DEFAULT_SOURCES = [
    "RBM Translation Z", "RBM Rotation X", "RBM Rotation Y",
    "Local Z'", "Global X", "Global Y",
]
_TS_DEFAULT_COLORS = ["red", "blue", "green", "orange", "purple", "brown"]
_TS_DEFAULT_MARKERS = ["o", "s", "^", "D", "v", "*"]


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_vec3(text: str) -> np.ndarray | None:
    text = (text or "").strip()
    if not text:
        return None
    parts = text.replace(",", " ").split()
    if len(parts) != 3:
        return None
    try:
        return np.array([float(p) for p in parts], dtype=np.float64)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Rigid body decomposition (Kabsch, anchored to fixed t=0 reference)
# ---------------------------------------------------------------------------

def _kabsch_rotation(a_centered: np.ndarray, b_centered: np.ndarray) -> np.ndarray:
    """
    Find the proper rotation R minimizing sum ||b_i - R a_i||^2.
    a_centered, b_centered: (n,3) row-vector point sets, already
    centered on their own centroids.
    """
    m = a_centered.T @ b_centered      # (3,3)
    u, _s, vt = np.linalg.svd(m)
    v = vt.T
    d = np.sign(np.linalg.det(v @ u.T))
    if d == 0:
        d = 1.0
    diag = np.diag([1.0, 1.0, d])
    return v @ diag @ u.T


def _pca_local_frame(
    pts0: np.ndarray, valid0: np.ndarray,
    x_hint: np.ndarray | None, normal_hint: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (centroid0, R0) where R0's columns are [x', y', z'] expressed
    in global coordinates (row-vector convention: local = global @ R0).
    Uniform (unweighted) point-cloud PCA about the centroid.
    """
    pts = pts0[valid0]
    centroid0 = pts.mean(axis=0)
    centered = pts - centroid0
    cov = (centered.T @ centered) / max(1, centered.shape[0])

    eigvals, eigvecs = np.linalg.eigh(cov)   # ascending eigenvalues
    z_axis = eigvecs[:, 0].copy()            # smallest -> out-of-plane normal
    y_axis = eigvecs[:, 1].copy()
    x_axis = eigvecs[:, 2].copy()            # largest -> wide in-plane direction

    if normal_hint is not None and np.linalg.norm(normal_hint) > 1e-9:
        if np.dot(z_axis, normal_hint) < 0:
            z_axis = -z_axis
    if x_hint is not None and np.linalg.norm(x_hint) > 1e-9:
        if np.dot(x_axis, x_hint) < 0:
            x_axis = -x_axis

    x_axis = x_axis - np.dot(x_axis, z_axis) * z_axis
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= np.linalg.norm(z_axis)

    r0 = np.column_stack([x_axis, y_axis, z_axis])
    return centroid0, r0


def _decompose_rbm(
    points: np.ndarray, x_hint: np.ndarray | None, normal_hint: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    points: (nt, mp, nq, 3).
    Returns (local_coords (nt,mp,nq,3), translations (nt,3),
             rotations_rvec (nt,3), r0 (3,3), n_bad_frames).
    """
    nt, mp, nq, _ = points.shape
    n_pts = mp * nq

    flat0 = points[0].reshape(n_pts, 3)
    valid0 = np.isfinite(flat0).all(axis=1)
    if valid0.sum() < 3:
        raise ValueError("t=0 frame has fewer than 3 valid (non-NaN) points")

    centroid0, r0 = _pca_local_frame(flat0, valid0, x_hint, normal_hint)

    local_coords = np.full((nt, n_pts, 3), np.nan, dtype=np.float64)
    translations = np.full((nt, 3), np.nan, dtype=np.float64)
    rotations = np.full((nt, 3), np.nan, dtype=np.float64)
    n_bad = 0

    for t in range(nt):
        flat_t = points[t].reshape(n_pts, 3)
        valid_t = np.isfinite(flat_t).all(axis=1)
        joint = valid0 & valid_t
        if joint.sum() < 3:
            n_bad += 1
            continue

        centroid_t = flat_t[joint].mean(axis=0)
        if t == 0:
            r_t = np.eye(3, dtype=np.float64)
        else:
            a = flat0[joint] - centroid0
            b = flat_t[joint] - centroid_t
            r_t = _kabsch_rotation(a, b)

        translations[t] = centroid_t
        rvec, _ = cv2.Rodrigues(r_t)
        rotations[t] = rvec.ravel()

        xt_centered = flat_t - centroid_t          # NaN stays NaN per point
        local_coords[t] = xt_centered @ r_t @ r0

    local_coords = local_coords.reshape(nt, mp, nq, 3)
    return local_coords, translations, rotations, r0, n_bad


# ---------------------------------------------------------------------------
# Q4 membrane strain (reference B-matrix computed once at t=0)
# ---------------------------------------------------------------------------

_G = 1.0 / np.sqrt(3.0)
_GAUSS_PTS = [(-_G, -_G), (_G, -_G), (_G, _G), (-_G, _G)]
_CORNER_OFFSETS = [(0, 0), (1, 0), (1, 1), (0, 1)]


def _precompute_q4_b_matrices(x0: np.ndarray, y0: np.ndarray):
    """
    x0, y0: (mp, nq) reference (t=0) local in-plane nodal coordinates.
    Returns (b_all (ncx,ncy,4,3,8), jac_ok (ncx,ncy) bool).
    """
    mp, nq = x0.shape
    ncx, ncy = mp - 1, nq - 1
    b_all = np.zeros((ncx, ncy, 4, 3, 8), dtype=np.float64)
    jac_ok = np.ones((ncx, ncy), dtype=bool)

    for i in range(ncx):
        for j in range(ncy):
            xs = np.array([x0[i + di, j + dj] for di, dj in _CORNER_OFFSETS])
            ys = np.array([y0[i + di, j + dj] for di, dj in _CORNER_OFFSETS])
            if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
                jac_ok[i, j] = False
                continue
            for gi, (xi, eta) in enumerate(_GAUSS_PTS):
                dn_dxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
                dn_deta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
                jac = np.array([
                    [dn_dxi @ xs, dn_dxi @ ys],
                    [dn_deta @ xs, dn_deta @ ys],
                ])
                det_j = np.linalg.det(jac)
                if abs(det_j) < 1e-12:
                    jac_ok[i, j] = False
                    continue
                jac_inv = np.linalg.inv(jac)
                dn_dxdy = jac_inv @ np.vstack([dn_dxi, dn_deta])   # (2,4)
                dndx, dndy = dn_dxdy[0], dn_dxdy[1]
                b = np.zeros((3, 8), dtype=np.float64)
                for k in range(4):
                    b[0, 2 * k] = dndx[k]
                    b[1, 2 * k + 1] = dndy[k]
                    b[2, 2 * k] = dndy[k]
                    b[2, 2 * k + 1] = dndx[k]
                b_all[i, j, gi] = b
    return b_all, jac_ok


def _compute_all_strains(local_coords: np.ndarray, b_all: np.ndarray, jac_ok: np.ndarray) -> np.ndarray:
    """
    local_coords: (nt, mp, nq, 3). Returns (nt, ncx, ncy, 4, 4)
    [eps_xx, eps_yy, gamma_xy, von_mises] per cell per Gauss point.
    """
    nt, mp, nq, _ = local_coords.shape
    ncx, ncy = mp - 1, nq - 1
    x0 = local_coords[0, :, :, 0]
    y0 = local_coords[0, :, :, 1]

    out = np.full((nt, ncx, ncy, 4, 4), np.nan, dtype=np.float64)
    for t in range(nt):
        xt = local_coords[t, :, :, 0]
        yt = local_coords[t, :, :, 1]
        ux = xt - x0
        uy = yt - y0

        u_vec = np.zeros((ncx, ncy, 8), dtype=np.float64)
        for k, (di, dj) in enumerate(_CORNER_OFFSETS):
            u_vec[:, :, 2 * k] = ux[di:di + ncx, dj:dj + ncy]
            u_vec[:, :, 2 * k + 1] = uy[di:di + ncx, dj:dj + ncy]

        strain = np.einsum("ijgab,ijb->ijga", b_all, u_vec)   # (ncx,ncy,4,3)
        exx, eyy, gxy = strain[..., 0], strain[..., 1], strain[..., 2]
        avg = (exx + eyy) / 2.0
        diff = (exx - eyy) / 2.0
        radius = np.sqrt(diff ** 2 + (gxy / 2.0) ** 2)
        e1 = avg + radius
        e2 = avg - radius
        von_mises = (2.0 / np.sqrt(3.0)) * np.sqrt(e1 ** 2 + e1 * e2 + e2 ** 2)

        frame_out = np.stack([exx, eyy, gxy, von_mises], axis=-1)
        frame_out[~jac_ok] = np.nan
        out[t] = frame_out
    return out


# ---------------------------------------------------------------------------
# StructuralMeshViewerNode
# ---------------------------------------------------------------------------

class StructuralMeshViewerNode(BaseNode):
    """
    See _HELP_TEXT (Ctrl-H) for the full reference.
    """

    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE = "structural_mesh_viewer"
    DISPLAY_NAME = "Structural Mesh Viewer"
    CATEGORY = "process"
    SEARCH_KEYWORDS = ("dic", "structural", "surface", "rigid body", "strain",
                       "deformation", "q4", "kabsch", "mesh viewer",
                       "displacement field", "von mises", "shm", "video export",
                       "time series", "overlay", "project points")
    NODE_WIDTH = 220
    NODE_HEIGHT = 120

    HELP_TEXT = _HELP_TEXT

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

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[
                PinDef("points", PinType.ARRAY, "pts", optional=False),
                PinDef("x_hint", PinType.ARRAY, "xHint", optional=True),
                PinDef("normal_hint", PinType.ARRAY, "nHint", optional=True),
                PinDef("img_files", PinType.STRING, "img_files", optional=True),
                PinDef("cmat", PinType.ARRAY, "cmat", optional=True),
                PinDef("dvec", PinType.ARRAY, "dvec", optional=True),
                PinDef("rvec", PinType.ARRAY, "rvec", optional=True),
                PinDef("tvec", PinType.ARRAY, "tvec", optional=True),
            ],
            outputs=[
                PinDef("local_coords", PinType.ARRAY, "local"),
                PinDef("rbm_translation", PinType.ARRAY, "trans"),
                PinDef("rbm_rotation", PinType.ARRAY, "rot"),
                PinDef("strain_fields", PinType.ARRAY, "strain"),
                PinDef("ready", PinType.SCALAR, "ready"),
            ],
        )

    def get_help_text(self) -> str:
        return _HELP_TEXT

    # ── state ─────────────────────────────────────────────────────

    def _init_state(self) -> None:
        if hasattr(self, "_x_hint_var"):
            return

        self._x_hint_var = tk.StringVar(value="")
        self._normal_hint_var = tk.StringVar(value="")
        self._status_var = tk.StringVar(value="waiting for points input")
        self._diag_var = tk.StringVar(value="")

        self._points_ref = None
        self._points_shape = None
        self._local_coords: np.ndarray | None = None
        self._translations: np.ndarray | None = None
        self._rotations: np.ndarray | None = None
        self._strain_fields: np.ndarray | None = None
        self._r0: np.ndarray | None = None
        self._b_all = None
        self._jac_ok = None
        self._nt = 0
        self._mp = 0
        self._nq = 0
        self._ready = False

        # camera inputs (Field Pcolor on Image)
        self._img_files: list[str] | None = None
        self._img_files_cache_path: str | None = None
        self._img_files_cache_frame: np.ndarray | None = None
        self._cam_cmat: np.ndarray | None = None
        self._cam_dvec: np.ndarray | None = None
        self._cam_rvec: np.ndarray | None = None
        self._cam_tvec: np.ndarray | None = None
        self._cam_status_var = tk.StringVar(value="")

        # shared time control
        self._frame_idx_var = tk.IntVar(value=0)
        self._fps_var = tk.DoubleVar(value=5.0)
        self._playing = False
        self._anim_after_id = None
        self._scrubber: tk.Scale | None = None
        self._frame_label_var = tk.StringVar(value="t = 0 / 0")
        self._notebook: ttk.Notebook | None = None
        self._first_btn: tk.Button | None = None
        self._prev_btn: tk.Button | None = None
        self._play_btn: tk.Button | None = None
        self._stop_btn: tk.Button | None = None
        self._next_btn: tk.Button | None = None
        self._last_btn: tk.Button | None = None
        self._record_btn: tk.Button | None = None
        self._record_status_var = tk.StringVar(value="")
        self._video_quality_var = tk.IntVar(value=95)
        self._video_nstripes_var = tk.IntVar(value=-1)
        self._video_start_frame_var = tk.IntVar(value=0)
        self._video_end_frame_var = tk.IntVar(value=-1)  # -1 = last frame

        self._field_label_to_key = {label: k for k, label in _FIELD_OPTIONS}
        self._field_key_to_label = dict(_FIELD_OPTIONS)
        self._surface_field_label_to_key = {"None (wireframe only)": "none"}
        self._surface_field_label_to_key.update(self._field_label_to_key)
        self._surface_field_key_to_label = {"none": "None (wireframe only)"}
        self._surface_field_key_to_label.update(self._field_key_to_label)

        # video recording state
        self._recording = False
        self._record_writer = None
        self._record_bbox = None
        self._record_frame_idx = 0
        self._record_start_frame = 0
        self._record_end_frame = 0
        self._record_win_size = (0, 0)
        self._record_path = ""
        self._ImageGrab = None

        # tab2: 3D animation
        self._view_mode_var = tk.StringVar(value="global")
        self._amp_x_var = tk.DoubleVar(value=1.0)
        self._amp_y_var = tk.DoubleVar(value=1.0)
        self._amp_z_var = tk.DoubleVar(value=10.0)
        self._show_points3d_var = tk.BooleanVar(value=True)
        self._show_lines3d_var = tk.BooleanVar(value=True)
        self._show_index_labels_var = tk.BooleanVar(value=True)
        self._preview3d_canvas: tk.Canvas | None = None
        self._cam_state = {
            "yaw": -0.9, "pitch": 0.7, "zoom": 1.0,
            "camera_target": [0.0, 0.0, 0.0],
            "dragging": False, "last_x": 0, "last_y": 0, "button": None,
        }
        # 0.0 = orthographic (parallel projection, no depth foreshortening).
        # 1.0 = strong perspective. Updated live from the "Perspective
        # Intensity" slider; see _cam_project() for how it's applied.
        self._perspective_var = tk.DoubleVar(value=0.35)
        self._cam_last_radius: float = 1.0
        self._surface_field_var = tk.StringVar(value="none")
        self._surface_cmap_var = tk.StringVar(value="jet")
        self._surface_clim_auto_var = tk.BooleanVar(value=True)
        self._surface_clim_min_var = tk.DoubleVar(value=0.0)
        self._surface_clim_max_var = tk.DoubleVar(value=1.0)
        self._surface_field_combo: ttk.Combobox | None = None

        # tab3: section plot
        self._section_axis_var = tk.StringVar(value="axis0")
        self._section_index_var = tk.IntVar(value=0)
        self._section_value_var = tk.StringVar(value="z")
        self._section_amp_var = tk.DoubleVar(value=10.0)
        self._section_xaxis_var = tk.StringVar(value="index")
        self._section_fig = None
        self._section_ax = None
        self._section_canvas_widget: FigureCanvasTkAgg | None = None
        self._section_index_sb: tk.Spinbox | None = None
        self._section_line = None
        self._section_needs_rebuild = True
        self._section_full_extent = None
        self._section_pz_state: dict = {}
        self._section_xlabel_var = tk.StringVar(value="point index along section")
        self._section_ylabel_var = tk.StringVar(value="local z'")

        # tab4: field pcolor
        self._field_var = tk.StringVar(value="vm")
        self._field_res_var = tk.StringVar(value="element")
        self._cmap_var = tk.StringVar(value="jet")
        self._clim_auto_var = tk.BooleanVar(value=True)
        self._clim_min_var = tk.DoubleVar(value=0.0)
        self._clim_max_var = tk.DoubleVar(value=1.0)
        self._field_fig = None
        self._field_ax = None
        self._field_cax = None
        self._field_canvas_widget: FigureCanvasTkAgg | None = None
        self._field_colorbar = None
        self._field_mesh = None
        self._field_grid_sig = None
        self._field_needs_rebuild = True
        self._field_full_extent = None
        self._field_pz_state: dict = {}
        self._field_combo: ttk.Combobox | None = None
        self._res_element_rb: tk.Radiobutton | None = None
        self._res_gauss_rb: tk.Radiobutton | None = None
        self._field_xlabel_var = tk.StringVar(value="local x'")
        self._field_ylabel_var = tk.StringVar(value="local y'")
        self._field_shading_var = tk.StringVar(value="flat")

        # tab0: time-series plot (new)
        self._ts_slots: dict[int, dict] = {}
        for k in range(1, 7):
            self._ts_slots[k] = {
                "enabled_var": tk.BooleanVar(value=(k == 1)),
                "source_var": tk.StringVar(value=_TS_DEFAULT_SOURCES[k - 1]),
                "point_i_var": tk.IntVar(value=0),
                "point_j_var": tk.IntVar(value=0),
                "mode_var": tk.StringVar(value="coord"),
                "color_var": tk.StringVar(value=_TS_DEFAULT_COLORS[k - 1]),
                "marker_var": tk.StringVar(value=_TS_DEFAULT_MARKERS[k - 1]),
                "marker_size_var": tk.DoubleVar(value=4.0),
                "line_width_var": tk.DoubleVar(value=1.2),
                "_src_combo": None,
                "_i_sb": None,
                "_j_sb": None,
            }
        self._ts_ylim_auto_var = tk.BooleanVar(value=True)
        self._ts_ylim_min_var = tk.DoubleVar(value=0.0)
        self._ts_ylim_max_var = tk.DoubleVar(value=1.0)
        self._ts_fig = None
        self._ts_ax = None
        self._ts_canvas_widget: FigureCanvasTkAgg | None = None
        self._ts_playhead = None
        self._ts_full_extent = None
        self._ts_pz_state: dict = {}
        self._ts_xlabel_var = tk.StringVar(value="frame index (t)")
        self._ts_ylabel_var = tk.StringVar(value="value")

        # tab5: field pcolor on image (new)
        self._img_field_var = tk.StringVar(value="vm")
        self._img_field_res_var = tk.StringVar(value="element")
        self._img_cmap_var = tk.StringVar(value="jet")
        self._img_clim_auto_var = tk.BooleanVar(value=True)
        self._img_clim_min_var = tk.DoubleVar(value=0.0)
        self._img_clim_max_var = tk.DoubleVar(value=1.0)
        self._img_alpha_var = tk.DoubleVar(value=0.6)
        self._img_fig = None
        self._img_ax = None
        self._img_cax = None
        self._img_canvas_widget: FigureCanvasTkAgg | None = None
        self._img_field_combo: ttk.Combobox | None = None
        self._img_res_element_rb: tk.Radiobutton | None = None
        self._img_res_gauss_rb: tk.Radiobutton | None = None
        self._img_full_extent = None
        self._img_pz_state: dict = {}
        self._img_xlabel_var = tk.StringVar(value="image x (pixels)")
        self._img_ylabel_var = tk.StringVar(value="image y (pixels)")
        self._img_shading_var = tk.StringVar(value="flat")

        self._help_popup: tk.Toplevel | None = None

    # ── body ──────────────────────────────────────────────────────

    def build_body(self) -> None:
        self._init_state()
        x, y, w, h = self.x, self.y, self.width, self.height
        self._body_rect = self.canvas.create_rectangle(
            x, y, x + w, y + h, fill="#eef3ff", outline="#33558a", width=2,
            tags=(self.node_id, "node_body"))
        self._title_item = self.canvas.create_text(
            x + w / 2, y + 13, text=self.DISPLAY_NAME,
            font=("Arial", 9, "bold"), fill="#1a2f55", tags=(self.node_id,))
        status_lbl = tk.Label(
            self.canvas, textvariable=self._status_var, font=("Arial", 8),
            bg="#eef3ff", fg="#33558a", wraplength=w - 12, justify="center")
        self.canvas.create_window(x + w / 2, y + h // 2 + 8, window=status_lbl,
                                  tags=(self.node_id,))
        self._canvas_items += [self._body_rect, self._title_item]
        self.canvas.tag_bind(self.node_id, "<Double-Button-1>", lambda e: self.open_inspector())

    # ── inspector ─────────────────────────────────────────────────

    def build_inspector(self, parent: tk.Frame) -> None:
        self._init_state()
        for child in parent.winfo_children():
            child.destroy()

        win = self._inspector_win
        if win is not None:
            for seq in ("<Control-h>", "<Control-H>"):
                win.bind(seq, lambda e: self._open_help())
            win.protocol("WM_DELETE_WINDOW", self._on_inspector_closing_wrapper(win))

        self._build_time_control_bar(parent)

        self._notebook = ttk.Notebook(parent)
        self._notebook.pack(fill="both", expand=True)
        self._notebook.bind("<<NotebookTabChanged>>", lambda e: self._redraw_active_tab())

        tab1 = tk.Frame(self._notebook)
        tab2 = tk.Frame(self._notebook)
        tab3 = tk.Frame(self._notebook)
        tab4 = tk.Frame(self._notebook)
        tab5 = tk.Frame(self._notebook)
        self._notebook.add(tab1, text="Data & Decomposition")
        self._notebook.add(tab2, text="3D Animation")
        self._notebook.add(tab3, text="Section Plot")
        self._notebook.add(tab4, text="Field Pcolor")
        self._notebook.add(tab5, text="Field Pcolor on Image")

        self._build_tab_data(tab1)
        self._build_tab_3d(tab2)
        self._build_tab_section(tab3)
        self._build_tab_field(tab4)
        self._build_tab_field_on_image(tab5)

        if win is not None:
            win.update_idletasks()
            win.minsize(1080, 740)
            win.geometry("1140x780")

        self._redraw_active_tab()

    def _on_inspector_closing_wrapper(self, win):
        def _close():
            self._stop_animation()
            if self._recording:
                self._finish_recording(cancelled=True)
            win.destroy()
        return _close

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

    # ── resizable left/right split, shared by all tabs ──────────────

    def _make_paned(self, parent: tk.Frame, left_bg: str = "#f7f7f7",
                    right_bg: str = "#ffffff", left_width: int = 260,
                    left_minsize: int = 180, right_minsize: int = 320
                    ) -> tuple[tk.PanedWindow, tk.Frame, tk.Frame]:
        panes = tk.PanedWindow(parent, orient=tk.HORIZONTAL, sashwidth=6,
                               sashrelief=tk.RAISED, showhandle=True)
        panes.pack(fill="both", expand=True)
        left = tk.Frame(panes, bg=left_bg)
        right = tk.Frame(panes, bg=right_bg)
        panes.add(left, width=left_width, minsize=left_minsize)
        panes.add(right, minsize=right_minsize)
        self._install_paned_autostretch(panes, right)
        return panes, left, right

    def _install_paned_autostretch(self, panes: tk.PanedWindow, right: tk.Frame) -> None:
        def _apply(total_width: int) -> None:
            if total_width < 10:
                return
            try:
                sash_x, _sash_y = panes.sash_coord(0)
            except Exception:
                return
            try:
                sash_w = int(panes.cget("sashwidth"))
            except Exception:
                sash_w = 6
            new_right_w = max(100, total_width - sash_x - sash_w - 2)
            try:
                panes.paneconfigure(right, width=new_right_w)
            except Exception:
                pass

        def _on_configure(event) -> None:
            if event.widget is not panes:
                return
            _apply(event.width)

        panes.bind("<Configure>", _on_configure)
        panes.after(60, lambda: _apply(panes.winfo_width()))

    # ── generic mouse pan/zoom for embedded matplotlib axes ─────────

    def _pz_on_press(self, event, ax, canvas, state, extent_getter) -> None:
        if event.inaxes != ax:
            return
        if event.dblclick and event.button == 1:
            extent = extent_getter()
            if extent is not None:
                (x0, x1), (y0, y1) = extent
                ax.set_xlim(x0, x1)
                ax.set_ylim(y0, y1)
                canvas.draw_idle()
            return
        if event.button == 1 and event.xdata is not None and event.ydata is not None:
            bbox = ax.get_window_extent()
            xlim0 = ax.get_xlim()
            ylim0 = ax.get_ylim()
            state["panning"] = True
            state["x0data"] = event.xdata
            state["y0data"] = event.ydata
            state["xlim0"] = xlim0
            state["ylim0"] = ylim0
            state["scale_x"] = (xlim0[1] - xlim0[0]) / max(bbox.width, 1e-6)
            state["scale_y"] = (ylim0[1] - ylim0[0]) / max(bbox.height, 1e-6)
            state["bbox_x0"] = bbox.x0
            state["bbox_y0"] = bbox.y0

    def _pz_on_motion(self, event, ax, canvas, state) -> None:
        if not state.get("panning"):
            return
        if event.x is None or event.y is None:
            return
        new_x0 = state["x0data"] - (event.x - state["bbox_x0"]) * state["scale_x"]
        new_y0 = state["y0data"] - (event.y - state["bbox_y0"]) * state["scale_y"]
        width0 = state["xlim0"][1] - state["xlim0"][0]
        height0 = state["ylim0"][1] - state["ylim0"][0]
        ax.set_xlim(new_x0, new_x0 + width0)
        ax.set_ylim(new_y0, new_y0 + height0)
        canvas.draw_idle()

    def _pz_on_release(self, _event, state) -> None:
        state["panning"] = False

    def _pz_on_scroll(self, event, ax, canvas) -> None:
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return
        factor = 0.9 if event.button == "up" else (1.0 / 0.9)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xdata, ydata = event.xdata, event.ydata
        new_w = (xlim[1] - xlim[0]) * factor
        new_h = (ylim[1] - ylim[0]) * factor
        relx = (xdata - xlim[0]) / (xlim[1] - xlim[0]) if xlim[1] != xlim[0] else 0.5
        rely = (ydata - ylim[0]) / (ylim[1] - ylim[0]) if ylim[1] != ylim[0] else 0.5
        ax.set_xlim(xdata - new_w * relx, xdata + new_w * (1 - relx))
        ax.set_ylim(ydata - new_h * rely, ydata + new_h * (1 - rely))
        canvas.draw_idle()

    def _install_pan_zoom(self, ax, canvas, state, extent_getter) -> None:
        canvas.mpl_connect(
            "button_press_event",
            lambda e: self._pz_on_press(e, ax, canvas, state, extent_getter))
        canvas.mpl_connect(
            "motion_notify_event",
            lambda e: self._pz_on_motion(e, ax, canvas, state))
        canvas.mpl_connect(
            "button_release_event",
            lambda e: self._pz_on_release(e, state))
        canvas.mpl_connect(
            "scroll_event",
            lambda e: self._pz_on_scroll(e, ax, canvas))

    def _get_cmap_obj(self, name: str):
        try:
            return matplotlib.colormaps.get_cmap(name)
        except Exception:
            return plt.get_cmap(name)

    # ── pcolormesh cell/node value & shading adaptation ──────────────
    # Shared by the Field Pcolor and Field Pcolor on Image tabs. Both tabs'
    # data comes back from _get_field_frame_data() following one of two
    # shape contracts: nodal fields have values/X/Y all the same shape,
    # while cell fields (element-averaged or per-Gauss-point) have X/Y one
    # larger than values in each dimension (the cell corners). matplotlib's
    # shading='flat' requires the cell-field shape relationship; 'nearest'
    # and 'gouraud' both require X/Y and C to share one shape instead.

    @staticmethod
    def _shrink_corners_to_cells(node_grid: np.ndarray) -> np.ndarray:
        """Average each cell's 4 corner values down to one value per cell."""
        n0, n1 = node_grid.shape[0] - 1, node_grid.shape[1] - 1
        stack = np.stack(
            [node_grid[di:di + n0, dj:dj + n1] for di, dj in _CORNER_OFFSETS], axis=-1)
        return np.nanmean(stack, axis=-1)

    @staticmethod
    def _expand_cells_to_nodes(cell_grid: np.ndarray) -> np.ndarray:
        """Average adjoining per-cell values onto each shared corner node."""
        ncx, ncy = cell_grid.shape
        mp, nq = ncx + 1, ncy + 1
        sums = np.zeros((mp, nq), dtype=np.float64)
        counts = np.zeros((mp, nq), dtype=np.float64)
        finite = np.isfinite(cell_grid)
        safe_vals = np.where(finite, cell_grid, 0.0)
        for di, dj in _CORNER_OFFSETS:
            sums[di:di + ncx, dj:dj + ncy] += safe_vals
            counts[di:di + ncx, dj:dj + ncy] += finite
        with np.errstate(invalid="ignore", divide="ignore"):
            nodal = sums / counts
        nodal[counts == 0] = np.nan
        return nodal

    def _prepare_mesh_shading(self, values: np.ndarray, xg: np.ndarray, yg: np.ndarray,
                              is_nodal: bool, shading_choice: str):
        """
        Returns (X, Y, C, shading) ready for ax.pcolormesh(), converting
        between cell- and node-resolution as needed for the requested
        shading. 'gouraud' needs nodal data -- cell data (is_nodal=False)
        is mapped onto nodes by averaging the surrounding cells at each
        node.
        """
        if is_nodal:
            if shading_choice == "flat":
                return xg, yg, self._shrink_corners_to_cells(values), "flat"
            return xg, yg, values, shading_choice
        if shading_choice == "gouraud":
            return xg, yg, self._expand_cells_to_nodes(values), "gouraud"
        if shading_choice == "nearest":
            xc = self._shrink_corners_to_cells(xg)
            yc = self._shrink_corners_to_cells(yg)
            return xc, yc, values, "nearest"
        return xg, yg, values, "flat"

    @staticmethod
    def _nearest_fill_bfs(values: np.ndarray, invalid: np.ndarray) -> np.ndarray:
        """Pure-numpy fallback for _fill_invalid_mesh_values() (no scipy):
        multi-source breadth-first nearest-neighbor fill."""
        from collections import deque
        filled = values.copy()
        visited = ~invalid
        nrows, ncols = values.shape
        dq = deque(zip(*np.where(visited)))
        while dq:
            r, c = dq.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < nrows and 0 <= nc < ncols and not visited[nr, nc]:
                    filled[nr, nc] = filled[r, c]
                    visited[nr, nc] = True
                    dq.append((nr, nc))
        return filled

    @staticmethod
    def _fill_invalid_mesh_values(values: np.ndarray) -> np.ndarray:
        """
        Return a copy of `values` with any non-finite (NaN/Inf) entries
        replaced by an interpolated estimate from the surrounding valid
        entries -- index-grid linear interpolation, with nearest-neighbor
        for entries outside the convex hull of valid data (e.g. mesh
        edges/corners) -- so pcolormesh() never receives non-finite data.
        """
        values = np.asarray(values, dtype=np.float64)
        invalid = ~np.isfinite(values)
        if not invalid.any():
            return values
        valid = ~invalid
        if not valid.any():
            return np.zeros_like(values)

        try:
            from scipy.interpolate import griddata

            rows, cols = np.indices(values.shape)
            valid_pts = np.column_stack([rows[valid], cols[valid]])
            valid_vals = values[valid]
            query_pts = np.column_stack([rows[invalid], cols[invalid]])

            estimate = griddata(valid_pts, valid_vals, query_pts, method="linear")
            still_missing = ~np.isfinite(estimate)
            if still_missing.any():
                estimate[still_missing] = griddata(
                    valid_pts, valid_vals, query_pts[still_missing], method="nearest")

            filled = values.copy()
            filled[invalid] = estimate
        except Exception:
            # Degenerate point sets (too few valid points, all collinear,
            # duplicate coordinates, etc.) can make griddata/Qhull raise
            # instead of just returning NaN -- fall back to the
            # dependency-free nearest-neighbor fill rather than propagating.
            filled = StructuralMeshViewerNode._nearest_fill_bfs(values, invalid)

        # Last-resort safety net: pcolormesh must never see non-finite data,
        # so if anything is still non-finite (e.g. griddata silently left
        # NaN behind), replace it with the mean of the valid data.
        still_invalid = ~np.isfinite(filled)
        if still_invalid.any():
            fallback_value = float(np.mean(values[valid])) if valid.any() else 0.0
            filled[still_invalid] = fallback_value
        return filled

    # ── shared time control bar ──────────────────────────────────

    def _build_time_control_bar(self, parent: tk.Frame) -> None:
        bar = tk.LabelFrame(parent, text="Time control (drives 3D Animation / Section Plot / Field Pcolor / Field on Image)",
                            font=("Arial", 9), padx=6, pady=4)
        bar.pack(fill="x", padx=6, pady=4)

        row = tk.Frame(bar)
        row.pack(fill="x")
        self._first_btn = tk.Button(row, text="|<", font=("Arial", 9), width=3,
                                    command=lambda: self._set_frame(0))
        self._first_btn.pack(side="left")
        self._prev_btn = tk.Button(row, text="Prev", font=("Arial", 9), width=5,
                                   command=lambda: self._set_frame(self._frame_idx_var.get() - 1))
        self._prev_btn.pack(side="left", padx=2)
        self._play_btn = tk.Button(row, text="Play", font=("Arial", 9), width=6,
                                   command=self._on_play_pause)
        self._play_btn.pack(side="left", padx=2)
        self._record_btn = tk.Button(row, text="Play & Save...", font=("Arial", 9),
                                     command=self._on_play_and_save)
        self._record_btn.pack(side="left", padx=2)
        self._stop_btn = tk.Button(row, text="Stop", font=("Arial", 9), width=5,
                                   command=self._on_stop)
        self._stop_btn.pack(side="left")
        self._next_btn = tk.Button(row, text="Next", font=("Arial", 9), width=5,
                                   command=lambda: self._set_frame(self._frame_idx_var.get() + 1))
        self._next_btn.pack(side="left", padx=2)
        self._last_btn = tk.Button(row, text=">|", font=("Arial", 9), width=3,
                                   command=lambda: self._set_frame(max(0, self._nt - 1)))
        self._last_btn.pack(side="left")

        tk.Label(row, text="FPS:", font=("Arial", 9)).pack(side="left", padx=(12, 2))
        tk.Spinbox(row, from_=0.1, to=120.0, increment=0.5, textvariable=self._fps_var,
                  width=6, font=("Arial", 9)).pack(side="left")

        tk.Label(row, textvariable=self._frame_label_var, font=("Arial", 9, "bold"),
                fg="#334477").pack(side="left", padx=(12, 0))

        tk.Button(row, text="Help (Ctrl-H)", font=("Arial", 8),
                  command=self._open_help).pack(side="right")

        self._scrubber = tk.Scale(
            bar, from_=0, to=0, orient=tk.HORIZONTAL, showvalue=False,
            variable=self._frame_idx_var, command=self._on_scrub)
        self._scrubber.pack(fill="x", pady=(4, 0))
        # The scrubber is rebuilt from scratch every time the inspector opens,
        # but decomposition may have already run earlier (e.g. before the
        # inspector was ever opened, or on a previous open); resync its range
        # to the current data now instead of leaving it stuck at 0.
        self._scrubber.configure(to=max(0, self._nt - 1))

        tk.Label(bar, textvariable=self._record_status_var, font=("Arial", 8),
                fg="#aa4400", anchor="w").pack(fill="x")

    def _on_play_pause(self) -> None:
        if self._playing:
            self._stop_animation()
        else:
            if self._nt <= 1:
                return
            self._playing = True
            self._play_btn.configure(text="Pause")
            self._tick()

    def _on_stop(self) -> None:
        self._stop_animation()
        self._set_frame(0)

    def _stop_animation(self) -> None:
        self._playing = False
        if self._play_btn is not None and self._play_btn.winfo_exists():
            try:
                self._play_btn.configure(text="Play")
            except Exception:
                pass
        if self._anim_after_id is not None and self._inspector_win is not None:
            try:
                self._inspector_win.after_cancel(self._anim_after_id)
            except Exception:
                pass
        self._anim_after_id = None

    def _tick(self) -> None:
        if not self._playing or self._inspector_win is None or not self._inspector_win.winfo_exists():
            self._playing = False
            return
        next_idx = (self._frame_idx_var.get() + 1) % max(1, self._nt)
        self._set_frame(next_idx)
        interval_ms = max(1, int(1000.0 / max(0.1, self._fps_var.get())))
        self._anim_after_id = self._inspector_win.after(interval_ms, self._tick)

    def _on_scrub(self, _value) -> None:
        self._update_frame_label()
        self._redraw_active_tab()

    def _set_frame(self, idx: int) -> None:
        idx = max(0, min(idx, max(0, self._nt - 1)))
        self._frame_idx_var.set(idx)
        self._update_frame_label()
        self._redraw_active_tab()

    def _update_frame_label(self) -> None:
        self._frame_label_var.set(f"t = {self._frame_idx_var.get()} / {max(0, self._nt - 1)}")

    def _redraw_active_tab(self) -> None:
        if not self._ready or self._notebook is None or not self._notebook.winfo_exists():
            return
        self._notebook.update_idletasks()
        try:
            current = self._notebook.index(self._notebook.select())
        except Exception:
            return
        if current == 0:
            self._update_ts_playhead()
        elif current == 1:
            self._redraw_3d()
        elif current == 2:
            self._redraw_section()
        elif current == 3:
            self._redraw_field()
        elif current == 4:
            self._redraw_field_on_image()

    # ── help popup ────────────────────────────────────────────────

    def _open_help(self) -> None:
        if self._help_popup is not None and self._help_popup.winfo_exists():
            self._help_popup.lift()
            return
        popup = tk.Toplevel()
        popup.title("Structural Mesh Viewer - Help")
        popup.geometry("800x700")
        popup.resizable(True, True)
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
        tk.Button(body, text="Close", font=("Arial", 9), command=popup.destroy
                 ).pack(anchor="e", pady=(8, 0))
        popup.bind("<Escape>", lambda _e: popup.destroy())
        popup.protocol("WM_DELETE_WINDOW", popup.destroy)
        self._help_popup = popup

    # ── Tab 1: Data & Decomposition (+ Time-Series plot) ────────────

    def _build_tab_data(self, parent: tk.Frame) -> None:
        panes, left, right = self._make_paned(parent, left_width=340,
                                              left_minsize=280, right_minsize=320)
        pad = {"padx": 8, "pady": 4}

        left_inner = self._make_scrollable(left)

        hint_frame = tk.LabelFrame(left_inner, text="Sign-disambiguation hints (optional)",
                                   font=("Arial", 9), **pad)
        hint_frame.pack(fill="x", **pad)
        for label, var in [("x' direction hint (e.g. \"1 0 0\"):", self._x_hint_var),
                           ("Normal (z') direction hint:", self._normal_hint_var)]:
            row = tk.Frame(hint_frame)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label, font=("Arial", 9), width=28, anchor="w").pack(side="left")
            tk.Entry(row, textvariable=var, font=("Courier", 9), width=16).pack(side="left")

        tk.Button(left_inner, text="Recompute", font=("Arial", 10, "bold"),
                  bg="#334477", fg="white", activebackground="#445588",
                  relief=tk.FLAT, padx=10, pady=4,
                  command=lambda: self._maybe_recompute(force=True)).pack(anchor="w", **pad)

        status_frame = tk.LabelFrame(left_inner, text="Status", font=("Arial", 9), **pad)
        status_frame.pack(fill="x", **pad)
        tk.Label(status_frame, textvariable=self._status_var, font=("Arial", 9),
                fg="#334477", anchor="w", justify="left", wraplength=300).pack(fill="x")
        tk.Label(status_frame, textvariable=self._diag_var, font=("Arial", 9),
                fg="#555555", anchor="w", justify="left", wraplength=300).pack(fill="x")

        # ── time-series plot settings (below Status) ────────────────
        ts_frame = tk.LabelFrame(left_inner, text="Time-Series Plot", font=("Arial", 9), **pad)
        ts_frame.pack(fill="x", **pad)
        tk.Label(ts_frame,
                text="Up to 6 series: 6-DOF rigid body motion, or global/local\n"
                     "coordinates/displacements at a chosen mesh point.",
                font=("Arial", 7), fg="#888888", justify="left").pack(anchor="w", pady=(0, 4))
        self._build_ts_slots_ui(ts_frame, pad)

        y_frame = tk.LabelFrame(left_inner, text="Y-axis", font=("Arial", 9), **pad)
        y_frame.pack(fill="x", **pad)
        tk.Checkbutton(y_frame, text="Auto Y limits", variable=self._ts_ylim_auto_var,
                      font=("Arial", 9), command=self._redraw_timeseries).pack(anchor="w")
        row = tk.Frame(y_frame)
        row.pack(fill="x", pady=2)
        tk.Label(row, text="Min:", font=("Arial", 9)).pack(side="left")
        tk.Entry(row, textvariable=self._ts_ylim_min_var, width=8, font=("Arial", 9)
                ).pack(side="left", padx=(2, 8))
        tk.Label(row, text="Max:", font=("Arial", 9)).pack(side="left")
        tk.Entry(row, textvariable=self._ts_ylim_max_var, width=8, font=("Arial", 9)
                ).pack(side="left", padx=(2, 0))
        tk.Button(y_frame, text="Apply", font=("Arial", 8),
                  command=self._redraw_timeseries).pack(anchor="w", pady=(2, 0))

        label_frame = tk.LabelFrame(left_inner, text="Axis Labels", font=("Arial", 9), **pad)
        label_frame.pack(fill="x", **pad)
        tk.Label(label_frame, text="X label:", font=("Arial", 9)).pack(anchor="w")
        tk.Entry(label_frame, textvariable=self._ts_xlabel_var, font=("Arial", 9)
                ).pack(fill="x")
        tk.Label(label_frame, text="Y label:", font=("Arial", 9)).pack(anchor="w", pady=(4, 0))
        tk.Entry(label_frame, textvariable=self._ts_ylabel_var, font=("Arial", 9)
                ).pack(fill="x")
        tk.Button(label_frame, text="Apply", font=("Arial", 8),
                  command=self._redraw_timeseries).pack(anchor="w", pady=(2, 0))

        plot_frame = tk.LabelFrame(right, text="Time series  (drag=pan, wheel=zoom, dbl-click=reset)",
                                   font=("Arial", 9), padx=4, pady=4)
        plot_frame.pack(fill="both", expand=True)
        self._ts_fig = plt.Figure(figsize=(5.5, 4.2), dpi=100)
        self._ts_ax = self._ts_fig.add_axes([0.12, 0.12, 0.83, 0.80])
        self._ts_canvas_widget = FigureCanvasTkAgg(self._ts_fig, master=plot_frame)
        self._ts_canvas_widget.get_tk_widget().pack(fill="both", expand=True)
        self._ts_pz_state = {"panning": False}
        self._install_pan_zoom(
            self._ts_ax, self._ts_canvas_widget, self._ts_pz_state,
            lambda: self._ts_full_extent)

        self._redraw_timeseries()

    def _build_ts_slots_ui(self, parent: tk.Frame, pad: dict) -> None:
        for k in range(1, 7):
            cfg = self._ts_slots[k]
            frame = tk.LabelFrame(parent, text=f"Series {k}", font=("Arial", 8), padx=4, pady=3)
            frame.pack(fill="x", pady=(0, 4))

            top = tk.Frame(frame)
            top.pack(fill="x")
            tk.Checkbutton(top, text="On", variable=cfg["enabled_var"], font=("Arial", 8),
                          command=self._redraw_timeseries).pack(side="left")
            src_combo = ttk.Combobox(top, textvariable=cfg["source_var"], values=_TS_SOURCE_OPTIONS,
                                     state="readonly", width=16, font=("Arial", 8))
            src_combo.pack(side="left", padx=(4, 0))
            src_combo.bind("<<ComboboxSelected>>", lambda e, kk=k: self._on_ts_source_changed(kk))
            cfg["_src_combo"] = src_combo

            pt_row = tk.Frame(frame)
            pt_row.pack(fill="x", pady=(2, 0))
            tk.Label(pt_row, text="Point (i,j):", font=("Arial", 8)).pack(side="left")
            i_sb = tk.Spinbox(pt_row, from_=0, to=0, textvariable=cfg["point_i_var"], width=4,
                              font=("Arial", 8), command=self._redraw_timeseries)
            i_sb.pack(side="left", padx=(2, 2))
            j_sb = tk.Spinbox(pt_row, from_=0, to=0, textvariable=cfg["point_j_var"], width=4,
                              font=("Arial", 8), command=self._redraw_timeseries)
            j_sb.pack(side="left")
            cfg["_i_sb"] = i_sb
            cfg["_j_sb"] = j_sb

            mode_row = tk.Frame(frame)
            mode_row.pack(fill="x", pady=(2, 0))
            tk.Radiobutton(mode_row, text="Coord", variable=cfg["mode_var"], value="coord",
                          font=("Arial", 8), command=self._redraw_timeseries).pack(side="left")
            tk.Radiobutton(mode_row, text="Displacement", variable=cfg["mode_var"], value="displacement",
                          font=("Arial", 8), command=self._redraw_timeseries).pack(side="left")

            style_row = tk.Frame(frame)
            style_row.pack(fill="x", pady=(2, 0))
            color_combo = ttk.Combobox(style_row, textvariable=cfg["color_var"], values=_COMMON_COLORS,
                                       width=7, font=("Arial", 8))
            color_combo.pack(side="left")
            color_combo.bind("<<ComboboxSelected>>", lambda e: self._redraw_timeseries())
            color_combo.bind("<FocusOut>", lambda e: self._redraw_timeseries())
            marker_combo = ttk.Combobox(style_row, textvariable=cfg["marker_var"],
                                        values=["None"] + _MARKERS, width=5, state="readonly",
                                        font=("Arial", 8))
            marker_combo.pack(side="left", padx=(2, 0))
            marker_combo.bind("<<ComboboxSelected>>", lambda e: self._redraw_timeseries())

            size_row = tk.Frame(frame)
            size_row.pack(fill="x", pady=(2, 0))
            tk.Label(size_row, text="Marker sz:", font=("Arial", 8)).pack(side="left")
            tk.Spinbox(size_row, from_=1, to=20, increment=0.5, textvariable=cfg["marker_size_var"],
                      width=4, font=("Arial", 8), command=self._redraw_timeseries).pack(
                side="left", padx=(2, 8))
            tk.Label(size_row, text="Line w:", font=("Arial", 8)).pack(side="left")
            tk.Spinbox(size_row, from_=0, to=10, increment=0.5, textvariable=cfg["line_width_var"],
                      width=4, font=("Arial", 8), command=self._redraw_timeseries).pack(
                side="left", padx=(2, 0))

            self._on_ts_source_changed(k, redraw=False)

    def _on_ts_source_changed(self, k: int, redraw: bool = True) -> None:
        cfg = self._ts_slots[k]
        needs_point = cfg["source_var"].get() in _TS_POINT_SOURCES
        state = "normal" if needs_point else "disabled"
        for key in ("_i_sb", "_j_sb"):
            w = cfg.get(key)
            if w is not None and w.winfo_exists():
                w.configure(state=state)
        if redraw:
            self._redraw_timeseries()

    def _get_ts_series(self, source: str, i: int, j: int) -> np.ndarray:
        i = max(0, min(i, max(0, self._mp - 1)))
        j = max(0, min(j, max(0, self._nq - 1)))
        if source == "RBM Translation X": return self._translations[:, 0].copy()
        if source == "RBM Translation Y": return self._translations[:, 1].copy()
        if source == "RBM Translation Z": return self._translations[:, 2].copy()
        if source == "RBM Rotation X": return self._rotations[:, 0].copy()
        if source == "RBM Rotation Y": return self._rotations[:, 1].copy()
        if source == "RBM Rotation Z": return self._rotations[:, 2].copy()
        if source == "Global X": return self._points_ref[:, i, j, 0].copy()
        if source == "Global Y": return self._points_ref[:, i, j, 1].copy()
        if source == "Global Z": return self._points_ref[:, i, j, 2].copy()
        if source == "Local X'": return self._local_coords[:, i, j, 0].copy()
        if source == "Local Y'": return self._local_coords[:, i, j, 1].copy()
        if source == "Local Z'": return self._local_coords[:, i, j, 2].copy()
        raise ValueError(f"unknown time-series source: {source}")

    def _ts_auto_label(self, source: str, i: int, j: int, mode: str) -> str:
        mode_label = "disp" if mode == "displacement" else "coord"
        if source in _TS_POINT_SOURCES:
            return f"{source} @({i},{j}) [{mode_label}]"
        return f"{source} [{mode_label}]"

    def _redraw_timeseries(self) -> None:
        if self._ts_ax is None or self._ts_canvas_widget is None:
            return
        ax = self._ts_ax
        ax.clear()
        self._ts_playhead = None

        if not self._ready:
            ax.text(0.5, 0.5, "Compute the decomposition first", ha="center", va="center",
                    transform=ax.transAxes, color="#888888")
            self._ts_canvas_widget.draw_idle()
            return

        t_axis = np.arange(self._nt)
        any_drawn = False
        for k in range(1, 7):
            cfg = self._ts_slots[k]
            if not cfg["enabled_var"].get():
                continue
            source = cfg["source_var"].get()
            i = cfg["point_i_var"].get()
            j = cfg["point_j_var"].get()
            try:
                series = self._get_ts_series(source, i, j)
            except Exception:
                continue
            if cfg["mode_var"].get() == "displacement":
                base = series[0] if series.size else np.nan
                series = series - base
            marker = cfg["marker_var"].get()
            try:
                marker_size = float(cfg["marker_size_var"].get())
                line_width = float(cfg["line_width_var"].get())
            except (tk.TclError, ValueError):
                marker_size, line_width = 4.0, 1.2
            ax.plot(
                t_axis, series,
                color=cfg["color_var"].get() or "black",
                marker=None if marker in ("None", "") else marker,
                markersize=marker_size,
                linewidth=line_width,
                label=self._ts_auto_label(source, i, j, cfg["mode_var"].get()),
            )
            any_drawn = True

        if any_drawn:
            ax.legend(loc="best", fontsize=8)
        else:
            ax.text(0.5, 0.5, "No series enabled -- check a Series checkbox on the left",
                    ha="center", va="center", transform=ax.transAxes,
                    color="#888888", wrap=True)

        ax.set_xlabel(self._ts_xlabel_var.get())
        ax.set_ylabel(self._ts_ylabel_var.get())
        ax.grid(True, linestyle="--", color="#cccccc")
        ax.set_xlim(0, max(1, self._nt - 1))
        if not self._ts_ylim_auto_var.get():
            try:
                ax.set_ylim(self._ts_ylim_min_var.get(), self._ts_ylim_max_var.get())
            except (tk.TclError, ValueError):
                ax.relim()
                ax.autoscale(axis="y")
        else:
            ax.relim()
            ax.autoscale(axis="y")

        self._ts_full_extent = (ax.get_xlim(), ax.get_ylim())
        self._ts_playhead = ax.axvline(
            self._frame_idx_var.get(), color="#333333", linewidth=1, linestyle=":")
        self._ts_canvas_widget.draw_idle()

    def _update_ts_playhead(self) -> None:
        if self._ts_playhead is None or self._ts_canvas_widget is None:
            return
        t = self._frame_idx_var.get()
        self._ts_playhead.set_xdata([t, t])
        self._ts_canvas_widget.draw_idle()

    # ── Tab 2: 3D Animation ───────────────────────────────────────

    def _build_tab_3d(self, parent: tk.Frame) -> None:
        panes, left, right = self._make_paned(parent)
        pad = {"padx": 6, "pady": 4}

        left_inner = self._make_scrollable(left)

        view_frame = tk.LabelFrame(left_inner, text="View", font=("Arial", 9), **pad)
        view_frame.pack(fill="x", **pad)
        tk.Radiobutton(view_frame, text="Global coordinates", variable=self._view_mode_var,
                      value="global", font=("Arial", 9), command=self._redraw_3d).pack(anchor="w")
        tk.Radiobutton(view_frame, text="Local coordinates", variable=self._view_mode_var,
                      value="local", font=("Arial", 9), command=self._redraw_3d).pack(anchor="w")

        cam_frame = tk.LabelFrame(left_inner, text="Camera", font=("Arial", 9), **pad)
        cam_frame.pack(fill="x", **pad)
        tk.Label(cam_frame, text="Perspective Intensity:", font=("Arial", 9)).pack(anchor="w")
        tk.Scale(cam_frame, from_=0.0, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
                variable=self._perspective_var,
                command=lambda _v: self._redraw_3d()).pack(fill="x")
        tk.Label(cam_frame, text="0 = orthographic (parallel projection)\n1 = strong perspective (near/far foreshortening)",
                font=("Arial", 7), fg="#888888", justify="left").pack(anchor="w")

        surf_frame = tk.LabelFrame(left_inner, text="Surface Color", font=("Arial", 9), **pad)
        surf_frame.pack(fill="x", **pad)
        surf_combo = ttk.Combobox(
            surf_frame,
            values=list(self._surface_field_label_to_key.keys()),
            state="readonly", width=26)
        surf_combo.set(self._surface_field_key_to_label.get(
            self._surface_field_var.get(), "None (wireframe only)"))
        surf_combo.bind("<<ComboboxSelected>>", self._on_surface_field_changed)
        surf_combo.pack(fill="x")
        self._surface_field_combo = surf_combo
        tk.Label(surf_frame, text="(always element-averaged; no per-Gauss option here)",
                font=("Arial", 7), fg="#888888", wraplength=220, justify="left").pack(anchor="w")

        cmap_row = tk.Frame(surf_frame)
        cmap_row.pack(fill="x", pady=(4, 0))
        tk.Label(cmap_row, text="Colormap:", font=("Arial", 9)).pack(side="left")
        surf_cmap_combo = ttk.Combobox(cmap_row, textvariable=self._surface_cmap_var,
                                       values=_CMAP_LIST, state="readonly", width=12)
        surf_cmap_combo.pack(side="left", padx=(4, 0))
        surf_cmap_combo.bind("<<ComboboxSelected>>", lambda e: self._redraw_3d())

        tk.Checkbutton(surf_frame, text="Auto color limits", variable=self._surface_clim_auto_var,
                      font=("Arial", 9), command=self._redraw_3d).pack(anchor="w", pady=(4, 0))
        clim_row = tk.Frame(surf_frame)
        clim_row.pack(fill="x")
        tk.Label(clim_row, text="Min:", font=("Arial", 9)).pack(side="left")
        tk.Entry(clim_row, textvariable=self._surface_clim_min_var, width=7,
                font=("Arial", 9)).pack(side="left", padx=(2, 8))
        tk.Label(clim_row, text="Max:", font=("Arial", 9)).pack(side="left")
        tk.Entry(clim_row, textvariable=self._surface_clim_max_var, width=7,
                font=("Arial", 9)).pack(side="left", padx=(2, 0))
        tk.Button(surf_frame, text="Apply", font=("Arial", 8),
                  command=self._redraw_3d).pack(anchor="w", pady=(2, 0))

        amp_frame = tk.LabelFrame(left_inner, text="Amplification (Local view only)",
                                  font=("Arial", 9), **pad)
        amp_frame.pack(fill="x", **pad)
        for label, var in [("x':", self._amp_x_var), ("y':", self._amp_y_var), ("z':", self._amp_z_var)]:
            row = tk.Frame(amp_frame)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=label, font=("Arial", 9), width=4).pack(side="left")
            tk.Spinbox(row, from_=-1000, to=1000, increment=0.5, textvariable=var,
                      width=8, font=("Arial", 9), command=self._redraw_3d).pack(side="left")

        disp_frame = tk.LabelFrame(left_inner, text="Display", font=("Arial", 9), **pad)
        disp_frame.pack(fill="x", **pad)
        tk.Checkbutton(disp_frame, text="Show points", variable=self._show_points3d_var,
                      font=("Arial", 9), command=self._redraw_3d).pack(anchor="w")
        tk.Checkbutton(disp_frame, text="Show mesh lines", variable=self._show_lines3d_var,
                      font=("Arial", 9), command=self._redraw_3d).pack(anchor="w")
        tk.Checkbutton(disp_frame, text="Show axis index labels", variable=self._show_index_labels_var,
                      font=("Arial", 9), command=self._redraw_3d).pack(anchor="w")
        tk.Label(disp_frame, text="Left-drag rotate, right-drag pan, wheel zoom",
                font=("Arial", 7), fg="#888888", wraplength=200, justify="left").pack(anchor="w", pady=(4, 0))

        self._preview3d_canvas = tk.Canvas(right, bg="#ffffff", highlightthickness=0)
        self._preview3d_canvas.pack(fill="both", expand=True)
        c = self._preview3d_canvas
        c.bind("<ButtonPress-1>", self._on_3d_drag_start)
        c.bind("<B1-Motion>", self._on_3d_drag_motion)
        c.bind("<ButtonRelease-1>", self._on_3d_drag_release)
        c.bind("<ButtonPress-3>", self._on_3d_pan_start)
        c.bind("<B3-Motion>", self._on_3d_pan_motion)
        c.bind("<ButtonRelease-3>", self._on_3d_pan_release)
        c.bind("<MouseWheel>", self._on_3d_wheel)
        c.bind("<Configure>", lambda e: self._redraw_3d())

    def _on_surface_field_changed(self, _event=None) -> None:
        label = self._surface_field_combo.get()
        self._surface_field_var.set(self._surface_field_label_to_key.get(label, "none"))
        self._redraw_3d()

    def _cam_rotation_matrix(self) -> np.ndarray:
        yaw, pitch = self._cam_state["yaw"], self._cam_state["pitch"]
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        ryaw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
        rpitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float64)
        return rpitch @ ryaw

    def _cam_scale(self, points_for_fit: np.ndarray) -> float:
        if points_for_fit.size == 0:
            self._cam_last_radius = 1.0
            return 1.0
        radius = max(float(np.nanmax(np.linalg.norm(points_for_fit, axis=1))), 1e-6)
        self._cam_last_radius = radius   # used by _cam_project() to scale the perspective effect
        canvas_min = 200.0
        c = self._preview3d_canvas
        if c is not None and c.winfo_exists():
            canvas_min = min(float(c.winfo_width()), float(c.winfo_height()))
        return canvas_min * 0.38 * self._cam_state["zoom"] / radius

    def _cam_project(self, pts: np.ndarray, scale: float) -> np.ndarray:
        c = self._preview3d_canvas
        if c is None or not c.winfo_exists():
            return np.zeros((len(pts), 2))
        target = np.asarray(self._cam_state["camera_target"], dtype=np.float64)
        rel = pts - target
        r = self._cam_rotation_matrix()
        rot = (r @ rel.T).T
        depth = rot[:, 1]   # +Y (rotated/camera space) = further from the viewer

        intensity = min(max(float(self._perspective_var.get()), 0.0), 1.0)
        if intensity > 1e-6:
            # The camera sits `focal` world-units back from camera_target
            # along the view axis. Smaller focal (higher intensity) puts the
            # camera closer to the model, giving stronger foreshortening; as
            # intensity -> 0, focal -> infinity and this converges to the
            # orthographic projection used below (scale independent of depth).
            radius = max(self._cam_last_radius, 1e-6)
            focal = radius * (4.0 / intensity - 3.0)
            # Floor the denominator so points that pass near/behind the
            # camera plane (denom -> 0 or negative) don't blow up or mirror;
            # they simply saturate at a large-but-finite magnification.
            denom = np.maximum(focal + depth, focal * 0.1)
            persp_factor = focal / denom
        else:
            persp_factor = np.ones_like(depth)

        sx = rot[:, 0] * persp_factor * scale
        sy = -rot[:, 2] * persp_factor * scale
        cw, ch = float(c.winfo_width()), float(c.winfo_height())
        return np.column_stack([sx + cw / 2.0, sy + ch / 2.0])

    def _on_3d_drag_start(self, event) -> None:
        self._cam_state.update(dragging=True, last_x=event.x, last_y=event.y, button=1)

    def _on_3d_drag_motion(self, event) -> None:
        if not self._cam_state["dragging"] or self._cam_state["button"] != 1:
            return
        dx = event.x - self._cam_state["last_x"]
        dy = event.y - self._cam_state["last_y"]
        self._cam_state["yaw"] += dx * 0.01
        self._cam_state["pitch"] += dy * 0.01
        self._cam_state["last_x"], self._cam_state["last_y"] = event.x, event.y
        self._redraw_3d()

    def _on_3d_drag_release(self, _e) -> None:
        self._cam_state.update(dragging=False, button=None)

    def _on_3d_pan_start(self, event) -> None:
        self._cam_state.update(dragging=True, last_x=event.x, last_y=event.y, button=3)

    def _on_3d_pan_motion(self, event) -> None:
        if not self._cam_state["dragging"] or self._cam_state["button"] != 3:
            return
        dx = event.x - self._cam_state["last_x"]
        dy = event.y - self._cam_state["last_y"]
        scale = self._cam_scale(self._last_3d_points_for_scale())
        if scale > 1e-12:
            r = self._cam_rotation_matrix()
            rt = r.T
            world_delta = (rt[:, 0] * dx - rt[:, 2] * dy) / scale
            target = np.asarray(self._cam_state["camera_target"], dtype=np.float64)
            self._cam_state["camera_target"] = (target - world_delta).tolist()
        self._cam_state["last_x"], self._cam_state["last_y"] = event.x, event.y
        self._redraw_3d()

    def _on_3d_pan_release(self, _e) -> None:
        self._cam_state.update(dragging=False, button=None)

    def _on_3d_wheel(self, event) -> str:
        delta = int(event.delta / 120)
        if delta != 0:
            self._cam_state["zoom"] = max(0.05, self._cam_state["zoom"] * (1.12 ** delta))
            self._redraw_3d()
        return "break"

    def _get_display_points_3d(self, frame_idx: int) -> np.ndarray:
        if self._view_mode_var.get() == "global":
            return self._points_ref[frame_idx]
        local_t = self._local_coords[frame_idx]
        local_0 = self._local_coords[0]
        amp = np.array([self._amp_x_var.get(), self._amp_y_var.get(), self._amp_z_var.get()])
        return local_0 + amp * (local_t - local_0)

    def _last_3d_points_for_scale(self) -> np.ndarray:
        if not self._ready:
            return np.empty((0, 3))
        pts = self._get_display_points_3d(self._frame_idx_var.get()).reshape(-1, 3)
        return pts[np.isfinite(pts).all(axis=1)]

    def _element_avg_of_field(self, frame_idx: int, key: str) -> np.ndarray | None:
        if key not in self._field_key_to_label:
            return None
        if key in _DISPLACEMENT_FIELD_KEYS:
            col = {"ux": 0, "uy": 1, "uz": 2}[key]
            nodal = self._local_coords[frame_idx, :, :, col] - self._local_coords[0, :, :, col]
            ncx, ncy = nodal.shape[0] - 1, nodal.shape[1] - 1
            stack = np.stack(
                [nodal[di:di + ncx, dj:dj + ncy] for di, dj in _CORNER_OFFSETS], axis=-1)
            return np.nanmean(stack, axis=-1)
        field_idx = {"exx": 0, "eyy": 1, "gxy": 2, "vm": 3}[key]
        cell_vals = self._strain_fields[frame_idx, :, :, :, field_idx]
        return np.nanmean(cell_vals, axis=2)

    def _draw_colored_surface(self, c: tk.Canvas, grid: np.ndarray, scale: float,
                              target: np.ndarray, r: np.ndarray, surf_key: str) -> None:
        mp, nq = grid.shape[0], grid.shape[1]
        ncx, ncy = mp - 1, nq - 1
        values = self._element_avg_of_field(self._frame_idx_var.get(), surf_key)
        if values is None:
            values = np.full((ncx, ncy), np.nan)

        if self._surface_clim_auto_var.get():
            finite = values[np.isfinite(values)]
            vmin = float(np.min(finite)) if finite.size else 0.0
            vmax = float(np.max(finite)) if finite.size else 1.0
            if vmin == vmax:
                vmax = vmin + 1e-9
        else:
            vmin, vmax = self._surface_clim_min_var.get(), self._surface_clim_max_var.get()

        cmap = self._get_cmap_obj(self._surface_cmap_var.get())
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        quads = []
        for i in range(ncx):
            for j in range(ncy):
                corners = np.array([grid[i, j], grid[i + 1, j], grid[i + 1, j + 1], grid[i, j + 1]])
                if not np.isfinite(corners).all():
                    continue
                rel = corners - target
                rot = (r @ rel.T).T
                depth = float(np.mean(rot[:, 1]))
                proj = self._cam_project(corners, scale)
                val = values[i, j] if i < values.shape[0] and j < values.shape[1] else np.nan
                if np.isfinite(val):
                    face_color = mcolors.to_hex(cmap(norm(val)))
                else:
                    face_color = ""
                edge_color = "#4466aa" if self._show_lines3d_var.get() else ""
                quads.append((depth, proj, face_color, edge_color))

        quads.sort(key=lambda q: q[0], reverse=True)
        for _depth, proj, face_color, edge_color in quads:
            pts = [coord for xy in proj for coord in xy]
            if face_color or edge_color:
                c.create_polygon(*pts, fill=face_color, outline=edge_color, width=1)

        self._draw_surface_legend(c, cmap, vmin, vmax, surf_key)

    def _draw_surface_legend(self, c: tk.Canvas, cmap, vmin: float, vmax: float, key: str) -> None:
        if not c.winfo_exists():
            return
        cw = c.winfo_width()
        ch = c.winfo_height()
        bar_w = 18
        bar_h = min(160, max(60, int(ch * 0.4)))
        x0 = cw - bar_w - 52
        y0 = 24
        n_steps = 40
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        for k in range(n_steps):
            frac_top = 1.0 - k / n_steps
            val = vmin + frac_top * (vmax - vmin)
            color = mcolors.to_hex(cmap(norm(val)))
            seg_y0 = y0 + (k / n_steps) * bar_h
            seg_y1 = y0 + ((k + 1) / n_steps) * bar_h
            c.create_rectangle(x0, seg_y0, x0 + bar_w, seg_y1, fill=color, outline="")
        c.create_rectangle(x0, y0, x0 + bar_w, y0 + bar_h, outline="#333333")
        label = self._field_key_to_label.get(key, key)
        c.create_text(x0 + bar_w / 2, y0 - 12, text=label, font=("Arial", 7),
                      fill="#333333", width=150)
        c.create_text(x0 + bar_w + 6, y0, text=f"{vmax:.4g}", font=("Arial", 7),
                      fill="#333333", anchor="w")
        c.create_text(x0 + bar_w + 6, y0 + bar_h, text=f"{vmin:.4g}", font=("Arial", 7),
                      fill="#333333", anchor="w")

    def _global_to_display_rotation(self) -> np.ndarray:
        if self._view_mode_var.get() != "local" or self._r0 is None or self._rotations is None:
            return np.eye(3)
        t = self._frame_idx_var.get()
        if t >= len(self._rotations):
            return np.eye(3)
        rvec = self._rotations[t]
        if not np.all(np.isfinite(rvec)):
            return np.eye(3)
        r_t, _ = cv2.Rodrigues(rvec)
        return (r_t @ self._r0).T

    def _draw_global_axis_triad(self, c: tk.Canvas) -> None:
        if not c.winfo_exists() or c.winfo_width() < 2:
            return
        cam_r = self._cam_rotation_matrix()
        to_display = self._global_to_display_rotation()

        corner = np.array([32.0, float(c.winfo_height()) - 32.0])
        axis_len = 22.0

        for axis, color, label in [
            (np.array([1.0, 0.0, 0.0]), "#cc0000", "X"),
            (np.array([0.0, 1.0, 0.0]), "#00aa00", "Y"),
            (np.array([0.0, 0.0, 1.0]), "#0000cc", "Z"),
        ]:
            rot = cam_r @ (to_display @ axis)
            end = corner + np.array([rot[0], -rot[2]]) * axis_len
            c.create_line(corner[0], corner[1], end[0], end[1], fill=color, width=2)
            c.create_text(
                end[0] + (end[0] - corner[0]) * 0.2, end[1] + (end[1] - corner[1]) * 0.2,
                text=label, fill=color, font=("Arial", 8, "bold"))

        cube_half = axis_len * 0.32
        verts = self._CUBE_VERTS * cube_half
        faces_by_depth = sorted(
            self._CUBE_FACES,
            key=lambda f: float((cam_r @ (to_display @ f["normal"]))[1]))
        for face in faces_by_depth[:3]:
            pts = []
            for vi in face["verts"]:
                rv = cam_r @ (to_display @ verts[vi])
                pts.append((corner[0] + rv[0], corner[1] - rv[2]))
            c.create_polygon(pts, fill=face["color"], outline="#333333", width=1.3)

    def _draw_mesh_index_labels(self, c: tk.Canvas, grid: np.ndarray, scale: float) -> None:
        if not self._show_index_labels_var.get():
            return
        mp, nq = grid.shape[0], grid.shape[1]
        flat = grid.reshape(-1, 3)
        valid_flat = flat[np.isfinite(flat).all(axis=1)]
        if valid_flat.shape[0] == 0:
            return
        centroid_screen = self._cam_project(valid_flat.mean(axis=0, keepdims=True), scale)[0]

        def _label_edge(edge_pts: np.ndarray, count: int, prefix: str) -> None:
            if count == 0:
                return
            valid = np.isfinite(edge_pts).all(axis=1)
            proj = self._cam_project(edge_pts, scale)
            max_labels = 20
            step = max(1, int(np.ceil(count / max_labels)))
            indices = sorted(set(list(range(0, count, step)) + [count - 1]))
            for idx in indices:
                if idx >= count or not valid[idx]:
                    continue
                px, py = proj[idx]
                dvec = np.array([px, py]) - centroid_screen
                n = np.linalg.norm(dvec)
                dvec = dvec / n if n > 1e-6 else np.array([0.0, -1.0])
                lx, ly = px + dvec[0] * 16, py + dvec[1] * 16
                c.create_text(lx, ly, text=f"{prefix} {idx}",
                              font=("Arial", 6), fill="#663300")

        _label_edge(grid[:, 0, :], mp, "axis-0 index")
        _label_edge(grid[0, :, :], nq, "axis-1 index")

    def _redraw_3d(self) -> None:
        c = self._preview3d_canvas
        if c is None or not c.winfo_exists():
            return
        c.delete("all")
        if not self._ready:
            c.create_text(c.winfo_width() / 2, c.winfo_height() / 2,
                          text="Compute the decomposition first", fill="#888888")
            return
        grid = self._get_display_points_3d(self._frame_idx_var.get())
        pts_flat = grid.reshape(-1, 3)
        scale = self._cam_scale(self._last_3d_points_for_scale())
        target = np.asarray(self._cam_state["camera_target"], dtype=np.float64)
        r = self._cam_rotation_matrix()

        surf_key = self._surface_field_var.get()
        if surf_key != "none":
            self._draw_colored_surface(c, grid, scale, target, r, surf_key)
        else:
            if self._show_lines3d_var.get():
                mp, nq = grid.shape[0], grid.shape[1]
                for i in range(mp):
                    seg = grid[i, :, :]
                    valid_seg = np.isfinite(seg).all(axis=1)
                    proj = self._cam_project(seg, scale)
                    for j in range(nq - 1):
                        if valid_seg[j] and valid_seg[j + 1]:
                            c.create_line(*proj[j], *proj[j + 1], fill="#4466aa", width=1)
                for j in range(grid.shape[1]):
                    seg = grid[:, j, :]
                    valid_seg = np.isfinite(seg).all(axis=1)
                    proj = self._cam_project(seg, scale)
                    for i in range(grid.shape[0] - 1):
                        if valid_seg[i] and valid_seg[i + 1]:
                            c.create_line(*proj[i], *proj[i + 1], fill="#4466aa", width=1)

        if self._show_points3d_var.get():
            valid = np.isfinite(pts_flat).all(axis=1)
            proj = self._cam_project(pts_flat, scale)
            for (px, py), ok in zip(proj, valid):
                if ok:
                    c.create_oval(px - 2, py - 2, px + 2, py + 2, fill="#cc3333", outline="")

        self._draw_mesh_index_labels(c, grid, scale)
        self._draw_global_axis_triad(c)

    # ── Tab 3: Section Plot ──────────────────────────────────────

    def _build_tab_section(self, parent: tk.Frame) -> None:
        panes, left, right = self._make_paned(parent)
        pad = {"padx": 6, "pady": 4}

        cut_frame = tk.LabelFrame(left, text="Section cut (Lagrangian / material-fixed)",
                                  font=("Arial", 9), **pad)
        cut_frame.pack(fill="x", **pad)
        tk.Radiobutton(cut_frame, text="Fix axis-0 index (mp direction)",
                      variable=self._section_axis_var, value="axis0", font=("Arial", 9),
                      command=self._on_section_setting_changed).pack(anchor="w")
        tk.Radiobutton(cut_frame, text="Fix axis-1 index (nq direction)",
                      variable=self._section_axis_var, value="axis1", font=("Arial", 9),
                      command=self._on_section_setting_changed).pack(anchor="w")
        idx_row = tk.Frame(cut_frame)
        idx_row.pack(fill="x", pady=2)
        tk.Label(idx_row, text="Index:", font=("Arial", 9)).pack(side="left")
        self._section_index_sb = tk.Spinbox(
            idx_row, from_=0, to=0, textvariable=self._section_index_var,
            width=6, font=("Arial", 9), command=self._on_section_setting_changed)
        self._section_index_sb.pack(side="left", padx=(4, 0))

        val_frame = tk.LabelFrame(left, text="Plotted quantity", font=("Arial", 9), **pad)
        val_frame.pack(fill="x", **pad)
        for key, label in _SECTION_VALUE_OPTIONS:
            tk.Radiobutton(val_frame, text=label, variable=self._section_value_var, value=key,
                          font=("Arial", 9), command=self._on_section_setting_changed).pack(anchor="w")
        amp_row = tk.Frame(val_frame)
        amp_row.pack(fill="x", pady=(4, 0))
        tk.Label(amp_row, text="Amplification:", font=("Arial", 9)).pack(side="left")
        tk.Spinbox(amp_row, from_=-1000, to=1000, increment=0.5, textvariable=self._section_amp_var,
                  width=8, font=("Arial", 9), command=self._on_section_setting_changed).pack(
            side="left", padx=(4, 0))
        tk.Label(val_frame,
                text="value = ref + amplification * (current - ref)\n(amplification=1 => true shape)",
                font=("Arial", 7), fg="#888888", justify="left").pack(anchor="w")

        xaxis_frame = tk.LabelFrame(left, text="Horizontal axis", font=("Arial", 9), **pad)
        xaxis_frame.pack(fill="x", **pad)
        tk.Radiobutton(xaxis_frame, text="Point index", variable=self._section_xaxis_var,
                      value="index", font=("Arial", 9), command=self._on_section_setting_changed).pack(anchor="w")
        tk.Radiobutton(xaxis_frame, text="In-plane coordinate (auto x' or y')",
                      variable=self._section_xaxis_var, value="coord", font=("Arial", 9),
                      command=self._on_section_setting_changed).pack(anchor="w")

        label_frame = tk.LabelFrame(left, text="Axis Labels", font=("Arial", 9), **pad)
        label_frame.pack(fill="x", **pad)
        tk.Label(label_frame, text="X label:", font=("Arial", 9)).pack(anchor="w")
        tk.Entry(label_frame, textvariable=self._section_xlabel_var, font=("Arial", 9)
                ).pack(fill="x")
        tk.Label(label_frame, text="Y label:", font=("Arial", 9)).pack(anchor="w", pady=(4, 0))
        tk.Entry(label_frame, textvariable=self._section_ylabel_var, font=("Arial", 9)
                ).pack(fill="x")
        tk.Button(label_frame, text="Apply", font=("Arial", 8),
                  command=self._redraw_section).pack(anchor="w", pady=(2, 0))

        plot_frame = tk.LabelFrame(right, text="Section curve  (drag=pan, wheel=zoom, dbl-click=reset)",
                                   font=("Arial", 9), padx=4, pady=4)
        plot_frame.pack(fill="both", expand=True)
        self._section_fig = plt.Figure(figsize=(5.5, 4.2), dpi=100)
        self._section_ax = self._section_fig.add_axes([0.12, 0.12, 0.83, 0.80])
        self._section_canvas_widget = FigureCanvasTkAgg(self._section_fig, master=plot_frame)
        self._section_canvas_widget.get_tk_widget().pack(fill="both", expand=True)
        self._section_pz_state = {"panning": False}
        self._install_pan_zoom(
            self._section_ax, self._section_canvas_widget, self._section_pz_state,
            lambda: self._section_full_extent)

    def _on_section_setting_changed(self) -> None:
        self._section_needs_rebuild = True
        if self._ready:
            axis = self._section_axis_var.get()
            max_idx = (self._mp - 1) if axis == "axis0" else (self._nq - 1)
            if self._section_index_sb is not None and self._section_index_sb.winfo_exists():
                self._section_index_sb.configure(to=max_idx)
            if self._section_index_var.get() > max_idx:
                self._section_index_var.set(0)
        self._redraw_section()

    def _get_section_line(self, frame_idx: int) -> np.ndarray:
        idx = self._section_index_var.get()
        if self._section_axis_var.get() == "axis0":
            idx = min(idx, self._mp - 1)
            return self._local_coords[frame_idx, idx, :, :]
        idx = min(idx, self._nq - 1)
        return self._local_coords[frame_idx, :, idx, :]

    def _redraw_section(self) -> None:
        if self._section_ax is None:
            return
        ax = self._section_ax
        if not self._ready:
            ax.clear()
            self._section_line = None
            ax.text(0.5, 0.5, "Compute the decomposition first", ha="center", va="center",
                    transform=ax.transAxes, color="#888888")
            self._section_canvas_widget.draw_idle()
            return

        t = self._frame_idx_var.get()
        line_ref = self._get_section_line(0)
        line_cur = self._get_section_line(t)
        col_map = {"x": 0, "y": 1, "z": 2}
        col = col_map[self._section_value_var.get()]
        amp = self._section_amp_var.get()
        values = line_ref[:, col] + amp * (line_cur[:, col] - line_ref[:, col])

        n = line_ref.shape[0]
        if self._section_xaxis_var.get() == "index":
            xvals = np.arange(n)
        else:
            rng_x = np.nanmax(line_ref[:, 0]) - np.nanmin(line_ref[:, 0])
            rng_y = np.nanmax(line_ref[:, 1]) - np.nanmin(line_ref[:, 1])
            xvals = line_ref[:, 0] if rng_x >= rng_y else line_ref[:, 1]

        need_rebuild = (self._section_needs_rebuild or self._section_line is None
                        or len(xvals) != len(self._section_line.get_xdata()))

        if need_rebuild:
            ax.clear()
            line, = ax.plot(xvals, values, marker="o", markersize=3, linewidth=1.5, color="#3355aa")
            self._section_line = line
            ax.grid(True, linestyle="--", color="#cccccc")
            ax.relim()
            ax.autoscale()
            self._section_full_extent = (ax.get_xlim(), ax.get_ylim())
            self._section_needs_rebuild = False
        else:
            self._section_line.set_data(xvals, values)

        ax.set_xlabel(self._section_xlabel_var.get())
        ax.set_ylabel(f"{self._section_ylabel_var.get()}  (amp x{amp:g})")
        ax.set_title(f"Section at {'axis0' if self._section_axis_var.get() == 'axis0' else 'axis1'} "
                    f"index {self._section_index_var.get()}  |  t = {t}")
        self._section_canvas_widget.draw_idle()

    # ── Tab 4: Field Pcolor ───────────────────────────────────────

    def _build_tab_field(self, parent: tk.Frame) -> None:
        panes, left, right = self._make_paned(parent)
        pad = {"padx": 6, "pady": 4}

        field_frame = tk.LabelFrame(left, text="Field", font=("Arial", 9), **pad)
        field_frame.pack(fill="x", **pad)
        field_combo = ttk.Combobox(
            field_frame,
            values=[label for _k, label in _FIELD_OPTIONS], state="readonly", width=30)
        field_combo.pack(fill="x")
        field_combo.bind("<<ComboboxSelected>>", self._on_field_combo_changed)
        if self._field_var.get() not in self._field_key_to_label:
            self._field_var.set("vm")
        field_combo.set(self._field_key_to_label[self._field_var.get()])
        self._field_combo = field_combo

        res_frame = tk.LabelFrame(left, text="Strain resolution", font=("Arial", 9), **pad)
        res_frame.pack(fill="x", **pad)
        self._res_element_rb = tk.Radiobutton(
            res_frame, text="Element-averaged (physical geometry)",
            variable=self._field_res_var, value="element", font=("Arial", 9),
            command=self._on_field_resolution_changed)
        self._res_element_rb.pack(anchor="w")
        self._res_gauss_rb = tk.Radiobutton(
            res_frame, text="Per-Gauss-point (index-based layout)",
            variable=self._field_res_var, value="gauss", font=("Arial", 9),
            command=self._on_field_resolution_changed)
        self._res_gauss_rb.pack(anchor="w")
        tk.Label(res_frame, text="(ignored for displacement fields, which are nodal)",
                font=("Arial", 7), fg="#888888").pack(anchor="w")
        self._update_field_res_controls_state()

        cmap_frame = tk.LabelFrame(left, text="Colormap", font=("Arial", 9), **pad)
        cmap_frame.pack(fill="x", **pad)
        cmap_combo = ttk.Combobox(cmap_frame, textvariable=self._cmap_var, values=_CMAP_LIST,
                                  state="readonly", width=14)
        cmap_combo.pack(anchor="w")
        cmap_combo.bind("<<ComboboxSelected>>", lambda e: self._redraw_field())

        clim_frame = tk.LabelFrame(left, text="Color limits", font=("Arial", 9), **pad)
        clim_frame.pack(fill="x", **pad)
        tk.Checkbutton(clim_frame, text="Auto (per-frame)", variable=self._clim_auto_var,
                      font=("Arial", 9), command=self._redraw_field).pack(anchor="w")
        row = tk.Frame(clim_frame)
        row.pack(fill="x", pady=2)
        tk.Label(row, text="Min:", font=("Arial", 9)).pack(side="left")
        tk.Entry(row, textvariable=self._clim_min_var, width=8, font=("Arial", 9)).pack(
            side="left", padx=(2, 8))
        tk.Label(row, text="Max:", font=("Arial", 9)).pack(side="left")
        tk.Entry(row, textvariable=self._clim_max_var, width=8, font=("Arial", 9)).pack(
            side="left", padx=(2, 0))
        tk.Button(clim_frame, text="Apply", font=("Arial", 8),
                  command=self._redraw_field).pack(anchor="w", pady=(2, 0))

        shading_frame = tk.LabelFrame(left, text="Shading", font=("Arial", 9), **pad)
        shading_frame.pack(fill="x", **pad)
        for value, text in (("flat", "Flat"), ("nearest", "Nearest"), ("gouraud", "Gouraud")):
            tk.Radiobutton(shading_frame, text=text, variable=self._field_shading_var,
                          value=value, font=("Arial", 9),
                          command=self._on_field_shading_changed).pack(anchor="w")
        tk.Label(shading_frame,
                text="Gouraud needs nodal data; cell data (element/Gauss)\n"
                     "is first averaged onto the mesh nodes.",
                font=("Arial", 7), fg="#888888", justify="left").pack(anchor="w")

        label_frame = tk.LabelFrame(left, text="Axis Labels", font=("Arial", 9), **pad)
        label_frame.pack(fill="x", **pad)
        tk.Label(label_frame, text="X label:", font=("Arial", 9)).pack(anchor="w")
        tk.Entry(label_frame, textvariable=self._field_xlabel_var, font=("Arial", 9)
                ).pack(fill="x")
        tk.Label(label_frame, text="Y label:", font=("Arial", 9)).pack(anchor="w", pady=(4, 0))
        tk.Entry(label_frame, textvariable=self._field_ylabel_var, font=("Arial", 9)
                ).pack(fill="x")
        tk.Button(label_frame, text="Apply", font=("Arial", 8),
                  command=self._redraw_field).pack(anchor="w", pady=(2, 0))

        plot_frame = tk.LabelFrame(right, text="Field  (drag=pan, wheel=zoom, dbl-click=reset)",
                                   font=("Arial", 9), padx=4, pady=4)
        plot_frame.pack(fill="both", expand=True)
        self._field_fig = plt.Figure(figsize=(5.5, 4.6), dpi=100)
        self._field_ax = self._field_fig.add_axes([0.10, 0.10, 0.72, 0.82])
        self._field_cax = self._field_fig.add_axes([0.86, 0.10, 0.04, 0.82])
        self._field_canvas_widget = FigureCanvasTkAgg(self._field_fig, master=plot_frame)
        self._field_canvas_widget.get_tk_widget().pack(fill="both", expand=True)
        self._field_pz_state = {"panning": False}
        self._install_pan_zoom(
            self._field_ax, self._field_canvas_widget, self._field_pz_state,
            lambda: self._field_full_extent)

    def _update_field_res_controls_state(self) -> None:
        if self._res_element_rb is None or not self._res_element_rb.winfo_exists():
            return
        is_disp = self._field_var.get() in _DISPLACEMENT_FIELD_KEYS
        state = "disabled" if is_disp else "normal"
        self._res_element_rb.configure(state=state)
        if self._res_gauss_rb is not None and self._res_gauss_rb.winfo_exists():
            self._res_gauss_rb.configure(state=state)

    def _on_field_combo_changed(self, _event=None) -> None:
        label = self._field_combo.get()
        self._field_var.set(self._field_label_to_key.get(label, "vm"))
        self._field_needs_rebuild = True
        self._update_field_res_controls_state()
        self._redraw_field()

    def _on_field_resolution_changed(self) -> None:
        self._field_needs_rebuild = True
        self._redraw_field()

    def _on_field_shading_changed(self) -> None:
        self._field_needs_rebuild = True
        self._redraw_field()

    def _get_field_frame_data(self, frame_idx: int,
                              field_key: str | None = None,
                              res_mode: str | None = None):
        """
        Returns (values, x_edges, y_edges, is_nodal) for pcolormesh.

        field_key/res_mode default to this tab's own selection (self._field_var /
        self._field_res_var) when omitted, so existing calls are unaffected;
        the "Field Pcolor on Image" tab passes its OWN field/resolution
        selection explicitly, reusing this same field-value logic without
        duplicating it.
        """
        key = field_key if field_key is not None else self._field_var.get()
        res = res_mode if res_mode is not None else self._field_res_var.get()
        if key not in self._field_key_to_label:
            key = "vm"
        x0 = self._local_coords[0, :, :, 0]
        y0 = self._local_coords[0, :, :, 1]

        if key in _DISPLACEMENT_FIELD_KEYS:
            col = {"ux": 0, "uy": 1, "uz": 2}.get(key, 2)
            values = self._local_coords[frame_idx, :, :, col] - self._local_coords[0, :, :, col]
            return values, x0, y0, True

        field_idx = {"exx": 0, "eyy": 1, "gxy": 2, "vm": 3}.get(key, 3)
        cell_vals = self._strain_fields[frame_idx, :, :, :, field_idx]

        if res == "element":
            avg = np.nanmean(cell_vals, axis=2)
            return avg, x0, y0, False

        ncx, ncy = cell_vals.shape[0], cell_vals.shape[1]
        expanded = np.full((ncx * 2, ncy * 2), np.nan, dtype=np.float64)
        gauss_to_sub = {0: (0, 0), 1: (1, 0), 2: (1, 1), 3: (0, 1)}
        for gi, (sdi, sdj) in gauss_to_sub.items():
            expanded[sdi::2, sdj::2] = cell_vals[:, :, gi]
        x_idx = np.arange(ncx * 2 + 1)
        y_idx = np.arange(ncy * 2 + 1)
        xg, yg = np.meshgrid(x_idx, y_idx, indexing="ij")
        return expanded, xg.astype(np.float64), yg.astype(np.float64), False

    def _redraw_field(self) -> None:
        if self._field_ax is None:
            return
        ax = self._field_ax
        if not self._ready:
            ax.clear()
            if self._field_cax is not None:
                self._field_cax.clear()
            self._field_mesh = None
            self._field_colorbar = None
            ax.text(0.5, 0.5, "Compute the decomposition first", ha="center", va="center",
                    transform=ax.transAxes, color="#888888")
            self._field_canvas_widget.draw_idle()
            return

        t = self._frame_idx_var.get()
        values, xg, yg, is_nodal = self._get_field_frame_data(t)
        shading_choice = self._field_shading_var.get()
        mesh_x, mesh_y, mesh_vals, shading = self._prepare_mesh_shading(
            values, xg, yg, is_nodal, shading_choice)
        # pcolormesh rejects non-finite X/Y just as strictly as non-finite C
        # (e.g. NaN mesh points from lost tracking at t=0), so these need
        # the same interpolated make-up as the color values.
        mesh_x_makeup = self._fill_invalid_mesh_values(mesh_x)
        mesh_y_makeup = self._fill_invalid_mesh_values(mesh_y)
        mesh_vals_makeup = self._fill_invalid_mesh_values(mesh_vals)
        grid_sig = (mesh_vals_makeup.shape, shading)

        cmap = self._cmap_var.get()
        if self._clim_auto_var.get():
            finite = mesh_vals_makeup[np.isfinite(mesh_vals_makeup)]
            vmin = float(np.min(finite)) if finite.size else 0.0
            vmax = float(np.max(finite)) if finite.size else 1.0
            if vmin == vmax:
                vmax = vmin + 1e-9
        else:
            vmin, vmax = self._clim_min_var.get(), self._clim_max_var.get()

        need_rebuild = (self._field_needs_rebuild or self._field_mesh is None
                        or self._field_grid_sig != grid_sig)

        if need_rebuild:
            ax.clear()
            self._field_cax.clear()
            try:
                mesh = ax.pcolormesh(mesh_x_makeup, mesh_y_makeup, mesh_vals_makeup,
                                     cmap=cmap, vmin=vmin, vmax=vmax, shading=shading)
                ax.set_aspect("equal", adjustable="box")
                ax.set_xlim(float(np.nanmin(mesh_x_makeup)), float(np.nanmax(mesh_x_makeup)))
                ax.set_ylim(float(np.nanmin(mesh_y_makeup)), float(np.nanmax(mesh_y_makeup)))
                self._field_full_extent = (ax.get_xlim(), ax.get_ylim())
                self._field_colorbar = self._field_fig.colorbar(mesh, cax=self._field_cax)
                self._field_mesh = mesh
                self._field_grid_sig = grid_sig
                self._field_needs_rebuild = False
            except Exception as e:
                # Too many/degenerate invalid cells can still defeat the
                # NaN make-up above (e.g. shading-specific edge cases) --
                # never let pcolormesh crash the redraw.
                ax.text(0.5, 0.5, f"overlay error: {e}", ha="center", va="center",
                        transform=ax.transAxes, color="#aa0000", wrap=True)
                self._field_mesh = None
                self._field_colorbar = None
                self._field_needs_rebuild = True
        else:
            mesh = self._field_mesh
            try:
                mesh.set_array(mesh_vals_makeup.ravel())
                mesh.set_cmap(cmap)
                mesh.set_clim(vmin, vmax)
                if self._field_colorbar is not None:
                    self._field_colorbar.update_normal(mesh)
            except Exception as e:
                ax.text(0.5, 0.5, f"overlay error: {e}", ha="center", va="center",
                        transform=ax.transAxes, color="#aa0000", wrap=True)
                self._field_needs_rebuild = True

        key_label = self._field_key_to_label.get(self._field_var.get(), self._field_key_to_label["vm"])
        res_label = "" if self._field_var.get() in _DISPLACEMENT_FIELD_KEYS else \
            ("  (element-avg)" if self._field_res_var.get() == "element" else "  (per Gauss pt)")
        ax.set_xlabel(self._field_xlabel_var.get())
        ax.set_ylabel(self._field_ylabel_var.get())
        ax.set_title(f"{key_label}{res_label}  |  t = {t}")
        self._field_canvas_widget.draw_idle()

    # ── Tab 5: Field Pcolor on Image (new) ──────────────────────────

    def _build_tab_field_on_image(self, parent: tk.Frame) -> None:
        panes, left, right = self._make_paned(parent)
        pad = {"padx": 6, "pady": 4}

        field_frame = tk.LabelFrame(left, text="Field", font=("Arial", 9), **pad)
        field_frame.pack(fill="x", **pad)
        img_field_combo = ttk.Combobox(
            field_frame, values=[label for _k, label in _FIELD_OPTIONS],
            state="readonly", width=30)
        img_field_combo.pack(fill="x")
        img_field_combo.bind("<<ComboboxSelected>>", self._on_img_field_combo_changed)
        if self._img_field_var.get() not in self._field_key_to_label:
            self._img_field_var.set("vm")
        img_field_combo.set(self._field_key_to_label[self._img_field_var.get()])
        self._img_field_combo = img_field_combo

        res_frame = tk.LabelFrame(left, text="Strain resolution", font=("Arial", 9), **pad)
        res_frame.pack(fill="x", **pad)
        self._img_res_element_rb = tk.Radiobutton(
            res_frame, text="Element-averaged (mesh node corners)",
            variable=self._img_field_res_var, value="element", font=("Arial", 9),
            command=self._redraw_field_on_image)
        self._img_res_element_rb.pack(anchor="w")
        self._img_res_gauss_rb = tk.Radiobutton(
            res_frame, text="Per-Gauss-point (bilinear sub-quads)",
            variable=self._img_field_res_var, value="gauss", font=("Arial", 9),
            command=self._redraw_field_on_image)
        self._img_res_gauss_rb.pack(anchor="w")
        tk.Label(res_frame,
                text="(per-Gauss uses true interpolated 3D edge-midpoint/\n"
                     "center positions, then projects those -- not an\n"
                     "abstract index grid, so quads align with the photo)",
                font=("Arial", 7), fg="#888888", justify="left").pack(anchor="w")
        self._update_img_field_res_controls_state()

        cmap_frame = tk.LabelFrame(left, text="Colormap", font=("Arial", 9), **pad)
        cmap_frame.pack(fill="x", **pad)
        img_cmap_combo = ttk.Combobox(cmap_frame, textvariable=self._img_cmap_var,
                                      values=_CMAP_LIST, state="readonly", width=14)
        img_cmap_combo.pack(anchor="w")
        img_cmap_combo.bind("<<ComboboxSelected>>", lambda e: self._redraw_field_on_image())

        clim_frame = tk.LabelFrame(left, text="Color limits", font=("Arial", 9), **pad)
        clim_frame.pack(fill="x", **pad)
        tk.Checkbutton(clim_frame, text="Auto (per-frame)", variable=self._img_clim_auto_var,
                      font=("Arial", 9), command=self._redraw_field_on_image).pack(anchor="w")
        row = tk.Frame(clim_frame)
        row.pack(fill="x", pady=2)
        tk.Label(row, text="Min:", font=("Arial", 9)).pack(side="left")
        tk.Entry(row, textvariable=self._img_clim_min_var, width=8, font=("Arial", 9)).pack(
            side="left", padx=(2, 8))
        tk.Label(row, text="Max:", font=("Arial", 9)).pack(side="left")
        tk.Entry(row, textvariable=self._img_clim_max_var, width=8, font=("Arial", 9)).pack(
            side="left", padx=(2, 0))
        tk.Button(clim_frame, text="Apply", font=("Arial", 8),
                  command=self._redraw_field_on_image).pack(anchor="w", pady=(2, 0))

        shading_frame = tk.LabelFrame(left, text="Shading", font=("Arial", 9), **pad)
        shading_frame.pack(fill="x", **pad)
        for value, text in (("flat", "Flat"), ("nearest", "Nearest"), ("gouraud", "Gouraud")):
            tk.Radiobutton(shading_frame, text=text, variable=self._img_shading_var,
                          value=value, font=("Arial", 9),
                          command=self._on_img_field_shading_changed).pack(anchor="w")
        tk.Label(shading_frame,
                text="Gouraud needs nodal data; cell data (element/Gauss)\n"
                     "is first averaged onto the mesh nodes.",
                font=("Arial", 7), fg="#888888", justify="left").pack(anchor="w")

        label_frame = tk.LabelFrame(left, text="Axis Labels", font=("Arial", 9), **pad)
        label_frame.pack(fill="x", **pad)
        tk.Label(label_frame, text="X label:", font=("Arial", 9)).pack(anchor="w")
        tk.Entry(label_frame, textvariable=self._img_xlabel_var, font=("Arial", 9)
                ).pack(fill="x")
        tk.Label(label_frame, text="Y label:", font=("Arial", 9)).pack(anchor="w", pady=(4, 0))
        tk.Entry(label_frame, textvariable=self._img_ylabel_var, font=("Arial", 9)
                ).pack(fill="x")
        tk.Button(label_frame, text="Apply", font=("Arial", 8),
                  command=self._redraw_field_on_image).pack(anchor="w", pady=(2, 0))

        trans_frame = tk.LabelFrame(left, text="Transparency", font=("Arial", 9), **pad)
        trans_frame.pack(fill="x", **pad)
        tk.Scale(trans_frame, from_=0.0, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
                variable=self._img_alpha_var,
                command=lambda _v: self._redraw_field_on_image()).pack(fill="x")
        tk.Label(trans_frame, text="0 = field invisible (see photo only)\n1 = field fully opaque",
                font=("Arial", 7), fg="#888888", justify="left").pack(anchor="w")

        cam_frame = tk.LabelFrame(left, text="Camera inputs", font=("Arial", 9), **pad)
        cam_frame.pack(fill="x", **pad)
        tk.Label(cam_frame, textvariable=self._cam_status_var, font=("Arial", 8),
                fg="#996600", anchor="w", justify="left", wraplength=230).pack(fill="x")

        plot_frame = tk.LabelFrame(
            right, text="Field on Image  (drag=pan, wheel=zoom, dbl-click=reset)",
            font=("Arial", 9), padx=4, pady=4)
        plot_frame.pack(fill="both", expand=True)
        self._img_fig = plt.Figure(figsize=(5.5, 4.6), dpi=100)
        self._img_ax = self._img_fig.add_axes([0.06, 0.06, 0.76, 0.88])
        self._img_cax = self._img_fig.add_axes([0.85, 0.06, 0.04, 0.88])
        self._img_canvas_widget = FigureCanvasTkAgg(self._img_fig, master=plot_frame)
        self._img_canvas_widget.get_tk_widget().pack(fill="both", expand=True)
        self._img_pz_state = {"panning": False}
        self._install_pan_zoom(
            self._img_ax, self._img_canvas_widget, self._img_pz_state,
            lambda: self._img_full_extent)

        self._redraw_field_on_image()

    def _on_img_field_combo_changed(self, _event=None) -> None:
        label = self._img_field_combo.get()
        self._img_field_var.set(self._field_label_to_key.get(label, "vm"))
        self._update_img_field_res_controls_state()
        self._redraw_field_on_image()

    def _on_img_field_shading_changed(self) -> None:
        self._redraw_field_on_image()

    def _update_img_field_res_controls_state(self) -> None:
        if self._img_res_element_rb is None or not self._img_res_element_rb.winfo_exists():
            return
        is_disp = self._img_field_var.get() in _DISPLACEMENT_FIELD_KEYS
        state = "disabled" if is_disp else "normal"
        self._img_res_element_rb.configure(state=state)
        if self._img_res_gauss_rb is not None and self._img_res_gauss_rb.winfo_exists():
            self._img_res_gauss_rb.configure(state=state)

    @staticmethod
    def _coerce_cmat(value) -> np.ndarray | None:
        try:
            return np.asarray(value, dtype=np.float64).reshape(3, 3)
        except Exception:
            return None

    @staticmethod
    def _coerce_dvec(value) -> np.ndarray | None:
        try:
            return np.asarray(value, dtype=np.float64).ravel().reshape(1, -1)
        except Exception:
            return None

    @staticmethod
    def _coerce_vec3(value) -> np.ndarray | None:
        try:
            return np.asarray(value, dtype=np.float64).ravel()[:3].reshape(3, 1)
        except Exception:
            return None

    def _camera_ready(self) -> bool:
        return (bool(self._img_files) and self._cam_cmat is not None
                and self._cam_dvec is not None and self._cam_rvec is not None
                and self._cam_tvec is not None)

    def _get_image_for_frame(self, t: int) -> np.ndarray | None:
        """Return the decoded (RGB) image for frame t from img_files, cached by path."""
        if not self._img_files or not (0 <= t < len(self._img_files)):
            return None
        path = self._img_files[t]
        if path == self._img_files_cache_path and self._img_files_cache_frame is not None:
            return self._img_files_cache_frame
        frame = cv2.imread(path)
        if frame is None:
            return None
        if frame.ndim == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._img_files_cache_path = path
        self._img_files_cache_frame = frame
        return frame

    @staticmethod
    def _subdivide_grid_bilinear(grid3d: np.ndarray, level: int) -> np.ndarray:
        """
        level=1: return grid3d unchanged (mp,nq,3) -- corner nodes only.
        level=2: return a (2*mp-1, 2*nq-1, 3) grid, inserting edge-midpoint
        and cell-center 3D positions via bilinear interpolation of each
        cell's 4 corner nodes. Since the Q4 GEOMETRIC mapping is itself
        bilinear by construction, this simple averaging gives the exact
        physical 3D position at the corresponding parametric location
        (no approximation) -- unlike the plain Field Pcolor tab's
        per-Gauss expansion, which only needs an abstract index grid
        since it never has to align with a real photo.
        """
        if level <= 1:
            return grid3d
        mp, nq, _ = grid3d.shape
        out = np.full((2 * mp - 1, 2 * nq - 1, 3), np.nan, dtype=np.float64)
        out[0::2, 0::2] = grid3d
        out[1::2, 0::2] = (grid3d[:-1, :, :] + grid3d[1:, :, :]) / 2.0
        out[0::2, 1::2] = (grid3d[:, :-1, :] + grid3d[:, 1:, :]) / 2.0
        out[1::2, 1::2] = (grid3d[:-1, :-1, :] + grid3d[1:, :-1, :]
                           + grid3d[:-1, 1:, :] + grid3d[1:, 1:, :]) / 4.0
        return out

    def _redraw_field_on_image(self) -> None:
        if self._img_ax is None or self._img_canvas_widget is None:
            return
        ax = self._img_ax

        had_content = self._img_full_extent is not None
        xlim_prev, ylim_prev = (ax.get_xlim(), ax.get_ylim()) if had_content else (None, None)

        ax.clear()
        if self._img_cax is not None:
            self._img_cax.clear()

        if not self._ready:
            ax.text(0.5, 0.5, "Compute the decomposition first", ha="center", va="center",
                    transform=ax.transAxes, color="#888888")
            self._img_canvas_widget.draw_idle()
            return

        if not self._camera_ready():
            ax.text(0.5, 0.5, "Connect img_files, cmat, dvec, rvec, tvec input pins",
                    ha="center", va="center", transform=ax.transAxes,
                    color="#888888", wrap=True)
            self._img_canvas_widget.draw_idle()
            return

        t = self._frame_idx_var.get()

        img = self._get_image_for_frame(t)
        if img is None:
            ax.text(0.5, 0.5, f"No readable image for frame t={t} in img_files",
                    ha="center", va="center", transform=ax.transAxes,
                    color="#aa0000", wrap=True)
            self._img_canvas_widget.draw_idle()
            return

        values, _xg, _yg, is_nodal = self._get_field_frame_data(
            t, self._img_field_var.get(), self._img_field_res_var.get())

        # The corner/node grid used for px/py must match what `values`
        # expects: pcolormesh with shading="gouraud" (nodal fields, e.g.
        # displacement) needs X/Y the SAME shape as C; shading="flat"
        # (cell fields, e.g. strain) needs X/Y one larger than C in each
        # dimension. Previously this method hard-coded shading="flat"
        # and always used the level-1 (node) grid for px/py, which broke
        # for nodal (displacement) fields -- values there is (mp,nq),
        # matching X/Y exactly, but "flat" demanded (mp-1,nq-1).
        if is_nodal:
            # element-averaged nodal field: X/Y and C are both (mp,nq)
            level = 1
        else:
            level = 1 if self._img_field_res_var.get() == "element" else 2

        global_grid = self._points_ref[t]                 # (mp,nq,3), GLOBAL coords
        sub_grid = self._subdivide_grid_bilinear(global_grid, level)

        flat = sub_grid.reshape(-1, 1, 3).astype(np.float64)
        valid_mask = np.isfinite(flat).all(axis=(1, 2))
        proj = np.full((flat.shape[0], 2), np.nan, dtype=np.float64)
        if valid_mask.any():
            try:
                proj_valid, _ = cv2.projectPoints(
                    flat[valid_mask], self._cam_rvec, self._cam_tvec,
                    self._cam_cmat, self._cam_dvec)
                proj[valid_mask] = proj_valid.reshape(-1, 2)
            except Exception as e:
                ax.text(0.5, 0.5, f"projectPoints error: {e}", ha="center", va="center",
                        transform=ax.transAxes, color="#aa0000", wrap=True)
                self._img_canvas_widget.draw_idle()
                return

        px = proj[:, 0].reshape(sub_grid.shape[0], sub_grid.shape[1])
        py = proj[:, 1].reshape(sub_grid.shape[0], sub_grid.shape[1])

        shading_choice = self._img_shading_var.get()
        mesh_x, mesh_y, mesh_vals, shading = self._prepare_mesh_shading(
            values, px, py, is_nodal, shading_choice)
        # pcolormesh rejects non-finite X/Y just as strictly as non-finite C
        # (px/py go NaN wherever cv2.projectPoints had no valid 3D point to
        # project, e.g. lost tracking), so these need the same interpolated
        # make-up as the color values.
        mesh_x_makeup = self._fill_invalid_mesh_values(mesh_x)
        mesh_y_makeup = self._fill_invalid_mesh_values(mesh_y)
        mesh_vals_makeup = self._fill_invalid_mesh_values(mesh_vals)

        cmap = self._img_cmap_var.get()
        if self._img_clim_auto_var.get():
            finite = mesh_vals_makeup[np.isfinite(mesh_vals_makeup)]
            vmin = float(np.min(finite)) if finite.size else 0.0
            vmax = float(np.max(finite)) if finite.size else 1.0
            if vmin == vmax:
                vmax = vmin + 1e-9
        else:
            vmin, vmax = self._img_clim_min_var.get(), self._img_clim_max_var.get()

        h, w = img.shape[0], img.shape[1]
        if img.ndim == 2:
            ax.imshow(img, extent=[0, w, h, 0], origin="upper", cmap="gray")
        else:
            ax.imshow(img, extent=[0, w, h, 0], origin="upper")

        alpha = float(self._img_alpha_var.get())
        try:
            mesh = ax.pcolormesh(mesh_x_makeup, mesh_y_makeup, mesh_vals_makeup,
                                 cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha, shading=shading)
            self._img_fig.colorbar(mesh, cax=self._img_cax)
        except Exception as e:
            ax.text(0.5, 0.5, f"overlay error: {e}", ha="center", va="center",
                    transform=ax.transAxes, color="#aa0000", wrap=True)

        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_aspect("equal", adjustable="box")
        key_label = self._field_key_to_label.get(self._img_field_var.get(), "?")
        res_label = "" if is_nodal else (
            "  (element-avg)" if self._img_field_res_var.get() == "element" else "  (per Gauss pt)")
        ax.set_xlabel(self._img_xlabel_var.get())
        ax.set_ylabel(self._img_ylabel_var.get())
        ax.set_title(f"{key_label}{res_label}  on image  |  t = {t}")

        self._img_full_extent = ((0, w), (h, 0))
        if had_content and xlim_prev is not None:
            ax.set_xlim(xlim_prev)
            ax.set_ylim(ylim_prev)

        self._img_canvas_widget.draw_idle()

    # ── decomposition trigger ────────────────────────────────────

    def _maybe_recompute(self, points: np.ndarray | None = None, force: bool = False) -> None:
        if points is None:
            points = self._points_ref
        if points is None:
            return
        same_ref = points is self._points_ref and points.shape == self._points_shape
        if same_ref and not force:
            return
        self._run_decomposition(points)

    def _run_decomposition(self, points: np.ndarray) -> None:
        if points.ndim != 4 or points.shape[-1] != 3:
            self._status_var.set(f"error: expected shape (nt,mp,nq,3), got {points.shape}")
            self._ready = False
            return

        x_hint = _parse_vec3(self._x_hint_var.get())
        normal_hint = _parse_vec3(self._normal_hint_var.get())

        try:
            local_coords, translations, rotations, r0, n_bad = _decompose_rbm(
                points, x_hint, normal_hint)
        except Exception as e:
            self._status_var.set(f"decomposition error: {e}")
            self._ready = False
            return

        x0 = local_coords[0, :, :, 0]
        y0 = local_coords[0, :, :, 1]
        b_all, jac_ok = _precompute_q4_b_matrices(x0, y0)
        strain_fields = _compute_all_strains(local_coords, b_all, jac_ok)

        self._points_ref = points
        self._points_shape = points.shape
        self._local_coords = local_coords
        self._translations = translations
        self._rotations = rotations
        self._r0 = r0
        self._strain_fields = strain_fields
        self._b_all = b_all
        self._jac_ok = jac_ok
        self._nt, self._mp, self._nq = points.shape[0], points.shape[1], points.shape[2]
        self._ready = True

        self._section_needs_rebuild = True
        self._field_needs_rebuild = True

        self._status_var.set(
            f"ok  nt={self._nt}  mp={self._mp}  nq={self._nq}"
            + (f"   ({n_bad} frame(s) had <3 valid points, left as NaN)" if n_bad else ""))
        n_bad_jac = int((~jac_ok).sum())
        self._diag_var.set(
            f"translation range: {np.nanmax(translations, axis=0) - np.nanmin(translations, axis=0)}\n"
            f"cells with degenerate reference geometry (no strain, always NaN): {n_bad_jac} / {jac_ok.size}"
        )

        if self._scrubber is not None and self._scrubber.winfo_exists():
            self._scrubber.configure(to=max(0, self._nt - 1))

        for k in range(1, 7):
            cfg = self._ts_slots[k]
            i_sb = cfg.get("_i_sb")
            if i_sb is not None and i_sb.winfo_exists():
                i_sb.configure(to=max(0, self._mp - 1))
            j_sb = cfg.get("_j_sb")
            if j_sb is not None and j_sb.winfo_exists():
                j_sb.configure(to=max(0, self._nq - 1))

        self._set_frame(min(self._frame_idx_var.get(), max(0, self._nt - 1)))
        self._on_section_setting_changed()
        self._redraw_timeseries()
        self._redraw_active_tab()

    # ── video recording ("Play & Save...") ──────────────────────────

    def _show_video_settings_dialog(self) -> bool:
        """
        Modal dialog collecting Quality (0-100), Number of parallel encoding
        stripes (-1 = automatic), and the Start/End frame range to record.
        Quality/stripes map directly to cv2.VIDEOWRITER_PROP_QUALITY and
        cv2.VIDEOWRITER_PROP_NSTRIPES. Pre-filled with sensible defaults
        (start=0, end=last frame); returns True if OK was clicked (values
        stored on self._video_quality_var / self._video_nstripes_var /
        self._video_start_frame_var / self._video_end_frame_var), False if
        cancelled.
        """
        win = self._inspector_win
        if win is None:
            return False

        dlg = tk.Toplevel(win)
        dlg.title("Video Recording Settings")
        dlg.resizable(False, False)
        dlg.transient(win)
        dlg.grab_set()

        result = {"ok": False}

        body = tk.Frame(dlg, padx=12, pady=10)
        body.pack(fill="both", expand=True)

        tk.Label(body, text="Quality (0-100, codec-dependent):", font=("Arial", 9)).grid(
            row=0, column=0, sticky="w", pady=4)
        quality_var = tk.IntVar(value=self._video_quality_var.get())
        tk.Spinbox(body, from_=0, to=100, textvariable=quality_var, width=8,
                  font=("Arial", 9)).grid(row=0, column=1, sticky="w", padx=(8, 0))

        tk.Label(body, text="Parallel stripes (-1 = automatic):", font=("Arial", 9)).grid(
            row=1, column=0, sticky="w", pady=4)
        nstripes_var = tk.IntVar(value=self._video_nstripes_var.get())
        tk.Spinbox(body, from_=-1, to=256, textvariable=nstripes_var, width=8,
                  font=("Arial", 9)).grid(row=1, column=1, sticky="w", padx=(8, 0))

        max_frame = max(0, self._nt - 1)
        tk.Label(body, text="Start frame:", font=("Arial", 9)).grid(
            row=2, column=0, sticky="w", pady=4)
        start_var = tk.IntVar(value=min(max(self._video_start_frame_var.get(), 0), max_frame))
        tk.Spinbox(body, from_=0, to=max_frame, textvariable=start_var, width=8,
                  font=("Arial", 9)).grid(row=2, column=1, sticky="w", padx=(8, 0))

        stored_end = self._video_end_frame_var.get()
        end_default = max_frame if stored_end < 0 or stored_end > max_frame else stored_end
        tk.Label(body, text="End frame:", font=("Arial", 9)).grid(
            row=3, column=0, sticky="w", pady=4)
        end_var = tk.IntVar(value=end_default)
        tk.Spinbox(body, from_=0, to=max_frame, textvariable=end_var, width=8,
                  font=("Arial", 9)).grid(row=3, column=1, sticky="w", padx=(8, 0))
        tk.Label(body, text=f"(0 to {max_frame}; end must be >= start)",
                font=("Arial", 7), fg="#888888").grid(
            row=4, column=0, columnspan=2, sticky="w")

        tk.Label(body, text="Defaults are fine for most users -- just click OK.",
                font=("Arial", 8), fg="#666666").grid(
            row=5, column=0, columnspan=2, sticky="w", pady=(4, 8))

        btn_row = tk.Frame(body)
        btn_row.grid(row=6, column=0, columnspan=2, sticky="e")

        def _on_ok():
            if end_var.get() < start_var.get():
                messagebox.showerror(
                    "Invalid range", "End frame must be >= start frame.", parent=dlg)
                return
            self._video_quality_var.set(quality_var.get())
            self._video_nstripes_var.set(nstripes_var.get())
            self._video_start_frame_var.set(start_var.get())
            self._video_end_frame_var.set(end_var.get())
            result["ok"] = True
            dlg.destroy()

        def _on_cancel():
            result["ok"] = False
            dlg.destroy()

        tk.Button(btn_row, text="OK", font=("Arial", 9), width=8,
                  command=_on_ok).pack(side="left", padx=4)
        tk.Button(btn_row, text="Cancel", font=("Arial", 9), width=8,
                  command=_on_cancel).pack(side="left")

        dlg.update_idletasks()
        try:
            px = win.winfo_rootx() + (win.winfo_width() - dlg.winfo_width()) // 2
            py = win.winfo_rooty() + (win.winfo_height() - dlg.winfo_height()) // 2
            dlg.geometry(f"+{max(0, px)}+{max(0, py)}")
        except Exception:
            pass

        dlg.protocol("WM_DELETE_WINDOW", _on_cancel)
        dlg.wait_window()
        return result["ok"]

    def _on_play_and_save(self) -> None:
        if self._recording:
            self._cancel_recording()
            return
        if not self._ready or self._nt < 1:
            messagebox.showinfo("Not ready", "Compute the decomposition first.")
            return
        if not self._show_video_settings_dialog():
            return
        path = filedialog.asksaveasfilename(
            title="Save animation as video",
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4"), ("AVI video", "*.avi"), ("All files", "*.*")])
        if not path:
            return
        self._start_recording(path)

    @staticmethod
    def _pick_fourcc(path: str):
        ext = path.lower().rsplit(".", 1)[-1] if "." in path else "mp4"
        if ext == "avi":
            return cv2.VideoWriter_fourcc(*"XVID")
        return cv2.VideoWriter_fourcc(*"mp4v")

    def _start_recording(self, path: str) -> None:
        win = self._inspector_win
        if win is None or not win.winfo_exists():
            return
        try:
            from PIL import ImageGrab
        except ImportError:
            messagebox.showerror(
                "Recording unavailable",
                "Saving to video requires Pillow's ImageGrab (screen capture), "
                "supported on Windows and macOS (recent Pillow adds limited "
                "Linux/X11 support). Please install/upgrade Pillow.")
            return
        self._ImageGrab = ImageGrab

        max_frame = max(0, self._nt - 1)
        start_frame = min(max(self._video_start_frame_var.get(), 0), max_frame)
        end_frame = self._video_end_frame_var.get()
        if end_frame < 0 or end_frame > max_frame:
            end_frame = max_frame
        end_frame = max(end_frame, start_frame)
        self._record_start_frame = start_frame
        self._record_end_frame = end_frame

        self._stop_animation()
        self._set_frame(start_frame)
        win.update_idletasks()
        win.lift()
        try:
            win.attributes("-topmost", True)
        except Exception:
            pass
        win.update()

        fps = max(0.1, self._fps_var.get())
        x = win.winfo_rootx()
        y = win.winfo_rooty()
        w = win.winfo_width()
        h = win.winfo_height()
        if w < 2 or h < 2:
            messagebox.showerror("Recording failed", "Inspector window has no visible size.")
            return

        fourcc = self._pick_fourcc(path)
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        if not writer.isOpened():
            messagebox.showerror("Recording failed", f"Could not open video writer for:\n{path}")
            return

        try:
            writer.set(cv2.VIDEOWRITER_PROP_QUALITY, float(self._video_quality_var.get()))
        except Exception:
            pass
        try:
            writer.set(cv2.VIDEOWRITER_PROP_NSTRIPES, float(self._video_nstripes_var.get()))
        except Exception:
            pass

        self._recording = True
        self._record_writer = writer
        self._record_bbox = (x, y, x + w, y + h)
        self._record_win_size = (w, h)
        self._record_frame_idx = start_frame
        self._record_path = path
        if self._record_btn is not None and self._record_btn.winfo_exists():
            self._record_btn.configure(text="Cancel Recording")
        for b in (self._first_btn, self._prev_btn, self._play_btn, self._stop_btn,
                  self._next_btn, self._last_btn):
            if b is not None and b.winfo_exists():
                b.configure(state="disabled")
        if self._scrubber is not None and self._scrubber.winfo_exists():
            self._scrubber.configure(state="disabled")
        total_frames = end_frame - start_frame + 1
        self._record_status_var.set(
            f"Recording frame 1/{total_frames} (t={start_frame}) to {path}  "
            f"(quality={self._video_quality_var.get()}, "
            f"stripes={self._video_nstripes_var.get()}) ...")
        win.after(80, self._record_tick)

    def _record_tick(self) -> None:
        win = self._inspector_win
        if not self._recording or win is None or not win.winfo_exists():
            self._finish_recording()
            return
        win.update_idletasks()
        win.update()
        try:
            img = self._ImageGrab.grab(bbox=self._record_bbox)
            frame_rgb = np.array(img.convert("RGB"))
        except Exception as e:
            messagebox.showerror("Recording error", f"Screen capture failed:\n{e}")
            self._finish_recording()
            return
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        target_w, target_h = self._record_win_size
        if frame_bgr.shape[1] != target_w or frame_bgr.shape[0] != target_h:
            frame_bgr = cv2.resize(frame_bgr, (target_w, target_h))
        self._record_writer.write(frame_bgr)

        total_frames = self._record_end_frame - self._record_start_frame + 1
        if self._record_frame_idx >= self._record_end_frame:
            self._finish_recording(success=True)
            return
        self._record_frame_idx += 1
        self._set_frame(self._record_frame_idx)
        done = self._record_frame_idx - self._record_start_frame + 1
        self._record_status_var.set(
            f"Recording frame {done}/{total_frames} (t={self._record_frame_idx}) "
            f"to {self._record_path} ...")
        win.after(10, self._record_tick)

    def _cancel_recording(self) -> None:
        self._finish_recording(cancelled=True)

    def _finish_recording(self, success: bool = False, cancelled: bool = False) -> None:
        self._recording = False
        if self._record_writer is not None:
            try:
                self._record_writer.release()
            except Exception:
                pass
            self._record_writer = None
        win = self._inspector_win
        if win is not None and win.winfo_exists():
            try:
                win.attributes("-topmost", False)
            except Exception:
                pass
            if self._record_btn is not None and self._record_btn.winfo_exists():
                self._record_btn.configure(text="Play & Save...")
            for b in (self._first_btn, self._prev_btn, self._play_btn, self._stop_btn,
                      self._next_btn, self._last_btn):
                if b is not None and b.winfo_exists():
                    b.configure(state="normal")
            if self._scrubber is not None and self._scrubber.winfo_exists():
                self._scrubber.configure(state="normal")
        if cancelled:
            self._record_status_var.set(f"Recording cancelled (partial file kept): {self._record_path}")
        elif success:
            self._record_status_var.set(f"Saved video: {self._record_path}")
        else:
            self._record_status_var.set("")

    # ── compute ───────────────────────────────────────────────────

    def compute(self, inputs: dict) -> dict:
        self._init_state()

        raw_img_files = inputs.get("img_files")
        if isinstance(raw_img_files, (list, tuple)):
            self._img_files = [str(p) for p in raw_img_files]
        else:
            self._img_files = None
        raw_cmat = inputs.get("cmat")
        self._cam_cmat = self._coerce_cmat(raw_cmat) if raw_cmat is not None else None
        raw_dvec = inputs.get("dvec")
        self._cam_dvec = self._coerce_dvec(raw_dvec) if raw_dvec is not None else None
        raw_rvec = inputs.get("rvec")
        self._cam_rvec = self._coerce_vec3(raw_rvec) if raw_rvec is not None else None
        raw_tvec = inputs.get("tvec")
        self._cam_tvec = self._coerce_vec3(raw_tvec) if raw_tvec is not None else None

        def _ok(v):
            if isinstance(v, np.ndarray):
                return "OK"
            return "OK" if v else "missing"
        self._cam_status_var.set(
            f"img_files: {_ok(self._img_files)}   cmat: {_ok(self._cam_cmat)}   "
            f"dvec: {_ok(self._cam_dvec)}\n"
            f"rvec: {_ok(self._cam_rvec)}   tvec: {_ok(self._cam_tvec)}"
        )

        points = inputs.get("points")
        if points is None:
            self._status_var.set("waiting for points input")
        elif isinstance(points, np.ndarray):
            self._maybe_recompute(points)

        if not self._ready:
            return {"ready": 0.0}

        return {
            "local_coords": self._local_coords,
            "rbm_translation": self._translations,
            "rbm_rotation": self._rotations,
            "strain_fields": self._strain_fields,
            "ready": 1.0,
        }

    # ── serialization ─────────────────────────────────────────────

    def get_params(self) -> dict:
        ts_params = []
        for k in range(1, 7):
            cfg = self._ts_slots[k]
            ts_params.append({
                "enabled": bool(cfg["enabled_var"].get()),
                "source": cfg["source_var"].get(),
                "point_i": cfg["point_i_var"].get(),
                "point_j": cfg["point_j_var"].get(),
                "mode": cfg["mode_var"].get(),
                "color": cfg["color_var"].get(),
                "marker": cfg["marker_var"].get(),
                "marker_size": float(cfg["marker_size_var"].get()),
                "line_width": float(cfg["line_width_var"].get()),
            })
        return {
            "x_hint": self._x_hint_var.get(),
            "normal_hint": self._normal_hint_var.get(),
            "view_mode": self._view_mode_var.get(),
            "amp_x": self._amp_x_var.get(),
            "amp_y": self._amp_y_var.get(),
            "amp_z": self._amp_z_var.get(),
            "perspective_intensity": self._perspective_var.get(),
            "show_points3d": bool(self._show_points3d_var.get()),
            "show_lines3d": bool(self._show_lines3d_var.get()),
            "show_index_labels": bool(self._show_index_labels_var.get()),
            "surface_field": self._surface_field_var.get(),
            "surface_cmap": self._surface_cmap_var.get(),
            "surface_clim_auto": bool(self._surface_clim_auto_var.get()),
            "surface_clim_min": self._surface_clim_min_var.get(),
            "surface_clim_max": self._surface_clim_max_var.get(),
            "section_axis": self._section_axis_var.get(),
            "section_index": self._section_index_var.get(),
            "section_value": self._section_value_var.get(),
            "section_amp": self._section_amp_var.get(),
            "section_xaxis": self._section_xaxis_var.get(),
            "section_xlabel": self._section_xlabel_var.get(),
            "section_ylabel": self._section_ylabel_var.get(),
            "field": self._field_var.get(),
            "field_res": self._field_res_var.get(),
            "field_shading": self._field_shading_var.get(),
            "field_xlabel": self._field_xlabel_var.get(),
            "field_ylabel": self._field_ylabel_var.get(),
            "cmap": self._cmap_var.get(),
            "clim_auto": bool(self._clim_auto_var.get()),
            "clim_min": self._clim_min_var.get(),
            "clim_max": self._clim_max_var.get(),
            "fps": self._fps_var.get(),
            "ts_slots": ts_params,
            "ts_ylim_auto": bool(self._ts_ylim_auto_var.get()),
            "ts_ylim_min": self._ts_ylim_min_var.get(),
            "ts_ylim_max": self._ts_ylim_max_var.get(),
            "ts_xlabel": self._ts_xlabel_var.get(),
            "ts_ylabel": self._ts_ylabel_var.get(),
            "img_field": self._img_field_var.get(),
            "img_field_res": self._img_field_res_var.get(),
            "img_shading": self._img_shading_var.get(),
            "img_xlabel": self._img_xlabel_var.get(),
            "img_ylabel": self._img_ylabel_var.get(),
            "img_cmap": self._img_cmap_var.get(),
            "img_clim_auto": bool(self._img_clim_auto_var.get()),
            "img_clim_min": self._img_clim_min_var.get(),
            "img_clim_max": self._img_clim_max_var.get(),
            "img_alpha": self._img_alpha_var.get(),
            "video_quality": self._video_quality_var.get(),
            "video_nstripes": self._video_nstripes_var.get(),
        }

    def set_params(self, params: dict) -> None:
        self._init_state()
        self._x_hint_var.set(str(params.get("x_hint", "")))
        self._normal_hint_var.set(str(params.get("normal_hint", "")))
        self._view_mode_var.set(str(params.get("view_mode", "global")))
        self._amp_x_var.set(float(params.get("amp_x", 1.0)))
        self._amp_y_var.set(float(params.get("amp_y", 1.0)))
        self._amp_z_var.set(float(params.get("amp_z", 10.0)))
        self._perspective_var.set(
            min(max(float(params.get("perspective_intensity", 0.35)), 0.0), 1.0))
        self._show_points3d_var.set(bool(params.get("show_points3d", True)))
        self._show_lines3d_var.set(bool(params.get("show_lines3d", True)))
        self._show_index_labels_var.set(bool(params.get("show_index_labels", True)))

        valid_field_keys = {k for k, _label in _FIELD_OPTIONS}
        loaded_surface = str(params.get("surface_field", "none"))
        self._surface_field_var.set(
            loaded_surface if loaded_surface in ({"none"} | valid_field_keys) else "none")
        self._surface_cmap_var.set(str(params.get("surface_cmap", "jet")))
        self._surface_clim_auto_var.set(bool(params.get("surface_clim_auto", True)))
        self._surface_clim_min_var.set(float(params.get("surface_clim_min", 0.0)))
        self._surface_clim_max_var.set(float(params.get("surface_clim_max", 1.0)))

        self._section_axis_var.set(str(params.get("section_axis", "axis0")))
        self._section_index_var.set(int(params.get("section_index", 0)))
        self._section_value_var.set(str(params.get("section_value", "z")))
        self._section_amp_var.set(float(params.get("section_amp", 10.0)))
        self._section_xaxis_var.set(str(params.get("section_xaxis", "index")))
        self._section_xlabel_var.set(str(params.get("section_xlabel", "point index along section")))
        self._section_ylabel_var.set(str(params.get("section_ylabel", "local z'")))

        valid_shading = {"flat", "nearest", "gouraud"}
        loaded_field = str(params.get("field", "vm"))
        self._field_var.set(loaded_field if loaded_field in valid_field_keys else "vm")
        self._field_res_var.set(str(params.get("field_res", "element")))
        loaded_field_shading = str(params.get("field_shading", "flat"))
        self._field_shading_var.set(loaded_field_shading if loaded_field_shading in valid_shading else "flat")
        self._field_xlabel_var.set(str(params.get("field_xlabel", "local x'")))
        self._field_ylabel_var.set(str(params.get("field_ylabel", "local y'")))
        self._cmap_var.set(str(params.get("cmap", "jet")))
        self._clim_auto_var.set(bool(params.get("clim_auto", True)))
        self._clim_min_var.set(float(params.get("clim_min", 0.0)))
        self._clim_max_var.set(float(params.get("clim_max", 1.0)))
        self._fps_var.set(float(params.get("fps", 5.0)))

        saved_ts = params.get("ts_slots", [])
        for k in range(1, 7):
            cfg = self._ts_slots[k]
            saved = saved_ts[k - 1] if k - 1 < len(saved_ts) else {}
            cfg["enabled_var"].set(bool(saved.get("enabled", k == 1)))
            source = str(saved.get("source", _TS_DEFAULT_SOURCES[k - 1]))
            cfg["source_var"].set(source if source in _TS_SOURCE_OPTIONS else _TS_DEFAULT_SOURCES[k - 1])
            cfg["point_i_var"].set(int(saved.get("point_i", 0)))
            cfg["point_j_var"].set(int(saved.get("point_j", 0)))
            cfg["mode_var"].set(str(saved.get("mode", "coord")))
            cfg["color_var"].set(str(saved.get("color", _TS_DEFAULT_COLORS[k - 1])))
            cfg["marker_var"].set(str(saved.get("marker", _TS_DEFAULT_MARKERS[k - 1])))
            cfg["marker_size_var"].set(float(saved.get("marker_size", 4.0)))
            cfg["line_width_var"].set(float(saved.get("line_width", 1.2)))
        self._ts_ylim_auto_var.set(bool(params.get("ts_ylim_auto", True)))
        self._ts_ylim_min_var.set(float(params.get("ts_ylim_min", 0.0)))
        self._ts_ylim_max_var.set(float(params.get("ts_ylim_max", 1.0)))
        self._ts_xlabel_var.set(str(params.get("ts_xlabel", "frame index (t)")))
        self._ts_ylabel_var.set(str(params.get("ts_ylabel", "value")))

        loaded_img_field = str(params.get("img_field", "vm"))
        self._img_field_var.set(loaded_img_field if loaded_img_field in valid_field_keys else "vm")
        self._img_field_res_var.set(str(params.get("img_field_res", "element")))
        loaded_img_shading = str(params.get("img_shading", "flat"))
        self._img_shading_var.set(loaded_img_shading if loaded_img_shading in valid_shading else "flat")
        self._img_xlabel_var.set(str(params.get("img_xlabel", "image x (pixels)")))
        self._img_ylabel_var.set(str(params.get("img_ylabel", "image y (pixels)")))
        self._img_cmap_var.set(str(params.get("img_cmap", "jet")))
        self._img_clim_auto_var.set(bool(params.get("img_clim_auto", True)))
        self._img_clim_min_var.set(float(params.get("img_clim_min", 0.0)))
        self._img_clim_max_var.set(float(params.get("img_clim_max", 1.0)))
        self._img_alpha_var.set(float(params.get("img_alpha", 0.6)))

        self._video_quality_var.set(int(params.get("video_quality", 95)))
        self._video_nstripes_var.set(int(params.get("video_nstripes", -1)))

    def close_inspector(self) -> None:
        self._stop_animation()
        if self._recording:
            self._finish_recording(cancelled=True)
        if self._section_fig is not None:
            plt.close(self._section_fig)
        if self._field_fig is not None:
            plt.close(self._field_fig)
        if self._ts_fig is not None:
            plt.close(self._ts_fig)
        if self._img_fig is not None:
            plt.close(self._img_fig)

        self._section_fig = None
        self._section_ax = None
        self._section_canvas_widget = None
        self._section_line = None
        self._section_needs_rebuild = True
        self._section_full_extent = None
        self._section_pz_state = {}

        self._field_fig = None
        self._field_ax = None
        self._field_cax = None
        self._field_canvas_widget = None
        self._field_colorbar = None
        self._field_mesh = None
        self._field_grid_sig = None
        self._field_needs_rebuild = True
        self._field_full_extent = None
        self._field_pz_state = {}

        self._ts_fig = None
        self._ts_ax = None
        self._ts_canvas_widget = None
        self._ts_playhead = None
        self._ts_full_extent = None
        self._ts_pz_state = {}
        for k in range(1, 7):
            cfg = self._ts_slots[k]
            cfg["_src_combo"] = None
            cfg["_i_sb"] = None
            cfg["_j_sb"] = None

        self._img_fig = None
        self._img_ax = None
        self._img_cax = None
        self._img_canvas_widget = None
        self._img_field_combo = None
        self._img_res_element_rb = None
        self._img_res_gauss_rb = None
        self._img_full_extent = None
        self._img_pz_state = {}

        self._preview3d_canvas = None
        self._notebook = None
        self._scrubber = None
        self._section_index_sb = None
        self._field_combo = None
        self._res_element_rb = None
        self._res_gauss_rb = None
        self._surface_field_combo = None
        self._first_btn = None
        self._prev_btn = None
        self._play_btn = None
        self._stop_btn = None
        self._next_btn = None
        self._last_btn = None
        self._record_btn = None
        self._record_status_var.set("")
        super().close_inspector()

    def on_destroy(self) -> None:
        self._stop_animation()
        if self._recording:
            self._finish_recording(cancelled=True)
        if self._record_writer is not None:
            try:
                self._record_writer.release()
            except Exception:
                pass
            self._record_writer = None
        if self._help_popup is not None and self._help_popup.winfo_exists():
            self._help_popup.destroy()
        if self._section_fig is not None:
            plt.close(self._section_fig)
        if self._field_fig is not None:
            plt.close(self._field_fig)
        if self._ts_fig is not None:
            plt.close(self._ts_fig)
        if self._img_fig is not None:
            plt.close(self._img_fig)
        super().on_destroy()