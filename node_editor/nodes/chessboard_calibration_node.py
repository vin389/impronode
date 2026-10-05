# node_editor/nodes/chessboard_calibration_node.py
"""
Chessboard-based camera calibration node.

STAGE 1 -- detect & select: scans a sequence of images of a planar
chessboard (ncol x nrow inner corners) captured while the board is moved
in front of a static camera, detects the corners with
cv2.findChessboardCorners + cv2.cornerSubPix, and greedily picks 10-30 of
them that are sharp, well-centred, cover a large fraction of the image,
and are each meaningfully DIFFERENT (in corner position) from every
image already picked. Outputs the selected object_points / image_points
arrays plus per-image sharpness and the source index into the input
file list. A "selected" checkbox per image lets the user exclude a few
images by hand (e.g. one whose corners look slightly mis-detected)
without re-running detection or selection.

STAGE 2 -- calibrate & analyse: runs cv2.calibrateCamera on the INCLUDED
(checked) points with user-chosen flags (fix principal point, fix
aspect ratio, zero tangential distortion, fix focal length, rational
model k4-k6, thin prism model s1-s4, tilted sensor model tau_x/tau_y,
individual fix-k1..k6, fix s1-s4, fix tau_x/tau_y, optional initial
intrinsic guess),
keeps every run the user computes in a comparison list, and visualises
per-image reprojection error, the spatial distribution of reprojection
residuals for one selected image at a time, and the recovered camera
poses relative to the board in an interactive 3D view. A leave-one-out
sensitivity analysis shows how much removing any single image would
change the result.

IMPORTANT LIMITATION: this node reads a LIST OF IMAGE FILES (the same
'files' convention as the Image Sequence node's output), not a video
file directly. If the source is a video, extract frames to a folder /
an image-sequence-producing node first.

SOURCE INDEX
    'source #' / source_index is the 0-based index into the node's input
    'files' list.

CORNER / OBJECT POINT ORDERING (identical to the OpenCV convention)
    patternSize passed to cv2.findChessboardCorners is (ncol, nrow) =
    (inner corners per chessboard ROW, inner corners per chessboard
    COLUMN) = (width, height), exactly as OpenCV documents it.
    The corners OpenCV returns are ROW-MAJOR: row by row, left to right
    within a row. Every array in this node keeps that order with NO
    transpose: shape (..., nrow, ncol, 2/3); index [r, c] is the corner in
    row r (0..nrow-1) and column c (0..ncol-1), with object point
    (c*square_size, r*square_size, 0). arr.reshape(-1, 2) therefore gives
    exactly the sequence cv2.findChessboardCorners returned (k = r*ncol + c),
    the same order as the standard
    np.mgrid[0:ncol, 0:nrow].T.reshape(-1, 2) recipe in the OpenCV
    tutorials. (Versions of this node before this change stored
    (ncol, nrow); old .npz files are converted automatically on load.)

    CANONICAL START CORNER: for a board with symmetric corner counts,
    findChessboardCorners may start counting at different board corners
    in different views (reversed, or for a square board rotated 90
    degrees / transposed). Every detection is therefore re-indexed by
    canonicalize_corners() so that [0, 0] is the corner at the TOP-LEFT
    of the image, rows run toward image +x and columns toward image +y.
    Without this, two near-identical views would look "very different"
    point by point and fool the selection below. Keep the board within
    about +-40 degrees of upright in every view.

SUB-PIXEL REFINEMENT
    Detection may run on a downscaled copy (Detection max width), so
    the coarse corners can be several full-resolution pixels off.
    cornerSubPix then refines them on the full-resolution image. With
    window = 0 (auto) the half-window is ~35% of the board's square size
    in that image (clamped to 5..60 px) -- large enough to reach the
    true corner, small enough to stay inside one square. A fixed small
    window (e.g. 11 px with ~130 px squares and a 3x downscale) leaves
    some corners 10-30 px off.

SELECTION ALGORITHM (farthest-point style, quality-weighted)
    Per detected image: sharpness = variance of the Laplacian computed
    on the image ROI bounding the detected corners (not the whole
    image, so an out-of-focus BACKGROUND does not affect the score);
    coverage = convex hull area of the corners / image area; centre_dist
    = normalized distance from the corners' centroid to the image
    centre. quality = 0.5*sharpness_norm + 0.35*coverage + 0.15*(1 -
    centre_dist), each term in [0, 1].

    Distance between two views A and B = mean distance between
    corresponding corners (A[r, c] vs B[r, c], canonical order), divided
    by the image diagonal.

    All usable candidates are RANKED ONCE (after Detect, or after the
    minimum sharpness changes). Rank 0 is the highest-quality candidate.
    Rank i is chosen, among the not-yet-ranked candidates, to MAXIMIZE
    its distance to its NEAREST already-ranked image (the classic
    "farthest point" / max-min criterion),
    multiplied by a quality factor (0.4 + 0.6*quality) so a sharp,
    well-covered candidate is preferred over an equally-"far" but poor
    one. This guarantees each image in the output, in order, is the
    MOST DIFFERENT remaining option from everything picked so far --
    not merely "different enough to pass a one-off threshold" -- which
    is what lets a small target count (10-30) still produce good board-
    pose coverage instead of a cluster of near-duplicate views.
    Selecting N images = taking the first N of the ranking (no
    recomputation), so changing the target count and pressing Reselect
    is instant and the first N images never change when N grows.
    Selection STOPS EARLY (before reaching the target count) at the
    first ranked image whose nearest-image distance at ranking time is
    below the diversity threshold, rather than padding the result with
    near-duplicate images just to hit the requested count.

THREADING
    Detection (Stage 1) and leave-one-out sensitivity analysis (Stage 2)
    run on background threads with progress polling via after(), same
    convention as the other long-running nodes in this project.
    Re-running the SELECTION step alone (after only the selection
    settings changed) is cheap and runs synchronously, since detection
    results are cached.
"""

from __future__ import annotations

import colorsys
import copy
import json
import math
import os
import threading
import time
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from node_editor.base_node import BaseNode
from node_editor.execution import ExecutionMode
from node_editor.pin_types import PinDef, PinSchema, PinType
from node_editor.project_context import get_project_directory
from node_editor.ui_style import apply_notebook_tab_style


PREVIEW_MAX_SIDE = 2000
IMG_MIN_SCALE, IMG_MAX_SCALE, IMG_ZOOM_FACTOR = 0.02, 40.0, 1.15
CHECK_ON, CHECK_OFF = "\u2611", "\u2610"   # checkbox glyphs for the Selected-images table
CORNER_LAYOUT = "row_major_nrow_ncol"      # tag written to .npz meta; absent = legacy (ncol, nrow) layout
# OpenCV distortion-vector order. calibrateCamera returns the first 5 (default),
# 8 (rational model), 12 (+ thin prism) or 14 (+ tilted sensor) of these.
DIST_NAMES = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4", "tau_x", "tau_y")
MANY_IMAGES_WARNING = 30                   # warn before calibrating with more checked images than this
UNDISTORT_CACHE_SIZE = 40                  # undistorted (display-size) images kept in memory

_HELP_TEXT = """\
Chessboard Calibration
=======================
STAGE 1 (Detect & Select): scans a sequence of images of a planar
chessboard, detects corners, and picks 10-30 images that are sharp,
well-centred, cover a large area of the frame, and are each the MOST
DIFFERENT remaining option (farthest-point style, see the module
docstring) from every image already picked. Outputs the selected
object_points / image_points arrays, sharpness, and the source index
into the input file list. Selection stops early if even the best
remaining candidate is too similar to what has already been picked,
rather than padding the result with near-duplicate images.

INPUT: 'files' (list of image paths, e.g. from the Image Sequence
node's 'files' output) -- NOT a video file. Extract frames first if
your source is a video. The Input panel also shows the image size
(w x h) once a sequence is connected. 'source #' everywhere in this
node is the 0-based index into that file list.

Board & Detection tab: chessboard inner-corner counts in OpenCV's
patternSize order -- columns first (= inner corners per row, the
patternSize WIDTH), then rows (= inner corners per column, the
patternSize HEIGHT); square size; cv2.findChessboardCorners flags; a
detection-time downscale width (speeds up detection on large images;
corner sub-pixel refinement still runs on the FULL-resolution image);
and the cornerSubPix window size / convergence criteria. Leave the
cornerSubPix window at 0 (auto, ~35% of the square size): a small
fixed window cannot pull corners found on a downscaled image back to
the true position, leaving some corners many pixels off.

Corner order is made consistent across images: [0, 0] is always the
board corner at the TOP-LEFT of the image (even when OpenCV started
counting elsewhere). Keep the board within about +-40 degrees of
upright.

Array layout (same as OpenCV): object_points (n, nrow, ncol, 3),
image_points (n, nrow, ncol, 2); reshape(-1, 2) of one image gives the
corners in exactly the order cv2.findChessboardCorners returns them
(row by row, left to right).

Selection tab: target image count (10-30 typical), minimum sharpness
(0 = accept anything that was detected), and the diversity threshold
(fraction of the image diagonal; selection stops at the first ranked
image that is closer than this to an image ranked before it). All
candidates are ranked ONCE, so "Reselect" after changing the target
count simply takes the first N ranked images (instant); "Detect &
Select" re-runs detection too (slow).

Selected images table: each row has a checkbox ("selected" column,
checked by default). Click it to exclude that image from the output
pins and from the NEXT calibration run -- useful when a few images have
slightly mis-detected corners. Click anywhere else on a row to preview
that image and show its reprojection residuals (once a run exists).

Calibration tab: the usual cv2.calibrateCamera flags, applied to
whichever images are currently checked. Checking "Fix principal point",
"Fix aspect ratio" or "Fix focal length" requires an initial intrinsic
guess (fx, fy, cx, cy) -- "Guess from image size" fills a reasonable
starting point. "Rational model (k4-k6)" must be on for the individual
k4/k5/k6 fix checkboxes to have any effect; likewise "Thin prism
(s1-s4)" for "Fix s1-s4" and "Tilted sensor (tau_x, tau_y)" for "Fix
tau_x, tau_y". The distortion vector has 5, 8, 12 or 14 coefficients
(k1 k2 p1 p2 k3 | k4 k5 k6 | s1 s2 s3 s4 | tau_x tau_y) depending on
the models enabled; columns a run did not estimate show "-" in the
runs table and the leave-one-out table. More than 30 checked images
asks for confirmation first (calibration can then take a long time,
and the window does not respond while it runs). Every "Run Calibration"
press adds a row to the runs list (and makes it the active run), using
a SNAPSHOT of whichever images were checked at that moment (later
checkbox changes do not retroactively change an existing run); select
another row and press "Set Active" (or double-click it) to switch back.
"Run Sensitivity (leave-one-out)" reruns calibration once per image of
the active run with that one image excluded, using the SAME flags.

Right-hand views (all for the ACTIVE run): image preview with detected
corners (first corner drawn larger, colour runs green -> red in OpenCV
corner order); "All corners": the corners of EVERY selected
image drawn together on the image frame (one colour per image, white =
current image, grey rings = unchecked) to judge whether the points
cover the part of the image you care about -- pause the mouse on a
point to see which image / file / corner it comes from, click it to
select that image; per-image RMS reprojection error (bar chart, x-axis =
source #); the reprojection RESIDUAL VECTOR field for ONE image at a
time (selected via the Selected images table, or the Prev/Next
buttons), at that image's corner pixel positions, exaggerated by the
magnification factor shown; an interactive 3D view of the recovered
camera poses relative to the board ("Fixed: Board" = one board and one
camera per image, as calibrateCamera defines rvecs / tvecs; "Fixed:
Camera" = one camera and one board per image -- a display-only change
of frame, the outputs are unchanged); "Undistort image": the current
image undistorted with the ACTIVE run's K and distortion coefficients
(same result as cv2.undistort). newCameraMatrix = K, or from
cv2.getOptimalNewCameraMatrix (alpha 0 = only valid pixels, 1 = keep
every source pixel; optional centerPrincipalPoint; the valid ROI is
drawn dashed). Overlays: the undistorted corners and straight lines
between the end corners of each row / column -- with a good
calibration every corner lies on those lines. Settings take effect on
"Apply"; switching images while this tab is shown undistorts the new
image automatically (the last 40 results are cached, and the pan/zoom
is kept); and the leave-one-out table
(k1 ... tau_y with their changes d_*).

Mouse controls: the image preview and both plot canvases support
drag-to-pan and wheel-to-zoom (Ctrl/Shift = single-axis zoom on the
plots). The image-preview and residual-vector pan/zoom is KEPT when you
select another image, so the same region can be compared image by
image ("Fit view" / "Reset view" / double-click to reset). The 3D pose
view: left-drag rotate, right-drag pan, wheel zoom.

LOAD / SAVE AS: a .npz file holds the settings, the Stage 1 selection
(object_points, image_points, sharpness, source_index, included mask,
coverage, centre distance, image paths, img_size) and, if present, the
active calibration run. Comparison runs and the leave-one-out table are
NOT saved -- redo those after loading. Files written by older versions
of this node (column-major corner layout) are converted on load.
"""


# ══ Geometry / detection helpers (module-level, no Tk) ═══════════════════

def _read_gray(path: str):
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    except Exception:
        return None


def _read_rgb(path: str):
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        if buf.size == 0:
            return None
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _reshape_corners(corners_flat: np.ndarray, ncol: int, nrow: int) -> np.ndarray:
    """cv2.findChessboardCorners with patternSize=(ncol, nrow) returns the
    corners ROW-MAJOR (nrow rows of ncol points, left to right). Keep that
    order unchanged -- (nrow, ncol, 2) -- so this node follows the OpenCV
    convention exactly (see module docstring)."""
    return corners_flat.reshape(nrow, ncol, 2).copy()


def canonicalize_corners(corners: np.ndarray) -> np.ndarray:
    """Return the (nrow, ncol, 2) corner grid re-indexed to ONE canonical
    order, independent of where cv2.findChessboardCorners happened to
    start counting. A chessboard with symmetric corner counts is
    ambiguous: the same view can come back starting at any board corner
    (180-degree reversal; for a square board also 90-degree rotations or a
    transpose). Among every re-indexing that keeps the grid shape, keep
    the one whose ROW direction (corner [r, 0] -> [r, ncol-1]) points
    closest to image +x and whose COLUMN direction ([0, c] -> [nrow-1, c])
    points to image +y (down) -- i.e. [0, 0] is the top-left corner as
    seen in the image. Consistent order is required for the per-point
    image-to-image distance used by the selection, and for calibration.
    (If the board is held rotated by ~45 degrees or more in some views,
    "top-left" can switch to a different physical corner between those
    views; keep the board within about +-40 degrees of upright.)
    """
    nrow, ncol = corners.shape[:2]
    best, best_cost = corners, None
    for k in range(4):
        rot = np.rot90(corners, k)
        for g in (rot, rot[:, ::-1]):
            if g.shape[:2] != (nrow, ncol):
                continue
            rd = (g[:, -1] - g[:, 0]).mean(axis=0)
            cd = (g[-1] - g[0]).mean(axis=0)
            if rd[0] * cd[1] - rd[1] * cd[0] <= 0:      # column direction must be clockwise of the row direction
                continue
            cost = math.atan2(float(rd[1]), float(rd[0])) ** 2
            if best_cost is None or cost < best_cost:
                best, best_cost = g, cost
    return np.ascontiguousarray(best)


def auto_subpix_window(corners: np.ndarray) -> int:
    """cornerSubPix half-window (px) for a (nrow, ncol, 2) full-resolution
    corner grid: ~35% of the median square size, clamped to [5, 60].
    The window must be large enough to pull a coarse corner (found on a
    downscaled image, so off by several full-res px) back to the true
    corner, but stay well inside one square so it never reaches the
    neighbouring corners.
    """
    steps = np.concatenate([np.linalg.norm(np.diff(corners, axis=1), axis=2).ravel(),
                            np.linalg.norm(np.diff(corners, axis=0), axis=2).ravel()])
    square_px = float(np.median(steps)) if steps.size else 0.0
    return int(min(60, max(5, round(0.35 * square_px))))


def build_object_grid(ncol: int, nrow: int, square_size: float) -> np.ndarray:
    """(nrow, ncol, 3) object points for one view of the board, z = 0;
    [r, c] = (c*square_size, r*square_size, 0). Flattened, this equals the
    OpenCV-tutorial np.mgrid[0:ncol, 0:nrow].T.reshape(-1, 2) order."""
    grid = np.zeros((nrow, ncol, 3), dtype=np.float64)
    grid[:, :, 0] = (np.arange(ncol, dtype=np.float64) * square_size)[None, :]
    grid[:, :, 1] = (np.arange(nrow, dtype=np.float64) * square_size)[:, None]
    return grid


def detect_chessboard(path: str, ncol: int, nrow: int, detect_flags: int,
                      max_width: int, subpix_win: int, subpix_criteria) -> dict:
    """Detect + sub-pixel refine the board in one image. Never touches Tk;
    safe to call from a worker thread. Returns a dict; 'found' is False
    if no board was detected (or the file could not be read).
    """
    gray = _read_gray(path)
    if gray is None:
        return {"path": path, "found": False, "error": "cannot read image"}
    h, w = gray.shape[:2]

    detect_img, scale = gray, 1.0
    if max_width > 0 and w > max_width:
        scale = max_width / float(w)
        detect_img = cv2.resize(gray, (max_width, max(1, int(round(h * scale)))),
                                interpolation=cv2.INTER_AREA)

    try:
        # patternSize = (points per row, points per column) = (ncol, nrow), per OpenCV docs.
        found, corners = cv2.findChessboardCorners(detect_img, (ncol, nrow), flags=detect_flags)
    except Exception as e:
        return {"path": path, "found": False, "error": f"{type(e).__name__}: {e}"}
    if not found:
        return {"path": path, "found": False, "size": (w, h)}

    if scale != 1.0:
        corners = corners / scale
    # subpix_win <= 0 = automatic, scaled to the board's square size in this image.
    win = subpix_win if subpix_win > 0 else auto_subpix_window(
        _reshape_corners(corners.reshape(-1, 2), ncol, nrow))
    try:
        corners = cv2.cornerSubPix(gray, corners.astype(np.float32),
                                   (win, win), (-1, -1), subpix_criteria)
    except Exception as e:
        return {"path": path, "found": False, "error": f"subpix failed: {e}", "size": (w, h)}

    corners2 = canonicalize_corners(_reshape_corners(corners.reshape(-1, 2), ncol, nrow))   # (nrow, ncol, 2)
    xs, ys = corners2[..., 0], corners2[..., 1]
    x0 = max(0, int(math.floor(xs.min())) - 15)
    x1 = min(w, int(math.ceil(xs.max())) + 15)
    y0 = max(0, int(math.floor(ys.min())) - 15)
    y1 = min(h, int(math.ceil(ys.max())) + 15)
    roi = gray[y0:y1, x0:x1]
    sharpness = float(cv2.Laplacian(roi, cv2.CV_64F).var()) if roi.size else 0.0

    hull = cv2.convexHull(corners2.reshape(-1, 1, 2).astype(np.float32))
    coverage = float(cv2.contourArea(hull)) / float(w * h)
    centroid = corners2.reshape(-1, 2).mean(axis=0)
    center_dist = float(np.linalg.norm(centroid - np.array([w / 2.0, h / 2.0]))) / (0.5 * math.hypot(w, h))

    return {"path": path, "found": True, "corners": corners2, "sharpness": sharpness,
            "coverage": coverage, "center_dist": center_dist, "size": (w, h)}


def image_distance(pts_a: np.ndarray, pts_b: np.ndarray, diag: float) -> float:
    """Difference between two views of the board: mean distance between
    CORRESPONDING corners (same [r, c] after canonicalize_corners), as a
    fraction of the image diagonal. pts_* are (..., 2) arrays."""
    return float(np.linalg.norm(pts_a.reshape(-1, 2) - pts_b.reshape(-1, 2), axis=1).mean()) / diag


def rank_calibration_images(candidates: list, min_sharpness: float) -> list:
    """Order ALL usable candidates once, farthest-point style -- see the
    module docstring's "SELECTION ALGORITHM". `candidates` is a list of
    detect_chessboard() result dicts. Returns them IN RANK ORDER; rank 0
    is the highest-quality image, and rank i is the remaining candidate
    whose distance to its NEAREST already-ranked image (ranks 0..i-1),
    times a quality factor, is largest. Each returned dict gets
    'quality' and 'pick_dist' (that nearest-image distance at the time it
    was ranked, x image diagonal; inf for rank 0). Selecting N images is
    then just taking the first N -- see take_ranked_images().
    """
    valid = [c for c in candidates if c.get("found") and c["sharpness"] >= min_sharpness]
    if not valid:
        return []
    max_sharp = max(c["sharpness"] for c in valid) or 1.0
    for c in valid:
        c["quality"] = (0.5 * (c["sharpness"] / max_sharp) + 0.35 * min(c["coverage"], 1.0)
                        + 0.15 * (1.0 - min(c["center_dist"], 1.0)))

    w, h = valid[0]["size"]
    diag = math.hypot(w, h)
    pts = np.stack([c["corners"].reshape(-1, 2) for c in valid]).astype(np.float64)   # (n, k, 2)
    quality_factor = np.array([0.4 + 0.6 * c["quality"] for c in valid])

    first = int(np.argmax([c["quality"] for c in valid]))
    order, pick_dist = [first], [math.inf]
    # nearest[j] = distance from candidate j to its NEAREST ranked image;
    # updated incrementally with each newly ranked image (O(n) per step).
    nearest = np.linalg.norm(pts - pts[first], axis=2).mean(axis=1) / diag
    taken = np.zeros(len(valid), dtype=bool)
    taken[first] = True
    while not taken.all():
        score = np.where(taken, -np.inf, nearest * quality_factor)
        i = int(np.argmax(score))
        order.append(i)
        pick_dist.append(float(nearest[i]))
        taken[i] = True
        nearest = np.minimum(nearest, np.linalg.norm(pts - pts[i], axis=2).mean(axis=1) / diag)

    ranked = []
    for i, d in zip(order, pick_dist):
        valid[i]["pick_dist"] = d
        ranked.append(valid[i])
    return ranked


def take_ranked_images(ranked: list, target_count: int, diversity_threshold: float) -> list:
    """The first `target_count` images of a rank_calibration_images()
    result, stopping early at the first image whose pick_dist is below
    `diversity_threshold` (everything after it is even more similar to
    what has already been picked). No recomputation."""
    out = []
    for c in ranked[:max(0, int(target_count))]:
        if c["pick_dist"] < diversity_threshold:
            break
        out.append(c)
    return out


# (opts key, cv2 flag, short label for flags_summary). opts.get() is used
# throughout so runs loaded from older files (missing newer keys) still work.
_CALIB_FLAG_TABLE = (
    ("use_intrinsic_guess", cv2.CALIB_USE_INTRINSIC_GUESS, "IntrinsicGuess"),
    ("fix_principal_point", cv2.CALIB_FIX_PRINCIPAL_POINT, "FixPP"),
    ("fix_aspect_ratio", cv2.CALIB_FIX_ASPECT_RATIO, "FixAspect"),
    ("fix_focal_length", cv2.CALIB_FIX_FOCAL_LENGTH, "FixFocal"),
    ("zero_tangent_dist", cv2.CALIB_ZERO_TANGENT_DIST, "ZeroTangent"),
    ("rational_model", cv2.CALIB_RATIONAL_MODEL, "Rational"),
    ("thin_prism_model", cv2.CALIB_THIN_PRISM_MODEL, "ThinPrism"),
    ("fix_s1_s4", cv2.CALIB_FIX_S1_S2_S3_S4, "FixS1-S4"),
    ("tilted_model", cv2.CALIB_TILTED_MODEL, "Tilted"),
    ("fix_taux_tauy", cv2.CALIB_FIX_TAUX_TAUY, "FixTau"),
)


def build_calibration_flags(opts: dict) -> int:
    flags = 0
    for key, bit, _label in _CALIB_FLAG_TABLE:
        if opts.get(key):
            flags |= bit
    for k in range(1, 7):
        if opts.get(f"fix_k{k}"):
            flags |= getattr(cv2, f"CALIB_FIX_K{k}")
    return flags


def flags_summary(opts: dict) -> str:
    parts = [label for key, _bit, label in _CALIB_FLAG_TABLE if opts.get(key)]
    fixed_k = [str(k) for k in range(1, 7) if opts.get(f"fix_k{k}")]
    if fixed_k:
        parts.append("FixK" + ",".join(fixed_k))
    return ", ".join(parts) if parts else "(default)"


def dist_value(dvec: np.ndarray, i: int) -> float:
    """Coefficient i (DIST_NAMES order) of a distortion vector, NaN when the
    model used did not estimate it (e.g. s1 without the thin-prism model)."""
    return float(dvec[i]) if i < dvec.size else float("nan")


def run_calibration(object_points: np.ndarray, image_points: np.ndarray,
                    img_size: tuple, opts: dict) -> dict:
    """object_points / image_points: (nimg, nrow, ncol, 3/2), OpenCV row-major
    corner order. Raises on failure (caller shows the exception to the user)."""
    nimg = object_points.shape[0]
    obj_list = [object_points[i].reshape(-1, 3).astype(np.float32) for i in range(nimg)]
    img_list = [image_points[i].reshape(-1, 2).astype(np.float32) for i in range(nimg)]
    flags = build_calibration_flags(opts)

    cmat_guess = None
    if opts["use_intrinsic_guess"]:
        cmat_guess = np.array([[opts["guess_fx"], 0.0, opts["guess_cx"]],
                               [0.0, opts["guess_fy"], opts["guess_cy"]],
                               [0.0, 0.0, 1.0]], dtype=np.float64)

    rms, cmat, dvec, rvecs, tvecs = cv2.calibrateCamera(
        obj_list, img_list, img_size, cmat_guess, None, flags=flags)

    npts = obj_list[0].shape[0]
    per_image_error = np.zeros(nimg)
    residuals = np.zeros((nimg, npts, 2))     # per corner, OpenCV order k = r*ncol + c
    for i in range(nimg):
        proj, _ = cv2.projectPoints(obj_list[i], rvecs[i], tvecs[i], cmat, dvec)
        diff = proj.reshape(-1, 2) - img_list[i]
        residuals[i] = diff
        per_image_error[i] = float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))

    return {"rms": float(rms), "cmat": cmat, "dvec": dvec.ravel(),
            "rvecs": np.array([r.ravel() for r in rvecs]), "tvecs": np.array([t.ravel() for t in tvecs]),
            "per_image_error": per_image_error, "residuals": residuals,
            "flags": flags, "opts": copy.deepcopy(opts), "img_size": img_size,
            "label": time.strftime("%H:%M:%S") + "  " + flags_summary(opts)}


def _legacy_grid_to_row_major(arr: np.ndarray, ncol: int, nrow: int) -> np.ndarray:
    """Legacy (n, ncol, nrow, d) -> OpenCV row-major (n, nrow, ncol, d)."""
    arr = np.asarray(arr)
    return np.ascontiguousarray(arr.reshape(arr.shape[0], ncol, nrow, -1).transpose(0, 2, 1, 3))


def _legacy_flat_to_row_major(arr: np.ndarray, ncol: int, nrow: int) -> np.ndarray:
    """Legacy flattened per-corner array (n, ncol*nrow, d) in column-major corner
    order -> OpenCV row-major corner order (k = r*ncol + c)."""
    arr = np.asarray(arr)
    n = arr.shape[0]
    return np.ascontiguousarray(
        arr.reshape(n, ncol, nrow, -1).transpose(0, 2, 1, 3).reshape(n, ncol * nrow, -1))


def _run_loo_worker(job: dict, state: dict) -> None:
    """Background thread: leave-one-out sensitivity analysis on the SAME
    point set used by the active run. Never touches Tk.
    """
    try:
        object_points, image_points = job["object_points"], job["image_points"]
        img_size, opts = job["img_size"], job["opts"]
        nimg = object_points.shape[0]
        rows = []
        for i in range(nimg):
            if state["cancel"].is_set():
                break
            mask = np.ones(nimg, dtype=bool)
            mask[i] = False
            try:
                r = run_calibration(object_points[mask], image_points[mask], img_size, opts)
                row = {"excluded_index": i, "rms": r["rms"],
                       "fx": float(r["cmat"][0, 0]), "fy": float(r["cmat"][1, 1]),
                       "cx": float(r["cmat"][0, 2]), "cy": float(r["cmat"][1, 2])}
                row.update({name: dist_value(r["dvec"], j) for j, name in enumerate(DIST_NAMES)})
                rows.append(row)
            except Exception as e:
                rows.append({"excluded_index": i, "error": str(e)})
            state["done"] = i + 1
        state["rows"] = rows
    except Exception as e:
        state["error"] = f"{type(e).__name__}: {e}"
    finally:
        state["finished"] = True


def _detect_worker(job: dict, state: dict) -> None:
    """Background thread: detect the board in every requested frame using a
    thread pool. Each result dict gets 'source_index' (0-based index into
    the node's input file list). Never touches Tk."""
    try:
        cfg, paths, indices = job["cfg"], job["paths"], job["indices"]
        results: list = [None] * len(paths)

        def task(k: int):
            if state["cancel"].is_set():
                return k, None
            r = detect_chessboard(paths[k], cfg["ncol"], cfg["nrow"], cfg["detect_flags"],
                                  cfg["max_width"], cfg["subpix_win"], cfg["subpix_criteria"])
            r["source_index"] = int(indices[k])
            return k, r

        with ThreadPoolExecutor(max_workers=max(1, int(cfg["threads"]))) as ex:
            futures = [ex.submit(task, k) for k in range(len(paths))]
            for fut in as_completed(futures):
                k, r = fut.result()
                results[k] = r
                state["done"] += 1
                if r is not None and r.get("found"):
                    state["found"] += 1
        state["results"] = results
    except Exception as e:
        state["error"] = f"{type(e).__name__}: {e}"
    finally:
        state["finished"] = True


# ══ Node ══════════════════════════════════════════════════════════════════

class ChessboardCalibrationNode(BaseNode):
    """Select good calibration images from a chessboard sequence, then
    calibrate the camera and analyse the result. See the module
    docstring and _HELP_TEXT for the full design and algorithms."""

    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE = "chessboard_calibration"
    DISPLAY_NAME = "Chessboard Calibration"
    CATEGORY = "process"
    SEARCH_KEYWORDS = ("chessboard", "calibration", "calibrateCamera", "corners",
                       "intrinsics", "distortion", "camera matrix")
    NODE_WIDTH = 240
    NODE_HEIGHT = 110
    HELP_TEXT = _HELP_TEXT

    DETECT_FLAG_BITS = {
        "adaptive_thresh": cv2.CALIB_CB_ADAPTIVE_THRESH,
        "normalize_image": cv2.CALIB_CB_NORMALIZE_IMAGE,
        "fast_check": cv2.CALIB_CB_FAST_CHECK,
    }

    def __init__(self, node_id: str, canvas: tk.Canvas):
        super().__init__(node_id, canvas)
        self._files: list | None = None
        self._candidates: list = []          # every found detect_chessboard() result from the last run
        self._ranking: list | None = None    # self._candidates in selection-rank order (rank_calibration_images)
        self._ranking_min_sharpness = None   # min_sharpness the cached ranking was built with
        self._detect_state: dict | None = None
        self._detect_poll_id = None
        self._detect_cfg: dict | None = None  # detection settings snapshot used for self._candidates
        self._detect_text = "Detection not run yet."
        self._selection_text = ""

        # Stage 1 result (the "applied" selection that feeds the output pins).
        self._stage1: dict | None = None      # {object_points, image_points, sharpness, source_index,
                                               #  img_size, selected, included, ncol, nrow, square_size}
                                               # point arrays are (n, nrow, ncol, 3/2), OpenCV row-major
        self._stage1_path = ""
        self._stage1_saved = True

        # Stage 2: every calibration the user has run, plus which one is active.
        self._runs: list = []
        self._active_run_index: int | None = None
        self._loo_state: dict | None = None
        self._loo_poll_id = None
        self._loo_rows: list | None = None
        self._loo_run_ref = None

        self._outputs_dirty = True
        self._destroyed = False
        self._status_item = None
        self._preview_selected_pos = 0        # position within self._stage1["selected"] -- drives the
                                               # preview, the residual plot and all highlights

        self._preview_view = None              # (ox, oy, s): canvas = o + preview_px * s
        self._preview_user_view = False
        self._preview_last_size = None
        self._preview_corner_scale = 1.0      # preview downscale factor; full-res corners are multiplied
                                               # by it before drawing (fixes the corner-offset bug)
        self._preview_pan = {"active": False, "x_px": None, "y_px": None, "ox0": None, "oy0": None}
        self._preview_rgb = None
        self._preview_rgb_path = None
        self._preview_photo = None
        self._preview_photo_key = None

        self._v3 = {"yaw": -0.9, "pitch": 0.7, "zoom": 1.0, "target": [0.0, 0.0, 0.0],
                    "dragging": False, "button": None, "last_x": 0, "last_y": 0, "last_scale": 1.0}
        self._v3_centered = False
        self._v3_extent = 1.0
        self._res_saved_view = None           # (xlim, ylim, img_size) of the residual plot, kept across images

        # "All corners" coverage view: (ox, oy, s) with canvas = o + full_res_px * s.
        self._cov_view = None
        self._cov_user_view = False
        self._cov_pan = {"active": False, "moved": False, "x_px": None, "y_px": None, "ox0": None, "oy0": None}
        self._cov_hit = None                  # (canvas xy (N, 2), image pos (N,), corner r (N,), corner c (N,))
        self._cov_hover_id = None

        # "Undistort image" tab. _und_cfg = the APPLIED settings (widgets take
        # effect only on Apply); the view (ox, oy, s) is kept across images.
        self._und_cfg = self._default_undistort()
        self._und_view = None
        self._und_user_view = False
        self._und_pan = {"active": False, "x_px": None, "y_px": None, "ox0": None, "oy0": None}
        self._und_maps: dict = {}             # remap tables per (K, dist, size, newK settings)
        self._und_cache: dict = {}            # (path, maps key) -> (display rgb, display scale), LRU order
        self._und_photo = None
        self._und_photo_key = None

        self._reset_widget_refs()

    def _reset_widget_refs(self) -> None:
        self._entries: dict = {}
        self._file_label_var = None
        self._detect_btn = self._reselect_btn = self._stop_detect_btn = None
        self._detect_progress = self._detect_progress_var = None
        self._detect_text_var = None
        self._selection_summary_var = None
        self._selected_tree = None
        self._c_preview = None
        self._preview_label_var = None
        self._c_cov = None
        self._cov_label_var = None
        self._right_nb = None
        self._und_tab = None
        self._c_und = None
        self._und_label_var = None
        self._und_vars = None
        self._und_photo, self._und_photo_key = None, None
        self._calib_btn = self._loo_btn = self._stop_loo_btn = None
        self._runs_tree = None
        self._set_active_btn = None
        self._active_summary_var = None
        self._loo_progress = self._loo_progress_var = None
        self._loo_tree = None
        self._fig_err = self._ax_err = self._canvas_err = None
        self._fig_res = self._ax_res = self._canvas_res = None
        self._residual_label_var = None
        self._pan_err = {"active": False, "x_px": None, "y_px": None, "xlim0": None, "ylim0": None,
                         "x_scale": None, "y_scale": None}
        self._pan_res = dict(self._pan_err)
        self._res_plot_size = None          # img_size of the residual plot currently drawn (None = message)
        self._c3d = None
        self._persp_var = None
        self._help_popup = None
        self._preview_photo, self._preview_photo_key = None, None

    # ── defaults ────────────────────────────────────────────────────────
    @staticmethod
    def _default_detection() -> dict:
        return {"ncol": "9", "nrow": "6", "square_size": "25.0",
                "adaptive_thresh": True, "normalize_image": True, "fast_check": True,
                "max_width": "1200", "subpix_win": "0", "subpix_iters": "30", "subpix_eps": "0.001",
                "threads": "4", "start": "1", "end": "", "step": "1"}

    @staticmethod
    def _default_undistort() -> dict:
        return {"newk_mode": "optimal", "alpha": 0.0, "center_pp": False,
                "show_corners": True, "show_lines": False, "show_roi": True}

    @staticmethod
    def _default_selection() -> dict:
        return {"target_count": "25", "min_sharpness": "0.0", "diversity_threshold": "0.06"}

    @staticmethod
    def _default_calib_opts() -> dict:
        opts = {"fix_principal_point": False, "fix_aspect_ratio": False, "fix_focal_length": False,
                "zero_tangent_dist": False, "rational_model": False, "use_intrinsic_guess": False,
                "thin_prism_model": False, "fix_s1_s4": False, "tilted_model": False, "fix_taux_tauy": False,
                "guess_fx": "1000.0", "guess_fy": "1000.0", "guess_cx": "0.0", "guess_cy": "0.0"}
        for k in range(1, 7):
            opts[f"fix_k{k}"] = False
        return opts

    def _init_state(self) -> None:
        if hasattr(self, "_det_vars"):
            return
        self._det_vars = {k: (tk.BooleanVar(value=v) if isinstance(v, bool) else tk.StringVar(value=v))
                          for k, v in self._default_detection().items()}
        self._sel_vars = {k: tk.StringVar(value=v) for k, v in self._default_selection().items()}
        d = self._default_calib_opts()
        self._calib_vars = {k: (tk.BooleanVar(value=v) if isinstance(v, bool) else tk.StringVar(value=v))
                            for k, v in d.items()}
        self._files_info_var = tk.StringVar(value="No image sequence connected.")
        self._resel_mag_var = tk.DoubleVar(value=50.0)   # residual-plot magnification factor
        self._frustum_scale_var = tk.DoubleVar(value=1.0)
        self._pose_frame_var = tk.StringVar(value="board")   # 3D view only: "board" or "camera" fixed

    # ── pins ────────────────────────────────────────────────────────────
    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[PinDef("files", PinType.STRING, "files", optional=True)],
            outputs=[
                PinDef("object_points", PinType.ARRAY, "objPts"),   # (n, nrow, ncol, 3)
                PinDef("image_points", PinType.ARRAY, "imgPts"),    # (n, nrow, ncol, 2)
                PinDef("sharpness", PinType.ARRAY, "sharp"),
                PinDef("source_index", PinType.ARRAY, "srcIdx"),
                PinDef("img_size", PinType.ARRAY, "imgSize"),
                PinDef("cmat", PinType.ARRAY, "cmat"),
                PinDef("dvec", PinType.ARRAY, "dvec"),
                PinDef("rvecs", PinType.ARRAY, "rvecs"),
                PinDef("tvecs", PinType.ARRAY, "tvecs"),
                PinDef("per_image_error", PinType.ARRAY, "perImgErr"),
                PinDef("rms_error", PinType.SCALAR, "rms"),
            ],
        )

    @staticmethod
    def _coerce_files(value):
        if value is None:
            return None
        if isinstance(value, (str, os.PathLike)):
            items = [str(value)]
        else:
            try:
                items = [str(v) for v in list(value)]
            except TypeError:
                return None
        items = [p for p in items if p.strip()]
        return items or None

    def compute(self, inputs: dict) -> dict:
        self._init_state()
        files = self._coerce_files(inputs.get("files"))
        if files != self._files:
            self._files = files
            self._update_files_info()
        if not self._outputs_dirty:
            return {"_skip_downstream": True, "_preserve_cache": True}
        self._outputs_dirty = False
        return self._outputs()

    def _update_files_info(self) -> None:
        """Input-panel summary, including the image size (w x h) -- read
        once from the first file whenever the file list actually changes
        (not on every compute() call).
        """
        n = len(self._files) if self._files else 0
        if not n:
            self._files_info_var.set("No image sequence connected.")
            return
        size_text = ""
        img = _read_gray(self._files[0])
        if img is not None:
            h, w = img.shape[:2]
            size_text = f", image size {w} x {h} px"
        self._files_info_var.set(f"{n} image files{size_text}")

    def _active_run(self) -> dict | None:
        if self._active_run_index is None or not (0 <= self._active_run_index < len(self._runs)):
            return None
        return self._runs[self._active_run_index]

    def _included_stage1_arrays(self) -> dict | None:
        """Stage 1 arrays filtered down to only the CHECKED ("included")
        images -- this is what actually feeds the output pins and the
        NEXT calibration run. Returns None if there is no selection yet.
        """
        s1 = self._stage1
        if s1 is None:
            return None
        mask = s1.get("included")
        if mask is None:
            mask = np.ones(s1["object_points"].shape[0], dtype=bool)
        return {"object_points": s1["object_points"][mask], "image_points": s1["image_points"][mask],
                "sharpness": s1["sharpness"][mask], "source_index": s1["source_index"][mask],
                "img_size": s1["img_size"]}

    def _outputs(self) -> dict:
        out: dict = {}
        filtered = self._included_stage1_arrays()
        if filtered is not None:
            out.update(object_points=filtered["object_points"], image_points=filtered["image_points"],
                       sharpness=filtered["sharpness"], source_index=filtered["source_index"],
                       img_size=np.array(filtered["img_size"], dtype=np.float64))
        run = self._active_run()
        if run is not None:
            out.update(cmat=run["cmat"], dvec=run["dvec"], rvecs=run["rvecs"], tvecs=run["tvecs"],
                       per_image_error=run["per_image_error"], rms_error=run["rms"])
        return out

    def _notify_outputs_changed(self) -> None:
        self._outputs_dirty = True
        if self._request_downstream is not None:
            self._request_downstream(self.node_id)

    # ── serialization ─────────────────────────────────────────────────
    def _settings_dict(self) -> dict:
        self._init_state()
        det = {k: v.get() for k, v in self._det_vars.items()}
        sel = {k: v.get() for k, v in self._sel_vars.items()}
        calib = {k: v.get() for k, v in self._calib_vars.items()}
        return {"detection": det, "selection": sel, "calibration": calib}

    def _apply_settings(self, s: dict) -> None:
        self._init_state()
        for group_attr, key, defaults in (
            (self._det_vars, "detection", self._default_detection()),
            (self._sel_vars, "selection", self._default_selection()),
            (self._calib_vars, "calibration", self._default_calib_opts()),
        ):
            saved = s.get(key, {}) or {}
            for name, default in defaults.items():
                value = saved.get(name, default)
                if isinstance(default, bool):
                    group_attr[name].set(bool(value))
                else:
                    group_attr[name].set(str(value))

    def get_params(self) -> dict:
        base = self._get_project_base()
        path = self._stage1_path
        rel = self._to_relative(self._to_absolute(path, base), base) if path else ""
        return {"settings": self._settings_dict(), "result_file": rel}

    def set_params(self, params: dict) -> None:
        self._init_state()
        self._apply_settings(params.get("settings", {}) or {})
        path = str(params.get("result_file", "") or "")
        if path:
            abs_path = self._to_absolute(path, self._get_project_base())
            try:
                self._load_result_file(abs_path)
                self._stage1_path, self._stage1_saved = abs_path, True
                self._outputs_dirty = True
            except Exception:
                pass  # a missing/corrupt result file must not break project loading

    @staticmethod
    def _get_project_base():
        return get_project_directory()

    @staticmethod
    def _to_relative(path: str, base) -> str:
        if not path or base is None:
            return path
        try:
            return os.path.relpath(str(path), str(base))
        except Exception:
            return path

    @staticmethod
    def _to_absolute(path: str, base) -> str:
        p = Path(path)
        if p.is_absolute() or base is None:
            return str(p)
        return str((Path(base) / p).resolve())

    # ── node face ─────────────────────────────────────────────────────
    def _status_text(self) -> str:
        s1 = self._stage1
        run = self._active_run()
        if s1 is None:
            return "no selection yet"
        n_included = int(s1.get("included", np.ones(s1["object_points"].shape[0], dtype=bool)).sum())
        text = f"{n_included}/{s1['object_points'].shape[0]} images included"
        if run is not None:
            text += f", RMS {run['rms']:.4g} px"
        return text

    def _update_status_item(self) -> None:
        if self._status_item is not None:
            try:
                self.canvas.itemconfig(self._status_item, text=self._status_text())
            except tk.TclError:
                pass

    def build_body(self) -> None:
        self._init_state()
        x, y, w, h = self.x, self.y, self.width, self.height
        self._body_rect = self.canvas.create_rectangle(
            x, y, x + w, y + h, fill="#f3eefb", outline="#6a3f9e", width=2,
            tags=(self.node_id, "node_body"))
        self._title_item = self.canvas.create_text(
            x + w / 2, y + 13, text=self.get_canvas_title(), font=("Arial", 9, "bold"),
            fill="#3a1f5e", tags=(self.node_id,))
        hint = self.canvas.create_text(
            x + w / 2, y + 42, text="files -> object/image points + calibration", font=("Arial", 8),
            fill="#5a3a7d", tags=(self.node_id,))
        self._status_item = self.canvas.create_text(
            x + w / 2, y + h - 14, text=self._status_text(), font=("Arial", 7),
            fill="#5a3a7d", width=w - 12, tags=(self.node_id,))
        self._canvas_items += [self._body_rect, self._title_item, hint, self._status_item]

    def on_resize(self, old_width, old_height, new_width, new_height) -> None:
        if self._status_item is not None:
            self.canvas.coords(self._status_item, self.x + new_width / 2, self.y + new_height - 14)

    def get_help_text(self) -> str:
        return _HELP_TEXT

    # ── validation helpers ────────────────────────────────────────────
    def _parse_int(self, var, minimum=None):
        try:
            v = int(var.get())
        except (ValueError, tk.TclError):
            return None
        if minimum is not None and v < minimum:
            return None
        return v

    def _parse_float(self, var, minimum=None):
        try:
            v = float(var.get())
        except (ValueError, tk.TclError):
            return None
        if not math.isfinite(v) or (minimum is not None and v < minimum):
            return None
        return v

    def _collect_detection_cfg(self) -> tuple[dict, list]:
        errs = []
        ncol = self._parse_int(self._det_vars["ncol"], 3)
        nrow = self._parse_int(self._det_vars["nrow"], 3)
        square = self._parse_float(self._det_vars["square_size"], 1e-9)
        max_width = self._parse_int(self._det_vars["max_width"], 0)
        subpix_win = self._parse_int(self._det_vars["subpix_win"], 0)    # 0 = auto (scaled to square size)
        subpix_iters = self._parse_int(self._det_vars["subpix_iters"], 1)
        subpix_eps = self._parse_float(self._det_vars["subpix_eps"], 1e-9)
        threads = self._parse_int(self._det_vars["threads"], 1)
        start = self._parse_int(self._det_vars["start"], 1)
        end_text = self._det_vars["end"].get().strip()
        end = self._parse_int(self._det_vars["end"], 1) if end_text else None
        step = self._parse_int(self._det_vars["step"], 1)
        for name, val in (("columns (inner corners per row)", ncol), ("rows (inner corners per column)", nrow),
                          ("square size", square), ("subpixel window", subpix_win),
                          ("subpixel iterations", subpix_iters), ("subpixel epsilon", subpix_eps),
                          ("threads", threads), ("start frame", start), ("frame step", step)):
            if val is None:
                errs.append(f"Invalid {name}.")
        if end_text and end is None:
            errs.append("Invalid last frame (leave blank for the last file).")
        if subpix_win is not None and 0 < subpix_win < 3:
            errs.append("cornerSubPix window must be 0 (auto) or >= 3 px.")
        if max_width is None:
            errs.append("Invalid detection max width (0 disables downscaling).")
        elif 0 < max_width < 320:
            errs.append(f"Detection max width {max_width} px is too small -- the board cannot be "
                        "detected at that resolution. Use >= 320 (e.g. 1200), or 0 for full size.")
        detect_flags = 0
        for key, bit in self.DETECT_FLAG_BITS.items():
            if self._det_vars[key].get():
                detect_flags |= bit
        cfg = {"ncol": ncol, "nrow": nrow, "square_size": square, "detect_flags": detect_flags,
               "max_width": max_width, "subpix_win": subpix_win,
               "subpix_criteria": (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, subpix_iters, subpix_eps)
                                  if subpix_iters is not None and subpix_eps is not None else None,
               "threads": threads, "start": start, "end": end, "step": step}
        return cfg, errs

    def _collect_selection_cfg(self) -> tuple[dict, list]:
        errs = []
        target = self._parse_int(self._sel_vars["target_count"], 1)
        min_sharp = self._parse_float(self._sel_vars["min_sharpness"], 0.0)
        diversity = self._parse_float(self._sel_vars["diversity_threshold"], 0.0)
        if target is None:
            errs.append("Invalid target image count.")
        if min_sharp is None:
            errs.append("Invalid minimum sharpness.")
        if diversity is None:
            errs.append("Invalid diversity threshold.")
        return {"target_count": target, "min_sharpness": min_sharp, "diversity_threshold": diversity}, errs

    def _collect_calib_opts(self) -> tuple[dict, list]:
        """Read the Calibration-tab widgets into a plain opts dict for
        run_calibration(). The intrinsic-guess fields are only validated
        when 'Use intrinsic guess' is on (OpenCV ignores them otherwise).
        """
        errs = []
        opts = {k: bool(v.get()) for k, v in self._calib_vars.items() if isinstance(v, tk.BooleanVar)}
        use_guess = opts["use_intrinsic_guess"]
        for key in ("guess_fx", "guess_fy", "guess_cx", "guess_cy"):
            val = self._parse_float(self._calib_vars[key])
            if val is None and use_guess:
                errs.append(f"Invalid {key.replace('guess_', 'guess ')}.")
            opts[key] = val if val is not None else 0.0
        if use_guess and (opts["guess_fx"] <= 0.0 or opts["guess_fy"] <= 0.0):
            errs.append("Initial guess fx and fy must be positive.")
        needs_guess = [label for key, label in (("fix_principal_point", "Fix principal point"),
                                                ("fix_aspect_ratio", "Fix aspect ratio"),
                                                ("fix_focal_length", "Fix focal length")) if opts[key]]
        if needs_guess and not use_guess:
            errs.append(", ".join(needs_guess) + " requires 'Use intrinsic guess' "
                        "(press 'Guess from image size' for a reasonable starting point).")
        return opts, errs

    # ══ Small Tk helpers ═════════════════════════════════════════════════
    def _dlg_parent(self):
        """Parent for message boxes: the inspector if open, else the main window."""
        win = getattr(self, "_inspector_win", None)
        try:
            if win is not None and win.winfo_exists():
                return win
        except tk.TclError:
            pass
        return self.canvas.winfo_toplevel()

    @staticmethod
    def _set_state(widget, state) -> None:
        if widget is not None:
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass

    @staticmethod
    def _wheel_steps(event) -> int:
        """+1 / -1 per wheel notch on Windows/macOS (<MouseWheel>) and X11 (<Button-4/5>)."""
        num = getattr(event, "num", None)
        if num == 4:
            return 1
        if num == 5:
            return -1
        delta = getattr(event, "delta", 0) or 0
        if delta == 0:
            return 0
        return 1 if delta > 0 else -1

    def _make_scrollable(self, parent: tk.Widget):
        """Vertically scrollable frame for the left (settings) pane."""
        outer = tk.Frame(parent)
        outer.pack(fill="both", expand=True)
        cv = tk.Canvas(outer, highlightthickness=0, borderwidth=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=cv.yview)
        cv.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        cv.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(cv)
        win_id = cv.create_window(0, 0, window=inner, anchor="nw")
        inner.bind("<Configure>", lambda _e: cv.configure(scrollregion=cv.bbox("all")))
        cv.bind("<Configure>", lambda e: cv.itemconfigure(win_id, width=e.width))
        return inner, cv

    def _bind_scroll_recursive(self, root: tk.Widget, cv: tk.Canvas) -> None:
        """Bind the mouse wheel on every descendant of `root` to scroll `cv`,
        EXCEPT widgets that use the wheel themselves (tables, spinboxes).
        Done per-widget rather than with bind_all so it cannot interfere
        with the image/plot canvases in the right-hand pane."""
        def on_wheel(event):
            steps = self._wheel_steps(event)
            if steps:
                cv.yview_scroll(-2 * steps, "units")
            return "break"

        skip = (ttk.Treeview, ttk.Combobox, tk.Spinbox, ttk.Spinbox)
        stack = [root, cv]
        while stack:
            w = stack.pop()
            if not isinstance(w, skip):
                for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                    w.bind(seq, on_wheel, add="+")
            stack.extend(w.winfo_children())

    def _field(self, parent, row: int, label: str, var, width: int = 10) -> tk.Entry:
        tk.Label(parent, text=label, font=("Arial", 8), anchor="w").grid(
            row=row, column=0, sticky="w", padx=2, pady=1)
        entry = tk.Entry(parent, textvariable=var, width=width, font=("Arial", 8))
        entry.grid(row=row, column=1, sticky="w", padx=2, pady=1)
        return entry

    def _file_label_text(self) -> str:
        name = os.path.basename(self._stage1_path) if self._stage1_path else "(not saved)"
        dirty = "" if self._stage1_saved else "  -- unsaved changes"
        return f"Result file: {name}{dirty}"

    def _refresh_file_label(self) -> None:
        if self._file_label_var is not None:
            try:
                self._file_label_var.set(self._file_label_text())
            except tk.TclError:
                pass

    def _mark_unsaved(self) -> None:
        self._stage1_saved = False
        self._refresh_file_label()

    # ══ Inspector ════════════════════════════════════════════════════════
    def build_inspector(self, parent: tk.Frame) -> None:
        self._init_state()
        self._reset_widget_refs()
        win = self._inspector_win
        if win is not None:
            sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
            win.geometry(f"{min(1500, sw - 80)}x{min(860, sh - 120)}")
            win.minsize(min(1000, sw - 80), min(560, sh - 120))
            for seq in ("<Control-h>", "<Control-H>"):
                win.bind(seq, lambda _e: self._open_help())

        panes = tk.PanedWindow(parent, orient=tk.HORIZONTAL, sashwidth=6, sashrelief=tk.RAISED)
        panes.pack(fill="both", expand=True)
        left, right = tk.Frame(panes), tk.Frame(panes)
        panes.add(left, width=450, minsize=320, stretch="never")
        panes.add(right, minsize=520, stretch="always")

        # ---- left: settings (scrollable) ----
        inner, scroll_cv = self._make_scrollable(left)
        self._build_top_bar(inner)
        self._build_input_group(inner)
        apply_notebook_tab_style()          # normally already applied by NodeEditorApp; harmless repeat
        nb = ttk.Notebook(inner)
        nb.pack(fill="x", padx=4, pady=(4, 2))
        for title, builder in (("Board & Detection", self._build_detection_tab),
                               ("Selection", self._build_selection_tab),
                               ("Calibration", self._build_calibration_tab)):
            tab = tk.Frame(nb, padx=4, pady=4)
            nb.add(tab, text=title)
            builder(tab)
        self._build_selected_table(inner)
        self._bind_scroll_recursive(inner, scroll_cv)

        # ---- right: views ----
        rnb = ttk.Notebook(right)
        rnb.pack(fill="both", expand=True, padx=2, pady=2)
        self._right_nb = rnb
        for title, builder in (("Image preview", self._build_preview_tab),
                               ("All corners", self._build_coverage_tab),
                               ("Reprojection error", self._build_error_tab),
                               ("Residual vectors", self._build_residual_tab),
                               ("Camera poses 3D", self._build_pose_tab),
                               ("Undistort image", self._build_undistort_tab),
                               ("Sensitivity (LOO)", self._build_loo_tab)):
            tab = tk.Frame(rnb)
            rnb.add(tab, text=title)
            builder(tab)
        # The undistort tab computes lazily: only when it is (or becomes) the visible tab.
        rnb.bind("<<NotebookTabChanged>>", lambda _e: self._redraw_undistort())

        # Re-attach to background jobs that kept running while the inspector was closed.
        detect_running = self._detect_state is not None and not self._detect_state["finished"]
        loo_running = self._loo_state is not None and not self._loo_state["finished"]
        self._set_detect_running(detect_running)
        self._set_loo_running(loo_running)
        if detect_running:
            self._schedule_detect_poll()
        if loo_running:
            self._schedule_loo_poll()
        self._refresh_all_views()

    def _build_top_bar(self, parent) -> None:
        bar = tk.Frame(parent)
        bar.pack(fill="x", padx=4, pady=(4, 0))
        tk.Button(bar, text="Load...", font=("Arial", 8), command=self._on_load).pack(side="left")
        tk.Button(bar, text="Save as...", font=("Arial", 8), command=self._on_save_as).pack(side="left", padx=(4, 0))
        tk.Button(bar, text="Help (Ctrl+H)", font=("Arial", 8), command=self._open_help).pack(side="right")
        self._file_label_var = tk.StringVar(value=self._file_label_text())
        tk.Label(parent, textvariable=self._file_label_var, font=("Arial", 8), fg="#555555",
                 anchor="w", justify="left", wraplength=420).pack(fill="x", padx=6, pady=(2, 0))

    def _build_input_group(self, parent) -> None:
        grp = tk.LabelFrame(parent, text="Input", font=("Arial", 8, "bold"), padx=4, pady=2)
        grp.pack(fill="x", padx=4, pady=2)
        tk.Label(grp, textvariable=self._files_info_var, font=("Arial", 8),
                 anchor="w", justify="left").pack(fill="x")

    def _build_detection_tab(self, tab) -> None:
        v = self._det_vars
        r = 0
        # Same order as OpenCV's patternSize = (ncol, nrow) = (width, height).
        for key, label in (("ncol", "Columns: inner corners per row (patternSize width)"),
                           ("nrow", "Rows: inner corners per column (patternSize height)"),
                           ("square_size", "Square size (world units)")):
            self._field(tab, r, label, v[key])
            r += 1
        flags = tk.Frame(tab)
        flags.grid(row=r, column=0, columnspan=2, sticky="w", pady=(2, 2))
        r += 1
        for key, label in (("adaptive_thresh", "Adaptive thresh"), ("normalize_image", "Normalize"),
                           ("fast_check", "Fast check")):
            tk.Checkbutton(flags, text=label, variable=v[key], font=("Arial", 8)).pack(side="left")
        for key, label in (("max_width", "Detection max width (px, 0 = full)"),
                           ("subpix_win", "cornerSubPix window (px, 0 = auto)"),
                           ("subpix_iters", "cornerSubPix max iterations"),
                           ("subpix_eps", "cornerSubPix epsilon"),
                           ("threads", "Worker threads"),
                           ("start", "First frame (1-based)"),
                           ("end", "Last frame (blank = last)"),
                           ("step", "Frame step")):
            self._field(tab, r, label, v[key])
            r += 1
        btns = tk.Frame(tab)
        btns.grid(row=r, column=0, columnspan=2, sticky="w", pady=(6, 2))
        r += 1
        self._detect_btn = tk.Button(btns, text="Detect & Select", font=("Arial", 8, "bold"),
                                     bg="#e8dcf5", command=self._on_detect_clicked)
        self._detect_btn.pack(side="left")
        self._stop_detect_btn = tk.Button(btns, text="Stop", font=("Arial", 8),
                                          command=self._on_stop_detect, state=tk.DISABLED)
        self._stop_detect_btn.pack(side="left", padx=(4, 0))
        self._detect_progress_var = tk.DoubleVar(value=0.0)
        self._detect_progress = ttk.Progressbar(tab, variable=self._detect_progress_var, maximum=100.0)
        self._detect_progress.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        r += 1
        self._detect_text_var = tk.StringVar(value=self._detect_text)
        tk.Label(tab, textvariable=self._detect_text_var, font=("Arial", 8), anchor="w",
                 justify="left", wraplength=380).grid(row=r, column=0, columnspan=2, sticky="w")
        tab.columnconfigure(1, weight=1)

    def _build_selection_tab(self, tab) -> None:
        v = self._sel_vars
        self._field(tab, 0, "Target image count", v["target_count"])
        self._field(tab, 1, "Minimum sharpness (0 = any)", v["min_sharpness"])
        self._field(tab, 2, "Diversity threshold (x diagonal)", v["diversity_threshold"])
        self._reselect_btn = tk.Button(tab, text="Reselect (uses cached detection)", font=("Arial", 8),
                                       command=self._on_reselect_clicked)
        self._reselect_btn.grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 2))
        self._selection_summary_var = tk.StringVar(value=self._selection_text)
        tk.Label(tab, textvariable=self._selection_summary_var, font=("Arial", 8), anchor="w",
                 justify="left", wraplength=380).grid(row=4, column=0, columnspan=2, sticky="w")

    def _build_calibration_tab(self, tab) -> None:
        v = self._calib_vars
        flags = tk.Frame(tab)
        flags.pack(fill="x")
        items = (("use_intrinsic_guess", "Use intrinsic guess"), ("fix_principal_point", "Fix principal point"),
                 ("fix_aspect_ratio", "Fix aspect ratio"), ("fix_focal_length", "Fix focal length"),
                 ("zero_tangent_dist", "Zero tangential dist."), ("rational_model", "Rational model (k4-k6)"),
                 ("thin_prism_model", "Thin prism (s1-s4)"), ("tilted_model", "Tilted sensor (tau_x, tau_y)"))
        for i, (key, label) in enumerate(items):
            tk.Checkbutton(flags, text=label, variable=v[key], font=("Arial", 8)).grid(
                row=i // 2, column=i % 2, sticky="w")
        krow = tk.Frame(tab)
        krow.pack(fill="x", pady=(2, 0))
        tk.Label(krow, text="Fix:", font=("Arial", 8)).pack(side="left")
        for k in range(1, 7):
            tk.Checkbutton(krow, text=f"k{k}", variable=v[f"fix_k{k}"], font=("Arial", 8)).pack(side="left")
        krow2 = tk.Frame(tab)
        krow2.pack(fill="x", pady=(0, 2))
        tk.Label(krow2, text="Fix:", font=("Arial", 8)).pack(side="left")
        tk.Checkbutton(krow2, text="s1-s4", variable=v["fix_s1_s4"], font=("Arial", 8)).pack(side="left")
        tk.Checkbutton(krow2, text="tau_x, tau_y", variable=v["fix_taux_tauy"], font=("Arial", 8)).pack(side="left")

        guess = tk.LabelFrame(tab, text="Initial intrinsic guess", font=("Arial", 8), padx=4, pady=2)
        guess.pack(fill="x", pady=(2, 2))
        for i, (key, label) in enumerate((("guess_fx", "fx"), ("guess_fy", "fy"),
                                          ("guess_cx", "cx"), ("guess_cy", "cy"))):
            tk.Label(guess, text=label, font=("Arial", 8)).grid(row=i // 2, column=(i % 2) * 2, sticky="e", padx=2)
            tk.Entry(guess, textvariable=v[key], width=10, font=("Arial", 8)).grid(
                row=i // 2, column=(i % 2) * 2 + 1, sticky="w", padx=2)
        tk.Button(guess, text="Guess from image size", font=("Arial", 8),
                  command=self._on_guess_intrinsics).grid(row=2, column=0, columnspan=4, sticky="w", pady=(2, 0))

        self._calib_btn = tk.Button(tab, text="Run Calibration (checked images)", font=("Arial", 8, "bold"),
                                    bg="#e8dcf5", command=self._on_run_calibration)
        self._calib_btn.pack(anchor="w", pady=(4, 2))

        runs = tk.LabelFrame(tab, text="Calibration runs", font=("Arial", 8), padx=2, pady=2)
        runs.pack(fill="x")
        cols = ("act", "n", "rms", "fx", "fy", "cx", "cy") + DIST_NAMES + ("flags",)
        heads = {c: c for c in cols}
        heads.update(act="", rms="RMS px", flags="time / flags")
        widths = {c: 64 for c in cols}
        widths.update(act=22, n=30, rms=60, cx=60, cy=60, flags=220)
        frm = tk.Frame(runs)
        frm.pack(fill="x")
        tree = ttk.Treeview(frm, columns=cols, show="headings", height=5, selectmode="browse")
        for c in cols:
            tree.heading(c, text=heads[c])
            tree.column(c, width=widths[c], anchor="center" if c != "flags" else "w", stretch=False)
        hsb = ttk.Scrollbar(frm, orient="horizontal", command=tree.xview)
        tree.configure(xscrollcommand=hsb.set)
        tree.pack(fill="x")
        hsb.pack(fill="x")
        tree.bind("<Double-1>", lambda _e: self._on_set_active())
        self._runs_tree = tree
        rb = tk.Frame(runs)
        rb.pack(fill="x", pady=(2, 0))
        self._set_active_btn = tk.Button(rb, text="Set Active", font=("Arial", 8), command=self._on_set_active)
        self._set_active_btn.pack(side="left")
        tk.Button(rb, text="Delete run", font=("Arial", 8), command=self._on_delete_run).pack(side="left", padx=(4, 0))

        self._active_summary_var = tk.StringVar(value="")
        tk.Label(tab, textvariable=self._active_summary_var, font=("Courier", 8), anchor="w",
                 justify="left").pack(fill="x", pady=(4, 0))

    def _build_selected_table(self, parent) -> None:
        grp = tk.LabelFrame(parent, text="Selected images (click a 'selected' cell to include / exclude)",
                            font=("Arial", 8, "bold"), padx=2, pady=2)
        grp.pack(fill="both", expand=True, padx=4, pady=(4, 4))
        cols = ("sel", "order", "src", "sharp", "cover", "centre", "err")
        heads = {"sel": "selected", "order": "#", "src": "source #", "sharp": "sharpness",
                 "cover": "coverage", "centre": "centre dist", "err": "RMS err (px)"}
        widths = {"sel": 60, "order": 34, "src": 62, "sharp": 72, "cover": 64, "centre": 72, "err": 80}
        frm = tk.Frame(grp)
        frm.pack(fill="both", expand=True)
        tree = ttk.Treeview(frm, columns=cols, show="headings", height=12, selectmode="browse")
        for c in cols:
            tree.heading(c, text=heads[c])
            tree.column(c, width=widths[c], anchor="center", stretch=False)
        vsb = ttk.Scrollbar(frm, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        tree.pack(side="left", fill="both", expand=True)
        tree.tag_configure("excluded", foreground="#999999")
        tree.bind("<Button-1>", self._on_selected_tree_click, add="+")
        tree.bind("<<TreeviewSelect>>", self._on_selected_row_changed)
        self._selected_tree = tree
        bar = tk.Frame(grp)
        bar.pack(fill="x", pady=(2, 0))
        tk.Button(bar, text="Check all", font=("Arial", 8),
                  command=lambda: self._set_all_included(True)).pack(side="left")

    def _build_preview_tab(self, tab) -> None:
        bar = tk.Frame(tab)
        bar.pack(fill="x", padx=2, pady=2)
        tk.Button(bar, text="< Prev", font=("Arial", 8), command=lambda: self._step_selected(-1)).pack(side="left")
        tk.Button(bar, text="Next >", font=("Arial", 8),
                  command=lambda: self._step_selected(1)).pack(side="left", padx=(2, 0))
        tk.Button(bar, text="Fit view", font=("Arial", 8),
                  command=self._reset_preview_view).pack(side="left", padx=(6, 0))
        self._preview_label_var = tk.StringVar(value="")
        tk.Label(bar, textvariable=self._preview_label_var, font=("Arial", 8), anchor="w").pack(
            side="left", padx=(8, 0), fill="x", expand=True)
        c = tk.Canvas(tab, bg="#202020", highlightthickness=0)
        c.pack(fill="both", expand=True)
        c.bind("<Configure>", lambda _e: self._redraw_preview())
        c.bind("<ButtonPress-1>", self._on_preview_press)
        c.bind("<B1-Motion>", self._on_preview_drag)
        c.bind("<ButtonRelease-1>", lambda _e: self._preview_pan.update(active=False))
        c.bind("<Double-Button-1>", lambda _e: self._reset_preview_view())
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            c.bind(seq, self._on_preview_wheel)
        self._c_preview = c

    def _build_coverage_tab(self, tab) -> None:
        bar = tk.Frame(tab)
        bar.pack(fill="x", padx=2, pady=2)
        tk.Button(bar, text="Fit view", font=("Arial", 8), command=self._reset_coverage_view).pack(side="left")
        tk.Label(bar, text="Hover a point to see its image; click it to select that image. "
                           "Drag = pan, wheel = zoom, double-click = fit.",
                 font=("Arial", 8), fg="#555555").pack(side="left", padx=(8, 0))
        self._cov_label_var = tk.StringVar(value="")
        tk.Label(tab, textvariable=self._cov_label_var, font=("Arial", 8), anchor="w",
                 justify="left", wraplength=900).pack(fill="x", padx=4)
        c = tk.Canvas(tab, bg="#202020", highlightthickness=0)
        c.pack(fill="both", expand=True)
        c.bind("<Configure>", lambda _e: self._redraw_coverage())
        c.bind("<ButtonPress-1>", self._on_cov_press)
        c.bind("<B1-Motion>", self._on_cov_drag)
        c.bind("<ButtonRelease-1>", self._on_cov_release)
        c.bind("<Double-Button-1>", lambda _e: self._reset_coverage_view())
        c.bind("<Motion>", self._on_cov_motion)
        c.bind("<Leave>", lambda _e: self._cov_hide_tip())
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            c.bind(seq, self._on_cov_wheel)
        self._c_cov = c

    def _build_error_tab(self, tab) -> None:
        bar = tk.Frame(tab)
        bar.pack(fill="x", padx=2, pady=2)
        tk.Button(bar, text="Reset view", font=("Arial", 8), command=self._redraw_err).pack(side="left")
        tk.Label(bar, text="Double-click a bar to select that image.", font=("Arial", 8),
                 fg="#555555").pack(side="left", padx=(8, 0))
        self._fig_err = Figure(figsize=(6, 3.2), dpi=100)
        self._fig_err.subplots_adjust(left=0.09, right=0.98, top=0.9, bottom=0.22)
        self._ax_err = self._fig_err.add_subplot(111)
        self._canvas_err = FigureCanvasTkAgg(self._fig_err, master=tab)
        self._canvas_err.get_tk_widget().pack(fill="both", expand=True)
        self._connect_mpl_pan_zoom(self._canvas_err, lambda: self._ax_err, self._pan_err,
                                   on_dblclick=self._on_err_dblclick)

    def _build_residual_tab(self, tab) -> None:
        bar = tk.Frame(tab)
        bar.pack(fill="x", padx=2, pady=2)
        tk.Button(bar, text="< Prev", font=("Arial", 8), command=lambda: self._step_selected(-1)).pack(side="left")
        tk.Button(bar, text="Next >", font=("Arial", 8),
                  command=lambda: self._step_selected(1)).pack(side="left", padx=(2, 0))
        tk.Label(bar, text="Magnification x", font=("Arial", 8)).pack(side="left", padx=(8, 0))
        spin = tk.Spinbox(bar, from_=1, to=100000, increment=10, width=7, font=("Arial", 8),
                          textvariable=self._resel_mag_var, command=self._redraw_res)
        spin.pack(side="left")
        spin.bind("<Return>", lambda _e: self._redraw_res())
        tk.Button(bar, text="Reset view", font=("Arial", 8), command=self._reset_res_view).pack(side="left", padx=(6, 0))
        self._residual_label_var = tk.StringVar(value="")
        tk.Label(tab, textvariable=self._residual_label_var, font=("Arial", 8), anchor="w").pack(fill="x", padx=4)
        self._fig_res = Figure(figsize=(6, 4), dpi=100)
        self._fig_res.subplots_adjust(left=0.1, right=0.98, top=0.92, bottom=0.1)
        self._ax_res = self._fig_res.add_subplot(111)
        self._canvas_res = FigureCanvasTkAgg(self._fig_res, master=tab)
        self._canvas_res.get_tk_widget().pack(fill="both", expand=True)
        self._connect_mpl_pan_zoom(self._canvas_res, lambda: self._ax_res, self._pan_res)

    def _build_pose_tab(self, tab) -> None:
        bar = tk.Frame(tab)
        bar.pack(fill="x", padx=2, pady=2)
        self._persp_var = tk.DoubleVar(value=0.3)
        tk.Label(bar, text="Perspective", font=("Arial", 8)).pack(side="left")
        tk.Scale(bar, from_=0.0, to=0.9, resolution=0.05, orient="horizontal", length=110, showvalue=False,
                 variable=self._persp_var, command=lambda _v: self._redraw_3d()).pack(side="left")
        tk.Label(bar, text="Frustum size", font=("Arial", 8)).pack(side="left", padx=(8, 0))
        tk.Scale(bar, from_=0.2, to=3.0, resolution=0.1, orient="horizontal", length=110, showvalue=False,
                 variable=self._frustum_scale_var, command=lambda _v: self._redraw_3d()).pack(side="left")
        tk.Button(bar, text="Reset view", font=("Arial", 8), command=self._reset_3d_view).pack(side="left", padx=(8, 0))
        tk.Label(bar, text="Fixed:", font=("Arial", 8)).pack(side="left", padx=(8, 0))
        for value, label in (("board", "Board"), ("camera", "Camera")):
            tk.Radiobutton(bar, text=label, value=value, variable=self._pose_frame_var, font=("Arial", 8),
                           command=self._reset_3d_view).pack(side="left")
        tk.Label(bar, text="L-drag rotate, R-drag pan, wheel zoom", font=("Arial", 8),
                 fg="#555555").pack(side="left", padx=(8, 0))
        c = tk.Canvas(tab, bg="white", highlightthickness=0)
        c.pack(fill="both", expand=True)
        c.bind("<Configure>", lambda _e: self._redraw_3d())
        c.bind("<ButtonPress-1>", lambda e: self._on_3d_press(e, 1))
        c.bind("<ButtonPress-3>", lambda e: self._on_3d_press(e, 3))
        c.bind("<ButtonPress-2>", lambda e: self._on_3d_press(e, 3))   # macOS right button
        c.bind("<B1-Motion>", self._on_3d_drag)
        c.bind("<B3-Motion>", self._on_3d_drag)
        c.bind("<B2-Motion>", self._on_3d_drag)
        for seq in ("<ButtonRelease-1>", "<ButtonRelease-2>", "<ButtonRelease-3>"):
            c.bind(seq, lambda _e: self._v3.update(dragging=False))
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            c.bind(seq, self._on_3d_wheel)
        self._c3d = c

    def _build_undistort_tab(self, tab) -> None:
        cfg = self._und_cfg
        v = self._und_vars = {
            "newk_mode": tk.StringVar(value=cfg["newk_mode"]),
            "alpha": tk.StringVar(value=f"{cfg['alpha']:g}"),
            "center_pp": tk.BooleanVar(value=cfg["center_pp"]),
            "show_corners": tk.BooleanVar(value=cfg["show_corners"]),
            "show_lines": tk.BooleanVar(value=cfg["show_lines"]),
            "show_roi": tk.BooleanVar(value=cfg["show_roi"]),
        }
        bar = tk.Frame(tab)
        bar.pack(fill="x", padx=2, pady=(2, 0))
        tk.Button(bar, text="< Prev", font=("Arial", 8), command=lambda: self._step_selected(-1)).pack(side="left")
        tk.Button(bar, text="Next >", font=("Arial", 8),
                  command=lambda: self._step_selected(1)).pack(side="left", padx=(2, 0))
        tk.Button(bar, text="Fit view", font=("Arial", 8),
                  command=self._reset_undistort_view).pack(side="left", padx=(6, 0))
        tk.Label(bar, text="newCameraMatrix:", font=("Arial", 8)).pack(side="left", padx=(10, 0))
        for value, label in (("same", "same as K"), ("optimal", "getOptimalNewCameraMatrix")):
            tk.Radiobutton(bar, text=label, value=value, variable=v["newk_mode"], font=("Arial", 8)).pack(side="left")
        tk.Label(bar, text="alpha (0-1)", font=("Arial", 8)).pack(side="left", padx=(4, 0))
        tk.Entry(bar, textvariable=v["alpha"], width=5, font=("Arial", 8)).pack(side="left")
        tk.Checkbutton(bar, text="centerPrincipalPoint", variable=v["center_pp"], font=("Arial", 8)).pack(side="left")
        bar2 = tk.Frame(tab)
        bar2.pack(fill="x", padx=2, pady=(0, 2))
        tk.Label(bar2, text="Overlay:", font=("Arial", 8)).pack(side="left")
        for key, label in (("show_corners", "undistorted corners"), ("show_lines", "row/column lines"),
                           ("show_roi", "valid ROI")):
            tk.Checkbutton(bar2, text=label, variable=v[key], font=("Arial", 8)).pack(side="left")
        tk.Button(bar2, text="Apply", font=("Arial", 8, "bold"), bg="#e8dcf5", width=8,
                  command=self._on_undistort_apply).pack(side="left", padx=(10, 0))
        self._und_label_var = tk.StringVar(value="")
        tk.Label(tab, textvariable=self._und_label_var, font=("Arial", 8), anchor="w",
                 justify="left").pack(fill="x", padx=4)
        c = tk.Canvas(tab, bg="#202020", highlightthickness=0)
        c.pack(fill="both", expand=True)
        c.bind("<Configure>", lambda _e: self._redraw_undistort())
        c.bind("<ButtonPress-1>", self._on_und_press)
        c.bind("<B1-Motion>", self._on_und_drag)
        c.bind("<ButtonRelease-1>", lambda _e: self._und_pan.update(active=False))
        c.bind("<Double-Button-1>", lambda _e: self._reset_undistort_view())
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            c.bind(seq, self._on_und_wheel)
        self._c_und = c
        self._und_tab = tab

    def _build_loo_tab(self, tab) -> None:
        bar = tk.Frame(tab)
        bar.pack(fill="x", padx=2, pady=2)
        self._loo_btn = tk.Button(bar, text="Run Sensitivity (leave-one-out)", font=("Arial", 8, "bold"),
                                  command=self._on_run_loo)
        self._loo_btn.pack(side="left")
        self._stop_loo_btn = tk.Button(bar, text="Stop", font=("Arial", 8), command=self._on_stop_loo,
                                       state=tk.DISABLED)
        self._stop_loo_btn.pack(side="left", padx=(4, 0))
        self._loo_progress_var = tk.DoubleVar(value=0.0)
        self._loo_progress = ttk.Progressbar(bar, variable=self._loo_progress_var, maximum=100.0, length=160)
        self._loo_progress.pack(side="left", padx=(8, 0))
        tk.Label(tab, font=("Arial", 8), anchor="w", justify="left", fg="#555555", wraplength=700,
                 text="Each row = the ACTIVE run recomputed WITHOUT that one image; d_* = change vs. the "
                      "active run. Highlighted rows shift the intrinsics much more than typical and are "
                      "worth inspecting. Click a row to preview that image.").pack(fill="x", padx=4)
        cols = ("src",) + tuple(x for name in ("rms", "fx", "fy", "cx", "cy") + DIST_NAMES
                                for x in (name, f"d_{name}"))
        frm = tk.Frame(tab)
        frm.pack(fill="both", expand=True, padx=2, pady=2)
        tree = ttk.Treeview(frm, columns=cols, show="headings", selectmode="browse")
        for c in cols:
            tree.heading(c, text="excluded source #" if c == "src" else c)
            tree.column(c, width=110 if c == "src" else 72, anchor="center", stretch=False)
        vsb = ttk.Scrollbar(frm, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(frm, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        tree.pack(side="left", fill="both", expand=True)
        tree.tag_configure("outlier", background="#ffe0e0")
        tree.tag_configure("error", foreground="#b00000")
        tree.bind("<<TreeviewSelect>>", self._on_loo_row_selected)
        self._loo_tree = tree

    # ══ Refresh hub ═════════════════════════════════════════════════════
    def _refresh_all_views(self) -> None:
        if self._selection_summary_var is not None:
            self._selection_summary_var.set(self._selection_text)
        self._refresh_file_label()
        self._refresh_selected_tree()
        self._refresh_runs_tree()
        self._refresh_active_summary()
        self._refresh_loo_tree()
        self._redraw_preview()
        self._redraw_coverage()
        self._redraw_err()
        self._redraw_res()
        self._redraw_3d()
        self._redraw_undistort()
        self._update_status_item()

    # ══ Stage 1: detection ══════════════════════════════════════════════
    def _set_detect_running(self, running: bool) -> None:
        self._set_state(self._detect_btn, tk.DISABLED if running else tk.NORMAL)
        self._set_state(self._reselect_btn, tk.DISABLED if running else tk.NORMAL)
        self._set_state(self._stop_detect_btn, tk.NORMAL if running else tk.DISABLED)

    def _set_detect_text(self, text: str) -> None:
        self._detect_text = text
        if self._detect_text_var is not None:
            try:
                self._detect_text_var.set(text)
            except tk.TclError:
                pass

    def _on_detect_clicked(self) -> None:
        if self._detect_state is not None and not self._detect_state["finished"]:
            return
        if not self._files:
            messagebox.showinfo("Detect", "Connect an image sequence ('files' input) first.",
                                parent=self._dlg_parent())
            return
        cfg, errs = self._collect_detection_cfg()
        _sel_cfg, errs2 = self._collect_selection_cfg()
        errs += errs2
        if errs:
            messagebox.showerror("Detect", "\n".join(errs), parent=self._dlg_parent())
            return
        n = len(self._files)
        end = min(cfg["end"] or n, n)
        indices = list(range(cfg["start"] - 1, end, cfg["step"]))
        if not indices:
            messagebox.showerror("Detect", "The frame range is empty.", parent=self._dlg_parent())
            return
        state = {"cancel": threading.Event(), "done": 0, "found": 0, "total": len(indices),
                 "results": None, "error": None, "finished": False}
        job = {"paths": [self._files[i] for i in indices], "indices": indices, "cfg": cfg}
        self._detect_state = state
        self._pending_detect_cfg = cfg
        threading.Thread(target=_detect_worker, args=(job, state), daemon=True).start()
        self._set_detect_running(True)
        self._set_detect_text(f"Detecting 0 / {len(indices)} ...")
        self._schedule_detect_poll()

    def _on_stop_detect(self) -> None:
        if self._detect_state is not None and not self._detect_state["finished"]:
            self._detect_state["cancel"].set()

    def _schedule_detect_poll(self) -> None:
        if self._detect_poll_id is None and not self._destroyed:
            try:
                self._detect_poll_id = self.canvas.after(100, self._poll_detect)
            except tk.TclError:
                pass

    def _poll_detect(self) -> None:
        self._detect_poll_id = None
        st = self._detect_state
        if st is None or self._destroyed:
            return
        total = max(1, st["total"])
        if self._detect_progress_var is not None:
            try:
                self._detect_progress_var.set(100.0 * st["done"] / total)
            except tk.TclError:
                pass
        if not st["finished"]:
            self._set_detect_text(f"Detecting {st['done']} / {st['total']} ... boards found: {st['found']}")
            self._schedule_detect_poll()
            return

        self._set_detect_running(False)
        if st["error"]:
            self._set_detect_text(f"Detection failed: {st['error']}")
            messagebox.showerror("Detect", st["error"], parent=self._dlg_parent())
            return
        results = [r for r in (st["results"] or []) if r is not None]
        found = [r for r in results if r.get("found")]
        # calibrateCamera needs one image size: keep the size of the first detected board.
        dropped = 0
        if found:
            ref_size = tuple(found[0]["size"])
            kept = [r for r in found if tuple(r["size"]) == ref_size]
            dropped = len(found) - len(kept)
            found = kept
        self._candidates = found
        self._ranking = None
        self._detect_cfg = getattr(self, "_pending_detect_cfg", None)
        cancelled = st["cancel"].is_set()
        text = (f"{'Stopped' if cancelled else 'Done'}: {len(results)} of {st['total']} images scanned, "
                f"board found in {len(found)}.")
        if dropped:
            text += f" {dropped} detected image(s) dropped because their size differs from the first one."
        self._set_detect_text(text)
        if found:
            self._run_selection(show_errors=True)
        else:
            messagebox.showinfo("Detect", "No chessboard was detected. Check the inner-corner counts "
                                "(INNER corners, not squares; columns = per row, rows = per column) "
                                "and the detection flags.", parent=self._dlg_parent())

    # ══ Stage 1: selection ══════════════════════════════════════════════
    def _on_reselect_clicked(self) -> None:
        if not self._candidates:
            messagebox.showinfo("Reselect", "Run 'Detect & Select' at least once first.",
                                parent=self._dlg_parent())
            return
        self._run_selection(show_errors=True)

    def _run_selection(self, show_errors: bool) -> None:
        sel_cfg, errs = self._collect_selection_cfg()
        if errs:
            if show_errors:
                messagebox.showerror("Selection", "\n".join(errs), parent=self._dlg_parent())
            return
        # The ranking depends only on the candidates and the minimum
        # sharpness; changing the target count or the diversity threshold
        # just takes a different prefix of the cached ranking.
        if self._ranking is None or self._ranking_min_sharpness != sel_cfg["min_sharpness"]:
            try:
                self._ranking = rank_calibration_images(list(self._candidates), sel_cfg["min_sharpness"])
            except Exception as e:
                messagebox.showerror("Selection", f"{type(e).__name__}: {e}", parent=self._dlg_parent())
                return
            self._ranking_min_sharpness = sel_cfg["min_sharpness"]
        selected = take_ranked_images(self._ranking, sel_cfg["target_count"], sel_cfg["diversity_threshold"])
        n_valid = len(self._ranking)
        if not selected:
            self._stage1 = None
            self._selection_text = ("No images passed the current filters -- lower the minimum sharpness "
                                    "or run detection again.")
        else:
            nrow, ncol = selected[0]["corners"].shape[:2]      # corners are (nrow, ncol, 2), OpenCV row-major
            cfg = self._detect_cfg or {}
            # Square size may be edited after detection without re-detecting:
            # it only scales the object points, never the corner detection.
            square = self._parse_float(self._det_vars["square_size"], 1e-9) or float(cfg.get("square_size") or 1.0)
            grid = build_object_grid(ncol, nrow, square)       # (nrow, ncol, 3)
            n = len(selected)
            self._stage1 = {
                "object_points": np.repeat(grid[None], n, axis=0),
                "image_points": np.stack([c["corners"] for c in selected]).astype(np.float64),
                "sharpness": np.array([c["sharpness"] for c in selected], dtype=np.float64),
                "source_index": np.array([c["source_index"] for c in selected], dtype=np.int64),
                "img_size": tuple(int(v) for v in selected[0]["size"]),
                "included": np.ones(n, dtype=bool),
                "selected": [{"path": c["path"], "source_index": int(c["source_index"]),
                              "corners": c["corners"], "sharpness": float(c["sharpness"]),
                              "coverage": float(c["coverage"]), "center_dist": float(c["center_dist"])}
                             for c in selected],
                "ncol": int(ncol), "nrow": int(nrow), "square_size": float(square),
            }
            text = (f"{n} image(s) selected from {n_valid} usable candidates "
                    f"(target {sel_cfg['target_count']}).")
            if n < sel_cfg["target_count"] and n < n_valid:
                text += (f" Stopped early: image #{n + 1} in the ranking is only "
                         f"{self._ranking[n]['pick_dist']:.4g} x diagonal from the images before it "
                         f"(threshold {sel_cfg['diversity_threshold']:g}).")
            self._selection_text = text
        self._preview_selected_pos = 0
        self._preview_user_view = False
        self._cov_user_view = False
        self._und_user_view = False
        self._mark_unsaved()
        self._notify_outputs_changed()
        self._refresh_all_views()

    # ══ Selected-images table ═══════════════════════════════════════════
    def _active_error_by_source(self) -> dict:
        run = self._active_run()
        if run is None:
            return {}
        return {int(s): float(e) for s, e in zip(run["source_index"], run["per_image_error"])}

    def _selected_row_values(self, pos: int, err_by_src: dict) -> tuple:
        s1 = self._stage1
        c = s1["selected"][pos]
        err = err_by_src.get(int(c["source_index"]))
        return (CHECK_ON if s1["included"][pos] else CHECK_OFF, pos + 1, c["source_index"],
                f"{c['sharpness']:.1f}",
                f"{c['coverage']:.3f}" if math.isfinite(c["coverage"]) else "-",
                f"{c['center_dist']:.3f}" if math.isfinite(c["center_dist"]) else "-",
                f"{err:.3f}" if err is not None else "-")

    def _refresh_selected_tree(self) -> None:
        tree = self._selected_tree
        if tree is None:
            return
        tree.delete(*tree.get_children())
        s1 = self._stage1
        if s1 is None:
            return
        err_by_src = self._active_error_by_source()
        for pos in range(len(s1["selected"])):
            tags = () if s1["included"][pos] else ("excluded",)
            tree.insert("", "end", iid=str(pos), values=self._selected_row_values(pos, err_by_src), tags=tags)
        if s1["selected"]:
            pos = min(max(self._preview_selected_pos, 0), len(s1["selected"]) - 1)
            tree.selection_set(str(pos))
            tree.see(str(pos))

    def _on_selected_tree_click(self, event):
        """Click in the 'selected' column toggles inclusion of that row."""
        tree = self._selected_tree
        if tree is None or tree.identify_region(event.x, event.y) != "cell":
            return None
        if tree.identify_column(event.x) != "#1":
            return None
        iid = tree.identify_row(event.y)
        if iid:
            self._toggle_included(int(iid))
        return None

    def _on_selected_row_changed(self, _event=None) -> None:
        tree = self._selected_tree
        if tree is None:
            return
        sel = tree.selection()
        if not sel:
            return
        pos = int(sel[0])
        if pos != self._preview_selected_pos:
            self._select_position(pos)

    def _toggle_included(self, pos: int) -> None:
        s1 = self._stage1
        if s1 is None or not (0 <= pos < len(s1["selected"])):
            return
        mask = s1["included"]
        if mask[pos] and int(mask.sum()) <= 1:
            messagebox.showinfo("Selected images", "At least one image must stay included.",
                                parent=self._dlg_parent())
            return
        mask[pos] = not mask[pos]
        if self._selected_tree is not None:
            self._selected_tree.item(str(pos), values=self._selected_row_values(pos, self._active_error_by_source()),
                                     tags=() if mask[pos] else ("excluded",))
        self._mark_unsaved()
        self._notify_outputs_changed()
        self._update_status_item()
        if pos == self._preview_selected_pos:
            self._redraw_preview()
        self._redraw_coverage()

    def _set_all_included(self, value: bool) -> None:
        s1 = self._stage1
        if s1 is None:
            return
        s1["included"][:] = value
        self._mark_unsaved()
        self._notify_outputs_changed()
        self._refresh_selected_tree()
        self._update_status_item()
        self._redraw_preview()
        self._redraw_coverage()

    def _pos_for_source(self, src: int) -> int | None:
        s1 = self._stage1
        if s1 is None:
            return None
        for pos, c in enumerate(s1["selected"]):
            if int(c["source_index"]) == int(src):
                return pos
        return None

    def _step_selected(self, delta: int) -> None:
        s1 = self._stage1
        if s1 is None or not s1["selected"]:
            return
        self._select_position((self._preview_selected_pos + delta) % len(s1["selected"]))

    def _select_position(self, pos: int) -> None:
        """Single entry point for 'the current image changed' -- keeps the
        table, preview, residual plot, error-bar highlight and 3D highlight
        in sync. The preview and residual-plot pan/zoom are KEPT, so the
        same region can be compared across images."""
        self._preview_selected_pos = pos
        tree = self._selected_tree
        if tree is not None and tree.exists(str(pos)):
            tree.selection_set(str(pos))
            tree.see(str(pos))
        self._redraw_preview()
        self._redraw_coverage()
        self._redraw_res()
        self._redraw_err()
        self._redraw_3d()
        self._redraw_undistort()

    # ══ Image preview ═══════════════════════════════════════════════════
    def _ensure_preview_image(self, cand: dict) -> bool:
        """Load (and downscale for display) the image for `cand`. Records the
        downscale factor in _preview_corner_scale -- corner coordinates are
        full-resolution and MUST be multiplied by it before drawing (this was
        the 'corners too large / shifted to the bottom-right' bug)."""
        path = cand.get("path") or ""
        if path and path == self._preview_rgb_path and self._preview_rgb is not None:
            return True
        rgb = _read_rgb(path) if path else None
        if rgb is None:
            self._preview_rgb, self._preview_rgb_path, self._preview_corner_scale = None, None, 1.0
            return False
        h, w = rgb.shape[:2]
        f = 1.0
        if max(h, w) > PREVIEW_MAX_SIDE:
            f = PREVIEW_MAX_SIDE / float(max(h, w))
            rgb = cv2.resize(rgb, (max(1, int(round(w * f))), max(1, int(round(h * f)))),
                             interpolation=cv2.INTER_AREA)
        self._preview_rgb, self._preview_rgb_path, self._preview_corner_scale = rgb, path, f
        self._preview_photo_key = None
        return True

    def _redraw_preview(self) -> None:
        c = self._c_preview
        if c is None:
            return
        try:
            cw, ch = c.winfo_width(), c.winfo_height()
        except tk.TclError:
            return
        if cw < 20 or ch < 20:
            return
        s1 = self._stage1
        c.delete("all")
        if s1 is None or not s1["selected"]:
            self._preview_view = None
            c.create_text(cw / 2, ch / 2, fill="#999999", justify="center", width=cw - 40,
                          text="No selected image yet.\nRun 'Detect & Select'.")
            if self._preview_label_var is not None:
                self._preview_label_var.set("No image selected.")
            return
        pos = min(max(self._preview_selected_pos, 0), len(s1["selected"]) - 1)
        self._preview_selected_pos = pos
        cand = s1["selected"][pos]

        has_img = self._ensure_preview_image(cand)
        if has_img:
            ph, pw = self._preview_rgb.shape[:2]
            cs = self._preview_corner_scale
        else:
            pw, ph = s1["img_size"]
            cs = 1.0

        if self._preview_view is None or not self._preview_user_view:
            s = 0.98 * min(cw / pw, ch / ph)
            self._preview_view = ((cw - pw * s) / 2.0, (ch - ph * s) / 2.0, s)
        self._preview_last_size = (cw, ch)
        ox, oy, s = self._preview_view

        if has_img:
            # Crop to the visible region first, then resize only that part.
            x0 = max(0, int(math.floor(-ox / s)))
            y0 = max(0, int(math.floor(-oy / s)))
            x1 = min(pw, int(math.ceil((cw - ox) / s)))
            y1 = min(ph, int(math.ceil((ch - oy) / s)))
            if x1 > x0 and y1 > y0:
                key = (self._preview_rgb_path, x0, y0, x1, y1, round(s, 6))
                if key != self._preview_photo_key:
                    crop = self._preview_rgb[y0:y1, x0:x1]
                    dw = max(1, int(round((x1 - x0) * s)))
                    dh = max(1, int(round((y1 - y0) * s)))
                    interp = cv2.INTER_NEAREST if s >= 1.0 else cv2.INTER_AREA
                    disp = cv2.resize(crop, (dw, dh), interpolation=interp)
                    self._preview_photo = ImageTk.PhotoImage(Image.fromarray(disp), master=c)
                    self._preview_photo_key = key
                c.create_image(ox + x0 * s, oy + y0 * s, image=self._preview_photo, anchor="nw")
        else:
            c.create_rectangle(ox, oy, ox + pw * s, oy + ph * s, fill="#333333", outline="#666666")
            c.create_text(cw / 2, 16, fill="#dddddd", text="Image file not available -- showing corners only")

        # Full-res pixel-centre coords -> preview coords (x cs) -> canvas (x s + offset).
        # The +0.5 converts OpenCV's pixel-centre convention to pixel-edge canvas coordinates.
        # OpenCV order: k = r*ncol + c (row by row) -- first corner drawn larger.
        pts = cand["corners"].reshape(-1, 2)
        n = len(pts)
        for k, (x, y) in enumerate(pts):
            X = ox + (x + 0.5) * cs * s
            Y = oy + (y + 0.5) * cs * s
            t = k / max(1, n - 1)
            color = f"#{int(255 * t):02x}{int(220 * (1 - t)):02x}30"
            r = 5 if k == 0 else 3
            c.create_oval(X - r, Y - r, X + r, Y + r, outline=color, width=2 if k == 0 else 1.5)

        if self._preview_label_var is not None:
            inc = "included" if s1["included"][pos] else "EXCLUDED"
            name = os.path.basename(cand.get("path") or "") or "(no path)"
            self._preview_label_var.set(f"Image {pos + 1}/{len(s1['selected'])} | source # "
                                        f"{cand['source_index']} | {name} | {inc}")

    def _reset_preview_view(self) -> None:
        self._preview_user_view = False
        self._redraw_preview()

    def _on_preview_press(self, event) -> None:
        if self._preview_view is None:
            return
        ox, oy, _s = self._preview_view
        self._preview_pan.update(active=True, x_px=event.x, y_px=event.y, ox0=ox, oy0=oy)

    def _on_preview_drag(self, event) -> None:
        st = self._preview_pan
        if not st["active"] or self._preview_view is None:
            return
        _ox, _oy, s = self._preview_view
        self._preview_view = (st["ox0"] + event.x - st["x_px"], st["oy0"] + event.y - st["y_px"], s)
        self._preview_user_view = True
        self._redraw_preview()

    def _on_preview_wheel(self, event):
        if self._preview_view is None:
            return "break"
        steps = self._wheel_steps(event)
        if steps == 0:
            return "break"
        ox, oy, s = self._preview_view
        new_s = float(min(max(s * (IMG_ZOOM_FACTOR ** steps), IMG_MIN_SCALE), IMG_MAX_SCALE))
        k = new_s / s
        ex, ey = float(event.x), float(event.y)
        self._preview_view = (ex - (ex - ox) * k, ey - (ey - oy) * k, new_s)
        self._preview_user_view = True
        self._redraw_preview()
        return "break"

    # ══ All corners of all selected images (coverage) ═══════════════════
    @staticmethod
    def _image_color(k: int) -> str:
        """Distinct, stable colour per selected-image position (golden-ratio hue steps)."""
        r, g, b = colorsys.hsv_to_rgb((k * 0.618034) % 1.0, 0.7, 1.0)
        return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

    def _redraw_coverage(self) -> None:
        c = self._c_cov
        if c is None:
            return
        try:
            cw, ch = c.winfo_width(), c.winfo_height()
        except tk.TclError:
            return
        if cw < 20 or ch < 20:
            return
        c.delete("all")
        self._cov_hit = None
        s1 = self._stage1
        if s1 is None or not s1["selected"]:
            self._cov_view = None
            c.create_text(cw / 2, ch / 2, fill="#999999", justify="center", width=cw - 40,
                          text="No selected images yet.\nRun 'Detect & Select'.")
            if self._cov_label_var is not None:
                self._cov_label_var.set("")
            return
        w, h = s1["img_size"]
        if self._cov_view is None or not self._cov_user_view:
            s = 0.96 * min(cw / w, ch / h)
            self._cov_view = ((cw - w * s) / 2.0, (ch - h * s) / 2.0, s)
        ox, oy, s = self._cov_view

        # Image frame + rule-of-thirds guides.
        c.create_rectangle(ox, oy, ox + w * s, oy + h * s, fill="#2b2b2b", outline="#9a9a9a")
        for f in (1 / 3, 2 / 3):
            c.create_line(ox + w * f * s, oy, ox + w * f * s, oy + h * s, fill="#444444", dash=(3, 3))
            c.create_line(ox, oy + h * f * s, ox + w * s, oy + h * f * s, fill="#444444", dash=(3, 3))

        cur = min(max(self._preview_selected_pos, 0), len(s1["selected"]) - 1)
        xy_all, pos_all, r_all, c_all = [], [], [], []
        # Draw the current image last so it stays on top.
        for k in [k for k in range(len(s1["selected"])) if k != cur] + [cur]:
            corners = s1["selected"][k]["corners"]                # (nrow, ncol, 2), full-res px
            ncol = corners.shape[1]
            pts = corners.reshape(-1, 2)
            X = ox + (pts[:, 0] + 0.5) * s
            Y = oy + (pts[:, 1] + 0.5) * s
            included = bool(s1["included"][k])
            if k == cur:
                color, rad = "#ffffff", 3.5
            else:
                color, rad = (self._image_color(k), 2.0) if included else ("#666666", 1.5)
            for x, y in zip(X, Y):
                if included or k == cur:
                    c.create_oval(x - rad, y - rad, x + rad, y + rad, fill=color, outline="")
                else:
                    c.create_oval(x - rad, y - rad, x + rad, y + rad, outline=color)
            xy_all.append(np.stack([X, Y], axis=1))
            pos_all.append(np.full(len(pts), k))
            rr, cc = np.divmod(np.arange(len(pts)), ncol)
            r_all.append(rr)
            c_all.append(cc)
        self._cov_hit = (np.vstack(xy_all), np.concatenate(pos_all), np.concatenate(r_all), np.concatenate(c_all))

        if self._cov_label_var is not None:
            inc = [k for k in range(len(s1["selected"])) if s1["included"][k]]
            if inc:
                pts = np.vstack([s1["selected"][k]["corners"].reshape(-1, 2) for k in inc])
                gx = np.clip((pts[:, 0] / w * 10).astype(int), 0, 9)
                gy = np.clip((pts[:, 1] / h * 10).astype(int), 0, 9)
                cells = len(set(zip(gx.tolist(), gy.tolist())))
                text = (f"{len(inc)} checked image(s), {len(pts)} points. Image area reached: {cells}% of a "
                        f"10 x 10 grid of cells; points span x {pts[:, 0].min():.0f}-{pts[:, 0].max():.0f}, "
                        f"y {pts[:, 1].min():.0f}-{pts[:, 1].max():.0f} px (image {w} x {h}). "
                        f"White = current image, grey rings = unchecked images.")
            else:
                text = "No checked images."
            self._cov_label_var.set(text)

    def _reset_coverage_view(self) -> None:
        self._cov_user_view = False
        self._redraw_coverage()

    def _on_cov_press(self, event) -> None:
        self._cov_hide_tip()
        if self._cov_view is None:
            return
        ox, oy, _s = self._cov_view
        self._cov_pan.update(active=True, moved=False, x_px=event.x, y_px=event.y, ox0=ox, oy0=oy)

    def _on_cov_drag(self, event) -> None:
        st = self._cov_pan
        if not st["active"] or self._cov_view is None:
            return
        dx, dy = event.x - st["x_px"], event.y - st["y_px"]
        if not st["moved"] and abs(dx) + abs(dy) < 3:
            return
        st["moved"] = True
        self._cov_view = (st["ox0"] + dx, st["oy0"] + dy, self._cov_view[2])
        self._cov_user_view = True
        self._redraw_coverage()

    def _on_cov_release(self, event) -> None:
        st = self._cov_pan
        clicked = st["active"] and not st["moved"]
        st["active"] = False
        if clicked:
            hit = self._cov_nearest(event.x, event.y)
            if hit is not None and hit[0] != self._preview_selected_pos:
                self._select_position(hit[0])

    def _on_cov_wheel(self, event):
        self._cov_hide_tip()
        if self._cov_view is None:
            return "break"
        steps = self._wheel_steps(event)
        if steps == 0:
            return "break"
        ox, oy, s = self._cov_view
        new_s = float(min(max(s * (IMG_ZOOM_FACTOR ** steps), IMG_MIN_SCALE * 0.1), IMG_MAX_SCALE))
        k = new_s / s
        ex, ey = float(event.x), float(event.y)
        self._cov_view = (ex - (ex - ox) * k, ey - (ey - oy) * k, new_s)
        self._cov_user_view = True
        self._redraw_coverage()
        return "break"

    def _cov_nearest(self, x: float, y: float, tol: float = 7.0):
        """(image pos, corner r, corner c) of the drawn point nearest to canvas
        (x, y) within `tol` px, or None. Ties go to the point drawn last
        (the current image), which is the one visible on top."""
        hit = self._cov_hit
        if hit is None or len(hit[0]) == 0:
            return None
        d2 = np.sum((hit[0] - np.array([x, y])) ** 2, axis=1)
        i = len(d2) - 1 - int(np.argmin(d2[::-1]))
        if d2[i] > tol * tol:
            return None
        return int(hit[1][i]), int(hit[2][i]), int(hit[3][i])

    def _on_cov_motion(self, event) -> None:
        self._cov_hide_tip()
        if self._cov_pan["active"] or self._c_cov is None:
            return
        x, y = event.x, event.y
        try:
            self._cov_hover_id = self._c_cov.after(300, lambda: self._cov_show_tip(x, y))
        except tk.TclError:
            pass

    def _cov_hide_tip(self) -> None:
        c = self._c_cov
        if self._cov_hover_id is not None and c is not None:
            try:
                c.after_cancel(self._cov_hover_id)
            except tk.TclError:
                pass
        self._cov_hover_id = None
        if c is not None:
            c.delete("tip")

    def _cov_show_tip(self, x: float, y: float) -> None:
        """Mouse paused near a point: ring every point of that image and show
        which image / file / corner the point comes from."""
        self._cov_hover_id = None
        c, s1 = self._c_cov, self._stage1
        hit = self._cov_nearest(x, y)
        if c is None or s1 is None or hit is None or self._cov_view is None:
            return
        pos, r, col = hit
        cand = s1["selected"][pos]
        ox, oy, s = self._cov_view
        for px, py in cand["corners"].reshape(-1, 2):
            X, Y = ox + (px + 0.5) * s, oy + (py + 0.5) * s
            c.create_oval(X - 5, Y - 5, X + 5, Y + 5, outline="#ffff00", width=1.5, tags="tip")
        px, py = cand["corners"][r, col]
        status = "checked" if s1["included"][pos] else "UNCHECKED (not used)"
        text = (f"image #{pos + 1}   source # {cand['source_index']}   ({status})\n"
                f"{os.path.basename(cand.get('path') or '') or '(no path)'}\n"
                f"corner [r={r}, c={col}]   x={px:.2f}, y={py:.2f} px")
        item = c.create_text(x + 14, y + 14, text=text, anchor="nw", fill="#000000", font=("Arial", 8), tags="tip")
        x0, y0, x1, y1 = c.bbox(item)
        cw, ch = c.winfo_width(), c.winfo_height()
        dx = -(x1 - x0) - 28 if x1 + 4 > cw else 0            # keep the tooltip inside the canvas
        dy = -(y1 - y0) - 28 if y1 + 4 > ch else 0
        if dx or dy:
            c.move(item, dx, dy)
            x0, y0, x1, y1 = c.bbox(item)
        bg = c.create_rectangle(x0 - 4, y0 - 3, x1 + 4, y1 + 3, fill="#ffffe0", outline="#808080", tags="tip")
        c.tag_lower(bg, item)

    # ══ Matplotlib pan / zoom (fixed-scale drag, see the pan-vibration fix) ══
    def _connect_mpl_pan_zoom(self, canvas, ax_getter, pan_state: dict, on_dblclick=None) -> None:
        canvas.mpl_connect("scroll_event", lambda e: self._mpl_scroll(e, ax_getter(), canvas))
        canvas.mpl_connect("button_press_event", lambda e: self._mpl_press(e, ax_getter(), pan_state, on_dblclick))
        canvas.mpl_connect("motion_notify_event", lambda e: self._mpl_drag(e, ax_getter(), pan_state, canvas))
        canvas.mpl_connect("button_release_event", lambda _e: pan_state.update(active=False))

    @staticmethod
    def _mpl_press(event, ax, st: dict, on_dblclick) -> None:
        if ax is None or event.inaxes is not ax:
            return
        if getattr(event, "dblclick", False) and on_dblclick is not None and event.xdata is not None:
            on_dblclick(event.xdata, event.ydata)
            return
        if event.button != 1:
            return
        bbox = ax.bbox
        if bbox.width <= 0 or bbox.height <= 0:
            return
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        # Data-units-per-pixel captured ONCE here; the drag handler never
        # re-derives positions from event.xdata (which depends on the very
        # limits it is changing -- the cause of the old jitter).
        st.update(active=True, x_px=event.x, y_px=event.y, xlim0=xlim, ylim0=ylim,
                  x_scale=(xlim[1] - xlim[0]) / bbox.width, y_scale=(ylim[1] - ylim[0]) / bbox.height)

    @staticmethod
    def _mpl_drag(event, ax, st: dict, canvas) -> None:
        if ax is None or not st.get("active") or event.x is None or event.y is None:
            return
        dx, dy = event.x - st["x_px"], event.y - st["y_px"]
        (x0, x1), (y0, y1) = st["xlim0"], st["ylim0"]
        ax.set_xlim(x0 - dx * st["x_scale"], x1 - dx * st["x_scale"])
        ax.set_ylim(y0 - dy * st["y_scale"], y1 - dy * st["y_scale"])
        canvas.draw_idle()

    @staticmethod
    def _mpl_scroll(event, ax, canvas) -> None:
        if ax is None or event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return
        factor = 0.85 if event.button == "up" else 1.0 / 0.85
        key = (event.key or "").lower()
        zoom_x, zoom_y = True, True
        if "control" in key or "ctrl" in key:
            zoom_y = False          # Ctrl+wheel: horizontal only
        elif "shift" in key:
            zoom_x = False          # Shift+wheel: vertical only
        if zoom_x:
            x0, x1 = ax.get_xlim()
            ax.set_xlim(event.xdata + (x0 - event.xdata) * factor, event.xdata + (x1 - event.xdata) * factor)
        if zoom_y:
            y0, y1 = ax.get_ylim()
            ax.set_ylim(event.ydata + (y0 - event.ydata) * factor, event.ydata + (y1 - event.ydata) * factor)
        canvas.draw_idle()

    # ══ Per-image error plot ════════════════════════════════════════════
    def _current_source(self) -> int | None:
        s1 = self._stage1
        if s1 is None or not s1["selected"]:
            return None
        pos = min(max(self._preview_selected_pos, 0), len(s1["selected"]) - 1)
        return int(s1["selected"][pos]["source_index"])

    def _redraw_err(self) -> None:
        ax = self._ax_err
        if ax is None:
            return
        ax.clear()
        run = self._active_run()
        if run is None:
            ax.text(0.5, 0.5, "No calibration run yet", transform=ax.transAxes, ha="center", va="center",
                    color="#888888")
            ax.set_xticks([])
            ax.set_yticks([])
            self._canvas_err.draw_idle()
            return
        src = run["source_index"]
        err = run["per_image_error"]
        x = np.arange(len(err))
        cur = self._current_source()
        colors = ["#d62728" if int(s) == cur else "#6a3f9e" for s in src]
        ax.bar(x, err, color=colors, width=0.8)
        ax.axhline(run["rms"], color="#ff7f0e", ls="--", lw=1, label=f"overall RMS {run['rms']:.3f} px")
        step = max(1, int(math.ceil(len(x) / 40)))
        ax.set_xticks(x[::step])
        ax.set_xticklabels([str(int(s)) for s in src[::step]], rotation=90, fontsize=7)
        ax.set_xlabel("source #", fontsize=8)
        ax.set_ylabel("RMS reprojection error (px)", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        ax.legend(fontsize=7, loc="upper right")
        ax.set_title(f"Per-image reprojection error -- run {self._active_run_index + 1}", fontsize=9)
        ax.set_xlim(-0.6, len(x) - 0.4)
        self._canvas_err.draw_idle()

    def _on_err_dblclick(self, xdata, _ydata) -> None:
        run = self._active_run()
        if run is None:
            return
        k = int(round(xdata))
        if 0 <= k < len(run["source_index"]):
            pos = self._pos_for_source(int(run["source_index"][k]))
            if pos is not None:
                self._select_position(pos)

    # ══ Residual vectors (ONE image at a time) ══════════════════════════
    def _reset_res_view(self) -> None:
        self._res_saved_view = None
        self._res_plot_size = None
        self._redraw_res()

    def _redraw_res(self) -> None:
        ax = self._ax_res
        if ax is None:
            return
        if self._res_plot_size is not None:
            # Remember the user's pan/zoom so switching images keeps the same region.
            self._res_saved_view = (ax.get_xlim(), ax.get_ylim(), self._res_plot_size)
        ax.clear()
        self._res_plot_size = None
        run = self._active_run()

        def message(text):
            ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color="#888888")
            ax.set_xticks([])
            ax.set_yticks([])
            if self._residual_label_var is not None:
                self._residual_label_var.set("")
            self._canvas_res.draw_idle()

        if run is None:
            message("No calibration run yet")
            return
        src = self._current_source()
        if src is None:
            message("No image selected")
            return
        hits = np.nonzero(run["source_index"] == src)[0]
        if hits.size == 0:
            message(f"source # {src} was not used by the active run\n"
                    "(it was unchecked when that run was computed)")
            return
        k = int(hits[0])
        try:
            mag = float(self._resel_mag_var.get())
        except (tk.TclError, ValueError):
            mag = 50.0
        mag = mag if mag > 0 else 1.0
        w, h = run["img_size"]
        pts = run["image_points"][k].reshape(-1, 2)      # OpenCV corner order, same as residuals
        res = run["residuals"][k]
        norms = np.linalg.norm(res, axis=1)
        ax.plot([0, w, w, 0, 0], [0, 0, h, h, 0], color="#bbbbbb", lw=1)
        ax.plot(pts[:, 0], pts[:, 1], ".", color="#444444", ms=3)
        ax.quiver(pts[:, 0], pts[:, 1], res[:, 0] * mag, res[:, 1] * mag, norms, cmap="plasma",
                  angles="xy", scale_units="xy", scale=1, width=0.003)
        saved = self._res_saved_view
        if saved is not None and tuple(saved[2]) == (w, h):
            ax.set_xlim(*saved[0])
            ax.set_ylim(*saved[1])
        else:
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)                      # image convention: y down
        ax.set_aspect("equal", adjustable="datalim")
        self._res_plot_size = (w, h)
        ax.tick_params(labelsize=7)
        ax.set_title(f"Residuals x{mag:g} -- source # {src}", fontsize=9)
        if self._residual_label_var is not None:
            self._residual_label_var.set(
                f"source # {src}: RMS {run['per_image_error'][k]:.4f} px, max {norms.max():.4f} px, "
                f"mean {norms.mean():.4f} px  (arrow = projected - detected)")
        self._canvas_res.draw_idle()

    # ══ 3D camera-pose view ═════════════════════════════════════════════
    def _v3_matrix(self) -> np.ndarray:
        yaw, pitch = self._v3["yaw"], self._v3["pitch"]
        cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
        rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
        rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
        return rx @ rz

    def _v3_project(self, P, cw: int, ch: int, extent: float) -> np.ndarray:
        M = self._v3_matrix()
        Q = (np.asarray(P, dtype=np.float64).reshape(-1, 3) - np.asarray(self._v3["target"])) @ M.T
        try:
            persp = float(self._persp_var.get()) if self._persp_var is not None else 0.3
        except tk.TclError:
            persp = 0.3
        f = 1.0 / np.maximum(0.15, 1.0 - persp * Q[:, 2] / (2.5 * extent))
        scale = self._v3["zoom"] * 0.42 * min(cw, ch) / extent
        self._v3["last_scale"] = scale
        return np.stack([cw / 2.0 + Q[:, 0] * scale * f, ch / 2.0 - Q[:, 1] * scale * f], axis=1)

    def _reset_3d_view(self) -> None:
        self._v3.update(yaw=-0.9, pitch=0.7, zoom=1.0)
        self._v3_centered = False
        self._redraw_3d()

    def _redraw_3d(self) -> None:
        c = self._c3d
        if c is None:
            return
        try:
            cw, ch = c.winfo_width(), c.winfo_height()
        except tk.TclError:
            return
        if cw < 20 or ch < 20:
            return
        c.delete("all")
        run = self._active_run()
        if run is None:
            c.create_text(cw / 2, ch / 2, text="No calibration run yet", fill="#888888")
            return
        grid = run["object_points"][0]                  # (nrow, ncol, 3), board frame, OpenCV row-major
        nrow, ncol = grid.shape[:2]
        rot = [cv2.Rodrigues(np.asarray(r, dtype=np.float64).reshape(3, 1))[0] for r in run["rvecs"]]
        tvecs = [np.asarray(t, dtype=np.float64).reshape(3) for t in run["tvecs"]]
        camera_fixed = self._pose_frame_var.get() == "camera"

        # Camera frustum rays in CAMERA coordinates (image corners at depth d).
        w, h = run["img_size"]
        fx, fy = float(run["cmat"][0, 0]), float(run["cmat"][1, 1])
        cx, cy = float(run["cmat"][0, 2]), float(run["cmat"][1, 2])
        board_diag = float(np.linalg.norm(grid[-1, -1] - grid[0, 0])) or 1.0
        try:
            fscale = float(self._frustum_scale_var.get())
        except tk.TclError:
            fscale = 1.0
        d = 0.35 * board_diag * fscale
        rays = np.array([[(u - cx) / fx, (v - cy) / fy, 1.0] for u, v in ((0, 0), (w, 0), (w, h), (0, h))]) * d
        sq = float(np.linalg.norm(grid[0, 1] - grid[0, 0])) if ncol > 1 else board_diag * 0.05

        if camera_fixed:
            # Display-only change of frame: X_cam = R @ X_board + t. Camera
            # coordinates (x right, y DOWN, z forward) are shown as
            # (x, z, -y) so the optical axis is horizontal and image-up is up.
            disp = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
            boards = [(grid.reshape(-1, 3) @ R.T + t).reshape(grid.shape) @ disp.T for R, t in zip(rot, tvecs)]
            frustum = rays @ disp.T
            all_pts = np.vstack([b.reshape(-1, 3) for b in boards] + [np.zeros((1, 3)), frustum])
        else:
            centers = [-R.T @ t for R, t in zip(rot, tvecs)]
            all_pts = np.vstack([grid.reshape(-1, 3), np.array(centers)])

        lo, hi = all_pts.min(axis=0), all_pts.max(axis=0)
        extent = float(max(np.max(hi - lo), 1e-9))
        self._v3_extent = extent
        if not self._v3_centered:
            self._v3["target"] = ((lo + hi) / 2.0).tolist()
            self._v3_centered = True

        def P(pts):
            return self._v3_project(pts, cw, ch, extent)

        def axes(origin, dirs, length):
            for vec, color in zip(dirs, ("#d62728", "#2ca02c", "#1f77b4")):
                seg = P(np.array([origin, origin + vec * length]))
                c.create_line(*seg.ravel().tolist(), fill=color, width=2, arrow=tk.LAST)

        cur = self._current_source()
        if camera_fixed:
            # One camera at the origin, one board per image.
            pk = P(frustum)
            pc = P(np.zeros((1, 3)))[0]
            for q in pk:
                c.create_line(pc[0], pc[1], q[0], q[1], fill="#555555", width=2)
            c.create_polygon(*pk.ravel().tolist(), outline="#555555", fill="", width=2)
            axes(np.zeros(3), np.eye(3) @ disp.T, 2.0 * sq)
            order = [k for k in range(len(boards)) if int(run["source_index"][k]) != cur]
            order += [k for k in range(len(boards)) if int(run["source_index"][k]) == cur]   # current on top
            for k in order:
                g = boards[k]
                src = int(run["source_index"][k])
                is_cur = src == cur
                color, width = ("#d62728", 2) if is_cur else ("#6a3f9e", 1)
                if is_cur:                                   # full grid only for the current board
                    for r in range(nrow):
                        c.create_line(*P(g[r, :, :]).ravel().tolist(), fill="#f0b0b0")
                    for col in range(ncol):
                        c.create_line(*P(g[:, col, :]).ravel().tolist(), fill="#f0b0b0")
                outline = P(np.array([g[0, 0], g[0, -1], g[-1, -1], g[-1, 0], g[0, 0]]))
                c.create_line(*outline.ravel().tolist(), fill=color, width=width)
                p0 = P(g[0, 0][None])[0]                     # board origin corner [0, 0]
                c.create_oval(p0[0] - 2.5, p0[1] - 2.5, p0[0] + 2.5, p0[1] + 2.5, fill=color, outline="")
                c.create_text(p0[0] + 4, p0[1] - 4, text=str(src), fill=color, font=("Arial", 7), anchor="sw")
            c.create_text(8, ch - 8, anchor="sw", fill="#777777", font=("Arial", 8),
                          text="CAMERA fixed (display only; rvecs/tvecs unchanged): X red (right), Y green (down), "
                               "Z blue (optical axis); dot = board corner [0,0]; labels = source #; "
                               "red = current image")
            return

        # Board grid: one line per chessboard row (constant y) and per column (constant x).
        for r in range(nrow):
            line = P(grid[r, :, :])
            c.create_line(*line.ravel().tolist(), fill="#c8c8c8")
        for col in range(ncol):
            line = P(grid[:, col, :])
            c.create_line(*line.ravel().tolist(), fill="#c8c8c8")
        outline = P(np.array([grid[0, 0], grid[0, -1], grid[-1, -1], grid[-1, 0], grid[0, 0]]))
        c.create_line(*outline.ravel().tolist(), fill="#555555", width=2)
        axes(grid[0, 0], np.eye(3), 3.0 * sq)

        # Camera frustums.
        for k, (R, C) in enumerate(zip(rot, centers)):
            corners = C + rays @ R                      # = C + R^T @ ray, row-wise
            src = int(run["source_index"][k])
            is_cur = src == cur
            color, width = ("#d62728", 2) if is_cur else ("#6a3f9e", 1)
            pc = P(C[None])[0]
            pk = P(corners)
            for q in pk:
                c.create_line(pc[0], pc[1], q[0], q[1], fill=color, width=width)
            c.create_polygon(*pk.ravel().tolist(), outline=color, fill="", width=width)
            c.create_text(pc[0] + 6, pc[1] - 6, text=str(src), fill=color, font=("Arial", 7), anchor="sw")
        c.create_text(8, ch - 8, anchor="sw", fill="#777777", font=("Arial", 8),
                      text="Board frame: X red (along a row), Y green (down the columns), Z blue; "
                           "labels = source #; red = current image")

    def _on_3d_press(self, event, button: int) -> None:
        self._v3.update(dragging=True, button=button, last_x=event.x, last_y=event.y)

    def _on_3d_drag(self, event) -> None:
        st = self._v3
        if not st["dragging"]:
            return
        dx, dy = event.x - st["last_x"], event.y - st["last_y"]
        st["last_x"], st["last_y"] = event.x, event.y
        if st["button"] == 1:
            st["yaw"] += dx * 0.01
            st["pitch"] = float(min(max(st["pitch"] + dy * 0.01, -1.55), 1.55))
        else:
            scale = st.get("last_scale") or 1.0
            view_delta = np.array([dx / scale, -dy / scale, 0.0])
            world_delta = self._v3_matrix().T @ view_delta
            st["target"] = (np.asarray(st["target"]) - world_delta).tolist()
        self._redraw_3d()

    def _on_3d_wheel(self, event):
        steps = self._wheel_steps(event)
        if steps:
            self._v3["zoom"] = float(min(max(self._v3["zoom"] * (1.15 ** steps), 0.05), 50.0))
            self._redraw_3d()
        return "break"

    # ══ Undistort image ═════════════════════════════════════════════════
    def _undistort_tab_visible(self) -> bool:
        nb, tab = self._right_nb, self._und_tab
        if nb is None or tab is None:
            return False
        try:
            return nb.select() == str(tab)
        except tk.TclError:
            return False

    def _on_undistort_apply(self) -> None:
        v = self._und_vars
        if v is None:
            return
        try:
            alpha = float(v["alpha"].get())
        except ValueError:
            alpha = None
        if alpha is None or not (0.0 <= alpha <= 1.0):
            messagebox.showerror("Undistort", "alpha must be a number between 0 and 1.", parent=self._dlg_parent())
            return
        self._und_cfg = {"newk_mode": v["newk_mode"].get(), "alpha": alpha, "center_pp": bool(v["center_pp"].get()),
                         "show_corners": bool(v["show_corners"].get()), "show_lines": bool(v["show_lines"].get()),
                         "show_roi": bool(v["show_roi"].get())}
        self._redraw_undistort()

    def _undistort_maps(self, run: dict) -> tuple:
        """(maps key, map1, map2, newK, valid roi) for the active run and the
        applied settings. initUndistortRectifyMap + remap gives exactly the
        cv2.undistort result, but the tables are built once per setting and
        reused for every image, so switching images is fast."""
        cfg = self._und_cfg
        K = np.asarray(run["cmat"], dtype=np.float64)
        d = np.asarray(run["dvec"], dtype=np.float64).ravel()
        w, h = (int(x) for x in run["img_size"])
        key = (K.tobytes(), d.tobytes(), w, h, cfg["newk_mode"], float(cfg["alpha"]), bool(cfg["center_pp"]))
        hit = self._und_maps.get(key)
        if hit is not None:
            return (key,) + hit
        roi = None
        if cfg["newk_mode"] == "optimal":
            newK, roi = cv2.getOptimalNewCameraMatrix(K, d, (w, h), float(cfg["alpha"]), (w, h),
                                                      centerPrincipalPoint=bool(cfg["center_pp"]))
        else:
            newK = K.copy()
        m1, m2 = cv2.initUndistortRectifyMap(K, d, None, newK, (w, h), cv2.CV_16SC2)
        self._und_maps = {key: (m1, m2, newK, roi)}          # keep only the current setting's tables
        return key, m1, m2, newK, roi

    def _undistorted_display(self, path: str, maps_key, m1, m2):
        """(display rgb, display scale) of the undistorted image, from the LRU
        cache or computed now. None if the file cannot be read."""
        ckey = (path, maps_key)
        hit = self._und_cache.pop(ckey, None)
        if hit is not None:
            self._und_cache[ckey] = hit                      # move to most-recent
            return hit, True
        rgb = _read_rgb(path) if path else None
        if rgb is None:
            return None, False
        und = cv2.remap(rgb, m1, m2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        h, w = und.shape[:2]
        f = 1.0
        if max(h, w) > PREVIEW_MAX_SIDE:
            f = PREVIEW_MAX_SIDE / float(max(h, w))
            und = cv2.resize(und, (max(1, int(round(w * f))), max(1, int(round(h * f)))),
                             interpolation=cv2.INTER_AREA)
        self._und_cache[ckey] = (und, f)
        while len(self._und_cache) > UNDISTORT_CACHE_SIZE:
            self._und_cache.pop(next(iter(self._und_cache)))
        return (und, f), False

    def _redraw_undistort(self) -> None:
        c = self._c_und
        if c is None or not self._undistort_tab_visible():
            return
        try:
            cw, ch = c.winfo_width(), c.winfo_height()
        except tk.TclError:
            return
        if cw < 20 or ch < 20:
            return
        c.delete("all")
        s1, run = self._stage1, self._active_run()
        msg = None
        if s1 is None or not s1["selected"]:
            msg = "No selected image yet.\nRun 'Detect & Select'."
        elif run is None:
            msg = "No calibration run yet.\nRun a calibration first -- the ACTIVE run's K and\ndistortion coefficients are used."
        if msg:
            c.create_text(cw / 2, ch / 2, fill="#999999", justify="center", width=cw - 40, text=msg)
            if self._und_label_var is not None:
                self._und_label_var.set("")
            return
        pos = min(max(self._preview_selected_pos, 0), len(s1["selected"]) - 1)
        cand = s1["selected"][pos]
        cfg = self._und_cfg

        parent = self._dlg_parent()
        t0 = time.perf_counter()
        try:
            parent.configure(cursor="watch")
            parent.update_idletasks()
        except tk.TclError:
            pass
        try:
            maps_key, m1, m2, newK, roi = self._undistort_maps(run)
            res, cached = self._undistorted_display(cand.get("path") or "", maps_key, m1, m2)
        except cv2.error as e:
            c.create_text(cw / 2, ch / 2, fill="#ff8080", width=cw - 40, text=f"cv2 error:\n{e}")
            return
        finally:
            try:
                parent.configure(cursor="")
            except tk.TclError:
                pass
        dt = time.perf_counter() - t0

        w, h = (int(x) for x in run["img_size"])
        if res is not None:
            rgb, f = res
            pw, ph = rgb.shape[1], rgb.shape[0]
        else:
            rgb, f, pw, ph = None, 1.0, w, h
        if self._und_view is None or not self._und_user_view:
            sc = 0.98 * min(cw / pw, ch / ph)
            self._und_view = ((cw - pw * sc) / 2.0, (ch - ph * sc) / 2.0, sc)
        ox, oy, sc = self._und_view

        if rgb is not None:
            x0 = max(0, int(math.floor(-ox / sc)))
            y0 = max(0, int(math.floor(-oy / sc)))
            x1 = min(pw, int(math.ceil((cw - ox) / sc)))
            y1 = min(ph, int(math.ceil((ch - oy) / sc)))
            if x1 > x0 and y1 > y0:
                key = (id(rgb), x0, y0, x1, y1, round(sc, 6))
                if key != self._und_photo_key:
                    crop = rgb[y0:y1, x0:x1]
                    dw = max(1, int(round((x1 - x0) * sc)))
                    dh = max(1, int(round((y1 - y0) * sc)))
                    disp = cv2.resize(crop, (dw, dh), interpolation=cv2.INTER_NEAREST if sc >= 1.0 else cv2.INTER_AREA)
                    self._und_photo = ImageTk.PhotoImage(Image.fromarray(disp), master=c)
                    self._und_photo_key = key
                c.create_image(ox + x0 * sc, oy + y0 * sc, image=self._und_photo, anchor="nw")
        else:
            c.create_rectangle(ox, oy, ox + pw * sc, oy + ph * sc, fill="#333333", outline="#666666")
            c.create_text(cw / 2, 16, fill="#dddddd", text="Image file not available")

        def to_canvas(x, y):                                  # full-res undistorted px -> canvas
            return ox + (x + 0.5) * f * sc, oy + (y + 0.5) * f * sc

        if cfg["show_roi"] and roi is not None and roi[2] > 0 and roi[3] > 0:
            rx, ry, rw, rh = roi
            ax0, ay0 = to_canvas(rx - 0.5, ry - 0.5)
            ax1, ay1 = to_canvas(rx + rw - 0.5, ry + rh - 0.5)
            c.create_rectangle(ax0, ay0, ax1, ay1, outline="#00e0ff", dash=(5, 3))
        if cfg["show_corners"] or cfg["show_lines"]:
            d = np.asarray(run["dvec"], dtype=np.float64).ravel()
            pts = cv2.undistortPoints(cand["corners"].reshape(-1, 1, 2).astype(np.float64),
                                      np.asarray(run["cmat"], dtype=np.float64), d, P=newK).reshape(-1, 2)
            nrow, ncol = cand["corners"].shape[:2]
            grid = pts.reshape(nrow, ncol, 2)
            if cfg["show_lines"]:
                # Straight lines between the END corners of each row / column: if the
                # undistortion is right, every corner lies on them.
                for a, b in [(grid[r, 0], grid[r, -1]) for r in range(nrow)] + \
                            [(grid[0, k], grid[-1, k]) for k in range(ncol)]:
                    c.create_line(*to_canvas(*a), *to_canvas(*b), fill="#ffd000", width=1)
            if cfg["show_corners"]:
                n = len(pts)
                for k, (x, y) in enumerate(pts):
                    X, Y = to_canvas(x, y)
                    t = k / max(1, n - 1)
                    color = f"#{int(255 * t):02x}{int(220 * (1 - t)):02x}30"
                    r = 5 if k == 0 else 3
                    c.create_oval(X - r, Y - r, X + r, Y + r, outline=color, width=2 if k == 0 else 1.5)

        if self._und_label_var is not None:
            name = os.path.basename(cand.get("path") or "") or "(no path)"
            mode = (f"getOptimalNewCameraMatrix(alpha={cfg['alpha']:g}"
                    f"{', centerPP' if cfg['center_pp'] else ''})" if cfg["newk_mode"] == "optimal" else "newK = K")
            self._und_label_var.set(
                f"Image {pos + 1}/{len(s1['selected'])} | source # {cand['source_index']} | {name} | "
                f"active run {self._active_run_index + 1} | {mode} | "
                f"{'cached' if cached else f'undistorted in {dt:.2f} s'}  "
                f"(newK: fx={newK[0, 0]:.1f} fy={newK[1, 1]:.1f} cx={newK[0, 2]:.1f} cy={newK[1, 2]:.1f})")

    def _reset_undistort_view(self) -> None:
        self._und_user_view = False
        self._redraw_undistort()

    def _on_und_press(self, event) -> None:
        if self._und_view is None:
            return
        ox, oy, _s = self._und_view
        self._und_pan.update(active=True, x_px=event.x, y_px=event.y, ox0=ox, oy0=oy)

    def _on_und_drag(self, event) -> None:
        st = self._und_pan
        if not st["active"] or self._und_view is None:
            return
        self._und_view = (st["ox0"] + event.x - st["x_px"], st["oy0"] + event.y - st["y_px"], self._und_view[2])
        self._und_user_view = True
        self._redraw_undistort()

    def _on_und_wheel(self, event):
        if self._und_view is None:
            return "break"
        steps = self._wheel_steps(event)
        if steps == 0:
            return "break"
        ox, oy, s = self._und_view
        new_s = float(min(max(s * (IMG_ZOOM_FACTOR ** steps), IMG_MIN_SCALE), IMG_MAX_SCALE))
        k = new_s / s
        ex, ey = float(event.x), float(event.y)
        self._und_view = (ex - (ex - ox) * k, ey - (ey - oy) * k, new_s)
        self._und_user_view = True
        self._redraw_undistort()
        return "break"

    # ══ Stage 2: calibration runs ═══════════════════════════════════════
    def _on_guess_intrinsics(self) -> None:
        size = self._stage1["img_size"] if self._stage1 is not None else None
        if size is None and self._files:
            img = _read_gray(self._files[0])
            if img is not None:
                size = (img.shape[1], img.shape[0])
        if size is None:
            messagebox.showinfo("Guess", "No image size known yet -- connect files or run detection.",
                                parent=self._dlg_parent())
            return
        w, h = size
        f = float(max(w, h))
        v = self._calib_vars
        v["guess_fx"].set(f"{f:.1f}")
        v["guess_fy"].set(f"{f:.1f}")
        v["guess_cx"].set(f"{(w - 1) / 2.0:.1f}")
        v["guess_cy"].set(f"{(h - 1) / 2.0:.1f}")
        v["use_intrinsic_guess"].set(True)

    def _on_run_calibration(self) -> None:
        filtered = self._included_stage1_arrays()
        if filtered is None:
            messagebox.showinfo("Calibration", "Run 'Detect & Select' (or Load) first.", parent=self._dlg_parent())
            return
        n = filtered["object_points"].shape[0]
        if n < 3:
            messagebox.showerror("Calibration", f"Need at least 3 checked images (have {n}).",
                                 parent=self._dlg_parent())
            return
        opts, errs = self._collect_calib_opts()
        if errs:
            messagebox.showerror("Calibration", "\n".join(errs), parent=self._dlg_parent())
            return
        if n > MANY_IMAGES_WARNING and not messagebox.askokcancel(
                "Calibration",
                f"{n} images are checked (more than {MANY_IMAGES_WARNING}).\n\n"
                "calibrateCamera time grows quickly with the number of images (and with the "
                "rational / thin-prism / tilted models), so this may take a long time, and the window "
                "will not respond until it finishes. 10-30 well-spread images are usually enough.\n\n"
                "Continue anyway?", icon="warning", parent=self._dlg_parent()):
            return
        img_size = tuple(int(v) for v in filtered["img_size"])
        parent = self._dlg_parent()
        try:
            parent.configure(cursor="watch")
            parent.update_idletasks()
        except tk.TclError:
            pass
        try:
            run = run_calibration(filtered["object_points"], filtered["image_points"], img_size, opts)
        except Exception as e:
            messagebox.showerror("Calibration", f"{type(e).__name__}: {e}", parent=parent)
            return
        finally:
            try:
                parent.configure(cursor="")
            except tk.TclError:
                pass
        # Snapshot of exactly what this run used -- later checkbox changes never alter it.
        run.update(object_points=filtered["object_points"].copy(), image_points=filtered["image_points"].copy(),
                   source_index=filtered["source_index"].copy(), img_size=img_size)
        self._runs.append(run)
        self._active_run_index = len(self._runs) - 1
        self._on_active_run_changed()

    def _on_active_run_changed(self) -> None:
        self._loo_rows = None
        self._v3_centered = False
        self._mark_unsaved()
        self._notify_outputs_changed()
        self._refresh_all_views()

    def _on_set_active(self) -> None:
        tree = self._runs_tree
        if tree is None:
            return
        sel = tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        if idx != self._active_run_index:
            self._active_run_index = idx
            self._on_active_run_changed()

    def _on_delete_run(self) -> None:
        tree = self._runs_tree
        if tree is None or not tree.selection():
            return
        idx = int(tree.selection()[0])
        if not (0 <= idx < len(self._runs)):
            return
        del self._runs[idx]
        if self._active_run_index is not None:
            if idx == self._active_run_index:
                self._active_run_index = len(self._runs) - 1 if self._runs else None
            elif idx < self._active_run_index:
                self._active_run_index -= 1
        self._on_active_run_changed()

    def _refresh_runs_tree(self) -> None:
        tree = self._runs_tree
        if tree is None:
            return
        tree.delete(*tree.get_children())
        for i, r in enumerate(self._runs):
            d = r["dvec"]
            dist = tuple(f"{d[j]:.4g}" if j < d.size else "-" for j in range(len(DIST_NAMES)))
            tree.insert("", "end", iid=str(i), values=(
                "*" if i == self._active_run_index else "", r["object_points"].shape[0], f"{r['rms']:.4f}",
                f"{r['cmat'][0, 0]:.2f}", f"{r['cmat'][1, 1]:.2f}", f"{r['cmat'][0, 2]:.2f}",
                f"{r['cmat'][1, 2]:.2f}") + dist + (r.get("label", ""),))
        if self._active_run_index is not None and tree.exists(str(self._active_run_index)):
            tree.selection_set(str(self._active_run_index))

    def _refresh_active_summary(self) -> None:
        var = self._active_summary_var
        if var is None:
            return
        run = self._active_run()
        if run is None:
            var.set("No active run.")
            return
        K, d = run["cmat"], run["dvec"]
        dist = "  ".join(f"{DIST_NAMES[i] if i < len(DIST_NAMES) else i}={d[i]:.5g}" for i in range(d.size))
        var.set(f"Active run {self._active_run_index + 1}: RMS {run['rms']:.4f} px, "
                f"{run['object_points'].shape[0]} images\n"
                f"flags: {flags_summary(run['opts'])}\n"
                f"fx={K[0, 0]:.3f}  fy={K[1, 1]:.3f}\ncx={K[0, 2]:.3f}  cy={K[1, 2]:.3f}\n{dist}")

    # ══ Stage 2: leave-one-out sensitivity ══════════════════════════════
    def _set_loo_running(self, running: bool) -> None:
        self._set_state(self._loo_btn, tk.DISABLED if running else tk.NORMAL)
        self._set_state(self._stop_loo_btn, tk.NORMAL if running else tk.DISABLED)
        self._set_state(self._calib_btn, tk.DISABLED if running else tk.NORMAL)

    def _on_run_loo(self) -> None:
        if self._loo_state is not None and not self._loo_state["finished"]:
            return
        run = self._active_run()
        if run is None:
            messagebox.showinfo("Sensitivity", "Run a calibration first.", parent=self._dlg_parent())
            return
        n = run["object_points"].shape[0]
        if n < 4:
            messagebox.showerror("Sensitivity", "Leave-one-out needs at least 4 images in the active run.",
                                 parent=self._dlg_parent())
            return
        if n > MANY_IMAGES_WARNING and not messagebox.askokcancel(
                "Sensitivity",
                f"The active run has {n} images (more than {MANY_IMAGES_WARNING}). Leave-one-out runs "
                f"{n} full calibrations of {n - 1} images each, so this may take a long time "
                "(it runs in the background and can be stopped).\n\nContinue?",
                icon="warning", parent=self._dlg_parent()):
            return
        state = {"cancel": threading.Event(), "done": 0, "total": n, "rows": None,
                 "error": None, "finished": False}
        job = {"object_points": run["object_points"], "image_points": run["image_points"],
               "img_size": tuple(run["img_size"]), "opts": run["opts"]}
        self._loo_state = state
        self._loo_run_ref = run
        threading.Thread(target=_run_loo_worker, args=(job, state), daemon=True).start()
        self._set_loo_running(True)
        self._schedule_loo_poll()

    def _on_stop_loo(self) -> None:
        if self._loo_state is not None and not self._loo_state["finished"]:
            self._loo_state["cancel"].set()

    def _schedule_loo_poll(self) -> None:
        if self._loo_poll_id is None and not self._destroyed:
            try:
                self._loo_poll_id = self.canvas.after(150, self._poll_loo)
            except tk.TclError:
                pass

    def _poll_loo(self) -> None:
        self._loo_poll_id = None
        st = self._loo_state
        if st is None or self._destroyed:
            return
        if self._loo_progress_var is not None:
            try:
                self._loo_progress_var.set(100.0 * st["done"] / max(1, st["total"]))
            except tk.TclError:
                pass
        if not st["finished"]:
            self._schedule_loo_poll()
            return
        self._set_loo_running(False)
        if st["error"]:
            messagebox.showerror("Sensitivity", st["error"], parent=self._dlg_parent())
            return
        # Discard results if the active run changed while the job was running.
        if self._loo_run_ref is self._active_run():
            self._loo_rows = st["rows"] or []
        self._refresh_loo_tree()

    def _refresh_loo_tree(self) -> None:
        tree = self._loo_tree
        if tree is None:
            return
        tree.delete(*tree.get_children())
        run, rows = self._active_run(), self._loo_rows
        if run is None or not rows:
            return
        K = run["cmat"]
        base = {"rms": run["rms"], "fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2]}
        w, h = run["img_size"]
        scores = []
        for r in rows:
            if "error" in r:
                scores.append(float("nan"))
                continue
            scores.append(max(abs(r["fx"] - base["fx"]) / base["fx"], abs(r["fy"] - base["fy"]) / base["fy"],
                              abs(r["cx"] - base["cx"]) / w, abs(r["cy"] - base["cy"]) / h))
        finite = [s for s in scores if math.isfinite(s)]
        med = float(np.median(finite)) if finite else 0.0
        for i, (r, score) in enumerate(zip(rows, scores)):
            src = int(run["source_index"][r["excluded_index"]])
            if "error" in r:
                tree.insert("", "end", iid=str(i), values=(src, "failed", r["error"]), tags=("error",))
                continue
            tags = ("outlier",) if med > 0 and score > 3.0 * med else ()
            dist = []
            for j, name in enumerate(DIST_NAMES):
                val, ref = r.get(name, float("nan")), dist_value(run["dvec"], j)
                if math.isfinite(val):
                    dist += [f"{val:.4g}", f"{val - ref:+.3g}" if math.isfinite(ref) else "-"]
                else:
                    dist += ["-", "-"]
            tree.insert("", "end", iid=str(i), tags=tags, values=(
                src, f"{r['rms']:.4f}", f"{r['rms'] - base['rms']:+.4f}",
                f"{r['fx']:.2f}", f"{r['fx'] - base['fx']:+.3f}", f"{r['fy']:.2f}", f"{r['fy'] - base['fy']:+.3f}",
                f"{r['cx']:.2f}", f"{r['cx'] - base['cx']:+.3f}", f"{r['cy']:.2f}", f"{r['cy'] - base['cy']:+.3f}")
                + tuple(dist))

    def _on_loo_row_selected(self, _event=None) -> None:
        tree, run = self._loo_tree, self._active_run()
        if tree is None or run is None or not tree.selection() or not self._loo_rows:
            return
        i = int(tree.selection()[0])
        if 0 <= i < len(self._loo_rows):
            src = int(run["source_index"][self._loo_rows[i]["excluded_index"]])
            pos = self._pos_for_source(src)
            if pos is not None and pos != self._preview_selected_pos:
                self._select_position(pos)

    # ══ Load / Save as (.npz) ═══════════════════════════════════════════
    def _on_save_as(self) -> None:
        if self._stage1 is None:
            messagebox.showinfo("Save as", "Nothing to save yet -- run 'Detect & Select' first.",
                                parent=self._dlg_parent())
            return
        path = filedialog.asksaveasfilename(parent=self._dlg_parent(), title="Save calibration data",
                                            defaultextension=".npz", initialfile="chessboard_calibration.npz",
                                            filetypes=[("Calibration files", "*.npz")])
        if not path:
            return
        final = path if path.lower().endswith(".npz") else path + ".npz"
        npz_dir = os.path.dirname(os.path.abspath(final))
        s1 = self._stage1
        paths_abs = [os.path.abspath(c["path"]) if c.get("path") else "" for c in s1["selected"]]
        paths_rel = []
        for p in paths_abs:
            try:
                paths_rel.append(os.path.relpath(p, npz_dir) if p else "")
            except ValueError:            # different drive on Windows
                paths_rel.append("")
        meta = {"ncol": int(s1["ncol"]), "nrow": int(s1["nrow"]), "square_size": float(s1["square_size"]),
                "img_size": [int(v) for v in s1["img_size"]], "paths": paths_abs, "paths_rel": paths_rel,
                "layout": CORNER_LAYOUT}
        arrays = {"settings_json": np.array(json.dumps(self._settings_dict())),
                  "meta_json": np.array(json.dumps(meta)),
                  "object_points": s1["object_points"], "image_points": s1["image_points"],
                  "sharpness": s1["sharpness"], "source_index": s1["source_index"],
                  "included": s1["included"].astype(bool),
                  "coverage": np.array([c["coverage"] for c in s1["selected"]], dtype=np.float64),
                  "center_dist": np.array([c["center_dist"] for c in s1["selected"]], dtype=np.float64)}
        run = self._active_run()
        if run is not None:
            arrays.update(active_cmat=run["cmat"], active_dvec=run["dvec"], active_rvecs=run["rvecs"],
                          active_tvecs=run["tvecs"], active_per_image_error=run["per_image_error"],
                          active_residuals=run["residuals"], active_rms=np.array(run["rms"]),
                          active_flags=np.array(run["flags"]),
                          active_opts_json=np.array(json.dumps(run["opts"])),
                          active_object_points=run["object_points"], active_image_points=run["image_points"],
                          active_source_index=run["source_index"])
        try:
            np.savez_compressed(final, **arrays)
        except Exception as e:
            messagebox.showerror("Save as", f"Cannot save the file:\n{e}", parent=self._dlg_parent())
            return
        self._stage1_path, self._stage1_saved = os.path.abspath(final), True
        self._refresh_file_label()

    def _on_load(self) -> None:
        path = filedialog.askopenfilename(parent=self._dlg_parent(), title="Load calibration data",
                                          filetypes=[("Calibration files", "*.npz"), ("All files", "*.*")])
        if not path:
            return
        try:
            self._load_result_file(path)
        except Exception as e:
            messagebox.showerror("Load", f"Cannot load the file:\n{type(e).__name__}: {e}",
                                 parent=self._dlg_parent())
            return
        self._stage1_path, self._stage1_saved = os.path.abspath(path), True
        self._notify_outputs_changed()
        self._refresh_all_views()

    def _resolve_saved_path(self, abs_path: str, rel_path: str, npz_dir: str, source_index: int) -> str:
        """Find the image file on THIS computer: saved absolute path, then
        the path relative to the .npz, then the connected 'files' input."""
        if abs_path and os.path.isfile(abs_path):
            return abs_path
        if rel_path:
            cand = os.path.normpath(os.path.join(npz_dir, rel_path))
            if os.path.isfile(cand):
                return cand
        if self._files and 0 <= source_index < len(self._files):
            return self._files[source_index]
        return ""

    def _load_result_file(self, path: str) -> None:
        """Load a .npz written by _on_save_as. Older files (without the
        'included' / coverage / path fields, and/or with the legacy
        (ncol, nrow) corner layout) are accepted and converted. Does not
        require the inspector to be open."""
        self._init_state()
        npz_dir = os.path.dirname(os.path.abspath(path))
        with np.load(path, allow_pickle=False) as z:
            files = set(z.files)
            settings = json.loads(z["settings_json"].item())
            meta = json.loads(z["meta_json"].item())
            obj = z["object_points"].astype(np.float64)
            img = z["image_points"].astype(np.float64)
            sharp = z["sharpness"].astype(np.float64)
            src = z["source_index"].astype(np.int64)
            n = obj.shape[0]
            included = z["included"].astype(bool) if "included" in files else np.ones(n, dtype=bool)
            coverage = z["coverage"] if "coverage" in files else np.full(n, np.nan)
            centre = z["center_dist"] if "center_dist" in files else np.full(n, np.nan)
            run = None
            if "active_cmat" in files:
                opts = json.loads(z["active_opts_json"].item())
                if "active_object_points" in files:
                    r_obj, r_img = z["active_object_points"], z["active_image_points"]
                    r_src = z["active_source_index"].astype(np.int64)
                else:                       # old format: the run used every stored image
                    r_obj, r_img, r_src = obj, img, src
                run = {"rms": float(z["active_rms"]), "cmat": z["active_cmat"], "dvec": z["active_dvec"].ravel(),
                       "rvecs": z["active_rvecs"], "tvecs": z["active_tvecs"],
                       "per_image_error": z["active_per_image_error"], "residuals": z["active_residuals"],
                       "flags": int(z["active_flags"]), "opts": opts,
                       "img_size": tuple(int(v) for v in meta["img_size"]),
                       "label": "loaded  " + flags_summary(opts),
                       "object_points": np.array(r_obj, dtype=np.float64),
                       "image_points": np.array(r_img, dtype=np.float64), "source_index": r_src}

        ncol_m, nrow_m = int(meta["ncol"]), int(meta["nrow"])
        if meta.get("layout") != CORNER_LAYOUT:
            # File written before the switch to OpenCV row-major layout: arrays are
            # (n, ncol, nrow, d). Same physical board frame, only the corner order
            # differs, so rvecs / tvecs / per-image errors stay valid as-is.
            obj = _legacy_grid_to_row_major(obj, ncol_m, nrow_m)
            img = _legacy_grid_to_row_major(img, ncol_m, nrow_m)
            if run is not None:
                run["object_points"] = _legacy_grid_to_row_major(run["object_points"], ncol_m, nrow_m)
                run["image_points"] = _legacy_grid_to_row_major(run["image_points"], ncol_m, nrow_m)
                run["residuals"] = _legacy_flat_to_row_major(run["residuals"], ncol_m, nrow_m)

        self._apply_settings(settings)
        paths_abs = meta.get("paths") or [""] * n
        paths_rel = meta.get("paths_rel") or [""] * n
        selected = []
        for k in range(n):
            selected.append({"path": self._resolve_saved_path(paths_abs[k], paths_rel[k], npz_dir, int(src[k])),
                             "source_index": int(src[k]), "corners": img[k], "sharpness": float(sharp[k]),
                             "coverage": float(coverage[k]), "center_dist": float(centre[k])})
        self._stage1 = {"object_points": obj, "image_points": img, "sharpness": sharp, "source_index": src,
                        "img_size": tuple(int(v) for v in meta["img_size"]), "included": included,
                        "selected": selected, "ncol": ncol_m, "nrow": nrow_m,
                        "square_size": float(meta["square_size"])}
        self._candidates = []               # detection cache is not saved; Reselect needs a new Detect
        self._ranking = None
        self._runs = [run] if run is not None else []
        self._active_run_index = 0 if run is not None else None
        self._loo_rows = None
        self._selection_text = f"Loaded {n} selected image(s) from {os.path.basename(path)}."
        self._preview_selected_pos = 0
        self._preview_user_view = False
        self._preview_rgb, self._preview_rgb_path = None, None
        self._v3_centered = False
        self._outputs_dirty = True
        self._update_status_item()

    # ══ Help / lifecycle ═════════════════════════════════════════════════
    def _open_help(self) -> None:
        if self._help_popup is not None:
            try:
                if self._help_popup.winfo_exists():
                    self._help_popup.lift()
                    return
            except tk.TclError:
                pass
        popup = tk.Toplevel(self._dlg_parent())
        popup.title("Chessboard Calibration - Help")
        popup.geometry("760x640")
        body = tk.Frame(popup, padx=8, pady=8)
        body.pack(fill="both", expand=True)
        txt = tk.Text(body, font=("Courier", 9), wrap=tk.WORD, relief=tk.FLAT)
        vsb = ttk.Scrollbar(body, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", _HELP_TEXT)
        txt.configure(state="disabled")
        popup.bind("<Escape>", lambda _e: popup.destroy())
        self._help_popup = popup

    def close_inspector(self) -> None:
        if self._help_popup is not None:
            try:
                self._help_popup.destroy()
            except tk.TclError:
                pass
        super().close_inspector()
        self._reset_widget_refs()          # widgets are gone; background jobs keep running

    def on_destroy(self) -> None:
        self._destroyed = True
        if self._detect_state is not None:
            self._detect_state["cancel"].set()
        if self._loo_state is not None:
            self._loo_state["cancel"].set()
        for attr in ("_detect_poll_id", "_loo_poll_id"):
            after_id = getattr(self, attr)
            if after_id is not None:
                try:
                    self.canvas.after_cancel(after_id)
                except tk.TclError:
                    pass
        super().on_destroy()