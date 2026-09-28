# node_editor/nodes/stereo_sync_node.py
"""
Stereo-triangulation-based time synchronization node -- PER-POINT SHIFT
VERSION.

This generalizes the original single-shift StereoSyncNode: instead of
one shared time offset applied uniformly to every tracked point,
EACH POINT GETS ITS OWN INDEPENDENT t_shift. The motivating case is a
rolling-shutter camera, where different rows of the sensor (and
therefore different image points, roughly) are exposed at different
instants -- and, deliberately, this node does NOT assume any particular
scan model (e.g. constant top-to-bottom scan rate). Treating every
point's shift as a fully independent, freely estimated parameter is the
most general way to absorb that effect (it also happens to absorb any
other per-feature timing quirk, such as per-track processing latency),
at the cost of estimating num_points values instead of one.

KEY STRUCTURAL CONSEQUENCE -- THE OPTIMIZATION DECOUPLES: the
reprojection error for point p, at any time sample, depends ONLY on
point p's own two image observations and point p's own t_shift -- never
on any other point's data or shift. So the num_points-parameter problem
is not a single hard joint optimization; it is num_points completely
independent 1D optimizations. This is exploited throughout: Auto Align's
coarse grid search still evaluates one shared set of candidate shift
values in a single BATCHED pipeline call (cheap, same cost profile as
the original single-shift version, independent of num_points), then
picks each point's best candidate independently from that one sweep.
Refine, by contrast, genuinely must run an independent local optimizer
per point -- its cost scales linearly with num_points, and this is
called out explicitly where it matters below.

OUTPUT SHAPE CHANGES from the single-shift version:
    t2_sync      : now (num_points, num_steps2) -- t2 + t_shift[p] per point.
    t_shift      : NEW output, (num_points,) -- the estimated offsets
                   themselves. This is likely the most useful output for
                   downstream rolling-shutter analysis: plot t_shift
                   against each point's known pixel row (outside this
                   node, which deliberately does not assume that
                   relationship) to check for a scan-rate pattern.
    points3d     : (num_points, 3, n_eval), same as before, but n_eval
                   is now always len(t1) -- see _run_pipeline's
                   docstring for why the once-shared "overlap window"
                   trimming no longer applies with per-point shifts.
    reproj_error : (num_points, n_eval), same reasoning as points3d.

IMPORTANT ASSUMPTIONS (unchanged / reinforced):
    - Both cameras are STATIC for the whole recording (camera matrix,
      distortion coefficients, rvec, tvec are each one fixed value).
    - Point index p in imgpts1 and point index p in imgpts2 MUST refer
      to the SAME physical tracked feature. This was implicit before,
      but matters far more now: with a global shift, a point-ordering
      mismatch degraded overall sync quality; with per-point shifts, it
      silently attributes one point's correct offset to the WRONG
      point. A point-COUNT mismatch is detected and reported; ordering
      mismatches are not detectable from this node's inputs alone and
      remain the caller's responsibility.
    - NaN-fill and interpolation method are still GLOBAL settings
      (applied identically to every point) rather than per-point
      configurable -- exposing those per point as well was judged to
      add UI complexity out of proportion to the benefit; revisit if a
      real dataset needs different treatment per point.

Distortion correction (undistortion) is ALWAYS applied before
triangulation -- there is no opt-out. This is a mathematical
requirement, not a quality setting: cv2.triangulatePoints has no notion
of lens distortion and assumes its input already corresponds to the
distortion-free pinhole geometry implied by P = K[R|t]. Since dvec1 and
dvec2 are required inputs, the caller has already committed to a
distortion model for these cameras; skipping undistortion would feed
distorted pixel coordinates into a function that assumes undistorted
ones, producing a geometrically inconsistent triangulation whose error
grows with distance from the principal point -- indistinguishable from,
and easily mistaken for, a genuine synchronization error.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import CubicSpline, interp1d
from scipy.optimize import minimize_scalar

from node_editor.base_node import BaseNode
from node_editor.pin_types import PinDef, PinSchema, PinType


NAN_FILL_CHOICES = ("linear", "cubic", "nearest", "none (exclude gaps)")
INTERP_CHOICES = ("linear", "cubic")
AGGREGATION_CHOICES = ("median", "mean", "rms")
COMPONENT_CHOICES = ("u", "v")
SCOPE_CHOICES = ("this point", "all points")


# ══ Geometry helpers (module-level: pure functions, no node state) ═══════

def _rodrigues_matrix(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))
    return R


def _build_projection_matrix(cmat: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """P = K [R | t], the 3x4 projection matrix cv2.triangulatePoints needs.
    Built once per camera per pipeline run since both cameras are assumed
    static -- this is NOT recomputed per time sample or per point.
    """
    R = _rodrigues_matrix(rvec)
    t = np.asarray(tvec, dtype=float).reshape(3, 1)
    return np.asarray(cmat, dtype=float) @ np.hstack([R, t])


def _resize_shift_array(shifts: np.ndarray | None, num_points: int) -> np.ndarray:
    """Resize a per-point t_shift array to match the current number of
    tracked points, preserving existing values by index and padding new
    indices with 0.0 (no shift).

    WHY THIS IS NEEDED NOW BUT WASN'T IN THE SINGLE-SHIFT VERSION: a
    single global t_shift was trivially compatible with any num_points,
    since it applied uniformly. Once shift is keyed BY point index, the
    shift array's length is coupled to imgpts1/imgpts2's point
    dimension. If the upstream tracking data is ever reconnected with a
    different number of tracked points -- a routine occurrence while
    iterating on a tracking pipeline -- a stale, wrong-length shift
    array would silently misalign point indices (point 5's old shift
    getting applied to a DIFFERENT physical point after the point count
    changes) unless it is explicitly resized here, at every point where
    num_points could have changed.
    """
    if shifts is None or len(shifts) == 0:
        return np.zeros(num_points, dtype=float)
    if len(shifts) == num_points:
        return shifts
    resized = np.zeros(num_points, dtype=float)
    n_copy = min(len(shifts), num_points)
    resized[:n_copy] = shifts[:n_copy]
    return resized


def _fill_time_gaps(pts: np.ndarray, t: np.ndarray, method: str) -> np.ndarray:
    """Fill NaN gaps in a (num_points, 2, num_steps) image-point array
    along the time axis, independently per (point, u/v) pair.

    WHY INTERPOLATE INSTEAD OF REJECTING A POINT OUTRIGHT: occlusion and
    tracking dropouts are routine in multi-camera feature tracking, and
    a real session may involve dozens of points, each occluded at
    different, unpredictable times. Rejecting a point the moment it has
    ANY missing sample would routinely discard most of the dataset.
    Interpolating across short gaps is the standard, defensible way to
    keep those points usable -- but only if the interpolation method is
    a visible, user-adjustable choice rather than a silently hardcoded
    one, since a bad choice of method can just as easily fabricate
    motion that was never observed. Hence this is exposed as an
    inspector setting ("NaN fill"), not baked into the node. This
    function operates identically whether called with the full
    (num_points, 2, num_steps) array or a single-point (1, 2, num_steps)
    slice, which is exploited by the single-point Refine path below.

    method:
        "linear"               -- np.interp. Fast, robust default.
        "cubic"                -- scipy CubicSpline. Smoother for
                                   slowly-varying trajectories, but can
                                   overshoot across large gaps -- avoid
                                   for trajectories with sharp features
                                   (impacts, sudden motion) unless the
                                   gaps are short relative to the motion.
        "nearest"               -- scipy interp1d(kind="nearest"). The
                                   most conservative *filling* option:
                                   never invents an in-between value.
        "none (exclude gaps)"  -- do not fill. NaNs are left in place
                                   and are naturally excluded later
                                   wherever a NaN check gates
                                   triangulation. The most conservative
                                   choice overall.

    Only INTERIOR gaps (surrounded by valid samples on both sides) are
    filled. Leading/trailing NaN runs are always left as NaN, regardless
    of method: extrapolating a feature track beyond its observed range
    has no physical justification.
    """
    filled = pts.copy()
    if method.startswith("none"):
        return filled

    num_points, _, num_steps = pts.shape
    for p in range(num_points):
        for c in range(2):  # u, v
            series = pts[p, c, :]
            valid = ~np.isnan(series)
            if valid.sum() < 2:
                continue  # not enough samples to interpolate at all

            first = int(np.argmax(valid))
            last = num_steps - 1 - int(np.argmax(valid[::-1]))
            interior = np.zeros(num_steps, dtype=bool)
            interior[first:last + 1] = True
            gap = interior & ~valid
            if not gap.any():
                continue

            t_valid, v_valid = t[valid], series[valid]
            if method == "cubic":
                filled[p, c, gap] = CubicSpline(t_valid, v_valid)(t[gap])
            elif method == "nearest":
                filled[p, c, gap] = interp1d(t_valid, v_valid, kind="nearest")(t[gap])
            else:  # "linear"
                filled[p, c, gap] = np.interp(t[gap], t_valid, v_valid)
    return filled


def _interpolate_points_time(pts_on_t2_grid: np.ndarray, t2: np.ndarray, t_shifts: np.ndarray,
                              t_eval: np.ndarray, method: str) -> np.ndarray:
    """Resample camera 2's (already NaN-filled) per-point trajectories
    from their own time base t2 onto a common evaluation grid t_eval,
    using an INDEPENDENT time shift per point.

    pts_on_t2_grid: (num_points, 2, num_steps2) -- camera 2's tracked
        points on their own, UNSHIFTED time grid t2.
    t_shifts: (num_points,) -- THIS is the key change from the
        single-shift version of this node: instead of one shift value
        shared by every tracked point, each point gets its own
        independent shift, t_shifts[p]. See the module docstring for the
        rolling-shutter motivation and the deliberate choice not to
        assume any scan model.
    t_eval: (n_eval,) -- the common evaluation grid every point's data
        is resampled onto. In this node t_eval is always t1 in full
        (see _run_pipeline) -- there is no longer a single shared
        "overlap window" to restrict to, since each point's own valid
        time range differs depending on its own shift.

    Evaluation points outside a given point's own
    [t_src.min(), t_src.max()] are left as NaN rather than extrapolated
    -- unchanged from the single-shift version, just now evaluated per
    point with that point's own shifted time base rather than one
    shared shifted grid for every point.
    """
    num_points = pts_on_t2_grid.shape[0]
    out = np.full((num_points, 2, len(t_eval)), np.nan)
    for p in range(num_points):
        t_src_p = t2 + t_shifts[p]
        for c in range(2):
            series = pts_on_t2_grid[p, c, :]
            valid = ~np.isnan(series)
            if valid.sum() < 2:
                continue
            t_valid, v_valid = t_src_p[valid], series[valid]
            if method == "cubic":
                spline = CubicSpline(t_valid, v_valid, extrapolate=False)
                out[p, c, :] = spline(t_eval)
            else:  # "linear"
                interior = (t_eval >= t_valid.min()) & (t_eval <= t_valid.max())
                out[p, c, interior] = np.interp(t_eval[interior], t_valid, v_valid)
    return out


def _undistort_pixel_points(pts_uv: np.ndarray, cmat: np.ndarray, dvec: np.ndarray) -> np.ndarray:
    """Undistort (N, 2) pixel coordinates, passing NaN rows through as
    NaN (cv2.undistortPoints cannot accept NaN input, so those rows are
    computed on a NaN-free subset and re-inserted afterward).
    """
    out = np.full_like(pts_uv, np.nan, dtype=float)
    valid = ~np.isnan(pts_uv).any(axis=1)
    if not valid.any():
        return out
    src = pts_uv[valid].reshape(-1, 1, 2).astype(np.float64)
    undist = cv2.undistortPoints(src, np.asarray(cmat, dtype=float), np.asarray(dvec, dtype=float),
                                  P=np.asarray(cmat, dtype=float))
    out[valid] = undist.reshape(-1, 2)
    return out


def _triangulate_and_reproject(imgpts1_flat: np.ndarray, imgpts2_flat: np.ndarray,
                                cam1: dict, cam2: dict):
    """Triangulate a flat list of (point, time) correspondences and
    reproject them back into both cameras.

    Distortion correction is ALWAYS applied before triangulation, with
    no opt-out -- see the module docstring for why this is a
    mathematical requirement rather than a quality setting.

    imgpts1_flat, imgpts2_flat: (N, 2) pixel coordinates, one row per
    (point, time-sample) pair. N = num_points * n_eval; the caller is
    responsible for flattening and later un-flattening this shape --
    see _run_pipeline's PERFORMANCE note for why this function
    deliberately does NOT loop over time samples (or points) itself.
    This same function is reused, unmodified, for both the full
    all-points batch AND the single-point Refine path -- it has no idea
    how many distinct points its input represents, which is exactly
    what makes it reusable for both.

    Returns:
        points3d: (N, 3), NaN rows where triangulation was not possible.
        err1, err2: (N,) per-camera reprojection error in pixels,
                  comparing the reprojected 3D point against the
                  ORIGINAL (still-distorted) observed pixel coordinates
                  -- never against the undistorted points used as the
                  triangulation input, so an incorrect distortion model
                  shows up as error instead of silently cancelling out.
    """
    n = imgpts1_flat.shape[0]
    points3d = np.full((n, 3), np.nan)
    err1 = np.full(n, np.nan)
    err2 = np.full(n, np.nan)

    valid = (~np.isnan(imgpts1_flat).any(axis=1)) & (~np.isnan(imgpts2_flat).any(axis=1))
    if not valid.any():
        return points3d, err1, err2

    obs1, obs2 = imgpts1_flat[valid], imgpts2_flat[valid]

    tri1 = _undistort_pixel_points(obs1, cam1["cmat"], cam1["dvec"])
    tri2 = _undistort_pixel_points(obs2, cam2["cmat"], cam2["dvec"])
    still_valid = (~np.isnan(tri1).any(axis=1)) & (~np.isnan(tri2).any(axis=1))

    if not still_valid.any():
        return points3d, err1, err2

    homog = cv2.triangulatePoints(
        cam1["P"], cam2["P"],
        tri1[still_valid].T.astype(np.float64),
        tri2[still_valid].T.astype(np.float64),
    )
    pts3d_valid = (homog[:3] / homog[3]).T

    proj1, _ = cv2.projectPoints(pts3d_valid, cam1["rvec"], cam1["tvec"], cam1["cmat"], cam1["dvec"])
    proj2, _ = cv2.projectPoints(pts3d_valid, cam2["rvec"], cam2["tvec"], cam2["cmat"], cam2["dvec"])
    proj1, proj2 = proj1.reshape(-1, 2), proj2.reshape(-1, 2)

    obs1_valid, obs2_valid = obs1[still_valid], obs2[still_valid]
    e1 = np.linalg.norm(proj1 - obs1_valid, axis=1)
    e2 = np.linalg.norm(proj2 - obs2_valid, axis=1)

    full_valid_idx = np.where(valid)[0][still_valid]
    points3d[full_valid_idx] = pts3d_valid
    err1[full_valid_idx] = e1
    err2[full_valid_idx] = e2
    return points3d, err1, err2


def _aggregate_error(err1: np.ndarray, err2: np.ndarray, method: str,
                      excluded_points: set[int], outlier_threshold: float | None) -> float | None:
    """Combine per-point, per-time-sample reprojection errors for both
    cameras into a single scalar sync-error cost (pixels).

    err1, err2: (num_rows, num_steps) arrays (NaN where triangulation
    was not possible at that sample). num_rows is num_points for the
    "overall" aggregate, or 1 for a single-point aggregate -- this
    function does not care which, since excluded_points is empty in the
    single-point case (index-based exclusion is meaningless against a
    1-row slice whose row 0 may not even be the excluded point's real
    index).

    WHY MEDIAN IS THE RECOMMENDED DEFAULT, NOT MEAN:
    Triangulation is highly sensitive to feature-tracking noise -- a
    single mistracked point, or a point seen under a poor triangulation
    angle for a few frames, can produce a reprojection error orders of
    magnitude larger than a healthy point's error. Averaging (mean) lets
    that ONE bad sample dominate the aggregate cost, which can pull
    Auto Align / Refine toward a t_shift that accidentally minimizes the
    bad sample's error rather than one that genuinely synchronizes the
    two cameras' timing. The median is far more robust: a small number
    of outlier samples do not move it at all. RMS is offered as a
    secondary option because it is the conventional photogrammetry
    convention for REPORTING reprojection error once tracking quality
    is already trusted; mean is offered mainly for side-by-side
    comparison against RMS on the same data.

    NOTE ON THE PER-POINT-SHIFT VERSION SPECIFICALLY: this robustness
    argument now applies WITHIN each point's own time series too (used
    when this function is called with a single-point slice during Auto
    Align's per-candidate-per-point aggregation and during Refine's
    objective function) -- a point that is well-tracked overall but has
    a few bad frames should still get a sensible, non-hijacked estimate
    of ITS OWN best shift.
    """
    per_sample = np.nanmean(np.stack([err1, err2], axis=0), axis=0)  # (num_rows, num_steps)

    if excluded_points:
        per_sample = per_sample.copy()
        idx = [i for i in sorted(excluded_points) if i < per_sample.shape[0]]
        if idx:
            per_sample[idx, :] = np.nan

    if outlier_threshold is not None and outlier_threshold > 0:
        per_sample = np.where(per_sample > outlier_threshold, np.nan, per_sample)

    valid = per_sample[~np.isnan(per_sample)]
    if valid.size == 0:
        return None
    if method == "mean":
        return float(np.mean(valid))
    if method == "rms":
        return float(np.sqrt(np.mean(valid ** 2)))
    return float(np.median(valid))  # "median" -- see docstring above


def _parse_excluded_points(text: str) -> set[int]:
    result = set()
    for token in text.replace(";", ",").split(","):
        token = token.strip()
        if token.isdigit():
            result.add(int(token))
    return result


# ══ Node class ════════════════════════════════════════════════════════════

class StereoSyncNode(BaseNode):
    """Find an INDEPENDENT time shift for every tracked point between
    two static cameras, by minimizing each point's own stereo-
    triangulation reprojection error. See the module docstring for the
    full rationale (rolling shutter, decoupled optimization, output
    shape changes).
    """

    NODE_TYPE = "stereo_sync"
    DISPLAY_NAME = "Stereo Sync"
    CATEGORY = "process"
    SEARCH_KEYWORDS = ("stereo", "triangulation", "reprojection", "synchronization",
                       "time shift", "rolling shutter")

    NODE_WIDTH = 160
    NODE_HEIGHT = 90

    DEFAULT_GRID_SEARCH_SAMPLES = 25
    # Above this many points, Refine("all points") asks for confirmation
    # before running, since its cost scales linearly with num_points --
    # see _on_refine_clicked.
    REFINE_ALL_POINTS_WARNING_THRESHOLD = 30

    # ══ Construction ════════════════════════════════════════════════

    def __init__(self, node_id: str, canvas: tk.Canvas):
        super().__init__(node_id, canvas)

        # --- Applied state: the ONLY settings that affect the output
        # pins. Committed from the trial settings by "Apply".
        # t_shift is now a PER-POINT array, not a scalar. ---
        self._t_shift_applied: np.ndarray = np.zeros(0, dtype=float)
        self._nan_fill_method_applied: str = "none (exclude gaps)"
        self._interp_method_applied: str = "linear"
        self._excluded_points_applied: set[int] = set()

        # --- Trial state: freely edited in the inspector, previewed
        # immediately (subject to the performance-driven live/deferred
        # split described below), with no effect on the output pins
        # until "Apply" is pressed. Also a per-point array now. ---
        self._t_shift_trial: np.ndarray = np.zeros(0, dtype=float)
        self._slider_min: float = -1.0
        self._slider_max: float = 1.0
        self._nan_fill_method: str = self._nan_fill_method_applied
        self._interp_method: str = self._interp_method_applied
        self._excluded_points: set[int] = set()

        # --- Display/cost-only settings: affect ONLY the sync-error
        # readouts and the error plots, re-derived from cached per-
        # sample error arrays without re-running triangulation. ---
        self._aggregation: str = "median"        # "median" | "mean" | "rms"
        self._outlier_threshold: float | None = None  # pixels; None/0 disables

        # Which point is currently being viewed / edited via the shift
        # slider. This single index now drives FOUR things: the
        # trajectory overlay, the error-vs-time plot, the scan plot, and
        # which point's value the shift slider is bound to.
        self._selected_point_index: int = 0
        self._selected_component: str = "u"       # "u" | "v"

        # Cached RAW inputs from the last compute() call.
        self._t1: np.ndarray | None = None
        self._t2: np.ndarray | None = None
        self._imgpts1: np.ndarray | None = None    # (num_points, 2, num_steps1)
        self._imgpts2: np.ndarray | None = None    # (num_points, 2, num_steps2)
        self._cam1: dict | None = None              # {"cmat","dvec","rvec","tvec","P"}
        self._cam2: dict | None = None
        self._point_count_warning: str | None = None

        # Cached result of the most recent PREVIEW pipeline run (using
        # trial per-point shifts), kept so aggregation/outlier-threshold
        # changes, point-selector changes, and plot redraws never need
        # to re-triangulate.
        self._preview_result: dict | None = None    # {"t_eval","points3d","err1","err2"}

        # From the last Auto Align grid search: one shared set of
        # candidate shifts, and a (num_points, num_candidates) error
        # matrix -- see _run_coarse_grid_search.
        self._scan_shifts: np.ndarray | None = None
        self._scan_error_matrix: np.ndarray | None = None

        self._last_applied_agg_error: float | None = None  # for the on-node status line

        # Matplotlib objects, created lazily in build_inspector(). Four
        # stacked subplots: image-space trajectory overlay, reprojection
        # error vs time (selected point), per-point shift/error summary,
        # and error vs t_shift scan (selected point, last Auto Align run).
        self._fig: Figure | None = None
        self._ax_traj = None
        self._ax_err_time = None
        self._ax_point_summary = None
        self._ax_point_summary_twin = None   # recreated every redraw -- see _redraw_per_point_summary_plot
        self._ax_scan = None
        self._canvas_widget: FigureCanvasTkAgg | None = None
        self._pan_state: dict = {}
        self._link_xaxis_var: tk.BooleanVar | None = None  # created in _build_plots
        self._relative_trajectory_var: tk.BooleanVar | None = None

        self._status_item: int | None = None

    # ══ Pin schema ══════════════════════════════════════════════════

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[
                PinDef(name="cmat1", type=PinType.ARRAY, label="cmat1"),
                PinDef(name="dvec1", type=PinType.ARRAY, label="dvec1"),
                PinDef(name="rvec1", type=PinType.ARRAY, label="rvec1"),
                PinDef(name="tvec1", type=PinType.ARRAY, label="tvec1"),
                PinDef(name="imgpts1", type=PinType.ARRAY, label="imgpts1"),
                PinDef(name="t1", type=PinType.ARRAY, label="t1"),
                PinDef(name="cmat2", type=PinType.ARRAY, label="cmat2"),
                PinDef(name="dvec2", type=PinType.ARRAY, label="dvec2"),
                PinDef(name="rvec2", type=PinType.ARRAY, label="rvec2"),
                PinDef(name="tvec2", type=PinType.ARRAY, label="tvec2"),
                PinDef(name="imgpts2", type=PinType.ARRAY, label="imgpts2"),
                PinDef(name="t2", type=PinType.ARRAY, label="t2"),
            ],
            outputs=[
                PinDef(name="t2_sync", type=PinType.ARRAY, label="t2_sync"),
                PinDef(name="t_shift", type=PinType.ARRAY, label="t_shift"),
                PinDef(name="points3d", type=PinType.ARRAY, label="points3d"),
                PinDef(name="reproj_error", type=PinType.ARRAY, label="reproj_error"),
            ],
        )

    # ══ Core pipeline (shared by compute(), preview, grid search, refine) ══

    def _run_pipeline(self, t_shifts: np.ndarray, nan_fill_method: str, interp_method: str,
                       excluded_points: set[int], *,
                       imgpts1: np.ndarray | None = None,
                       imgpts2: np.ndarray | None = None) -> dict | None:
        """Run the full triangulation/reprojection pipeline for a given
        set of PER-POINT trial or applied t_shifts. Returns None only
        when the required inputs are entirely missing.

        UNLIKE THE SINGLE-SHIFT VERSION, a lack of time overlap is no
        longer an all-or-nothing failure: since each point can have a
        completely different shift, one point's data can be fully out
        of range while another's is well aligned. "No overlap" is now
        expressed PER POINT via NaN in the output arrays (produced by
        _interpolate_points_time's per-point domain check) rather than
        by this method returning None for the whole node.

        imgpts1 / imgpts2 overrides: normally omitted, in which case
        this reads the node's cached self._imgpts1 / self._imgpts2
        covering ALL tracked points. The override is used ONLY by
        _evaluate_cost_single_point, which slices a SINGLE point's row
        out of these arrays before calling this method -- this lets
        Refine's per-point local optimizer reuse the exact same pipeline
        logic while running against just one point's data. See that
        method's docstring for why this distinction matters for
        performance.

        PERFORMANCE NOTE: all (point, time-sample) correspondences are
        triangulated in a SINGLE vectorized batch rather than looping
        per time step (or per point) in Python -- see
        _triangulate_and_reproject. The evaluation grid is now always
        the FULL t1 array, not trimmed to a shared overlap window: with
        independent per-point shifts there is no single shared overlap
        window left to trim to, so trimming would need to track
        num_points different windows for no real benefit. Every point's
        outputs simply get the same, predictable length (len(t1)), with
        NaN outside that point's own valid sub-range.
        """
        t1 = self._t1
        t2 = self._t2
        src_imgpts1 = self._imgpts1 if imgpts1 is None else imgpts1
        src_imgpts2 = self._imgpts2 if imgpts2 is None else imgpts2

        if (t1 is None or t2 is None or src_imgpts1 is None or src_imgpts2 is None
                or self._cam1 is None or self._cam2 is None):
            return None

        t_eval = t1  # full grid; see docstring above

        imgpts1_filled = _fill_time_gaps(src_imgpts1, t1, nan_fill_method)
        imgpts2_filled = _fill_time_gaps(src_imgpts2, t2, nan_fill_method)

        imgpts1_eval = imgpts1_filled  # already on t1's own grid == t_eval
        imgpts2_eval = _interpolate_points_time(imgpts2_filled, t2, t_shifts, t_eval, interp_method)

        num_points, _, n_eval = imgpts1_eval.shape

        # Flatten (point, time) into one batch: time-major, point-minor,
        # so the un-flatten reshape below is a plain transpose, no
        # fancy indexing.
        flat1 = imgpts1_eval.transpose(2, 0, 1).reshape(-1, 2)
        flat2 = imgpts2_eval.transpose(2, 0, 1).reshape(-1, 2)

        p3d_flat, e1_flat, e2_flat = _triangulate_and_reproject(
            flat1, flat2, self._cam1, self._cam2,
        )

        points3d = p3d_flat.reshape(n_eval, num_points, 3).transpose(1, 2, 0)  # (P, 3, n_eval)
        err1 = e1_flat.reshape(n_eval, num_points).T                           # (P, n_eval)
        err2 = e2_flat.reshape(n_eval, num_points).T

        if excluded_points:
            idx = [i for i in sorted(excluded_points) if i < num_points]
            if idx:
                points3d[idx, :, :] = np.nan
                err1[idx, :] = np.nan
                err2[idx, :] = np.nan

        return {"t_eval": t_eval, "points3d": points3d, "err1": err1, "err2": err2}

    def _evaluate_cost_single_point(self, point_index: int, shift: float) -> float | None:
        """Cheap, single-point cost evaluation used as the objective
        function inside Refine's scipy.optimize.minimize_scalar call.

        WHY THIS EXISTS SEPARATELY FROM THE FULL BATCHED PIPELINE: as
        explained in the module docstring, the reprojection error for
        point p depends ONLY on point p's own data and its own shift --
        the num_points per-point optimization problems are fully
        DECOUPLED. A local optimizer typically needs several dozen
        objective evaluations to converge; running each of those
        against the FULL num_points-point batched pipeline (as Auto
        Align's grid search does) would multiply the already-expensive
        triangulation cost by roughly num_points for NO benefit, since
        every other point's result would be discarded anyway. Slicing
        the input down to just this one point before calling
        _run_pipeline keeps each evaluation as cheap as a single-point
        triangulation, independent of how many points exist in total --
        this is what keeps Refine("this point") fast; it is also why
        Refine("all points") still costs O(num_points) overall, since it
        must run this independently-cheap evaluation loop once per point.
        """
        if self._imgpts1 is None or self._imgpts2 is None:
            return None
        result = self._run_pipeline(
            np.array([shift]), self._nan_fill_method, self._interp_method, set(),
            imgpts1=self._imgpts1[point_index:point_index + 1],
            imgpts2=self._imgpts2[point_index:point_index + 1],
        )
        if result is None:
            return None
        return _aggregate_error(result["err1"], result["err2"], self._aggregation, set(), self._outlier_threshold)

    def _run_coarse_grid_search(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Sweep DEFAULT_GRID_SEARCH_SAMPLES candidate shift values,
        applying EACH candidate UNIFORMLY to every point in a single
        batched pipeline call per candidate, and record the per-point
        (aggregated over TIME ONLY) error at every candidate.

        Returns (shifts, error_matrix) with error_matrix of shape
        (num_points, num_candidates). The best candidate index differs
        per point in general -- that is the entire point of a per-point
        shift model. This grid still explores one SHARED set of
        candidate values for computational efficiency: the whole reason
        this stays cheap regardless of num_points is that the batched
        pipeline call's cost per candidate is the same as it was in the
        single-shift version of this node (see _run_pipeline's
        PERFORMANCE note) -- aggregating the resulting error matrix
        column-by-column, per point, AFTER the sweep is what turns one
        shared sweep into num_points independent coarse estimates.
        Refine (_evaluate_cost_single_point) then locally improves each
        point's estimate beyond this grid's resolution.
        """
        if self._imgpts1 is None:
            return None
        num_points = self._imgpts1.shape[0]
        shifts = np.linspace(self._slider_min, self._slider_max, self.DEFAULT_GRID_SEARCH_SAMPLES)
        error_matrix = np.full((num_points, len(shifts)), np.inf)

        for i, s in enumerate(shifts):
            uniform_shifts = np.full(num_points, s)
            result = self._run_pipeline(
                uniform_shifts, self._nan_fill_method, self._interp_method, self._excluded_points,
            )
            if result is None:
                continue
            # Aggregate over TIME ONLY here, per point, per candidate --
            # aggregating over points as well would collapse exactly the
            # per-point information this grid search exists to produce.
            for p in range(num_points):
                agg = _aggregate_error(
                    result["err1"][p:p + 1], result["err2"][p:p + 1],
                    self._aggregation, set(), self._outlier_threshold,
                )
                if agg is not None:
                    error_matrix[p, i] = agg

        return shifts, error_matrix

    # ══ Compute ═════════════════════════════════════════════════════

    def compute(self, inputs: dict) -> dict:
        required = ("cmat1", "dvec1", "rvec1", "tvec1", "imgpts1", "t1",
                    "cmat2", "dvec2", "rvec2", "tvec2", "imgpts2", "t2")
        if any(inputs.get(name) is None for name in required):
            self._clear_input_cache()
            return {"t2_sync": None, "t_shift": None, "points3d": None, "reproj_error": None}

        self._t1 = np.asarray(inputs["t1"], dtype=float)
        self._t2 = np.asarray(inputs["t2"], dtype=float)
        imgpts1_raw = np.asarray(inputs["imgpts1"], dtype=float)
        imgpts2_raw = np.asarray(inputs["imgpts2"], dtype=float)

        num_points1, num_points2 = imgpts1_raw.shape[0], imgpts2_raw.shape[0]
        if num_points1 != num_points2:
            # Point index p in imgpts1 and point index p in imgpts2 MUST
            # refer to the same physical tracked feature: each point now
            # carries its OWN independent shift, so a mismatched point
            # count (or, more subtly, a matching count but misordered
            # points) means shifts get silently attributed to the wrong
            # feature. A count mismatch is at least detectable here;
            # misordering is not, and remains the caller's responsibility.
            self._point_count_warning = (
                f"imgpts1 has {num_points1} points but imgpts2 has {num_points2}; "
                "using the smaller count and ignoring the extra points."
            )
        else:
            self._point_count_warning = None

        num_points = min(num_points1, num_points2)
        self._imgpts1 = imgpts1_raw[:num_points]
        self._imgpts2 = imgpts2_raw[:num_points]

        self._t_shift_applied = _resize_shift_array(self._t_shift_applied, num_points)
        self._t_shift_trial = _resize_shift_array(self._t_shift_trial, num_points)

        self._cam1 = self._build_camera_dict(inputs["cmat1"], inputs["dvec1"], inputs["rvec1"], inputs["tvec1"])
        self._cam2 = self._build_camera_dict(inputs["cmat2"], inputs["dvec2"], inputs["rvec2"], inputs["tvec2"])

        result = self._run_pipeline(
            self._t_shift_applied, self._nan_fill_method_applied,
            self._interp_method_applied, self._excluded_points_applied,
        )

        t2_sync = self._t2[None, :] + self._t_shift_applied[:, None]  # (num_points, num_steps2)
        t_shift_out = self._t_shift_applied.copy()

        if result is None:
            self._last_applied_agg_error = None
            self._update_status_text()
            return {"t2_sync": t2_sync, "t_shift": t_shift_out, "points3d": None, "reproj_error": None}

        reproj_error = np.nanmean(np.stack([result["err1"], result["err2"]], axis=0), axis=0)
        self._last_applied_agg_error = _aggregate_error(
            result["err1"], result["err2"], self._aggregation,
            self._excluded_points_applied, self._outlier_threshold,
        )
        self._update_status_text()

        return {
            "t2_sync": t2_sync, "t_shift": t_shift_out,
            "points3d": result["points3d"], "reproj_error": reproj_error,
        }

    @staticmethod
    def _build_camera_dict(cmat, dvec, rvec, tvec) -> dict:
        cam = {
            "cmat": np.asarray(cmat, dtype=float),
            "dvec": np.asarray(dvec, dtype=float),
            "rvec": np.asarray(rvec, dtype=float).reshape(3, 1),
            "tvec": np.asarray(tvec, dtype=float).reshape(3, 1),
        }
        cam["P"] = _build_projection_matrix(cam["cmat"], cam["rvec"], cam["tvec"])
        return cam

    def _clear_input_cache(self) -> None:
        self._t1 = self._t2 = self._imgpts1 = self._imgpts2 = None
        self._cam1 = self._cam2 = None
        self._last_applied_agg_error = None
        self._point_count_warning = None
        self._update_status_text()

    # ══ Serialization ═══════════════════════════════════════════════

    def get_params(self) -> dict:
        # Serialize the APPLIED settings: what actually produced the
        # current outputs. Trial edits not yet Applied are NOT
        # persisted. t_shift is now a list (JSON-friendly) instead of a
        # single float.
        return {
            "t_shift_applied": self._t_shift_applied.tolist(),
            "slider_min": self._slider_min,
            "slider_max": self._slider_max,
            "nan_fill_method": self._nan_fill_method_applied,
            "interp_method": self._interp_method_applied,
            "excluded_points": sorted(self._excluded_points_applied),
            "aggregation": self._aggregation,
            "outlier_threshold": self._outlier_threshold,
        }

    def set_params(self, params: dict) -> None:
        self._t_shift_applied = np.array(params.get("t_shift_applied", []), dtype=float)
        self._slider_min = float(params.get("slider_min", -1.0))
        self._slider_max = float(params.get("slider_max", 1.0))
        self._nan_fill_method_applied = params.get("nan_fill_method", "none (exclude gaps)")
        self._interp_method_applied = params.get("interp_method", "linear")
        self._excluded_points_applied = set(params.get("excluded_points", []))
        self._aggregation = params.get("aggregation", "median")
        self._outlier_threshold = params.get("outlier_threshold")

        # Trial settings start out matching the applied ones on load, so
        # the inspector opens in a clean, non-dirty state.
        self._t_shift_trial = self._t_shift_applied.copy()
        self._nan_fill_method = self._nan_fill_method_applied
        self._interp_method = self._interp_method_applied
        self._excluded_points = set(self._excluded_points_applied)

    # ══ Canvas body ═════════════════════════════════════════════════

    def _status_text(self) -> str:
        if self._last_applied_agg_error is None:
            return "no result yet"
        n = len(self._t_shift_applied)
        return f"{n} pts, {self._aggregation} reproj err: {self._last_applied_agg_error:.3g} px"

    def _update_status_text(self) -> None:
        if getattr(self, "_status_item", None) is not None:
            self.canvas.itemconfig(self._status_item, text=self._status_text())

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
        self._ensure_trial_shift_size()

        self._insp_error_var = tk.StringVar(value="")
        tk.Label(parent, textvariable=self._insp_error_var, anchor="w", justify="left",
                 font=("Arial", 9)).pack(fill="x", pady=(4, 0))

        if self._point_count_warning:
            tk.Label(parent, text=self._point_count_warning, anchor="w",
                     fg="#aa5500", font=("Arial", 8)).pack(fill="x")

        main_row = tk.Frame(parent)
        main_row.pack(fill="both", expand=True, pady=(4, 0))

        left = tk.Frame(main_row)
        left.pack(side="left", fill="y", anchor="n")

        self._build_pipeline_controls(left)
        self._build_cost_controls(left)
        self._build_point_selector_controls(left)
        self._build_shift_controls(left)

        action_row = tk.Frame(left)
        action_row.pack(fill="x", pady=(8, 0))
        tk.Label(action_row, text="Scope:").pack(side="left")
        self._insp_scope_var = tk.StringVar(value="this point")
        scope_box = ttk.Combobox(
            action_row, textvariable=self._insp_scope_var, values=SCOPE_CHOICES,
            state="readonly", width=10,
        )
        scope_box.pack(side="left", padx=(4, 8))
        tk.Button(action_row, text="Auto Align", command=self._on_auto_align_clicked).pack(side="left")
        tk.Button(action_row, text="Refine", command=self._on_refine_clicked).pack(side="left", padx=(6, 0))

        apply_row = tk.Frame(left)
        apply_row.pack(fill="x", pady=(4, 0))
        tk.Button(apply_row, text="Apply", width=12, command=self._on_apply_clicked).pack(side="right")

        plot_frame = tk.Frame(main_row)
        plot_frame.pack(side="left", fill="both", expand=True, padx=(8, 0))
        self._build_plots(plot_frame)

        # Run one preview pass immediately so the plots and sync-error
        # readout aren't blank the first time the inspector opens.
        self._recompute_preview()

    def _ensure_trial_shift_size(self) -> None:
        """Keep the trial shift array in sync with the current number of
        tracked points. Called defensively at the start of every
        inspector action that reads or writes into it, since the
        upstream point count can change at any time (a reconnected
        tracking node), not only at compute() time.
        """
        if self._imgpts1 is None:
            return
        num_points = self._imgpts1.shape[0]
        self._t_shift_trial = _resize_shift_array(self._t_shift_trial, num_points)

    # -- Plot embedding: 4 stacked, independently pannable/zoomable axes --

    def _build_plots(self, parent: tk.Frame) -> None:
        link_row = tk.Frame(parent)
        link_row.pack(fill="x")
        self._link_xaxis_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            link_row, text="Link horizontal axis across time-domain plots (Ctrl+wheel)",
            variable=self._link_xaxis_var,
        ).pack(side="left")

        relative_row = tk.Frame(parent)
        relative_row.pack(fill="x")
        self._relative_trajectory_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            relative_row, text="Plot camera values relative to time index [0]",
            variable=self._relative_trajectory_var,
            command=self._redraw_trajectory_plot,
        ).pack(side="left")

        # The 4 stacked subplots can be taller than the inspector
        # window, so wrap them in a vertically scrollable canvas.
        outer_canvas = tk.Canvas(parent, highlightthickness=0)
        vbar = ttk.Scrollbar(parent, orient="vertical", command=outer_canvas.yview)
        outer_canvas.configure(yscrollcommand=vbar.set)
        vbar.pack(side="right", fill="y")
        outer_canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(outer_canvas)
        inner_id = outer_canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda _e: outer_canvas.configure(scrollregion=outer_canvas.bbox("all")))
        outer_canvas.bind("<Configure>", lambda e: outer_canvas.itemconfig(inner_id, width=e.width))

        def _on_mousewheel(e):
            outer_canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        outer_canvas.bind("<MouseWheel>", _on_mousewheel)
        inner.bind("<MouseWheel>", _on_mousewheel)

        self._fig = Figure(figsize=(6, 11), dpi=100)
        self._ax_traj = self._fig.add_subplot(411)
        self._ax_err_time = self._fig.add_subplot(412)
        self._ax_point_summary = self._fig.add_subplot(413)
        self._ax_scan = self._fig.add_subplot(414)
        self._fig.tight_layout(pad=2.0)

        self._canvas_widget = FigureCanvasTkAgg(self._fig, master=inner)
        self._canvas_widget.get_tk_widget().pack(fill="both", expand=True)

        # NOTE: _ax_point_summary has POINT INDEX on its x-axis, not
        # time -- "Link horizontal axis" intentionally does not apply to
        # it (see _on_plot_scroll). _ax_scan's x-axis is candidate shift
        # values, not time either, so it is also excluded from linking.
        for ax in (self._ax_traj, self._ax_err_time, self._ax_point_summary, self._ax_scan):
            self._pan_state[ax] = {
                "active": False, "x0": None, "y0": None, "xlim0": None, "ylim0": None,
            }

        self._canvas_widget.mpl_connect("scroll_event", self._on_plot_scroll)
        self._canvas_widget.mpl_connect("button_press_event", self._on_plot_press)
        self._canvas_widget.mpl_connect("motion_notify_event", self._on_plot_drag)
        self._canvas_widget.mpl_connect("button_release_event", self._on_plot_release)

    def _on_plot_scroll(self, event) -> None:
        # Same convention as elsewhere: no modifier zooms both axes,
        # Ctrl zooms x (horizontal) only, Shift zooms y (vertical) only.
        # "Link horizontal axis" only applies between _ax_traj and
        # _ax_err_time, since those are the only two plots that share a
        # literal time axis -- _ax_point_summary's x-axis is point
        # index and _ax_scan's x-axis is candidate shift value, so
        # linking either of those to a time axis would be meaningless.
        ax = event.inaxes
        if ax is None or event.xdata is None or event.ydata is None:
            return

        factor = 0.9 if event.button == "up" else 1.1
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        key = getattr(event, "key", None)
        linkable_time_axes = (self._ax_traj, self._ax_err_time)

        if key in ("control", "ctrl"):
            new_xlim = [event.xdata - (event.xdata - v) * factor for v in xlim]
            if (self._link_xaxis_var is not None and self._link_xaxis_var.get()
                    and ax in linkable_time_axes):
                for other_ax in linkable_time_axes:
                    other_ax.set_xlim(new_xlim)
            else:
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

    def _on_plot_press(self, event) -> None:
        ax = event.inaxes
        if ax is None or event.button != 1 or event.xdata is None:
            return
        self._pan_state[ax].update(
            active=True, x0=event.xdata, y0=event.ydata,
            xlim0=ax.get_xlim(), ylim0=ax.get_ylim(),
        )

    def _on_plot_drag(self, event) -> None:
        for ax, state in self._pan_state.items():
            if not state["active"] or event.xdata is None or event.inaxes is not ax:
                continue
            dx = event.xdata - state["x0"]
            dy = event.ydata - state["y0"]
            x0, x1 = state["xlim0"]
            y0, y1 = state["ylim0"]
            ax.set_xlim(x0 - dx, x1 - dx)
            ax.set_ylim(y0 - dy, y1 - dy)
            self._canvas_widget.draw_idle()

    def _on_plot_release(self, _event) -> None:
        for state in self._pan_state.values():
            state["active"] = False

    # -- Cheap redraw: single-point trajectory overlay (drag-safe) ----------

    def _redraw_trajectory_plot(self) -> None:
        """Redraw ONLY the image-space trajectory overlay for the
        currently selected point/component, using that point's own
        current TRIAL shift. O(num_steps), no cv2 calls -- safe to call
        on every slider-drag tick.
        """
        ax = self._ax_traj
        had_data = bool(ax.lines)
        xlim, ylim = (ax.get_xlim(), ax.get_ylim()) if had_data else (None, None)
        ax.clear()

        if self._t1 is not None and self._imgpts1 is not None and self._imgpts2 is not None:
            num_points = self._imgpts1.shape[0]
            p = max(0, min(self._selected_point_index, num_points - 1))
            self._selected_point_index = p
            c = 0 if self._selected_component == "u" else 1

            series1 = _fill_time_gaps(self._imgpts1[p:p + 1], self._t1, self._nan_fill_method)[0, c, :]
            series2 = _fill_time_gaps(self._imgpts2[p:p + 1], self._t2, self._nan_fill_method)[0, c, :]
            shift_p = self._t_shift_trial[p] if p < len(self._t_shift_trial) else 0.0
            t2_trial = self._t2 + shift_p

            relative = (self._relative_trajectory_var is not None
                        and self._relative_trajectory_var.get())
            if relative:
                series1 = series1 - series1[0]
                series2 = series2 - series2[0]

            ax.plot(self._t1, series1, color="#1f77b4", linewidth=1.2, label="camera 1")
            ax.plot(t2_trial, series2, color="#d62728", linewidth=1.2,
                    label=f"camera 2 (trial shift={shift_p:.4g})")
            ax.legend(loc="upper right", fontsize=8)

        ax.set_xlabel("time")
        relative = (self._relative_trajectory_var is not None
                and self._relative_trajectory_var.get())
        suffix = " relative to index [0]" if relative else ""
        ax.set_ylabel(f"{self._selected_component}{suffix} (pixels)")
        ax.set_title(f"point {self._selected_point_index}: image-space overlay", fontsize=9)

        if had_data:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        self._canvas_widget.draw_idle()

    # -- Expensive redraw: full pipeline (slider-release / setting-change) --

    def _recompute_preview(self) -> None:
        """Run the full batched pipeline over ALL points at the current
        trial shifts, cache the raw per-sample errors, and refresh
        everything that depends on them. NEVER call this from a slider
        drag tick.

        NOTE: even though tweaking ONE point's shift only actually
        changes that point's own numbers (the optimization decouples --
        see the module docstring), this still reruns the pipeline for
        every point rather than trying to isolate just the changed one.
        That is intentional: the batched call's cost is dominated by its
        total element count, not by how many points logically changed,
        so there is no real saving to be had from a partial recompute
        here, and always reusing the same one code path keeps this
        simpler and less error-prone than tracking per-point staleness.
        """
        self._ensure_trial_shift_size()
        self._preview_result = self._run_pipeline(
            self._t_shift_trial, self._nan_fill_method, self._interp_method, self._excluded_points,
        )
        self._update_sync_error_label()
        self._redraw_error_time_plot()
        self._redraw_per_point_summary_plot()
        self._redraw_trajectory_plot()

    def _update_sync_error_label(self) -> None:
        """Show BOTH the selected point's own sync error and the overall
        (all-points) aggregate. With a single global shift these were
        the same number by definition; with per-point shifts they
        usually are not, so showing only one would hide the other.
        """
        if self._preview_result is None:
            self._insp_error_var.set("Sync error: n/a (missing input)")
            return

        err1, err2 = self._preview_result["err1"], self._preview_result["err2"]
        overall = _aggregate_error(err1, err2, self._aggregation, self._excluded_points, self._outlier_threshold)

        p = self._selected_point_index
        point_err = None
        if 0 <= p < err1.shape[0]:
            point_err = _aggregate_error(err1[p:p + 1], err2[p:p + 1], self._aggregation, set(),
                                          self._outlier_threshold)

        overall_text = f"{overall:.4g} px" if overall is not None else "n/a"
        point_text = f"{point_err:.4g} px" if point_err is not None else "n/a"
        self._insp_error_var.set(
            f"Point {p} sync error: {point_text}   |   Overall ({self._aggregation}): {overall_text}\n"
            f"nan_fill={self._nan_fill_method}, interp={self._interp_method}"
        )

    def _redraw_error_time_plot(self) -> None:
        """Reprojection error over time for the SELECTED POINT only.
        Unlike the single-shift version, this no longer aggregates
        across points at each time sample -- with independent per-point
        shifts, a cross-point aggregate curve would blur together points
        that are individually well- or poorly-synchronized, which is
        exactly the detail this plot exists to show. The cross-point
        view lives in _redraw_per_point_summary_plot instead.
        """
        ax = self._ax_err_time
        had_data = bool(ax.lines)
        xlim, ylim = (ax.get_xlim(), ax.get_ylim()) if had_data else (None, None)
        ax.clear()

        if self._preview_result is not None:
            p = self._selected_point_index
            err1_all, err2_all = self._preview_result["err1"], self._preview_result["err2"]
            if 0 <= p < err1_all.shape[0]:
                t_eval = self._preview_result["t_eval"]
                ax.plot(t_eval, err1_all[p], color="#1f77b4", linewidth=1.0, label="camera 1 error")
                ax.plot(t_eval, err2_all[p], color="#d62728", linewidth=1.0, label="camera 2 error")
                ax.legend(loc="upper right", fontsize=8)

        ax.set_xlabel("time")
        ax.set_ylabel("reproj. error (px)")
        ax.set_title(f"point {self._selected_point_index}: reprojection error vs time", fontsize=9)

        if had_data:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        self._canvas_widget.draw_idle()

    def _redraw_per_point_summary_plot(self) -> None:
        """Plot every point's current TRIAL shift (left axis, bars) and
        its aggregate reprojection error (right axis, markers) against
        point index.

        THIS IS THE MAIN DIAGNOSTIC VIEW MOTIVATING THE PER-POINT SHIFT
        MODEL: if the two-camera timing offset varies systematically in
        a rolling-shutter-like pattern, plotting shift against point
        index is how that pattern becomes visible -- WITHOUT this node
        assuming or fitting any particular scan-rate model itself; that
        analysis (e.g. correlating shift against each point's known
        pixel row) is left to the person reading this plot, or to a
        downstream node consuming the t_shift output pin. High-error
        points are also easy to spot here, which is often a useful cue
        for adding a point to "Excluded points" rather than trusting its
        individually estimated shift.
        """
        ax = self._ax_point_summary
        ax.clear()
        # A colorbar-style secondary axis (twinx) is NOT cleared by
        # ax.clear() -- it must be explicitly removed before recreating,
        # or it silently accumulates on every redraw (the same lifecycle
        # issue as cwt_1d_node.py's colorbar, and ifft_1d_node.py's
        # phase twin-axis).
        if self._ax_point_summary_twin is not None:
            self._ax_point_summary_twin.remove()
            self._ax_point_summary_twin = None

        if self._preview_result is not None and len(self._t_shift_trial) > 0:
            num_points = len(self._t_shift_trial)
            idx = np.arange(num_points)

            ax.bar(idx, self._t_shift_trial, color="#9ecae1", width=0.6)
            ax.set_ylabel("t_shift", color="#3182bd")

            err1, err2 = self._preview_result["err1"], self._preview_result["err2"]
            per_point_error = np.array([
                _aggregate_error(err1[p:p + 1], err2[p:p + 1], self._aggregation, set(), self._outlier_threshold)
                if _aggregate_error(err1[p:p + 1], err2[p:p + 1], self._aggregation, set(),
                                    self._outlier_threshold) is not None else np.nan
                for p in range(num_points)
            ], dtype=float)

            self._ax_point_summary_twin = ax.twinx()
            self._ax_point_summary_twin.scatter(idx, per_point_error, color="#d62728", marker="o", s=14)
            self._ax_point_summary_twin.set_ylabel(f"{self._aggregation} reproj. error (px)", color="#d62728")

            # Shade excluded points so a gap in the error scatter reads
            # as "deliberately excluded" rather than "no valid data".
            for p in sorted(self._excluded_points):
                if p < num_points:
                    ax.axvspan(p - 0.5, p + 0.5, color="#888888", alpha=0.15)

            # Highlight the currently selected point so it's easy to
            # find on this plot too.
            if 0 <= self._selected_point_index < num_points:
                ax.axvline(self._selected_point_index, color="#333333", linewidth=0.8, linestyle=":")

        ax.set_xlabel("point index")
        ax.set_title("per-point shift and error", fontsize=9)
        self._canvas_widget.draw_idle()

    def _redraw_scan_plot(self) -> None:
        """Auto Align scan for the SELECTED point, sliced out of the
        cached (num_points, num_candidates) matrix from the last grid
        search -- switching the selected point redraws this instantly
        without re-running the search.
        """
        ax = self._ax_scan
        ax.clear()
        if self._scan_shifts is not None and self._scan_error_matrix is not None:
            p = self._selected_point_index
            if 0 <= p < self._scan_error_matrix.shape[0]:
                row = self._scan_error_matrix[p]
                finite = np.isfinite(row)
                if finite.any():
                    ax.plot(self._scan_shifts[finite], row[finite], color="#2ca02c", marker="o", markersize=3)
                if 0 <= p < len(self._t_shift_trial):
                    ax.axvline(self._t_shift_trial[p], color="#888888", linestyle="--", linewidth=1)
        ax.set_xlabel("t_shift")
        ax.set_ylabel(f"{self._aggregation} reproj. error (px)")
        ax.set_title(f"point {self._selected_point_index}: Auto Align scan (last run)", fontsize=9)
        self._canvas_widget.draw_idle()

    # -- Pipeline-affecting controls (nan fill, interp, excluded points) ----

    def _build_pipeline_controls(self, parent: tk.Frame) -> None:
        row = tk.Frame(parent)
        row.pack(fill="x", pady=(6, 0))

        tk.Label(row, text="NaN fill:").pack(side="left")
        self._insp_nanfill_var = tk.StringVar(value=self._nan_fill_method)
        nanfill_box = ttk.Combobox(
            row, textvariable=self._insp_nanfill_var, values=NAN_FILL_CHOICES,
            state="readonly", width=16,
        )
        nanfill_box.pack(side="left", padx=(4, 12))
        nanfill_box.bind("<<ComboboxSelected>>", lambda _e: self._on_pipeline_setting_changed())

        tk.Label(row, text="Interp:").pack(side="left")
        self._insp_interp_var = tk.StringVar(value=self._interp_method)
        interp_box = ttk.Combobox(
            row, textvariable=self._insp_interp_var, values=INTERP_CHOICES,
            state="readonly", width=8,
        )
        interp_box.pack(side="left", padx=(4, 12))
        interp_box.bind("<<ComboboxSelected>>", lambda _e: self._on_pipeline_setting_changed())

        row2 = tk.Frame(parent)
        row2.pack(fill="x", pady=(4, 0))
        tk.Label(row2, text="Excluded points (comma-separated indices):").pack(side="left")
        self._insp_excluded_var = tk.StringVar(value=",".join(str(i) for i in sorted(self._excluded_points)))
        excl_entry = tk.Entry(row2, textvariable=self._insp_excluded_var, width=24)
        excl_entry.pack(side="left", padx=(4, 0))
        excl_entry.bind("<Return>", lambda _e: self._on_pipeline_setting_changed())
        excl_entry.bind("<FocusOut>", lambda _e: self._on_pipeline_setting_changed())

    def _on_pipeline_setting_changed(self) -> None:
        # These settings change what gets computed for every point, so a
        # change here re-runs the full batched pipeline once -- unlike
        # the aggregation/outlier controls below, which never touch it.
        self._nan_fill_method = self._insp_nanfill_var.get()
        self._interp_method = self._insp_interp_var.get()
        self._excluded_points = _parse_excluded_points(self._insp_excluded_var.get())
        self._recompute_preview()

    # -- Cost-only controls (aggregation, outlier threshold) -----------------

    def _build_cost_controls(self, parent: tk.Frame) -> None:
        row = tk.Frame(parent)
        row.pack(fill="x", pady=(6, 0))

        tk.Label(row, text="Aggregation:").pack(side="left")
        self._insp_agg_var = tk.StringVar(value=self._aggregation)
        agg_box = ttk.Combobox(
            row, textvariable=self._insp_agg_var, values=AGGREGATION_CHOICES,
            state="readonly", width=8,
        )
        agg_box.pack(side="left", padx=(4, 12))
        agg_box.bind("<<ComboboxSelected>>", lambda _e: self._on_cost_setting_changed())

        tk.Label(row, text="Outlier threshold (px, blank = off):").pack(side="left")
        self._insp_outlier_var = tk.StringVar(
            value="" if not self._outlier_threshold else str(self._outlier_threshold)
        )
        outlier_entry = tk.Entry(row, textvariable=self._insp_outlier_var, width=8)
        outlier_entry.pack(side="left", padx=(4, 0))
        outlier_entry.bind("<Return>", lambda _e: self._on_cost_setting_changed())
        outlier_entry.bind("<FocusOut>", lambda _e: self._on_cost_setting_changed())

    def _on_cost_setting_changed(self) -> None:
        # These settings only change how the ALREADY-COMPUTED per-sample
        # errors are summarized -- re-aggregate from cache, no pipeline
        # re-run.
        self._aggregation = self._insp_agg_var.get()
        text = self._insp_outlier_var.get().strip()
        try:
            self._outlier_threshold = float(text) if text else None
        except ValueError:
            self._outlier_threshold = None
        self._update_sync_error_label()
        self._redraw_error_time_plot()
        self._redraw_per_point_summary_plot()

    # -- Point selector (drives trajectory / error-vs-time / scan / slider) -

    def _build_point_selector_controls(self, parent: tk.Frame) -> None:
        row = tk.Frame(parent)
        row.pack(fill="x", pady=(6, 0))

        tk.Label(row, text="Selected point:").pack(side="left")
        self._insp_point_var = tk.IntVar(value=self._selected_point_index)
        point_entry = tk.Spinbox(
            row, from_=0, to=9999, textvariable=self._insp_point_var, width=5,
            command=self._on_selected_point_changed,
        )
        point_entry.pack(side="left", padx=(4, 12))
        point_entry.bind("<Return>", lambda _e: self._on_selected_point_changed())

        tk.Label(row, text="Component:").pack(side="left")
        self._insp_component_var = tk.StringVar(value=self._selected_component)
        comp_box = ttk.Combobox(
            row, textvariable=self._insp_component_var, values=COMPONENT_CHOICES,
            state="readonly", width=4,
        )
        comp_box.pack(side="left", padx=(4, 0))
        comp_box.bind("<<ComboboxSelected>>", lambda _e: self._on_selected_point_changed())

    def _on_selected_point_changed(self) -> None:
        """Switching the selected point changes WHICH point's shift the
        slider edits and which point's data the trajectory / error-vs-
        time / scan plots show -- it does NOT itself change any shift
        value. This is a cheap, redraw-only operation: _preview_result
        and _scan_error_matrix already hold every point's data from the
        last full run, so no pipeline re-run is needed.
        """
        try:
            self._selected_point_index = int(self._insp_point_var.get())
        except (tk.TclError, ValueError):
            pass
        self._selected_component = self._insp_component_var.get()
        self._ensure_trial_shift_size()

        if 0 <= self._selected_point_index < len(self._t_shift_trial):
            self._insp_shift_var.set(self._t_shift_trial[self._selected_point_index])

        self._redraw_trajectory_plot()
        self._redraw_error_time_plot()
        self._redraw_scan_plot()

    # -- Time-shift slider: edits ONLY the SELECTED point's trial shift -----

    def _build_shift_controls(self, parent: tk.Frame) -> None:
        row = tk.Frame(parent)
        row.pack(fill="x", pady=(8, 0))

        tk.Label(row, text="t_shift range:").pack(side="left")

        self._insp_min_var = tk.StringVar(value=str(self._slider_min))
        min_entry = tk.Entry(row, textvariable=self._insp_min_var, width=8)
        min_entry.pack(side="left", padx=(4, 0))

        current = (self._t_shift_trial[self._selected_point_index]
                   if 0 <= self._selected_point_index < len(self._t_shift_trial) else 0.0)
        self._insp_shift_var = tk.DoubleVar(value=current)
        self._insp_shift_scale = tk.Scale(
            row, variable=self._insp_shift_var, orient=tk.HORIZONTAL,
            from_=self._slider_min, to=self._slider_max,
            resolution=0.001, showvalue=True, length=280,
            # PERFORMANCE: same drag/release split as before. Dragging
            # updates ONLY the selected point's trial shift value and
            # the single-point trajectory overlay -- no triangulation
            # runs until the mouse is released.
            command=lambda _v: self._on_shift_dragging(),
        )
        self._insp_shift_scale.pack(side="left", fill="x", expand=True, padx=6)

        self._insp_max_var = tk.StringVar(value=str(self._slider_max))
        max_entry = tk.Entry(row, textvariable=self._insp_max_var, width=8)
        max_entry.pack(side="left")

        min_entry.bind("<Return>", lambda _e: self._apply_slider_range())
        max_entry.bind("<Return>", lambda _e: self._apply_slider_range())

        self._insp_shift_scale.bind("<ButtonRelease-1>", lambda _e: self._on_shift_released())
        self._insp_shift_scale.bind("<KeyRelease>", lambda _e: self._on_shift_released())

    def _apply_slider_range(self) -> None:
        try:
            lo = float(self._insp_min_var.get())
            hi = float(self._insp_max_var.get())
        except ValueError:
            return
        if hi <= lo:
            return
        self._slider_min, self._slider_max = lo, hi
        self._insp_shift_scale.configure(from_=lo, to=hi)

        self._ensure_trial_shift_size()
        p = self._selected_point_index
        if 0 <= p < len(self._t_shift_trial):
            clamped = min(max(self._t_shift_trial[p], lo), hi)
            self._t_shift_trial[p] = clamped
            self._insp_shift_var.set(clamped)
        self._recompute_preview()

    def _on_shift_dragging(self) -> None:
        self._ensure_trial_shift_size()
        p = self._selected_point_index
        if 0 <= p < len(self._t_shift_trial):
            self._t_shift_trial[p] = self._insp_shift_var.get()
        self._redraw_trajectory_plot()

    def _on_shift_released(self) -> None:
        self._ensure_trial_shift_size()
        p = self._selected_point_index
        if 0 <= p < len(self._t_shift_trial):
            self._t_shift_trial[p] = self._insp_shift_var.get()
        self._recompute_preview()

    # -- Auto Align (coarse grid search) + Refine (local optimization) ------

    def _on_auto_align_clicked(self) -> None:
        """Runs the shared-grid, batched sweep across ALL points
        (cheap, same cost profile regardless of num_points -- see
        _run_coarse_grid_search), then commits the resulting per-point
        best candidates to the TRIAL shifts of either just the selected
        point or every point, depending on the Scope combobox.
        """
        if self._imgpts1 is None:
            messagebox.showwarning("Auto Align", "No input data connected.", parent=self._inspector_win)
            return
        self._ensure_trial_shift_size()

        scan = self._run_coarse_grid_search()
        if scan is None:
            messagebox.showwarning("Auto Align", "Could not run grid search (missing input).",
                                    parent=self._inspector_win)
            return
        shifts, error_matrix = scan
        self._scan_shifts, self._scan_error_matrix = shifts, error_matrix

        num_points = error_matrix.shape[0]
        best_idx_per_point = np.argmin(error_matrix, axis=1)

        scope = self._insp_scope_var.get()
        targets = range(num_points) if scope == "all points" else [self._selected_point_index]
        for p in targets:
            if not (0 <= p < num_points):
                continue
            if not np.isfinite(error_matrix[p, best_idx_per_point[p]]):
                continue  # this point never produced a valid sample anywhere in the grid
            self._t_shift_trial[p] = float(shifts[best_idx_per_point[p]])

        if 0 <= self._selected_point_index < len(self._t_shift_trial):
            self._insp_shift_var.set(self._t_shift_trial[self._selected_point_index])

        self._redraw_scan_plot()
        self._recompute_preview()

    def _on_refine_clicked(self) -> None:
        """Runs an independent local optimizer per target point.

        PERFORMANCE WARNING (see the module docstring and
        _evaluate_cost_single_point): unlike Auto Align, this genuinely
        costs O(num_points) when the scope is "all points" -- each point
        needs its own local search of typically a few dozen single-point
        pipeline evaluations. Use Auto Align first to get every point
        close, so each point's local search here converges quickly.
        """
        if self._imgpts1 is None:
            messagebox.showwarning("Refine", "No input data connected.", parent=self._inspector_win)
            return
        self._ensure_trial_shift_size()

        num_points = self._imgpts1.shape[0]
        scope = self._insp_scope_var.get()
        targets = list(range(num_points)) if scope == "all points" else [self._selected_point_index]

        if scope == "all points" and num_points > self.REFINE_ALL_POINTS_WARNING_THRESHOLD:
            proceed = messagebox.askyesno(
                "Refine All Points",
                f"This runs an independent local optimization for each of the "
                f"{num_points} points, which can take a while. Continue?",
                parent=self._inspector_win,
            )
            if not proceed:
                return

        failed = []
        for p in targets:
            if not (0 <= p < num_points) or p in self._excluded_points:
                continue

            def _cost(shift: float, point_index=p) -> float:
                agg = self._evaluate_cost_single_point(point_index, shift)
                return agg if agg is not None else float("inf")

            result = minimize_scalar(_cost, bounds=(self._slider_min, self._slider_max), method="bounded")
            if result.success:
                self._t_shift_trial[p] = float(result.x)
            else:
                failed.append(p)

        if 0 <= self._selected_point_index < len(self._t_shift_trial):
            self._insp_shift_var.set(self._t_shift_trial[self._selected_point_index])
        if failed:
            messagebox.showwarning(
                "Refine", f"Local optimization did not converge for point(s): {failed}",
                parent=self._inspector_win,
            )
        self._recompute_preview()

    # -- Apply ----------------------------------------------------------------

    def _on_apply_clicked(self) -> None:
        if self._imgpts1 is None:
            messagebox.showwarning("Apply", "No input data connected.", parent=self._inspector_win)
            return

        self._t_shift_applied = self._t_shift_trial.copy()
        self._nan_fill_method_applied = self._nan_fill_method
        self._interp_method_applied = self._interp_method
        self._excluded_points_applied = set(self._excluded_points)

        if self._request_downstream is not None:
            self._request_downstream(self.node_id)