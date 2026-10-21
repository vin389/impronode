# node_editor/calib_engine.py
"""
Camera calibration engine (no Tk) for the Camera Calibration node.

One weighted least-squares problem covers every use case:

  parameters   fx, fy, cx, cy; distortion (OpenCV order k1 k2 p1 p2 k3 k4 k5 k6
               s1 s2 s3 s4 taux tauy, as many as the model uses); per view a
               rotation vector and a translation vector; one scalar per object
               point with an unknown coordinate ('?'). Any intrinsic /
               distortion parameter can be FIXED; a view's pose can be fixed.

  residuals (pixels, each group with its own weight)
    point      projection of a known object point minus its image point
    unknown    the same for a point with ONE unknown coordinate: the object
               point lies on a 3D line (the unknown axis); its coordinate is a
               free parameter, so the optimum is the smallest reprojection
               error possible along that line -- exactly, distortion included.
    line       image points of one straight 3D line whose 3D coordinates are
               unknown: undistorted to the normalised image plane, where a
               3D line is a straight line (pinhole), a line is fitted (total
               least squares) and each point's perpendicular distance to it,
               times the mean focal length (-> pixels), is the residual
               (plumb-line constraint: mainly fixes the distortion).
    chessboard reprojection of the corners of a chessboard view.

Solved with scipy.optimize.least_squares (optionally Huber-robust), using
the sparsity of the Jacobian: a view's residuals depend only on the shared
intrinsics / distortion, its own pose and its own unknowns.

When a problem needs nothing beyond OpenCV -- no '?' unknowns, no lines, no
Huber loss, equal weights, fixed parameters that OpenCV's flags can express
-- run_plan() uses cv2.calibrateCamera (and cv2.solvePnP for the points
image's pose in 'two steps') directly: the same least-squares problem,
solved much faster. solver_route() tells which way a plan is solved and why.

Coordinates follow OpenCV: image (0, 0) is the centre of the top-left pixel.
"""

from __future__ import annotations

import dataclasses
import math
import re
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

INTR_NAMES = ("fx", "fy", "cx", "cy")
DIST_NAMES = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6",
              "s1", "s2", "s3", "s4", "taux", "tauy")
DIST_MODELS = {"simple": 5, "rational": 8, "thin_prism": 12, "tilted": 14}
RESIDUAL_TYPES = ("point", "unknown", "line", "chessboard")

_UNDIST_CRITERIA = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 50, 1e-12)
# The OpenCV route uses cv2.calibrateCamera's own stopping rule (30
# iterations). With many free coefficients (e.g. rational k1-k6 all free) the
# minimum is a long flat valley: OpenCV then stops a hair above the
# least-squares optimum (RMS within ~1e-4 px), while more iterations cost far
# more time than the least-squares solver. plan['force_lsq'] refines it.


class Cancelled(Exception):
    """Raised inside a solve when its cancel event is set."""


# ── object-point cells: "12.5", "?", "?=40" ─────────────────────────────────

_CELL_RE = re.compile(r"\s*\?\s*(?:=\s*([-+0-9.eE]+))?\s*")


def parse_object_cell(text) -> tuple[float, bool, float | None]:
    """'12.5' -> (12.5, False, None); '?' -> (nan, True, None);
    '?=40' -> (nan, True, 40.0). Raises ValueError for anything else."""
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        return float(text), False, None
    s = str(text).strip()
    m = _CELL_RE.fullmatch(s)
    if m:
        return math.nan, True, (float(m.group(1)) if m.group(1) else None)
    return float(s), False, None


# ── problem description ───────────────────────────────────────────────────

@dataclass
class View:
    """One image. kind 'points': object / image points (object coordinates
    may contain ONE unknown each, NaN + unknown_axis) and straight-line
    groups; kind 'chessboard': corners of a board view."""
    name: str
    kind: str = "points"                       # "points" | "chessboard"
    obj: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))   # (n, 3), NaN = unknown
    img: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))   # (n, 2)
    weights: np.ndarray | None = None          # (n,) per-point weights (default 1)
    unknown_init: dict = field(default_factory=dict)                    # point index -> initial value
    lines: list = field(default_factory=list)  # [(m, 2) image points of one 3D line]
    line_weights: list | None = None
    rvec: np.ndarray | None = None             # initial / fixed pose
    tvec: np.ndarray | None = None
    fix_pose: bool = False


@dataclass
class Problem:
    image_size: tuple[int, int]                # (width, height)
    views: list
    K: np.ndarray | None = None                # initial camera matrix (None: guessed)
    D: np.ndarray | None = None                # initial distortion
    dist_model: str = "simple"
    fixed: set = field(default_factory=set)    # names from INTR_NAMES / DIST_NAMES
    fix_aspect: bool = False                   # fy / fx kept at its initial ratio
    weights: dict = field(default_factory=lambda: {t: 1.0 for t in RESIDUAL_TYPES})
    loss: str = "linear"                       # "linear" | "huber"
    huber_px: float = 1.0
    max_nfev: int | None = None


@dataclass
class Result:
    K: np.ndarray
    D: np.ndarray
    rvecs: list
    tvecs: list
    unknowns: dict                             # (view, point) -> solved coordinate
    rms: dict                                  # residual type -> RMS pixels ("all" too)
    per_point: list                            # per view (n, 2) residual vectors (NaN if unused)
    projected: list                            # per view (n, 2) projected points
    line_fits: list                            # per view, per line: dict (see line_geometry)
    cost: float
    success: bool
    message: str
    nfev: int
    seconds: float


# ── validation ────────────────────────────────────────────────────────────

def unknown_axes(obj: np.ndarray) -> list:
    """Per point: None (fully known), axis 0/1/2 (one unknown), or 'all'
    (no object point: an image-only point) / 'many' (2 unknowns: invalid)."""
    out = []
    for p in np.asarray(obj, dtype=float).reshape(-1, 3):
        nan = np.isnan(p)
        n = int(nan.sum())
        out.append(None if n == 0 else int(np.argmax(nan)) if n == 1 else "all" if n == 3 else "many")
    return out


def check_view(view: View) -> list[str]:
    """Problems that make (parts of) a points view unusable, as messages."""
    msgs = []
    for i, a in enumerate(unknown_axes(view.obj)):
        if a == "many":
            msgs.append(f"point {i + 1}: two unknown coordinates -- one image point can only fix "
                        f"ONE unknown (a viewing ray meets a 3D line, not a plane); the point is ignored")
    for j, ln in enumerate(view.lines):
        if len(ln) < 3:
            msgs.append(f"line {j + 1}: needs at least 3 image points (has {len(ln)})")
    return msgs


# ── geometry helpers ──────────────────────────────────────────────────────

def guess_K(image_size) -> np.ndarray:
    w, h = image_size
    f = float(max(w, h))
    return np.array([[f, 0, (w - 1) / 2.0], [0, f, (h - 1) / 2.0], [0, 0, 1.0]])


def undistort_normalized(img_pts, K, D) -> np.ndarray:
    """Image pixels -> undistorted normalised coordinates (x / z, y / z)."""
    p = np.asarray(img_pts, dtype=np.float64).reshape(-1, 1, 2)
    if len(p) == 0:
        return np.zeros((0, 2))
    return cv2.undistortPointsIter(p, K, D, None, None, _UNDIST_CRITERIA).reshape(-1, 2)


def distort_normalized(norm_pts, K, D) -> np.ndarray:
    """Undistorted normalised coordinates -> image pixels (with distortion)."""
    n = np.asarray(norm_pts, dtype=np.float64).reshape(-1, 2)
    if len(n) == 0:
        return np.zeros((0, 2))
    obj = np.column_stack([n, np.ones(len(n))])
    px, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), K, D)
    return px.reshape(-1, 2)


def project(obj, rvec, tvec, K, D) -> np.ndarray:
    o = np.asarray(obj, dtype=np.float64).reshape(-1, 3)
    if len(o) == 0:
        return np.zeros((0, 2))
    px, _ = cv2.projectPoints(o, np.asarray(rvec, float), np.asarray(tvec, float), K, D)
    return px.reshape(-1, 2)


def fit_line_normalized(norm: np.ndarray):
    """Total-least-squares line through (m, 2) points: (centre, unit
    direction, unit normal)."""
    c = norm.mean(axis=0)
    _u, _s, vt = np.linalg.svd(norm - c, full_matrices=False)
    d = vt[0]
    return c, d, np.array([-d[1], d[0]])


def line_geometry(img_pts, K, D, n_curve: int = 60) -> dict:
    """For drawing / reporting one straight-line group with the current
    camera: the fitted line as an image polyline (curved by the distortion),
    each point's foot on it (pixels) and its residual (pixels)."""
    pts = np.asarray(img_pts, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return {"curve": np.zeros((0, 2)), "feet": pts.copy(), "residual": np.zeros(len(pts))}
    norm = undistort_normalized(pts, K, D)
    c, d, nrm = fit_line_normalized(norm)
    s = (norm - c) @ d
    span = s.max() - s.min()
    ss = np.linspace(s.min() - 0.08 * span, s.max() + 0.08 * span, n_curve)
    f = 0.5 * (K[0, 0] + K[1, 1])
    return {"curve": distort_normalized(c + ss[:, None] * d, K, D),
            "feet": distort_normalized(c + s[:, None] * d, K, D),
            "residual": ((norm - c) @ nrm) * f}


def project_grid(xs, ys, zs, rvec, tvec, K, D, samples: int = 24) -> list:
    """Polylines (image pixels) of a 3D grid: lines along x for every (y, z),
    along y for every (x, z) and along z for every (x, y), each densified
    so the distortion curves it correctly. Lines with a point behind the
    camera are dropped."""
    xs, ys, zs = (np.atleast_1d(np.asarray(a, dtype=np.float64)) for a in (xs, ys, zs))
    R, _ = cv2.Rodrigues(np.asarray(rvec, float))
    t = np.asarray(tvec, float).reshape(3)
    out = []

    def add(a, b):
        seg = np.linspace(0.0, 1.0, samples)[:, None] * (b - a) + a
        if np.all((seg @ R.T + t)[:, 2] > 1e-9):
            out.append(project(seg, rvec, tvec, K, D))
    for axis, (u, v, w) in enumerate(((xs, ys, zs), (ys, xs, zs), (zs, xs, ys))):
        if len(u) < 2:
            continue
        for p in v:
            for q in w:
                a, b = np.zeros(3), np.zeros(3)
                a[axis], b[axis] = u.min(), u.max()
                others = [i for i in range(3) if i != axis]
                a[others[0]] = b[others[0]] = p
                a[others[1]] = b[others[1]] = q
                add(a, b)
    return out


def solve_unknown_closest(base, axis, img_pt, R, t, K, D) -> float:
    """Initial value of an unknown coordinate: the point of the object line
    X(s) = base + s e_axis closest to the viewing ray of the image point."""
    r = np.append(undistort_normalized(img_pt, K, D)[0], 1.0)
    r /= np.linalg.norm(r)
    P = np.eye(3) - np.outer(r, r)
    e = np.zeros(3)
    e[axis] = 1.0
    b0 = np.where(np.isnan(base), 0.0, base)
    a_vec = P @ (R @ e)
    b_vec = P @ (R @ b0 + t)
    aa = float(a_vec @ a_vec)
    return -float(a_vec @ b_vec) / aa if aa > 1e-18 else 0.0


# ── the solver ────────────────────────────────────────────────────────────

class _Layout:
    """Maps the free parameters to positions in the optimisation vector."""

    def __init__(self, prob: Problem, K0, D0, poses):
        self.prob, self.K0, self.D0 = prob, K0, D0
        nd = DIST_MODELS[prob.dist_model]
        self.intr = [n for n in INTR_NAMES if n not in prob.fixed and not (prob.fix_aspect and n == "fy")]
        self.dist = [n for n in DIST_NAMES[:nd] if n not in prob.fixed]
        self.nd = nd
        self.ratio = K0[1, 1] / K0[0, 0]
        pos = len(self.intr) + len(self.dist)
        self.n_global = pos
        self.pose_at = []
        for v in prob.views:
            if v.fix_pose:
                self.pose_at.append(None)
            else:
                self.pose_at.append(pos)
                pos += 6
        self.unknowns = []                     # (view, point, axis, position)
        self.axes = []
        for vi, v in enumerate(prob.views):
            ax = unknown_axes(v.obj) if v.kind == "points" else [None] * len(v.obj)
            self.axes.append(ax)
            for i, a in enumerate(ax):
                if isinstance(a, int):
                    self.unknowns.append((vi, i, a, pos))
                    pos += 1
        self.n = pos
        self.poses = poses

    def pack(self, K, D, poses, unknown_vals) -> np.ndarray:
        x = np.zeros(self.n)
        vals = {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2]}
        for j, n in enumerate(self.intr):
            x[j] = vals[n]
        for j, n in enumerate(self.dist):
            x[len(self.intr) + j] = D[DIST_NAMES.index(n)]
        for vi, at in enumerate(self.pose_at):
            if at is not None:
                x[at:at + 3], x[at + 3:at + 6] = poses[vi][0], poses[vi][1]
        for vi, i, _a, at in self.unknowns:
            x[at] = unknown_vals[(vi, i)]
        return x

    def unpack(self, x):
        vals = {"fx": self.K0[0, 0], "fy": self.K0[1, 1], "cx": self.K0[0, 2], "cy": self.K0[1, 2]}
        for j, n in enumerate(self.intr):
            vals[n] = x[j]
        if self.prob.fix_aspect:
            vals["fy"] = vals["fx"] * self.ratio
        K = np.array([[vals["fx"], 0, vals["cx"]], [0, vals["fy"], vals["cy"]], [0, 0, 1.0]])
        D = np.array(self.D0, dtype=np.float64).copy()
        for j, n in enumerate(self.dist):
            D[DIST_NAMES.index(n)] = x[len(self.intr) + j]
        poses = []
        for vi, at in enumerate(self.pose_at):
            poses.append(self.poses[vi] if at is None else (x[at:at + 3], x[at + 3:at + 6]))
        unk = {(vi, i): x[at] for vi, i, _a, at in self.unknowns}
        return K, D, poses, unk


def _full_obj(view: View, vi: int, axes, unk) -> np.ndarray:
    o = np.array(view.obj, dtype=np.float64).reshape(-1, 3).copy()
    for i, a in enumerate(axes):
        if isinstance(a, int):
            o[i, a] = unk[(vi, i)]
    return o


def _point_mask(axes) -> np.ndarray:
    """Points that take part in point / unknown residuals."""
    return np.array([a is None or isinstance(a, int) for a in axes], dtype=bool)


def _residual_blocks(prob: Problem, lay: _Layout, x):
    """Residual vector pieces, in a fixed order, with their type and
    (view, point) owner -- shared by the solver, the sparsity pattern and
    the reports."""
    K, D, poses, unk = lay.unpack(x)
    w = prob.weights
    blocks = []
    for vi, v in enumerate(prob.views):
        axes, mask = lay.axes[vi], _point_mask(lay.axes[vi])
        if mask.any():
            obj = _full_obj(v, vi, axes, unk)
            proj = project(obj[mask], poses[vi][0], poses[vi][1], K, D)
            res = proj - np.asarray(v.img, dtype=np.float64).reshape(-1, 2)[mask]
            pw = (np.ones(len(v.obj)) if v.weights is None else np.asarray(v.weights, float))[mask]
            kinds = ["chessboard" if v.kind == "chessboard" else ("unknown" if isinstance(a, int) else "point")
                     for a in np.array(axes, dtype=object)[mask]]
            tw = np.array([w.get(k, 1.0) for k in kinds])
            blocks.append(("points", vi, np.flatnonzero(mask), kinds,
                           res, np.sqrt(np.maximum(tw * pw, 0.0))))
        for j, ln in enumerate(v.lines):
            if len(ln) < 3:
                continue
            norm = undistort_normalized(ln, K, D)
            c, _d, nrm = fit_line_normalized(norm)
            f = 0.5 * (K[0, 0] + K[1, 1])
            lw = 1.0 if v.line_weights is None else float(v.line_weights[j])
            blocks.append(("line", vi, j, ["line"] * len(ln), ((norm - c) @ nrm) * f,
                           np.full(len(ln), math.sqrt(max(w.get("line", 1.0) * lw, 0.0)))))
    return blocks, (K, D, poses, unk)


def _residual_vector(prob, lay, x) -> np.ndarray:
    blocks, _ = _residual_blocks(prob, lay, x)
    parts = []
    for b in blocks:
        r, sw = b[4], b[5]
        parts.append((r * sw[:, None]).ravel() if r.ndim == 2 else r * sw)
    return np.concatenate(parts) if parts else np.zeros(0)


def _sparsity(prob, lay, x):
    blocks, _ = _residual_blocks(prob, lay, x)
    rows = sum(b[4].size for b in blocks)
    S = lil_matrix((rows, lay.n), dtype=np.int8)
    r0 = 0
    unk_at = {(vi, i): at for vi, i, _a, at in lay.unknowns}
    for b in blocks:
        n_r = b[4].size
        S[r0:r0 + n_r, :lay.n_global] = 1                 # intrinsics / distortion
        if b[0] == "points":
            vi, idx = b[1], b[2]
            at = lay.pose_at[vi]
            if at is not None:
                S[r0:r0 + n_r, at:at + 6] = 1
            for k, i in enumerate(idx):
                u = unk_at.get((vi, int(i)))
                if u is not None:
                    S[r0 + 2 * k:r0 + 2 * k + 2, u] = 1
        r0 += n_r
    return S


def _column_groups(lay: _Layout) -> list:
    """Columns that can be perturbed TOGETHER in one finite-difference step
    because they affect disjoint residual rows: each global (intrinsic /
    distortion) column alone; pose component j of every view together; all
    unknown coordinates together. A Jacobian then costs n_global + 7
    residual evaluations, whatever the number of views."""
    groups = [[j] for j in range(lay.n_global)]
    for comp in range(6):
        g = [at + comp for at in lay.pose_at if at is not None]
        if g:
            groups.append(g)
    if lay.unknowns:
        groups.append([at for _vi, _i, _a, at in lay.unknowns])
    return groups


def _make_jacobian(prob: Problem, lay: _Layout, sparsity):
    """Forward-difference Jacobian using _column_groups and the sparsity
    pattern (rows a column can change). Returned dense, so least_squares
    can use its exact trust-region solver (the iterative sparse solver
    converged very slowly on these problems)."""
    mask = sparsity.toarray().astype(bool)
    groups = _column_groups(lay)

    def jac(x, *_args):
        f0 = _residual_vector(prob, lay, x)
        J = np.zeros((f0.size, x.size))
        for g in groups:
            h = np.sqrt(np.finfo(float).eps) * np.maximum(1.0, np.abs(x[g]))
            xp = x.copy()
            xp[g] += h
            df = _residual_vector(prob, lay, xp) - f0
            for col, step in zip(g, h):
                rows = mask[:, col]
                J[rows, col] = df[rows] / step
        return J
    return jac


def initial_poses(prob: Problem, K, D) -> list:
    """A pose per view: the given one, or solvePnP on its fully known points."""
    poses = []
    for v in prob.views:
        if v.rvec is not None and v.tvec is not None:
            poses.append((np.asarray(v.rvec, float).reshape(3), np.asarray(v.tvec, float).reshape(3)))
            continue
        axes = unknown_axes(v.obj)
        known = np.array([a is None for a in axes], dtype=bool)
        if known.sum() < 4:
            raise ValueError(f"view '{v.name}': at least 4 points with fully known 3D coordinates "
                             f"are needed to start (has {int(known.sum())})")
        obj = np.asarray(v.obj, float)[known]
        img = np.asarray(v.img, float).reshape(-1, 2)[known]
        ok, rv, tv = cv2.solvePnP(obj, img, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise ValueError(f"view '{v.name}': cannot find an initial camera pose (solvePnP)")
        poses.append((rv.reshape(3), tv.reshape(3)))
    return poses


def initial_unknowns(prob: Problem, K, D, poses) -> dict:
    vals = {}
    for vi, v in enumerate(prob.views):
        if v.kind != "points":
            continue
        R, _ = cv2.Rodrigues(np.asarray(poses[vi][0], float))
        t = np.asarray(poses[vi][1], float).reshape(3)
        for i, a in enumerate(unknown_axes(v.obj)):
            if isinstance(a, int):
                init = v.unknown_init.get(i)
                vals[(vi, i)] = float(init) if init is not None else solve_unknown_closest(
                    np.asarray(v.obj[i], float), a, np.asarray(v.img, float).reshape(-1, 2)[i], R, t, K, D)
    return vals


def evaluate(prob: Problem, K, D, poses, unknowns, t0: float | None = None,
             opt=None, lay: _Layout | None = None) -> Result:
    """Residuals and statistics for given parameters (also used live by the
    manual-adjust sliders)."""
    nd = DIST_MODELS[prob.dist_model]
    d = np.asarray(D, dtype=np.float64).ravel()
    D = np.zeros(nd)
    D[:min(nd, len(d))] = d[:nd]                      # the model's coefficients, zero-padded
    if lay is None:
        lay = _Layout(prob, K, D, poses)
    x = lay.pack(K, D, poses, unknowns)
    blocks, (K, D, poses, unk) = _residual_blocks(prob, lay, x)
    sq = {t: [] for t in RESIDUAL_TYPES}
    per_point = [np.full((len(v.obj), 2), np.nan) for v in prob.views]
    projected = [np.full((len(v.obj), 2), np.nan) for v in prob.views]
    line_fits = [[None] * len(v.lines) for v in prob.views]
    for b in blocks:
        if b[0] == "points":
            vi, idx, kinds, res = b[1], b[2], b[3], b[4]
            per_point[vi][idx] = res
            projected[vi][idx] = res + np.asarray(prob.views[vi].img, float).reshape(-1, 2)[idx]
            for k, kind in enumerate(kinds):
                sq[kind].append(float(res[k] @ res[k]))
        else:
            vi, j, res = b[1], b[2], b[4]
            sq["line"].extend((res * res).tolist())
    for vi, v in enumerate(prob.views):
        for j, ln in enumerate(v.lines):
            if len(ln) >= 2:
                line_fits[vi][j] = line_geometry(ln, K, D)
    rms = {}
    for t, vals in sq.items():
        if vals:
            # point-like residuals are 2-D: RMS of the distance; lines are 1-D.
            rms[t] = math.sqrt(sum(vals) / len(vals))
    allv = [v for vals in sq.values() for v in vals]
    rms["all"] = math.sqrt(sum(allv) / len(allv)) if allv else math.nan
    cost = float(0.5 * np.sum(_residual_vector(prob, lay, x) ** 2))
    return Result(K=K, D=D, rvecs=[p[0] for p in poses], tvecs=[p[1] for p in poses], unknowns=unk,
                  rms=rms, per_point=per_point, projected=projected, line_fits=line_fits, cost=cost,
                  success=bool(getattr(opt, "success", True)), message=str(getattr(opt, "message", "")),
                  nfev=int(getattr(opt, "nfev", 0)), seconds=(time.perf_counter() - t0) if t0 else 0.0)


def solve(prob: Problem, cancel=None) -> Result:
    """Run the weighted (optionally robust) least-squares calibration.
    `cancel`: a threading.Event; when set, the solve stops with Cancelled at
    the next residual evaluation."""
    t0 = time.perf_counter()
    nd = DIST_MODELS[prob.dist_model]
    K0 = np.array(prob.K, dtype=np.float64) if prob.K is not None else guess_K(prob.image_size)
    D0 = np.zeros(nd)
    if prob.D is not None:
        d = np.asarray(prob.D, dtype=np.float64).ravel()
        D0[:min(nd, len(d))] = d[:nd]
    poses = initial_poses(prob, K0, D0)
    unk0 = initial_unknowns(prob, K0, D0, poses)
    lay = _Layout(prob, K0, D0, poses)
    x0 = lay.pack(K0, D0, poses, unk0)
    if lay.n == 0:
        return evaluate(prob, K0, D0, poses, unk0, t0)
    n_res = _residual_vector(prob, lay, x0).size
    if n_res < lay.n:
        raise ValueError(f"not enough measurements: {n_res} residuals for {lay.n} unknown parameters "
                         f"-- add points / lines, or fix more parameters")
    jac = _make_jacobian(prob, lay, _sparsity(prob, lay, x0))

    def fun(x):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        return _residual_vector(prob, lay, x)

    def jac_c(x, *args):
        if cancel is not None and cancel.is_set():
            raise Cancelled()
        return jac(x, *args)

    opt = least_squares(fun, x0, jac=jac_c, x_scale="jac", method="trf",
                        loss="huber" if prob.loss == "huber" else "linear",
                        f_scale=prob.huber_px, max_nfev=prob.max_nfev or 200 * lay.n,
                        xtol=1e-12, ftol=1e-12, gtol=1e-12)
    K, D, poses, unk = lay.unpack(opt.x)
    return evaluate(prob, K, D, poses, unk, t0, opt, lay)


def model_flags(dist_model: str) -> int:
    return {"rational": cv2.CALIB_RATIONAL_MODEL,
            "thin_prism": cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL,
            "tilted": cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL | cv2.CALIB_TILTED_MODEL
            }.get(dist_model, 0)


def chessboard_initial(views, image_size, dist_model="simple", extra_flags: int = 0, criteria=None):
    """cv2.calibrateCamera on the chessboard views (planar targets): a good
    start for intrinsics / distortion and every board's pose (criteria None =
    OpenCV's default; a start does not need to be exact)."""
    objs = [np.asarray(v.obj, np.float32).reshape(-1, 3) for v in views]
    imgs = [np.asarray(v.img, np.float32).reshape(-1, 1, 2) for v in views]
    rms, K, D, rv, tv = cv2.calibrateCamera(objs, imgs, tuple(int(s) for s in image_size), None, None,
                                            flags=model_flags(dist_model) | extra_flags,
                                            **({} if criteria is None else {"criteria": criteria}))
    return K, D.ravel(), [r.reshape(3) for r in rv], [t.reshape(3) for t in tv], rms


# ── which solver: OpenCV alone or the least-squares solver ─────────────────

def _fixed_value(prob: Problem, user: dict, name: str) -> float:
    """The value a FIXED distortion parameter keeps (merge_initial)."""
    v = user.get(name)
    return 0.0 if v is None else float(v)


def opencv_camera_flags(prob: Problem, user: dict):
    """(flags with values, flags fixing at zero, reasons): how the fixed
    parameters of `prob` map to cv2.calibrateCamera flags.
    flags with values -- need CALIB_USE_INTRINSIC_GUESS (start values given);
    flags at zero      -- also work without start values (stage 1);
    reasons            -- why OpenCV cannot solve it (empty = it can)."""
    reasons = []
    fixed = prob.fixed
    nd = DIST_MODELS[prob.dist_model]
    names = DIST_NAMES[:nd]
    fx_free = "fx" not in fixed
    fy_free = "fy" not in fixed and not prob.fix_aspect
    guess, zero = 0, 0
    if not fx_free and not fy_free:
        guess |= cv2.CALIB_FIX_FOCAL_LENGTH
    elif fx_free and not fy_free:
        if prob.fix_aspect:
            guess |= cv2.CALIB_FIX_ASPECT_RATIO
        else:
            reasons.append("only fy is fixed (OpenCV fixes fx and fy together)")
    elif not fx_free and fy_free:
        reasons.append("only fx is fixed (OpenCV fixes fx and fy together)")
    cxy = [n in fixed for n in ("cx", "cy")]
    if all(cxy):
        guess |= cv2.CALIB_FIX_PRINCIPAL_POINT
    elif any(cxy):
        reasons.append("only one of cx / cy is fixed (OpenCV fixes both together)")
    for k, flag in enumerate((cv2.CALIB_FIX_K1, cv2.CALIB_FIX_K2, cv2.CALIB_FIX_K3, cv2.CALIB_FIX_K4,
                              cv2.CALIB_FIX_K5, cv2.CALIB_FIX_K6), start=1):
        n = f"k{k}"
        if n in names and n in fixed:
            guess |= flag
            if _fixed_value(prob, user, n) == 0.0:
                zero |= flag

    def group(group_names, flag, zero_flag, label):
        nonlocal guess, zero
        present = [n for n in group_names if n in names]
        if not present:
            return
        f = [n in fixed for n in present]
        if all(f):
            guess |= flag
            if all(_fixed_value(prob, user, n) == 0.0 for n in present):
                zero |= zero_flag
        elif any(f):
            reasons.append(f"only some of {label} are fixed (OpenCV fixes them together)")

    group(("p1", "p2"), cv2.CALIB_FIX_TANGENT_DIST, cv2.CALIB_ZERO_TANGENT_DIST, "p1 / p2")
    group(("s1", "s2", "s3", "s4"), cv2.CALIB_FIX_S1_S2_S3_S4, cv2.CALIB_FIX_S1_S2_S3_S4, "s1-s4")
    group(("taux", "tauy"), cv2.CALIB_FIX_TAUX_TAUY, cv2.CALIB_FIX_TAUX_TAUY, "taux / tauy")
    if prob.loss == "huber":
        reasons.append("robust (huber) loss")
    return guess, zero, reasons


def _points_view_reasons(view: View, prob: Problem) -> list:
    """Why the points view needs the least-squares solver (empty = OpenCV)."""
    reasons = []
    axes = unknown_axes(view.obj)
    if any(isinstance(a, int) for a in axes):
        reasons.append("'?' unknown coordinates in the Points table")
    if any(len(ln) >= 3 for ln in view.lines):
        reasons.append("straight lines in the Points table")
    known = np.array([a is None for a in axes], dtype=bool)
    w = np.ones(len(axes)) if view.weights is None else np.asarray(view.weights, float)
    if known.any() and np.ptp(w[known]) > 0:
        reasons.append("Points rows with different weights")
    if prob.weights.get("point", 1.0) <= 0:
        reasons.append("the 'known points' weight is 0")
    return reasons


def solver_route(plan: dict) -> dict:
    """How run_plan() solves `plan`:
    boards  'opencv' (cv2.calibrateCamera) | 'lsq' (least squares) | None
    points  'opencv' (with the boards / alone in calibrateCamera) | 'pnp'
            (cv2.solvePnP, two steps) | 'lsq' | None
    reasons why the least-squares solver is needed (empty = OpenCV only)
    text    a one-line summary for the user."""
    prob: Problem = plan["prob"]
    has_pts = plan["pts"] is not None
    views = list(prob.views)
    cbv = views[1:] if has_pts else views
    _g, _z, cam_reasons = opencv_camera_flags(prob, plan.get("user_init", {}))
    if cbv and prob.weights.get("chessboard", 1.0) <= 0:
        cam_reasons.append("the 'board corners' weight is 0")
    pt_reasons = _points_view_reasons(views[0], prob) if has_pts else []
    boards = points = None
    if cbv and has_pts and plan.get("mode") == "two_step":
        boards = "lsq" if cam_reasons else "opencv"
        loss = ["robust (huber) loss"] if prob.loss == "huber" else []
        points = "lsq" if (pt_reasons or loss) else "pnp"
        reasons = cam_reasons + [r for r in pt_reasons + loss if r not in cam_reasons]
    else:
        reasons = cam_reasons + pt_reasons
        if cbv and has_pts and prob.weights.get("point", 1.0) != prob.weights.get("chessboard", 1.0):
            reasons.append("known points and board corners are weighted differently (joint)")
        way = "lsq" if reasons else "opencv"
        boards = way if cbv else None
        points = way if has_pts else None
    if not reasons:
        text = ("OpenCV only: cv2.calibrateCamera" + (" + cv2.solvePnP for the points image" if points == "pnp"
                                                       else "") + " -- no extra minimization (fast)")
    elif boards == "opencv" and points == "lsq":
        text = ("Boards: OpenCV cv2.calibrateCamera; points image (camera fixed): least-squares minimization, "
                "because: " + "; ".join(reasons))
    else:
        text = ("Least-squares minimization (started from OpenCV" if cbv else "Least-squares minimization") + (
            "), because: " if cbv else ", because: ") + "; ".join(reasons)
    return {"boards": boards, "points": points, "reasons": reasons, "text": text}


def calibrate_opencv(prob: Problem, views: list, user: dict, has_pts: bool):
    """cv2.calibrateCamera for `views` (points view first when has_pts):
    (K, D, poses, report). Stage 1 (boards only, no start values) gives the
    start; stage 2 (with start values, CALIB_USE_INTRINSIC_GUESS) is run
    when parameters are fixed at given values or the points view (possibly
    not planar) takes part."""
    size = tuple(int(s) for s in prob.image_size)
    guess, zero, _r = opencv_camera_flags(prob, user)
    nd = DIST_MODELS[prob.dist_model]
    boards = views[1:] if has_pts else views
    report = []
    t = time.perf_counter()
    if boards:
        Kc, Dc, rvs, tvs, rms = chessboard_initial(boards, size, prob.dist_model, zero)
        report.append(f"cv2.calibrateCamera on {len(boards)} board images: RMS {rms:.4f} px "
                      f"({time.perf_counter() - t:.2f} s)")
        K0, D0 = merge_initial(Kc, Dc, user, prob)
        poses = list(zip(rvs, tvs))
    else:
        K0, D0, poses = np.asarray(prob.K, float), np.zeros(nd), []
        if prob.D is not None:
            d = np.asarray(prob.D, float).ravel()
            D0[:min(nd, len(d))] = d[:nd]
    intr_fixed = guess & (cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_PRINCIPAL_POINT)
    if has_pts or intr_fixed or guess != zero:
        t = time.perf_counter()
        objs, imgs = [], []
        for k, v in enumerate(views):
            obj = np.asarray(v.obj, np.float64).reshape(-1, 3)
            img = np.asarray(v.img, np.float64).reshape(-1, 2)
            if has_pts and k == 0:
                known = np.array([a is None for a in unknown_axes(obj)], dtype=bool)
                obj, img = obj[known], img[known]
            objs.append(obj.astype(np.float32))
            imgs.append(img.astype(np.float32).reshape(-1, 1, 2))
        rms, K, D, rvs, tvs = cv2.calibrateCamera(
            objs, imgs, size, K0.copy(), D0.copy(), flags=model_flags(prob.dist_model) | guess
            | cv2.CALIB_USE_INTRINSIC_GUESS)
        poses = [(r.reshape(3), tt.reshape(3)) for r, tt in zip(rvs, tvs)]
        report.append(f"cv2.calibrateCamera with start values{' (points image included)' if has_pts else ''}"
                      f": RMS {rms:.4f} px ({time.perf_counter() - t:.2f} s)")
        K0, D0 = K, D.ravel()
    D = np.zeros(nd)
    d = np.asarray(D0, float).ravel()
    D[:min(nd, len(d))] = d[:nd]
    return np.asarray(K0, float), D, [(np.asarray(r, float).reshape(3), np.asarray(tt, float).reshape(3))
                                      for r, tt in poses], report


# ── solve plans (used by the Camera Calibration node; no Tk) ─────────────────

def merge_initial(base_K, base_D, user: dict, prob: Problem):
    """Start values: a value typed by the user wins; a FIXED parameter left
    empty takes its automatic value (guess_K / 0, as documented); every
    other parameter starts from `base` (e.g. cv2.calibrateCamera)."""
    auto_K = guess_K(prob.image_size)
    vals = {}
    for n, (r, c) in zip(INTR_NAMES, ((0, 0), (1, 1), (0, 2), (1, 2))):
        if user.get(n) is not None:
            vals[n] = user[n]
        elif n in prob.fixed:
            vals[n] = auto_K[r, c]
        else:
            vals[n] = float(base_K[r, c])
    if prob.fix_aspect:
        # fy / fx stays at the ratio of the INITIAL values: typed ones, else
        # the automatic ones (1) -- not the ratio found by a free start fit.
        ratio = ((user.get("fy") if user.get("fy") is not None else auto_K[1, 1])
                 / (user.get("fx") if user.get("fx") is not None else auto_K[0, 0]))
        vals["fy"] = vals["fx"] * ratio
    K = np.array([[vals["fx"], 0, vals["cx"]], [0, vals["fy"], vals["cy"]], [0, 0, 1.0]])
    nd = DIST_MODELS[prob.dist_model]
    bd = np.asarray(base_D, dtype=np.float64).ravel()
    D = np.zeros(nd)
    for j, n in enumerate(DIST_NAMES[:nd]):
        if user.get(n) is not None:
            D[j] = user[n]
        elif n not in prob.fixed and j < len(bd):
            D[j] = bd[j]
    return K, D


def _check_cancel(cancel) -> None:
    if cancel is not None and cancel.is_set():
        raise Cancelled()


def _run_opencv(plan: dict, route: dict) -> dict | None:
    """run_plan() without the least-squares solver, or None when the route
    needs it for the camera (see solver_route)."""
    prob: Problem = plan["prob"]
    cancel = plan.get("cancel")
    views = list(prob.views)
    has_pts = plan["pts"] is not None
    cbv = views[1:] if has_pts else views
    user = plan.get("user_init", {})
    t0 = time.perf_counter()
    out = {"cb_poses": {}, "points_pose": None, "unknowns": {}, "nfev": 0, "success": True}
    if route["points"] == "opencv" or not has_pts:              # one calibrateCamera with every view
        K, D, poses, report = calibrate_opencv(prob, views, user, has_pts)
        _check_cancel(cancel)
        r = evaluate(prob, K, D, poses, {}, t0)
        if has_pts:
            out["points_pose"] = (r.rvecs[0], r.tvecs[0])
        for j, v in enumerate(cbv):
            out["cb_poses"][v.name] = (r.rvecs[j + has_pts], r.tvecs[j + has_pts])
        rms = r.rms.get("all", math.nan)
    else:                                                       # two steps: boards, then the points pose
        bprob = dataclasses.replace(prob, views=cbv)
        K, D, poses, report = calibrate_opencv(bprob, cbv, user, False)
        _check_cancel(cancel)
        r1 = evaluate(bprob, K, D, poses, {}, t0)
        for v, rv, tv in zip(cbv, r1.rvecs, r1.tvecs):
            out["cb_poses"][v.name] = (rv, tv)
        fixed_all = set(INTR_NAMES) | set(DIST_NAMES)
        pprob = dataclasses.replace(prob, views=[views[0]], K=r1.K, D=r1.D, fixed=fixed_all)
        if route["points"] == "pnp":
            v = views[0]
            known = np.array([a is None for a in unknown_axes(v.obj)], dtype=bool)
            obj = np.asarray(v.obj, float).reshape(-1, 3)[known]
            img = np.asarray(v.img, float).reshape(-1, 2)[known]
            ok, rv, tv = cv2.solvePnP(obj, img, r1.K, r1.D, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                raise ValueError("cv2.solvePnP could not find the pose of the points image")
            rv, tv = cv2.solvePnPRefineLM(obj, img, r1.K, r1.D, rv, tv)
            r2 = evaluate(pprob, r1.K, r1.D, [(rv.reshape(3), tv.reshape(3))], {}, t0)
            report.append(f"points image pose: cv2.solvePnP, RMS {r2.rms.get('all', math.nan):.4f} px")
        else:
            r2 = solve(pprob, cancel)
            out["nfev"] = r2.nfev
            out["success"] = r2.success
            report.append(f"points image pose (camera fixed): least squares, RMS "
                          f"{r2.rms.get('all', math.nan):.4f} px, {r2.nfev} evaluations")
            out["unknowns"] = {k: v for (vi, k), v in r2.unknowns.items() if vi == 0}
        out["points_pose"] = (r2.rvecs[0], r2.tvecs[0])
        rms = r1.rms.get("all", math.nan)
    out.update(K=K, D=D, rms=rms, seconds=time.perf_counter() - t0,
               message="OpenCV" + (" (calibrateCamera" + (", solvePnP)" if route["points"] == "pnp" else ")")),
               report="; ".join(report))
    return out


def run_plan(plan: dict) -> dict:
    """Solve a plan made by the node (see _prepare_solve). Views: the points
    view first (if any), then the chessboard views. Returns plain data.
    plan['cancel'] (optional threading.Event) stops it with Cancelled;
    plan['force_lsq'] = True always uses the least-squares solver."""
    prob: Problem = plan["prob"]
    route = solver_route(plan)
    if plan.get("force_lsq"):
        route = dict(route, boards="lsq" if route["boards"] else None, points="lsq" if route["points"] else None,
                     text="Least-squares minimization (started from OpenCV), because 'Always refine' is ticked"
                          + ("" if route["reasons"] else " (OpenCV alone would be enough)")
                          + ("" if not route["reasons"] else "; also needed for: " + "; ".join(route["reasons"])))
    elif route["boards"] in ("opencv", None) and route["points"] in ("opencv", "pnp", None) or (
            route["boards"] == "opencv" and route["points"] == "lsq" and plan.get("mode") == "two_step"):
        out = _run_opencv(plan, route)
        out["route"] = route
        return out
    cancel = plan.get("cancel")
    views = list(prob.views)
    has_pts = plan["pts"] is not None
    cbv = views[1:] if has_pts else views
    report = []
    if cbv:
        Kc, Dc, rvs, tvs, rms_cv = chessboard_initial(cbv, prob.image_size, prob.dist_model)
        for v, rv, tv in zip(cbv, rvs, tvs):
            v.rvec, v.tvec = rv, tv
        prob.K, prob.D = merge_initial(Kc, Dc, plan["user_init"], prob)
        report.append(f"start: cv2.calibrateCamera on {len(cbv)} board images, RMS {rms_cv:.3f} px")
    out = {"cb_poses": {}, "points_pose": None, "unknowns": {}}
    if has_pts and cbv and plan["mode"] == "two_step":
        r1 = solve(dataclasses.replace(prob, views=cbv), cancel)
        fixed_all = set(INTR_NAMES) | set(DIST_NAMES)
        r2 = solve(dataclasses.replace(prob, views=[views[0]], K=r1.K, D=r1.D, fixed=fixed_all), cancel)
        report.append(f"step 1 (boards only): RMS {r1.rms.get('all', math.nan):.4f} px, "
                      f"{'converged' if r1.success else 'NOT converged'}, {r1.nfev} evaluations")
        report.append(f"step 2 (points image pose, camera fixed): RMS {r2.rms.get('all', math.nan):.4f} px, "
                      f"{'converged' if r2.success else 'NOT converged'}, {r2.nfev} evaluations")
        out.update(K=r1.K, D=r1.D, success=r1.success and r2.success, seconds=r1.seconds + r2.seconds,
                   message=r1.message, nfev=r1.nfev + r2.nfev)
        out["points_pose"] = (r2.rvecs[0], r2.tvecs[0])
        out["unknowns"] = {k: v for (vi, k), v in r2.unknowns.items() if vi == 0}
        for v, rv, tv in zip(cbv, r1.rvecs, r1.tvecs):
            out["cb_poses"][v.name] = (rv, tv)
        out["rms"] = r1.rms.get("all", math.nan)
    else:
        r = solve(prob, cancel)
        out.update(K=r.K, D=r.D, success=r.success, seconds=r.seconds, message=r.message, nfev=r.nfev,
                   rms=r.rms.get("all", math.nan))
        if has_pts:
            out["points_pose"] = (r.rvecs[0], r.tvecs[0])
            out["unknowns"] = {k: v for (vi, k), v in r.unknowns.items() if vi == 0}
        for j, v in enumerate(cbv):
            out["cb_poses"][v.name] = (r.rvecs[j + has_pts], r.tvecs[j + has_pts])
    out["report"] = "; ".join(report)
    out["route"] = route
    return out



def loo_task(job: dict) -> dict:
    """Worker-process entry of the leave-one-out sensitivity analysis: solve
    one plan (the full plan without one item) and return a small summary.
    Top-level and light (numpy / cv2 / scipy only), so it can run in a
    separate process."""
    t0 = time.perf_counter()
    try:
        out = run_plan(job["plan"])
        return {"label": job["label"], "kind": job["kind"], "key": job["key"],
                "K": np.asarray(out["K"]).tolist(), "D": np.asarray(out["D"]).ravel().tolist(),
                "rms": float(out["rms"]), "success": bool(out["success"]), "error": None,
                "seconds": time.perf_counter() - t0}
    except Exception as ex:                              # reported per row
        return {"label": job["label"], "kind": job["kind"], "key": job["key"], "error": f"{ex}",
                "seconds": time.perf_counter() - t0}


# Many worker processes, each with a multi-threaded BLAS (MKL / OpenBLAS) and
# OpenCV, oversubscribe the cores so badly that one 0.2 s solve takes over a
# minute. Workers must therefore run single-threaded: the environment is set
# BEFORE they start (read when numpy loads) and limited again on start-up.
WORKER_ENV = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
              "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}


def worker_init() -> None:
    """ProcessPoolExecutor initializer for loo_task workers."""
    cv2.setNumThreads(1)
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:                                    # optional package
        pass


def start_worker_pool(n_workers: int, jobs: list, fn):
    """A ProcessPoolExecutor with single-threaded workers, all `jobs`
    submitted: (pool, futures). The worker environment is applied only while
    the workers are being started, then the parent's is restored."""
    import os
    from concurrent.futures import ProcessPoolExecutor
    saved = {k: os.environ.get(k) for k in WORKER_ENV}
    os.environ.update(WORKER_ENV)
    try:
        pool = ProcessPoolExecutor(max_workers=n_workers, initializer=worker_init)
        futures = [pool.submit(fn, job) for job in jobs]
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return pool, futures
