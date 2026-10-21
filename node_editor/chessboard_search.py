"""
Brute-force chessboard detection search (no Tk; used by the Camera
Calibration node).

Every image is searched with every combination of the detection settings
(pattern size, detection width, adaptive threshold / normalize / fast
check). One image can give several detections -- for example a 7 x 12
sub-board where the whole board is partly outside the image, and an 8 x 13
one with another setting. They are MERGED into one set of corners on one
board lattice:

  Every detection is a grid of corners with lattice indices (r, c). Two
  detections of the same board in the same image differ only by one of the
  8 lattice symmetries (rotations by 90 degrees, mirror) plus an integer
  offset. A new detection is merged into the existing lattice only if
    * at least 3 of its corners coincide (within a quarter of the smallest
      square) with lattice corners, and those shared corners are NOT on
      one straight lattice line (otherwise the mapping is ambiguous);
    * exactly ONE symmetry + offset maps ALL shared corners onto their
      lattice indices;
    * none of its other corners lands on a lattice index that is already
      taken by a corner somewhere else in the image.
  Anything else is rejected (never guessed).

  Before that, a detection whose grid is not smooth is rejected as a false
  detection (see lattice_irregularity(): with many settings tried, OpenCV
  occasionally returns a grid with some corners on an occluding edge). A detection with no shared
  corner starts a separate lattice; only the largest lattice of an image is
  kept (two lattices without shared corners have an unknown offset).

A corner found by several settings is then refined ONCE more with
cv2.cornerSubPix on the full-resolution image, starting from the median of
its detections ('most reliable' position); corners that the refinement
moves away (> the matching tolerance) are dropped.

Object points: lattice index (r, c) -> (c * square, r * square, 0), the same
convention as chessboard_detect.build_object_grid(). The lattice of an image is shifted so
its smallest r and c are 0 -- the origin may be a corner that was not
found. Each image has its own lattice (own origin), which is correct for
calibration: every board view has its own pose.
"""

from __future__ import annotations

import dataclasses
import math
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from node_editor.chessboard_detect import (
    _read_gray, _reshape_corners, auto_subpix_window, canonicalize_corners)

SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
MATCH_FRACTION = 0.25          # matching tolerance = this x the smallest square (px)
MIN_SHARED = 3                 # shared corners needed to merge a detection
MAX_IRREGULARITY = 0.3         # see lattice_irregularity()


def parse_number_list(text: str) -> np.ndarray | None:
    """Numbers separated by spaces / commas, with MATLAB-style ranges:
    '7 8 9', '7:9', '800:400:1600' -> float array, or None if not valid.
    (Moved here from the retired calib_nodes.py.)"""
    if text is None:
        return None
    raw = str(text).strip()
    if not raw or raw.lower() in {"none", "null"}:
        return None

    # Treat the ghost-text placeholder as empty so it cannot be mistaken for real input.
    if raw.startswith("e.g.,"):
        return None

    tokens = [p for p in re.split(r"[\s,]+", raw) if p.strip()]
    if not tokens:
        return None

    values: list[float] = []
    for token in tokens:
        if ":" not in token:
            try:
                values.append(float(token))
            except ValueError:
                return None
            continue

        colon_parts = [part.strip() for part in token.split(":") if part.strip()]
        if not colon_parts:
            continue
        if len(colon_parts) == 1:
            try:
                values.append(float(colon_parts[0]))
            except ValueError:
                return None
            continue

        try:
            start = float(colon_parts[0])
            stop = float(colon_parts[-1])
            if len(colon_parts) == 2:
                step = 1.0 if stop >= start else -1.0
            else:
                step = float(colon_parts[1])
                if step == 0:
                    return None
        except ValueError:
            return None

        if step > 0:
            arr = np.arange(start, stop + step, step, dtype=np.float64)
        else:
            arr = np.arange(start, stop + step, step, dtype=np.float64)
        values.extend(float(v) for v in arr)

    if not values:
        return None
    return np.asarray(values, dtype=np.float64)


@dataclasses.dataclass(frozen=True)
class Setting:
    ncol: int
    nrow: int
    max_width: int             # 0 = full size
    adaptive: bool
    normalize: bool
    fast: bool

    @property
    def flags(self) -> int:
        f = 0
        if self.adaptive:
            f |= cv2.CALIB_CB_ADAPTIVE_THRESH
        if self.normalize:
            f |= cv2.CALIB_CB_NORMALIZE_IMAGE
        if self.fast:
            f |= cv2.CALIB_CB_FAST_CHECK
        return f

    def label(self) -> str:
        opts = [n for n, on in (("adaptive", self.adaptive), ("normalize", self.normalize), ("fast", self.fast)) if on]
        return (f"{self.ncol} x {self.nrow}, width {self.max_width or 'full'}"
                + (", " + " + ".join(opts) if opts else ""))


def build_settings(ncols, nrows, widths, adaptive, normalize, fast) -> list:
    """Every combination; `adaptive` / `normalize` / `fast` are lists of the
    bools to try ([True], [False] or [False, True])."""
    return [Setting(int(c), int(r), int(w), bool(a), bool(n), bool(f))
            for c in ncols for r in nrows for w in widths
            for a in adaptive for n in normalize for f in fast]


# ── one detection ──────────────────────────────────────────────────────────

def _effective_width(max_width: int, w: int) -> int:
    return 0 if max_width <= 0 or max_width >= w else int(max_width)


def find_corners(gray: np.ndarray, setting: Setting, scaled: dict | None = None):
    """(nrow, ncol, 2) sub-pixel corners (canonical order, full-resolution
    pixels) or None. `scaled` caches the shrunk images per width."""
    h, w = gray.shape[:2]
    eff = _effective_width(setting.max_width, w)
    if eff:
        key = eff
        if scaled is not None and key in scaled:
            img, scale = scaled[key]
        else:
            scale = eff / float(w)
            img = cv2.resize(gray, (eff, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
            if scaled is not None:
                scaled[key] = (img, scale)
    else:
        img, scale = gray, 1.0
    try:
        found, corners = cv2.findChessboardCorners(img, (setting.ncol, setting.nrow), flags=setting.flags)
    except cv2.error:
        return None
    if not found or corners is None or len(corners) != setting.ncol * setting.nrow:
        return None
    corners = corners.reshape(-1, 1, 2).astype(np.float32)
    if scale != 1.0:
        corners = corners / scale
    win = auto_subpix_window(_reshape_corners(corners.reshape(-1, 2), setting.ncol, setting.nrow))
    try:
        corners = cv2.cornerSubPix(gray, corners, (win, win), (-1, -1), SUBPIX_CRITERIA)
    except cv2.error:
        return None
    grid = canonicalize_corners(_reshape_corners(corners.reshape(-1, 2), setting.ncol, setting.nrow))
    if lattice_irregularity(grid) > MAX_IRREGULARITY:
        return None
    return grid.astype(np.float64)


def _square_steps(grid: np.ndarray) -> np.ndarray:
    return np.concatenate([np.linalg.norm(np.diff(grid, axis=1), axis=2).ravel(),
                           np.linalg.norm(np.diff(grid, axis=0), axis=2).ravel()])


def lattice_irregularity(grid: np.ndarray) -> float:
    """Largest second difference along the grid rows / columns,
    |p[k+1] - 2 p[k] + p[k-1]|, divided by the local square size. A real
    board under perspective and strong lens distortion stays near 0.05; a
    false corner (e.g. on the edge of an occluding object) gives ~0.5-1."""
    worst = 0.0
    for g in (grid, grid.transpose(1, 0, 2)):
        if g.shape[1] < 3:
            continue
        d2 = np.linalg.norm(g[:, 2:] - 2 * g[:, 1:-1] + g[:, :-2], axis=2)
        local = 0.5 * (np.linalg.norm(g[:, 2:] - g[:, 1:-1], axis=2) + np.linalg.norm(g[:, 1:-1] - g[:, :-2], axis=2))
        worst = max(worst, float(np.max(d2 / np.maximum(local, 1e-9))))
    return worst


def grid_rc(nrow: int, ncol: int) -> np.ndarray:
    """(nrow * ncol, 2) lattice indices (r, c) in row-major order -- the order
    of corners.reshape(-1, 2) for an (nrow, ncol, 2) grid."""
    r, c = np.divmod(np.arange(nrow * ncol), ncol)
    return np.stack([r, c], axis=1).astype(np.int32)


# ── merging detections of one image ───────────────────────────────────────

def _lattice_ops() -> list:
    rot = np.array([[0, -1], [1, 0]], dtype=np.int64)
    flip = np.array([[0, 1], [1, 0]], dtype=np.int64)
    ops, g = [], np.eye(2, dtype=np.int64)
    for _ in range(4):
        ops += [g.copy(), g @ flip]
        g = rot @ g
    return ops


LATTICE_OPS = _lattice_ops()


class _Lattice:
    def __init__(self):
        self.obs: dict = {}            # (r, c) -> list of (u, v, setting index)
        self.wins: list = []           # cornerSubPix half-windows of the merged detections
        self.tol = math.inf
        self.settings: list = []       # setting indices merged
        self._keys = np.zeros((0, 2), dtype=np.int64)
        self._uv = np.zeros((0, 2))

    def _rebuild(self) -> None:
        self._keys = np.array(list(self.obs.keys()), dtype=np.int64).reshape(-1, 2)
        self._uv = np.array([np.mean([(u, v) for u, v, _k in o], axis=0) for o in self.obs.values()]).reshape(-1, 2)

    def add(self, rc: np.ndarray, uv: np.ndarray, setting: int, win: int, tol: float) -> None:
        for (r, c), (u, v) in zip(rc.tolist(), uv.tolist()):
            self.obs.setdefault((int(r), int(c)), []).append((u, v, setting))
        self.wins.append(win)
        self.tol = min(self.tol, tol)
        self.settings.append(setting)
        self._rebuild()

    def register(self, rc: np.ndarray, uv: np.ndarray, tol: float):
        """('disjoint' | 'rejected' | 'ok', mapped rc or None)."""
        tol = min(tol, self.tol)
        d = np.linalg.norm(uv[:, None, :] - self._uv[None, :, :], axis=2)          # (n, m)
        j = np.argmin(d, axis=1)
        near = d[np.arange(len(uv)), j] < tol
        if not near.any():
            return "disjoint", None
        i_sh, j_sh = np.nonzero(near)[0], j[near]
        if len(set(j_sh.tolist())) != len(j_sh) or len(i_sh) < MIN_SHARED:
            return "rejected", None
        b = rc[i_sh].astype(np.float64)
        if np.linalg.matrix_rank(b - b.mean(axis=0), tol=1e-6) < 2:            # all on one lattice line
            return "rejected", None
        a = self._keys[j_sh]
        fits = []
        for g in LATTICE_OPS:
            t = a - rc[i_sh].astype(np.int64) @ g.T
            if (t == t[0]).all():
                fits.append((g, t[0]))
        if len(fits) != 1:
            return "rejected", None
        g, t = fits[0]
        mapped = rc.astype(np.int64) @ g.T + t
        taken = {tuple(k) for k in self._keys.tolist()}
        for k in np.nonzero(~near)[0]:
            if tuple(mapped[k].tolist()) in taken:      # lands on a lattice corner that is elsewhere in the image
                return "rejected", None
        return "ok", mapped

    def size(self) -> int:
        return len(self.obs)


class ImageMerger:
    """Detections of ONE image -> lattices (see the module docstring)."""

    def __init__(self):
        self.lattices: list = []
        self.rejected = 0

    def add(self, grid: np.ndarray, setting: int) -> str:
        nrow, ncol = grid.shape[:2]
        rc = grid_rc(nrow, ncol)
        uv = grid.reshape(-1, 2)
        steps = _square_steps(grid)
        tol = MATCH_FRACTION * float(np.percentile(steps, 5)) if steps.size else 1.0
        win = auto_subpix_window(grid)
        for lat in sorted(self.lattices, key=lambda x: -x.size()):
            status, mapped = lat.register(rc, uv, tol)
            if status == "ok":
                lat.add(mapped, uv, setting, win, tol)
                return "merged"
            if status == "rejected":
                self.rejected += 1
                return "rejected"
        lat = _Lattice()
        lat.add(rc, uv, setting, win, tol)
        self.lattices.append(lat)
        return "new"

    def best(self):
        return max(self.lattices, key=lambda x: (x.size(), sum(len(o) for o in x.obs.values())), default=None)

    def best_count(self) -> int:
        b = self.best()
        return 0 if b is None else b.size()


def _merge_order(detections: list) -> list:
    return sorted(detections, key=lambda d: (-d[1].shape[0] * d[1].shape[1], d[0]))


def merge_detections(gray: np.ndarray, detections: list) -> dict | None:
    """detections: [(setting index, (nrow, ncol, 2) grid)]. Largest
    detections are merged first (deterministic, the most trusted lattice
    first). Returns rc / corners / support / spread arrays or None."""
    merger = ImageMerger()
    for k, grid in _merge_order(detections):
        merger.add(grid, k)
    lat = merger.best()
    if lat is None:
        return None
    keys = list(lat.obs.keys())
    obs = [lat.obs[k] for k in keys]
    start = np.array([np.median([(u, v) for u, v, _s in o], axis=0) for o in obs], dtype=np.float32)
    win = int(min(lat.wins))
    refined = cv2.cornerSubPix(gray, start.reshape(-1, 1, 2).copy(), (win, win), (-1, -1),
                               SUBPIX_CRITERIA).reshape(-1, 2).astype(np.float64)
    keep = np.linalg.norm(refined - start, axis=1) < lat.tol
    # Two lattice corners refined onto the same image spot: contradictory, drop both.
    d = np.linalg.norm(refined[:, None] - refined[None], axis=2) + np.eye(len(refined)) * 1e9
    keep &= ~(d < 0.5 * lat.tol).any(axis=1)
    rc = np.array(keys, dtype=np.int64)[keep]
    uv = refined[keep]
    if len(rc) < 4 or np.linalg.matrix_rank((rc - rc.mean(axis=0)).astype(float), tol=1e-6) < 2:
        return None
    support = np.array([len({s for _u, _v, s in o}) for o, k in zip(obs, keep) if k], dtype=np.int32)
    spread = np.array([max(math.hypot(u - p[0], v - p[1]) for u, v, _s in o)
                       for o, p, k in zip(obs, refined, keep) if k])
    order = np.lexsort((rc[:, 1], rc[:, 0]))               # row-major
    rc = rc[order] - rc.min(axis=0)
    return {"rc": rc.astype(np.int32), "corners": uv[order].astype(np.float32),
            "support": support[order], "spread": spread[order].astype(np.float32),
            "n_lattices": len(merger.lattices), "rejected": merger.rejected,
            "settings_merged": sorted(set(lat.settings))}


def image_stats(gray: np.ndarray, corners: np.ndarray) -> dict:
    """sharpness / coverage / center_dist as in detect_chessboard()."""
    h, w = gray.shape[:2]
    xs, ys = corners[:, 0], corners[:, 1]
    x0, x1 = max(0, int(math.floor(xs.min())) - 15), min(w, int(math.ceil(xs.max())) + 15)
    y0, y1 = max(0, int(math.floor(ys.min())) - 15), min(h, int(math.ceil(ys.max())) + 15)
    roi = gray[y0:y1, x0:x1]
    hull = cv2.convexHull(corners.reshape(-1, 1, 2).astype(np.float32))
    centroid = corners.mean(axis=0)
    return {"sharpness": float(cv2.Laplacian(roi, cv2.CV_64F).var()) if roi.size else 0.0,
            "coverage": float(cv2.contourArea(hull)) / float(w * h),
            "center_dist": float(np.linalg.norm(centroid - np.array([w / 2.0, h / 2.0]))) / (0.5 * math.hypot(w, h)),
            "size": (w, h)}


# ── background search ──────────────────────────────────────────────────────

def new_search_state(n_images: int, n_settings: int) -> dict:
    return {"cancel": threading.Event(), "lock": threading.Lock(), "done": 0,
            "total": n_images * n_settings, "images_done": 0, "live": {}, "results": [None] * n_images,
            "setting_hits": [[0, 0] for _ in range(n_settings)],      # [images, corners] per setting
            "error": None, "finished": False}


def search_worker(paths: list, settings: list, threads: int, state: dict) -> None:
    """Background thread: every (image, setting) pair is one task, image by
    image, run by a thread pool. Fills state['results'][i] per image with
    a dict (or {'path', 'error'} / {'path', 'found': False}). Never touches Tk.
    After a cancel, images that were started are still merged with what
    they found ('partial')."""
    n_set = len(settings)
    lock = state["lock"]
    ctx: dict = {}                                     # image index -> {"gray", "scaled", "dets", "left", "merger"}

    def context(i: int):
        with lock:
            c = ctx.get(i)
            if c is None:
                c = ctx[i] = {"gray": None, "scaled": {}, "dets": [], "left": n_set, "merger": ImageMerger(),
                              "seen": {}, "lock": threading.Lock(), "loaded": False}
        with c["lock"]:
            if not c["loaded"]:
                c["gray"] = _read_gray(paths[i])
                c["loaded"] = True
        return c

    def finish(i: int, partial: bool) -> None:
        c = ctx.get(i)
        if c is None:
            return
        gray = c["gray"]
        if gray is None:
            res = {"path": paths[i], "found": False, "error": "cannot read image"}
        else:
            merged = merge_detections(gray, c["dets"]) if c["dets"] else None
            if merged is None:
                res = {"path": paths[i], "found": False, "size": (gray.shape[1], gray.shape[0])}
            else:
                res = {"path": paths[i], "found": True, **merged, **image_stats(gray, merged["corners"])}
            res["n_det"] = len(c["dets"])
            res["best_single"] = max((g.shape[0] * g.shape[1] for _k, g in c["dets"]), default=0)
        res["source_index"] = i
        res["partial"] = partial
        with lock:
            state["results"][i] = res
            state["live"][i] = len(res["rc"]) if res.get("found") else 0
            state["images_done"] += 1
        c["gray"] = c["scaled"] = None
        del ctx[i]

    def task(i: int, k: int) -> None:
        try:
            _task(i, k)
        except Exception as e:                          # pool futures are not awaited: record, then stop
            state["error"] = state["error"] or f"{type(e).__name__}: {e}"
            state["cancel"].set()

    def _task(i: int, k: int) -> None:
        if state["cancel"].is_set():
            return
        c = context(i)
        grid = None
        if c["gray"] is not None:
            s = settings[k]
            key = (s.ncol, s.nrow, _effective_width(s.max_width, c["gray"].shape[1]), s.flags)
            with c["lock"]:
                cached = c["seen"].get(key, "todo")
            if cached == "todo":
                with c["lock"]:
                    scaled = c["scaled"]
                grid = find_corners(c["gray"], s, scaled)
                with c["lock"]:
                    c["seen"][key] = grid
            else:
                grid = cached
        with c["lock"]:
            if grid is not None:
                c["dets"].append((k, grid))
                c["merger"] = ImageMerger()
                for kk, gg in _merge_order(c["dets"]):
                    c["merger"].add(gg, kk)
            c["left"] -= 1
            last = c["left"] == 0
            live = c["merger"].best_count()
        with lock:
            state["done"] += 1
            state["live"][i] = live
            if grid is not None:
                state["setting_hits"][k][0] += 1
                state["setting_hits"][k][1] += grid.shape[0] * grid.shape[1]
        if last:
            finish(i, partial=False)

    try:
        with ThreadPoolExecutor(max_workers=max(1, int(threads))) as ex:
            for i in range(len(paths)):
                for k in range(n_set):
                    ex.submit(task, i, k)
        for i in list(ctx.keys()):                     # cancelled: merge what the started images found
            finish(i, partial=True)
    except Exception as e:
        state["error"] = f"{type(e).__name__}: {e}"
    finally:
        state["finished"] = True
