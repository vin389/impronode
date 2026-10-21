# node_editor/chessboard_detect.py
"""
Chessboard corner detection and calibration-image ranking (no Tk), shared by
the Camera Calibration node and node_editor/chessboard_search.py. Moved here
unchanged from the retired Chessboard Calibration node
(recycle_bin/node_editor/nodes/chessboard_calibration_node.py).

Corner layout: cv2.findChessboardCorners with patternSize=(ncol, nrow)
returns the corners row-major -- (nrow, ncol, 2) after _reshape_corners --
re-indexed to one canonical order by canonicalize_corners().
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np


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
