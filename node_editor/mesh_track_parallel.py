# node_editor/mesh_track_parallel.py
"""
Multi-process helper for the Mesh Template Track node (no Tk).

Why the work is split over CELLS, not frames: frame k is matched starting
from where every point was found in frame k-1 (and, with "update template",
from templates cut out of earlier frames), so frames must run in order. The
points (cells) of ONE frame are independent, so each frame's points are
split into chunks, one per worker process.

Images are passed through shared memory: the tracking thread decodes a
frame once, converts it to grey and copies it into a SharedMemory block;
the workers map that block instead of receiving a pickled copy of a
multi-megabyte image for every chunk of every frame (which made more than
~4 processes SLOWER than fewer).

Grey conversion: cv2.COLOR_BGR2GRAY applied to the RGB frame -- exactly what
calcTemplateMatchPyr did internally when it was given the RGB frame -- so
results are identical to the single-process version.

This module is imported by the worker processes, so it must stay light:
numpy, cv2 and the matcher only (no Tk, no node_editor imports).
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import shared_memory
from pathlib import Path

import cv2
import numpy as np

# calcTemplateMatchPyr lives in a top-level script module at the repo root.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from calcTemplateMatchPyr_codex import calcTemplateMatchPyr  # noqa: E402

MIN_POINTS_PER_PROCESS = 8          # smaller chunks cost more in overhead than they save


def read_gray(path: str) -> np.ndarray | None:
    """Decode an image file and convert it to grey (see module docstring)."""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        if buf.size == 0:
            return None
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)


def read_gray_timed(path: str) -> tuple[np.ndarray | None, float]:
    """read_gray() plus the seconds it took (for the Run panel's timing)."""
    t = time.perf_counter()
    gray = read_gray(path)
    return gray, time.perf_counter() - t


def max_processes() -> int:
    """Upper limit for the worker count: leave one CPU core for the UI."""
    return max(1, min(16, (os.cpu_count() or 2) - 1))


def candidate_counts(n_points: int, limit: int) -> list[int]:
    """Process counts tried by the auto-tuner, smallest first."""
    limit = max(1, min(limit, n_points // MIN_POINTS_PER_PROCESS or 1))
    counts = [c for c in (1, 2, 4, 6, 8, 12, 16) if c <= limit]
    if limit not in counts:
        counts.append(limit)
    return counts


class SharedFrame:
    """A grey uint8 image held in shared memory (owned by the tracking thread)."""

    def __init__(self, gray: np.ndarray):
        self.shape = tuple(gray.shape)
        self.shm = shared_memory.SharedMemory(create=True, size=max(1, gray.nbytes))
        self.array = np.ndarray(self.shape, dtype=np.uint8, buffer=self.shm.buf)
        self.array[...] = gray

    @property
    def ref(self) -> tuple[str, tuple]:
        return self.shm.name, self.shape

    def release(self) -> None:
        self.array = None
        try:
            self.shm.close()
            self.shm.unlink()
        except (FileNotFoundError, OSError):
            pass


# ── worker side ─────────────────────────────────────────────────────────
_attached: dict[str, shared_memory.SharedMemory] = {}


def _worker_init() -> None:
    # One OpenCV thread per worker: the parallelism comes from the processes.
    cv2.setNumThreads(1)


def _attach(ref) -> np.ndarray:
    name, shape = ref
    shm = _attached.get(name)
    if shm is None:
        shm = shared_memory.SharedMemory(name=name)
        _attached[name] = shm
    return np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)


def _forget_stale(alive: set) -> None:
    for name in [n for n in _attached if n not in alive]:
        try:
            _attached.pop(name).close()
        except Exception:
            pass


def _warm_up(_i: int) -> int:
    return os.getpid()


def _match_chunk(task: dict):
    _forget_stale(task["alive"])
    x0, y0, x1, y1 = task["box"]
    # Views into shared memory (no copy); the matcher copies only this band.
    prev_img = _attach(task["prev"])[y0:y1, x0:x1]
    next_img = _attach(task["next"])[y0:y1, x0:x1]
    origin = np.array([x0, y0], dtype=np.float64)
    nxt, st, cf = calcTemplateMatchPyr(prev_img, next_img, task["pts"] - origin,
                                       next_pts_init=task["init"] - origin, ctime=None, **task["cfg"])
    return np.asarray(nxt, dtype=np.float64) + origin, st, cf


def _pair(v) -> tuple[int, int]:
    return (int(v), int(v)) if np.isscalar(v) else (int(v[0]), int(v[1]))


def crop_box(pts: np.ndarray, init: np.ndarray, cfg: dict, shape: tuple) -> tuple[int, int, int, int]:
    """Image region a chunk of points can touch, so a worker processes only
    that band instead of the whole image.

    Reach from a point: half the template + the coarse search range + the
    fine search radius at every finer level + a margin for the pyramid's
    Gaussian filter (so pixels the matcher reads are not affected by the
    band's edge) and rounding. The origin is aligned to 2**(levels-1) so
    every pyramid level of the band lines up exactly with the same level of
    the full image -- results equal the whole-image computation. Where the
    band reaches the real image border it is clipped there, exactly as the
    whole image would be."""
    levels = int(cfg["num_levels"])
    tw, th = _pair(cfg["template_size"])
    sx, sy = _pair(cfg["search_range"])
    fx, fy = _pair(cfg["fine_search_radius"])
    top = 2 ** (levels - 1)
    finer = sum(2 ** lv for lv in range(levels - 1))      # fine search at levels L-2 .. 0
    margin = 8 * 2 ** levels + 2 * top                      # pyramid filter support + rounding
    rx = tw // 2 + sx + fx * finer + top + margin
    ry = th // 2 + sy + fy * finer + top + margin
    both = np.vstack([pts, init])
    H, W = shape
    x0 = max(0, int(np.floor(both[:, 0].min())) - rx)
    y0 = max(0, int(np.floor(both[:, 1].min())) - ry)
    x0, y0 = (x0 // top) * top, (y0 // top) * top
    x1 = min(W, int(np.ceil(both[:, 0].max())) + rx + 1)
    y1 = min(H, int(np.ceil(both[:, 1].max())) + ry + 1)
    return x0, y0, x1, y1


# ── tracking-thread side ────────────────────────────────────────────────
class ParallelMatcher:
    """Runs calcTemplateMatchPyr for one group of points, split over `n`
    worker processes (n = 1 runs in the calling thread, no IPC)."""

    def __init__(self, n_workers: int):
        self.n_workers = n_workers
        self.pool = (ProcessPoolExecutor(max_workers=n_workers, initializer=_worker_init)
                     if n_workers > 1 else None)

    def warm_up(self) -> None:
        """Start every worker process now (spawn + imports take ~1 s on
        Windows) so it is not counted in the auto-tuning timings."""
        if self.pool is not None:
            list(self.pool.map(_warm_up, range(self.n_workers * 2)))

    def match(self, prev: SharedFrame, nxt: SharedFrame, pts: np.ndarray, init: np.ndarray,
              cfg: dict, n: int, alive: set):
        n = max(1, min(n, self.n_workers, len(pts)))
        if n == 1 or self.pool is None:
            return calcTemplateMatchPyr(prev.array, nxt.array, pts, next_pts_init=init, **cfg)
        # Points are in row-major mesh order, so consecutive chunks are
        # horizontal bands of the mesh -- each worker needs only its band.
        tasks = [{"prev": prev.ref, "next": nxt.ref, "pts": p, "init": q, "cfg": cfg, "alive": alive,
                  "box": crop_box(p, q, cfg, nxt.shape)}
                 for p, q in zip(np.array_split(pts, n), np.array_split(init, n))]
        parts = list(self.pool.map(_match_chunk, tasks))
        return (np.concatenate([np.asarray(r[0], dtype=np.float64).reshape(-1, 2) for r in parts]),
                np.concatenate([np.asarray(r[1]).reshape(-1) for r in parts]),
                np.concatenate([np.asarray(r[2], dtype=np.float64).reshape(-1) for r in parts]))

    def shutdown(self) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None
