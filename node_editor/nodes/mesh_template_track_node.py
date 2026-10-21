# node_editor/nodes/mesh_template_track_node.py
"""
Mesh template tracking node.

Tracks a regular M x N mesh of points through an image sequence with the
coarse-to-fine template matcher calcTemplateMatchPyr (the same function used
by the Template Match node) and lets the user inspect how the mesh moves.

Tracking strategy
    Two reference styles are available, selected by the "Update template
    every frame" checkbox:

    The mesh is defined on the REFERENCE frame (default: frame 1). Tracking
    runs outward from it: forward to the end frame and, when the reference
    lies inside the start..end range, backward to the start frame; each of
    the two passes starts again from the reference frame and the mesh. With
    a reference outside the range, tracking starts at the end of the range
    nearest to it.

    * Fixed reference (checkbox off, the default). Every template is cut
      from the REFERENCE frame. For frame k the search is centred on the
      point's last successfully tracked position (next_pts_init), so
      "Search range" only has to cover the motion BETWEEN two consecutive
      frames even though the template itself never changes. Errors do not
      accumulate from frame to frame, but a point whose appearance has
      drifted far from the reference frame (large rotation, out-of-plane
      deformation, lighting change) may fail to match at all.

    * Incremental reference (checkbox on). Each point's template is re-cut,
      every time that point is successfully tracked, from the frame it was
      just found in and the position it was just found at -- so frame k+1
      is matched against frame k's own appearance of that point, not the
      original one. This is far more robust to appearance changes, at the
      cost of small per-step sub-pixel interpolation differences that CAN
      accumulate into drift over a long sequence. A point that fails keeps
      its old template and last known position (exactly like the fixed
      style) until it succeeds again.

      For a rigid, appearance-preserving motion (pure translation, nothing
      else changing), the patch compared against frame k+1 is pixel-for-
      pixel identical whichever style is used -- mesh points are already
      generally at sub-pixel positions in BOTH styles, so both already
      require the same sub-pixel patch extraction capability -- so both
      styles should recover essentially the same position at every frame.
      They are only expected to diverge once the target's appearance
      actually changes between frames.

    A point that fails at frame k is NOT abandoned: it keeps whatever
    template/position it last had and is retried against every following
    frame, so it can reacquire tracking later.

    The reference frame's own result (when it is inside the range) is the
    mesh, status 1, confidence 1.

Coordinates
    Image coordinates follow OpenCV: (0, 0) is the CENTRE of the top-left
    pixel, u is horizontal, v is vertical. Mesh points are the centres of the
    M x N cells spanned by the top-left / bottom-right corners.

Result array
    points: (num_steps, M, N, 2) with the last axis (u, v) in pixels.
    Frame numbers are 1-based, inclusive.

Threading
    Tracking and preview decoding run on worker threads. Worker threads never
    touch Tk or matplotlib; the Tk thread polls shared state with after().
    In incremental-reference mode, more than one past frame may be held in
    memory at once when some points are lagging behind (still referencing an
    older frame because they have not re-succeeded yet); any cached frame no
    longer referenced by any point is dropped immediately after each step.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import threading
import time
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from PIL import Image, ImageTk

from node_editor import frame_copy
from node_editor.base_node import BaseNode
from node_editor.execution import ExecutionMode
from node_editor.pin_types import PinDef, PinSchema, PinType
from node_editor.project_context import get_project_directory
from node_editor.tooltip import add_tooltip
from node_editor.video_frame_reader import VIDEO_EXTS, VideoFrameReader
from node_editor.nodes.template_match_node import TEMPLATE_MATCH_TIPS

# calcTemplateMatchPyr lives as a top-level script module (not inside the
# node_editor package); make the repo root importable exactly like
# template_match_node.py does. A failed import must not hide the node: viewing
# and loading results still works, only "Run" is blocked.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
try:
    from calcTemplateMatchPyr_codex import calcTemplateMatchPyr
    _TRACKER_IMPORT_ERROR = None
except Exception as _exc:
    calcTemplateMatchPyr = None
    _TRACKER_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"


MATCH_METHODS = {
    "TM_SQDIFF": cv2.TM_SQDIFF,
    "TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED,
    "TM_CCORR": cv2.TM_CCORR,
    "TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
    "TM_CCOEFF": cv2.TM_CCOEFF,
    "TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
}

PREVIEW_MAX_SIDE = 2000          # longest side of the cached preview image (px)
MAX_DRAWN_POINTS = 20000         # above this, dots are not drawn in the preview
MAX_RESULT_ENTRIES = 20_000_000  # frames * M * N guard (~25 bytes per entry)
POLL_MS = 250
_LOADED = object()                # marker: template size just loaded, adopt the mesh as is

GRID_COLOR = "#22ccee"
SELECT_COLOR = "#ff3d00"
DOT_COLOR = "#ffe000"
SELECT_DOT_COLOR = "#ff2d2d"
OUTSIDE_DOT_COLOR = "#ff8c00"
PICK_COLOR = "#7CFC00"         # crosshair while picking a mesh corner
INVALID_FG = "#c62828"
NORMAL_FG = "#111111"

# Image-preview pan/zoom tuning.
IMG_MIN_SCALE = 0.02       # minimum canvas-px-per-image-px (zoomed all the way out)
IMG_MAX_SCALE = 40.0       # maximum canvas-px-per-image-px (zoomed all the way in)
IMG_ZOOM_FACTOR = 1.15     # scale multiplier applied per wheel notch
IMG_CLICK_DRAG_PX = 3      # movement (px) before a press+release counts as a pan, not a click

_SEPARATORS = re.compile(r"[,;\s]+")

_HELP_TEXT = """\
Mesh Template Track
===================
Tracks a regular mesh of points through an image sequence and shows how the
mesh moves. Useful to understand object/camera motion and overall tracking
quality before doing detailed analysis.

INPUT PIN
  files   STRING   list of image paths (the 'files' output of the Image
                   Sequence node). Frame numbers are 1-based positions in it.

INPUT WITHOUT THE PIN
  When the 'files' input delivers no images, the Input group can select
  the frames directly:
    Select a video ...   one video file. Nothing is read in advance: the
                         frame count comes from the file header (corrected
                         when the real end is reached), and frames are read
                         in order while tracking -- always exact. With a
                         backward pass (reference frame after Start), that
                         stretch is first read once in order so the
                         reverse reads can jump with verification.
    Select images ...    several image files, used in file-name order.
    Clear                forget the selection.
  The selection is saved in the project with paths relative to the project
  file (a very long image list goes to a side file next to the project).
  A connected input with images always takes precedence.

OUTPUT PINS (after Run, or after a result was loaded)
  points       (S, M, N, 2)  tracked (u, v) in pixels, NaN where tracking failed
  status       (S, M, N)     1 = tracked (correlation >= min. correlation)
  confidence   (S, M, N)     match quality score reported by the tracker
  frames       (S,)          frame numbers (1-based)

ALGORITHM
  Two reference styles, chosen with 'Update template every frame':
    off (default) - every template is cut from the REFERENCE frame ('fixed'
                    reference), so errors do not accumulate from frame to
                    frame. The search in frame k is still centred on the
                    point's last successfully tracked position, so 'Search
                    range' only has to cover the motion between two
                    consecutive frames.
    on             - each point's template is re-cut, every time it is
                    successfully tracked, from the frame and position it was
                    just found at ('incremental' reference), so tracking
                    follows the target's own most recent appearance rather
                    than the original one. For pure rigid motion with no
                    appearance change the two styles should give nearly the
                    same result; they are mainly useful when the target's
                    appearance itself changes over time (large rotation,
                    out-of-plane deformation, lighting).
  A point that fails is stored as NaN but is not abandoned: it keeps its old
  template/position and is retried on every later frame.
  Tracking starts at the REFERENCE frame and runs outward: forward to the
  End frame and backward to the Start frame (each pass starts again from
  the reference). With a reference before Start / after End, tracking
  starts at the end of the range nearest to it.

MESH SETTINGS
  Cells        M N   cells along the vertical / horizontal image edge
  Top-left     u v   corner of the mesh (pixels)
  Bottom-right u v   corner of the mesh (pixels; filled from the image size)
  One tracking point sits at the centre of every cell. Numbers may be
  separated by space, tab or comma. Coordinates are OpenCV pixel coordinates:
  (0, 0) is the centre of the top-left pixel.
  "Pick TL" / "Pick BR" pick a corner with the mouse in the image preview:
  a crosshair follows the cursor and the corner field (and the mesh) updates
  live with the integer pixel under it. Click to accept; Esc or right-click
  cancels and restores the previous value. Dragging and wheel zoom still
  work while picking.
  "Fit Mesh to Full Image" sets the corners to (0, 0) and
  (image width - 1, image height - 1), read from the currently loaded
  preview image.

TRACKING SETTINGS
  Reference frame (1-based, default 1): the frame the mesh is defined on --
  the corners / cells describe the points in THIS frame, and templates are
  cut from it. Editing it shows that frame in the preview (the slider
  covers Start..End and the reference frame). Projects saved before this
  setting existed use their Start frame. 'Update template every frame
  (else: fixed reference frame)' chooses the reference style (ALGORITHM).
  The other settings have the same meaning as in the Template Match node.
  'Auto (from mesh)' sets the
  template size to the mesh cell size (rounded to the nearest integer,
  minimum 3); the fields can still be edited afterwards. The same happens
  automatically whenever the mesh (cells or corners) changes to valid
  values -- re-type the template size after the mesh is final if you want
  a different one. A template size loaded with a project / result file is
  kept until the mesh is changed. Start / End frame
  are 1-based and inclusive (End defaults to the number of frames).
  Invalid values are shown in red; Run stays disabled until all are valid.

RUN / STOP
  Run tracks in a background thread with a progress bar. Stop takes effect
  after the frame currently being processed; the frames tracked so far are
  kept (the rest is NaN).

  Speed: frames are tracked in order (each frame starts from the previous
  frame's result), but the cells of one frame are split over several worker
  PROCESSES, each matching one horizontal band of the mesh. The number of
  processes is chosen automatically: the first frames try 1, 2, 4, ... and
  the fastest is used for the rest of the run (small meshes stay at 1).
  The next frame is decoded while the current one is matched. Results are
  the same as with a single process. The read-only "Tracking details" box
  shows the processes in use, the auto-tune timings and the time per frame.

LOAD / SAVE / SAVE AS
  Save (Ctrl+S in this window) writes to the result file shown in the
  "File" field, replacing it; the first time (no file yet) it asks for a
  name like Save as. Save as always asks.
  A .npz file holds the settings used for the result and the result itself.
  The project file stores the settings plus the (relative) path of the last
  saved/loaded result file and re-loads it when the project is opened. The
  result is NOT embedded in the project file: save it to a file, otherwise a
  long tracking run has to be repeated after re-opening the project.
  The result-file path is a property of the node: the "File" field in the
  Data group shows it (and how the project stores it, relative to the
  project file). It is kept after a new Run (the note then says the current
  result is not saved in it yet) and when the file is missing at project
  load. Load / Save as open in that file's folder.

PREVIEW AND PLOT
  Image preview: drag to pan, wheel to zoom (like a normal image viewer); a
  plain click (no drag) selects the cell under the cursor (bold rectangle).
  "Reset view" returns to fit-to-window. The slider changes the displayed
  frame (the image loads when the slider is released). Dots show the
  tracked positions at the displayed frame (red = selected). Orange dots:
  the template of that point would extend outside the image.
  Selected point panel: shows the selected cell's row/column and the pixel
  coordinates of its centre, computed from the current mesh settings --
  available even before Run. Plot: u and v of the selected point relative
  to its position in the reference frame (the mesh point). Drag to pan,
  wheel to zoom; Ctrl+wheel zooms
  horizontally only, Shift+wheel vertically only. If the mesh settings
  differ from the result's mesh, the plot is disabled to avoid showing a
  different point.

COPY SELECTED FRAMES TO A FOLDER (below the plot)
  Keep only the frames you need from a long sequence / video as new files,
  e.g. to make a smaller Image Sequence of them.
  Folder      'Select folder...': where the new files go.
  Frames      1-based, as the Image Sequence selector: '1:100 201:500
              600:900', '1:1000:10' (every 10th), '5', '700:' (to the last);
              numbers may be expressions with + - * ( ): '(31-30):(80+50-30)',
              ':' binds weakest (1:2+3 = 1:5). Valid frames are shaded green
              on the plot above.
  Names       original: image files keep their names; video frames become
              <video name>_FRM000123 (the 1-based frame number).
              new: a pattern like selected_%06d, numbered by the original
              frame number or 1, 2, 3 ... in the order of the selection.
  Format      jpg / png (decoded and written), or 'same' for image files:
              copied byte for byte (exact and fastest). JPEG quality 1-100.
  'Copy selected frames' runs in the background (progress, time left,
  Cancel); existing files: overwrite or skip (asked first).
  'Create Image Sequence node' then adds an Image Sequence node with the
  copied files next to this node.
"""


# ══ Helpers (pure functions, no Tk) ═══════════════════════════════════════

def parse_numbers(text, count, integer=False):
    """Parse exactly `count` numbers separated by space, tab, comma or ';'.
    Returns a tuple, or None if the text is invalid (wrong count, not a
    number, or not finite)."""
    tokens = [t for t in _SEPARATORS.split((text or "").strip()) if t]
    if len(tokens) != count:
        return None
    values = []
    for token in tokens:
        try:
            value = int(token) if integer else float(token)
        except ValueError:
            return None
        if not integer and not math.isfinite(value):
            return None
        values.append(value)
    return tuple(values)


def mesh_centers(M, N, u0, v0, u1, v1) -> np.ndarray:
    """(M, N, 2) array of cell-centre points, last axis = (u, v)."""
    us = u0 + (np.arange(N) + 0.5) * (u1 - u0) / N
    vs = v0 + (np.arange(M) + 0.5) * (v1 - v0) / M
    uu, vv = np.meshgrid(us, vs)
    return np.stack([uu, vv], axis=-1)


def read_rgb(path):
    """Decode an image file to an RGB uint8 array (same convention as the
    Image Sequence node). np.fromfile + imdecode also handles non-ASCII
    paths on Windows. Returns None if the file cannot be read."""
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


def gray_of(rgb):
    """Grey image for the tracker: cv2.COLOR_BGR2GRAY applied to the RGB
    frame, exactly as mesh_track_parallel.read_gray does for image files, so
    video and image sources give identical results."""
    return None if rgb is None else cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)


class FrameSource:
    """The frames the node tracks (0-based index): a list of image files, or
    one video file (exact frame access through VideoFrameReader).

    read_rgb / read_gray may be called from worker threads.

    Video: nothing has to be read before tracking. The frame count comes
    from the file header at once (corrected automatically when in-order
    reading reaches the real end). Frames read in order are always exact;
    a JUMP (seek) is used only to a frame whose fingerprint is already
    known, and is checked against it (see video_frame_reader.py). A
    background pass learns fingerprints for the preview slider; it is
    stopped when tracking starts. Reading BACKWARD (the backward tracking
    pass) decodes a short block forward per jump and serves it from a
    small cache; prepare() first reads that stretch once in order so those
    jumps can be verified."""

    BLOCK = 32                             # frames decoded at once for backward reading

    def __init__(self, kind: str, paths: list[str]):
        self.kind = kind                   # "images" or "video"
        self.paths = list(paths)
        self.reader = None
        self._lock = threading.Lock()
        self._gray_cache: dict = {}
        self._last = None
        if kind == "video":
            self.reader = VideoFrameReader(self.paths[0])
            # Learn fingerprints in the background (fast verified jumps for the
            # preview slider); nothing waits for it.
            threading.Thread(target=self.reader.build_index, daemon=True, name="mtt-video-index").start()

    def count(self) -> int:
        return self.reader.count_known() if self.reader is not None else len(self.paths)

    def stop_background(self) -> None:
        """Stop the background fingerprint pass (tracking needs the CPU)."""
        if self.reader is not None:
            self.reader.stop_index()

    def prepare(self, first: int, last: int, cancelled=lambda: False, progress=None) -> None:
        """Video: read frames first..last once IN ORDER, so that every one of
        them can later be reached by a verified jump (the backward pass reads
        them in reverse). Frames already known are skipped. Images: nothing."""
        if self.reader is None:
            return
        for k in range(first, last + 1):
            if cancelled():
                return
            if not self.reader.knows(k):
                if self.reader.read(k) is None:
                    return                          # past the real end
            if progress is not None:
                progress(k - first + 1, last - first + 1)

    def read_rgb(self, i: int):
        if self.reader is not None:
            return self.reader.read(i)
        return read_rgb(self.paths[i]) if 0 <= i < len(self.paths) else None

    def read_gray(self, i: int):
        if self.reader is None:
            return gray_of(self.read_rgb(i))
        with self._lock:
            backward = self._last is not None and i < self._last
            self._last = i
            hit = self._gray_cache.pop(i, None)
            if hit is not None:
                return hit
            if not backward:
                self._gray_cache.clear()
                return gray_of(self.reader.read(i))
            # Backward: decode frames i-BLOCK+1 .. i in order (one jump, then
            # sequential reads) and keep them for the next requests.
            self._gray_cache.clear()
            first = max(0, i - self.BLOCK + 1)
            for k in range(first, i):
                self._gray_cache[k] = gray_of(self.reader.read(k))
            return gray_of(self.reader.read(i))

    def read_gray_timed(self, i: int):
        t = time.perf_counter()
        gray = self.read_gray(i)
        return gray, time.perf_counter() - t

    def name(self, i: int) -> str:
        if self.reader is not None:
            return f"{os.path.basename(self.paths[0])}  frame {i + 1}"
        return os.path.basename(self.paths[i]) if 0 <= i < len(self.paths) else ""

    def describe(self) -> str:
        n = self.count()
        if self.reader is not None:
            r = self.reader
            if r.frame_count is not None:
                state = f"{n} frames"
            else:
                state = f"{n} frames (from the file header)"
            return (f"Video: {os.path.basename(self.paths[0])}\n"
                    f"{r.width} x {r.height}, {r.fps:.3f} fps, {state}")
        return (f"{n} image files\nfirst: {os.path.basename(self.paths[0])}\n"
                f"last:  {os.path.basename(self.paths[-1])}")

    def close(self) -> None:
        if self.reader is not None:
            self.reader.close()


def _tracking_passes(S: int, ref_index: int | None, ref_before: bool) -> list[list[int]]:
    """Order in which the result frames 0..S-1 are tracked: outward from the
    reference frame, each pass starting again from the reference.
    ref_index: the reference frame's index in the result, or None when it is
    outside the tracked range (then ref_before tells on which side)."""
    if ref_index is not None:
        return [p for p in (list(range(ref_index + 1, S)), list(range(ref_index - 1, -1, -1))) if p]
    return [list(range(S))] if ref_before else [list(range(S - 1, -1, -1))]


def _tracking_worker(job: dict, result: dict, state: dict) -> None:
    """Background thread: track the mesh through all frames of the job.
    Writes frame k into the preallocated result arrays, marks it in
    result['filled'] and only THEN advances state['done'] (a count), so the
    Tk thread can safely read every frame whose 'filled' flag is set.

    The mesh is defined on the REFERENCE frame (job["ref_seq"]). Tracking
    runs outward from it (see _tracking_passes): forward to the end and,
    when the reference lies inside the range, backward to the start -- each
    pass starts again from the reference frame and the mesh positions.

    Two reference styles, chosen by job["update_template"] -- see the module
    docstring's "Tracking strategy" section for the full rationale:

      * Fixed reference (update_template=False): every point's template
        always comes from the reference frame at the original mesh position.
        Only ONE image (the reference) is ever needed as a template source.
      * Incremental reference (update_template=True): each point's template
        is refreshed, every time that point is found, to the frame and
        position it was just found at. Points differ in which past frame
        they currently reference (most will share the same one -- typically
        "the previous frame" -- but a point that has been failing keeps
        pointing at whatever frame it last succeeded in), so points are
        grouped by their current reference frame and matched in one batched
        call per group. A small frame cache holds only frames that are
        still referenced by at least one point.

    Speed-up (see node_editor/mesh_track_parallel.py):
      * each frame's points are split over several worker PROCESSES (frames
        must stay in order: frame k starts from frame k-1's result);
      * frames are passed to the workers through shared memory;
      * the next frame is decoded in a helper thread while the current one
        is being matched;
      * the number of processes is AUTO-TUNED on the first frames: 1, 2,
        4, ... processes are tried on one frame each, stopping when it gets
        slower, and the fastest count is used for the rest of the run.
    state["info"] carries what the Run panel shows (processes, timings).
    """
    from node_editor import mesh_track_parallel as mtp

    REF = -1                               # frame-cache key of the reference frame
    info = state["info"]
    frames: dict = {}                      # template-source key (REF or result index) -> SharedFrame
    matcher = None
    prefetch = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mtt-read")
    try:
        # source: FrameSource (image files or a video); seq[k] = the source's
        # frame index of result row k.
        source, seq = job["source"], job["seq"]
        cfg, update_template = job["cfg"], job["update_template"]
        S = len(seq)
        ref_index = job["ref_index"]
        # A backward pass reads frames in reverse. For a video, first read
        # that stretch once in order (exact) so the reverse reads can jump
        # with verification instead of decoding from the start every time.
        backward = (ref_index > 0) if ref_index is not None else not job["ref_before"]
        if backward:
            hi = job["ref_seq"] - 1 if ref_index is not None else seq[-1]

            def _progress(done, total):
                info["phase"] = f"reading the video in order for the backward pass ({done}/{total})"
            source.prepare(seq[0], hi, cancelled=state["cancel"].is_set, progress=_progress)
            if state["cancel"].is_set():
                return
        gray_ref = source.read_gray(job["ref_seq"])
        if gray_ref is None:
            raise RuntimeError(f"cannot read the reference frame: {source.name(job['ref_seq'])}")
        mesh_pts = job["mesh_pts"]
        M, N = mesh_pts.shape[:2]
        P = M * N
        pts0 = np.ascontiguousarray(mesh_pts.reshape(-1, 2), dtype=np.float64)

        if ref_index is not None:          # the reference frame's result IS the mesh
            result["points"][ref_index] = mesh_pts
            result["status"][ref_index] = 1
            result["confidence"][ref_index] = 1.0
            result["filled"][ref_index] = True
            state["done"] = 1

        candidates = mtp.candidate_counts(P, job.get("max_processes") or mtp.max_processes())
        info.update(points=P, image=f"{gray_ref.shape[1]} x {gray_ref.shape[0]}", cpus=os.cpu_count(),
                    candidates=candidates, tune={}, processes=candidates[0],
                    phase="starting worker processes" if len(candidates) > 1 else "running")
        matcher = mtp.ParallelMatcher(max(candidates))
        matcher.warm_up()
        n_now, tuning, tune_pos = candidates[0], len(candidates) > 1, 0
        if tuning:
            info["phase"] = "auto-tuning"
        frames[REF] = mtp.SharedFrame(gray_ref)
        # Per-frame timings (seconds) for the Run panel: total wall-clock time
        # of the frame step, image read+decode (done in the read-ahead thread,
        # overlapping the previous frame's matching), and template matching.
        times = {"total": [], "read": [], "tm": []}

        for order in _tracking_passes(S, ref_index, job["ref_before"]):
            if state["cancel"].is_set():
                break
            # Every pass starts from the reference frame and the mesh.
            for key in [k for k in frames if k != REF]:
                frames.pop(key).release()
            # template_frame[p] / template_pos[p]: which frame, and which
            # sub-pixel position within it, point p's template is cut from.
            # search_center[p]: current best guess of point p's location, used
            # as next_pts_init -- this always advances to the latest successful
            # match in BOTH reference styles; only template_frame/template_pos
            # are held back (never updated) in the fixed style.
            template_frame = np.full(P, REF, dtype=np.int64)
            template_pos = pts0.copy()
            search_center = pts0.copy()

            pending = prefetch.submit(source.read_gray_timed, seq[order[0]])
            for pos, k in enumerate(order):
                if state["cancel"].is_set():
                    break
                t_frame = time.perf_counter()
                gray, t_read = pending.result()
                pending = (prefetch.submit(source.read_gray_timed, seq[order[pos + 1]])
                           if pos + 1 < len(order) else None)
                if gray is None:            # one bad frame must not kill a long run
                    state["bad_frames"] += 1
                    result["filled"][k] = True
                    state["done"] += 1
                    continue
                cur = mtp.SharedFrame(gray)
                alive = {f.shm.name for f in frames.values()} | {cur.shm.name}

                nxt_all = np.full((P, 2), np.nan)
                status_all = np.zeros(P, dtype=np.uint8)
                conf_all = np.full(P, np.nan)

                t_match = time.perf_counter()
                # Group points by their current template-source frame -- in the
                # common case (no failures, incremental or fixed) this is a
                # single group and costs exactly one (parallel) call.
                for frame_key in np.unique(template_frame):
                    mask = template_frame == frame_key
                    nxt, st, cf = matcher.match(frames[int(frame_key)], cur, template_pos[mask],
                                                search_center[mask].copy(), cfg, n_now, alive)
                    group_idx = np.nonzero(mask)[0]
                    nxt_all[group_idx] = np.asarray(nxt, dtype=np.float64).reshape(-1, 2)
                    status_all[group_idx] = np.asarray(st).reshape(-1).astype(np.uint8)
                    conf_all[group_idx] = np.asarray(cf, dtype=np.float64).reshape(-1)
                dt = time.perf_counter() - t_match

                if tuning:
                    info["tune"][n_now] = dt
                    best = min(info["tune"], key=info["tune"].get)
                    tune_pos += 1
                    # Stop at the end of the list, or as soon as more processes got
                    # clearly slower (beyond that point it only gets worse).
                    if tune_pos >= len(candidates) or (n_now != best and dt > 1.05 * info["tune"][best]):
                        tuning, n_now = False, best
                        info["phase"] = "running"
                    else:
                        n_now = candidates[tune_pos]
                info["processes"] = n_now

                good = (status_all != 0) & np.isfinite(nxt_all).all(axis=1)
                search_center[good] = nxt_all[good]
                if update_template:
                    template_frame[good] = k
                    template_pos[good] = nxt_all[good]
                    frames[k] = cur
                    referenced = set(int(v) for v in template_frame)
                    for cached_key in [i for i in frames if i not in referenced and i != REF]:
                        frames.pop(cached_key).release()
                else:
                    # Fixed reference: the reference frame is the only template
                    # source, so the current frame is not needed after this step.
                    cur.release()

                out = np.where(good[:, None], nxt_all, np.nan)
                result["points"][k] = out.reshape(M, N, 2)
                result["status"][k] = good.reshape(M, N).astype(np.uint8)
                result["confidence"][k] = conf_all.reshape(M, N)
                result["filled"][k] = True
                state["done"] += 1

                for key, value in (("total", time.perf_counter() - t_frame), ("read", t_read), ("tm", dt)):
                    times[key].append(value)
                    info[f"avg_{key}_ms"] = 1000.0 * float(np.mean(times[key][-20:]))   # last 20 frames
                info["last_total_ms"] = 1000.0 * times["total"][-1]
            if pending is not None:
                pending.cancel()
    except Exception as e:              # reported to the UI thread by the poller
        state["error"] = f"{type(e).__name__}: {e}"
    finally:
        prefetch.shutdown(wait=True, cancel_futures=True)
        if matcher is not None:
            matcher.shutdown()
        for f in frames.values():
            f.release()
        info["phase"] = "finished"
        state["finished"] = True


# ══ Node ══════════════════════════════════════════════════════════════════

class MeshTemplateTrackNode(BaseNode):
    """Track a regular mesh of points through an image sequence. See the
    module docstring and _HELP_TEXT."""

    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE = "mesh_template_track"
    DISPLAY_NAME = "Mesh Template Track"
    CATEGORY = "process"
    SEARCH_KEYWORDS = ("mesh", "grid", "template", "match", "track", "tracking",
                       "image sequence", "displacement", "dic", "camera motion")
    NODE_WIDTH = 240
    NODE_HEIGHT = 110
    HELP_TEXT = _HELP_TEXT

    # (StringVar attribute, key in saved settings, key in _defaults())
    _SETTING_KEYS = (
        ("_cells_var", "cells", "cells"), ("_tl_var", "top_left", "tl"),
        ("_lr_var", "lower_right", "lr"),
        ("_tmpl_w_var", "template_w", "tmpl_w"), ("_tmpl_h_var", "template_h", "tmpl_h"),
        ("_search_w_var", "search_w", "search_w"), ("_search_h_var", "search_h", "search_h"),
        ("_levels_var", "num_levels", "levels"), ("_method_var", "match_method", "method"),
        ("_mincorr_var", "min_correlation", "min_corr"),
        ("_fine_w_var", "fine_w", "fine_w"), ("_fine_h_var", "fine_h", "fine_h"),
        ("_sidelobe_var", "sidelobe", "sidelobe"),
        ("_start_var", "start_frame", "start"), ("_end_var", "end_frame", "end"),
        ("_ref_var", "reference_frame", "ref"),
    )
    _FIELD_LABELS = {
        "cells": "cells (M, N)", "tl": "top-left corner", "lr": "bottom-right corner",
        "tmpl_w": "template width", "tmpl_h": "template height",
        "search_w": "search range width", "search_h": "search range height",
        "levels": "number of levels", "min_corr": "min. correlation",
        "fine_w": "fine radius width", "fine_h": "fine radius height",
        "sidelobe": "sidelobe radius", "start": "start frame", "end": "end frame",
        "ref": "reference frame",
    }

    def __init__(self, node_id: str, canvas: tk.Canvas):
        super().__init__(node_id, canvas)
        # Plain Python state only. The toolbox instantiates throw-away nodes
        # while dragging, so Tk variables are created lazily in _init_state().
        self._files = None             # image paths from the 'files' input pin
        self._pin_source = None        # FrameSource of those files (when the pin delivers any)
        self._local_spec = None        # {"kind": "images"|"video", "paths": [abs]} chosen in the inspector
        self._local_source = None      # FrameSource opened from _local_spec (None if it failed)
        self._local_error = ""
        self._video_poll_id = None
        self._result = None            # dict, see _new_result()
        self._run_info_last = "Not run yet. Press Run: the number of worker processes\nis chosen automatically on the first frames."
        self._result_path = ""
        self._result_saved = True
        self._result_error = ""
        self._running = False
        self._run_state = None
        self._poll_id = None
        self._outputs_dirty = True
        self._destroyed = False
        self._status_item = None
        self._lr_auto = True           # bottom-right follows the image size until edited
        self._tmpl_mesh_seen = None    # mesh the template size last followed (or _LOADED)
        self._end_auto = True          # end frame follows the sequence length until edited
        self._sel = (0, 0)             # selected cell (row, column)
        self._pick = None              # None, "tl" or "lr": corner being picked in the preview
        self._pick_orig = None         # (text, lr_auto) restored when the pick is cancelled
        self._pick_hover = None        # (u, v) integer pixel under the cursor while picking
        self._view_frame = 1
        self._preview_rgb = None
        self._image_size = None        # (W, H) of the full-resolution image
        self._preview_frame = None
        self._preview_msg = ""
        self._image_serial = 0
        self._preview_token = 0
        self._preview_req = None
        self._preview_poll_id = None
        self._preview_debounce_id = None
        self._preview_executor = None
        self._photo = None
        self._photo_key = None
        self._view = None              # (ox, oy, scale) image -> canvas mapping, current render
        self._series_user_view = False
        self._last_plot_done = 0
        self._view_filled_drawn = False
        self._series_match_state = None
        self._suspend_validate = False
        self._programmatic = False
        self._copy_job = None          # running frame copy: {"state", "kind", ...}
        self._copy_done = None         # last finished copy: {"files", "dest"} for 'Create Image Sequence node'
        self._copy_check_id = None
        self._reset_widget_refs()

    def _reset_widget_refs(self) -> None:
        self._entries = {}
        self._run_info_text = None
        self._c_img = None
        self._slider = None
        self._fig = self._ax = self._canvas_mpl = None
        self._pan = {"active": False, "x_px": None, "y_px": None,
                     "xlim0": None, "ylim0": None, "x_scale": None, "y_scale": None}
        # Image-preview pan/zoom state. Reset whenever the inspector (and
        # its canvas widget) is rebuilt: the persisted ox/oy/scale are in
        # the OLD canvas's pixel coordinates, and a freshly built canvas
        # may have a different size -- simplest and safest is to just
        # re-fit on (re)open rather than carry an absolute pixel-space
        # view across different canvas widgets.
        self._img_view = None                 # (ox, oy, scale) once computed/user-set
        self._img_user_view = False           # True once the user has panned/zoomed
        self._img_last_size = None            # (W, H) last seen, to detect a new image resolution
        self._img_pan = {"active": False, "x_px": None, "y_px": None,
                         "moved": False, "ox0": None, "oy0": None}
        self._progress = None
        self._summary_label = self._result_label = None
        self._btn_run = self._btn_stop = self._btn_load = self._btn_save = None
        self._btn_save_now = None
        self._local_buttons = []
        self._btn_pick = {}                   # "tl"/"lr" -> Pick button
        self._help_popup = None
        self._copy_btn = self._copy_cancel_btn = self._copy_node_btn = None
        self._copy_info_label = self._copy_sel_entry = self._copy_pattern_entry = None
        self._copy_number_cb = self._copy_quality_entry = None

    @staticmethod
    def _defaults() -> dict:
        return {"cells": "6 8", "tl": "0 0", "lr": "", "tmpl_w": "31", "tmpl_h": "31",
                "search_w": "48", "search_h": "48", "levels": "3",
                "method": "TM_CCOEFF_NORMED", "min_corr": "0.5",
                "fine_w": "4", "fine_h": "4", "sidelobe": "2", "start": "1", "end": "",
                "ref": "1"}

    def _init_state(self) -> None:
        if hasattr(self, "_cells_var"):
            return
        d = self._defaults()
        self._cells_var = tk.StringVar(value=d["cells"])
        self._tl_var = tk.StringVar(value=d["tl"])
        self._lr_var = tk.StringVar(value=d["lr"])
        self._tmpl_w_var = tk.StringVar(value=d["tmpl_w"])
        self._tmpl_h_var = tk.StringVar(value=d["tmpl_h"])
        self._search_w_var = tk.StringVar(value=d["search_w"])
        self._search_h_var = tk.StringVar(value=d["search_h"])
        self._levels_var = tk.StringVar(value=d["levels"])
        self._method_var = tk.StringVar(value=d["method"])
        self._subpixel_var = tk.BooleanVar(value=True)
        self._update_template_var = tk.BooleanVar(value=False)
        self._mincorr_var = tk.StringVar(value=d["min_corr"])
        self._fine_w_var = tk.StringVar(value=d["fine_w"])
        self._fine_h_var = tk.StringVar(value=d["fine_h"])
        self._sidelobe_var = tk.StringVar(value=d["sidelobe"])
        self._start_var = tk.StringVar(value=d["start"])
        self._ref_var = tk.StringVar(value=d["ref"])          # frame the mesh / templates are defined on
        self._end_var = tk.StringVar(value=d["end"])
        self._show_tracked_var = tk.BooleanVar(value=True)
        self._show_cells_var = tk.BooleanVar(value=True)      # cell grid lines in the preview
        self._show_centers_var = tk.BooleanVar(value=True)    # cell-centre dots in the preview
        self._files_info_var = tk.StringVar(value="No image sequence connected.")
        self._summary_var = tk.StringVar(value="")
        self._mesh_info_var = tk.StringVar(value="")
        self._result_info_var = tk.StringVar(value="")
        self._result_file_var = tk.StringVar(value="")       # read-only "File" field in the Data group
        self._result_file_note_var = tk.StringVar(value="")
        self._progress_var = tk.StringVar(value="Idle")
        self._frame_label_var = tk.StringVar(value="")
        self._sel_info_var = tk.StringVar(value="")
        self._slider_var = tk.IntVar(value=1)
        # 'Copy selected frames to a folder'
        self._copy_dir_var = tk.StringVar(value="")              # absolute folder ("" = none)
        self._copy_frames_var = tk.StringVar(value="")
        self._copy_mode_var = tk.StringVar(value="original")     # "original" | "pattern"
        self._copy_pattern_var = tk.StringVar(value="selected_%06d")
        self._copy_number_var = tk.StringVar(value=self._COPY_NUMBERING[0])
        self._copy_format_var = tk.StringVar(value="png")       # "jpg" | "png" | "same"
        self._copy_quality_var = tk.StringVar(value="95")
        self._copy_info_var = tk.StringVar(value="")
        self._copy_status_var = tk.StringVar(value="")
        self._copy_progress_var = tk.DoubleVar(value=0.0)
        for var in (self._copy_dir_var, self._copy_frames_var, self._copy_mode_var, self._copy_pattern_var,
                    self._copy_number_var, self._copy_format_var, self._copy_quality_var):
            var.trace_add("write", lambda *_a: self._schedule_copy_check())
        for attr, _saved, _key in self._SETTING_KEYS:
            if attr not in ("_lr_var", "_end_var"):
                getattr(self, attr).trace_add("write", self._on_var_written)
        self._subpixel_var.trace_add("write", self._on_var_written)
        self._update_template_var.trace_add("write", self._on_var_written)
        self._lr_var.trace_add("write", self._on_lr_written)
        self._end_var.trace_add("write", self._on_end_written)
        self._ref_var.trace_add("write", self._on_ref_written)
        self._show_tracked_var.trace_add("write", lambda *_a: self._redraw_preview())
        self._show_cells_var.trace_add("write", lambda *_a: self._redraw_preview())
        self._show_centers_var.trace_add("write", lambda *_a: self._redraw_preview())

    # ── variable helpers ────────────────────────────────────────────────
    def _set_var(self, var, value) -> None:
        """Programmatic write. Writes that do NOT go through here are user
        edits (used to switch the two 'auto' fields to manual)."""
        self._programmatic = True
        try:
            var.set(value)
        finally:
            self._programmatic = False

    def _on_var_written(self, *_args) -> None:
        self._validate_all()

    def _on_lr_written(self, *_args) -> None:
        if not self._programmatic:
            self._lr_auto = False
        self._validate_all()

    def _on_ref_written(self, *_args) -> None:
        """A new (valid) reference frame typed by the user: show that frame
        in the preview, since the mesh is defined on it."""
        if self._programmatic:
            return
        cfg, bad = self._collect()
        ref = cfg.get("ref")
        if ref is None or "ref" in bad or not self._n_frames() or ref == self._view_frame:
            return
        self._view_frame = ref
        self._sync_slider(cfg, bad)
        self._schedule_preview()

    def _on_end_written(self, *_args) -> None:
        if not self._programmatic:
            self._end_auto = False
        self._validate_all()

    # ── pins ────────────────────────────────────────────────────────────
    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[PinDef("files", PinType.STRING, "files", optional=True)],
            outputs=[
                PinDef("points", PinType.ARRAY, "points", shape=(-1, -1, -1, 2), dtype="float64"),
                PinDef("status", PinType.ARRAY, "status", shape=(-1, -1, -1), dtype="uint8"),
                PinDef("confidence", PinType.ARRAY, "conf", shape=(-1, -1, -1), dtype="float64"),
                PinDef("frames", PinType.ARRAY, "frames", shape=(-1,), dtype="float64"),
            ],
        )

    # ── frame source: the pin's images, else the inspector's own selection ──
    def _source(self):
        return self._pin_source if self._pin_source is not None else self._local_source

    def _n_frames(self) -> int:
        src = self._source()
        return src.count() if src is not None else 0

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
            old_n = self._n_frames()
            self._files = files
            self._pin_source = FrameSource("images", files) if files else None
            self._on_files_changed(old_n)
        # Do not push partial data downstream while tracking, and do not
        # re-push unchanged results when only the (unchanged) input file list
        # is re-delivered (e.g. on every frame emitted by an Image Sequence).
        if self._running or not self._outputs_dirty:
            return {"_skip_downstream": True, "_preserve_cache": True}
        self._outputs_dirty = False
        return self._outputs()

    def _outputs(self) -> dict:
        r = self._result
        if r is None or r["n_done"] <= 0:
            return {}
        S = r["points"].shape[0]
        return {"points": r["points"], "status": r["status"], "confidence": r["confidence"],
                "frames": (r["start"] + np.arange(S)).astype(np.float64)}

    def _notify_result_changed(self) -> None:
        self._outputs_dirty = True
        if self._request_downstream is not None:
            self._request_downstream(self.node_id)

    # ── serialization ───────────────────────────────────────────────────
    def _settings_dict(self) -> dict:
        self._init_state()
        s = {saved: getattr(self, attr).get() for attr, saved, _k in self._SETTING_KEYS}
        s.update(subpixel=bool(self._subpixel_var.get()), show_tracked=bool(self._show_tracked_var.get()),
                 show_cells=bool(self._show_cells_var.get()),
                 show_centers=bool(self._show_centers_var.get()),
                 update_template=bool(self._update_template_var.get()),
                 lr_auto=bool(self._lr_auto), end_auto=bool(self._end_auto))
        return s

    def _apply_settings(self, s: dict, from_file: bool) -> None:
        self._init_state()
        d = self._defaults()
        self._suspend_validate = True
        try:
            for attr, saved, key in self._SETTING_KEYS:
                self._set_var(getattr(self, attr), str(s.get(saved, d[key])))
            if "reference_frame" not in s:
                # Saved before the reference frame existed: the reference
                # was always the start frame.
                self._set_var(self._ref_var, str(s.get("start_frame", d["ref"])))
            self._set_var(self._subpixel_var, bool(s.get("subpixel", True)))
            self._set_var(self._show_tracked_var, bool(s.get("show_tracked", True)))
            self._set_var(self._show_cells_var, bool(s.get("show_cells", True)))
            self._set_var(self._show_centers_var, bool(s.get("show_centers", True)))
            self._set_var(self._update_template_var, bool(s.get("update_template", False)))
        finally:
            self._suspend_validate = False
        # Values loaded from a result file are explicit: never overwritten by
        # the current sequence's image size / length.
        self._lr_auto = False if from_file else bool(s.get("lr_auto", False))
        self._end_auto = False if from_file else bool(s.get("end_auto", False))
        self._tmpl_mesh_seen = _LOADED    # keep the loaded template size (see _follow_mesh_with_template)
        self._validate_all()

    def get_params(self) -> dict:
        base = self._get_project_base()
        path = self._result_path
        rel = self._to_relative(self._to_absolute(path, base), base) if path else ""
        dest = self._copy_dir_var.get().strip() if hasattr(self, "_copy_dir_var") else ""
        copy_cfg = {} if not hasattr(self, "_copy_dir_var") else {
            "folder": self._to_relative(dest, base) if dest else "", "frames": self._copy_frames_var.get(),
            "names": self._copy_mode_var.get(), "pattern": self._copy_pattern_var.get(),
            "numbering": "frame" if self._copy_number_var.get() == self._COPY_NUMBERING[0] else "sequence",
            "format": self._copy_format_var.get(), "jpeg_quality": self._copy_quality_var.get()}
        return {"settings": self._settings_dict(), "result_file": rel,
                "input": self._local_input_params(), "copy_frames": copy_cfg}

    def set_params(self, params: dict) -> None:
        self._init_state()
        self._apply_settings(params.get("settings", {}) or {}, from_file=False)
        self._restore_local_input(params.get("input"))
        c = params.get("copy_frames") or {}
        if c:
            folder = str(c.get("folder", "") or "")
            self._copy_dir_var.set(self._to_absolute(folder, self._get_project_base()) if folder else "")
            self._copy_frames_var.set(str(c.get("frames", "")))
            self._copy_mode_var.set("pattern" if c.get("names") == "pattern" else "original")
            self._copy_pattern_var.set(str(c.get("pattern", "selected_%06d")))
            self._copy_number_var.set(self._COPY_NUMBERING[1 if c.get("numbering") == "sequence" else 0])
            self._copy_format_var.set(c.get("format") if c.get("format") in self._COPY_FORMATS else "png")
            self._copy_quality_var.set(str(c.get("jpeg_quality", "95")))
        path = str(params.get("result_file", "") or "")
        if path:
            abs_path = self._to_absolute(path, self._get_project_base())
            # The path is the node's property: keep it even when the file is
            # missing or unreadable, so saving the project does not drop it.
            self._result_path = abs_path
            try:
                _settings, result = self._read_track_file(abs_path)
                if result is not None:
                    self._result, self._result_path = result, abs_path
                    self._result_saved, self._result_error = True, ""
                    self._outputs_dirty = True
            except Exception as e:      # missing/corrupt file must not break project loading
                self._result_error = f"Result file could not be loaded:\n{abs_path}\n({e})"
        self._update_result_info()
        self._update_status_item()

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

    # ── node face ───────────────────────────────────────────────────────
    def _status_text(self) -> str:
        """Node face: line 1 = current input file count, line 2 = tracking
        state / result. Kept apart because the result's frame count is NOT
        the input file count -- after the upstream list changed (e.g. 4004 ->
        3911 files) the result still has the old number of frames, which on
        a single line looked like a stale file count."""
        n = self._n_frames()
        src = self._source()
        files = (f"{n} {'video frames' if src.kind == 'video' else 'files'}" if n else "no input")
        if self._running and self._run_state is not None:
            return f"{files}\ntracking {self._run_state['done']}/{self._run_state['total']}"
        r = self._result
        if r is None:
            return f"{files}\n" + ("result file error" if self._result_error else "no result")
        S, M, N = r["points"].shape[:3]
        # Short lines: the face has room for ~4 unwrapped lines between the pins.
        lines = [files, f"result: {M}x{N} pts",
                 f"{r['n_done']} frames" + (" (partial)" if r["n_done"] < S else "")]
        if n and r.get("n_files") and r["n_files"] != n:
            lines.append(f"(made from {r['n_files']} files)")
        return "\n".join(lines)

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
            x, y, x + w, y + h, fill="#eef3fb", outline="#3f6db3", width=2,
            tags=(self.node_id, "node_body"))
        self._title_item = self.canvas.create_text(
            x + w / 2, y + 13, text=self.get_canvas_title(), font=("Arial", 9, "bold"),
            fill="#1c3a6b", tags=(self.node_id,))
        cx, tw = self._text_column(x, w)
        self._hint_item = self.canvas.create_text(
            cx, y + 33, text="files -> mesh points", font=("Arial", 8),
            fill="#35507d", width=tw, justify="center", tags=(self.node_id,))
        # Anchored at the bottom edge so the 2-3 line status grows upward.
        self._status_item = self.canvas.create_text(
            cx, y + h - 5, text=self._status_text(), font=("Arial", 7),
            fill="#456", width=tw, anchor="s", justify="center", tags=(self.node_id,))
        self._canvas_items += [self._body_rect, self._title_item, self._hint_item, self._status_item]

    _PIN_LABEL_IN, _PIN_LABEL_OUT = 40, 58      # px taken by input / output pin labels

    def _text_column(self, x, w) -> tuple[float, float]:
        """Centre x and wrap width of the free column between the input-pin
        labels (left) and the output-pin labels (right), so the face text
        does not run under 'points' / 'status' / 'frames'."""
        left, right = x + self._PIN_LABEL_IN, x + w - self._PIN_LABEL_OUT
        return (left + right) / 2, max(60, right - left)

    def on_resize(self, old_width, old_height, new_width, new_height) -> None:
        cx, tw = self._text_column(self.x, new_width)
        if self._status_item is not None:
            self.canvas.coords(self._status_item, cx, self.y + new_height - 5)
            self.canvas.itemconfigure(self._status_item, width=tw)
        if getattr(self, "_hint_item", None) is not None:
            self.canvas.coords(self._hint_item, cx, self.y + 33)
            self.canvas.itemconfigure(self._hint_item, width=tw)

    def get_help_text(self) -> str:
        return _HELP_TEXT

    # ══ Validation ═══════════════════════════════════════════════════════
    def _collect(self):
        """Parse every setting. Returns (cfg, bad): cfg holds the parsed
        values (None where invalid or still pending), bad is the set of
        field keys with invalid text."""
        self._init_state()
        bad: set = set()
        cfg: dict = {}

        def int_of(var, key, minimum):
            v = parse_numbers(var.get(), 1, integer=True)
            if v is None or v[0] < minimum:
                bad.add(key)
                return None
            return v[0]

        cells = parse_numbers(self._cells_var.get(), 2, integer=True)
        if cells is None or cells[0] < 1 or cells[1] < 1:
            bad.add("cells")
            cells = None
        cfg["cells"] = cells

        tl = parse_numbers(self._tl_var.get(), 2)
        if tl is None:
            bad.add("tl")
        cfg["tl"] = tl

        lr_text = self._lr_var.get().strip()
        if not lr_text and self._lr_auto:
            lr = None                                # pending: image size unknown yet
        else:
            lr = parse_numbers(lr_text, 2)
            if lr is None:
                bad.add("lr")
        cfg["lr"] = lr
        if tl is not None and lr is not None and (lr[0] <= tl[0] or lr[1] <= tl[1]):
            bad.update(("tl", "lr"))

        def pair(vw, vh, kw, kh, minimum):
            w, h = int_of(vw, kw, minimum), int_of(vh, kh, minimum)
            return (w, h) if w is not None and h is not None else None

        cfg["template"] = pair(self._tmpl_w_var, self._tmpl_h_var, "tmpl_w", "tmpl_h", 3)
        cfg["search"] = pair(self._search_w_var, self._search_h_var, "search_w", "search_h", 1)
        cfg["levels"] = int_of(self._levels_var, "levels", 1)
        cfg["fine"] = pair(self._fine_w_var, self._fine_h_var, "fine_w", "fine_h", 0)
        cfg["sidelobe"] = int_of(self._sidelobe_var, "sidelobe", 0)

        mc = parse_numbers(self._mincorr_var.get(), 1)
        if mc is None or not (0.0 <= mc[0] <= 1.0):
            bad.add("min_corr")
            cfg["min_corr"] = None
        else:
            cfg["min_corr"] = mc[0]

        cfg["method"] = self._method_var.get()
        cfg["subpixel"] = bool(self._subpixel_var.get())
        cfg["update_template"] = bool(self._update_template_var.get())

        ref = int_of(self._ref_var, "ref", 1)
        n_frames = self._n_frames()
        if ref is not None and n_frames and ref > n_frames:
            bad.add("ref")
        cfg["ref"] = ref
        start = int_of(self._start_var, "start", 1)
        if not self._end_var.get().strip() and self._end_auto:
            end = None                               # pending: sequence length unknown yet
        else:
            end = int_of(self._end_var, "end", 1)
        if start is not None and end is not None and start > end:
            bad.update(("start", "end"))
        if n_frames:
            n = n_frames
            if start is not None and start > n:
                bad.add("start")
            if end is not None and end > n:
                bad.add("end")
        cfg["start"], cfg["end"] = start, end
        return cfg, bad

    @staticmethod
    def _mesh_from_cfg(cfg, bad):
        if any(k in bad for k in ("cells", "tl", "lr")):
            return None
        if cfg["cells"] is None or cfg["tl"] is None or cfg["lr"] is None:
            return None
        (M, N), (u0, v0), (u1, v1) = cfg["cells"], cfg["tl"], cfg["lr"]
        return (M, N, u0, v0, u1, v1)

    @staticmethod
    def _filled(res) -> np.ndarray:
        """Bool (S,): result frames that have been tracked (or are the
        reference). Frames are not filled in order: tracking runs outward
        from the reference frame."""
        filled = res.get("filled")
        if filled is None:                  # older in-memory results: a prefix
            filled = np.arange(res["points"].shape[0]) < res["n_done"]
        return filled

    @staticmethod
    def _result_matches(res, mesh) -> bool:
        if res is None or mesh is None:
            return False
        rm = res["mesh"]
        return (int(rm[0]) == mesh[0] and int(rm[1]) == mesh[1]
                and bool(np.allclose(rm[2:], mesh[2:], rtol=0.0, atol=1e-9)))

    def _run_blocker(self, cfg, bad):
        """Return why Run is not possible right now, or None."""
        if calcTemplateMatchPyr is None:
            return f"calcTemplateMatchPyr could not be imported: {_TRACKER_IMPORT_ERROR}"
        src = self._source()
        if src is None or not src.count():
            return ("No images: connect the 'files' input, or use 'Select a video ...' / "
                    "'Select images ...'.")
        if bad:
            return "Fix the values shown in red."
        if cfg["lr"] is None:
            return "Image size not known yet; enter the bottom-right corner."
        if cfg["end"] is None:
            return "End frame is not set."
        if cfg["ref"] is None:
            return "Reference frame is not set."
        return None

    def _outside_mask(self, mesh, cfg):
        """Bool (M, N): points whose template would extend outside the image."""
        size, tmpl = self._image_size, cfg["template"]
        if size is None or tmpl is None:
            return None
        W, H = size
        tw, th = tmpl
        c = mesh_centers(*mesh)
        u, v = c[..., 0], c[..., 1]
        return ((u - tw / 2.0 < -0.5) | (u + tw / 2.0 > W - 0.5)
                | (v - th / 2.0 < -0.5) | (v + th / 2.0 > H - 0.5))

    def _validate_all(self) -> None:
        if self._suspend_validate:
            return
        self._follow_mesh_with_template()
        cfg, bad = self._collect()
        for key, entries in self._entries.items():
            color = INVALID_FG if key in bad else NORMAL_FG
            for ent in entries:
                try:
                    ent.configure(fg=color)
                except tk.TclError:
                    pass
        blocker = self._run_blocker(cfg, bad)
        if bad:
            text, color = "Invalid: " + ", ".join(self._FIELD_LABELS[k] for k in sorted(bad)), INVALID_FG
        elif blocker:
            text, color = blocker, "#b36b00"
        else:
            text, color = "Settings OK - ready to run.", "#1a7a1a"
        self._summary_var.set(text)
        if self._summary_label is not None:
            try:
                self._summary_label.configure(fg=color)
            except tk.TclError:
                pass
        self._update_mesh_info(cfg, bad)
        self._set_state(self._btn_run, tk.NORMAL if (blocker is None and not self._running) else tk.DISABLED)
        self._sync_slider(cfg, bad)
        self._redraw_preview()
        # _redraw_preview() -> _draw_overlay() may clamp self._sel if the
        # mesh shrank, so refresh the selected-cell label AFTER it, not
        # before, to reflect any clamping.
        self._update_sel_info()
        match = self._result_matches(self._result, self._mesh_from_cfg(cfg, bad))
        if match != self._series_match_state:
            self._series_match_state = match
            self._redraw_series()

    @staticmethod
    def _set_state(widget, state) -> None:
        if widget is not None:
            try:
                widget.configure(state=state)
            except tk.TclError:
                pass

    def _update_mesh_info(self, cfg, bad) -> None:
        mesh = self._mesh_from_cfg(cfg, bad)
        if mesh is None:
            self._mesh_info_var.set("Waiting for valid cells and corners.")
            return
        M, N, u0, v0, u1, v1 = mesh
        text = f"{M} x {N} = {M * N} points; cell {(u1 - u0) / N:.2f} x {(v1 - v0) / M:.2f} px"
        outside = self._outside_mask(mesh, cfg)
        if outside is not None and outside.any():
            text += (f"\nWARNING: {int(outside.sum())} point(s) have a template extending outside "
                     "the image (orange in the preview); they may fail to track.")
        if M * N > MAX_DRAWN_POINTS:
            text += "\nToo many points: dots are not drawn in the preview."
        self._mesh_info_var.set(text)

    def _update_sel_info(self) -> None:
        """Refresh the persistent 'row X, col Y, pixel (u, v)' label shown
        above the plot. The pixel coordinate is always the geometric
        centre of the SELECTED MESH CELL, computed from the current mesh
        settings -- independent of whether a tracking result exists yet,
        so it is useful immediately, not only after Run.
        """
        cfg, bad = self._collect()
        mesh = self._mesh_from_cfg(cfg, bad)
        i, j = self._sel
        if mesh is None:
            self._sel_info_var.set(f"Selected cell: row {i + 1}, col {j + 1}  (mesh not valid yet)")
            return
        M, N = mesh[0], mesh[1]
        i = min(i, M - 1)
        j = min(j, N - 1)
        u, v = mesh_centers(*mesh)[i, j]
        self._sel_info_var.set(
            f"Selected cell: row {i + 1}, col {j + 1}   |   pixel centre (u, v) = ({u:.2f}, {v:.2f})"
        )

    # ══ Files / preview image ════════════════════════════════════════════
    def _on_files_changed(self, old_n: int = 0) -> None:
        """The frame source changed (pin input, own selection, or a video's
        frame count became exact)."""
        src = self._source()
        n = src.count() if src is not None else 0
        self._update_source_info()
        if n:
            # End follows the sequence length when it is automatic, and also
            # when it was set to the OLD last frame ("track to the end"): it is
            # often no longer automatic (after loading a result file or a
            # project), and would otherwise keep the old length (e.g. 4004
            # after the upstream list shrank to 3911) and block Run. An End
            # set to any other frame is the user's choice and is kept.
            if not self._end_auto and old_n and self._end_var.get().strip() == str(old_n):
                self._end_auto = True
            if self._end_auto:
                self._set_var(self._end_var, str(n))
        self._update_status_item()                   # node face shows the file count
        self._update_result_info()
        self._validate_all()
        self._schedule_preview(60)
        self._schedule_video_poll()

    def _update_source_info(self) -> None:
        """Input group text + which own-selection buttons are usable."""
        pin = self._pin_source is not None
        if pin:
            text = ("From the 'files' input:\n" + self._pin_source.describe()
                    + "\n(The Select buttons are used only when the input has no images.)")
        elif self._local_source is not None:
            text = "Selected here:\n" + self._local_source.describe()
        elif self._local_error:
            text = self._local_error
        else:
            text = ("No input. Connect the 'files' output of an Image Sequence node,\n"
                    "or select a video or image files below.")
        self._files_info_var.set(text)
        for btn in self._local_buttons:
            self._set_state(btn, tk.DISABLED if (pin or self._running) else tk.NORMAL)

    def _schedule_video_poll(self) -> None:
        """A selected video starts with the file header's frame count. Watch
        (cheaply, in the background) for the exact count, which arrives when
        the background pass or in-order reading reaches the real end; then
        End / validation follow. Nothing waits for it."""
        src = self._local_source
        r = src.reader if src is not None else None
        if (self._video_poll_id is not None or self._destroyed or r is None
                or r.frame_count is not None or r.index_error or r.index_stopped):
            return
        try:
            self._video_poll_id = self.canvas.after(1000, self._poll_video, src.count())
        except tk.TclError:
            self._video_poll_id = None

    def _poll_video(self, last_count: int) -> None:
        self._video_poll_id = None
        src = self._local_source
        if self._destroyed or src is None or src.reader is None:
            return
        if src.count() != last_count:
            self._on_files_changed(last_count)       # exact count: End / validation follow
        else:
            self._schedule_video_poll()

    def _frame_text(self, f: int) -> str:
        src = self._source()
        name = src.name(f - 1) if src is not None and 1 <= f <= src.count() else ""
        end = self._end_var.get().strip()
        text = f"Frame {f}" + (f" / {end}" if end.isdigit() else "")
        return text + (f"   {name}" if name else "")

    def _sync_slider(self, cfg, bad) -> None:
        sl = self._slider
        if sl is None:
            return
        s, e = cfg["start"], cfg["end"]
        if s is None or e is None or "start" in bad or "end" in bad or not self._n_frames():
            self._set_state(sl, tk.DISABLED)
            self._frame_label_var.set("Frame range not valid yet")
            return
        ref = cfg.get("ref")
        if ref is not None and "ref" not in bad:     # the reference may lie outside start..end
            s, e = min(s, ref), max(e, ref)
        try:
            sl.configure(state=tk.NORMAL, from_=s, to=e)
        except tk.TclError:
            return
        f = min(max(self._view_frame, s), e)
        if f != self._view_frame:
            self._view_frame = f
            self._schedule_preview()
        if self._slider_var.get() != f:
            self._slider_var.set(f)
        self._frame_label_var.set(self._frame_text(f))

    def _on_slider_moved(self, value) -> None:
        try:
            f = int(float(value))
        except ValueError:
            return
        self._frame_label_var.set(self._frame_text(f))     # cheap: label only while dragging

    def _on_slider_released(self, _event=None) -> None:
        f = int(self._slider_var.get())
        if f == self._view_frame and self._preview_frame == f:
            return
        self._view_frame = f
        self._request_preview()
        self._redraw_series()                               # moves the frame marker

    def _schedule_preview(self, delay: int = 300) -> None:
        if self._c_img is None:
            return
        if self._preview_debounce_id is not None:
            try:
                self.canvas.after_cancel(self._preview_debounce_id)
            except tk.TclError:
                pass
        self._preview_debounce_id = self.canvas.after(delay, self._request_preview)

    def _request_preview(self) -> None:
        self._preview_debounce_id = None
        if self._c_img is None or self._destroyed:
            return
        src, f = self._source(), self._view_frame
        if src is None or not (1 <= f <= src.count()):
            self._preview_rgb = self._image_size = None
            self._preview_msg = ""
            self._photo_key = None
            self._redraw_preview()
            return
        self._preview_token += 1
        req = {"token": self._preview_token, "frame": f, "source": src,
               "image": None, "size": None, "error": None, "done": False}
        self._preview_req = req
        self._frame_label_var.set(self._frame_text(f) + "   (loading ...)")
        if self._preview_executor is None:
            self._preview_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mtt-preview")
        self._preview_executor.submit(self._preview_worker, req)
        self._schedule_preview_poll()

    def _preview_worker(self, req: dict) -> None:
        """Worker thread: decode + downscale. Never touches Tk."""
        try:
            if req["token"] != self._preview_token:          # superseded before it started
                return
            rgb = req["source"].read_rgb(req["frame"] - 1)
            if rgb is None:
                req["error"] = "cannot read image"
                return
            h, w = rgb.shape[:2]
            longest = max(h, w)
            if longest > PREVIEW_MAX_SIDE:
                f = PREVIEW_MAX_SIDE / longest
                rgb = cv2.resize(rgb, (max(1, round(w * f)), max(1, round(h * f))),
                                 interpolation=cv2.INTER_AREA)
            req["image"], req["size"] = rgb, (w, h)
        except Exception as e:
            req["error"] = f"{type(e).__name__}: {e}"
        finally:
            req["done"] = True

    def _schedule_preview_poll(self) -> None:
        if self._preview_poll_id is None:
            try:
                self._preview_poll_id = self.canvas.after(40, self._poll_preview)
            except tk.TclError:
                pass

    def _poll_preview(self) -> None:
        self._preview_poll_id = None
        req = self._preview_req
        if req is None or self._destroyed:
            return
        if not req["done"]:
            self._schedule_preview_poll()
            return
        self._preview_req = None
        self._apply_preview(req)

    def _apply_preview(self, req: dict) -> None:
        if req["error"] or req["image"] is None:
            self._preview_rgb = self._image_size = None
            self._preview_frame = None
            self._preview_msg = f"Cannot read frame {req['frame']}:\n{req['source'].name(req['frame'] - 1)}"
        else:
            self._preview_rgb, self._image_size = req["image"], req["size"]
            self._preview_frame, self._preview_msg = req["frame"], ""
            self._image_serial += 1
            if self._lr_auto:
                W, H = req["size"]
                text = f"{W - 1} {H - 1}"
                if self._lr_var.get() != text:
                    self._set_var(self._lr_var, text)        # -> validation + redraw
        self._frame_label_var.set(self._frame_text(self._view_frame))
        self._validate_all()
        self._redraw_series()

    # ══ Inspector ════════════════════════════════════════════════════════
    def build_inspector(self, parent: tk.Frame) -> None:
        self._init_state()
        self._reset_widget_refs()
        win = self._inspector_win
        if win is not None:
            sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
            win.geometry(f"{min(1500, sw - 80)}x{min(820, sh - 120)}")
            win.minsize(min(1000, sw - 80), min(560, sh - 120))
            for seq in ("<Control-h>", "<Control-H>"):
                win.bind(seq, lambda _e: self._open_help())
            # Ctrl+S in the main window saves the project (node canvas binding);
            # inside this inspector it saves the tracking result instead.
            for seq in ("<Control-s>", "<Control-S>"):
                win.bind(seq, lambda _e: (self._on_save(), "break")[1])
        panes = tk.PanedWindow(parent, orient=tk.HORIZONTAL, sashwidth=6, sashrelief=tk.RAISED)
        panes.pack(fill="both", expand=True)
        left, right = tk.Frame(panes), tk.Frame(panes)
        panes.add(left, width=400, minsize=300)
        panes.add(right, minsize=460)
        inner, cv = self._make_scrollable(left)
        self._build_settings_panel(inner)
        self._bind_wheel_recursive(left, cv)
        self._build_right_panel(right)
        self._set_running_ui(self._running)
        self._update_result_info()
        self._validate_all()
        self._request_preview()

    @staticmethod
    def _make_scrollable(parent):
        outer = tk.Frame(parent)
        outer.pack(fill="both", expand=True)
        outer.grid_rowconfigure(0, weight=1)
        outer.grid_columnconfigure(0, weight=1)
        cv = tk.Canvas(outer, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=cv.yview)
        hsb = ttk.Scrollbar(outer, orient="horizontal", command=cv.xview)
        cv.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        cv.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        inner = tk.Frame(cv)
        cv.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all")))
        return inner, cv

    @staticmethod
    def _wheel_steps(e) -> int:
        if getattr(e, "num", None) == 4:
            return 1
        if getattr(e, "num", None) == 5:
            return -1
        d = getattr(e, "delta", 0)
        return int(d / 120) if abs(d) >= 120 else (1 if d > 0 else -1 if d < 0 else 0)

    def _bind_wheel_recursive(self, widget, cv) -> None:
        """Wheel scrolls the settings panel (Shift = horizontal). Bound per
        widget (not bind_all) so the editor's own wheel bindings stay intact;
        returning "break" stops comboboxes changing value on scroll."""
        def on_wheel(e):
            steps = self._wheel_steps(e)
            (cv.xview_scroll if e.state & 0x0001 else cv.yview_scroll)(-steps * 3, "units")
            return "break"

        def bind(w):
            for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                w.bind(seq, on_wheel, add="+")
            for child in w.winfo_children():
                bind(child)
        bind(widget)

    def _grid_row(self, parent, row, text, fields) -> None:
        tk.Label(parent, text=text, font=("Arial", 8), anchor="w").grid(
            row=row, column=0, sticky="w", padx=(0, 6), pady=2)
        for col, (var, key, width) in enumerate(fields, start=1):
            ent = tk.Entry(parent, textvariable=var, width=width, justify="center", font=("Arial", 9))
            ent.grid(row=row, column=col, sticky="w", padx=(0 if col == 1 else 4, 0), pady=2)
            self._entries.setdefault(key, []).append(ent)

    def _build_settings_panel(self, f: tk.Frame) -> None:
        pad = {"padx": 6, "pady": 4}

        box = tk.LabelFrame(f, text="Input", font=("Arial", 9), padx=6, pady=4)
        box.pack(fill="x", **pad)
        tk.Label(box, textvariable=self._files_info_var, font=("Arial", 8), anchor="w",
                 justify="left", wraplength=340).pack(fill="x")
        # Own selection, used only when the 'files' input delivers no images.
        sel = tk.Frame(box)
        sel.pack(fill="x", pady=(4, 0))
        video_btn = tk.Button(sel, text="Select a video ...", font=("Arial", 8),
                              command=self._on_select_video)
        video_btn.pack(side="left")
        images_btn = tk.Button(sel, text="Select images ...", font=("Arial", 8),
                               command=self._on_select_images)
        images_btn.pack(side="left", padx=(4, 0))
        clear_btn = tk.Button(sel, text="Clear", font=("Arial", 8), command=self._on_clear_local)
        clear_btn.pack(side="left", padx=(4, 0))
        self._local_buttons = [video_btn, images_btn, clear_btn]
        add_tooltip(video_btn, "Track the frames of an existing video file (mp4, avi, mov, ...) "
                               "directly, without an Image Sequence node. Frames are read in "
                               "order while tracking (exact); nothing is scanned in advance. "
                               "Only used when "
                               "the 'files' input has no images.")
        add_tooltip(images_btn, "Track existing image files directly, without an Image Sequence "
                                "node: pick several files (Ctrl / Shift-click); they are used in "
                                "file-name order. Only used when the 'files' input has no images.")
        add_tooltip(clear_btn, "Forget the video / images selected here.")
        self._update_source_info()

        box = tk.LabelFrame(f, text="Mesh settings", font=("Arial", 9), padx=6, pady=4)
        box.pack(fill="x", **pad)
        self._grid_row(box, 0, "Cells (M rows, N columns)", [(self._cells_var, "cells", 16)])
        self._grid_row(box, 1, "Top-left corner (u v)", [(self._tl_var, "tl", 16)])
        self._grid_row(box, 2, "Bottom-right corner (u v)", [(self._lr_var, "lr", 16)])
        for row, which, text in ((1, "tl", "Pick TL"), (2, "lr", "Pick BR")):
            btn = tk.Button(box, text=text, font=("Arial", 8), width=8,
                            command=lambda w=which: self._start_pick(w))
            btn.grid(row=row, column=2, sticky="w", padx=(4, 0), pady=2)
            self._btn_pick[which] = btn
            add_tooltip(btn, f"Pick the {'top-left' if which == 'tl' else 'bottom-right'} corner "
                             f"with the mouse in the image preview. The field follows the cursor; "
                             f"click to accept, Esc or right-click to cancel.")
        for widget in box.grid_slaves(row=0):
            add_tooltip(widget, "M = number of cell rows, N = number of cell columns. One "
                                "tracking point is placed at the centre of each cell, so the "
                                "mesh has M x N points.")
        tk.Button(box, text="Fit Mesh to Full Image", font=("Arial", 8),
                  command=self._on_fit_mesh_to_image).grid(
            row=3, column=0, columnspan=3, sticky="w", pady=(2, 4))
        tk.Label(box, text="Separate numbers with space, tab or comma.\n"
                           "One tracking point sits at the centre of each cell.",
                 font=("Arial", 8), fg="#666666", anchor="w", justify="left").grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(2, 0))
        tk.Label(box, textvariable=self._mesh_info_var, font=("Arial", 8), fg="#334477",
                 anchor="w", justify="left", wraplength=340).grid(
            row=5, column=0, columnspan=3, sticky="w", pady=(4, 0))

        box = tk.LabelFrame(f, text="Tracking settings", font=("Arial", 9), padx=6, pady=4)
        box.pack(fill="x", **pad)

        # Reference frame: the frame the mesh is drawn on and the templates
        # are cut from; tracking runs outward from it.
        self._grid_row(box, 0, "Reference frame", [(self._ref_var, "ref", 6)])
        tk.Checkbutton(box, text="Update template every frame (else: fixed reference frame)",
                       variable=self._update_template_var, font=("Arial", 8)).grid(
            row=1, column=0, columnspan=4, sticky="w", pady=2)

        # Template size row, with an "Auto (from mesh)" button that fills it
        # from the current cell size -- users can still edit it afterwards.
        tk.Label(box, text="Template size (w,h)", font=("Arial", 8), anchor="w").grid(
            row=2, column=0, sticky="w", padx=(0, 6), pady=2)
        tmpl_w_entry = tk.Entry(box, textvariable=self._tmpl_w_var, width=6, justify="center", font=("Arial", 9))
        tmpl_w_entry.grid(row=2, column=1, sticky="w", pady=2)
        tmpl_h_entry = tk.Entry(box, textvariable=self._tmpl_h_var, width=6, justify="center", font=("Arial", 9))
        tmpl_h_entry.grid(row=2, column=2, sticky="w", padx=(4, 0), pady=2)
        self._entries.setdefault("tmpl_w", []).append(tmpl_w_entry)
        self._entries.setdefault("tmpl_h", []).append(tmpl_h_entry)
        auto_btn = tk.Button(box, text="Auto (from mesh)", font=("Arial", 8),
                             command=self._on_auto_window_size)
        auto_btn.grid(row=2, column=3, sticky="w", padx=(8, 0), pady=2)

        self._grid_row(box, 3, "Search range (w,h)",
                       [(self._search_w_var, "search_w", 6), (self._search_h_var, "search_h", 6)])
        self._grid_row(box, 4, "Num levels", [(self._levels_var, "levels", 6)])
        tk.Label(box, text="Match method", font=("Arial", 8), anchor="w").grid(
            row=5, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Combobox(box, textvariable=self._method_var, values=list(MATCH_METHODS),
                     width=18, state="readonly").grid(row=5, column=1, columnspan=2, sticky="w", pady=2)
        tk.Checkbutton(box, text="Subpixel refinement", variable=self._subpixel_var,
                       font=("Arial", 8)).grid(row=6, column=0, columnspan=3, sticky="w", pady=2)
        self._grid_row(box, 7, "Min correlation (0-1)", [(self._mincorr_var, "min_corr", 6)])
        self._grid_row(box, 8, "Fine search radius (w,h)",
                       [(self._fine_w_var, "fine_w", 6), (self._fine_h_var, "fine_h", 6)])
        self._grid_row(box, 9, "Sidelobe exclusion radius", [(self._sidelobe_var, "sidelobe", 6)])
        self._grid_row(box, 10, "Start frame", [(self._start_var, "start", 6)])
        self._grid_row(box, 11, "End frame", [(self._end_var, "end", 6)])
        tk.Button(box, text="Reset tracking defaults", font=("Arial", 8),
                  command=self._reset_tracking_defaults).grid(
            row=12, column=0, columnspan=3, sticky="w", pady=(6, 0))
        tips = {
            0: "The frame the mesh is defined on: the corners and cells you set describe the "
               "points in THIS frame, and the templates are cut from it (1-based; default 1 = "
               "the first frame). Choose a frame where the target is clearly visible. Tracking "
               "starts here and runs forward to End and backward to Start. Changing it shows "
               "that frame in the preview.",
            1: "ON: each frame is matched against the patch from the previous frame tracked -- "
               "follows large deformation or lighting changes, but small errors add up (drift).\n"
               "OFF: always matched against the reference frame -- no drift; best for small "
               "deformations.",
            2: TEMPLATE_MATCH_TIPS["template"], 3: TEMPLATE_MATCH_TIPS["search"],
            4: TEMPLATE_MATCH_TIPS["levels"], 5: TEMPLATE_MATCH_TIPS["method"],
            6: TEMPLATE_MATCH_TIPS["subpixel"], 7: TEMPLATE_MATCH_TIPS["min_corr"],
            8: TEMPLATE_MATCH_TIPS["fine"], 9: TEMPLATE_MATCH_TIPS["sidelobe"],
        }
        for row, tip in tips.items():
            for widget in box.grid_slaves(row=row):
                if widget is not auto_btn:
                    add_tooltip(widget, tip)
        add_tooltip(auto_btn, "Set the template size to the mesh cell size, so neighbouring "
                              "templates just touch. This also happens automatically whenever "
                              "the cells or corners change; you can still edit it afterwards.")

        box = tk.LabelFrame(f, text="Run", font=("Arial", 9), padx=6, pady=4)
        box.pack(fill="x", **pad)
        row = tk.Frame(box)
        row.pack(fill="x")
        self._btn_run = tk.Button(row, text="Run", width=10, font=("Arial", 9, "bold"),
                                  bg="#33558a", fg="white", command=self._on_run)
        self._btn_run.pack(side="left")
        self._btn_stop = tk.Button(row, text="Stop", width=10, command=self._on_stop, state=tk.DISABLED)
        self._btn_stop.pack(side="left", padx=6)
        tk.Button(row, text="Help", command=self._open_help).pack(side="right")
        self._progress = ttk.Progressbar(box, mode="determinate", length=300)
        self._progress.pack(fill="x", pady=(6, 2))
        tk.Label(box, textvariable=self._progress_var, font=("Arial", 8), anchor="w",
                 justify="left", wraplength=340).pack(fill="x")
        self._summary_label = tk.Label(box, textvariable=self._summary_var, font=("Arial", 8, "bold"),
                                       anchor="w", justify="left", wraplength=340)
        self._summary_label.pack(fill="x", pady=(4, 0))
        tk.Label(box, text="Tracking details", font=("Arial", 8), anchor="w").pack(fill="x", pady=(4, 0))
        self._run_info_text = tk.Text(box, height=12, width=46, font=("Consolas", 8), wrap="word",
                                      bg="#f4f4f4", relief=tk.SUNKEN, borderwidth=1)
        self._run_info_text.pack(fill="x")
        self._set_run_info(self._run_info_last)

        box = tk.LabelFrame(f, text="Data", font=("Arial", 9), padx=6, pady=4)
        box.pack(fill="x", **pad)
        row = tk.Frame(box)
        row.pack(fill="x")
        self._btn_load = tk.Button(row, text="Load ...", width=10, command=self._on_load)
        self._btn_load.pack(side="left")
        self._btn_save_now = tk.Button(row, text="Save", width=10, command=self._on_save)
        self._btn_save_now.pack(side="left", padx=(6, 0))
        add_tooltip(self._btn_save_now, "Save settings and result to the file shown below, "
                                        "replacing it (Ctrl+S). The first time, asks for a "
                                        "file name like 'Save as ...'.")
        self._btn_save = tk.Button(row, text="Save as ...", width=10, command=self._on_save_as)
        self._btn_save.pack(side="left", padx=6)
        frow = tk.Frame(box)
        frow.pack(fill="x", pady=(4, 0))
        tk.Label(frow, text="File:", font=("Arial", 8)).pack(side="left")
        tk.Entry(frow, textvariable=self._result_file_var, state="readonly", font=("Arial", 8),
                 readonlybackground="#f4f4f4").pack(side="left", fill="x", expand=True, padx=(4, 0))
        self._result_file_note = tk.Label(box, textvariable=self._result_file_note_var, font=("Arial", 8),
                                          anchor="w", justify="left", wraplength=340, fg="#555555")
        self._result_file_note.pack(fill="x")
        self._result_label = tk.Label(box, textvariable=self._result_info_var, font=("Arial", 8),
                                      anchor="w", justify="left", wraplength=340)
        self._result_label.pack(fill="x", pady=(4, 0))

        box = tk.LabelFrame(f, text="Display", font=("Arial", 9), padx=6, pady=4)
        box.pack(fill="x", **pad)
        tk.Checkbutton(box, text="Show tracked positions at the displayed frame",
                       variable=self._show_tracked_var, font=("Arial", 8)).pack(anchor="w")

    def _build_right_panel(self, right: tk.Frame) -> None:
        rp = tk.PanedWindow(right, orient=tk.VERTICAL, sashwidth=6, sashrelief=tk.RAISED)
        rp.pack(fill="both", expand=True)
        top = tk.LabelFrame(rp, text="Image preview (drag to pan, wheel to zoom, click a cell to select its point)",
                            font=("Arial", 9))
        bottom = tk.LabelFrame(rp, text="Selected point: displacement relative to the reference frame",
                               font=("Arial", 9))
        copy_box = tk.LabelFrame(rp, text="Copy selected frames to a folder (e.g. to keep only the frames "
                                          "you need as a new image sequence)", font=("Arial", 9))
        rp.add(top, minsize=220, stretch="always")
        rp.add(bottom, height=280, minsize=160, stretch="never")
        rp.add(copy_box, height=150, minsize=120, stretch="never")

        # Packed (side="bottom") BEFORE the canvas so the canvas, which expands,
        # takes only the space left above the slider and these checkboxes.
        srow = tk.Frame(top)
        srow.pack(side="bottom", fill="x")
        show_row = tk.Frame(top)
        show_row.pack(side="bottom", fill="x", padx=6)
        tk.Checkbutton(show_row, text="Show/hide cells", variable=self._show_cells_var,
                       font=("Arial", 8)).pack(side="left")
        tk.Checkbutton(show_row, text="Show/hide cell centers", variable=self._show_centers_var,
                       font=("Arial", 8)).pack(side="left", padx=(10, 0))

        self._c_img = tk.Canvas(top, bg="#202020", highlightthickness=0)
        self._c_img.pack(fill="both", expand=True)
        self._c_img.bind("<Configure>", lambda _e: self._redraw_preview())
        self._c_img.bind("<ButtonPress-1>", self._on_img_press)
        self._c_img.bind("<B1-Motion>", self._on_img_drag)
        self._c_img.bind("<ButtonRelease-1>", self._on_img_release)
        self._c_img.bind("<MouseWheel>", self._on_img_wheel)
        self._c_img.bind("<Button-4>", self._on_img_wheel)
        self._c_img.bind("<Button-5>", self._on_img_wheel)
        self._c_img.bind("<Motion>", self._on_img_motion)
        self._c_img.bind("<Leave>", self._on_img_leave)
        self._c_img.bind("<ButtonPress-3>", self._on_img_cancel_pick)
        self._c_img.bind("<Escape>", self._on_img_cancel_pick)

        self._slider = tk.Scale(srow, from_=1, to=1, orient=tk.HORIZONTAL, resolution=1,
                                showvalue=False, variable=self._slider_var,
                                command=self._on_slider_moved, state=tk.DISABLED)
        self._slider.pack(fill="x", padx=6)
        self._slider.bind("<ButtonRelease-1>", self._on_slider_released)
        self._slider.bind("<KeyRelease>", self._on_slider_released)

        info_row = tk.Frame(srow)
        info_row.pack(fill="x", padx=6, pady=(0, 2))
        tk.Label(info_row, textvariable=self._frame_label_var, font=("Arial", 9), anchor="w").pack(side="left")
        tk.Button(info_row, text="Reset view", font=("Arial", 8),
                  command=self._on_reset_img_view).pack(side="right")

        self._build_series_plot(bottom)
        self._build_copy_panel(copy_box)

    # ══ Preview drawing / selection ══════════════════════════════════════
    def _redraw_preview(self) -> None:
        c = self._c_img
        if c is None:
            return
        try:
            cw, ch = c.winfo_width(), c.winfo_height()
        except tk.TclError:
            return
        if cw < 30 or ch < 30:
            return
        rgb, size = self._preview_rgb, self._image_size
        if rgb is None or size is None:
            c.delete("all")
            self._photo_key, self._view = None, None
            c.create_text(cw / 2, ch / 2, fill="#999999", justify="center", width=cw - 40,
                          text=self._preview_msg or
                          "No image available.\nConnect an image sequence to the 'files' input.")
            return

        W, H = size
        if self._img_last_size != (W, H):
            # A new image resolution makes any previous manual pan/zoom
            # meaningless (it was framed for a differently-sized image), so
            # fall back to auto-fit again.
            self._img_user_view = False
            self._img_last_size = (W, H)

        if self._img_user_view and self._img_view is not None:
            ox, oy, s = self._img_view
        else:
            margin = 6
            s = min((cw - 2 * margin) / W, (ch - 2 * margin) / H)
            dw0, dh0 = max(1, int(round(W * s))), max(1, int(round(H * s)))
            ox, oy = (cw - dw0) / 2.0, (ch - dh0) / 2.0
            self._img_view = (ox, oy, s)
        self._view = (ox, oy, s)

        # Render only the part of the source image that is actually visible
        # in the canvas, then resize just that crop -- rendering the WHOLE
        # image at an arbitrary zoom scale would blow up memory/time at high
        # zoom (e.g. a 5x zoom on a 2000 px preview would need a ~10000 px
        # PhotoImage). The view (ox, oy, s) is in FULL-resolution pixels:
        # full-res column u occupies screen [ox + u*s, ox + (u+1)*s),
        # consistent with _draw_overlay's X(u) = ox + (u+0.5)*s convention.
        # The cached preview `rgb` is downscaled (longest side <=
        # PREVIEW_MAX_SIDE), so the visible region is converted to the
        # PREVIEW's own pixel indices before cropping: preview column p
        # covers full-res columns [p/kx, (p+1)/kx). Cropping the small
        # preview with full-res indices (the old code) was clipped by
        # NumPy at the array edge and then stretched to the full-res size --
        # an amount that changed while panning, differently in x and y, so
        # panning looked like zooming with unequal x/y factors.
        ph_, pw_ = rgb.shape[:2]
        kx, ky = pw_ / W, ph_ / H                    # preview px per full-res px
        col0 = max(0, int(math.floor((0 - ox) / s * kx)))
        col1 = min(pw_, int(math.ceil((cw - ox) / s * kx)))
        row0 = max(0, int(math.floor((0 - oy) / s * ky)))
        row1 = min(ph_, int(math.ceil((ch - oy) / s * ky)))

        if col1 <= col0 or row1 <= row0:
            # Panned/zoomed entirely out of view: nothing of the image to show.
            c.delete("all")
            self._photo, self._photo_key = None, None
            self._draw_overlay(c, ox, oy, s)
            self._draw_pick_cursor(c, ox, oy, s, cw, ch)
            return

        # Screen position of the crop's edges, from the same full-res mapping.
        px, py = ox + col0 / kx * s, oy + row0 / ky * s
        dw = max(1, int(round(ox + col1 / kx * s - px)))
        dh = max(1, int(round(oy + row1 / ky * s - py)))
        key = (self._image_serial, col0, row0, col1, row1, dw, dh)

        if key != self._photo_key:
            crop = np.ascontiguousarray(rgb[row0:row1, col0:col1])
            interp = cv2.INTER_AREA if s < 1.0 else (cv2.INTER_NEAREST if s >= 3.0 else cv2.INTER_LINEAR)
            small = cv2.resize(crop, (dw, dh), interpolation=interp)
            self._photo = ImageTk.PhotoImage(Image.fromarray(small))
            self._photo_key = key
            c.delete("all")
            c.create_image(px, py, anchor="nw", image=self._photo, tags="img")
        else:
            c.coords("img", px, py)
            c.delete("overlay")

        self._draw_overlay(c, ox, oy, s)
        self._draw_pick_cursor(c, ox, oy, s, cw, ch)

    def _draw_pick_cursor(self, c, ox, oy, s, cw, ch) -> None:
        """Crosshair + hint while a corner is being picked (tagged 'overlay'
        so the next redraw removes it like the rest of the overlay)."""
        if self._pick is None:
            return
        name = "top-left" if self._pick == "tl" else "bottom-right"
        hint = f"Picking {name} corner: click to accept, Esc / right-click to cancel"
        c.create_text(9, ch - 7, anchor="sw", fill="#000000", text=hint, tags="overlay")
        c.create_text(8, ch - 8, anchor="sw", fill=PICK_COLOR, text=hint, tags="overlay")
        if self._pick_hover is None:
            return
        u, v = self._pick_hover
        x, y = ox + (u + 0.5) * s, oy + (v + 0.5) * s       # centre of the picked pixel
        for dash, color, width in (((), "#000000", 3), ((6, 4), PICK_COLOR, 1)):
            c.create_line(0, y, cw, y, fill=color, width=width, dash=dash, tags="overlay")
            c.create_line(x, 0, x, ch, fill=color, width=width, dash=dash, tags="overlay")
        if s >= 4:                                           # zoomed in: outline the whole pixel
            h = s / 2.0
            c.create_rectangle(x - h, y - h, x + h, y + h, outline=PICK_COLOR, width=2, tags="overlay")
        else:
            c.create_oval(x - 4, y - 4, x + 4, y + 4, outline=PICK_COLOR, width=2, tags="overlay")
        label = f"{'TL' if self._pick == 'tl' else 'BR'} ({u}, {v})"
        # Keep the label inside the canvas: flip to the other side near the edges.
        lx, anchor_x = (x + 10, "w") if x + 140 < cw else (x - 10, "e")
        ly, anchor_y = (y - 10, "s") if y - 30 > 0 else (y + 10, "n")
        anchor = {("s", "w"): "sw", ("s", "e"): "se", ("n", "w"): "nw", ("n", "e"): "ne"}[(anchor_y, anchor_x)]
        c.create_text(lx + 1, ly + 1, anchor=anchor, fill="#000000", text=label,
                      font=("Arial", 10, "bold"), tags="overlay")
        c.create_text(lx, ly, anchor=anchor, fill=PICK_COLOR, text=label,
                      font=("Arial", 10, "bold"), tags="overlay")

    def _draw_overlay(self, c, ox, oy, s) -> None:
        cfg, bad = self._collect()
        mesh = self._mesh_from_cfg(cfg, bad)
        if mesh is None:
            c.create_text(8, 8, anchor="nw", fill="#ff5252", tags="overlay",
                          text="Mesh settings are not valid yet.")
            return
        M, N, u0, v0, u1, v1 = mesh
        X = lambda u: ox + (u + 0.5) * s          # OpenCV pixel-centre convention
        Y = lambda v: oy + (v + 0.5) * s
        if self._show_cells_var.get():
            for i in range(M + 1):
                v = v0 + i * (v1 - v0) / M
                c.create_line(X(u0), Y(v), X(u1), Y(v), fill=GRID_COLOR, width=1, tags="overlay")
            for j in range(N + 1):
                u = u0 + j * (u1 - u0) / N
                c.create_line(X(u), Y(v0), X(u), Y(v1), fill=GRID_COLOR, width=1, tags="overlay")

        si, sj = min(self._sel[0], M - 1), min(self._sel[1], N - 1)
        self._sel = (si, sj)
        cu0, cu1 = u0 + sj * (u1 - u0) / N, u0 + (sj + 1) * (u1 - u0) / N
        cv0, cv1 = v0 + si * (v1 - v0) / M, v0 + (si + 1) * (v1 - v0) / M
        c.create_rectangle(X(cu0), Y(cv0), X(cu1), Y(cv1), outline=SELECT_COLOR, width=3, tags="overlay")

        if M * N > MAX_DRAWN_POINTS:
            return
        pos, tracked = mesh_centers(*mesh), False
        res = self._result
        if self._show_tracked_var.get() and self._result_matches(res, mesh):
            idx = self._view_frame - res["start"]
            if 0 <= idx < res["points"].shape[0] and self._filled(res)[idx]:
                pos, tracked = res["points"][idx], True
        outside = None if tracked else self._outside_mask(mesh, cfg)
        show_centers = self._show_centers_var.get()
        for i in range(M):
            for j in range(N):
                x, y = pos[i, j]
                if not (math.isfinite(x) and math.isfinite(y)):
                    continue
                selected = (i, j) == (si, sj)
                if not (selected or show_centers):
                    continue                  # the selected point stays visible
                color = SELECT_DOT_COLOR if selected else (
                    OUTSIDE_DOT_COLOR if outside is not None and outside[i, j] else DOT_COLOR)
                r = 2.5 if selected else 1.25
                px, py = X(x), Y(y)
                c.create_oval(px - r, py - r, px + r, py + r, fill=color, outline="", tags="overlay")

    def _select_cell_at(self, event) -> None:
        if self._view is None:
            return
        cfg, bad = self._collect()
        mesh = self._mesh_from_cfg(cfg, bad)
        if mesh is None:
            return
        ox, oy, s = self._view
        u, v = (event.x - ox) / s - 0.5, (event.y - oy) / s - 0.5
        M, N, u0, v0, u1, v1 = mesh
        j = int(math.floor((u - u0) / (u1 - u0) * N))
        i = int(math.floor((v - v0) / (v1 - v0) * M))
        if 0 <= i < M and 0 <= j < N and (i, j) != self._sel:
            self._sel = (i, j)
            self._series_user_view = False           # new point -> autoscale
            self._update_sel_info()
            self._redraw_preview()
            self._redraw_series()

    # -- Image preview pan (drag) and wheel zoom -----------------------------
    # A left-drag pans the image; a plain click (press+release with little
    # or no movement) selects the cell under the cursor instead -- see
    # _on_img_release. Panning uses screen-PIXEL deltas directly (event.x,
    # event.y and ox/oy are already in the same canvas-pixel space), unlike
    # the matplotlib-based plots elsewhere in this node: there is no data-
    # coordinate transform here to get invalidated between events, so this
    # does not need the "fix the scale at press time" workaround those use.

    def _on_img_press(self, event) -> None:
        if self._view is None:
            return
        self._img_pan.update(active=True, x_px=event.x, y_px=event.y, moved=False,
                             ox0=self._view[0], oy0=self._view[1])

    def _on_img_drag(self, event) -> None:
        pan = self._img_pan
        if not pan["active"] or self._view is None:
            return
        dx = event.x - pan["x_px"]
        dy = event.y - pan["y_px"]
        if not pan["moved"] and (abs(dx) > IMG_CLICK_DRAG_PX or abs(dy) > IMG_CLICK_DRAG_PX):
            pan["moved"] = True
        if not pan["moved"]:
            return
        _ox, _oy, s = self._view
        self._img_view = (pan["ox0"] + dx, pan["oy0"] + dy, s)
        self._img_user_view = True
        self._redraw_preview()

    def _on_img_release(self, event) -> None:
        pan = self._img_pan
        was_drag = pan["active"] and pan["moved"]
        pan["active"] = False
        if was_drag:
            self._update_pick(event)
        elif self._pick is not None:
            self._update_pick(event)
            self._end_pick(commit=True)
        else:
            self._select_cell_at(event)

    # -- Corner picking ("Pick TL" / "Pick BR") ------------------------------
    # While picking, the hovered pixel is written straight into the corner
    # field, so the field and the mesh overlay follow the mouse; a plain
    # click keeps it, Esc / right-click restores the value from before.

    def _start_pick(self, which: str) -> None:
        if self._pick == which:
            return
        if self._pick is not None:
            self._end_pick(commit=False)
        if self._image_size is None or self._c_img is None:
            messagebox.showwarning("Pick corner", "No image is loaded yet to pick from.",
                                   parent=self._dlg_parent())
            return
        var = self._tl_var if which == "tl" else self._lr_var
        self._pick, self._pick_orig, self._pick_hover = which, (var.get(), self._lr_auto), None
        if which == "lr":
            self._lr_auto = False      # keep a new preview image from overwriting the pick
        for btn in self._btn_pick.values():
            self._set_state(btn, tk.DISABLED)
        try:
            self._c_img.configure(cursor="crosshair")
            self._c_img.focus_set()    # receive <Escape>
        except tk.TclError:
            pass
        self._redraw_preview()

    def _end_pick(self, commit: bool) -> None:
        which = self._pick
        if which is None:
            return
        orig_text, orig_auto = self._pick_orig
        self._pick, self._pick_orig, self._pick_hover = None, None, None
        var = self._tl_var if which == "tl" else self._lr_var
        if not commit:
            self._set_var(var, orig_text)
            if which == "lr":
                self._lr_auto = orig_auto
        for btn in self._btn_pick.values():
            self._set_state(btn, tk.NORMAL)
        if self._c_img is not None:
            try:
                self._c_img.configure(cursor="")
            except tk.TclError:
                pass
        # Full pass (not just a redraw): an accepted corner is a mesh change,
        # so the template size follows it now (it was held while picking).
        self._validate_all()

    def _update_pick(self, event) -> None:
        """Map the cursor to the integer FULL-resolution pixel under it (the
        view (ox, oy, s) is in full-res pixels, see _redraw_preview) and
        write it into the corner field being picked."""
        if self._pick is None or self._view is None or self._image_size is None:
            return
        ox, oy, s = self._view
        W, H = self._image_size
        # Pixel u spans screen [ox + u*s, ox + (u+1)*s), so floor() gives the
        # pixel under the cursor; clamp to the image.
        u = min(max(int(math.floor((event.x - ox) / s)), 0), W - 1)
        v = min(max(int(math.floor((event.y - oy) / s)), 0), H - 1)
        if self._pick_hover == (u, v):
            return
        self._pick_hover = (u, v)
        var = self._tl_var if self._pick == "tl" else self._lr_var
        self._set_var(var, f"{u} {v}")   # -> _validate_all -> redraws mesh + crosshair

    def _on_img_cancel_pick(self, _event) -> str | None:
        if self._pick is None:
            return None
        self._end_pick(commit=False)
        return "break"                     # don't let Esc also reach window-level bindings

    def _on_img_motion(self, event) -> None:
        self._update_pick(event)

    def _on_img_leave(self, _event) -> None:
        if self._pick is not None and self._pick_hover is not None:
            self._pick_hover = None        # hide the crosshair; the field keeps the last value
            self._redraw_preview()

    def _on_img_wheel(self, event) -> str | None:
        """Zoom the image preview, anchored on the point under the cursor
        (that image point stays under the cursor across the zoom), same
        convention as image_data_simulator_node.py's 3D preview: scrolling
        "up" zooms in.
        """
        if self._preview_rgb is None or self._image_size is None or self._view is None:
            return None
        steps = self._wheel_steps(event)
        if steps == 0:
            return None
        ox, oy, s = self._view
        new_s = float(min(max(s * (IMG_ZOOM_FACTOR ** steps), IMG_MIN_SCALE), IMG_MAX_SCALE))
        applied = new_s / s
        ex, ey = float(event.x), float(event.y)
        new_ox = ex - (ex - ox) * applied
        new_oy = ey - (ey - oy) * applied
        self._img_view = (new_ox, new_oy, new_s)
        self._img_user_view = True
        self._redraw_preview()
        self._update_pick(event)                     # another pixel may now be under the cursor
        return "break"

    def _on_reset_img_view(self) -> None:
        self._img_user_view = False
        self._redraw_preview()

    # ══ Series plot ══════════════════════════════════════════════════════
    def _build_series_plot(self, parent: tk.Frame) -> None:
        tk.Label(parent, textvariable=self._sel_info_var, font=("Arial", 8),
                 anchor="w", justify="left").pack(fill="x", padx=2, pady=(2, 0))
        bar = tk.Frame(parent)
        bar.pack(fill="x")
        tk.Button(bar, text="Reset zoom", font=("Arial", 8), command=self._reset_series_view).pack(side="left")
        self._fig = Figure(figsize=(5, 2.6), dpi=100)
        self._fig.subplots_adjust(left=0.09, right=0.985, top=0.88, bottom=0.2)
        self._ax = self._fig.add_subplot(111)
        self._canvas_mpl = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas_mpl.get_tk_widget().pack(fill="both", expand=True)
        self._canvas_mpl.mpl_connect("scroll_event", self._on_series_scroll)
        self._canvas_mpl.mpl_connect("button_press_event", self._on_series_press)
        self._canvas_mpl.mpl_connect("motion_notify_event", self._on_series_drag)
        self._canvas_mpl.mpl_connect("button_release_event", self._on_series_release)

    def _reset_series_view(self) -> None:
        self._series_user_view = False
        self._redraw_series()

    def _on_series_scroll(self, event) -> None:
        ax = self._ax
        if ax is None or event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return
        factor = 0.9 if event.button == "up" else 1.1
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        key = getattr(event, "key", None)
        # Ctrl+wheel: horizontal zoom only; Shift+wheel: vertical only; else both.
        if key in ("control", "ctrl"):
            ax.set_xlim([event.xdata - (event.xdata - v) * factor for v in xlim])
        elif key == "shift":
            ax.set_ylim([event.ydata - (event.ydata - v) * factor for v in ylim])
        else:
            ax.set_xlim([event.xdata - (event.xdata - v) * factor for v in xlim])
            ax.set_ylim([event.ydata - (event.ydata - v) * factor for v in ylim])
        self._series_user_view = True
        self._canvas_mpl.draw_idle()

    def _on_series_press(self, event) -> None:
        if self._ax is None or event.inaxes is not self._ax or event.button != 1 or event.x is None:
            return
        xlim0, ylim0 = self._ax.get_xlim(), self._ax.get_ylim()
        bbox = self._ax.bbox
        if bbox.width <= 0 or bbox.height <= 0:
            return
        # Scale (data units per pixel) is captured ONCE here and reused for
        # the whole drag -- see _on_series_drag's docstring for why this
        # avoids a feedback loop that re-deriving it from event.xdata/ydata
        # on every move would create.
        self._pan.update(
            active=True, x_px=event.x, y_px=event.y, xlim0=xlim0, ylim0=ylim0,
            x_scale=(xlim0[1] - xlim0[0]) / bbox.width,
            y_scale=(ylim0[1] - ylim0[0]) / bbox.height,
        )

    def _on_series_drag(self, event) -> None:
        """Pan by PIXEL displacement, converted to data units with the scale
        fixed at button-press time -- NOT by re-deriving a delta from
        event.xdata/event.ydata on every move.

        event.xdata/event.ydata are computed by matplotlib using the axes'
        CURRENT transform, and this handler itself changes that transform on
        every call (via set_xlim/set_ylim). Comparing a value computed under
        the just-changed transform against a reference captured under the
        ORIGINAL (press-time) transform produces an alternating-sign error
        series: the applied shift both lags behind the mouse and visibly
        jitters while dragging, since each step over/under-corrects for the
        one before it. Pixel coordinates and the data-per-pixel scale stay
        constant throughout a pure pan (only zooming changes the scale), so
        converting displacement once with a scale fixed at press time avoids
        that feedback loop entirely.
        """
        if not self._pan["active"] or self._ax is None or event.x is None:
            return
        dx = (event.x - self._pan["x_px"]) * self._pan["x_scale"]
        dy = (event.y - self._pan["y_px"]) * self._pan["y_scale"]
        x0, x1 = self._pan["xlim0"]
        y0, y1 = self._pan["ylim0"]
        self._ax.set_xlim(x0 - dx, x1 - dx)
        self._ax.set_ylim(y0 - dy, y1 - dy)
        self._series_user_view = True
        self._canvas_mpl.draw_idle()

    def _on_series_release(self, _event) -> None:
        self._pan["active"] = False

    def _redraw_series(self) -> None:
        ax = self._ax
        if ax is None or self._canvas_mpl is None:
            return
        keep = self._series_user_view and bool(ax.lines)
        xlim, ylim = (ax.get_xlim(), ax.get_ylim()) if keep else (None, None)
        ax.clear()
        cfg, bad = self._collect()
        mesh = self._mesh_from_cfg(cfg, bad)
        res = self._result
        message = "No tracking result yet.\nPress Run, or load a result file."
        if res is not None and res["n_done"] > 0:
            if not self._result_matches(res, mesh):
                message = ("The mesh settings differ from the tracking result.\n"
                           "Run again, or restore the mesh settings used for the result.")
            else:
                S, M, N = res["points"].shape[:3]
                i, j = min(self._sel[0], M - 1), min(self._sel[1], N - 1)
                n = S
                frames = res["start"] + np.arange(S)
                # Frames not tracked yet are NaN, so lines simply stop there.
                xy = np.where(self._filled(res)[:, None], res["points"][:, i, j, :], np.nan)
                # Displacement relative to the reference frame (its position is
                # the mesh point); a reference outside the tracked range uses
                # the mesh point too. Older results without one: first valid.
                mesh = res["mesh"]
                ref_uv = mesh_centers(*mesh)[i, j] if res.get("ref") is not None else None
                for k, (label, color) in enumerate((("u", "#1f77b4"), ("v", "#d62728"))):
                    series = xy[:, k].copy()
                    finite = np.isfinite(series)
                    if ref_uv is not None:
                        series = series - ref_uv[k]
                    elif finite.any():
                        series = series - series[int(np.argmax(finite))]
                    ax.plot(frames, series, color=color, linewidth=1.3, label=label)
                ax.legend(loc="upper right", fontsize=8)
                if frames[0] <= self._view_frame <= frames[-1]:
                    ax.axvline(self._view_frame, color="#888888", linestyle="--", linewidth=1)
                valid = int(np.isfinite(xy[:, 0]).sum())
                ref_txt = f"; reference frame {res['ref']}" if res.get("ref") is not None else ""
                ax.set_title(f"cell (row {i + 1}, col {j + 1}); valid {valid}/{n} frames{ref_txt}", fontsize=8)
                # Frame numbers are 1-based. Without this, matplotlib's 5% x
                # margin pads the axis to below frame 1 and labels a "0" tick
                # (and fractional ticks for short runs), which reads as 0-based.
                ax.set_xmargin(0)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                if n == 1:
                    ax.set_xlim(frames[0] - 0.5, frames[0] + 0.5)
                ax.set_xlabel("frame (1-based)", fontsize=8)
                ax.set_ylabel("displacement (px)", fontsize=8)
                ax.tick_params(labelsize=8)
                ax.grid(True, alpha=0.3)
                message = None
        if message is not None:
            # Leave the axes' default (0..1) autoscale range and ticks in
            # place here, matching ifft_1d_node.py and the other plot-based
            # nodes: this is what lets a user test panning/zooming on the
            # plot before any tracking result exists, instead of an axis
            # with no ticks at all to grab onto.
            ax.set_xlabel("frame", fontsize=8)
            ax.set_ylabel("displacement (px)", fontsize=8)
            ax.text(0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center",
                    color="#777777", fontsize=9)
        self._draw_copy_selection(ax, message is not None)
        if keep:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
        self._canvas_mpl.draw_idle()

    _COPY_SHADE = "#5cb85c"            # soft green: easy to see, does not hide the u / v lines

    def _draw_copy_selection(self, ax, no_result: bool) -> None:
        """Shade the frames chosen in 'Copy selected frames' (when the text is
        valid): one artist for all ranges, behind the curves."""
        try:
            n = self._n_frames()
            sel, _dropped = frame_copy.parse_frames(self._copy_frames_var.get(), n)
        except (ValueError, AttributeError, tk.TclError):
            return
        if not sel:
            return
        spans = [(a + 0.5, b - a + 1) for a, b in frame_copy.runs(sel)]       # 1-based a+1 -> a+1-0.5
        ax.broken_barh(spans, (0, 1), transform=ax.get_xaxis_transform(), facecolors=self._COPY_SHADE,
                       alpha=0.22, linewidth=0, zorder=0, label=f"to copy ({len(sel)} frames)")
        if no_result and n > 0:
            ax.set_xlim(0.5, n + 0.5)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlabel("frame (1-based)", fontsize=8)
        ax.legend(loc="upper right", fontsize=8)

    # ══ Copy selected frames to a folder ═══════════════════════════════════
    _COPY_NUMBERING = ("original frame number", "1, 2, 3 ... (in order)")
    _COPY_FORMATS = ("png", "jpg", "same")

    def _build_copy_panel(self, f: tk.Frame) -> None:
        small = ("Arial", 8)
        row = tk.Frame(f)
        row.pack(fill="x", padx=4, pady=(2, 0))
        b = tk.Button(row, text="Select folder...", font=small, command=self._on_copy_select_folder)
        b.pack(side="left")
        add_tooltip(b, "Choose the folder the copied frames are written to (an empty folder is best).")
        lbl = tk.Label(row, textvariable=self._copy_dir_var, font=small, fg="#334455", anchor="w")
        lbl.pack(side="left", fill="x", expand=True, padx=6)
        add_tooltip(lbl, "Destination folder of the copied frames.")

        row = tk.Frame(f)
        row.pack(fill="x", padx=4, pady=(3, 0))
        tk.Label(row, text="Frames (1-based):", font=small).pack(side="left")
        self._copy_sel_entry = e = tk.Entry(row, textvariable=self._copy_frames_var, font=small, width=40)
        e.pack(side="left", fill="x", expand=True, padx=(4, 0))
        add_tooltip(e, "Frames to copy, as in the Image Sequence selector: '1:100 201:500 600:900' "
                       "(inclusive ranges), '1:1000:10' (every 10th), '5' (one frame), '700:' (to the last). "
                       "Numbers may be expressions with + - * ( ), e.g. (31-30):(80+50-30); ':' binds "
                       "weakest, so 1:2+3 is 1:5. Valid frames are shaded green on the plot above.")
        self._copy_info_label = tk.Label(f, textvariable=self._copy_info_var, font=small, anchor="w",
                                         justify="left")
        self._copy_info_label.pack(fill="x", padx=4)

        row = tk.Frame(f)
        row.pack(fill="x", padx=4, pady=(2, 0))
        tk.Label(row, text="Names:", font=small).pack(side="left")
        rb = tk.Radiobutton(row, text="original", value="original", variable=self._copy_mode_var, font=small)
        rb.pack(side="left")
        add_tooltip(rb, "Image files keep their names. Video frames become <video name>_FRM000123 "
                        "(the 1-based frame number).")
        rb = tk.Radiobutton(row, text="new:", value="pattern", variable=self._copy_mode_var, font=small)
        rb.pack(side="left", padx=(6, 0))
        add_tooltip(rb, "New names from the pattern on the right.")
        self._copy_pattern_entry = e = tk.Entry(row, textvariable=self._copy_pattern_var, font=small, width=16)
        e.pack(side="left")
        add_tooltip(e, "File name pattern WITHOUT extension, with one number field: selected_%06d -> "
                       "selected_000001. The extension comes from Format.")
        self._copy_number_cb = cb = ttk.Combobox(row, textvariable=self._copy_number_var, state="readonly",
                                                 values=self._COPY_NUMBERING, width=21, font=small)
        cb.pack(side="left", padx=(4, 0))
        add_tooltip(cb, "Number in the new names: the ORIGINAL frame number (you can trace each file back), or "
                        "1, 2, 3 ... in the order of the selection (a gap-free sequence).")
        tk.Label(row, text="  Format:", font=small).pack(side="left")
        cb = ttk.Combobox(row, textvariable=self._copy_format_var, state="readonly", values=self._COPY_FORMATS,
                          width=5, font=small)
        cb.pack(side="left")
        add_tooltip(cb, "png: lossless (recommended for measurements). jpg: smaller files. same: image files are "
                        "copied byte for byte (exact, fastest; not for a video).")
        tk.Label(row, text=" JPEG quality", font=small).pack(side="left")
        self._copy_quality_entry = e = tk.Entry(row, textvariable=self._copy_quality_var, font=small, width=4)
        e.pack(side="left")
        add_tooltip(e, "1-100 (95 = very good). Only used for jpg.")

        row = tk.Frame(f)
        row.pack(fill="x", padx=4, pady=(4, 2))
        self._copy_btn = tk.Button(row, text="Copy selected frames", font=("Arial", 9, "bold"), bg="#d7ecd9",
                                   state="disabled", command=self._on_copy_start)
        self._copy_btn.pack(side="left")
        add_tooltip(self._copy_btn, "Write the chosen frames to the folder (in the background). Enabled when "
                                    "the folder, the frames and the names are valid.")
        self._copy_cancel_btn = tk.Button(row, text="Cancel", font=small, state="disabled",
                                          command=self._on_copy_cancel)
        self._copy_cancel_btn.pack(side="left", padx=(3, 0))
        add_tooltip(self._copy_cancel_btn, "Stop copying; the files written so far stay.")
        ttk.Progressbar(row, variable=self._copy_progress_var, maximum=100.0, length=140).pack(side="left", padx=6)
        tk.Label(row, textvariable=self._copy_status_var, font=small, fg="#334455", anchor="w").pack(
            side="left", fill="x", expand=True)
        self._copy_node_btn = tk.Button(row, text="Create Image Sequence node", font=small, state="disabled",
                                        command=self._on_copy_create_node)
        self._copy_node_btn.pack(side="right")
        add_tooltip(self._copy_node_btn, "Add an Image Sequence node with the copied files to the canvas, next to "
                                         "this node (enabled after a copy).")
        self._copy_check()
        if self._copy_job is not None:
            self._set_copy_running(True)
            self._poll_copy()

    def _schedule_copy_check(self) -> None:
        if self._copy_check_id is not None:
            return
        try:
            self._copy_check_id = self.canvas.after(200, self._copy_check)
        except tk.TclError:
            self._copy_check_id = None

    def _copy_plan(self):
        """(targets, kind, paths, info) for the current fields; ValueError
        with a message for the user when something is missing / invalid."""
        src = self._source()
        if src is None or self._n_frames() <= 0:
            raise ValueError("no frames: connect an image sequence or select a video / images above")
        n = self._n_frames()
        sel, dropped = frame_copy.parse_frames(self._copy_frames_var.get(), n)
        if not sel:
            raise ValueError("type the frames to copy, e.g. 1:100 201:500")
        dest = self._copy_dir_var.get().strip()
        if not dest:
            raise ValueError(f"{len(sel)} frames chosen -- select the folder to copy them to")
        if not os.path.isdir(dest):
            raise ValueError(f"the folder does not exist: {dest}")
        if self._copy_format_var.get() == "jpg":
            try:
                q = int(self._copy_quality_var.get())
                if not 1 <= q <= 100:
                    raise ValueError
            except ValueError:
                raise ValueError("JPEG quality must be a whole number 1-100") from None
        naming = {"mode": self._copy_mode_var.get(), "pattern": self._copy_pattern_var.get(),
                  "number": "frame" if self._copy_number_var.get() == self._COPY_NUMBERING[0] else "sequence",
                  "format": self._copy_format_var.get()}
        targets = frame_copy.plan_targets(src.kind, src.paths, sel, dest, naming)
        info = f"{len(sel)} frames chosen (of {n}) in {len(frame_copy.runs(sel))} range(s)"
        if dropped:
            info += f"; {dropped} numbers beyond the last frame ({n}) are ignored"
        first = os.path.basename(targets[0][2])
        info += f"  ->  {first}" + (f" ... {os.path.basename(targets[-1][2])}" if len(targets) > 1 else "")
        return targets, src.kind, list(src.paths), info

    def _copy_check(self) -> None:
        """Validate the fields: info line, Copy button, plot shading."""
        self._copy_check_id = None
        if not hasattr(self, "_copy_frames_var"):
            return
        color, ok = "#2f6b2f", False
        try:
            _t, _k, _p, info = self._copy_plan()
            ok = True
        except ValueError as e:
            info, color = str(e), "#b36b00"
            try:
                frame_copy.parse_frames(self._copy_frames_var.get(), self._n_frames())
            except ValueError as pe:
                info, color = f"Invalid frames: {pe}", "#c62828"
        self._copy_info_var.set(info)
        pattern = self._copy_mode_var.get() == "pattern"
        try:
            if self._copy_info_label is not None:
                self._copy_info_label.configure(fg=color)
            if self._copy_btn is not None:
                self._copy_btn.configure(state="normal" if ok and self._copy_job is None else "disabled")
            if self._copy_pattern_entry is not None:
                self._copy_pattern_entry.configure(state="normal" if pattern else "disabled")
            if self._copy_number_cb is not None:
                self._copy_number_cb.configure(state="readonly" if pattern else "disabled")
            if self._copy_quality_entry is not None:
                self._copy_quality_entry.configure(state="normal" if self._copy_format_var.get() == "jpg"
                                                   else "disabled")
        except tk.TclError:
            pass
        self._redraw_series()

    def _on_copy_select_folder(self) -> None:
        folder = self.ask_directory("save", parent=self._dlg_parent(), title="Folder for the copied frames")
        if folder:
            self._copy_dir_var.set(os.path.abspath(folder))

    def _set_copy_running(self, running: bool) -> None:
        try:
            if self._copy_btn is not None:
                self._copy_btn.configure(state="disabled" if running else "normal",
                                         text="Copying..." if running else "Copy selected frames")
            if self._copy_cancel_btn is not None:
                self._copy_cancel_btn.configure(state="normal" if running else "disabled")
            if self._copy_node_btn is not None:
                self._copy_node_btn.configure(state="normal" if (self._copy_done and not running) else "disabled")
        except tk.TclError:
            pass
        if not running:
            self._copy_check()

    def _on_copy_start(self) -> None:
        if self._copy_job is not None:
            return
        try:
            targets, kind, paths, _info = self._copy_plan()
        except ValueError as e:
            messagebox.showerror("Copy selected frames", str(e), parent=self._dlg_parent())
            return
        existing = sum(1 for _k, _s, dst, _e in targets if os.path.exists(dst))
        skip = False
        if existing:
            answer = messagebox.askyesnocancel(
                "Copy selected frames",
                f"{existing} of the {len(targets)} files already exist in the folder.\n\n"
                f"Yes = overwrite them\nNo = keep them (skip)\nCancel = do not copy",
                parent=self._dlg_parent())
            if answer is None:
                return
            skip = not answer
        quality = int(self._copy_quality_var.get()) if self._copy_quality_var.get().strip().isdigit() else 95
        state = frame_copy.new_copy_state(len(targets))
        self._copy_job = {"state": state, "dest": self._copy_dir_var.get().strip()}
        self._copy_done = None
        threading.Thread(target=frame_copy.copy_worker, args=(kind, paths, targets, state, quality, skip),
                         daemon=True, name="mtt-copy-frames").start()
        self._set_copy_running(True)
        self._poll_copy()

    def _on_copy_cancel(self) -> None:
        if self._copy_job is not None:
            self._copy_job["state"]["cancel"].set()

    def _poll_copy(self) -> None:
        job = self._copy_job
        if job is None or self._destroyed:
            return
        st = job["state"]
        done, total = st["done"], max(1, st["total"])
        elapsed = time.monotonic() - st["t0"]
        try:
            self._copy_progress_var.set(100.0 * done / total)
        except tk.TclError:
            pass
        if not st["finished"]:
            left = ""
            if done:
                t = elapsed * (total - done) / done
                left = f", about {int(t // 60)}:{int(t % 60):02d} left"
            self._copy_status_var.set(f"{done} / {total} frames{left}")
            try:
                self.canvas.after(200, self._poll_copy)
            except tk.TclError:
                pass
            return
        self._copy_job = None
        written = st["written"]
        text = (f"{'Cancelled' if st['cancel'].is_set() else 'Done'}: {len(written)} files in "
                f"{int(elapsed // 60)}:{int(elapsed % 60):02d}")
        if st["errors"]:
            text += f"; {len(st['errors'])} frame(s) failed ({st['errors'][0]})"
        if st["error"]:
            text += f"; stopped: {st['error']}"
        self._copy_status_var.set(text)
        self._copy_done = {"files": list(written), "dest": job["dest"]} if written else None
        self._set_copy_running(False)
        if st["errors"] or st["error"]:
            messagebox.showwarning("Copy selected frames", text + ("\n\n" + "\n".join(st["errors"][:10])
                                                                  if st["errors"] else ""),
                                   parent=self._dlg_parent())

    def _on_copy_create_node(self) -> None:
        done = self._copy_done
        if not done:
            return
        files = done["files"]
        params = {"source": "images", "wait_for_files": False, "method": "folder",
                  "folder": done["dest"], "file_list": "\n".join(files)}
        src = self._source()
        what = os.path.basename(src.paths[0]) if src is not None and src.kind == "video" else (
            self.node_name if getattr(self, "node_name", "") else self.DISPLAY_NAME)
        name = f"{len(files)} frames of {what}"
        node = self.spawn_node_near("image_sequence", params, name)
        if node is None:
            messagebox.showinfo("Create Image Sequence node",
                                "The node could not be created here. Make an Image Sequence node yourself and "
                                f"select the folder:\n{done['dest']}", parent=self._dlg_parent())
            return
        self._copy_status_var.set(f"Created the Image Sequence node '{name}' next to this node.")

    # ══ Run / stop ═══════════════════════════════════════════════════════
    def _dlg_parent(self):
        return self._inspector_win if self.is_inspector_open() else None

    def _reset_tracking_defaults(self) -> None:
        d = self._defaults()
        for attr, key in (("_tmpl_w_var", "tmpl_w"), ("_tmpl_h_var", "tmpl_h"),
                          ("_search_w_var", "search_w"), ("_search_h_var", "search_h"),
                          ("_levels_var", "levels"), ("_method_var", "method"),
                          ("_mincorr_var", "min_corr"), ("_fine_w_var", "fine_w"),
                          ("_fine_h_var", "fine_h"), ("_sidelobe_var", "sidelobe")):
            self._set_var(getattr(self, attr), d[key])
        self._set_var(self._subpixel_var, True)
        self._set_var(self._update_template_var, False)

    def _on_fit_mesh_to_image(self) -> None:
        """Set the mesh to span the whole image: top-left (0, 0), bottom-right
        (W-1, H-1), where (W, H) is the size of the currently loaded preview
        image. Also re-enables bottom-right's "auto-fit to image size" mode,
        so it keeps tracking the image size afterwards for as long as it is
        not edited by hand -- the same behaviour a fresh node starts with.
        """
        size = self._image_size
        if size is None:
            messagebox.showwarning(
                "Fit Mesh to Full Image",
                "No image is loaded yet to read the image size from.",
                parent=self._dlg_parent(),
            )
            return
        W, H = size
        self._set_var(self._tl_var, "0 0")
        self._lr_auto = True
        self._set_var(self._lr_var, f"{W - 1} {H - 1}")

    def _on_auto_window_size(self) -> None:
        """Set the template size to the mesh cell size (rounded to the
        nearest integer, minimum 3), computed from the current cells (M, N)
        and corners. The fields remain freely editable afterwards.
        """
        cfg, bad = self._collect()
        mesh = self._mesh_from_cfg(cfg, bad)
        if mesh is None:
            messagebox.showwarning(
                "Auto window size",
                "Cells and corners must be valid before the window size can be computed.",
                parent=self._dlg_parent(),
            )
            return
        tmpl_w, tmpl_h = self._auto_template_size(mesh)
        self._set_var(self._tmpl_w_var, str(tmpl_w))
        self._set_var(self._tmpl_h_var, str(tmpl_h))

    @staticmethod
    def _auto_template_size(mesh) -> tuple[int, int]:
        M, N, u0, v0, u1, v1 = mesh
        return max(3, int(round((u1 - u0) / N))), max(3, int(round((v1 - v0) / M)))

    def _follow_mesh_with_template(self) -> None:
        """When the mesh (cells, corners) changes to a VALID new value, set
        the template size to the cell size -- as if "Auto (from mesh)" were
        pressed. Settings just loaded (project / result file) are adopted
        as they are: their mesh becomes the baseline without touching the
        saved template size. Skipped while a corner is being picked (the
        mesh follows the mouse); the picked corner counts on accept."""
        if self._pick is not None:
            return
        cfg, bad = self._collect()
        mesh = self._mesh_from_cfg(cfg, bad)
        if mesh is None or mesh == self._tmpl_mesh_seen:
            return
        adopt_only = self._tmpl_mesh_seen is _LOADED
        self._tmpl_mesh_seen = mesh
        if adopt_only:
            return
        tmpl_w, tmpl_h = self._auto_template_size(mesh)
        if (self._tmpl_w_var.get().strip(), self._tmpl_h_var.get().strip()) == (str(tmpl_w), str(tmpl_h)):
            return
        # Both writes, then one validation pass (by the caller).
        self._suspend_validate = True
        try:
            self._set_var(self._tmpl_w_var, str(tmpl_w))
            self._set_var(self._tmpl_h_var, str(tmpl_h))
        finally:
            self._suspend_validate = False

    def _on_run(self) -> None:
        if self._running:
            return
        cfg, bad = self._collect()
        problem = self._run_blocker(cfg, bad)
        if problem:
            messagebox.showwarning("Run", problem, parent=self._dlg_parent())
            return
        start, end, ref = cfg["start"], cfg["end"], cfg["ref"]
        M, N = cfg["cells"]
        (u0, v0), (u1, v1) = cfg["tl"], cfg["lr"]
        S = end - start + 1
        if S * M * N > MAX_RESULT_ENTRIES:
            messagebox.showerror("Run", f"Too many results ({S} frames x {M * N} points). "
                                        "Reduce the number of frames or cells.", parent=self._dlg_parent())
            return
        if self._result is not None and not self._result_saved and not messagebox.askokcancel(
                "Run", "The current tracking result has not been saved to a file.\n"
                       "Running again will replace it. Continue?", parent=self._dlg_parent()):
            return

        mesh_pts = mesh_centers(M, N, u0, v0, u1, v1)
        result = {
            "points": np.full((S, M, N, 2), np.nan), "status": np.zeros((S, M, N), np.uint8),
            "confidence": np.full((S, M, N), np.nan), "start": start, "end": end, "n_done": 0,
            "filled": np.zeros(S, dtype=bool), "ref": ref,
            "mesh": (M, N, u0, v0, u1, v1), "bad_frames": 0, "n_files": self._n_frames(),
            "settings": self._settings_dict(),        # snapshot: saved together with the result
        }
        job = {"source": self._source(), "seq": list(range(start - 1, end)), "mesh_pts": mesh_pts,
               "ref_seq": ref - 1,
               "ref_index": ref - start if start <= ref <= end else None,
               "ref_before": ref < start,
               "update_template": cfg["update_template"],
               "cfg": {"template_size": cfg["template"], "search_range": cfg["search"],
                       "num_levels": cfg["levels"], "match_method": MATCH_METHODS[cfg["method"]],
                       "subpixel": cfg["subpixel"], "min_correlation": cfg["min_corr"],
                       "fine_search_radius": cfg["fine"], "sidelobe_exclusion_radius": cfg["sidelobe"]}}
        state = {"cancel": threading.Event(), "done": 0, "total": S, "finished": False,
                 "error": None, "bad_frames": 0, "t0": time.time(),
                 "info": {"phase": "starting", "tune": {}}}     # filled by the worker, shown in the Run panel
        self._source().stop_background()             # tracking gets the CPU / disk
        self._result, self._run_state, self._running = result, state, True
        # Keep self._result_path: it stays the node's result-file property (the
        # default for "Save as"); the new result is marked as not saved in it.
        self._result_saved, self._result_error = True, ""
        self._last_plot_done = 0
        self._view_filled_drawn = False
        self._series_user_view = False
        self._set_running_ui(True)
        self._update_result_info()
        self._update_status_item()
        self._redraw_series()
        threading.Thread(target=_tracking_worker, args=(job, result, state),
                         daemon=True, name="mesh-track").start()
        self._poll_id = self.canvas.after(POLL_MS, self._poll_run)

    def _on_stop(self) -> None:
        st = self._run_state
        if st is not None and not st["finished"]:
            st["cancel"].set()
            self._progress_var.set("Stopping after the current frame ...")

    def _set_running_ui(self, running: bool) -> None:
        self._set_state(self._btn_stop, tk.NORMAL if running else tk.DISABLED)
        self._set_state(self._btn_load, tk.DISABLED if running else tk.NORMAL)
        self._set_state(self._btn_save, tk.DISABLED if running else tk.NORMAL)
        self._set_state(self._btn_save_now, tk.DISABLED if running else tk.NORMAL)
        self._update_source_info()                  # no changing the input while tracking
        if running:
            self._set_state(self._btn_run, tk.DISABLED)
        else:
            cfg, bad = self._collect()
            self._set_state(self._btn_run, tk.NORMAL if self._run_blocker(cfg, bad) is None else tk.DISABLED)

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        seconds = max(0, int(seconds))
        return f"{seconds // 60:d}:{seconds % 60:02d}"

    def _update_progress_ui(self) -> None:
        st = self._run_state
        if self._progress is None or st is None:
            return
        try:
            total, done = max(1, st["total"]), st["done"]
            self._progress.configure(maximum=total, value=done)
            elapsed = time.time() - st["t0"]
            eta = "estimating ..." if done <= 1 else self._fmt_time(elapsed * (total - done) / (done - 1))
            phase = str((st.get("info") or {}).get("phase", ""))
            if phase.startswith("reading the video"):    # before the first tracked frame
                self._progress_var.set(f"{phase[0].upper()}{phase[1:]}   "
                                       f"elapsed {self._fmt_time(elapsed)}")
                return self._set_run_info(self._run_info_lines(st))
            self._progress_var.set(f"Frame {done}/{total} ({100 * done // total}%)   "
                                   f"elapsed {self._fmt_time(elapsed)}   ETA {eta}")
        except tk.TclError:
            pass
        self._set_run_info(self._run_info_lines(st))

    def _run_info_lines(self, st: dict) -> str:
        """Text for the read-only 'Tracking details' box."""
        info = st.get("info") or {}
        lines = [f"Status     : {info.get('phase', '-')}"]
        if "points" in info:
            n = info.get("processes", 1)
            lines.append(f"Processes  : {n} (of {info.get('cpus', '?')} CPUs)"
                         + ("  -- trying counts" if info.get("phase") == "auto-tuning" else ""))
            lines.append(f"Points     : {info['points']} per frame, ~{-(-info['points'] // max(1, n))} per process")
            lines.append(f"Image      : {info.get('image', '?')} px")
            tune = info.get("tune") or {}
            if tune:
                best = min(tune, key=tune.get)
                lines.append("Auto-tune  : " + ", ".join(
                    f"{k}p {v * 1000:.0f} ms" + ("*" if k == best and info.get("phase") != "auto-tuning" else "")
                    for k, v in sorted(tune.items())))
            if "avg_total_ms" in info:
                # total = wall-clock time per frame step. imread runs in the
                # read-ahead thread while the previous frame is being matched,
                # so total can be less than imread + tm.
                fps = 1000.0 / info["avg_total_ms"] if info["avg_total_ms"] > 0 else 0.0
                lines.append(f"Per frame  : {info['avg_total_ms']:.0f} ms matching "
                             f"({info['avg_read_ms']:.0f} ms imread, {info['avg_tm_ms']:.0f} ms tm) "
                             f"(last {info['last_total_ms']:.0f} ms), {fps:.1f} frames/s")
        lines.append(f"Frames     : {st['done']}/{st['total']}"
                     + (f", {st['bad_frames']} unreadable" if st.get("bad_frames") else ""))
        return "\n".join(lines)

    def _set_run_info(self, text: str) -> None:
        self._run_info_last = text
        w = self._run_info_text
        if w is None:
            return
        try:
            w.configure(state=tk.NORMAL)
            w.delete("1.0", tk.END)
            w.insert("1.0", text)
            w.configure(state=tk.DISABLED)            # read-only (still selectable / copyable)
        except tk.TclError:
            pass

    def _poll_run(self) -> None:
        """Tk-thread poller: progress, live plot, completion."""
        self._poll_id = None
        st = self._run_state
        if st is None or self._destroyed:
            return
        res = self._result
        done = st["done"]
        if res is not None:
            res["n_done"] = done
        self._update_progress_ui()
        self._update_status_item()
        if done != self._last_plot_done:
            self._last_plot_done = done
            self._redraw_series()
            idx = self._view_frame - res["start"] if res is not None else -1
            if res is not None and 0 <= idx < res["points"].shape[0]:
                filled_now = bool(self._filled(res)[idx])
                if filled_now != self._view_filled_drawn:
                    self._view_filled_drawn = filled_now
                    self._redraw_preview()           # displayed frame just became available
        if st["finished"]:
            self._finish_run(st)
            return
        try:
            self._poll_id = self.canvas.after(POLL_MS, self._poll_run)
        except tk.TclError:
            pass

    def _finish_run(self, st: dict) -> None:
        res = self._result
        self._running, self._run_state = False, None
        if res is not None:
            res["n_done"], res["bad_frames"] = st["done"], st["bad_frames"]
        if st["done"] == 0:                           # no frame at all: nothing to keep
            self._result = None
        self._result_saved = self._result is None
        if st["error"]:
            msg = "Failed: " + st["error"]
        elif st["cancel"].is_set():
            msg = f"Stopped by user after {st['done']}/{st['total']} frames."
        else:
            msg = f"Finished: {st['done']} frames."
        if st["bad_frames"]:
            msg += f" {st['bad_frames']} frame(s) could not be read."
        self._progress_var.set(msg)
        self._set_run_info(self._run_info_lines(st))
        self._set_running_ui(False)
        self._update_result_info()
        self._update_status_item()
        self._redraw_series()
        self._redraw_preview()
        self._notify_result_changed()
        if st["error"]:
            messagebox.showerror("Mesh Template Track", st["error"], parent=self._dlg_parent())

    # ══ Load / save ══════════════════════════════════════════════════════
    def _update_result_file_field(self) -> None:
        """Show the remembered result file: full path in the read-only field,
        and below it how the project file stores it and whether the current
        result is in it."""
        path = self._result_path
        self._result_file_var.set(path or "")
        if not path:
            note = "No result file yet (use Save or Save as ... to choose one)."
        else:
            base = self._get_project_base()
            if base is not None:
                note = "Stored in the project as: " + self._to_relative(path, base)
            else:
                note = "Project not saved yet: the path is made relative when it is."
            if not os.path.isfile(path):
                note += "\nFile not found."
            elif self._result is not None and not self._result_saved and not self._running:
                note += "\nThe current result is NOT saved in this file yet (use Save, Ctrl+S)."
        self._result_file_note_var.set(note)

    def _file_dialog_start(self) -> dict:
        """initialdir / initialfile for Load and Save as: the remembered file."""
        path = self._result_path
        if path:
            return {"initialdir": os.path.dirname(path) or None, "initialfile": os.path.basename(path)}
        base = self._get_project_base()
        return {"initialdir": str(base) if base is not None else None, "initialfile": "mesh_tracking.npz"}

    def _update_result_info(self) -> None:
        res = self._result
        if res is None:
            text = "No tracking result."
            if self._result_error:
                text += "\n" + self._result_error
            color = INVALID_FG if self._result_error else "#555555"
        else:
            S, M, N = res["points"].shape[:3]
            text = (f"Result: {res['n_done']}/{S} frames (frames {res['start']}-{res['end']}), "
                    f"{M} x {N} points")
            if res["n_done"] < S:
                text += " (partial)"
            if res.get("bad_frames"):
                text += f", {res['bad_frames']} unreadable frame(s)"
            if self._running:
                text += "\nTracking in progress ..."
            elif self._result_path and self._result_saved:
                text += "\nSaved in the file above."
            else:
                text += "\nNOT saved to a file yet."
            n_now = self._n_frames()
            if n_now and res.get("n_files") and res["n_files"] != n_now:
                text += (f"\nNote: the result was computed for a sequence of {res['n_files']} files; "
                         f"the current input has {n_now}.")
            color = "#b36b00" if (not self._result_saved and not self._running) else "#334477"
        self._result_info_var.set(text)
        self._update_result_file_field()
        if self._result_label is not None:
            try:
                self._result_label.configure(fg=color)
            except tk.TclError:
                pass

    @staticmethod
    def _read_track_file(path: str):
        """Returns (settings dict, result dict or None)."""
        with np.load(path, allow_pickle=False) as data:
            names = set(data.files)
            settings = json.loads(str(data["settings_json"])) if "settings_json" in names else {}
            result = None
            if "points" in names:
                pts = np.array(data["points"], dtype=np.float64)
                if pts.ndim != 4 or pts.shape[-1] != 2:
                    raise ValueError("'points' must have shape (frames, M, N, 2)")
                S, M, N, _ = pts.shape
                status = (np.array(data["status"]).astype(np.uint8) if "status" in names
                          else np.zeros((S, M, N), np.uint8))
                conf = (np.array(data["confidence"], dtype=np.float64) if "confidence" in names
                        else np.full((S, M, N), np.nan))
                if status.shape != (S, M, N) or conf.shape != (S, M, N):
                    raise ValueError("'status' / 'confidence' shapes do not match 'points'")
                meta = json.loads(str(data["meta_json"])) if "meta_json" in names else {}
                mesh = meta.get("mesh")
                if mesh is None or len(mesh) != 6:
                    raise ValueError("the file has no mesh description")
                start = int(meta.get("start", 1))
                n_done = int(min(max(meta.get("n_done", S), 0), S))
                # Files saved before tracking ran outward from a reference
                # frame have no 'filled' array: their frames were a prefix.
                filled = (np.array(data["filled"]).astype(bool).reshape(-1) if "filled" in names
                          else np.arange(S) < n_done)
                if filled.shape != (S,):
                    raise ValueError("'filled' does not match 'points'")
                result = {"points": pts, "status": status, "confidence": conf, "start": start,
                          "end": start + S - 1, "n_done": n_done, "filled": filled,
                          "ref": (int(meta["ref"]) if meta.get("ref") is not None else None),
                          "mesh": (int(mesh[0]), int(mesh[1])) + tuple(float(v) for v in mesh[2:]),
                          "bad_frames": int(meta.get("bad_frames", 0)),
                          "n_files": int(meta.get("n_files", 0)), "settings": settings}
        return settings, result

    # ── own input selection (used when the 'files' pin has no images) ─────
    _IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm")

    def _local_dialog_dir(self):
        spec = self._local_spec
        if spec and spec.get("paths"):
            return os.path.dirname(spec["paths"][0]) or None
        base = self._get_project_base()
        return str(base) if base is not None else None

    def _on_select_video(self) -> None:
        if self._running or self._pin_source is not None:
            return
        exts = " ".join(f"*{e}" for e in VIDEO_EXTS)
        path = self.ask_open_filename(
            parent=self._dlg_parent(), title="Select a video to track",
            filetypes=[("Video files", exts), ("All files", "*.*")], initialdir=self._local_dialog_dir())
        if path:
            self._set_local({"kind": "video", "paths": [os.path.abspath(path)]}, show_errors=True)

    def _on_select_images(self) -> None:
        if self._running or self._pin_source is not None:
            return
        exts = " ".join(f"*{e}" for e in self._IMAGE_EXTS)
        names = self.ask_open_filenames(
            parent=self._dlg_parent(), title="Select the image files to track (Ctrl / Shift-click)",
            filetypes=[("Image files", exts), ("All files", "*.*")], initialdir=self._local_dialog_dir())
        if names:
            self._set_local({"kind": "images", "paths": sorted(os.path.abspath(n) for n in names)},
                            show_errors=True)

    def _on_clear_local(self) -> None:
        if not self._running:
            self._set_local(None)

    def _set_local(self, spec, show_errors: bool = False) -> None:
        """Use `spec` ({"kind", "paths"} with absolute paths, or None) as the
        node's own input. A video that cannot be opened keeps its path (so a
        project save does not drop it) and shows why in the Input group."""
        old_n = self._n_frames()
        if self._local_source is not None:
            self._local_source.close()
        self._local_spec, self._local_source, self._local_error = spec, None, ""
        if spec:
            try:
                self._local_source = FrameSource(spec["kind"], spec["paths"])
            except Exception as e:
                what = "video" if spec["kind"] == "video" else "images"
                self._local_error = f"Cannot open the selected {what}:\n{spec['paths'][0]}\n({e})"
                if show_errors:
                    messagebox.showerror("Select input", self._local_error, parent=self._dlg_parent())
        self._on_files_changed(old_n)

    def _local_input_params(self) -> dict | None:
        """The own selection for the project file, with paths RELATIVE to the
        project file where possible. Images from one folder are stored as
        that folder + file names; a very long list goes to a side file next
        to the project ('<project>.<node id>.mesh_track_images.txt')."""
        spec = self._local_spec
        if not spec:
            return None
        base = self._get_project_base()
        rel = [self._to_relative(p, base) for p in spec["paths"]]
        if spec["kind"] == "video":
            return {"kind": "video", "path": rel[0]}
        folders = {os.path.dirname(p) for p in rel}
        if len(folders) == 1:
            out = {"kind": "images", "folder": folders.pop(), "names": [os.path.basename(p) for p in rel]}
        else:
            out = {"kind": "images", "paths": rel}
        # However long: long params go to the project file's Data sheet
        # (project_io.py); 'list_file' side files of older projects are still read.
        return out

    def _restore_local_input(self, data) -> None:
        if not isinstance(data, dict) or data.get("kind") not in ("video", "images"):
            self._set_local(None)
            return
        base = self._get_project_base()
        if data["kind"] == "video":
            rel = [str(data.get("path", "") or "")]
        elif data.get("list_file"):
            try:
                text = (Path(base) / Path(str(data["list_file"])).name).read_text(encoding="utf-8")
                rel = [line.strip() for line in text.splitlines() if line.strip()]
            except (OSError, TypeError):
                self._local_spec, self._local_source = None, None
                self._local_error = f"The saved image list {data['list_file']} was not found."
                self._on_files_changed(self._n_frames())
                return
        elif "names" in data:
            # One folder: resolve it once, not once per file (thousands of
            # resolve() calls are slow on network / external drives).
            folder = self._to_absolute(str(data.get("folder", "") or "."), base)
            paths = [os.path.join(folder, str(n)) for n in data.get("names", []) if n]
            self._set_local({"kind": "images", "paths": paths} if paths else None)
            return
        else:
            rel = [str(p) for p in data.get("paths", [])]
        paths = [self._to_absolute(p, base) for p in rel if p]
        self._set_local({"kind": data["kind"], "paths": paths} if paths else None)

    def _on_load(self) -> None:
        if self._running:
            return
        path = self.ask_open_filename(
            parent=self._inspector_win, title="Load mesh tracking settings and result",
            filetypes=[("Mesh tracking files", "*.npz"), ("All files", "*.*")],
            **self._file_dialog_start())
        if not path:
            return
        if self._result is not None and not self._result_saved and not messagebox.askokcancel(
                "Load", "The current tracking result has not been saved and will be replaced. Continue?",
                parent=self._dlg_parent()):
            return
        try:
            settings, result = self._read_track_file(path)
        except Exception as e:
            messagebox.showerror("Load", f"Cannot load the file:\n{e}", parent=self._dlg_parent())
            return
        self._apply_settings(settings, from_file=True)
        self._result, self._result_path = result, os.path.abspath(path)
        self._result_saved, self._result_error = True, ""
        self._sel, self._series_user_view = (0, 0), False
        if result is not None:
            self._progress_var.set(f"Loaded {result['n_done']} frames from file.")
        self._update_result_info()
        self._update_status_item()
        self._validate_all()
        self._redraw_series()
        self._notify_result_changed()

    def _on_save(self) -> None:
        """Save to the node's result file without asking (Ctrl+S); the first
        time, when there is no result file yet, this is Save as."""
        if self._running:
            return
        if not self._result_path:
            self._on_save_as()
            return
        self._write_track_file(self._result_path, "Save")

    def _on_save_as(self) -> None:
        if self._running:
            return
        path = self.ask_save_as_filename(
            parent=self._inspector_win, title="Save mesh tracking settings and result",
            defaultextension=".npz", filetypes=[("Mesh tracking files", "*.npz")],
            **self._file_dialog_start())
        if not path:
            return
        self._write_track_file(path, "Save as")

    def _write_track_file(self, path: str, title: str) -> None:
        res = self._result
        # A result is saved with the settings that produced it, so the file is
        # self-consistent even if the UI was edited after the run.
        settings = res["settings"] if (res is not None and res.get("settings")) else self._settings_dict()
        arrays = {"settings_json": np.array(json.dumps(settings))}
        if res is not None:
            S = res["points"].shape[0]
            mesh = res["mesh"]
            meta = {"version": 1, "start": int(res["start"]), "end": int(res["end"]),
                    "ref": (int(res["ref"]) if res.get("ref") is not None else None),
                    "n_done": int(res["n_done"]), "mesh": [int(mesh[0]), int(mesh[1])] + [float(v) for v in mesh[2:]],
                    "bad_frames": int(res.get("bad_frames", 0)), "n_files": int(res.get("n_files", 0))}
            arrays.update(points=res["points"], status=res["status"], confidence=res["confidence"],
                          filled=self._filled(res),
                          frames=(res["start"] + np.arange(S)), meta_json=np.array(json.dumps(meta)))
        try:
            np.savez_compressed(path, **arrays)
        except Exception as e:
            messagebox.showerror(title, f"Cannot save the file:\n{e}", parent=self._dlg_parent())
            return
        final = path if path.lower().endswith(".npz") else path + ".npz"
        self._result_path, self._result_saved = os.path.abspath(final), True
        self._update_result_info()
        self._progress_var.set("Saved: " + final)

    # ══ Help / lifecycle ═════════════════════════════════════════════════
    def _open_help(self) -> None:
        if self._help_popup is not None and self._help_popup.winfo_exists():
            self._help_popup.lift()
            return
        popup = tk.Toplevel(self._inspector_win)
        popup.title("Mesh Template Track - Help")
        popup.geometry("720x600")
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

    def _cancel_after(self, attr: str) -> None:
        after_id = getattr(self, attr)
        if after_id is not None:
            try:
                self.canvas.after_cancel(after_id)
            except tk.TclError:
                pass
            setattr(self, attr, None)

    def close_inspector(self) -> None:
        if self._help_popup is not None:
            try:
                self._help_popup.destroy()
            except tk.TclError:
                pass
        self._cancel_after("_preview_debounce_id")
        self._end_pick(commit=False)  # an unfinished pick restores the corner
        super().close_inspector()
        self._reset_widget_refs()     # widgets are gone; a running job keeps going

    def on_destroy(self) -> None:
        self._destroyed = True
        if self._run_state is not None:
            self._run_state["cancel"].set()
        if self._copy_job is not None:
            self._copy_job["state"]["cancel"].set()
        for attr in ("_poll_id", "_preview_poll_id", "_preview_debounce_id", "_video_poll_id", "_copy_check_id"):
            self._cancel_after(attr)
        if self._local_source is not None:
            self._local_source.close()               # stops the video's frame counting too
        if self._preview_executor is not None:
            self._preview_executor.shutdown(wait=False)
        super().on_destroy()