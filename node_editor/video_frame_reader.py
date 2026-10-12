# node_editor/video_frame_reader.py
"""
Exact frame access for video files (no Tk).

Problem: reading frames from a video in order (cap.read() again and again)
is always correct, but jumping to frame k with
cap.set(cv2.CAP_PROP_POS_FRAMES, k) is NOT guaranteed to land on frame k:
depending on the codec, the container, the OpenCV / FFmpeg version and on
broken timestamps, a seek can return a neighbouring frame, a key frame, or a
partly decoded frame. CAP_PROP_FRAME_COUNT can also be wrong (it is read
from the container header, not counted).

Approach -- every frame returned by read(k) is the k-th frame of a plain
sequential decode:
  1. build_index() decodes the whole video once, in order, on a background
     thread with its own VideoCapture. It records the exact frame count and
     a fingerprint of every frame (64-bit hash of a 32 x 32 grey thumbnail).
  2. read(k) then uses the cheapest exact way:
       - k is the next frame                 -> cap.read()
       - k is a little ahead (<= 240 frames) -> decode forward to it (grab)
       - otherwise, index ready              -> seek, read, and VERIFY the
                                                fingerprint against frame k
       - verification failed / no index yet  -> decode in order from the
                                                start (or from the current
                                                position if k is ahead) --
                                                always exact, can be slow
  slow_cost(k) tells the caller beforehand how many frames a request will
  have to decode, so a UI can warn before a slow jump.

All reads go through one VideoCapture guarded by a lock (VideoCapture is not
thread-safe). Frames are returned as RGB uint8 (this framework's convention).
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict

import cv2
import numpy as np

VIDEO_EXTS = (".mp4", ".m4v", ".mov", ".avi", ".mkv", ".mpg", ".mpeg", ".wmv", ".webm",
              ".mts", ".m2ts", ".ts", ".3gp", ".flv", ".ogv", ".mjpeg", ".mjpg")


def frame_fingerprint(bgr: np.ndarray) -> bytes:
    """64-bit hash of a 32 x 32 grey thumbnail: identical for the same decoded
    frame, different for different frames (except truly identical images)."""
    gray = bgr if bgr.ndim == 2 else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    thumb = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    return hashlib.blake2b(thumb.tobytes(), digest_size=8).digest()


def _fourcc_text(value: float) -> str:
    code = int(value)
    text = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
    return text if text.isprintable() and text.strip() else "?"


class VideoFrameReader:
    FORWARD_DECODE_MAX = 240           # forward jumps up to this are decoded through, never sought
    SLOW_WARN_FRAMES = 300             # slow_cost() above this deserves a warning in the UI

    def __init__(self, path: str, cache_size: int = 16):
        self.path = path
        self._lock = threading.Lock()
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise IOError(f"OpenCV cannot open the video: {path}")
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self.reported_count = max(0, int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        self.codec = _fourcc_text(self._cap.get(cv2.CAP_PROP_FOURCC))
        self._next = 0                 # index of the frame the next cap.read() returns
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = cache_size
        # Filled by build_index():
        self._fingerprints: list | None = None
        self.frame_count: int | None = None        # exact, once indexed
        self.index_progress = 0.0
        self.index_error: str | None = None
        self._cancel = threading.Event()
        # Statistics shown in the UI.
        self.seeks = 0
        self.seek_failures = 0
        self.last_method = ""

    # ── state ─────────────────────────────────────────────────────────
    @property
    def index_ready(self) -> bool:
        return self._fingerprints is not None

    def count_known(self) -> int:
        """Exact count once indexed; until then the container's report."""
        return self.frame_count if self.frame_count is not None else self.reported_count

    def time_of(self, k: int) -> float | None:
        return k / self.fps if self.fps > 0 else None

    # ── index ─────────────────────────────────────────────────────────
    def build_index(self) -> None:
        """Decode the whole video in order (own VideoCapture; call from a
        background thread). Sets frame_count and the fingerprints."""
        cap = cv2.VideoCapture(self.path)
        try:
            if not cap.isOpened():
                raise IOError("cannot open the video for indexing")
            prints, total = [], max(1, self.reported_count)
            while not self._cancel.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                prints.append(frame_fingerprint(frame))
                self.index_progress = min(0.999, len(prints) / total)
            if self._cancel.is_set():
                return
            self.frame_count = len(prints)
            self._fingerprints = prints
            self.index_progress = 1.0
        except Exception as e:                       # reported in the UI
            self.index_error = f"{type(e).__name__}: {e}"
        finally:
            cap.release()

    # ── reading ───────────────────────────────────────────────────────
    def seek_reliable(self) -> bool:
        """False once seeking has failed verification often enough (then jumps
        go straight to the exact in-order decode instead of trying to seek)."""
        return self.seeks < 5 or self.seek_failures * 4 < self.seeks

    def slow_cost(self, k: int) -> int:
        """Frames read(k) would have to decode one by one (0 = fast)."""
        if k in self._cache or k == self._next or 0 < k - self._next <= self.FORWARD_DECODE_MAX:
            return 0
        if self.index_ready and self.seek_reliable():
            return 0                                 # seek + verify (seeking has been working)
        return k - self._next if k > self._next else k

    def read(self, k: int) -> np.ndarray | None:
        """The k-th frame (0-based) as RGB uint8, exactly; None past the end."""
        if k < 0:
            return None
        with self._lock:
            hit = self._cache.get(k)
            if hit is not None:
                self._cache.move_to_end(k)
                self.last_method = "cached"
                return hit
            if self.frame_count is not None and k >= self.frame_count:
                return None
            bgr = self._read_exact(k)
            if bgr is None:
                return None
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self._cache[k] = rgb
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
            return rgb

    def _grab_to(self, k: int) -> bool:
        while self._next < k:
            if not self._cap.grab():
                return False
            self._next += 1
        return True

    def _read_next(self):
        ok, frame = self._cap.read()
        if not ok:
            return None
        self._next += 1
        return frame

    def _verified(self, k: int, frame) -> bool:
        return self._fingerprints is None or frame_fingerprint(frame) == self._fingerprints[k]

    def _read_exact(self, k: int):
        if k == self._next:
            self.last_method = "next frame"
            return self._read_next()
        if 0 < k - self._next <= self.FORWARD_DECODE_MAX:
            self.last_method = "decoded forward"
            return self._read_next() if self._grab_to(k) else None
        if self.index_ready and self.seek_reliable():
            self.seeks += 1
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            ok, frame = self._cap.read()
            if ok and self._verified(k, frame):
                self._next = k + 1
                self.last_method = "seek (verified)"
                return frame
            self.seek_failures += 1
        # Exact fallback: decode in order (from the start if k is behind).
        if k < self._next or self.index_ready:   # after a failed seek the position is unknown
            self._reopen()
        self.last_method = "decoded from start" if k > 0 else "first frame"
        if not self._grab_to(k):
            return None
        frame = self._read_next()
        if frame is not None and not self._verified(k, frame):
            # Should not happen (in-order decode IS the reference); report it.
            self.last_method = "decoded from start (fingerprint mismatch!)"
        return frame

    def _reopen(self) -> None:
        self._cap.release()
        self._cap = cv2.VideoCapture(self.path)
        self._next = 0

    def close(self) -> None:
        self._cancel.set()
        with self._lock:
            self._cap.release()
            self._cache.clear()
