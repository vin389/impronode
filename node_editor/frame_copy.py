"""
Copy chosen frames of an image list or a video into a folder, as new image
files (no Tk; used by the Mesh Template Track node's "Copy selected frames
to a folder" panel).

Frames are chosen with the Image Sequence selector syntax, 1-based:
'1:100 201:500 600:900', '1:1000:10', '5', 'a:' (to the last), ':b'.

File names (naming dict):
  mode    "original": image files keep their name; video frames are
                      '<video name>_FRM<frame, 6 digits>' (1-based frame)
          "pattern" : pattern % number, e.g. 'selected_%06d'
  number  "frame"   : the original frame number (1-based) -- traceable
          "sequence": 1, 2, 3 ... in the order of the selection
  format  "same"    : image files are copied byte for byte (exact, fastest)
          "jpg" / "png": decoded and written in that format
The extension comes from the format, never from the pattern.
"""

from __future__ import annotations

import os
import re
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
_BAD_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def parse_frames(text: str, total: int):
    """(0-based indices, in the given order, without repeats; requested
    numbers beyond `total` that were dropped). Empty text -> ([], 0).
    ValueError for malformed text (message for the user)."""
    from node_editor.nodes.image_sequence_node import ImageSequenceNode
    sel, dropped = ImageSequenceNode._parse_selector(text, total)
    if sel is None:
        return [], 0
    return list(dict.fromkeys(sel)), dropped


def runs(indices) -> list:
    """Sorted contiguous runs [(first, last), ...] of 0-based indices."""
    out = []
    for k in sorted(set(int(i) for i in indices)):
        if out and k == out[-1][1] + 1:
            out[-1][1] = k
        else:
            out.append([k, k])
    return [tuple(r) for r in out]


def pattern_error(pattern: str) -> str | None:
    """Why a file-name pattern cannot be used, or None."""
    p = (pattern or "").strip()
    if not p:
        return "the name pattern is empty"
    if _BAD_CHARS.search(p):
        return 'the name pattern contains a character not allowed in file names (< > : " / \\ | ? *)'
    if Path(p).suffix.lower() in IMAGE_EXTS:
        return f"leave the extension ('{Path(p).suffix}') out of the pattern -- choose it under Format"
    body = p.replace("%%", "")
    if body.count("%") != 1 or not re.search(r"%0?\d*d", body):
        return "the pattern needs exactly one number field like %06d (e.g. selected_%06d)"
    try:
        p % 1
    except (TypeError, ValueError):
        return "the pattern needs exactly one number field like %06d (e.g. selected_%06d)"
    return None


def plan_targets(kind: str, paths: list, indices: list, dest: str, naming: dict) -> list:
    """[(index, source path or None, target path, exact copy)] in the order
    of `indices`. ValueError (message for the user) when the names do not
    work: duplicates, writing onto a source file, a video with 'same'."""
    fmt = naming.get("format", "jpg")
    mode = naming.get("mode", "original")
    if kind == "video" and fmt == "same":
        raise ValueError("video frames must be written as jpg or png")
    if mode == "pattern":
        err = pattern_error(naming.get("pattern", ""))
        if err:
            raise ValueError(err)
    video_stem = Path(paths[0]).stem if kind == "video" and paths else ""
    out, seen = [], {}
    for pos, k in enumerate(indices):
        src = paths[k] if kind == "images" else None
        if kind == "images" and (k < 0 or k >= len(paths)):
            raise ValueError(f"frame {k + 1} is beyond the {len(paths)} images")
        ext = Path(src).suffix if fmt == "same" else "." + fmt
        if mode == "pattern":
            number = k + 1 if naming.get("number", "frame") == "frame" else pos + 1
            stem = naming["pattern"].strip() % number
        elif kind == "video":
            stem = f"{video_stem}_FRM{k + 1:06d}"
        else:
            stem = Path(src).stem
        target = os.path.join(dest, stem + ext)
        key = os.path.normcase(os.path.abspath(target))
        if key in seen:
            raise ValueError(f"frames {seen[key] + 1} and {k + 1} would get the same file name "
                             f"'{stem + ext}' -- use new names (a pattern)")
        seen[key] = k
        if src is not None and key == os.path.normcase(os.path.abspath(src)):
            raise ValueError(f"'{stem + ext}' would be written onto its own source file -- choose another "
                             f"folder or new names")
        out.append((k, src, target, kind == "images" and fmt == "same"))
    return out


def _read_image(path: str):
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED) if buf.size else None


def _write_image(img, target: str, jpeg_quality: int) -> None:
    ext = Path(target).suffix.lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)] if ext in (".jpg", ".jpeg") else []
    if ext in (".jpg", ".jpeg"):
        if img.dtype != np.uint8:                       # JPEG holds 8 bits per channel
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        if img.ndim == 3 and img.shape[2] == 4:          # no alpha in JPEG
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    ok, data = cv2.imencode(ext, img, params)
    if not ok:
        raise ValueError(f"cannot encode as {ext}")
    tmp = target + ".part"
    data.tofile(tmp)
    os.replace(tmp, target)


def new_copy_state(total: int) -> dict:
    return {"cancel": threading.Event(), "done": 0, "total": total, "written": [], "errors": [],
            "t0": time.monotonic(), "finished": False, "error": None}


def copy_worker(kind: str, paths: list, targets: list, state: dict, jpeg_quality: int = 95,
                skip_existing: bool = False) -> None:
    """Background thread: write every target (see plan_targets). Video
    frames are read in increasing order with an own VideoFrameReader (exact
    frames, mostly sequential decoding). state['written'] lists the files
    written (or kept) in the order of `targets`. Never touches Tk."""
    reader = None
    written = {}
    try:
        if kind == "video":
            from node_editor.video_frame_reader import VideoFrameReader
            reader = VideoFrameReader(paths[0])
        order = sorted(range(len(targets)), key=lambda t: targets[t][0]) if kind == "video" else range(len(targets))
        for t in order:
            if state["cancel"].is_set():
                break
            k, src, dst, exact = targets[t]
            try:
                if skip_existing and os.path.exists(dst):
                    written[t] = dst
                elif exact:
                    tmp = dst + ".part"
                    shutil.copy2(src, tmp)
                    os.replace(tmp, dst)
                    written[t] = dst
                else:
                    if reader is not None:
                        rgb = reader.read(k)
                        img = None if rgb is None else cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    else:
                        img = _read_image(src)
                    if img is None:
                        raise ValueError("cannot read the frame")
                    _write_image(img, dst, jpeg_quality)
                    written[t] = dst
            except Exception as e:                       # reported, the rest goes on
                state["errors"].append(f"frame {k + 1}: {type(e).__name__}: {e}")
            state["done"] += 1
    except Exception as e:
        state["error"] = f"{type(e).__name__}: {e}"
    finally:
        if reader is not None:
            reader.close()
        state["written"] = [written[t] for t in range(len(targets)) if t in written]
        state["finished"] = True
