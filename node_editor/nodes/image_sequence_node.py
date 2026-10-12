# node_editor/nodes/image_sequence_node.py

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np

from node_editor.base_node import BaseNode
from node_editor.pin_types import PinSchema, PinDef, PinType
from node_editor.execution import ExecutionMode
from node_editor.project_context import get_project_directory, get_project_file_path
from node_editor.video_frame_reader import VideoFrameReader, VIDEO_EXTS


class ImageSequenceNode(BaseNode):
    """
    Provides frames from a list of IMAGE FILES or from a VIDEO FILE.

    Image files: the list is made from a folder, from picked files, or from
    numbered names (printf pattern + number range; the files may be created
    later -- "Wait for files" then waits for each one). Video: frames are
    read with exact frame access (see node_editor/video_frame_reader.py).

    Positions are 1-based everywhere (1 = first image / first video frame).
    The Batch section emits a pair of frames (Frame 1, Frame 2) for each
    loop index i. Paths are stored relative to the project file.
    """

    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE      = "image_sequence"
    DISPLAY_NAME   = "Image Sequence"
    CATEGORY       = "source"
    NODE_WIDTH     = 260
    NODE_HEIGHT    = 270

    HELP_TEXT = """\
Image Sequence
==============
Sends frames from a list of IMAGE FILES or from a VIDEO FILE, one pair of
frames (frame 1, frame 2) at a time.

1. SOURCE
---------
Image files -- the frames are a LIST of files (position 1 = first line).
  Make the list with one of:
    Folder...          every image in a folder (sorted by name).
    Files...           pick image files yourself.
    Numbered names...  names made from a pattern and a number range, e.g.
                       C:/data/img_%04d.png, numbers 1..500  ->
                       img_0001.png ... img_0500.png. The files do NOT have
                       to exist yet (another program may still write them).
                       %d = plain number, %04d = 4 digits with leading zeros.
    Show full list     read-only view of the whole list with 1-based numbers
                       (select + Ctrl+C to copy; unselected lines are grey).
    Edit selected list...  see / edit / reorder / delete lines of the
                       SELECTED files. With a selector, Apply makes those
                       lines the new list and clears the selector.
  Wait for files that are not there yet:
    when a file in the list is missing (or still being written), wait and
    check again every N seconds until it can be read, instead of stopping.
    Turned on automatically when "Numbered names" makes names of files
    that do not exist yet. The summary line shows how many are missing.

Video file -- the frames are the frames of a video (mp4, avi, mov, mkv,
  mpg, wmv, ... whatever OpenCV/FFmpeg can decode). Position 1 = first frame.
  Exact frame access:
    Seeking in a video (jumping to frame k) is not guaranteed to land on
    frame k with every codec / file / OpenCV version, and the frame count
    written in the file header can be wrong. So when a video is opened the
    node first reads it once in order (in the background) to COUNT the
    frames exactly and to record a fingerprint of every frame. Then:
      - the next frame / a short jump forward: decoded in order (always exact);
      - a long jump: seek, then the frame is CHECKED against its fingerprint;
      - if the check fails (or the index is not ready yet): decoded in order
        from the start -- exact, but slow for far frames.
    The "Frame access" line shows whether seeking works for the file. If a
    jump will need slow in-order decoding of many frames, you are asked once
    per video. Playing / stepping forward is always fast.
  Extract frames to image files...
    writes the frames (or a range) as PNG/JPG files and can switch the
    source to them: fastest random access, and nodes that need a file list
    (e.g. Mesh Template Track, Chessboard Calibration via the 'files' pin)
    can then use them. For a video source the 'files' output is empty.

Selector (both sources) -- use only some of the images / video frames.
    1:30, 61:90, 100:105   images 1-30, 61-90 and 100-105 (1-based,
                           inclusive): 30 + 30 + 6 = 66 images.
    7                      a single image.
    1:100:5                every 5th image of 1..100.
    200:  /  :50           from 200 to the last  /  from the first to 50.
  Separate items with commas or spaces; they are used in the order given.
  Empty = all images. Press Enter (or leave the box) to apply; the line
  below shows how many are selected. Numbers beyond the last image are
  ignored (and reported).
  After applying, EVERYTHING else works on the selected images only, as if
  they were the whole sequence: Start / End / i / Frame 1 / Frame 2, the
  index pins and the index / count / files outputs all count 1..66 in the
  example ('files' lists the 66 selected files). "Now sending" shows
  "#k: name" = position k in the selection, then the file / video frame.

2. FRAMES TO SEND (batch)
-------------------------
  Start / End / Step   the loop index i runs from Start to End inclusive
                       (End is always included). Negative Step runs backward.
  Frame 1 / Frame 2    which positions are sent for each i: i, i+N, i-N or a
                       fixed number. Default Frame 1 = 1 (reference frame),
                       Frame 2 = i. For "previous / current" use i-1 and i.
  Current i            the loop index sent by Current / Play.
  First Prev Current Next Last   send that step of the loop.
  Play / Stop          run the loop at FPS, or -- when the 'next' input pin
                       is connected -- one step per trigger pulse.
  Red text = invalid value (e.g. a position outside 1..number of frames).

INPUT PINS
  index 1 / index 2  select the two frames directly (1-based); they take
                     precedence over the batch.
  next               trigger: advance one batch step per pulse.
  current            trigger: re-send the current batch step.

OUTPUT PINS
  frame 1 / frame 2  the images (RGB).   index 1 / index 2  their positions.
  files              absolute paths of the image list (empty for a video).
  count              number of frames (list length / exact video frame count).
  trig               pulse on every batch step.  start  pulse on the first step.

Ctrl+H shows this help.
"""

    # Shared decode pool across all ImageSequence instances.
    _DECODE_POOL = ThreadPoolExecutor(max_workers=4)

    # supported extensions (OpenCV-readable)
    _IMG_EXTS = {
        ".jpg", ".jpeg", ".png", ".bmp",
        ".tif", ".tiff", ".webp", ".ppm", ".pgm",
    }

    def __init__(self, node_id: str, canvas: tk.Canvas):
        super().__init__(node_id, canvas)

        self._file_paths: list[str] = []
        self._current_index: int = 0
        self._playing: bool = False
        self._after_id: str | None = None
        self._last_frame: np.ndarray | None = None
        self._last_frame_index: int = -1
        self._last_frame_1: np.ndarray | None = None
        self._last_frame_1_index: int = -1
        self._last_frame_2: np.ndarray | None = None
        self._last_frame_2_index: int = -1

        self._decode_lock = threading.Lock()
        self._decode_future = None
        self._decode_pending_req: tuple[int, str, int] | None = None
        self._decode_request_id: int = 0
        self._destroyed: bool = False
        self._batch_fetch_token: int = 0
        self._slider_dragging: bool = False

        self._frame_index_input_connected: bool = False
        self._trigger_linked: bool = False
        self._batch_running: bool = False
        self._trigger_pulse_counter: int = 0

        self._init_ui_state()

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[
                PinDef("frame_index_1", PinType.SCALAR, "index 1", optional=True),
                PinDef("frame_index_2", PinType.SCALAR, "index 2", optional=True),
                PinDef("current_trigger", PinType.TRIGGER, "current", optional=True),
                PinDef("trigger", PinType.TRIGGER, "next", optional=True),
            ],
            outputs=[
                PinDef("image_1",        PinType.IMAGE,  "frame 1"),
                PinDef("image_2",        PinType.IMAGE,  "frame 2"),
                PinDef("frame_index_1",    PinType.SCALAR, "index 1"),
                PinDef("frame_index_2",  PinType.SCALAR, "index 2"),
                PinDef("files",          PinType.STRING, "files"),
                PinDef("frame_count",    PinType.SCALAR, "count"),
                PinDef("trigger",        PinType.TRIGGER, "trig"),
                PinDef("batch_start",    PinType.TRIGGER, "start"),
            ]
        )

    def _init_ui_state(self) -> None:
        if hasattr(self, "_method_var"):
            return

        self._method_var = tk.StringVar(value="pattern")     # legacy; kept for old projects
        # Source: "images" (a list of files) or "video" (one video file).
        self._source_var = tk.StringVar(value="images")
        # Wait for list files that do not exist yet / are still being written.
        self._wait_files_var = tk.BooleanVar(value=False)
        self._pattern_first_var = tk.StringVar(value="1")
        self._pattern_last_var = tk.StringVar(value="100")
        self._pattern_step_var = tk.StringVar(value="1")
        self._list_summary_var = tk.StringVar(
            value="No images yet. Make the list with Folder..., Files... or Numbered names...")
        self._missing_cache = (None, 0, False)          # (list key, missing count, fully checked)
        self._video_path_var = tk.StringVar(value="")   # project-relative path of the video
        self._video_name_var = tk.StringVar(value="No video. Use Open video...")
        self._video_info_var = tk.StringVar(value="")
        self._video_access_var = tk.StringVar(value="")
        self._video_reader = None
        self._video_warned: set = set()
        self._images_panel = None
        self._video_panel = None
        self._video_access_label = None
        self._help_popup = None
        # Selector: which source frames are used ("1:30, 61:90"; empty = all).
        # _selector_var is the entry text; _selector_applied is the text in
        # effect (applied on Enter / leaving the box) -- a plain str, since
        # the frame-read worker threads map positions through it.
        self._selector_var = tk.StringVar(value="")
        self._selector_applied = ""
        self._selector_info_var = tk.StringVar(value="")
        self._selector_cache = None          # ((text, source total), (map | None, dropped))
        self._selector_frame = None
        self._selector_entry = None
        self._selector_info_label = None
        self._selector_var.trace_add("write", lambda *_a: self._update_selector_info())
        self._pattern_var = tk.StringVar(value="/images/DCIM%04d.JPG")
        self._start_var = tk.StringVar(value="0")
        self._end_var = tk.StringVar(value="100")
        self._pattern_wait_var = tk.DoubleVar(value=0.5)
        self._folder_var = tk.StringVar(value="(no folder)")

        self._info_var = tk.StringVar(value="no files loaded")
        self._play_btn_var = tk.StringVar(value="Play")
        self._fps_var = tk.StringVar(value="5")
        self._slider_var = tk.IntVar(value=0)
        self._idx_var = tk.StringVar(value="1")
        self._count_var = tk.StringVar(value="/ 0")
        self._fname_var = tk.StringVar(value="")
        self._size_var = tk.StringVar(value="")
        self._status_var = tk.StringVar(value="ready")
        self._frame_index_input_connected = False
        self._batch_start_var = tk.StringVar(value="1")
        self._batch_end_var = tk.StringVar(value="0")
        self._batch_step_var = tk.StringVar(value="1")
        self._batch_current_var = tk.StringVar(value="1")
        # Default pair: frame 1 fixed at the first frame (reference), frame 2
        # walks through the sequence (i = 1 .. number of frames).
        self._batch_frame_1_var = tk.StringVar(value="1")
        self._batch_frame_2_var = tk.StringVar(value="i")
        self._batch_mode_var = tk.StringVar(value="Timer mode: advances at FPS")
        self._loop_entries: dict[str, tk.Entry] = {}
        self._trigger_linked = False
        self._batch_running = False
        self._batch_indices: list[int] = []
        self._batch_position = 0
        self._batch_after_id: str | None = None
        self._batch_first_btn = None
        self._batch_prev_btn = None
        self._batch_current_btn = None
        self._batch_next_btn = None
        self._batch_last_btn = None
        self._batch_play_btn = None
        for variable in (
            self._batch_start_var,
            self._batch_end_var,
            self._batch_step_var,
            self._batch_current_var,
            self._batch_frame_1_var,
            self._batch_frame_2_var,
        ):
            variable.trace_add("write", self._validate_loop_fields)

        # Inspector-only widgets (exist only while popup is open).
        self._pat_entry = None
        self._pattern_wait_entry = None
        self._folder_label = None
        self._slider = None
        self._idx_entry = None
        self._pat_frame = None
        self._fld_frame = None
        self._batch_frame = None

    # ── build_body ────────────────────────────────────────────────

    def build_body(self) -> None:
        self._init_ui_state()
        x, y, w, h = self.x, self.y, self.width, self.height

        # Compact canvas shell (Phase 2). Full controls are in inspector popup.
        self._body_rect = self.canvas.create_rectangle(
            x, y, x+w, y+h,
            fill="#DDD1AF", outline="#cc8844", width=2,
            tags=(self.node_id, "node_body"))
        self._title_item = self.canvas.create_text(
            x+w/2, y+13,
            text=self.DISPLAY_NAME,
            font=("Arial", 9, "bold"), fill="#161512",
            tags=(self.node_id,))

        status_lbl = tk.Label(
            self.canvas, textvariable=self._status_var,
            font=("Arial", 7), bg="#DDD1AF", fg="#5b4a2f"
        )
        self.canvas.create_window(
            x + w / 2, y + h - 14, window=status_lbl,
            tags=(self.node_id,),
        )

        self._canvas_items += [self._body_rect, self._title_item]

        # ── runtime state ─────────────────────────────────────────
        if not hasattr(self, "_file_paths"):
            self._file_paths:     list[str] = []
            self._current_index:  int = 0
            self._playing:        bool = False
            self._after_id:       str | None = None
            self._last_frame_1:     np.ndarray | None = None
            self._last_frame_1_index: int = -1
            self._last_frame_2:   np.ndarray | None = None
            self._last_frame_2_index: int = -1

            self._decode_lock = threading.Lock()
            self._decode_future = None
            self._decode_pending_req: tuple[int, str, int] | None = None
            self._decode_request_id: int = 0
            self._destroyed: bool = False
            # Bumped to cancel superseded/in-flight background frame fetches
            # (including a pattern-mode wait-for-file retry loop).
            self._batch_fetch_token: int = 0

            self._slider_dragging: bool = False

    def build_inspector(self, parent: tk.Frame) -> None:
        self._init_ui_state()
        win = self._inspector_win
        if win is not None:
            for seq in ("<Control-h>", "<Control-H>"):
                win.bind(seq, lambda _e: self._open_help())
        top = tk.Frame(parent)
        top.pack(fill="both", expand=True)
        small, hint = ("Arial", 8), ("Arial", 7)

        # ── 1. Source ─────────────────────────────────────────────────
        src = tk.LabelFrame(top, text="1. Source", padx=6, pady=4, font=("Arial", 9, "bold"))
        src.pack(fill="x", pady=(0, 6))
        row = tk.Frame(src)
        row.pack(fill="x")
        for value, label in (("images", "Image files"), ("video", "Video file")):
            tk.Radiobutton(row, text=label, value=value, variable=self._source_var, font=("Arial", 9),
                           command=lambda: self._on_source_change(user=True)).pack(side="left", padx=(0, 12))
        tk.Button(row, text="Help (Ctrl+H)", font=small, command=self._open_help).pack(side="right")

        # Image files panel: one list, three ways to make it.
        p = tk.Frame(src)
        self._images_panel = p
        tk.Label(p, textvariable=self._list_summary_var, font=small, anchor="w", justify="left",
                 wraplength=430, fg="#2f3e5c").pack(fill="x", pady=(4, 3))
        b = tk.Frame(p)
        b.pack(fill="x")
        tk.Label(b, text="Make the list:", font=small).pack(side="left")
        for text, cmd in (("Folder...", self._browse_folder), ("Files...", self._browse_files),
                          ("Numbered names...", self._open_pattern_dialog),
                          ("Show full list", self._show_full_list),
                          ("Edit selected list...", self._edit_file_list)):
            tk.Button(b, text=text, font=small, command=cmd).pack(side="left", padx=(4, 0))
        tk.Label(p, font=hint, fg="#666666", anchor="w", justify="left", wraplength=430, text=(
            "Folder / Files: images that exist now.  Numbered names: names such as img_%04d.png "
            "for a number range -- the files may be written later by another program.")).pack(
            fill="x", pady=(2, 0))
        w = tk.Frame(p)
        w.pack(fill="x", pady=(4, 0))
        tk.Checkbutton(w, text="Wait for files that are not there yet", variable=self._wait_files_var,
                       font=small, command=self._update_list_summary).pack(side="left")
        tk.Label(w, text="check every", font=small).pack(side="left", padx=(6, 2))
        self._pattern_wait_entry = tk.Entry(w, textvariable=self._pattern_wait_var, width=5, font=small)
        self._pattern_wait_entry.pack(side="left")
        tk.Label(w, text="s", font=small).pack(side="left", padx=(2, 0))

        # Video panel.
        v = tk.Frame(src)
        self._video_panel = v
        tk.Label(v, textvariable=self._video_name_var, font=small, anchor="w", justify="left",
                 wraplength=430, fg="#2f3e5c").pack(fill="x", pady=(4, 2))
        vb = tk.Frame(v)
        vb.pack(fill="x")
        tk.Button(vb, text="Open video...", font=small, command=self._browse_video).pack(side="left")
        tk.Button(vb, text="Extract frames to image files...", font=small,
                  command=self._open_extract_dialog).pack(side="left", padx=(4, 0))
        tk.Label(v, textvariable=self._video_info_var, font=small, anchor="w", justify="left",
                 wraplength=430).pack(fill="x", pady=(4, 0))
        self._video_access_label = tk.Label(v, textvariable=self._video_access_var, font=small, anchor="w",
                                            justify="left", wraplength=430)
        self._video_access_label.pack(fill="x")

        # Selector (both sources). The panels above are re-packed BEFORE
        # this frame when the source changes (see _on_source_change).
        sf = tk.Frame(src)
        self._selector_frame = sf
        sf.pack(fill="x", side="bottom", pady=(6, 0))
        srow = tk.Frame(sf)
        srow.pack(fill="x")
        tk.Label(srow, text="Selector:", font=small).pack(side="left")
        self._selector_entry = se = tk.Entry(srow, textvariable=self._selector_var, font=small)
        se.pack(side="left", fill="x", expand=True, padx=(4, 0))
        se.bind("<Return>", self._apply_selector)
        se.bind("<FocusOut>", self._apply_selector)
        # Grey usage hint shown inside the empty box (Tk entries have no
        # placeholder); clicking it focuses the entry.
        placeholder = tk.Label(se, text="e.g.  1:30, 61:90, 100:105   (1-based, inclusive; empty = all)",
                               font=hint, fg="#9a9a9a", bg="white", anchor="w", cursor="xterm")
        placeholder.bind("<Button-1>", lambda _e: se.focus_set())

        def _toggle_placeholder(*_a):
            try:
                if self._selector_var.get() or se.focus_get() is se:
                    placeholder.place_forget()
                else:
                    placeholder.place(x=2, rely=0.5, anchor="w", relwidth=0.97)
            except (tk.TclError, KeyError):
                pass
        se.bind("<FocusIn>", _toggle_placeholder, add="+")
        se.bind("<FocusOut>", _toggle_placeholder, add="+")
        self._selector_placeholder_trace = self._selector_var.trace_add("write", _toggle_placeholder)
        _toggle_placeholder()
        self._selector_info_label = tk.Label(sf, textvariable=self._selector_info_var, font=small,
                                             anchor="w", justify="left", wraplength=430)
        self._selector_info_label.pack(fill="x")

        # ── 2. Frames to send ─────────────────────────────────────────
        self._batch_frame = bf = tk.LabelFrame(top, text="2. Frames to send", padx=6, pady=4,
                                               font=("Arial", 9, "bold"))
        bf.pack(fill="x", pady=(0, 6))
        tk.Label(bf, text="Positions are 1-based (1 = first image / first video frame). "
                          "The loop index i runs from Start to End by Step.",
                 font=hint, fg="#555555", wraplength=430, justify="left").grid(
            row=0, column=0, columnspan=6, sticky="w")

        def entry(row_, col, label, var, key, width=6, padx=(0, 2)):
            tk.Label(bf, text=label, font=small).grid(row=row_, column=col, sticky="w", padx=padx, pady=(4, 0))
            e = tk.Entry(bf, textvariable=var, width=width, justify="center", font=small)
            e.grid(row=row_, column=col + 1, sticky="w", pady=(4, 0))
            e.bind("<Return>", self._on_batch_range_commit)
            e.bind("<FocusOut>", self._on_batch_range_commit)
            self._loop_entries[key] = e

        entry(1, 0, "Start:", self._batch_start_var, "start")
        entry(1, 2, "End:", self._batch_end_var, "end", padx=(10, 2))
        entry(1, 4, "Step:", self._batch_step_var, "step", padx=(10, 2))
        entry(2, 0, "Frame 1:", self._batch_frame_1_var, "frame_1", width=7)
        entry(2, 2, "Frame 2:", self._batch_frame_2_var, "frame_2", width=7, padx=(10, 2))
        tk.Label(bf, text="i, i+N, i-N or a fixed number", font=hint, fg="#555555").grid(
            row=2, column=4, columnspan=2, sticky="w", padx=(10, 0), pady=(4, 0))
        entry(3, 0, "Current i:", self._batch_current_var, "current")
        tk.Label(bf, text="the step sent by Current / Play", font=hint, fg="#555555").grid(
            row=3, column=2, columnspan=4, sticky="w", padx=(10, 0), pady=(4, 0))

        nav = tk.Frame(bf)
        nav.grid(row=4, column=0, columnspan=6, sticky="ew", pady=(6, 0))
        cfg = dict(font=small, width=7)
        self._batch_first_btn = tk.Button(nav, text="First", command=self._go_first, **cfg)
        self._batch_prev_btn = tk.Button(nav, text="Prev", command=self._prev_frame, **cfg)
        self._batch_current_btn = tk.Button(nav, text="Current", command=self._send_current_batch_frame, **cfg)
        self._batch_next_btn = tk.Button(nav, text="Next", command=self._next_frame, **cfg)
        self._batch_last_btn = tk.Button(nav, text="Last", command=self._go_last, **cfg)
        for btn in (self._batch_first_btn, self._batch_prev_btn, self._batch_current_btn,
                    self._batch_next_btn, self._batch_last_btn):
            btn.pack(side="left", padx=1)
        self._batch_play_btn = tk.Button(nav, textvariable=self._play_btn_var, command=self._toggle_play,
                                         font=("Arial", 8, "bold"), width=7)
        self._batch_play_btn.pack(side="left", padx=(6, 1))
        tk.Spinbox(nav, from_=0.0001, to=1000.0, increment=0.1, textvariable=self._fps_var, width=7,
                   font=small).pack(side="right", padx=2)
        tk.Label(nav, text="FPS:", font=small).pack(side="right", padx=(8, 0))
        tk.Label(bf, textvariable=self._batch_mode_var, font=small, fg="#356a49").grid(
            row=5, column=0, columnspan=6, sticky="w", pady=(4, 0))

        # ── 3. Now sending ────────────────────────────────────────────
        now = tk.LabelFrame(top, text="3. Now sending", padx=6, pady=4, font=("Arial", 9, "bold"))
        now.pack(fill="x")
        tk.Label(now, textvariable=self._fname_var, font=small, anchor="nw", justify="left",
                 height=2).pack(fill="x")
        tk.Label(now, textvariable=self._size_var, font=small, anchor="w", justify="left").pack(fill="x")

        self._on_source_change(user=False)
        self._update_list_summary()
        self._update_video_labels()
        self._refresh_inspector_widgets()
        self._validate_loop_fields()
        self._refresh_current_file_display()

    def _refresh_inspector_widgets(self) -> None:
        if self._slider is not None and self._slider.winfo_exists():
            self._slider.configure(from_=0, to=max(0, self._frame_total() - 1))
            self._slider_var.set(max(0, min(self._current_index, max(0, self._frame_total() - 1))))
            self._slider.configure(state=tk.DISABLED if self._frame_index_input_connected else tk.NORMAL)
        self._validate_loop_fields()

    def _set_external_index_connected(self, connected: bool) -> None:
        self._frame_index_input_connected = bool(connected)
        if connected and self._batch_running:
            self._stop_batch("external index connected")
        if self._slider is not None and self._slider.winfo_exists():
            self._slider.configure(state=tk.DISABLED if connected else tk.NORMAL)

    def on_input_link_changed(self, pin_name: str, connected: bool) -> None:
        """Switch batch progression immediately when the next link changes."""
        if pin_name != "trigger":
            return
        self._trigger_linked = bool(connected)
        self._batch_mode_var.set(
            "Trigger mode: waiting for next pulse"
            if self._trigger_linked
            else "Timer mode: advances at FPS"
        )
        if not self._batch_running:
            return
        if self._trigger_linked:
            self._cancel_batch_timer()
            self._status_var.set("batch waiting for trigger")
        else:
            self._status_var.set("batch running at FPS")
            self._schedule_batch_tick(0)

    def _is_video(self) -> bool:
        return hasattr(self, "_source_var") and self._source_var.get() == "video"

    # ── selector ──────────────────────────────────────────────────
    # Two index spaces:
    #   source index   -- 0-based position in the FULL list / video;
    #   position       -- 0-based position in the SELECTED frames.
    # Everything the user and downstream nodes see (batch Start/End/i,
    # index pins, 'count', 'files', _current_index, the frame caches) is in
    # positions; only the actual frame reads map a position to its source
    # index (_src_index). With an empty selector the two are identical.

    def _source_total(self) -> int:
        """Number of frames of the whole source (list length / video frames)."""
        if self._is_video():
            reader = getattr(self, "_video_reader", None)
            return reader.count_known() if reader is not None else 0
        return len(getattr(self, "_file_paths", []))

    def _frame_total(self) -> int:
        """Number of SELECTED frames -- the frames the batch loops over."""
        sel = self._selection()[0]
        return self._source_total() if sel is None else len(sel)

    @staticmethod
    def _parse_selector(text: str, total: int) -> tuple[list[int] | None, int]:
        """'1:30, 61:90, 100' (1-based, inclusive) -> (0-based source indices
        in the given order, number of requested indices beyond `total` that
        were dropped). Also 'a:b:step', 'a:' (to the last) and ':b' (from
        the first). Empty text -> (None, 0) = everything.
        Raises ValueError for malformed text."""
        text = re.sub(r"\s*:\s*", ":", (text or "").strip())
        if not text:
            return None, 0
        out: list[int] = []
        dropped = 0
        for token in re.split(r"[,;\s]+", text):
            if not token:
                continue
            m = re.fullmatch(r"(\d*)(?::(\d*)(?::(\d+))?)?", token)
            if m is None or token in (":", "::"):
                raise ValueError(f"'{token}' is not a number or a range a:b")
            a_s, b_s, step_s = m.groups()
            is_range = ":" in token
            if not is_range:
                a = b = int(a_s)
            else:
                a = int(a_s) if a_s else 1
                b = int(b_s) if b_s else max(total, a)
            step = int(step_s) if step_s else 1
            if a < 1 or b < 1:
                raise ValueError(f"'{token}': numbers start at 1")
            if step < 1:
                raise ValueError(f"'{token}': step must be 1 or more")
            if a > b:
                raise ValueError(f"'{token}': the first number is larger than the last")
            requested = (b - a) // step + 1
            stop = min(b, total)              # never build a huge range past the source
            if a <= stop:
                kept = range(a - 1, stop, step)
                out.extend(kept)
                dropped += requested - len(kept)
            else:
                dropped += requested
        return out, dropped

    def _selection(self) -> tuple[list[int] | None, int]:
        """(source indices of the selected frames or None for all, dropped)
        for the APPLIED selector; cached per (text, source total). Touches no
        Tk variable, so frame-read worker threads may call it."""
        text = getattr(self, "_selector_applied", "")
        if not text:
            return None, 0
        key = (text, self._source_total())
        cache = self._selector_cache
        if cache is None or cache[0] != key:
            try:
                value = self._parse_selector(text, key[1])
            except ValueError:
                value = (None, 0)             # cannot happen: only valid text is applied
            cache = self._selector_cache = (key, value)
        return cache[1]

    def _src_index(self, position: int) -> int:
        """Source index of a 0-based position in the selected frames."""
        sel = self._selection()[0]
        if sel is None:
            return position
        return sel[position] if 0 <= position < len(sel) else -1

    def _update_selector_info(self) -> None:
        """'66 images are selected' -- for the text being typed (live), or an
        error; applied on Enter / leaving the box."""
        unit = "frames" if self._is_video() else "images"
        total = self._source_total()
        typed = self._selector_var.get()
        color = "#2f3e5c"
        try:
            sel, dropped = self._parse_selector(typed, total)
        except ValueError as e:
            current = self._frame_total()
            text = (f"Invalid selector: {e}.  Still using the previous selection "
                    f"({current} {unit}).")
            color = "#c62828"
        else:
            n = total if sel is None else len(sel)
            text = f"{n} {unit} are selected" + ("" if sel is None else f"  (of {total})")
            if dropped:
                text += f"\n{dropped} requested numbers are beyond the last ({total}) and are ignored."
                color = "#b36b00"
            if typed.strip() != self._selector_applied.strip():
                text += "   -- press Enter to apply"
        self._selector_info_var.set(text)
        label = getattr(self, "_selector_info_label", None)
        if label is not None:
            try:
                label.configure(fg=color)
            except tk.TclError:
                pass
        entry = getattr(self, "_selector_entry", None)
        if entry is not None:
            try:
                entry.configure(fg="#c62828" if color == "#c62828" else "#111111")
            except tk.TclError:
                pass

    def _apply_selector(self, _event=None) -> None:
        """Make the typed selector the one in effect (if valid and changed):
        the batch then runs over the selected frames, i = 1 .. count."""
        typed = self._selector_var.get().strip()
        if typed == self._selector_applied:
            self._update_selector_info()
            return
        try:
            self._parse_selector(typed, self._source_total())
        except ValueError:
            self._update_selector_info()          # shows the error; keep the old selection
            return
        self._set_selector(typed, emit=True)

    def _set_selector(self, text: str, *, emit: bool) -> None:
        self._selector_applied = text.strip()
        if self._selector_var.get().strip() != self._selector_applied:
            self._selector_var.set(self._selector_applied)
        self._invalidate_frame_cache()            # positions now mean other frames
        self._stop_batch("ready")
        n = self._frame_total()
        # Keep Step and the Frame 1 / Frame 2 expressions; the loop restarts
        # over the new selection.
        self._batch_start_var.set("1")
        self._batch_end_var.set(str(max(1, n)))
        self._batch_current_var.set("1")
        self._current_index = 0
        self._update_selector_info()
        self._refresh_current_file_display()
        if emit and n and self._validate_loop_fields():
            self._emit_batch_frames(1)

    def _invalidate_frame_cache(self) -> None:
        """Forget decoded frames: their positions refer to an old list /
        selection, and _emit_batch_frames re-uses frames by position."""
        self._batch_fetch_token += 1
        self._last_frame, self._last_frame_index = None, -1
        self._last_frame_1, self._last_frame_1_index = None, -1
        self._last_frame_2, self._last_frame_2_index = None, -1

    def _validate_loop_fields(self, *_args) -> bool:
        """Show valid loop values in black and invalid values in red."""
        count = self._frame_total()
        values: dict[str, int | None] = {}
        for key, variable in (
            ("start", self._batch_start_var),
            ("end", self._batch_end_var),
            ("step", self._batch_step_var),
            ("current", self._batch_current_var),
        ):
            try:
                values[key] = int(variable.get().strip())
            except (TypeError, ValueError):
                values[key] = None

        # Positions are always positions in the CURRENT source (the list --
        # "Numbered names" makes the list explicitly -- or the video), so
        # every value is limited to 1..number of frames.
        limit = count

        def in_range(v):
            return v is not None and v >= 1 and (limit is None or v <= limit)

        valid = {
            "start": in_range(values["start"]),
            "end": in_range(values["end"]),
            "step": values["step"] is not None and values["step"] != 0
                    and (limit is None or abs(values["step"]) <= limit),
        }
        if (valid["start"] and valid["end"] and valid["step"]
                and ((values["step"] > 0 and values["start"] > values["end"])
                     or (values["step"] < 0 and values["start"] < values["end"]))):
            valid["start"] = valid["end"] = False

        offsets = {
            "frame_1": self._parse_batch_expression(self._batch_frame_1_var.get()),
            "frame_2": self._parse_batch_expression(self._batch_frame_2_var.get()),
        }
        loop_indices = self._build_batch_indices(
            values["start"], values["end"], values["step"],
        ) if all(valid[key] for key in ("start", "end", "step")) else []
        valid["current"] = (
            values["current"] is not None and values["current"] in loop_indices
        )
        for key, offset in offsets.items():
            valid[key] = (
                offset is not None
                and bool(loop_indices)
                # frame number = index + offset (or a constant): monotonic, so
                # the smallest and largest index decide (lists can be 10k+ long)
                and all(
                    in_range(self._expression_frame_number(offset, index))
                    for index in (min(loop_indices), max(loop_indices))
                )
            )

        for key, entry in self._loop_entries.items():
            if entry.winfo_exists():
                entry.configure(fg="#111111" if valid[key] else "#c62828")
        self._update_batch_navigation(loop_indices, valid)
        return all(valid.values())

    def _update_batch_navigation(self, loop_indices: list[int],
                                 valid: dict[str, bool]) -> None:
        """Disable manual navigation at the inclusive loop boundaries."""
        if not self._frame_total():
            states = {
                "first": tk.DISABLED, "prev": tk.DISABLED,
                "current": tk.DISABLED, "next": tk.DISABLED,
                "last": tk.DISABLED, "play": tk.DISABLED,
            }
        elif not loop_indices or not valid.get("current", False):
            states = {
                "first": tk.NORMAL, "prev": tk.DISABLED,
                "current": tk.NORMAL, "next": tk.DISABLED,
                "last": tk.NORMAL, "play": tk.NORMAL,
            }
        else:
            current = int(self._batch_current_var.get())
            states = {
                "first": tk.NORMAL,
                "prev": tk.DISABLED if current == loop_indices[0] else tk.NORMAL,
                "current": tk.NORMAL,
                "next": tk.DISABLED if current == loop_indices[-1] else tk.NORMAL,
                "last": tk.NORMAL,
                "play": tk.NORMAL,
            }
        for button, state in (
            (self._batch_first_btn, states["first"]),
            (self._batch_prev_btn, states["prev"]),
            (self._batch_current_btn, states["current"]),
            (self._batch_next_btn, states["next"]),
            (self._batch_last_btn, states["last"]),
            (self._batch_play_btn, states["play"]),
        ):
            if button is not None and button.winfo_exists():
                button.configure(state=state)

    @staticmethod
    def _parse_batch_expression(text: str) -> tuple[str, int] | None:
        """Parse i-based expressions or a fixed, one-based frame number."""
        match = re.fullmatch(r"\s*i\s*(?:([+-])\s*(\d+))?\s*", text or "")
        if match is not None:
            sign, magnitude = match.groups()
            if sign is None:
                return "relative", 0
            offset = int(magnitude)
            return "relative", offset if sign == "+" else -offset
        if re.fullmatch(r"\s*\d+\s*", text or ""):
            return "constant", int(text.strip())
        return None

    @staticmethod
    def _expression_frame_number(expression: tuple[str, int],
                                 loop_index: int) -> int:
        kind, value = expression
        return loop_index + value if kind == "relative" else value

    @staticmethod
    def _build_batch_indices(start: int | None, end: int | None,
                             step: int | None) -> list[int]:
        if start is None or end is None or step is None or step == 0:
            return []
        if (step > 0 and start > end) or (step < 0 and start < end):
            return []
        indices = list(range(start, end + (1 if step > 0 else -1), step))
        # The selected End is always part of the batch, even if Step does not
        # divide the interval exactly.
        if indices and indices[-1] != end:
            indices.append(end)
        return indices

    def _on_batch_range_commit(self, _event=None) -> None:
        if self._validate_loop_fields():
            self._status_var.set("batch range ready")
        else:
            self._status_var.set("correct red batch values")

    def _batch_range(self) -> list[int] | None:
        if not self._validate_loop_fields():
            self._status_var.set("correct red batch values")
            return None
        start = int(self._batch_start_var.get())
        end = int(self._batch_end_var.get())
        step = int(self._batch_step_var.get())
        return self._build_batch_indices(start, end, step)

    def _start_batch(self) -> None:
        if self._frame_index_input_connected:
            self._status_var.set("disconnect index inputs for batch")
            return
        indices = self._batch_range()
        if not indices:
            return
        if not self._confirm_video_jump(int(self._batch_current_var.get())):
            return
        self._stop_play()
        self._batch_indices = indices
        self._batch_position = indices.index(int(self._batch_current_var.get()))
        self._batch_running = True
        self._play_btn_var.set("Stop")
        if self._trigger_linked:
            self._status_var.set("batch waiting for trigger")
        else:
            self._advance_batch()

    def _stop_batch(self, status: str = "batch stopped") -> None:
        self._batch_running = False
        self._batch_indices = []
        self._batch_position = 0
        # Cancel any in-flight pattern-mode file wait so it stops retrying.
        self._batch_fetch_token += 1
        self._play_btn_var.set("Play")
        self._cancel_batch_timer()
        self._status_var.set(status)

    def _cancel_batch_timer(self) -> None:
        if self._batch_after_id:
            try:
                self.canvas.after_cancel(self._batch_after_id)
            except tk.TclError:
                pass
            self._batch_after_id = None

    def _schedule_batch_tick(self, delay_ms: int | None = None) -> None:
        if not self._batch_running or self._trigger_linked:
            return
        self._cancel_batch_timer()
        if delay_ms is None:
            try:
                delay_ms = self._fps_interval_ms(self._fps_var.get())
            except ValueError as exc:
                self._stop_batch(str(exc))
                self.set_status("invalid FPS", "#cc0000")
                return
        self._batch_after_id = self.canvas.after(delay_ms, self._advance_batch)

    @staticmethod
    def _fps_interval_ms(value) -> int:
        try:
            fps = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("FPS must be a positive number") from exc
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError("FPS must be a positive number")
        return max(1, int(round(1000.0 / fps)))

    def _advance_batch(self) -> None:
        self._batch_after_id = None
        if not self._batch_running:
            return
        if self._batch_position >= len(self._batch_indices):
            self._stop_batch("batch complete")
            return

        loop_index = self._batch_indices[self._batch_position]
        self._batch_position += 1
        self._batch_current_var.set(str(loop_index))
        is_first_step = self._batch_position == 1
        self._status_var.set(
            f"batch {self._batch_position}/{len(self._batch_indices)}"
        )
        self._emit_batch_frames(
            loop_index,
            emit_trigger=True,
            emit_batch_start=is_first_step,
            on_complete=self._on_batch_step_complete,
        )

    def _on_batch_step_complete(self) -> None:
        """Continue only after the requested frame pair was emitted."""
        if not self._batch_running:
            return
        if self._batch_position >= len(self._batch_indices):
            self._stop_batch("batch complete")
        elif not self._trigger_linked:
            self._schedule_batch_tick()

    def _emit_batch_frames(self, loop_index: int,
                           emit_trigger: bool = False,
                           emit_batch_start: bool = False,
                           on_complete=None) -> None:
        """Decode and publish the two frame expressions for one batch index.

        Frame reads happen through _fetch_frame_async() on a background
        thread, so a Pattern-mode file that an external process has not
        finished writing yet can be waited on / retried without freezing
        the UI. Both frames are pushed together once both are ready.
        """
        frame_1_expression = self._parse_batch_expression(self._batch_frame_1_var.get())
        frame_2_expression = self._parse_batch_expression(self._batch_frame_2_var.get())
        if frame_1_expression is None or frame_2_expression is None:
            self._stop_batch("invalid frame expression")
            return

        frame_1_index = self._expression_frame_number(frame_1_expression, loop_index) - 1
        frame_2_index = self._expression_frame_number(frame_2_expression, loop_index) - 1
        total = self._frame_total()
        if not (0 <= frame_1_index < total and 0 <= frame_2_index < total):
            self._stop_batch("frame expression out of range")
            return

        self._batch_fetch_token += 1
        token = self._batch_fetch_token
        pending: dict[str, np.ndarray | None] = {}

        def _finish() -> None:
            if token != self._batch_fetch_token:
                return   # superseded by a newer request
            if "frame_1" not in pending or "frame_2" not in pending:
                return

            frame_1 = pending["frame_1"]
            frame_2 = pending["frame_2"]
            if frame_1 is None or frame_2 is None:
                self._stop_batch("cannot read batch frame")
                return

            self._current_index = frame_1_index
            self._last_frame_1 = frame_1
            self._last_frame_1_index = frame_1_index
            self._last_frame_2 = frame_2
            self._last_frame_2_index = frame_2_index
            self._slider_var.set(frame_1_index)
            self._idx_var.set(str(frame_1_index + 1))
            self._current_index = frame_1_index
            self._refresh_current_file_display()
            outputs = {
                "image_1": frame_1,
                "image_2": frame_2,
                "frame_index_1": float(frame_1_index + 1),
                "frame_index_2": float(frame_2_index + 1),
                "files": self._absolute_file_paths(),
                "frame_count": float(self._frame_total()),
            }
            if self._is_video():
                self._update_video_labels()
            if emit_trigger:
                # _emit_trigger_pulse() delivers `outputs` on its high edge, so a
                # separate plain push_output() here would recompute downstream twice.
                self._emit_trigger_pulse(outputs, emit_batch_start=emit_batch_start)
            else:
                self.push_output(outputs)
            if on_complete is not None:
                on_complete()

        def _resolve(key: str, index: int) -> None:
            if self._last_frame_1 is not None and self._last_frame_1_index == index:
                pending[key] = self._last_frame_1
                _finish()
                return
            if self._last_frame_2 is not None and self._last_frame_2_index == index:
                pending[key] = self._last_frame_2
                _finish()
                return
            if self._waits_for_files():
                self._status_var.set(f"waiting for frame {index + 1}...")

            def _on_ready(frame, key=key):
                pending[key] = frame
                _finish()

            self._fetch_frame_async(index, _on_ready)

        if frame_2_index == frame_1_index:
            def _on_ready_shared(frame):
                pending["frame_1"] = frame
                pending["frame_2"] = frame
                _finish()

            if self._last_frame_1 is not None and self._last_frame_1_index == frame_1_index:
                _on_ready_shared(self._last_frame_1)
            else:
                if self._waits_for_files():
                    self._status_var.set(f"waiting for frame {frame_1_index + 1}...")
                self._fetch_frame_async(frame_1_index, _on_ready_shared)
        else:
            _resolve("frame_1", frame_1_index)
            _resolve("frame_2", frame_2_index)

    def close_inspector(self) -> None:
        trace_id = getattr(self, "_selector_placeholder_trace", None)
        if trace_id is not None:
            try:
                self._selector_var.trace_remove("write", trace_id)
            except tk.TclError:
                pass
            self._selector_placeholder_trace = None
        self._selector_frame = self._selector_entry = self._selector_info_label = None
        super().close_inspector()
        self._pat_entry = None
        self._pattern_wait_entry = None
        self._folder_label = None
        self._slider = None
        self._idx_entry = None
        self._pat_frame = None
        self._fld_frame = None
        self._batch_frame = None
        self._images_panel = None
        self._video_panel = None
        self._video_access_label = None
        self._loop_entries = {}
        self._batch_first_btn = None
        self._batch_prev_btn = None
        self._batch_current_btn = None
        self._batch_next_btn = None
        self._batch_last_btn = None
        self._batch_play_btn = None

    def on_resize(self, old_width: int, old_height: int,
                  new_width: int, new_height: int) -> None:
        super().on_resize(old_width, old_height, new_width, new_height)

        if self._slider is not None and self._slider.winfo_exists():
            self._slider.configure(length=max(80, int(new_width) - 20))
        if self._folder_label is not None and self._folder_label.winfo_exists():
            self._folder_label.configure(wraplength=max(80, int(new_width) - 20))

    # ── source switching ──────────────────────────────────────────

    def _on_source_change(self, user: bool = False) -> None:
        """Show the panel of the selected source. user=True (radio button):
        restart the batch for the new source and send its first pair."""
        video = self._is_video()
        self._update_selector_info()          # "images" <-> "frames", other total
        for panel, show in ((self._images_panel, not video), (self._video_panel, video)):
            if panel is None or not panel.winfo_exists():
                continue
            if show and panel.winfo_manager() != "pack":
                panel.pack(fill="x")
            elif not show and panel.winfo_manager() == "pack":
                panel.pack_forget()
        if not user:
            return
        self._stop_batch("ready")
        self._reset_batch_defaults(self._frame_total())
        self._refresh_current_file_display()
        if self._frame_total() and self._validate_loop_fields():
            self._emit_batch_frames(1)

    def _on_method_change(self) -> None:          # legacy name (old projects / callers)
        self._on_source_change(user=False)

    def _reset_batch_defaults(self, n: int) -> None:
        """Batch fields for a new source: i = 1..n, Frame 1 = 1, Frame 2 = i."""
        self._batch_start_var.set("1")
        self._batch_end_var.set(str(max(1, n)))
        self._batch_step_var.set("1")
        self._batch_current_var.set("1")
        self._batch_frame_1_var.set("1")
        self._batch_frame_2_var.set("i")

    def _waits_for_files(self) -> bool:
        return (not self._is_video()) and bool(self._wait_files_var.get())

    @staticmethod
    def _show_warning_with_terminal_log(title: str, message: str) -> None:
        """Show warning dialog and mirror message to terminal for easy copy."""
        print(f"[ImageSequenceNode][Warning] {title}: {message}", flush=True)
        messagebox.showwarning(title, message)

    def _dialog_parent(self):
        win = getattr(self, "_inspector_win", None)
        try:
            if win is not None and win.winfo_exists():
                return win
        except tk.TclError:
            pass
        return self.canvas.winfo_toplevel()

    # ── image list: summary ───────────────────────────────────────

    _MISSING_CHECK_MAX = 20000        # existence is checked for at most this many files

    def _update_list_summary(self) -> None:
        """'523 images: a.png ... b.png / in <folder> / all exist' (or how
        many are missing and whether the node waits for them)."""
        if not self._is_video():
            self._update_selector_info()
        n = len(self._file_paths)
        if not n:
            self._list_summary_var.set(
                "No images yet. Make the list with Folder..., Files... or Numbered names...")
            return
        absp = self._absolute_file_paths()
        key = (n, absp[0], absp[-1], id(self._file_paths))
        if self._missing_cache[0] != key:
            checked = absp[: self._MISSING_CHECK_MAX]
            missing = self._count_missing(checked)
            self._missing_cache = (key, missing, len(checked) == n)
        _key, missing, complete = self._missing_cache
        folders = {os.path.dirname(path) for path in absp[:2000]}
        where = next(iter(folders)) if len(folders) == 1 else f"{len(folders)}+ folders"
        if missing == 0:
            status = "all exist" if complete else f"first {self._MISSING_CHECK_MAX} exist"
        elif self._wait_files_var.get():
            status = f"{missing} not there yet -- the node will wait for them"
        else:
            status = (f"{missing} not there yet -- turn on 'Wait for files...' if another program "
                      f"will write them")
        self._list_summary_var.set(
            f"{n} images:  {Path(absp[0]).name}  ...  {Path(absp[-1]).name}\nin {where}\n{status}")

    @staticmethod
    def _count_missing(paths: list[str]) -> int:
        """How many of `paths` do not exist. Each FOLDER is listed once
        (os.scandir) instead of asking the file system about every file --
        thousands of os.path.exists calls took seconds per project load on
        external / network drives."""
        by_folder: dict[str, list[str]] = {}
        for path in paths:
            folder, name = os.path.split(path)
            by_folder.setdefault(folder, []).append(name)
        missing = 0
        for folder, names in by_folder.items():
            if len(names) < 8:                       # a few files: stat is cheaper
                missing += sum(1 for name in names if not os.path.isfile(os.path.join(folder, name)))
                continue
            try:
                with os.scandir(folder or ".") as it:
                    present = {os.path.normcase(e.name) for e in it if not e.is_dir()}
            except OSError:
                missing += len(names)                # folder does not exist (yet)
                continue
            missing += sum(1 for name in names if os.path.normcase(name) not in present)
        return missing

    # ── image list: folder / files / numbered names ───────────────

    def _browse_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select image folder", parent=self._dialog_parent())
        if not folder:
            return

        base = self._get_project_base()
        paths = sorted(
            str(p) for p in Path(folder).iterdir()
            if p.suffix.lower() in self._IMG_EXTS)

        if not paths:
            self._show_warning_with_terminal_log(
                "No Images",
                f"No supported image files found in:\n{folder}")
            return

        self._folder_var.set(self._to_relative(self._to_absolute(folder, base), base))
        self._set_file_paths(paths)

    def _browse_files(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(self._IMG_EXTS))
        names = filedialog.askopenfilenames(
            title="Select image files", parent=self._dialog_parent(),
            filetypes=[("Image files", exts), ("All files", "*.*")])
        if not names:
            return
        self._set_file_paths(sorted(names))

    @staticmethod
    def _numbered_paths(pattern: str, first: int, last: int, step: int) -> list[str]:
        """Names for numbers first..last (inclusive) by step, e.g. img_%04d.png.
        Raises ValueError for a pattern without exactly one integer field."""
        if step == 0:
            raise ValueError("Step must not be 0.")
        if (step > 0 and first > last) or (step < 0 and first < last):
            raise ValueError("The number range is empty (check First / Last / Step).")
        fields = re.findall(r"%[-+ 0#]*\d*[diu]", pattern)
        if len(fields) != 1 or pattern.count("%") != 1:
            raise ValueError("The pattern needs exactly one number field such as %d or %04d "
                             "(and no other % sign).")
        stop = last + (1 if step > 0 else -1)
        return [pattern % k for k in range(first, stop, step)]

    def _open_pattern_dialog(self) -> None:
        """Make the list from numbered names (the files may not exist yet)."""
        top = tk.Toplevel(self._dialog_parent())
        top.title("Numbered names")
        top.transient(self._dialog_parent())
        f = tk.Frame(top, padx=10, pady=8)
        f.pack(fill="both", expand=True)
        tk.Label(f, justify="left", anchor="w", wraplength=520, font=("Arial", 8), fg="#444444", text=(
            "Make the list from a name PATTERN and a NUMBER RANGE. Use one printf number field: "
            "%d (1, 2, ...) or %04d (0001, 0002, ...).  Example: C:/data/img_%04d.png with numbers "
            "1 to 500 gives img_0001.png ... img_0500.png.  The files may not exist yet: another "
            "program can write them later.")).grid(row=0, column=0, columnspan=4, sticky="w")
        tk.Label(f, text="Pattern:", font=("Arial", 9)).grid(row=1, column=0, sticky="w", pady=(8, 2))
        tk.Entry(f, textvariable=self._pattern_var, width=58, font=("Arial", 9)).grid(
            row=1, column=1, columnspan=2, sticky="ew", pady=(8, 2))

        def browse_dir():
            d = filedialog.askdirectory(title="Folder of the numbered files", parent=top)
            if d:
                name = Path(self._pattern_var.get().strip()).name or "img_%04d.png"
                self._pattern_var.set(str(Path(d) / name))
        tk.Button(f, text="Folder...", font=("Arial", 8), command=browse_dir).grid(
            row=1, column=3, sticky="w", padx=(4, 0), pady=(8, 2))
        rng = tk.Frame(f)
        rng.grid(row=2, column=0, columnspan=4, sticky="w", pady=(2, 2))
        for label, var in (("First number:", self._pattern_first_var), ("Last number:", self._pattern_last_var),
                           ("Step:", self._pattern_step_var)):
            tk.Label(rng, text=label, font=("Arial", 9)).pack(side="left", padx=(0, 2))
            tk.Entry(rng, textvariable=var, width=8, font=("Arial", 9)).pack(side="left", padx=(0, 12))
        preview = tk.StringVar(value="")
        tk.Label(f, textvariable=preview, justify="left", anchor="w", wraplength=520,
                 font=("Arial", 8, "bold"), fg="#2f3e5c").grid(row=3, column=0, columnspan=4, sticky="w",
                                                              pady=(4, 6))
        state = {"paths": None}

        def update(*_a):
            try:
                paths = self._numbered_paths(self._pattern_var.get().strip(), int(self._pattern_first_var.get()),
                                             int(self._pattern_last_var.get()), int(self._pattern_step_var.get()))
            except (ValueError, TypeError) as e:
                state["paths"] = None
                preview.set(f"Not valid: {e}")
                return
            state["paths"] = paths
            base = self._get_project_base()
            sample = paths[:5000]
            exist = sum(1 for path in sample if os.path.exists(self._to_absolute(path, base)))
            checked = "" if len(sample) == len(paths) else f" (checked the first {len(sample)})"
            preview.set(f"-> {len(paths)} names:  {Path(paths[0]).name}  ...  {Path(paths[-1]).name}\n"
                        f"   exist now: {exist}{checked}, not yet: {len(sample) - exist}")
        for var in (self._pattern_var, self._pattern_first_var, self._pattern_last_var, self._pattern_step_var):
            var.trace_add("write", update)
        update()

        def create():
            paths = state["paths"]
            if not paths:
                return
            # Names of files still to come: turn waiting on BEFORE the list is
            # set, so the first frame is waited for instead of failing to read.
            base = self._get_project_base()
            if any(not os.path.exists(self._to_absolute(path, base)) for path in paths[:5000]):
                self._wait_files_var.set(True)
            self._set_file_paths(paths)
            top.destroy()
        btns = tk.Frame(f)
        btns.grid(row=4, column=0, columnspan=4, sticky="e")
        tk.Button(btns, text="Make the list", font=("Arial", 9, "bold"), command=create).pack(side="left")
        tk.Button(btns, text="Cancel", font=("Arial", 9), command=top.destroy).pack(side="left", padx=(6, 0))
        top.bind("<Escape>", lambda _e: top.destroy())

    # ── video ─────────────────────────────────────────────────────

    def _browse_video(self) -> None:
        exts = " ".join(f"*{e}" for e in VIDEO_EXTS)
        path = filedialog.askopenfilename(title="Open video", parent=self._dialog_parent(),
                                          filetypes=[("Video files", exts), ("All files", "*.*")])
        if path:
            self._open_video(path, reset_batch=True)

    def _close_video(self) -> None:
        reader, self._video_reader = self._video_reader, None
        if reader is not None:
            reader.close()

    def _open_video(self, path: str, reset_batch: bool = True, show_errors: bool = True) -> bool:
        base = self._get_project_base()
        abs_path = self._to_absolute(path, base)
        self._stop_batch("ready")
        self._close_video()
        try:
            reader = VideoFrameReader(abs_path)
        except Exception as e:
            self._status_var.set("cannot open video")
            self._video_name_var.set(f"Cannot open: {abs_path}")
            if show_errors:
                messagebox.showerror("Open video", f"{e}", parent=self._dialog_parent())
            return False
        self._video_reader = reader
        self._invalidate_frame_cache()
        self._video_path_var.set(self._to_relative(abs_path, base))
        self._source_var.set("video")
        self._on_source_change(user=False)
        if reset_batch:
            self._reset_batch_defaults(self._frame_total())
        # Count the frames exactly + fingerprint them, in the background.
        threading.Thread(target=reader.build_index, daemon=True, name="video-index").start()
        self._poll_video_index(reader)
        self._update_video_labels()
        self._status_var.set(f"video: {Path(abs_path).name}")
        if self._validate_loop_fields():
            self._emit_batch_frames(int(self._batch_current_var.get()))
        return True

    def _poll_video_index(self, reader) -> None:
        if reader is not self._video_reader or self._destroyed:
            return
        self._update_video_labels()
        if reader.index_ready or reader.index_error:
            exact = reader.frame_count
            if exact is not None and exact != reader.reported_count:
                # The header's count was wrong: an End that meant "the last
                # (selected) frame" follows the exact count.
                try:
                    old_sel = self._parse_selector(self._selector_applied, reader.reported_count)[0]
                except ValueError:
                    old_sel = None
                old_n = reader.reported_count if old_sel is None else len(old_sel)
                if self._batch_end_var.get().strip() == str(old_n):
                    self._batch_end_var.set(str(self._frame_total()))
            self._validate_loop_fields()
            return
        try:
            self.canvas.after(300, lambda: self._poll_video_index(reader))
        except tk.TclError:
            pass

    def _update_video_labels(self) -> None:
        if self._is_video():
            self._update_selector_info()      # the frame count grows while indexing
        r = self._video_reader
        if r is None:
            self._video_name_var.set("No video. Use Open video...")
            self._video_info_var.set("")
            self._video_access_var.set("")
            return
        self._video_name_var.set(r.path)
        count = (f"{r.frame_count} frames (counted)" if r.index_ready else
                 f"~{r.reported_count} frames (from the file header; counting ... "
                 f"{r.index_progress * 100:.0f}%)")
        dur = f"  |  {r.count_known() / r.fps:.2f} s" if r.fps > 0 else ""
        self._video_info_var.set(f"{r.width} x {r.height}  |  {r.fps:.3f} fps  |  codec {r.codec}  |  {count}{dur}")
        if r.index_error:
            text, color = (f"Frame access: the index failed ({r.index_error}); every jump is decoded "
                           f"in order from the start (exact, but slow).", "#b36b00")
        elif not r.index_ready:
            text, color = ("Frame access: building the frame index ... until it is ready, jumps are "
                           "decoded in order (exact).", "#666666")
        elif not r.seek_reliable():
            text, color = (f"Frame access: seeking is NOT reliable for this file ({r.seek_failures} of "
                           f"{r.seeks} jumps failed the check); jumps are decoded from the start (exact, "
                           f"slow). Play / Next are fast. Extracting the frames to images avoids this.",
                           "#b36b00")
        else:
            checked = f" ({r.seeks - r.seek_failures}/{r.seeks} jumps verified)" if r.seeks else ""
            text, color = (f"Frame access: exact. Jumps use seeking, each frame checked against the "
                           f"index{checked}.", "#2e7d32")
        self._video_access_var.set(text)
        if self._video_access_label is not None and self._video_access_label.winfo_exists():
            self._video_access_label.configure(fg=color)

    def _confirm_video_jump(self, loop_index: int) -> bool:
        """Before a UI-initiated step: if a frame will need slow in-order
        decoding of many frames, explain it once per video and ask."""
        r = self._video_reader
        if not self._is_video() or r is None:
            return True
        cost = 0
        for var in (self._batch_frame_1_var, self._batch_frame_2_var):
            expr = self._parse_batch_expression(var.get())
            if expr is not None:
                src = self._src_index(self._expression_frame_number(expr, loop_index) - 1)
                if src >= 0:
                    cost = max(cost, r.slow_cost(src))
        if cost <= r.SLOW_WARN_FRAMES or r.path in self._video_warned:
            return True
        self._video_warned.add(r.path)
        reason = ("the frame index of this video is still being built" if not r.index_ready
                  else "seeking is not reliable for this video")
        return messagebox.askokcancel(
            "Video: slow jump",
            f"The requested frame is far from the current position and {reason}.\n\n"
            f"To be sure it is exactly the right frame, the video will be decoded in order: "
            f"about {cost} frames, which can take a while.\n\n"
            f"Playing / Next (in order) is always fast. For frequent jumps use "
            f"'Extract frames to image files'.\n\nThis message is shown once per video. Continue?",
            icon="warning", parent=self._dialog_parent())

    def _open_extract_dialog(self) -> None:
        """Write the video's frames as image files, then optionally use them."""
        r = self._video_reader
        if r is None:
            messagebox.showinfo("Extract frames", "Open a video first.", parent=self._dialog_parent())
            return
        stem = Path(r.path).stem
        top = tk.Toplevel(self._dialog_parent())
        top.title("Extract video frames to image files")
        top.transient(self._dialog_parent())
        f = tk.Frame(top, padx=10, pady=8)
        f.pack(fill="both", expand=True)
        folder_var = tk.StringVar(value=str(Path(r.path).parent / f"{stem}_frames"))
        name_var = tk.StringVar(value=f"{stem}_%06d.png")
        first_var = tk.StringVar(value="1")
        last_var = tk.StringVar(value=str(r.count_known()))
        step_var = tk.StringVar(value="1")
        quality_var = tk.StringVar(value="95")
        tk.Label(f, justify="left", anchor="w", wraplength=540, font=("Arial", 8), fg="#444444", text=(
            "Frames are read IN ORDER (always exact). The number in each name is the video frame "
            "number (1 = first frame). .png = lossless (recommended for measurements), .jpg = smaller "
            "files.")).grid(row=0, column=0, columnspan=4, sticky="w")
        tk.Label(f, text="Folder:").grid(row=1, column=0, sticky="w", pady=(8, 2))
        tk.Entry(f, textvariable=folder_var, width=60).grid(row=1, column=1, columnspan=2, sticky="ew", pady=(8, 2))
        tk.Button(f, text="...", command=lambda: folder_var.set(
            filedialog.askdirectory(parent=top) or folder_var.get())).grid(row=1, column=3, padx=(4, 0), pady=(8, 2))
        tk.Label(f, text="File names:").grid(row=2, column=0, sticky="w", pady=2)
        tk.Entry(f, textvariable=name_var, width=36).grid(row=2, column=1, sticky="w", pady=2)
        tk.Label(f, text="JPG quality:").grid(row=2, column=2, sticky="e", pady=2)
        tk.Entry(f, textvariable=quality_var, width=5).grid(row=2, column=3, sticky="w", pady=2)
        rr = tk.Frame(f)
        rr.grid(row=3, column=0, columnspan=4, sticky="w", pady=2)
        for label, var in (("Frames from:", first_var), ("to:", last_var), ("every:", step_var)):
            tk.Label(rr, text=label).pack(side="left", padx=(0, 2))
            tk.Entry(rr, textvariable=var, width=8).pack(side="left", padx=(0, 10))
        progress = tk.StringVar(value="")
        tk.Label(f, textvariable=progress, anchor="w", justify="left", font=("Arial", 8, "bold"),
                 fg="#2f3e5c").grid(row=4, column=0, columnspan=4, sticky="w", pady=(6, 4))
        btns = tk.Frame(f)
        btns.grid(row=5, column=0, columnspan=4, sticky="e")
        job = {"cancel": threading.Event(), "done": 0, "total": 0, "error": None,
               "finished": False, "paths": []}

        def worker(abs_video, out_dir, pattern, first, last, step, quality):
            try:
                os.makedirs(out_dir, exist_ok=True)
                cap = cv2.VideoCapture(abs_video)
                wanted = set(range(first, last + 1, step))
                job["total"] = len(wanted)
                ext = Path(pattern).suffix.lower()
                params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in (".jpg", ".jpeg") else []
                k = 0
                while k < last and not job["cancel"].is_set():
                    ok = cap.grab()
                    if not ok:
                        break
                    k += 1
                    if k in wanted:
                        ok, frame = cap.retrieve()
                        if not ok:
                            break
                        out = os.path.join(out_dir, pattern % k)
                        ok, buf = cv2.imencode(ext, frame, params)
                        if not ok:
                            raise IOError(f"cannot encode {out}")
                        buf.tofile(out)                      # also works for non-ASCII paths
                        job["paths"].append(out)
                        job["done"] += 1
                cap.release()
            except Exception as e:
                job["error"] = f"{type(e).__name__}: {e}"
            finally:
                job["finished"] = True

        def poll():
            if not top.winfo_exists():
                return
            progress.set(f"Written {job['done']} / {job['total']} images ...")
            if not job["finished"]:
                top.after(200, poll)
                return
            start_btn.configure(state=tk.NORMAL)
            if job["error"]:
                progress.set(f"Failed: {job['error']}")
                return
            msg = ("Stopped" if job["cancel"].is_set() else "Done") + f": {job['done']} images written."
            progress.set(msg)
            if job["paths"] and messagebox.askyesno(
                    "Extract frames", f"{msg}\n\nUse these images as this node's source now?", parent=top):
                self._set_file_paths(job["paths"])
                top.destroy()

        def start():
            try:
                pattern = name_var.get().strip()
                self._numbered_paths(pattern, 1, 1, 1)           # validates the number field
                first, last, step = int(first_var.get()), int(last_var.get()), int(step_var.get())
                quality = max(1, min(100, int(quality_var.get())))
                if Path(pattern).suffix.lower() not in self._IMG_EXTS:
                    raise ValueError("File names must end with an image extension such as .png or .jpg.")
                if not (1 <= first <= last) or step < 1:
                    raise ValueError("Use 1 <= from <= to and every >= 1.")
            except (ValueError, TypeError) as e:
                progress.set(f"Not valid: {e}")
                return
            job.update(done=0, total=0, error=None, finished=False, paths=[])
            job["cancel"].clear()
            start_btn.configure(state=tk.DISABLED)
            threading.Thread(target=worker, daemon=True, args=(r.path, folder_var.get().strip(), pattern,
                                                               first, last, step, quality)).start()
            poll()
        start_btn = tk.Button(btns, text="Extract", font=("Arial", 9, "bold"), command=start)
        start_btn.pack(side="left")
        tk.Button(btns, text="Stop", command=job["cancel"].set).pack(side="left", padx=(6, 0))
        tk.Button(btns, text="Close", command=lambda: (job["cancel"].set(), top.destroy())).pack(
            side="left", padx=(6, 0))

    # ── help ──────────────────────────────────────────────────────

    def _open_help(self) -> None:
        if self._help_popup is not None:
            try:
                if self._help_popup.winfo_exists():
                    self._help_popup.lift()
                    return
            except tk.TclError:
                pass
        popup = tk.Toplevel(self._dialog_parent())
        popup.title("Image Sequence - Help")
        popup.geometry("760x640")
        body = tk.Frame(popup, padx=8, pady=8)
        body.pack(fill="both", expand=True)
        txt = tk.Text(body, font=("Courier", 9), wrap=tk.WORD, relief=tk.FLAT)
        vsb = ttk.Scrollbar(body, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", self.HELP_TEXT)
        txt.configure(state="disabled")
        popup.bind("<Escape>", lambda _e: popup.destroy())
        self._help_popup = popup

    @staticmethod
    def _popup_position_near_pointer(pointer_x: int, pointer_y: int, width: int, height: int,
                                     screen_w: int, screen_h: int, gap: int = 16) -> tuple[int, int]:
        """Top-left corner for a popup beside the mouse pointer: below-right
        of it, flipped to the left / above when it would leave the screen."""
        x = pointer_x + gap
        if x + width > screen_w:
            x = pointer_x - gap - width
        y = pointer_y + gap
        if y + height > screen_h:
            y = pointer_y - gap - height
        return max(0, x), max(0, y)

    def _show_full_list(self) -> None:
        """Read-only view of the WHOLE list with 1-based numbers (what the
        selector refers to). Text can be selected and copied (Ctrl+C,
        Ctrl+A); unselected lines are greyed out."""
        if not self._file_paths:
            messagebox.showinfo(
                "Full List",
                "The list is empty. Make it with Folder..., Files... or Numbered names... first.",
                parent=self._dialog_parent())
            return
        n = len(self._file_paths)
        sel = self._selection()[0]
        win = tk.Toplevel()
        win.title(f"Full List ({n} images" + ("" if sel is None else f", {len(sel)} selected") + ")")
        try:
            px, py = win.winfo_pointerxy()
            x, y = self._popup_position_near_pointer(
                px, py, 760, 520, win.winfo_screenwidth(), win.winfo_screenheight())
            win.geometry(f"760x520+{x}+{y}")
        except tk.TclError:
            win.geometry("760x520")
        tk.Label(win, font=("Arial", 9), pady=4, text=(
            "Number (1-based, as used by the Selector) and path. Read-only: select + Ctrl+C to copy."
            + ("" if sel is None else "  Grey = not selected."))).pack()
        frame = tk.Frame(win)
        frame.pack(fill="both", expand=True)
        txt = tk.Text(frame, font=("Courier", 9), wrap=tk.NONE)
        vsb = tk.Scrollbar(frame, orient="vertical", command=txt.yview)
        hsb = tk.Scrollbar(frame, orient="horizontal", command=txt.xview)
        txt.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        hsb.pack(side="bottom", fill="x")
        vsb.pack(side="right", fill="y")
        txt.pack(fill="both", expand=True)
        width = len(str(n))
        txt.insert("1.0", "\n".join(f"{k + 1:>{width}}  {p}" for k, p in enumerate(self._file_paths)))
        if sel is not None:
            txt.tag_configure("unselected", foreground="#a0a0a0")
            chosen = set(sel)
            k = 0
            while k < n:                       # tag runs of unselected lines, not single lines
                if k in chosen:
                    k += 1
                    continue
                j = k
                while j + 1 < n and (j + 1) not in chosen:
                    j += 1
                txt.tag_add("unselected", f"{k + 1}.0", f"{j + 1}.end")
                k = j + 1

        # Read-only but still selectable / copyable: a disabled Text does
        # not take keyboard focus on every platform, so block edits instead.
        def _block(event):
            ctrl = bool(event.state & 0x4)
            if ctrl and event.keysym.lower() in ("c", "a", "insert", "home", "end"):
                return None
            if event.keysym in ("Left", "Right", "Up", "Down", "Prior", "Next", "Home", "End",
                                "Shift_L", "Shift_R", "Control_L", "Control_R"):
                return None
            return "break"

        def _select_all(_e):
            txt.tag_add("sel", "1.0", "end-1c")
            return "break"
        txt.bind("<Key>", _block)
        for seq in ("<<Paste>>", "<<Cut>>", "<<Clear>>", "<<PasteSelection>>"):
            txt.bind(seq, lambda _e: "break")
        txt.bind("<Control-a>", _select_all)
        txt.bind("<Control-A>", _select_all)
        txt.focus_set()
        btn_frame = tk.Frame(win)
        btn_frame.pack(fill="x", pady=4)
        tk.Button(btn_frame, text="Close", command=win.destroy, font=("Arial", 9)).pack(side="right", padx=8)
        win.bind("<Escape>", lambda _e: win.destroy())

    def _edit_file_list(self, *, preserve_batch: bool = False) -> None:
        """
        Open a popup text editor showing the current file list,
        one path per line. Experienced users can add, remove, or
        reorder paths. Changes take effect when they click Apply.
        preserve_batch keeps the Batch sequence fields on Apply (used when
        the list was generated from those fields in Pattern mode).
        """
        if not self._file_paths:
            messagebox.showinfo(
                "Edit Selected List",
                "The list is empty. Make it with Folder..., Files... or Numbered names... first.",
                parent=self._dialog_parent())
            return

        selected = self._selected_file_paths()
        has_selector = self._selection()[0] is not None
        win = tk.Toplevel()
        win.title("Edit Selected List" + (f" ({len(selected)} of {len(self._file_paths)})"
                                          if has_selector else f" ({len(selected)})"))
        try:
            px, py = win.winfo_pointerxy()
            x, y = self._popup_position_near_pointer(
                px, py, 700, 500, win.winfo_screenwidth(), win.winfo_screenheight())
            win.geometry(f"700x500+{x}+{y}")
        except tk.TclError:
            win.geometry("700x500")

        tk.Label(win,
                 text="One file path per line. "
                      "Paths relative to project file or absolute.",
                 font=("Arial", 9), pady=4).pack()
        if has_selector:
            tk.Label(win, fg="#b36b00", font=("Arial", 8), wraplength=660, justify="left", text=(
                f"Only the {len(selected)} SELECTED files are shown. Apply makes these lines the new "
                f"list and clears the selector -- the {len(self._file_paths) - len(selected)} "
                f"unselected files are dropped from the list.")).pack(padx=8)

        txt = tk.Text(win, font=("Courier", 9), wrap=tk.NONE)
        vsb = tk.Scrollbar(win, orient="vertical",
                           command=txt.yview)
        hsb = tk.Scrollbar(win, orient="horizontal",
                           command=txt.xview)
        txt.configure(yscrollcommand=vsb.set,
                      xscrollcommand=hsb.set)
        hsb.pack(side="bottom", fill="x")
        vsb.pack(side="right",  fill="y")
        txt.pack(fill="both", expand=True)

        txt.insert("1.0", "\n".join(selected))

        def _apply():
            base = self._get_project_base()
            lines = [l.strip()
                     for l in txt.get("1.0", tk.END).splitlines()
                     if l.strip()]
            # With "Wait for files" on, missing paths are expected (another
            # program may still write them); otherwise ask before applying.
            if not self._wait_files_var.get():
                bad = [l for l in lines
                       if not Path(self._to_absolute(l, base)).exists()]
                if bad:
                    sample = "\n".join(bad[:5])
                    more   = f"\n... and {len(bad)-5} more" \
                             if len(bad) > 5 else ""
                    if not messagebox.askyesno(
                            "Missing Files",
                            f"{len(bad)} paths do not exist:\n"
                            f"{sample}{more}\n\n"
                            f"Apply anyway?"):
                        return
            if has_selector:
                # The edited lines ARE the selection now: every line is used.
                self._selector_applied = ""
                self._selector_var.set("")
            self._set_file_paths(lines, reset_batch=not preserve_batch)
            win.destroy()

        btn_frame = tk.Frame(win)
        btn_frame.pack(fill="x", pady=4)
        tk.Button(btn_frame, text="Apply",
                  command=_apply,
                  font=("Arial", 9)).pack(side="right", padx=8)
        tk.Button(btn_frame, text="Cancel",
                  command=win.destroy,
                  font=("Arial", 9)).pack(side="right")

    # ── file list management ──────────────────────────────────────

    def _set_file_paths(self, paths: list[str], *, reset_batch: bool = True) -> None:
        # Cancel any in-flight background frame fetch/wait tied to the old list.
        self._batch_fetch_token += 1
        base = self._get_project_base()
        # Keep canonical stored paths project-relative when possible.
        normalized = self._normalize_paths(paths, base)

        self._file_paths    = normalized
        self._current_index = 0
        self._invalidate_frame_cache()
        self._stop_batch("ready")
        self._source_var.set("images")
        self._on_source_change(user=False)

        # The selector (if any) applies to the new list too.
        n = self._frame_total()
        self._update_selector_info()
        if reset_batch:
            # Default pair 1 / i: frame 1 is the first frame, frame 2 runs over
            # every frame, so End is the last frame.
            self._reset_batch_defaults(n)
        if self._slider is not None and self._slider.winfo_exists():
            self._slider.configure(from_=0, to=max(0, n - 1))
        self._slider_var.set(0)
        self._count_var.set(f"/ {n}")
        self._info_var.set(
            f"{len(normalized)} files  "
            f"{Path(normalized[0]).name} ... {Path(normalized[-1]).name}" if normalized else "no files")
        self._update_list_summary()
        self._idx_var.set("1")
        self._fname_var.set("")
        self._size_var.set("")
        # Publish the configured pair immediately (Frame 1=1, Frame 2=i by
        # default; at i=1 both pins carry frame 1, the reference frame).
        if self._validate_loop_fields():
            try:
                current = int(self._batch_current_var.get())
            except ValueError:
                current = 1
            self._emit_batch_frames(current)
        else:
            self._load_and_push(0)

    # ── frame loading ─────────────────────────────────────────────

    @staticmethod
    def _decode_frame(abs_path: str) -> np.ndarray | None:
        frame = cv2.imread(abs_path)
        if frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _frame_label(self, index: int) -> str:
        """Display name of a position: file name, or 'frame k  (t s)' for a
        video (k = the frame number in the whole video)."""
        src = self._src_index(index)
        if src < 0:
            return ""
        prefix = "" if self._selection()[0] is None else f"#{index + 1}: "
        if self._is_video():
            r = self._video_reader
            t = r.time_of(src) if r is not None else None
            return f"{prefix}frame {src + 1}" + (f"  (t = {t:.3f} s)" if t is not None else "")
        if src < len(self._file_paths):
            return prefix + Path(self._file_paths[src]).name
        return ""

    def _read_index_once(self, index: int):
        """Decode one position now (any thread). None if it cannot be read."""
        src = self._src_index(index)
        if src < 0:
            return None
        if self._is_video():
            reader = self._video_reader
            return reader.read(src) if reader is not None else None
        if src >= len(self._file_paths):
            return None
        return self._decode_frame(self._to_absolute(self._file_paths[src], self._get_project_base()))

    def _request_decode(self, index: int) -> None:
        total = self._frame_total()
        if not total:
            return

        index = max(0, min(index, total - 1))
        self._current_index = index
        label = self._frame_label(index)

        with self._decode_lock:
            self._decode_request_id += 1
            req_id = self._decode_request_id
            self._decode_pending_req = (req_id, label, index)

        self._try_start_decode()

    def _try_start_decode(self) -> None:
        with self._decode_lock:
            if self._destroyed:
                return
            if self._decode_future is not None:
                return
            req = self._decode_pending_req
            if req is None:
                return
            self._decode_pending_req = None

            req_id, label, index = req
            fut = self._DECODE_POOL.submit(self._read_index_once, index)
            self._decode_future = fut

        def _done_cb(future, rid=req_id, lab=label, idx=index):
            try:
                self.canvas.after(0, lambda: self._on_decode_done(rid, lab, idx, future))
            except tk.TclError:
                pass

        fut.add_done_callback(_done_cb)

    def _on_decode_done(self, req_id: int, label: str, index: int, future) -> None:
        with self._decode_lock:
            self._decode_future = None
            destroyed = self._destroyed

        if destroyed:
            return

        try:
            frame_rgb = future.result()
        except Exception:
            frame_rgb = None

        with self._decode_lock:
            latest_req_id = self._decode_request_id

        # Drop stale results when a newer decode request already exists.
        if req_id != latest_req_id:
            self._try_start_decode()
            return

        if frame_rgb is None:
            self._info_var.set(f"cannot read: {label}")
            self._size_var.set("read error")
            self._try_start_decode()
            return

        self._last_frame = frame_rgb
        self._last_frame_index = index

        # update slider and entry without triggering callbacks
        self._slider_var.set(index)
        self._idx_var.set(str(index + 1))
        self._refresh_current_file_display()

        # push to downstream
        self.push_output({
            "image_1":       frame_rgb,
            "frame_index_1": float(index + 1),
            "files": self._absolute_file_paths(),
            "frame_count": float(self._frame_total()),
        })

        self._try_start_decode()

    def _load_and_push(self, index: int) -> None:
        self._request_decode(index)

    # ── slider callbacks ──────────────────────────────────────────

    def _refresh_current_file_display(self) -> None:
        """Keep the inspector's 'Now sending' text aligned with the active frame pair."""
        total = self._frame_total()
        if not total:
            self._fname_var.set("frame 1: \nframe 2: ")
            self._size_var.set("")
            return

        idx = max(0, min(self._current_index, total - 1))
        name_2 = ""
        if 0 <= self._last_frame_2_index < total and self._last_frame_2 is not None:
            name_2 = self._frame_label(self._last_frame_2_index)
        self._fname_var.set(f"frame 1: {self._frame_label(idx)}\nframe 2: {name_2}")

        frame = self._last_frame if self._last_frame_index == idx else None
        if frame is None and self._last_frame_1 is not None and self._last_frame_1_index == idx:
            frame = self._last_frame_1
        if frame is None and self._last_frame_2 is not None and self._last_frame_2_index == idx:
            frame = self._last_frame_2
        if frame is None:
            self._size_var.set("")
            return
        h, w = frame.shape[:2]
        self._size_var.set(f"{w} x {h}")

    def _on_slider_move(self, val) -> None:
        """Called continuously while dragging — update entry only."""
        if self._frame_index_input_connected:
            self._slider_var.set(self._current_index)
            return
        idx = int(float(val))
        self._idx_var.set(str(idx + 1))
        self._current_index = idx
        self._refresh_current_file_display()
        self._slider_dragging = True

    def _on_slider_release(self, event) -> None:
        """Load and push only when user releases the slider."""
        if self._frame_index_input_connected:
            return
        self._slider_dragging = False
        if self._frame_total():
            self._load_and_push(self._slider_var.get())

    # ── frame index entry ─────────────────────────────────────────

    def _on_idx_entry(self, event) -> None:
        if self._frame_index_input_connected:
            self._idx_var.set(str(self._current_index + 1))
            return
        try:
            idx_1based = int(self._idx_var.get())
        except ValueError:
            return
        if self._frame_total():
            idx_0based = max(0, idx_1based - 1)
            self._load_and_push(idx_0based)

    # ── playback controls ─────────────────────────────────────────

    def _go_first(self) -> None:
        self._move_batch_current("first")

    def _go_last(self) -> None:
        self._move_batch_current("last")

    def _prev_frame(self) -> None:
        self._move_batch_current("prev")

    def _next_frame(self) -> None:
        self._move_batch_current("next")

    def _send_current_batch_frame(self) -> None:
        self._move_batch_current("current")

    def _move_batch_current(self, action: str, ask: bool = True) -> None:
        """Navigate the configured loop manually and emit the selected pair.
        ask=False (trigger-driven moves) never opens the slow-jump dialog."""
        if self._frame_index_input_connected:
            self._status_var.set("disconnect index inputs for batch")
            return
        if not self._validate_loop_fields():
            self._status_var.set("correct red batch values")
            return
        values = self._batch_range()
        if not values:
            return

        current = int(self._batch_current_var.get())
        position = values.index(current)
        if action == "first":
            position = 0
        elif action == "last":
            position = len(values) - 1
        elif action == "prev":
            position = max(0, position - 1)
        elif action == "next":
            position = min(len(values) - 1, position + 1)

        loop_index = values[position]
        if ask and not self._confirm_video_jump(loop_index):
            return
        if self._batch_running:
            self._stop_batch("manual batch control")
        self._batch_current_var.set(str(loop_index))
        self._emit_batch_frames(loop_index, emit_trigger=True)
        if not self._batch_running:
            self._status_var.set(f"sent i={loop_index}")

    def _toggle_play(self) -> None:
        if self._batch_running:
            self._stop_batch("batch stopped")
        else:
            self._start_batch()

    def _start_play(self) -> None:
        self._start_batch()

    def _stop_play(self) -> None:
        self._playing = False
        self._play_btn_var.set("Play")
        if self._after_id:
            self.canvas.after_cancel(self._after_id)
            self._after_id = None

    def _play_tick(self) -> None:
        """Advance one frame and schedule the next tick."""
        if not self._playing:
            return

        next_idx = self._current_index + 1
        if next_idx >= self._frame_total():
            # reached end — stop
            self._stop_play()
            return

        self._load_and_push(next_idx)

        try:
            interval_ms = self._fps_interval_ms(self._fps_var.get())
        except ValueError as exc:
            self._stop_play()
            self._status_var.set(str(exc))
            self.set_status("invalid FPS", "#cc0000")
            return
        self._after_id = self.canvas.after(
            interval_ms, self._play_tick)

    # ── override push_output for SYNC mode ───────────────────────

    def _emit_trigger_pulse(self, base_outputs: dict | None = None,
                            emit_batch_start: bool = False) -> None:
        """Emit a high->low trigger pulse, optionally with a first-step pulse.

        Only the high edge drives a full downstream recompute; the low/reset
        edge is delivered as a cache-only update so trigger-consumers still see
        a correct 1->0 transition without re-running downstream nodes a second
        time (which would double their expensive compute() work, e.g. an
        Undistortion node or an Image Save node in Continuous mode).

        Because the low/reset edge never reaches a downstream node's compute(),
        edge-detecting trigger consumers (Template Match, Optical Flow, Image
        Save's Triggered mode, ...) never observe a real 1->0 transition
        between two pulses. A fixed 1.0 high value would then look like "no
        change" on the second and later pulses and get silently ignored, so
        each high pulse instead carries a monotonically increasing counter --
        the same "ever-incrementing counter-style trigger" value those nodes
        already special-case in their edge detection.
        """
        if not self._on_output_ready:
            return
        self._trigger_pulse_counter += 1
        pulse_outputs = dict(base_outputs or {})
        pulse_outputs["trigger"] = float(self._trigger_pulse_counter)
        if emit_batch_start:
            pulse_outputs["batch_start"] = 1.0
        self._on_output_ready(self.node_id, pulse_outputs)

        reset_outputs = dict(base_outputs or {})
        reset_outputs["trigger"] = 0.0
        if emit_batch_start:
            reset_outputs["batch_start"] = 0.0
        reset_outputs["_cache_only"] = True
        self._on_output_ready(self.node_id, reset_outputs)

    def push_output(self, outputs: dict) -> None:
        """
        ImageSequenceNode is SYNC but needs to push frames
        like a STREAMING node so that VideoPlayOutputNode
        receives them without the user having to manually
        trigger downstream.
        We call the engine callback directly.
        """
        if not self._on_output_ready:
            return
        self._on_output_ready(self.node_id, outputs)

    # ── compute (called by engine for SYNC nodes) ─────────────────

    def compute(self, inputs: dict) -> dict:
        """
        Return the last loaded frame when the engine pulls data.
        This handles the case where a downstream node is added
        after a frame is already loaded.
        """
        raw_frame_index_1 = inputs.get("frame_index_1")
        raw_frame_index_2 = inputs.get("frame_index_2")
        trigger_is_linked = "trigger" in inputs
        current_trigger_is_linked = "current_trigger" in inputs
        if trigger_is_linked != self._trigger_linked:
            self.on_input_link_changed("trigger", trigger_is_linked)
        has_external_index = raw_frame_index_1 is not None
        has_external_index_2 = raw_frame_index_2 is not None
        self._set_external_index_connected(
            has_external_index or has_external_index_2)

        if not self._frame_total():
            return {}

        # When indices are connected, decode both requested frames in this
        # synchronous call.  The existing asynchronous path is intentionally
        # retained for inspector/playback interaction, but cannot be used here:
        # it coalesces requests and would drop one of the two frame requests.
        if has_external_index or has_external_index_2:
            requested_index = self._parse_frame_index(
                raw_frame_index_1, "frame_index_1",
                self._current_index)
            requested_index_2 = self._parse_frame_index(
                raw_frame_index_2, "frame_index_2",
                self._current_index)
            if requested_index is None or requested_index_2 is None:
                return {}

            self._stop_play()
            self._current_index = requested_index
            self._last_frame = self._frame_for_index(
                requested_index, self._last_frame, self._last_frame_index)
            self._last_frame_index = requested_index
            if requested_index_2 == requested_index:
                self._last_frame_2 = self._last_frame
            else:
                self._last_frame_2 = self._frame_for_index(
                    requested_index_2, self._last_frame_2,
                    self._last_frame_2_index)
            self._last_frame_2_index = requested_index_2
            self._refresh_current_file_display()

            if self._last_frame is None or self._last_frame_2 is None:
                return {}
            return {
                "image_1": self._last_frame,
                "image_2": self._last_frame_2,
                "frame_index_1": float(requested_index + 1),
                "frame_index_2": float(requested_index_2 + 1),
                "files": self._absolute_file_paths(),
                "frame_count": float(self._frame_total()),
            }

        if trigger_is_linked or current_trigger_is_linked:
            if trigger_is_linked and self._truthy_trigger(inputs.get("trigger")):
                self._move_batch_current("next", ask=False)
            if current_trigger_is_linked and self._truthy_trigger(inputs.get("current_trigger")):
                self._move_batch_current("current", ask=False)
            # A truthy pulse already pushed frame output and the trigger pulse
            # through push_output(). Avoid emitting a second sync-path payload
            # that could race with trigger propagation (also applies to the
            # low/reset edge, which must suppress downstream recompute).
            return {
                "_skip_downstream": True,
                "_preserve_cache": True,
            }

        if self._last_frame is None or self._last_frame_index != self._current_index:
            return {}
        outputs = {
            "image_1": self._last_frame,
            "frame_index_1": float(self._current_index + 1),
            "files": self._absolute_file_paths(),
            "frame_count": float(self._frame_total()),
        }
        if self._last_frame_2 is not None:
            outputs["image_2"] = self._last_frame_2
            outputs["frame_index_2"] = float(self._last_frame_2_index + 1)
        return outputs

    @staticmethod
    def _truthy_trigger(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() not in ("", "0", "false", "no", "off")
        try:
            return bool(float(value))
        except (TypeError, ValueError):
            return bool(value)

    def _parse_frame_index(self, value, input_name: str,
                           default: int) -> int | None:
        """Convert a one-based index input to a valid zero-based index."""
        if value is None:
            return default
        try:
            index = int(round(float(value))) - 1
        except (TypeError, ValueError, OverflowError):
            self._status_var.set(f"invalid {input_name} input")
            return None
        return max(0, min(index, self._frame_total() - 1))

    def _frame_for_index(self, index: int, cached_frame,
                         cached_index: int) -> np.ndarray | None:
        """Return a decoded frame, reusing the matching index cache."""
        if cached_frame is not None and cached_index == index:
            return cached_frame
        frame = self._read_index_once(index)
        if frame is None:
            self._status_var.set(f"cannot read: {self._frame_label(index)}")
        return frame

    @staticmethod
    def _read_frame_blocking(abs_path: str, wait_s: float, cancelled) -> np.ndarray | None:
        """Poll for a file an external process may still be writing (Pattern
        mode), then decode it once its size looks stable and cv2 can read it.

        Runs on a background thread only -- must not touch any Tk widget/var.
        Returns None once `cancelled()` reports True (batch stopped/node
        destroyed/superseded by a newer request).
        """
        path_obj = Path(abs_path)
        wait_s = max(0.05, wait_s)
        last_size = -1
        while not cancelled():
            if path_obj.exists():
                try:
                    size = path_obj.stat().st_size
                except OSError:
                    size = -1
                # Only pay for a real cv2.imread() once size has held steady
                # across two polls -- cuts down on attempts against a file
                # that is still mid-write and would just decode as garbage/None.
                if size > 0 and size == last_size:
                    frame = cv2.imread(abs_path)
                    if frame is not None:
                        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                last_size = size
            time.sleep(wait_s)
        return None

    def _fetch_frame_async(self, index: int, on_ready) -> None:
        """Resolve file_paths[index] to a decoded frame on a background
        thread, then hand the result to on_ready(frame_or_None) on the main
        thread. In Pattern mode this waits for/retries a file that has not
        finished being generated yet; in Folder mode it is a one-shot decode.
        """
        token = self._batch_fetch_token
        src = self._src_index(index)

        def _cancelled() -> bool:
            return self._destroyed or self._batch_fetch_token != token

        if self._is_video():
            reader = self._video_reader

            def _task():
                return reader.read(src) if reader is not None and src >= 0 else None
        elif not (0 <= src < len(self._file_paths)):
            def _task():
                return None
        else:
            abs_path = self._to_absolute(self._file_paths[src], self._get_project_base())
            waits = self._waits_for_files()
            try:
                wait_s = float(self._pattern_wait_var.get())
            except (tk.TclError, ValueError):
                wait_s = 0.5

            def _task():
                if waits:
                    return self._read_frame_blocking(abs_path, wait_s, _cancelled)
                return self._decode_frame(abs_path)

        future = self._DECODE_POOL.submit(_task)

        def _done(fut) -> None:
            if _cancelled():
                return
            try:
                frame = fut.result()
            except Exception:
                frame = None
            try:
                self.canvas.after(0, lambda: on_ready(frame))
            except tk.TclError:
                pass

        future.add_done_callback(_done)

    # ── serialization ─────────────────────────────────────────────

    def _normalize_paths(self, paths, base: Path | None) -> list[str]:
        """_to_relative(_to_absolute(p)) for a whole list, computed once per
        FOLDER (a sequence has thousands of files in a few folders)."""
        folders: dict[str, tuple[str, str]] = {}
        out = []
        for p in paths:
            folder, name = os.path.split(str(p))
            if not name or name in (".", ".."):
                out.append(self._to_relative(self._to_absolute(p, base), base))
                continue
            conv = folders.get(folder)
            if conv is None:
                # Convert a probe path in this folder and keep its folder part.
                probe = self._to_relative(self._to_absolute(os.path.join(folder, "x"), base), base)
                conv = folders[folder] = (os.path.split(probe)[0],)
            out.append(os.path.join(conv[0], name) if conv[0] else name)
        return out

    def _to_relative(self, path: str,
                     base: Path | None) -> str:
        """Convert a path to a project-relative path using relpath semantics."""
        if not path:
            return path
        if base is None:
            return path
        # relpath is computed once per FOLDER and the file name appended:
        # a sequence has thousands of files in a few folders, and calling
        # os.path.relpath per file cost ~2 s per project load/save.
        folder, name = os.path.split(str(path))
        cache = self.__dict__.setdefault("_relpath_cache", {})
        key = (folder, str(base))
        rel_folder = cache.get(key)
        if rel_folder is None:
            try:
                rel_folder = os.path.relpath(folder, str(base)) if folder else ""
            except Exception:
                return path            # e.g. a different drive on Windows
            if len(cache) > 1000:
                cache.clear()
            cache[key] = rel_folder
        return os.path.join(rel_folder, name) if rel_folder not in ("", ".") else name

    def _to_absolute(self, path: str,
                     base: Path | None) -> str:
        """Resolve relative path against base.

        Pure string work (join + normpath, which also folds '..'). It used
        Path.resolve(), which on Windows asks the FILE SYSTEM for every
        path's final name -- with thousands of files per sequence, converted
        several times per load/save and on every frame emit, that was ~80k
        disk queries and 10-17 s per project load/save (worse on slow or
        external drives)."""
        folder, name = os.path.split(str(path))
        if not name or name in (".", ".."):          # rare (folder fields): no per-folder cache
            p = Path(path)
            if p.is_absolute() or base is None:
                return str(p)
            return os.path.normpath(os.path.join(str(base), str(p)))
        cache = self.__dict__.setdefault("_abspath_cache", {})
        key = (folder, None if base is None else str(base))
        abs_folder = cache.get(key)
        if abs_folder is None:
            p = Path(folder) if folder else None
            if p is None:
                abs_folder = "" if base is None else os.path.normpath(str(base))
            elif p.is_absolute() or base is None:
                abs_folder = str(p)
            else:
                abs_folder = os.path.normpath(os.path.join(str(base), str(p)))
            if len(cache) > 1000:
                cache.clear()
            cache[key] = abs_folder
        return os.path.join(abs_folder, name) if abs_folder else name

    def _get_project_base(self) -> Path | None:
        return get_project_directory()

    def _absolute_file_paths(self) -> list[str]:
        """Return the SELECTED files as absolute paths, in selection order
        (all files, in list order, when the selector is empty) -- so that
        'files', 'count' and the index pins describe the same sequence.

        Called on every frame emit, so the converted list is cached and only
        rebuilt when the file list, the selection or the project folder changes."""
        if self._is_video():
            return []                      # a video has no file list
        base = self._get_project_base()
        key = (str(base), tuple(self._file_paths), self._selector_applied)
        cache = getattr(self, "_abs_paths_cache", None)
        if cache is None or cache[0] != key:
            sel = self._selection()[0]
            paths = self._file_paths if sel is None else [self._file_paths[k] for k in sel]
            cache = (key, [self._to_absolute(p, base) for p in paths])
            self._abs_paths_cache = cache
        return list(cache[1])

    def _selected_file_paths(self) -> list[str]:
        """Stored (project-relative) paths of the selected files, in order."""
        sel = self._selection()[0]
        return list(self._file_paths) if sel is None else [self._file_paths[k] for k in sel]

    def get_help_text(self) -> str:
        return self.HELP_TEXT

    _INLINE_FILE_LIST_MAX = 12000   # longer lists go to a side file (Excel cell limit)

    def _project_sidecar_filename(self) -> str | None:
        """'<project>.<node_id>.image_sequence_files.txt' next to the project
        file, or None while the project has no file yet."""
        project_path = get_project_file_path()
        if project_path is None:
            return None
        safe_node_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in self.node_id)
        return f"{project_path.stem}.{safe_node_id}.image_sequence_files.txt"

    def _write_file_list_sidecar(self, filename: str, paths: list[str]) -> bool:
        """One project-relative path per line, UTF-8; written atomically."""
        project_dir = get_project_directory()
        if project_dir is None:
            return False
        path = project_dir / filename
        text = "\n".join(paths) + "\n"
        if self._sidecar_unchanged(path, text):
            return True                              # same content already on disk
        tmp = path.with_name(path.name + ".tmp")
        try:
            tmp.write_text(text, encoding="utf-8")
            tmp.replace(path)
            self._remember_sidecar(path, text)
            return True
        except OSError:
            try:
                tmp.unlink()
            except OSError:
                pass
            return False

    def _read_file_list_sidecar(self, filename: str) -> str | None:
        project_dir = get_project_directory()
        if project_dir is None or Path(filename).name != filename:
            return None
        try:
            text = (project_dir / filename).read_text(encoding="utf-8")
        except OSError:
            return None
        self._remember_sidecar(project_dir / filename, text)
        return text

    # The side file is skipped on save when its text equals what was last
    # read / written AND the file was not touched since (same size + mtime).
    # Opening a file costs tens of ms on some drives (antivirus, USB, network).
    def _remember_sidecar(self, path: Path, text: str) -> None:
        try:
            st = os.stat(path)
            self._sidecar_known = (str(path), hash(text), st.st_size, st.st_mtime_ns)
        except OSError:
            self._sidecar_known = None

    def _sidecar_unchanged(self, path: Path, text: str) -> bool:
        known = getattr(self, "_sidecar_known", None)
        if not known or known[0] != str(path) or known[1] != hash(text):
            return False
        try:
            st = os.stat(path)
        except OSError:
            return False
        return (st.st_size, st.st_mtime_ns) == known[2:]

    def get_params(self) -> dict:
        base  = self._get_project_base()
        key = (str(base), tuple(self._file_paths))
        cached = getattr(self, "_params_paths_cache", None)
        if cached is None or cached[0] != key:
            cached = (key, self._normalize_paths(self._file_paths, base))
            self._params_paths_cache = cached
        paths = cached[1]
        method = "pattern" if self._wait_files_var.get() else "folder"
        pattern = self._to_relative(self._pattern_var.get().strip(), base)

        folder_value = self._folder_var.get().strip()
        if folder_value and folder_value != "(no folder)":
            folder_value = self._to_relative(self._to_absolute(folder_value, base), base)
        elif self._file_paths:
            # If folder text was lost/stale, infer it from current file list.
            abs_files = [Path(self._to_absolute(p, base)) for p in self._file_paths]
            if abs_files:
                inferred = str(abs_files[0].parent)
                if all(p.parent == abs_files[0].parent for p in abs_files):
                    folder_value = self._to_relative(inferred, base)

        file_list_text = "\n".join(paths)
        # All params of a node are stored as JSON in ONE Excel cell (max
        # 32,767 characters), so a long file list cannot stay inline. It is
        # written to a side file next to the project instead (same scheme as
        # the Array Viewer's .npy side file) and set_params reads it back.
        # Previously a long list was simply dropped in folder mode, and the
        # project reload re-scanned the folder -- so files the user had
        # removed from the list came back (the saved edit was lost).
        file_list_file = ""
        if len(file_list_text) > self._INLINE_FILE_LIST_MAX:
            sidecar = self._project_sidecar_filename()
            if sidecar is not None and self._write_file_list_sidecar(sidecar, paths):
                file_list_file, file_list_text = sidecar, ""
            elif method == "folder":
                file_list_text = ""          # last resort: the folder re-scan

        video_value = self._video_path_var.get().strip()
        if video_value:
            video_value = self._to_relative(self._to_absolute(video_value, base), base)
        return {
            "source":        self._source_var.get(),            # "images" | "video"
            "wait_for_files": bool(self._wait_files_var.get()),
            "video":         video_value,
            "pattern_first": self._pattern_first_var.get(),
            "pattern_last":  self._pattern_last_var.get(),
            "pattern_step":  self._pattern_step_var.get(),
            # Older versions read "method": pattern = wait for files, folder = do not.
            "method":        "pattern" if self._wait_files_var.get() else "folder",
            "pattern":       pattern,
            "pattern_wait":  self._pattern_wait_var.get(),
            "folder":        folder_value or "(no folder)",
            "file_list":     file_list_text,
            "file_list_file": file_list_file,
            "current_index": self._current_index,
            "fps":           self._fps_var.get(),
            "batch_start":   self._batch_start_var.get(),
            "batch_end":     self._batch_end_var.get(),
            "batch_step":    self._batch_step_var.get(),
            "batch_current": self._batch_current_var.get(),
            "batch_frame_1": self._batch_frame_1_var.get(),
            "batch_frame_2": self._batch_frame_2_var.get(),
            "selector":      self._selector_applied,
        }

    def set_params(self, params: dict) -> None:
        self._init_ui_state()
        method = params.get("method", "pattern")
        self._method_var.set(method)
        # Projects saved before the redesign have no "source" / "wait_for_files":
        # Pattern mode waited for files, Folder mode did not.
        source = str(params.get("source", "images"))
        self._wait_files_var.set(bool(params.get("wait_for_files", method == "pattern")))
        for key, var, default in (("pattern_first", self._pattern_first_var, "1"),
                                  ("pattern_last", self._pattern_last_var, "100"),
                                  ("pattern_step", self._pattern_step_var, "1")):
            var.set(str(params.get(key, default)))

        base = self._get_project_base()
        pattern_value = str(params.get("pattern", "") or "")
        self._pattern_var.set(
            self._to_relative(self._to_absolute(pattern_value, base), base)
        )
        try:
            self._pattern_wait_var.set(float(params.get("pattern_wait", 0.5)))
        except (TypeError, ValueError):
            self._pattern_wait_var.set(0.5)
        folder_value = str(params.get("folder", "(no folder)") or "(no folder)")
        if folder_value != "(no folder)":
            folder_value = self._to_relative(self._to_absolute(folder_value, base), base)
        self._folder_var.set(folder_value)
        try:
            fps = float(params.get("fps", 5))
            self._fps_interval_ms(fps)
            self._fps_var.set(str(fps))
        except (TypeError, ValueError):
            self._fps_var.set("5")

        # The selector is set BEFORE the source is loaded, so the frames
        # emitted while loading are already the selected ones. A text that
        # no longer parses (should not happen) falls back to "all".
        selector = str(params.get("selector", "") or "").strip()
        try:
            self._parse_selector(selector, 0)
        except ValueError:
            selector = ""
        self._selector_applied = selector
        self._selector_var.set(selector)
        self._invalidate_frame_cache()

        # A video source: open it (the batch fields are restored below).
        if source == "video":
            video = str(params.get("video", "") or "")
            if video and not self._open_video(video, reset_batch=False, show_errors=False):
                self._status_var.set("saved video not found")
            self._source_var.set("video")
            self._on_source_change(user=False)

        # Restore file list first. This is the most explicit source of truth:
        # inline, or (long lists) from the side file written by get_params.
        file_list = params.get("file_list", "")
        sidecar = str(params.get("file_list_file", "") or "").strip()
        sidecar_missing = False
        if not file_list and sidecar:
            file_list = self._read_file_list_sidecar(sidecar) or ""
            sidecar_missing = not file_list
        if file_list:
            paths = self._normalize_paths(
                [p.strip() for p in file_list.splitlines() if p.strip()], base)
            if paths:
                if method == "folder" and (folder_value == "(no folder)" or not folder_value):
                    # Recover folder label from file list for older payloads.
                    abs_files = [Path(self._to_absolute(p, base)) for p in paths]
                    if abs_files and all(p.parent == abs_files[0].parent for p in abs_files):
                        self._folder_var.set(self._to_relative(str(abs_files[0].parent), base))
                self._set_file_paths(paths)
                idx = int(params.get("current_index", 0))
                self._load_and_push(idx)
        # Folder-mode fallback for legacy/broken params_json payloads where
        # file_list may be missing but folder was persisted.
        elif source != "video" and method == "pattern" and params.get("pattern"):
            # Old Pattern-mode project without a saved list: list position i
            # was pattern % i for i = 1 .. batch End (the old semantics).
            try:
                end = int(params.get("batch_end", 0))
                paths = self._numbered_paths(str(params.get("pattern")), 1, end, 1) if end >= 1 else []
            except (TypeError, ValueError):
                paths = []
            if paths:
                self._pattern_first_var.set("1")
                self._pattern_last_var.set(str(end))
                self._set_file_paths(paths)
        elif method == "folder" and folder_value and folder_value != "(no folder)":
            try:
                abs_folder = self._to_absolute(folder_value, base)
                folder_path = Path(abs_folder)
                if folder_path.exists() and folder_path.is_dir():
                    paths = sorted(
                        str(p) for p in folder_path.iterdir()
                        if p.suffix.lower() in self._IMG_EXTS
                    )
                    if paths:
                        self._set_file_paths(paths)
                        idx = int(params.get("current_index", 0))
                        self._load_and_push(idx)
                        if sidecar_missing:
                            # Set after loading, which resets the status to "ready".
                            self._status_var.set(f"saved file list {sidecar} not found: "
                                                 "re-scanned the whole folder")
                    else:
                        self._status_var.set("folder restored but no readable images")
                else:
                    self._status_var.set("saved folder not found")
            except Exception:
                self._status_var.set("failed to restore saved folder")

        self._batch_start_var.set(str(params.get("batch_start", "1")))
        self._batch_end_var.set(str(params.get(
            "batch_end", max(1, self._frame_total() - 1))))
        self._batch_step_var.set(str(params.get("batch_step", "1")))
        self._batch_current_var.set(str(params.get(
            "batch_current", self._batch_start_var.get())))
        # Fallbacks keep the behaviour of projects saved before the batch
        # fields existed (Frame 1 = current frame, Frame 2 = the next one).
        # New file lists get the 1 / i defaults from _set_file_paths().
        self._batch_frame_1_var.set(str(params.get("batch_frame_1", "i")))
        self._batch_frame_2_var.set(str(params.get("batch_frame_2", "i+1")))
        if source == "video":
            self._source_var.set("video")
            self._on_source_change(user=False)
        if self._validate_loop_fields() and self._frame_total():
            self._emit_batch_frames(int(self._batch_current_var.get()))

    def on_destroy(self) -> None:
        with self._decode_lock:
            self._destroyed = True
            self._decode_pending_req = None
        self._stop_play()
        self._stop_batch("destroyed")
        self._close_video()
        super().on_destroy()
