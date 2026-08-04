# node_editor/nodes/image_sequence_node.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np

from node_editor.base_node import BaseNode
from node_editor.pin_types import PinSchema, PinDef, PinType
from node_editor.execution import ExecutionMode
from node_editor.project_context import get_project_directory


class ImageSequenceNode(BaseNode):
    """
    Provides a sequence of images from disk as IMAGE output.

    Two input methods:
      A. Pattern mode  — printf-style path pattern with start/end indices
                         e.g. /images/DCIM%04d.JPG  start=123  end=987
      B. Folder mode   — scan a folder for all OpenCV-readable image files,
                         sorted alphanumerically

    Paths are stored and serialized as relative paths with respect to the
    XLSX project file location (app._current_file.parent). If no project
    file is saved yet, absolute paths are used and a warning is shown on
    save.

    Playback uses tk.after() on the main thread (no extra thread needed).
    Output pins: image, image_2 (PinType.IMAGE)
    Additional output pins: frame_index (PinType.SCALAR), 
                            frame_count (PinType.SCALAR)
    """

    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE      = "image_sequence"
    DISPLAY_NAME   = "Image Sequence"
    CATEGORY       = "source"
    NODE_WIDTH     = 260
    NODE_HEIGHT    = 270

    HELP_TEXT = (
        "Image Sequence\n\n"
        "Load images with Pattern or Folder mode, then use the Batch sequence "
        "section to choose an inclusive, 1-based Start, End, and Step. "
        "Positive and negative steps are supported. Set Frame 1 and Frame 2 "
        "with i, i+N, i-N, or a fixed frame number; they are checked across every loop iteration. "
        "A black value is valid; red text must be corrected before a batch can run.\n\n"
        "Play: with no link on the next input, one frame pair is emitted at "
        "each FPS interval. Connect a TRIGGER output to next to advance one "
        "frame for each trigger pulse instead.\n\n"
        "The index and index 2 inputs still select the two image outputs "
        "directly. They take precedence over manual and batch playback."
    )

    # Shared decode pool across all ImageSequence instances.
    _DECODE_POOL = ThreadPoolExecutor(max_workers=4)

    # supported extensions (OpenCV-readable)
    _IMG_EXTS = {
        ".jpg", ".jpeg", ".png", ".bmp",
        ".tif", ".tiff", ".webp", ".ppm", ".pgm",
    }

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[
                PinDef("frame_index", PinType.SCALAR, "index", optional=True),
                PinDef("frame_index_2", PinType.SCALAR, "index 2", optional=True),
                PinDef("trigger", PinType.TRIGGER, "next", optional=True),
            ],
            outputs=[
                PinDef("image",       PinType.IMAGE,  "frame"),
                PinDef("image_2",     PinType.IMAGE,  "frame 2"),
                PinDef("frame_index", PinType.SCALAR, "index"),
                PinDef("frame_count", PinType.SCALAR, "count"),
            ]
        )

    def _init_ui_state(self) -> None:
        if hasattr(self, "_method_var"):
            return

        self._method_var = tk.StringVar(value="pattern")
        self._pattern_var = tk.StringVar(value="/images/DCIM%04d.JPG")
        self._start_var = tk.StringVar(value="0")
        self._end_var = tk.StringVar(value="100")
        self._folder_var = tk.StringVar(value="(no folder)")

        self._info_var = tk.StringVar(value="no files loaded")
        self._play_btn_var = tk.StringVar(value="Play")
        self._fps_var = tk.IntVar(value=5)
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
        self._batch_frame_1_var = tk.StringVar(value="i")
        self._batch_frame_2_var = tk.StringVar(value="i+1")
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
        self._start_entry = None
        self._end_entry = None
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
            self._last_frame:     np.ndarray | None = None
            self._last_frame_index: int = -1
            self._last_frame_2:   np.ndarray | None = None
            self._last_frame_2_index: int = -1

            self._decode_lock = threading.Lock()
            self._decode_future = None
            self._decode_pending_req: tuple[int, str, int] | None = None
            self._decode_request_id: int = 0
            self._destroyed: bool = False

            self._slider_dragging: bool = False

    def build_inspector(self, parent: tk.Frame) -> None:
        self._init_ui_state()

        top = tk.Frame(parent)
        top.pack(fill="both", expand=True)

        method_frame = tk.Frame(top)
        method_frame.pack(fill="x", pady=(0, 6))
        tk.Radiobutton(
            method_frame, text="Pattern",
            variable=self._method_var, value="pattern",
            font=("Arial", 8),
            command=self._on_method_change,
        ).pack(side="left")
        tk.Radiobutton(
            method_frame, text="Folder",
            variable=self._method_var, value="folder",
            font=("Arial", 8),
            command=self._on_method_change,
        ).pack(side="left", padx=6)

        self._pat_frame = tk.LabelFrame(top, text="Pattern Mode", padx=6, pady=4)
        self._pat_frame.pack(fill="x", pady=(0, 6))

        tk.Label(self._pat_frame, text="Pattern:", font=("Arial", 8)).grid(row=0, column=0, sticky="w")
        self._pat_entry = tk.Entry(self._pat_frame, textvariable=self._pattern_var, width=34, font=("Arial", 8))
        self._pat_entry.grid(row=0, column=1, columnspan=3, padx=2, pady=1, sticky="ew")

        tk.Label(self._pat_frame, text="Start:", font=("Arial", 8)).grid(row=1, column=0, sticky="w")
        self._start_entry = tk.Entry(self._pat_frame, textvariable=self._start_var, width=8, font=("Arial", 8))
        self._start_entry.grid(row=1, column=1, padx=2, pady=1, sticky="w")

        tk.Label(self._pat_frame, text="End:", font=("Arial", 8)).grid(row=1, column=2, sticky="w", padx=(8, 0))
        self._end_entry = tk.Entry(self._pat_frame, textvariable=self._end_var, width=8, font=("Arial", 8))
        self._end_entry.grid(row=1, column=3, padx=2, pady=1, sticky="w")

        tk.Button(self._pat_frame, text="Load Pattern", font=("Arial", 8), command=self._load_pattern).grid(
            row=2, column=0, columnspan=4, pady=3, sticky="ew"
        )

        self._fld_frame = tk.LabelFrame(top, text="Folder Mode", padx=6, pady=4)
        self._fld_frame.pack(fill="x", pady=(0, 6))

        self._folder_label = tk.Label(
            self._fld_frame,
            textvariable=self._folder_var,
            font=("Arial", 8),
            anchor="w",
            justify="left",
            wraplength=360,
        )
        self._folder_label.grid(row=0, column=0, columnspan=2, sticky="w")

        tk.Button(self._fld_frame, text="Browse Folder...", font=("Arial", 8), command=self._browse_folder).grid(
            row=1, column=0, pady=3, sticky="ew"
        )
        tk.Button(self._fld_frame, text="Edit File List...", font=("Arial", 8), command=self._edit_file_list).grid(
            row=1, column=1, pady=3, sticky="ew", padx=(4, 0)
        )

        info_lbl = tk.Label(top, textvariable=self._info_var, font=("Arial", 8), anchor="w", justify="left")
        info_lbl.pack(fill="x", pady=(0, 6))

        self._batch_frame = tk.LabelFrame(
            top, text="Batch sequence", padx=6, pady=4,
        )
        self._batch_frame.pack(fill="x", pady=(0, 6))
        tk.Label(
            self._batch_frame,
            text="Inclusive 1-based loop. Use i, i+N, i-N, or a fixed frame number.",
            font=("Arial", 8), fg="#555555",
        ).grid(row=0, column=0, columnspan=6, sticky="w")

        for column, (label, variable, key) in enumerate((
            ("Start", self._batch_start_var, "start"),
            ("End", self._batch_end_var, "end"),
            ("Step", self._batch_step_var, "step"),
        )):
            tk.Label(self._batch_frame, text=label + ":", font=("Arial", 8)).grid(
                row=1, column=column * 2, sticky="w", padx=(0 if column == 0 else 8, 2), pady=(4, 0),
            )
            entry = tk.Entry(
                self._batch_frame, textvariable=variable, width=6,
                justify="center", font=("Arial", 8),
            )
            entry.grid(row=1, column=column * 2 + 1, sticky="w", pady=(4, 0))
            entry.bind("<Return>", self._on_batch_range_commit)
            entry.bind("<FocusOut>", self._on_batch_range_commit)
            self._loop_entries[key] = entry

        tk.Label(self._batch_frame, text="Current i:", font=("Arial", 8)).grid(
            row=2, column=0, sticky="w", pady=(5, 0),
        )
        current_entry = tk.Entry(
            self._batch_frame, textvariable=self._batch_current_var,
            width=6, justify="center", font=("Arial", 8),
        )
        current_entry.grid(row=2, column=1, sticky="w", pady=(5, 0))
        current_entry.bind("<Return>", self._on_batch_range_commit)
        current_entry.bind("<FocusOut>", self._on_batch_range_commit)
        self._loop_entries["current"] = current_entry
        tk.Label(
            self._batch_frame, text="Current loop position (1-based)",
            font=("Arial", 8), fg="#555555",
        ).grid(row=2, column=2, columnspan=4, sticky="w", padx=(8, 0), pady=(5, 0))

        nav_frame = tk.Frame(self._batch_frame)
        nav_frame.grid(row=3, column=0, columnspan=6, sticky="ew", pady=(5, 0))
        nav_cfg = dict(font=("Arial", 8), width=7)
        self._batch_first_btn = tk.Button(
            nav_frame, text="First", command=self._go_first, **nav_cfg,
        )
        self._batch_first_btn.pack(side="left", padx=1)
        self._batch_prev_btn = tk.Button(
            nav_frame, text="Prev", command=self._prev_frame, **nav_cfg,
        )
        self._batch_prev_btn.pack(side="left", padx=1)
        self._batch_current_btn = tk.Button(
            nav_frame, text="Current", command=self._send_current_batch_frame, **nav_cfg,
        )
        self._batch_current_btn.pack(side="left", padx=1)
        self._batch_next_btn = tk.Button(
            nav_frame, text="Next", command=self._next_frame, **nav_cfg,
        )
        self._batch_next_btn.pack(side="left", padx=1)
        self._batch_last_btn = tk.Button(
            nav_frame, text="Last", command=self._go_last, **nav_cfg,
        )
        self._batch_last_btn.pack(side="left", padx=1)
        self._batch_play_btn = tk.Button(
            nav_frame, textvariable=self._play_btn_var,
            command=self._toggle_play, **nav_cfg,
        )
        self._batch_play_btn.pack(side="left", padx=(4, 1))
        tk.Label(nav_frame, text="FPS:", font=("Arial", 8)).pack(side="right", padx=(8, 0))
        tk.Spinbox(nav_frame, from_=1, to=60, textvariable=self._fps_var, width=4, font=("Arial", 8)).pack(side="right", padx=2)

        tk.Label(self._batch_frame, text="Frame 1:", font=("Arial", 8)).grid(
            row=4, column=0, sticky="w", pady=(5, 0),
        )
        frame_1_entry = tk.Entry(
            self._batch_frame, textvariable=self._batch_frame_1_var,
            width=8, justify="center", font=("Arial", 8),
        )
        frame_1_entry.grid(row=4, column=1, sticky="w", pady=(5, 0))
        tk.Label(self._batch_frame, text="Frame 2:", font=("Arial", 8)).grid(
            row=4, column=2, sticky="w", padx=(8, 2), pady=(5, 0),
        )
        frame_2_entry = tk.Entry(
            self._batch_frame, textvariable=self._batch_frame_2_var,
            width=8, justify="center", font=("Arial", 8),
        )
        frame_2_entry.grid(row=4, column=3, sticky="w", pady=(5, 0))
        for entry, key in ((frame_1_entry, "frame_1"), (frame_2_entry, "frame_2")):
            entry.bind("<Return>", self._on_batch_range_commit)
            entry.bind("<FocusOut>", self._on_batch_range_commit)
            self._loop_entries[key] = entry

        tk.Label(
            self._batch_frame, textvariable=self._batch_mode_var,
            font=("Arial", 8), fg="#356a49",
        ).grid(row=5, column=0, columnspan=6, sticky="w", pady=(5, 0))

        tk.Label(top, textvariable=self._fname_var, font=("Arial", 8), anchor="w", justify="left").pack(fill="x")
        tk.Label(top, textvariable=self._size_var, font=("Arial", 8), anchor="w", justify="left").pack(fill="x")

        self._on_method_change()
        self._refresh_inspector_widgets()
        self._validate_loop_fields()

    def _refresh_inspector_widgets(self) -> None:
        if self._slider is not None and self._slider.winfo_exists():
            self._slider.configure(from_=0, to=max(0, len(self._file_paths) - 1))
            self._slider_var.set(max(0, min(self._current_index, max(0, len(self._file_paths) - 1))))
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

    def _validate_loop_fields(self, *_args) -> bool:
        """Show valid loop values in black and invalid values in red."""
        count = len(getattr(self, "_file_paths", []))
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

        valid = {
            "start": values["start"] is not None and 1 <= values["start"] <= count,
            "end": values["end"] is not None and 1 <= values["end"] <= count,
            "step": values["step"] is not None and values["step"] != 0 and abs(values["step"]) <= count,
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
                and all(
                    1 <= self._expression_frame_number(offset, index) <= count
                    for index in loop_indices
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
        if not self._file_paths:
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
            delay_ms = max(1, int(1000 / max(1, self._fps_var.get())))
        self._batch_after_id = self.canvas.after(delay_ms, self._advance_batch)

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
        self._emit_batch_frames(loop_index)
        if not self._batch_running:
            return
        self._status_var.set(
            f"batch {self._batch_position}/{len(self._batch_indices)}"
        )

        if self._batch_position >= len(self._batch_indices):
            self._stop_batch("batch complete")
        elif not self._trigger_linked:
            self._schedule_batch_tick()

    def _emit_batch_frames(self, loop_index: int) -> None:
        """Decode and publish the two frame expressions for one batch index."""
        frame_1_expression = self._parse_batch_expression(self._batch_frame_1_var.get())
        frame_2_expression = self._parse_batch_expression(self._batch_frame_2_var.get())
        if frame_1_expression is None or frame_2_expression is None:
            self._stop_batch("invalid frame expression")
            return

        frame_1_index = self._expression_frame_number(frame_1_expression, loop_index) - 1
        frame_2_index = self._expression_frame_number(frame_2_expression, loop_index) - 1
        if not (0 <= frame_1_index < len(self._file_paths)
                and 0 <= frame_2_index < len(self._file_paths)):
            self._stop_batch("frame expression out of range")
            return
        frame_1 = self._frame_for_index(
            frame_1_index, self._last_frame, self._last_frame_index)
        if frame_2_index == frame_1_index:
            frame_2 = frame_1
        else:
            frame_2 = self._frame_for_index(
                frame_2_index, self._last_frame_2, self._last_frame_2_index)
        if frame_1 is None or frame_2 is None:
            self._stop_batch("cannot read batch frame")
            return

        self._current_index = frame_1_index
        self._last_frame = frame_1
        self._last_frame_index = frame_1_index
        self._last_frame_2 = frame_2
        self._last_frame_2_index = frame_2_index
        self._slider_var.set(frame_1_index)
        self._idx_var.set(str(frame_1_index + 1))
        self._fname_var.set(Path(self._file_paths[frame_1_index]).name)
        height, width = frame_1.shape[:2]
        self._size_var.set(f"{width} x {height}")
        self.push_output({
            "image": frame_1,
            "image_2": frame_2,
            "frame_index": float(frame_1_index + 1),
            "frame_count": float(len(self._file_paths)),
        })

    def close_inspector(self) -> None:
        super().close_inspector()
        self._pat_entry = None
        self._start_entry = None
        self._end_entry = None
        self._folder_label = None
        self._slider = None
        self._idx_entry = None
        self._pat_frame = None
        self._fld_frame = None
        self._batch_frame = None
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

    # ── method switching ──────────────────────────────────────────

    def _on_method_change(self) -> None:
        if self._pat_frame is None or self._fld_frame is None:
            return
        if self._method_var.get() == "pattern":
            if self._fld_frame.winfo_manager() == "pack":
                self._fld_frame.pack_forget()
            if self._pat_frame.winfo_manager() != "pack":
                self._pat_frame.pack(fill="x", pady=(0, 6))
        else:
            if self._pat_frame.winfo_manager() == "pack":
                self._pat_frame.pack_forget()
            if self._fld_frame.winfo_manager() != "pack":
                self._fld_frame.pack(fill="x", pady=(0, 6))

    @staticmethod
    def _show_warning_with_terminal_log(title: str, message: str) -> None:
        """Show warning dialog and mirror message to terminal for easy copy."""
        print(f"[ImageSequenceNode][Warning] {title}: {message}", flush=True)
        messagebox.showwarning(title, message)

    # ── method A: pattern ─────────────────────────────────────────

    def _load_pattern(self) -> None:
        pattern = self._pattern_var.get().strip()
        base = self._get_project_base()
        try:
            start = int(self._start_var.get())
            end   = int(self._end_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid Input",
                "Start and End must be integers.")
            return

        if start > end:
            messagebox.showerror(
                "Invalid Input",
                "Start must be less than or equal to End.")
            return

        paths = []
        missing = []
        for i in range(start, end + 1):
            try:
                p = pattern % i
            except TypeError:
                messagebox.showerror(
                    "Invalid Pattern",
                    "Pattern must contain a printf-style "
                    "integer format specifier such as %04d.")
                return
            abs_p = self._to_absolute(p, base)
            if Path(abs_p).exists():
                paths.append(abs_p)
            else:
                missing.append(abs_p)

        if not paths:
            self._show_warning_with_terminal_log(
                "No Files Found",
                f"No files found for pattern:\n{pattern}\n"
                f"start={start}  end={end}")
            return

        if missing:
            # warn but continue with found files
            sample = "\n".join(missing[:5])
            more   = f"\n... and {len(missing)-5} more" \
                     if len(missing) > 5 else ""
            self._show_warning_with_terminal_log(
                "Missing Files",
                f"{len(missing)} files not found "
                f"(skipped):\n{sample}{more}")

            rel_pattern = self._to_relative(self._to_absolute(pattern, base), base)
            self._pattern_var.set(rel_pattern)

        self._set_file_paths(paths)

    # ── method B: folder ─────────────────────────────────────────

    def _browse_folder(self) -> None:
        folder = filedialog.askdirectory(
            title="Select image folder")
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

    def _edit_file_list(self) -> None:
        """
        Open a popup text editor showing the current file list,
        one path per line. Experienced users can add, remove, or
        reorder paths. Changes take effect when they click Apply.
        """
        if not self._file_paths:
            messagebox.showinfo(
                "Edit File List",
                "No files loaded yet. "
                "Use Browse Folder first.")
            return

        win = tk.Toplevel()
        win.title("Edit File List")
        win.geometry("700x500")

        tk.Label(win,
                 text="One file path per line. "
                      "Paths relative to project file or absolute.",
                 font=("Arial", 9), pady=4).pack()

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

        txt.insert("1.0", "\n".join(self._file_paths))

        def _apply():
            base = self._get_project_base()
            lines = [l.strip()
                     for l in txt.get("1.0", tk.END).splitlines()
                     if l.strip()]
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
            self._set_file_paths(lines)
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

    def _set_file_paths(self, paths: list[str]) -> None:
        base = self._get_project_base()
        # Keep canonical stored paths project-relative when possible.
        normalized = [
            self._to_relative(self._to_absolute(p, base), base)
            for p in paths
        ]

        self._file_paths    = normalized
        self._current_index = 0
        self._stop_batch("ready")

        n = len(normalized)
        self._batch_start_var.set("1")
        # The default pair i / i+1 needs one following frame available.
        self._batch_end_var.set(str(max(1, n - 1)))
        self._batch_step_var.set("1")
        self._batch_current_var.set("1")
        self._batch_frame_1_var.set("i")
        self._batch_frame_2_var.set("i+1")
        if self._slider is not None and self._slider.winfo_exists():
            self._slider.configure(from_=0, to=max(0, n - 1))
        self._slider_var.set(0)
        self._count_var.set(f"/ {n}")
        self._info_var.set(
            f"{n} files  "
            f"{Path(normalized[0]).name} ... {Path(normalized[-1]).name}")
        self._idx_var.set("1")
        self._fname_var.set("")
        self._size_var.set("")
        # Publish the configured pair immediately (Frame 1=i, Frame 2=i+1
        # by default), so the two output pins carry different frames.
        if self._validate_loop_fields():
            self._emit_batch_frames(1)
        else:
            self._load_and_push(0)

    # ── frame loading ─────────────────────────────────────────────

    @staticmethod
    def _decode_frame(abs_path: str) -> np.ndarray | None:
        frame = cv2.imread(abs_path)
        if frame is None:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _request_decode(self, index: int) -> None:
        if not self._file_paths:
            return

        index = max(0, min(index, len(self._file_paths) - 1))
        self._current_index = index

        base = self._get_project_base()
        abs_path = self._to_absolute(self._file_paths[index], base)

        with self._decode_lock:
            self._decode_request_id += 1
            req_id = self._decode_request_id
            self._decode_pending_req = (req_id, abs_path, index)

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

            req_id, abs_path, index = req
            fut = self._DECODE_POOL.submit(self._decode_frame, abs_path)
            self._decode_future = fut

        def _done_cb(future, rid=req_id, path=abs_path, idx=index):
            try:
                self.canvas.after(0, lambda: self._on_decode_done(rid, path, idx, future))
            except tk.TclError:
                pass

        fut.add_done_callback(_done_cb)

    def _on_decode_done(self, req_id: int, abs_path: str, index: int, future) -> None:
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
            self._info_var.set(
                f"cannot read: {Path(abs_path).name}")
            self._size_var.set("read error")
            self._try_start_decode()
            return

        self._last_frame = frame_rgb
        self._last_frame_index = index

        h, w = frame_rgb.shape[:2]
        self._fname_var.set(Path(abs_path).name)
        self._size_var.set(f"{w} x {h}")

        # update slider and entry without triggering callbacks
        self._slider_var.set(index)
        self._idx_var.set(str(index + 1))

        # push to downstream
        self.push_output({
            "image":       frame_rgb,
            "frame_index": float(index + 1),
            "frame_count": float(len(self._file_paths)),
        })

        self._try_start_decode()

    def _load_and_push(self, index: int) -> None:
        self._request_decode(index)

    # ── slider callbacks ──────────────────────────────────────────

    def _on_slider_move(self, val) -> None:
        """Called continuously while dragging — update entry only."""
        if self._frame_index_input_connected:
            self._slider_var.set(self._current_index)
            return
        idx = int(float(val))
        self._idx_var.set(str(idx + 1))
        self._fname_var.set(
            Path(self._file_paths[idx]).name
            if self._file_paths else "")
        self._slider_dragging = True

    def _on_slider_release(self, event) -> None:
        """Load and push only when user releases the slider."""
        if self._frame_index_input_connected:
            return
        self._slider_dragging = False
        if self._file_paths:
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
        if self._file_paths:
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

    def _move_batch_current(self, action: str) -> None:
        """Navigate the configured loop manually and emit the selected pair."""
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

        if self._batch_running:
            self._stop_batch("manual batch control")
        loop_index = values[position]
        self._batch_current_var.set(str(loop_index))
        self._emit_batch_frames(loop_index)
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
        if next_idx >= len(self._file_paths):
            # reached end — stop
            self._stop_play()
            return

        self._load_and_push(next_idx)

        interval_ms = max(1, int(1000 / self._fps_var.get()))
        self._after_id = self.canvas.after(
            interval_ms, self._play_tick)

    # ── override push_output for SYNC mode ───────────────────────

    def push_output(self, outputs: dict) -> None:
        """
        ImageSequenceNode is SYNC but needs to push frames
        like a STREAMING node so that VideoPlayOutputNode
        receives them without the user having to manually
        trigger downstream.
        We call the engine callback directly.
        """
        if self._on_output_ready:
            self._on_output_ready(self.node_id, outputs)

    # ── compute (called by engine for SYNC nodes) ─────────────────

    def compute(self, inputs: dict) -> dict:
        """
        Return the last loaded frame when the engine pulls data.
        This handles the case where a downstream node is added
        after a frame is already loaded.
        """
        raw_frame_index = inputs.get("frame_index")
        raw_frame_index_2 = inputs.get("frame_index_2")
        trigger_is_linked = "trigger" in inputs
        if trigger_is_linked != self._trigger_linked:
            self.on_input_link_changed("trigger", trigger_is_linked)
        has_external_index = raw_frame_index is not None
        has_external_index_2 = raw_frame_index_2 is not None
        self._set_external_index_connected(
            has_external_index or has_external_index_2)

        if not self._file_paths:
            return {}

        # When indices are connected, decode both requested frames in this
        # synchronous call.  The existing asynchronous path is intentionally
        # retained for inspector/playback interaction, but cannot be used here:
        # it coalesces requests and would drop one of the two frame requests.
        if has_external_index or has_external_index_2:
            requested_index = self._parse_frame_index(
                raw_frame_index, "frame_index",
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

            if self._last_frame is None or self._last_frame_2 is None:
                return {}
            return {
                "image": self._last_frame,
                "image_2": self._last_frame_2,
                "frame_index": float(requested_index + 1),
                "frame_count": float(len(self._file_paths)),
            }

        if trigger_is_linked:
            if self._truthy_trigger(inputs.get("trigger")):
                self._move_batch_current("next")
                if self._last_frame is None:
                    return {
                        "_skip_downstream": True,
                        "_preserve_cache": True,
                    }
                outputs = {
                    "image": self._last_frame,
                    "frame_index": float(self._current_index + 1),
                    "frame_count": float(len(self._file_paths)),
                }
                if self._last_frame_2 is not None:
                    outputs["image_2"] = self._last_frame_2
                return outputs

            # Low/reset edge from trigger source: suppress downstream recompute.
            return {
                "_skip_downstream": True,
                "_preserve_cache": True,
            }

        if self._last_frame is None or self._last_frame_index != self._current_index:
            return {}
        outputs = {
            "image": self._last_frame,
            "frame_index": float(self._current_index + 1),
            "frame_count": float(len(self._file_paths)),
        }
        if self._last_frame_2 is not None:
            outputs["image_2"] = self._last_frame_2
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
        return max(0, min(index, len(self._file_paths) - 1))

    def _frame_for_index(self, index: int, cached_frame,
                         cached_index: int) -> np.ndarray | None:
        """Return a decoded frame, reusing the matching index cache."""
        if cached_frame is not None and cached_index == index:
            return cached_frame
        abs_path = self._to_absolute(
            self._file_paths[index], self._get_project_base())
        frame = self._decode_frame(abs_path)
        if frame is None:
            self._status_var.set(f"cannot read: {Path(abs_path).name}")
        return frame

    # ── serialization ─────────────────────────────────────────────

    def _to_relative(self, path: str,
                     base: Path | None) -> str:
        """Convert absolute path to relative if base is known."""
        if base is None:
            return path
        try:
            return str(Path(path).relative_to(base))
        except ValueError:
            return path   # different drive on Windows — keep absolute

    def _to_absolute(self, path: str,
                     base: Path | None) -> str:
        """Resolve relative path against base."""
        p = Path(path)
        if p.is_absolute():
            return str(p)
        if base is None:
            return str(p)
        return str((base / p).resolve())

    def _get_project_base(self) -> Path | None:
        return get_project_directory()

    def get_help_text(self) -> str:
        return self.HELP_TEXT

    def get_params(self) -> dict:
        base  = self._get_project_base()
        paths = [self._to_relative(self._to_absolute(p, base), base)
                 for p in self._file_paths]
        method = self._method_var.get()
        pattern = self._to_relative(self._pattern_var.get().strip(), base)

        folder_value = self._folder_var.get().strip()
        if folder_value and folder_value != "(no folder)":
            folder_value = self._to_relative(self._to_absolute(folder_value, base), base)
        elif self._method_var.get() == "folder" and self._file_paths:
            # If folder text was lost/stale, infer it from current file list.
            abs_files = [Path(self._to_absolute(p, base)) for p in self._file_paths]
            if abs_files:
                inferred = str(abs_files[0].parent)
                if all(p.parent == abs_files[0].parent for p in abs_files):
                    folder_value = self._to_relative(inferred, base)

        file_list_text = "\n".join(paths)
        # Excel cell text has practical limits; a very long file list can be
        # truncated and break params_json on reload. In folder mode, the
        # folder path is enough to reconstruct the list.
        if method == "folder" and len(file_list_text) > 12000:
            file_list_text = ""

        return {
            "method":        method,
            "pattern":       pattern,
            "start":         self._start_var.get(),
            "end":           self._end_var.get(),
            "folder":        folder_value or "(no folder)",
            "file_list":     file_list_text,
            "current_index": self._current_index,
            "fps":           self._fps_var.get(),
            "batch_start":   self._batch_start_var.get(),
            "batch_end":     self._batch_end_var.get(),
            "batch_step":    self._batch_step_var.get(),
            "batch_current": self._batch_current_var.get(),
            "batch_frame_1": self._batch_frame_1_var.get(),
            "batch_frame_2": self._batch_frame_2_var.get(),
        }

    def set_params(self, params: dict) -> None:
        self._init_ui_state()
        method = params.get("method", "pattern")
        self._method_var.set(method)
        self._on_method_change()

        base = self._get_project_base()
        pattern_value = str(params.get("pattern", "") or "")
        self._pattern_var.set(
            self._to_relative(self._to_absolute(pattern_value, base), base)
        )
        self._start_var.set(str(params.get("start", "0")))
        self._end_var.set(str(params.get("end", "0")))
        folder_value = str(params.get("folder", "(no folder)") or "(no folder)")
        if folder_value != "(no folder)":
            folder_value = self._to_relative(self._to_absolute(folder_value, base), base)
        self._folder_var.set(folder_value)
        try:
            self._fps_var.set(int(params.get("fps", 5)))
        except Exception:
            self._fps_var.set(5)

        # Restore file list first. This is the most explicit source of truth.
        file_list = params.get("file_list", "")
        if file_list:
            paths = [
                self._to_relative(self._to_absolute(p.strip(), base), base)
                for p in file_list.splitlines()
                if p.strip()]
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
                    else:
                        self._status_var.set("folder restored but no readable images")
                else:
                    self._status_var.set("saved folder not found")
            except Exception:
                self._status_var.set("failed to restore saved folder")

        self._batch_start_var.set(str(params.get("batch_start", "1")))
        self._batch_end_var.set(str(params.get(
            "batch_end", max(1, len(self._file_paths) - 1))))
        self._batch_step_var.set(str(params.get("batch_step", "1")))
        self._batch_current_var.set(str(params.get(
            "batch_current", self._batch_start_var.get())))
        self._batch_frame_1_var.set(str(params.get("batch_frame_1", "i")))
        self._batch_frame_2_var.set(str(params.get("batch_frame_2", "i+1")))
        if self._validate_loop_fields() and self._file_paths:
            self._emit_batch_frames(int(self._batch_current_var.get()))

    def on_destroy(self) -> None:
        with self._decode_lock:
            self._destroyed = True
            self._decode_pending_req = None
        self._stop_play()
        self._stop_batch("destroyed")
        super().on_destroy()
