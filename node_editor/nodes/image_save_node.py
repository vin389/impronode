# node_editor/nodes/image_save_node.py

import os
import time
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np

from node_editor.base_node import BaseNode
from node_editor.execution import ExecutionMode
from node_editor.pin_types import PinDef, PinSchema, PinType
from node_editor.project_context import get_project_directory


_HELP_TEXT = (
    "Image Save Node\n\n"
    "Purpose:\n"
    "- Continuously (or on trigger) write incoming IMAGE frames to disk\n"
    "  as a numbered file sequence, e.g. STEP0001.jpg, STEP0002.jpg, ...\n"
    "  Designed to sit after image-producing/processing nodes such as\n"
    "  Image Sequence -> Undistortion -> Image Save, to export a whole\n"
    "  processed sequence to a new folder.\n\n"
    "Input Pins:\n"
    "- image [IMAGE]: frame to save (required). RGB, uint8, 2D or 3D.\n"
    "- index [SCALAR, optional]: overrides the internal auto-increment\n"
    "  counter for this save. Wire ImageSequenceNode.frame_index here to\n"
    "  keep output filenames aligned with the source sequence's numbers,\n"
    "  even if frames are skipped upstream. When connected, the internal\n"
    "  counter is NOT advanced -- the external index fully owns numbering.\n"
    "- trig [TRIGGER, optional]: only used in 'Triggered' save mode (see\n"
    "  inspector). Ignored entirely in 'Continuous' mode.\n"
    "- reset [TRIGGER, optional]: resets the internal counter back to\n"
    "  'Start index' from the inspector.\n\n"
    "Output Pins:\n"
    "- saved_path [STRING]: absolute path of the most recently written\n"
    "  file (not produced if the save failed or was skipped as 'exists').\n"
    "- index_out [SCALAR]: the index value used for the most recent save.\n"
    "- save_done [TRIGGER]: fires once per successful (or skipped-as-\n"
    "  exists) save. Connect to a LoopNode's advance pin, or any other\n"
    "  trigger consumer, to synchronise a batch pipeline with disk I/O.\n\n"
    "Inspector Settings:\n"
    "- Output pattern: a path with a printf-style placeholder, e.g.\n"
    "    C:/projects/images/undistorted/STEP%04d.jpg\n"
    "  Use %04d for zero-padded 4-digit numbers, %d for unpadded, etc.\n"
    "  A pattern with NO '%' placeholder at all is treated as a fixed\n"
    "  filename (every save overwrites the same file) -- useful for a\n"
    "  'always latest snapshot' output.\n"
    "  Stored relative to the project (.xlsx) file's folder, exactly like\n"
    "  Image Sequence's pattern field, so the project folder can be\n"
    "  copied to another computer/drive intact.\n"
    "- Browse folder...: pick a destination folder; rewrites the pattern's\n"
    "  directory portion, keeping (or creating) a default filename template.\n"
    "- Start index / Step: internal counter parameters. 'Next index' is\n"
    "  directly editable too, so you can resume or jump numbering.\n"
    "- Save mode:\n"
    "    Continuous  -- save on every compute() call with a valid image\n"
    "                   (the default; needs no trig wiring at all).\n"
    "    Triggered   -- only save on a new trig pulse (edge-detected, so\n"
    "                   both boolean pulse-style and ever-incrementing\n"
    "                   counter-style trigger sources work correctly).\n"
    "- Skip if file exists: if the target path already exists, do not\n"
    "  overwrite it -- but still advance numbering and fire save_done,\n"
    "  so a batch pipeline is not stalled by pre-existing files.\n"
    "- JPEG quality / PNG compression: passed to cv2.imwrite when the\n"
    "  output extension is .jpg/.jpeg or .png respectively; ignored for\n"
    "  other extensions.\n"
    "- Reset Counter button: same as the 'reset' trigger pin.\n\n"
    "Notes:\n"
    "- Missing output directories are created automatically.\n"
    "- On write failure, the counter is NOT advanced and save_done does\n"
    "  NOT fire, so the same index is retried on the next attempt rather\n"
    "  than silently skipping a frame.\n"
    "- This node writes synchronously (no background thread). For very\n"
    "  high frame-rate continuous sources (e.g. a live webcam at 30 FPS)\n"
    "  this may add visible latency; for the batch/offline export use\n"
    "  case this node is designed for, this is not a practical concern.\n"
)


class ImageSaveNode(BaseNode):
    """
    Writes incoming IMAGE frames to a numbered file sequence on disk.

    See _HELP_TEXT (Ctrl-H in the app) for the full pin/inspector reference.
    """

    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE = "image_save"
    DISPLAY_NAME = "Image Save"
    CATEGORY = "output"   # new category: file-writing / sink nodes
    SEARCH_KEYWORDS = ("save", "write", "export", "output", "imwrite", "sequence")
    NODE_WIDTH = 220
    NODE_HEIGHT = 120

    HELP_TEXT = _HELP_TEXT

    _DEFAULT_START = 1
    _DEFAULT_STEP = 1
    _DEFAULT_JPEG_QUALITY = 95
    _DEFAULT_PNG_COMPRESSION = 3

    def _init_state(self) -> None:
        if hasattr(self, "_status_var"):
            return

        self._pattern_var = tk.StringVar(value="")
        self._start_var = tk.IntVar(value=self._DEFAULT_START)
        self._step_var = tk.IntVar(value=self._DEFAULT_STEP)
        # The authoritative counter value. A plain tk.IntVar so the user
        # can see AND directly edit/resume it from the inspector.
        self._next_index_var = tk.IntVar(value=self._DEFAULT_START)

        self._save_mode_var = tk.StringVar(value="continuous")
        self._skip_if_exists_var = tk.BooleanVar(value=False)
        self._jpeg_quality_var = tk.IntVar(value=self._DEFAULT_JPEG_QUALITY)
        self._png_compression_var = tk.IntVar(value=self._DEFAULT_PNG_COMPRESSION)

        self._status_var = tk.StringVar(value="no output pattern set")
        self._last_saved_var = tk.StringVar(value="(nothing saved yet)")
        self._saved_count = 0

        # --- trigger / reset edge-detection state ----------------------
        # Same edge-detection design as OpticalFlowNode/TemplateMatchNode:
        # fires on a classic rising edge (falsy -> truthy) for boolean
        # pulse-style sources, AND on any value change while already
        # truthy, for ever-incrementing counter-style sources.
        self._last_trig_value = None
        self._trig_was_truthy = False
        self._last_reset_value = None
        self._reset_was_truthy = False

        self._last_saved_path: str | None = None
        self._last_warn_key: str | None = None

        # Inspector widget refs (exist only while the inspector is open)
        self._pattern_entry: tk.Entry | None = None

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[
                PinDef("image", PinType.IMAGE, "img"),
                PinDef("index", PinType.SCALAR, "idx", optional=True),
                PinDef("trig", PinType.TRIGGER, "trig", optional=True),
                PinDef("reset", PinType.TRIGGER, "reset", optional=True),
            ],
            outputs=[
                PinDef("saved_path", PinType.STRING, "path"),
                PinDef("index_out", PinType.SCALAR, "idx"),
                PinDef("save_done", PinType.TRIGGER, "done"),
            ],
        )

    def get_help_text(self) -> str:
        return _HELP_TEXT

    # ── body ──────────────────────────────────────────────────────────

    def build_body(self) -> None:
        self._init_state()
        x, y, w, h = self.x, self.y, self.width, self.height

        self._body_rect = self.canvas.create_rectangle(
            x, y, x + w, y + h,
            fill="#fdeee0", outline="#cc7a33", width=2,
            tags=(self.node_id, "node_body"),
        )
        self._title_item = self.canvas.create_text(
            x + w / 2, y + 13,
            text=self.DISPLAY_NAME,
            font=("Arial", 9, "bold"), fill="#8a4a1a",
            tags=(self.node_id,),
        )

        last_lbl = tk.Label(
            self.canvas, textvariable=self._last_saved_var,
            font=("Arial", 7), bg="#fdeee0", fg="#8a4a1a",
            wraplength=w - 12, justify="center",
        )
        self.canvas.create_window(
            x + w / 2, y + h // 2 + 6, window=last_lbl,
            tags=(self.node_id,),
        )

        status_lbl = tk.Label(
            self.canvas, textvariable=self._status_var,
            font=("Arial", 7), bg="#fdeee0", fg="#996633",
            wraplength=w - 12, justify="center",
        )
        self.canvas.create_window(
            x + w / 2, y + h - 12, window=status_lbl,
            tags=(self.node_id,),
        )

        self._canvas_items += [self._body_rect, self._title_item]

    # ── inspector ─────────────────────────────────────────────────────

    def build_inspector(self, parent: tk.Frame) -> None:
        self._init_state()
        pad = {"padx": 6, "pady": 4}

        # ── output pattern ───────────────────────────────────────────
        pat_frame = tk.LabelFrame(parent, text="Output pattern", font=("Arial", 9), **pad)
        pat_frame.pack(fill="x", **pad)

        self._pattern_entry = tk.Entry(pat_frame, textvariable=self._pattern_var,
                                       font=("Courier", 9))
        self._pattern_entry.pack(fill="x")

        tk.Label(
            pat_frame,
            text='e.g.  C:/projects/images/undistorted/STEP%04d.jpg\n'
                 'No "%" placeholder = fixed filename, overwritten every save.',
            font=("Arial", 8), fg="#666666", justify="left", anchor="w",
        ).pack(fill="x", pady=(2, 4))

        tk.Button(pat_frame, text="Browse folder...", font=("Arial", 8),
                  command=self._on_browse_folder).pack(anchor="w")

        # ── numbering ─────────────────────────────────────────────────
        num_frame = tk.LabelFrame(parent, text="Numbering", font=("Arial", 9), **pad)
        num_frame.pack(fill="x", **pad)

        row = tk.Frame(num_frame)
        row.pack(fill="x", pady=2)
        tk.Label(row, text="Start index:", font=("Arial", 9), width=14, anchor="w").pack(side="left")
        tk.Spinbox(row, from_=0, to=10_000_000, textvariable=self._start_var,
                   width=10, font=("Arial", 9)).pack(side="left")

        row = tk.Frame(num_frame)
        row.pack(fill="x", pady=2)
        tk.Label(row, text="Step:", font=("Arial", 9), width=14, anchor="w").pack(side="left")
        tk.Spinbox(row, from_=1, to=100_000, textvariable=self._step_var,
                   width=10, font=("Arial", 9)).pack(side="left")

        row = tk.Frame(num_frame)
        row.pack(fill="x", pady=2)
        tk.Label(row, text="Next index:", font=("Arial", 9), width=14, anchor="w").pack(side="left")
        tk.Spinbox(row, from_=0, to=10_000_000, textvariable=self._next_index_var,
                   width=10, font=("Arial", 9)).pack(side="left")
        tk.Label(row, text="(editable; overridden per-save by the index pin)",
                 font=("Arial", 8), fg="#666666").pack(side="left", padx=(6, 0))

        tk.Button(num_frame, text="Reset Counter", font=("Arial", 8),
                  command=self._on_reset_counter).pack(anchor="w", pady=(4, 0))

        # ── save mode ─────────────────────────────────────────────────
        mode_frame = tk.LabelFrame(parent, text="Save mode", font=("Arial", 9), **pad)
        mode_frame.pack(fill="x", **pad)

        tk.Radiobutton(
            mode_frame, text="Continuous (save every incoming frame)",
            variable=self._save_mode_var, value="continuous", font=("Arial", 9),
        ).pack(anchor="w")
        tk.Radiobutton(
            mode_frame, text="Triggered (save only on a new trig pulse)",
            variable=self._save_mode_var, value="triggered", font=("Arial", 9),
        ).pack(anchor="w")

        tk.Checkbutton(
            mode_frame, text="Skip save if file already exists",
            variable=self._skip_if_exists_var, font=("Arial", 9),
        ).pack(anchor="w", pady=(4, 0))

        # ── format options ───────────────────────────────────────────
        fmt_frame = tk.LabelFrame(parent, text="Format options", font=("Arial", 9), **pad)
        fmt_frame.pack(fill="x", **pad)

        row = tk.Frame(fmt_frame)
        row.pack(fill="x", pady=2)
        tk.Label(row, text="JPEG quality (0-100):", font=("Arial", 9), width=20, anchor="w").pack(side="left")
        tk.Spinbox(row, from_=0, to=100, textvariable=self._jpeg_quality_var,
                   width=8, font=("Arial", 9)).pack(side="left")

        row = tk.Frame(fmt_frame)
        row.pack(fill="x", pady=2)
        tk.Label(row, text="PNG compression (0-9):", font=("Arial", 9), width=20, anchor="w").pack(side="left")
        tk.Spinbox(row, from_=0, to=9, textvariable=self._png_compression_var,
                   width=8, font=("Arial", 9)).pack(side="left")

        # ── status ────────────────────────────────────────────────────
        stat_frame = tk.LabelFrame(parent, text="Status", font=("Arial", 9), **pad)
        stat_frame.pack(fill="x", **pad)
        tk.Label(stat_frame, textvariable=self._last_saved_var, font=("Arial", 9),
                 fg="#8a4a1a", anchor="w", justify="left", wraplength=380).pack(fill="x")
        tk.Label(stat_frame, textvariable=self._status_var, font=("Arial", 9),
                 fg="#996633", anchor="w", justify="left", wraplength=380).pack(fill="x")

    def close_inspector(self) -> None:
        super().close_inspector()
        self._pattern_entry = None

    # ── path helpers (same round-trip pattern as ImageSequenceNode) ────

    def _get_project_base(self) -> Path | None:
        return get_project_directory()

    @staticmethod
    def _to_relative(path: str, base: Path | None) -> str:
        if base is None or not path:
            return path
        try:
            return os.path.relpath(str(path), str(base))
        except Exception:
            return path

    @staticmethod
    def _to_absolute(path: str, base: Path | None) -> str:
        p = Path(path)
        if p.is_absolute() or base is None:
            return str(p)
        return str((base / p).resolve())

    # ── inspector actions ────────────────────────────────────────────

    def _on_browse_folder(self) -> None:
        base = self._get_project_base()
        initial = str(base) if base else "."
        folder = filedialog.askdirectory(title="Select output folder", initialdir=initial)
        if not folder:
            return
        existing = self._pattern_var.get().strip()
        filename_part = Path(existing).name if existing else "STEP%04d.jpg"
        new_pattern = str(Path(folder) / filename_part)
        rel = self._to_relative(self._to_absolute(new_pattern, base), base)
        self._pattern_var.set(rel)

    def _on_reset_counter(self) -> None:
        self._next_index_var.set(self._start_var.get())
        self._status_var.set("counter reset")

    # ── save mechanics ───────────────────────────────────────────────

    def _resolve_target_path(self, index: int) -> Path:
        base = self._get_project_base()
        pattern_abs = self._to_absolute(self._pattern_var.get().strip(), base)
        try:
            resolved = pattern_abs % index
        except TypeError:
            if "%" not in pattern_abs:
                # No placeholder at all: treat as a fixed, always-overwritten filename.
                resolved = pattern_abs
            else:
                raise
        return Path(resolved)

    @staticmethod
    def _prepare_for_imwrite(image: np.ndarray) -> np.ndarray:
        """Convert this framework's RGB convention to the BGR order cv2.imwrite expects."""
        arr = np.asarray(image)
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3 and arr.shape[2] == 3:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        if arr.ndim == 3 and arr.shape[2] == 4:
            return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGRA)
        raise ValueError(f"unsupported image shape for saving: {arr.shape}")

    def _imwrite_params(self, abs_path: Path) -> list:
        ext = abs_path.suffix.lower()
        if ext in (".jpg", ".jpeg"):
            return [cv2.IMWRITE_JPEG_QUALITY, int(self._jpeg_quality_var.get())]
        if ext == ".png":
            return [cv2.IMWRITE_PNG_COMPRESSION, int(self._png_compression_var.get())]
        return []

    def _warn_once(self, key: str, title: str, message: str) -> None:
        if self._last_warn_key == key:
            return
        self._last_warn_key = key
        print(f"[ImageSaveNode][Warning] {title}: {message}", flush=True)
        try:
            messagebox.showwarning(title, message)
        except Exception:
            pass

    @staticmethod
    def _coerce_bool_like(value) -> bool:
        if isinstance(value, str):
            text = value.strip().lower()
            if text in ("1", "true", "yes", "y", "on"):
                return True
            if text in ("0", "false", "no", "n", "off", ""):
                return False
        try:
            return bool(int(float(value)))
        except Exception:
            return bool(value)

    def on_upstream_changed(self) -> None:
        # Only the ephemeral edge-detection state is reset here -- the
        # counter and inspector settings are durable and must survive
        # rewiring, same reasoning as why get_params()/set_params() persist
        # them across project save/load.
        self._last_trig_value = None
        self._trig_was_truthy = False
        self._last_reset_value = None
        self._reset_was_truthy = False
        self._last_warn_key = None

    def compute(self, inputs: dict) -> dict:
        self._init_state()

        # ── reset pin (edge-detected, same logic as trig) ──────────────
        raw_reset = inputs.get("reset")
        reset_truthy_now = raw_reset is not None and self._coerce_bool_like(raw_reset)
        reset_value_changed = raw_reset is not None and raw_reset != self._last_reset_value
        if reset_truthy_now and (not self._reset_was_truthy or reset_value_changed):
            self._next_index_var.set(self._start_var.get())
            self._status_var.set("counter reset (via pin)")
        self._reset_was_truthy = reset_truthy_now
        self._last_reset_value = raw_reset

        image = inputs.get("image")
        if image is None or not isinstance(image, np.ndarray):
            self._status_var.set("no image")
            return {}

        pattern = self._pattern_var.get().strip()
        if not pattern:
            self._status_var.set("no output pattern set")
            return {}

        # ── save-mode gating ────────────────────────────────────────
        if self._save_mode_var.get() == "triggered":
            raw_trig = inputs.get("trig")
            truthy_now = raw_trig is not None and self._coerce_bool_like(raw_trig)
            value_changed = raw_trig is not None and raw_trig != self._last_trig_value
            new_pulse = truthy_now and (not self._trig_was_truthy or value_changed)
            self._trig_was_truthy = truthy_now
            self._last_trig_value = raw_trig

            if not new_pulse:
                if raw_trig is None:
                    self._status_var.set("frozen: trig not connected/pulsed")
                    self.set_status("frozen", "#666666")
                return {"_skip_downstream": True, "_preserve_cache": True}

        # ── determine index for this save ──────────────────────────
        # The incoming idx pin always wins over the inspector's numbering.
        # Only when no external index is connected do we use the UI counter.
        raw_index = inputs.get("index")
        external_index = raw_index is not None
        if external_index:
            index_used = int(round(float(raw_index)))
        else:
            index_used = int(self._next_index_var.get())

        try:
            abs_path = self._resolve_target_path(index_used)
        except Exception as e:
            self._status_var.set(f"pattern error: {e}")
            self.set_status("error", "#cc0000")
            return {}

        if self._skip_if_exists_var.get() and abs_path.exists():
            self._status_var.set(f"skipped (exists): {abs_path.name}")
            self.set_status("ok", "#55aa55")
        else:
            try:
                parent = abs_path.parent
                if str(parent):
                    os.makedirs(parent, exist_ok=True)

                out_img = self._prepare_for_imwrite(image)
                params = self._imwrite_params(abs_path)
                ok = cv2.imwrite(str(abs_path), out_img, params)
                if not ok:
                    raise IOError(f"cv2.imwrite returned False for {abs_path}")

                self._saved_count += 1
                self._status_var.set(
                    f"ok: saved #{self._saved_count} "
                    f"(time: {time.strftime('%H:%M:%S')})"
                )
                self.set_status("ok", "#55aa55")
            except Exception as e:
                self._status_var.set(f"error: {e}")
                self.set_status("error", "#cc0000")
                # Do not advance the counter and do not fire save_done on
                # failure, so the same index is retried on the next attempt
                # rather than silently skipping a frame.
                return {}

        self._last_saved_path = str(abs_path)
        self._last_saved_var.set(abs_path.name)

        if not external_index:
            self._next_index_var.set(index_used + int(self._step_var.get()))

        self._fire_counter = getattr(self, "_fire_counter", 0) + 1
        return {
            "saved_path": self._last_saved_path,
            "index_out": float(index_used),
            "save_done": self._fire_counter,
        }

    # ── serialization ────────────────────────────────────────────────

    def get_params(self) -> dict:
        self._init_state()
        base = self._get_project_base()
        pattern_rel = self._to_relative(self._to_absolute(self._pattern_var.get().strip(), base), base)
        return {
            "pattern": pattern_rel,
            "start": self._start_var.get(),
            "step": self._step_var.get(),
            "next_index": self._next_index_var.get(),
            "save_mode": self._save_mode_var.get(),
            "skip_if_exists": bool(self._skip_if_exists_var.get()),
            "jpeg_quality": self._jpeg_quality_var.get(),
            "png_compression": self._png_compression_var.get(),
        }

    def set_params(self, params: dict) -> None:
        self._init_state()
        base = self._get_project_base()
        pattern_value = str(params.get("pattern", "") or "")
        self._pattern_var.set(self._to_relative(self._to_absolute(pattern_value, base), base))
        self._start_var.set(int(params.get("start", self._DEFAULT_START)))
        self._step_var.set(int(params.get("step", self._DEFAULT_STEP)))
        self._next_index_var.set(int(params.get("next_index", self._start_var.get())))
        self._save_mode_var.set(params.get("save_mode", "continuous"))
        self._skip_if_exists_var.set(bool(params.get("skip_if_exists", False)))
        self._jpeg_quality_var.set(int(params.get("jpeg_quality", self._DEFAULT_JPEG_QUALITY)))
        self._png_compression_var.set(int(params.get("png_compression", self._DEFAULT_PNG_COMPRESSION)))