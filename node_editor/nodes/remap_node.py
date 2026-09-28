# node_editor/nodes/remap_node.py

import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np

from node_editor.base_node import BaseNode
from node_editor.pin_types import PinSchema, PinDef, PinType
from node_editor.execution import ExecutionMode


_HELP_TEXT = """\
Remap Node
==========

PURPOSE
-------
Wraps cv2.remap() to resample an image according to a pair of
per-pixel coordinate maps (e.g. the map1/map2 produced by
cv2.initUndistortRectifyMap, as output by the Undistortion node's
method B / map-cache path).

Unlike the underlying cv2.remap() signature, interpolation method,
border mode, and border value are NOT input pins here -- they are
INSPECTOR SETTINGS. This is deliberate: they are almost always fixed
choices for a given pipeline (e.g. "always use linear interpolation,
always fill with black"), not something that changes frame to frame,
so exposing them as pins would just add three connections you'd wire
to constants every time. Wire actual per-frame DATA (image, map1,
map2) through pins; wire STYLE/POLICY through the inspector.

INPUT PINS
----------
image   IMAGE  (required)
    Source image, RGB uint8, 2D or 3D.

map1    ARRAY  (required)
    x-coordinate map (or combined map from a fixed-point map pair),
    same convention as cv2.remap's map1 -- e.g. straight from an
    Undistortion node's map1 output pin.

map2    ARRAY  (optional)
    y-coordinate map. Leave unconnected if map1 already encodes both
    (CV_16SC2 fixed-point combined maps); most float32 map pairs need
    both map1 and map2 connected.

OUTPUT PINS
-----------
image_out   IMAGE
    The remapped image.

INSPECTOR SETTINGS
-------------------
Interpolation
    Nearest / Linear / Cubic / Lanczos4 / Area. Passed as
    cv2.INTER_* to cv2.remap(). Linear is the default and is correct
    for almost all photographic resampling; Nearest is useful for
    label/index images where blending values would be meaningless.

Border mode
    Constant / Replicate / Reflect / Reflect101 / Wrap. Controls how
    cv2.remap() fills pixels whose source location falls outside the
    original image. Reflect101 (cv2.BORDER_REFLECT_101) matches
    OpenCV's own undistortion defaults and avoids duplicating the
    edge pixel the way plain Reflect does.

Border value
    Only used when Border mode = Constant. Enter one number (applied
    to all channels) or three comma-separated numbers (per RGB
    channel), e.g. "0" or "0, 0, 0" or "255, 0, 0".

Update button
    Re-runs remap with the current inspector settings and pushes the
    result downstream. Useful after changing a setting without
    changing any upstream data.

KEYBOARD SHORTCUT
-----------------
Ctrl-H / Ctrl-h -- show this help window.
"""

_INTERP_OPTIONS = [
    ("nearest", "Nearest", cv2.INTER_NEAREST),
    ("linear", "Linear", cv2.INTER_LINEAR),
    ("cubic", "Cubic", cv2.INTER_CUBIC),
    ("lanczos4", "Lanczos4", cv2.INTER_LANCZOS4),
    ("area", "Area", cv2.INTER_AREA),
]
_INTERP_LABEL_TO_KEY = {label: key for key, label, _ in _INTERP_OPTIONS}
_INTERP_KEY_TO_LABEL = {key: label for key, label, _ in _INTERP_OPTIONS}
_INTERP_KEY_TO_CV = {key: cv for key, _label, cv in _INTERP_OPTIONS}

_BORDER_OPTIONS = [
    ("constant", "Constant", cv2.BORDER_CONSTANT),
    ("replicate", "Replicate", cv2.BORDER_REPLICATE),
    ("reflect", "Reflect", cv2.BORDER_REFLECT),
    ("reflect101", "Reflect101", cv2.BORDER_REFLECT_101),
    ("wrap", "Wrap", cv2.BORDER_WRAP),
]
_BORDER_LABEL_TO_KEY = {label: key for key, label, _ in _BORDER_OPTIONS}
_BORDER_KEY_TO_LABEL = {key: label for key, label, _ in _BORDER_OPTIONS}
_BORDER_KEY_TO_CV = {key: cv for key, _label, cv in _BORDER_OPTIONS}


def _parse_border_value(text: str) -> tuple:
    """
    "0" -> (0,0,0,0)   single number applied to all channels
    "0, 0, 0" -> (0,0,0,0)   per-channel, padded/truncated to 4
    Falls back to (0,0,0,0) on any parse failure.
    """
    text = (text or "").strip()
    if not text:
        return (0, 0, 0, 0)
    try:
        parts = [float(p) for p in text.replace(",", " ").split()]
    except ValueError:
        return (0, 0, 0, 0)
    if not parts:
        return (0, 0, 0, 0)
    if len(parts) == 1:
        parts = parts * 3
    parts = (parts + [0.0, 0.0, 0.0, 0.0])[:4]
    return tuple(parts)


class RemapNode(BaseNode):
    """
    Wraps cv2.remap(). Interpolation, border mode, and border value are
    inspector settings, not input pins -- see _HELP_TEXT for rationale.
    """

    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE = "remap"
    DISPLAY_NAME = "Remap"
    CATEGORY = "process"
    SEARCH_KEYWORDS = ("remap", "resample", "warp", "map1", "map2",
                       "interpolation", "border")
    NODE_WIDTH = 200
    NODE_HEIGHT = 110

    HELP_TEXT = _HELP_TEXT

    _BODY_BG = "#f0f4ee"
    _OUTLINE = "#557a3a"
    _TITLE_FG = "#2a4a1a"
    _STATUS_FG = "#557a3a"

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[
                PinDef("image", PinType.IMAGE, "img", optional=False),
                PinDef("map1", PinType.ARRAY, "map1", optional=False),
                PinDef("map2", PinType.ARRAY, "map2", optional=True),
            ],
            outputs=[
                PinDef("image_out", PinType.IMAGE, "imgOut"),
            ],
        )

    def get_help_text(self) -> str:
        return _HELP_TEXT

    # ── state ─────────────────────────────────────────────────────

    def _init_state(self) -> None:
        if hasattr(self, "_interp_var"):
            return
        self._interp_var = tk.StringVar(value="linear")
        self._border_var = tk.StringVar(value="constant")
        self._border_value_text_var = tk.StringVar(value="0")
        self._status_var = tk.StringVar(value="waiting for image and map1")
        self._interp_combo: ttk.Combobox | None = None
        self._border_combo: ttk.Combobox | None = None
        self._border_value_entry: tk.Entry | None = None
        self._help_popup: tk.Toplevel | None = None

    # ── body ──────────────────────────────────────────────────────

    def build_body(self) -> None:
        self._init_state()
        x, y, w, h = self.x, self.y, self.width, self.height
        self._body_rect = self.canvas.create_rectangle(
            x, y, x + w, y + h, fill=self._BODY_BG, outline=self._OUTLINE,
            width=2, tags=(self.node_id, "node_body"))
        self._title_item = self.canvas.create_text(
            x + w / 2, y + 13, text=self.DISPLAY_NAME,
            font=("Arial", 9, "bold"), fill=self._TITLE_FG, tags=(self.node_id,))

        summary_lbl = tk.Label(
            self.canvas, textvariable=self._interp_var, font=("Arial", 8),
            bg=self._BODY_BG, fg=self._TITLE_FG)
        self.canvas.create_window(x + w / 2, y + h * 0.42, window=summary_lbl,
                                  tags=(self.node_id,))

        status_lbl = tk.Label(
            self.canvas, textvariable=self._status_var, font=("Arial", 7),
            bg=self._BODY_BG, fg=self._STATUS_FG, wraplength=w - 12, justify="center")
        self.canvas.create_window(x + w / 2, y + h - 12, window=status_lbl,
                                  tags=(self.node_id,))

        self._canvas_items += [self._body_rect, self._title_item]

    # ── inspector ─────────────────────────────────────────────────

    def build_inspector(self, parent: tk.Frame) -> None:
        self._init_state()
        win = self._inspector_win
        if win is not None:
            for seq in ("<Control-h>", "<Control-H>"):
                win.bind(seq, lambda e: self._open_help())

        pad = {"padx": 8, "pady": 5}

        interp_frame = tk.LabelFrame(parent, text="Interpolation", font=("Arial", 9), **pad)
        interp_frame.pack(fill="x", **pad)
        self._interp_combo = ttk.Combobox(
            interp_frame,
            values=[label for _k, label, _cv in _INTERP_OPTIONS],
            state="readonly", width=14)
        self._interp_combo.set(_INTERP_KEY_TO_LABEL[self._interp_var.get()])
        self._interp_combo.bind("<<ComboboxSelected>>", self._on_interp_changed)
        self._interp_combo.pack(anchor="w")

        border_frame = tk.LabelFrame(parent, text="Border mode", font=("Arial", 9), **pad)
        border_frame.pack(fill="x", **pad)
        self._border_combo = ttk.Combobox(
            border_frame,
            values=[label for _k, label, _cv in _BORDER_OPTIONS],
            state="readonly", width=14)
        self._border_combo.set(_BORDER_KEY_TO_LABEL[self._border_var.get()])
        self._border_combo.bind("<<ComboboxSelected>>", self._on_border_changed)
        self._border_combo.pack(anchor="w")

        value_row = tk.Frame(border_frame)
        value_row.pack(fill="x", pady=(6, 0))
        tk.Label(value_row, text="Border value:", font=("Arial", 9)).pack(side="left")
        self._border_value_entry = tk.Entry(
            value_row, textvariable=self._border_value_text_var,
            font=("Courier", 9), width=14)
        self._border_value_entry.pack(side="left", padx=(4, 0))
        self._border_value_entry.bind("<Return>", lambda e: self._on_update())
        self._border_value_entry.bind("<FocusOut>", lambda e: self._on_update())
        tk.Label(border_frame, text='e.g. "0" or "0, 0, 0" (only used for Constant)',
                font=("Arial", 8), fg="#666666").pack(anchor="w", pady=(2, 0))

        btn_row = tk.Frame(parent)
        btn_row.pack(fill="x", **pad)
        tk.Button(
            btn_row, text="Update", font=("Arial", 9, "bold"),
            bg="#557a3a", fg="white", activebackground="#668a4a",
            relief=tk.FLAT, padx=10, pady=3, command=self._on_update).pack(side="left")
        tk.Button(btn_row, text="Help (Ctrl-H)", font=("Arial", 8),
                  command=self._open_help).pack(side="right")

        tk.Label(parent, textvariable=self._status_var, font=("Arial", 9),
                fg="#2a4a1a", anchor="w", justify="left").pack(fill="x", **pad)

        self._update_border_value_state()

    def _on_interp_changed(self, _event=None) -> None:
        label = self._interp_combo.get()
        self._interp_var.set(_INTERP_LABEL_TO_KEY.get(label, "linear"))
        self._on_update()

    def _on_border_changed(self, _event=None) -> None:
        label = self._border_combo.get()
        self._border_var.set(_BORDER_LABEL_TO_KEY.get(label, "constant"))
        self._update_border_value_state()
        self._on_update()

    def _update_border_value_state(self) -> None:
        if self._border_value_entry is None or not self._border_value_entry.winfo_exists():
            return
        state = "normal" if self._border_var.get() == "constant" else "disabled"
        self._border_value_entry.configure(state=state)

    def _on_update(self) -> None:
        if self._request_downstream:
            self._request_downstream(self.node_id)

    # ── help ──────────────────────────────────────────────────────

    def _open_help(self) -> None:
        if self._help_popup is not None and self._help_popup.winfo_exists():
            self._help_popup.lift()
            return
        popup = tk.Toplevel()
        popup.title("Remap - Help")
        popup.geometry("680x560")
        try:
            px, py = self.canvas.winfo_pointerxy()
            popup.geometry(f"+{px + 16}+{py + 16}")
        except Exception:
            pass
        body = tk.Frame(popup, bg="#f8f8f8", padx=10, pady=8)
        body.pack(fill="both", expand=True)
        txt = tk.Text(body, font=("Courier", 9), bg="#f8f8f8", fg="#222222",
                      wrap=tk.WORD, relief=tk.FLAT)
        vsb = ttk.Scrollbar(body, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", _HELP_TEXT)
        txt.configure(state="disabled")
        tk.Button(body, text="Close", font=("Arial", 9),
                  command=popup.destroy).pack(anchor="e", pady=(8, 0))
        popup.bind("<Escape>", lambda _e: popup.destroy())
        popup.protocol("WM_DELETE_WINDOW", popup.destroy)
        self._help_popup = popup

    # ── compute ───────────────────────────────────────────────────

    def compute(self, inputs: dict) -> dict:
        self._init_state()
        image = inputs.get("image")
        map1 = inputs.get("map1")
        map2 = inputs.get("map2")

        if image is None or not isinstance(image, np.ndarray):
            self._status_var.set("waiting for image")
            return {}
        if map1 is None or not isinstance(map1, np.ndarray):
            self._status_var.set("waiting for map1")
            return {}

        interp_cv = _INTERP_KEY_TO_CV.get(self._interp_var.get(), cv2.INTER_LINEAR)
        border_cv = _BORDER_KEY_TO_CV.get(self._border_var.get(), cv2.BORDER_CONSTANT)
        border_value = _parse_border_value(self._border_value_text_var.get())

        try:
            # This project's images are RGB (see UndistortNode/other
            # image nodes); cv2.remap itself is channel-order agnostic,
            # so no BGR conversion is needed here.
            out = cv2.remap(
                image, map1, map2, interpolation=interp_cv,
                borderMode=border_cv, borderValue=border_value)
        except cv2.error as e:
            self._status_var.set(f"cv2 error: {e}")
            self.set_status("error", "#cc0000")
            return {}
        except Exception as e:
            self._status_var.set(f"error: {e}")
            self.set_status("error", "#cc0000")
            return {}

        self._status_var.set(
            f"ok  {out.shape[1]}x{out.shape[0]}  "
            f"{_INTERP_KEY_TO_LABEL[self._interp_var.get()]} / "
            f"{_BORDER_KEY_TO_LABEL[self._border_var.get()]}")
        self.set_status("ok", "#557a3a")
        return {"image_out": out}

    # ── serialization ─────────────────────────────────────────────

    def get_params(self) -> dict:
        return {
            "interp": self._interp_var.get(),
            "border": self._border_var.get(),
            "border_value_text": self._border_value_text_var.get(),
        }

    def set_params(self, params: dict) -> None:
        self._init_state()
        interp = str(params.get("interp", "linear"))
        self._interp_var.set(interp if interp in _INTERP_KEY_TO_LABEL else "linear")
        border = str(params.get("border", "constant"))
        self._border_var.set(border if border in _BORDER_KEY_TO_LABEL else "constant")
        self._border_value_text_var.set(str(params.get("border_value_text", "0")))

    def close_inspector(self) -> None:
        super().close_inspector()
        self._interp_combo = None
        self._border_combo = None
        self._border_value_entry = None

    def on_destroy(self) -> None:
        if self._help_popup is not None and self._help_popup.winfo_exists():
            self._help_popup.destroy()
        super().on_destroy()