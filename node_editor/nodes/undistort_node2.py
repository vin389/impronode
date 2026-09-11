# node_editor/nodes/undistort_node2.py

import tkinter as tk
from tkinter import ttk
import threading
import hashlib

import cv2
import numpy as np

from node_editor.base_node import BaseNode
from node_editor.pin_types import PinSchema, PinDef, PinType
from node_editor.execution import ExecutionMode


# ---------------------------------------------------------------------------
# Help text
# ---------------------------------------------------------------------------

_HELP_NODE = """\
Undistortion Node
=================

INPUT PINS
----------
cmat         ARRAY  (required)
             3x3 camera matrix from camera calibration.

dvec         ARRAY  (required)
             Distortion coefficient vector (1xN or Nx1).
             Supports 4, 5, 8, 12, or 14 coefficients.

image        IMAGE  (optional)
             Image to undistort (H x W x 3, uint8, RGB).
             If not connected, image_out will be empty.

points       ARRAY  (optional)
             2D image points to undistort, shape (N, 2)
             or (N, 1, 2), float32 or float64.
             If not connected, points_out will be empty.

new_cmat     ARRAY  (optional)
             Override new camera matrix (3x3).
             Whether this pin or the inspector value takes
             priority is controlled by the radio buttons
             in the inspector.

new_img_size ARRAY  (optional)
             Override new image size [W, H].
             Same priority radio button logic applies.

OUTPUT PINS
-----------
image_out    IMAGE
             Undistorted image. Empty if no image input.

points_out   ARRAY
             Undistorted points in pixel coordinates of
             the undistorted image. Empty if no points.

new_cmat_out ARRAY
             The actual new camera matrix used.

roi          ARRAY
             Valid pixel ROI [x, y, w, h] returned by
             getOptimalNewCameraMatrix. Useful for cropping.

map1         ARRAY
             Undistortion/rectification map 1 (method B).
             Can be reused by a downstream Remap node.

map2         ARRAY
             Undistortion/rectification map 2 (method B).
"""

_HELP_INSPECTOR = """\
Undistortion Inspector Settings
================================

METHOD
------
Method A: cv2.undistort()
  Simple one-call undistortion. Recomputes the mapping
  every time. Suitable for single images.

Method B: getOptimalNewCameraMatrix + initUndistortRectifyMap + remap
  Pre-computes map1/map2 once and reuses them for
  subsequent frames with the same camera parameters.
  Strongly recommended for image sequences.
  The maps are automatically recomputed only when cmat,
  dvec, new_cmat, or new_img_size change.

ALPHA (method B only)
-----
Controls how much black border appears in the result.
  0.0 = all black border pixels are cropped out
        (tightest crop, no wasted pixels)
  1.0 = all original pixels are retained
        (largest output, black corners visible)
  0.5 = balanced compromise

NEW CAMERA MATRIX SOURCE
------------------------
Three mutually exclusive options:

  "Calculate automatically"
    Uses cv2.getOptimalNewCameraMatrix() with the
    alpha value above and the two convenience options:
      - Equal focal lengths: sets fx_new = fy_new = (fx+fy)/2
      - Center principal point: sets cx_new = W/2-.5, cy_new = H/2-.5
    This is the recommended starting point.

  "Use inspector text box"
    Uses the 3x3 matrix typed directly in the text box.
    Ignores any connected new_cmat pin.

  "Use input pin (if connected)"
    Uses the new_cmat array arriving on the input pin.
    Falls back to automatic calculation if no pin is connected.

NEW IMAGE SIZE SOURCE
---------------------
Same three-option radio button logic, applied to image size.

  Scale factor (automatic mode):
    new_size = round(original_size * scale)
    Scale = 1.0 means same size as the input image.

  If no image is connected, set the original image size
  manually in the "Original image size" fields so the
  node can compute the correct output dimensions.

UPDATE BUTTON
-------------
Re-runs the undistortion with the current inspector
values and pushes the results to all output pins.
Useful when you change inspector settings without
changing upstream input data.

UNDISTORT POINTS  (P parameter)
---------------------------------
Points are undistorted with P = new_cmat, so the
output coordinates are in the pixel space of the
undistorted image. This is the correct choice for
tracking applications where you need pixel positions.
"""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _parse_array(text: str, cols: int) -> np.ndarray | None:
    try:
        nums = [float(t)
                for t in text.replace(",", " ")
                             .replace("\t", " ")
                             .split()
                if t.strip()]
        if not nums:
            return None
        return np.array(nums, dtype=np.float64).reshape(-1, cols)
    except Exception:
        return None


def _arr_to_text(arr: np.ndarray, fmt: str = "%.6g") -> str:
    a = np.asarray(arr)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    return "\n".join(
        "  ".join(fmt % v for v in row) for row in a)


def _param_hash(*arrays) -> str:
    """Compute a short hash of a set of numpy arrays for cache invalidation."""
    h = hashlib.md5()
    for a in arrays:
        if a is None:
            h.update(b"None")
        else:
            h.update(np.asarray(a).tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# UndistortNode
# ---------------------------------------------------------------------------

class UndistortNode(BaseNode):
    """
    Undistortion node: corrects lens distortion in images and/or 2D points.

    Wraps cv2.undistort() (method A) and
    cv2.getOptimalNewCameraMatrix + cv2.initUndistortRectifyMap + cv2.remap
    (method B, default).

    Method B caches the undistortion maps and only recomputes them when
    the camera parameters or output size change, making it efficient for
    image sequences.

    Points are undistorted with cv2.undistortPoints(P=new_cmat) so the
    output coordinates remain in pixel space of the undistorted image.
    """

    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE      = "undistort"
    DISPLAY_NAME   = "Undistortion"
    CATEGORY       = "process"
    NODE_WIDTH     = 200
    NODE_HEIGHT    = 110

    # Help text exposed to the NodeEditorApp hotkey handler.
    HELP_TEXT = _HELP_NODE

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[
                PinDef("cmat",        PinType.ARRAY,  "cmat",
                       optional=False),
                PinDef("dvec",        PinType.ARRAY,  "dvec",
                       optional=False),
                PinDef("image",       PinType.IMAGE,  "image",
                       optional=True),
                PinDef("points",      PinType.ARRAY,  "points",
                       optional=True),
                PinDef("new_cmat",    PinType.ARRAY,  "newCmat",
                       optional=True),
                PinDef("new_img_size",PinType.ARRAY,  "newSize",
                       optional=True),
            ],
            outputs=[
                PinDef("image_out",   PinType.IMAGE,  "imageOut"),
                PinDef("points_out",  PinType.ARRAY,  "ptsOut"),
                PinDef("new_cmat_out",PinType.ARRAY,  "newCmat"),
                PinDef("roi",         PinType.ARRAY,  "roi"),
                PinDef("map1",        PinType.ARRAY,  "map1"),
                PinDef("map2",        PinType.ARRAY,  "map2"),
            ]
        )

    # ── state init ────────────────────────────────────────────────

    def _init_state(self) -> None:
        if hasattr(self, "_method_var"):
            return

        # method
        self._method_var = tk.StringVar(value="B")

        # alpha (method B)
        self._alpha_var = tk.DoubleVar(value=1.0)

        # convenience options (auto mode)
        self._equal_f_var  = tk.BooleanVar(value=False)
        self._center_pp_var = tk.BooleanVar(value=False)

        # new camera matrix source: "auto" | "inspector" | "pin"
        self._ncmat_src_var = tk.StringVar(value="auto")
        self._ncmat_text = ""           # inspector text box content

        # new image size source: "auto" | "inspector" | "pin"
        self._nsize_src_var = tk.StringVar(value="auto")
        self._nsize_scale_var = tk.DoubleVar(value=1.0)

        # original image size (used when no image pin is connected)
        self._orig_w_var = tk.IntVar(value=1920)
        self._orig_h_var = tk.IntVar(value=1080)

        # cache for method B maps
        self._map1: np.ndarray | None = None
        self._map2: np.ndarray | None = None
        self._cached_new_cmat: np.ndarray | None = None
        self._cached_roi: tuple | None = None
        self._map_hash: str = ""

        # last computed outputs (returned by compute() between updates)
        self._last_outputs: dict = {}

        # status
        self._status_var = tk.StringVar(value="waiting")

        # inspector widget refs
        self._ncmat_text_widget: tk.Text | None = None
        self._alpha_sb:          tk.Spinbox | None = None
        self._help_popup:        tk.Toplevel | None = None

    def open_inspector(self) -> None:
        super().open_inspector()
        if self._inspector_win is None or not self._inspector_win.winfo_exists():
            return

        self._inspector_win.update_idletasks()
        req_w = max(640, self._inspector_win.winfo_reqwidth())
        req_h = max(480, self._inspector_win.winfo_reqheight())
        self._inspector_win.geometry(f"{req_w * 3 / 2}x{req_h * 3}")
        self._inspector_win.update_idletasks()

    # ── build_body ────────────────────────────────────────────────

    def build_body(self) -> None:
        self._init_state()
        x, y, w, h = self.x, self.y, self.width, self.height

        self._body_rect = self.canvas.create_rectangle(
            x, y, x+w, y+h,
            fill="#f3f6fb", outline="#4c8fd8", width=2,
            tags=(self.node_id, "node_body"))
        self._title_item = self.canvas.create_text(
            x+w/2, y+13,
            text=self.DISPLAY_NAME,
            font=("Arial", 9, "bold"), fill="#1f3b5b",
            tags=(self.node_id,))

        method_lbl = tk.Label(
            self.canvas,
            textvariable=self._method_var,
            font=("Arial", 8), bg="#f3f6fb", fg="#2f4d6d")
        self.canvas.create_window(
            x+w/2, y+33, window=method_lbl,
            tags=(self.node_id,))

        status_lbl = tk.Label(
            self.canvas, textvariable=self._status_var,
            font=("Arial", 7), bg="#f3f6fb", fg="#4a4a4a",
            wraplength=w-10, justify="center")
        self.canvas.create_window(
            x+w/2, y+h-12, window=status_lbl,
            tags=(self.node_id,))

        self._canvas_items += [self._body_rect,
                               self._title_item]

    # ── build_inspector ───────────────────────────────────────────

    def build_inspector(self, parent: tk.Frame) -> None:
        self._init_state()

        # Ctrl-H binding inside inspector window
        insp_win = self._inspector_win
        if insp_win is not None:
            insp_win.bind(
                "<Control-h>",
                lambda e: self._show_inspector_help())
            insp_win.bind(
                "<Control-H>",
                lambda e: self._show_inspector_help())

        # scrollable inner frame
        outer = tk.Frame(parent)
        outer.pack(fill="both", expand=True)

        canvas = tk.Canvas(outer, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient="vertical",
                             command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas)
        win_id = canvas.create_window(
            (0, 0), window=inner, anchor="nw")
        inner.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")))
        canvas.bind(
            "<Configure>",
            lambda e: canvas.itemconfigure(
                win_id, width=e.width))

        self._build_inspector_content(inner)

    def _build_inspector_content(self,
                                  parent: tk.Frame) -> None:
        pad = {"padx": 6, "pady": 3}

        # ── method ────────────────────────────────────────────────
        mf = tk.LabelFrame(parent, text="Undistortion method",
                           font=("Arial", 9), **pad)
        mf.pack(fill="x", **pad)

        tk.Radiobutton(
            mf,
            text=("Method A  —  cv2.undistort()  "
                  "(simple, recomputes every frame)"),
            variable=self._method_var, value="A",
            font=("Arial", 9),
            justify=tk.LEFT,
            wraplength=560,
            anchor="w",
            command=self._on_method_change).pack(anchor="w", fill="x")
        tk.Radiobutton(
            mf,
            text=("Method B  —  getOptimalNewCameraMatrix "
                  "+ initUndistortRectifyMap + remap  "
                  "(cached maps, fast for sequences)"),
            variable=self._method_var, value="B",
            font=("Arial", 9),
            justify=tk.LEFT,
            wraplength=560,
            anchor="w",
            command=self._on_method_change).pack(anchor="w", fill="x")

        # ── alpha ─────────────────────────────────────────────────
        self._alpha_frame = tk.LabelFrame(
            parent,
            text="alpha  (0=crop black borders  "
                 "→  1=keep all pixels)",
            font=("Arial", 9), **pad)
        self._alpha_frame.pack(fill="x", **pad)

        alpha_row = tk.Frame(self._alpha_frame)
        alpha_row.pack(fill="x")

        alpha_slider = tk.Scale(
            alpha_row,
            variable=self._alpha_var,
            from_=0.0, to=1.0,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            length=260,
            showvalue=False)
        alpha_slider.pack(side="left")

        self._alpha_sb = tk.Spinbox(
            alpha_row,
            from_=0.0, to=1.0,
            increment=0.05,
            textvariable=self._alpha_var,
            width=6, font=("Arial", 9),
            format="%.2f")
        self._alpha_sb.pack(side="left", padx=4)

        # ── new camera matrix source ──────────────────────────────
        ncmat_frame = tk.LabelFrame(
            parent, text="New camera matrix source",
            font=("Arial", 9), **pad)
        ncmat_frame.pack(fill="x", **pad)

        tk.Radiobutton(
            ncmat_frame,
            text="Calculate automatically  "
                 "(getOptimalNewCameraMatrix + options below)",
            variable=self._ncmat_src_var, value="auto",
            font=("Arial", 9),
            justify=tk.LEFT,
            wraplength=560,
            anchor="w",
            command=self._on_ncmat_src_change).pack(anchor="w", fill="x")

        # convenience options (auto mode)
        self._auto_options_frame = tk.Frame(ncmat_frame)
        self._auto_options_frame.pack(
            fill="x", padx=12, pady=2)
        tk.Checkbutton(
            self._auto_options_frame,
            text="Equal focal lengths  "
                 "(fx_new = fy_new = (fx+fy)/2)",
            variable=self._equal_f_var,
            font=("Arial", 9)).pack(anchor="w")
        tk.Checkbutton(
            self._auto_options_frame,
            text="Center principal point  "
                 "(cx_new = W/2-0.5,  cy_new = H/2-0.5)",
            variable=self._center_pp_var,
            font=("Arial", 9)).pack(anchor="w")

        tk.Radiobutton(
            ncmat_frame,
            text="Use inspector text box  "
                 "(ignores connected pin)",
            variable=self._ncmat_src_var,
            value="inspector",
            font=("Arial", 9),
            command=self._on_ncmat_src_change).pack(anchor="w")

        # inspector text box (inspector mode)
        self._ncmat_inspector_frame = tk.Frame(ncmat_frame)
        self._ncmat_inspector_frame.pack(
            fill="x", padx=12, pady=2)
        tk.Label(
            self._ncmat_inspector_frame,
            text="New camera matrix (3x3):",
            font=("Arial", 8)).pack(anchor="w")
        self._ncmat_text_widget = tk.Text(
            self._ncmat_inspector_frame,
            width=40, height=3,
            font=("Courier", 8))
        self._ncmat_text_widget.pack(fill="x")
        if self._ncmat_text:
            self._ncmat_text_widget.insert(
                "1.0", self._ncmat_text)

        tk.Radiobutton(
            ncmat_frame,
            text="Use input pin  "
                 "(falls back to auto if pin not connected)",
            variable=self._ncmat_src_var, value="pin",
            font=("Arial", 9),
            command=self._on_ncmat_src_change).pack(anchor="w")

        # ── new image size source ─────────────────────────────────
        nsize_frame = tk.LabelFrame(
            parent, text="New image size source",
            font=("Arial", 9), **pad)
        nsize_frame.pack(fill="x", **pad)

        tk.Radiobutton(
            nsize_frame,
            text="Scale from original  "
                 "(scale × original size)",
            variable=self._nsize_src_var, value="auto",
            font=("Arial", 9),
            command=self._on_nsize_src_change).pack(anchor="w")

        # scale (auto mode)
        self._scale_frame = tk.Frame(nsize_frame)
        self._scale_frame.pack(
            fill="x", padx=12, pady=2)
        scale_row = tk.Frame(self._scale_frame)
        scale_row.pack(fill="x")
        tk.Label(scale_row, text="Scale:",
                 font=("Arial", 9)).pack(side="left")
        tk.Spinbox(
            scale_row,
            from_=0.1, to=10.0,
            increment=0.1,
            textvariable=self._nsize_scale_var,
            width=6, font=("Arial", 9),
            format="%.2f").pack(side="left", padx=4)
        tk.Label(
            scale_row,
            text="(1.0 = same as original)",
            font=("Arial", 8), fg="#666666").pack(
            side="left")

        tk.Radiobutton(
            nsize_frame,
            text="Use inspector text box  "
                 "(ignores connected pin)",
            variable=self._nsize_src_var,
            value="inspector",
            font=("Arial", 9),
            command=self._on_nsize_src_change).pack(anchor="w")

        # inspector text box for size (inspector mode)
        self._nsize_inspector_frame = tk.Frame(nsize_frame)
        self._nsize_inspector_frame.pack(
            fill="x", padx=12, pady=2)
        tk.Label(
            self._nsize_inspector_frame,
            text="New image size [W  H]:",
            font=("Arial", 8)).pack(anchor="w")
        self._nsize_text_var = tk.StringVar(value="1920 1080")
        tk.Entry(
            self._nsize_inspector_frame,
            textvariable=self._nsize_text_var,
            font=("Courier", 8), width=20).pack(anchor="w")

        tk.Radiobutton(
            nsize_frame,
            text="Use input pin  "
                 "(falls back to scale if pin not connected)",
            variable=self._nsize_src_var, value="pin",
            font=("Arial", 9),
            command=self._on_nsize_src_change).pack(anchor="w")

        # ── original image size ───────────────────────────────────
        orig_frame = tk.LabelFrame(
            parent,
            text="Original image size  "
                 "(used when no image pin is connected)",
            font=("Arial", 9), **pad)
        orig_frame.pack(fill="x", **pad)

        orig_row = tk.Frame(orig_frame)
        orig_row.pack(fill="x")
        tk.Label(orig_row, text="W:",
                 font=("Arial", 9)).pack(side="left")
        tk.Spinbox(
            orig_row,
            from_=1, to=100000,
            textvariable=self._orig_w_var,
            width=7, font=("Arial", 9)).pack(
            side="left", padx=2)
        tk.Label(orig_row, text="  H:",
                 font=("Arial", 9)).pack(side="left")
        tk.Spinbox(
            orig_row,
            from_=1, to=100000,
            textvariable=self._orig_h_var,
            width=7, font=("Arial", 9)).pack(
            side="left", padx=2)

        # ── status + update button ────────────────────────────────
        bottom_row = tk.Frame(parent)
        bottom_row.pack(fill="x", **pad)

        tk.Button(
            bottom_row,
            text="Update  (re-run undistortion)",
            font=("Arial", 9, "bold"),
            bg="#224466", fg="white",
            activebackground="#335577",
            relief=tk.FLAT, padx=8, pady=3,
            command=self._on_update).pack(side="left")

        tk.Button(
            bottom_row,
            text="Help  (Ctrl-H)",
            font=("Arial", 8),
            command=self._show_inspector_help).pack(
            side="right", padx=4)

        tk.Label(
            parent, textvariable=self._status_var,
            font=("Arial", 9), fg="#446688",
            anchor="w", justify="left").pack(
            fill="x", **pad)

        # sync visibility
        self._on_method_change()
        self._on_ncmat_src_change()
        self._on_nsize_src_change()

    # ── inspector helpers ─────────────────────────────────────────

    def _on_method_change(self) -> None:
        if not hasattr(self, "_alpha_frame"):
            return
        if self._method_var.get() == "B":
            self._alpha_frame.pack(
                fill="x", padx=6, pady=3)
        else:
            self._alpha_frame.pack_forget()

    def _on_ncmat_src_change(self) -> None:
        if not hasattr(self, "_auto_options_frame"):
            return
        src = self._ncmat_src_var.get()
        # auto options
        if src == "auto":
            self._auto_options_frame.pack(
                fill="x", padx=12, pady=2)
        else:
            self._auto_options_frame.pack_forget()
        # inspector text box
        if src == "inspector":
            self._ncmat_inspector_frame.pack(
                fill="x", padx=12, pady=2)
        else:
            self._ncmat_inspector_frame.pack_forget()

    def _on_nsize_src_change(self) -> None:
        if not hasattr(self, "_scale_frame"):
            return
        src = self._nsize_src_var.get()
        if src == "auto":
            self._scale_frame.pack(
                fill="x", padx=12, pady=2)
            self._nsize_inspector_frame.pack_forget()
        elif src == "inspector":
            self._scale_frame.pack_forget()
            self._nsize_inspector_frame.pack(
                fill="x", padx=12, pady=2)
        else:
            self._scale_frame.pack_forget()
            self._nsize_inspector_frame.pack_forget()

    def _on_update(self) -> None:
        """Re-run undistortion with current inspector values
        and push results downstream."""
        self._sync_ncmat_text()
        # invalidate cache so parameters are re-read
        self._map_hash = ""
        if self._request_downstream:
            self._request_downstream(self.node_id)

    def _sync_ncmat_text(self) -> None:
        if (self._ncmat_text_widget is not None
                and self._ncmat_text_widget.winfo_exists()):
            self._ncmat_text = \
                self._ncmat_text_widget.get(
                    "1.0", tk.END).strip()

    # ── help popups ───────────────────────────────────────────────

    def _show_inspector_help(self) -> None:
        self._show_help_popup(
            "Undistortion Inspector Help",
            _HELP_INSPECTOR)

    def _show_node_help(self) -> None:
        self._show_help_popup(
            "Undistortion Node Help",
            _HELP_NODE)

    def get_help_text(self) -> str:
        """Called by NodeEditorApp when Ctrl-H is pressed
        on a selected node (node body, not inspector)."""
        return _HELP_NODE

    def _show_help_popup(self,
                          title: str,
                          text: str) -> None:
        if (self._help_popup is not None
                and self._help_popup.winfo_exists()):
            self._help_popup.lift()
            return

        popup = tk.Toplevel()
        popup.title(title)
        popup.resizable(True, True)
        popup.geometry("640x500")

        try:
            px, py = (self.canvas
                      .winfo_pointerxy())
            popup.geometry(
                f"+{px + 16}+{py + 16}")
        except Exception:
            pass

        body = tk.Frame(
            popup, bg="#111111",
            padx=10, pady=8)
        body.pack(fill="both", expand=True)

        txt = tk.Text(
            body,
            font=("Courier", 9),
            bg="#111111", fg="#dddddd",
            wrap=tk.WORD,
            relief=tk.FLAT)
        vsb = ttk.Scrollbar(
            body, orient="vertical",
            command=txt.yview)
        txt.configure(
            yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", text)
        txt.configure(state="disabled")

        tk.Button(
            body, text="Close",
            command=popup.destroy).pack(
            anchor="e",
            pady=(8, 0))

        popup.bind(
            "<Escape>",
            lambda _e: popup.destroy())
        popup.protocol(
            "WM_DELETE_WINDOW",
            popup.destroy)
        self._help_popup = popup

    # ── undistortion logic ────────────────────────────────────────

    def _resolve_orig_size(
        self,
        image: np.ndarray | None
    ) -> tuple[int, int]:
        """Return (W, H) of the original image."""
        if image is not None:
            h, w = image.shape[:2]
            return w, h
        return (self._orig_w_var.get(),
                self._orig_h_var.get())

    def _resolve_new_size(
        self,
        orig_w: int,
        orig_h: int,
        pin_size: np.ndarray | None
    ) -> tuple[int, int]:
        """Resolve output image size from the three sources."""
        src = self._nsize_src_var.get()
        if src == "pin" and pin_size is not None:
            a = np.asarray(pin_size).ravel()
            return int(a[0]), int(a[1])
        if src == "inspector":
            t = getattr(
                self, "_nsize_text_var",
                None)
            if t is not None:
                arr = _parse_array(t.get(), 1)
                if arr is not None and arr.size >= 2:
                    return (int(arr.ravel()[0]),
                            int(arr.ravel()[1]))
        # auto (scale)
        scale = self._nsize_scale_var.get()
        return (max(1, int(round(orig_w * scale))),
                max(1, int(round(orig_h * scale))))

    def _resolve_new_cmat(
        self,
        cmat: np.ndarray,
        dvec: np.ndarray,
        orig_w: int,
        orig_h: int,
        new_w: int,
        new_h: int,
        pin_ncmat: np.ndarray | None,
    ) -> tuple[np.ndarray, tuple]:
        """
        Resolve new camera matrix and ROI from the three sources.
        Returns (new_cmat_3x3, roi_tuple).
        """
        src = self._ncmat_src_var.get()

        if src == "pin" and pin_ncmat is not None:
            nc = np.asarray(
                pin_ncmat,
                dtype=np.float64).reshape(3, 3)
            return nc, (0, 0, new_w, new_h)

        if src == "inspector":
            self._sync_ncmat_text()
            m = _parse_array(
                self._ncmat_text, 3)
            if m is not None and m.shape == (3, 3):
                return m, (0, 0, new_w, new_h)

        # auto: getOptimalNewCameraMatrix
        alpha = self._alpha_var.get() \
            if self._method_var.get() == "B" \
            else 1.0
        nc, roi = cv2.getOptimalNewCameraMatrix(
            cmat, dvec,
            (orig_w, orig_h),
            alpha,
            (new_w, new_h))

        # apply convenience adjustments
        if self._equal_f_var.get():
            f_avg = (nc[0, 0] + nc[1, 1]) / 2.0
            nc[0, 0] = f_avg
            nc[1, 1] = f_avg
        if self._center_pp_var.get():
            nc[0, 2] = new_w / 2.0 - .5
            nc[1, 2] = new_h / 2.0 - .5

        return nc, roi

    def _ensure_maps(
        self,
        cmat: np.ndarray,
        dvec: np.ndarray,
        new_cmat: np.ndarray,
        new_w: int,
        new_h: int,
    ) -> None:
        """
        Compute undistortion maps only when parameters change.
        Uses a MD5 hash of all relevant arrays for cache validation.
        """
        key = _param_hash(
            cmat, dvec, new_cmat,
            np.array([new_w, new_h]))
        if key == self._map_hash:
            return   # cache hit — skip recomputation
        self._map1, self._map2 = \
            cv2.initUndistortRectifyMap(
                cmat, dvec, None, new_cmat,
                (new_w, new_h),
                cv2.CV_32FC1)
        self._map_hash = key

    def _undistort_image(
        self,
        image: np.ndarray,
        cmat: np.ndarray,
        dvec: np.ndarray,
        new_cmat: np.ndarray,
        new_w: int,
        new_h: int,
    ) -> np.ndarray:
        # convert RGB → BGR for OpenCV, process, convert back
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if self._method_var.get() == "A":
            out_bgr = cv2.undistort(
                bgr, cmat, dvec, None, new_cmat)
        else:
            self._ensure_maps(
                cmat, dvec, new_cmat,
                new_w, new_h)
            out_bgr = cv2.remap(
                bgr, self._map1, self._map2,
                cv2.INTER_LINEAR)
        return cv2.cvtColor(
            out_bgr, cv2.COLOR_BGR2RGB)

    def _undistort_points(
        self,
        points: np.ndarray,
        cmat: np.ndarray,
        dvec: np.ndarray,
        new_cmat: np.ndarray,
    ) -> np.ndarray:
        """
        Undistort 2D points with P=new_cmat so output
        coordinates are in the pixel space of the
        undistorted image.
        """
        pts = np.asarray(
            points, dtype=np.float32).reshape(
            -1, 1, 2)
        out = cv2.undistortPoints(
            pts, cmat, dvec, P=new_cmat)
        return out.reshape(-1, 2)

    # ── compute ───────────────────────────────────────────────────

    def compute(self, inputs: dict) -> dict:
        cmat = inputs.get("cmat")
        dvec = inputs.get("dvec")
        if cmat is None or dvec is None:
            self._status_var.set(
                "waiting for cmat and dvec")
            return self._last_outputs

        try:
            cmat = np.asarray(
                cmat,
                dtype=np.float64).reshape(3, 3)
            dvec = np.asarray(
                dvec,
                dtype=np.float64).reshape(1, -1)
        except Exception as e:
            self._status_var.set(
                f"cmat/dvec parse error: {e}")
            return self._last_outputs

        image    = inputs.get("image")
        points   = inputs.get("points")
        pin_nc   = inputs.get("new_cmat")
        pin_sz   = inputs.get("new_img_size")

        orig_w, orig_h = self._resolve_orig_size(image)
        new_w,  new_h  = self._resolve_new_size(
            orig_w, orig_h, pin_sz)
        new_cmat, roi  = self._resolve_new_cmat(
            cmat, dvec,
            orig_w, orig_h,
            new_w, new_h,
            pin_nc)

        outputs: dict = {
            "new_cmat_out": new_cmat,
            "roi": np.array(roi,
                            dtype=np.int32),
        }

        # method B: always ensure maps are ready
        if self._method_var.get() == "B":
            self._ensure_maps(
                cmat, dvec, new_cmat,
                new_w, new_h)
            outputs["map1"] = self._map1
            outputs["map2"] = self._map2

        # undistort image
        if image is not None and \
                isinstance(image, np.ndarray):
            try:
                img_out = self._undistort_image(
                    image, cmat, dvec,
                    new_cmat, new_w, new_h)
                outputs["image_out"] = img_out
                self._status_var.set(
                    f"ok  {new_w}×{new_h}"
                    f"  method {self._method_var.get()}")
            except Exception as e:
                self._status_var.set(
                    f"image error: {e}")
                self.set_status(
                    "error", "#cc0000")

        # undistort points
        if points is not None and \
                isinstance(points, np.ndarray):
            try:
                pts_out = self._undistort_points(
                    points, cmat, dvec, new_cmat)
                outputs["points_out"] = pts_out
            except Exception as e:
                self._status_var.set(
                    f"points error: {e}")

        self._last_outputs = outputs
        return outputs

    # ── serialization ─────────────────────────────────────────────

    def get_params(self) -> dict:
        self._sync_ncmat_text()
        nsize_text = ""
        if hasattr(self, "_nsize_text_var"):
            nsize_text = self._nsize_text_var.get()
        return {
            "method":        self._method_var.get(),
            "alpha":         self._alpha_var.get(),
            "equal_f":       self._equal_f_var.get(),
            "center_pp":     self._center_pp_var.get(),
            "ncmat_src":     self._ncmat_src_var.get(),
            "ncmat_text":    self._ncmat_text,
            "nsize_src":     self._nsize_src_var.get(),
            "nsize_scale":   self._nsize_scale_var.get(),
            "nsize_text":    nsize_text,
            "orig_w":        self._orig_w_var.get(),
            "orig_h":        self._orig_h_var.get(),
        }

    def set_params(self, params: dict) -> None:
        self._init_state()
        self._method_var.set(
            params.get("method", "B"))
        self._alpha_var.set(
            float(params.get("alpha", 1.0)))
        self._equal_f_var.set(
            bool(params.get("equal_f", False)))
        self._center_pp_var.set(
            bool(params.get("center_pp", False)))
        self._ncmat_src_var.set(
            params.get("ncmat_src", "auto"))
        self._ncmat_text = str(
            params.get("ncmat_text", ""))
        self._nsize_src_var.set(
            params.get("nsize_src", "auto"))
        self._nsize_scale_var.set(
            float(params.get("nsize_scale", 1.0)))
        self._orig_w_var.set(
            int(params.get("orig_w", 1920)))
        self._orig_h_var.set(
            int(params.get("orig_h", 1080)))
        if hasattr(self, "_nsize_text_var"):
            self._nsize_text_var.set(
                str(params.get("nsize_text",
                               "1920 1080")))

    def close_inspector(self) -> None:
        self._sync_ncmat_text()
        if hasattr(self, "_nsize_text_var"):
            pass   # StringVar persists
        super().close_inspector()
        self._ncmat_text_widget = None
        self._alpha_sb          = None

    def on_destroy(self) -> None:
        if (self._help_popup is not None
                and self._help_popup.winfo_exists()):
            self._help_popup.destroy()
        self._map1 = None
        self._map2 = None
        super().on_destroy()