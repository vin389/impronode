# node_editor/nodes/project_points_node.py

import tkinter as tk
from tkinter import ttk
import numpy as np
import cv2

from node_editor.base_node import BaseNode
from node_editor.pin_types import PinSchema, PinDef, PinType
from node_editor.execution import ExecutionMode


# ---------------------------------------------------------------------------
# Help text
# ---------------------------------------------------------------------------

_HELP_TEXT = """\
Project Points Node
===================

PURPOSE
-------
Wraps cv2.projectPoints() to project 3D object points into
2D image coordinates using a calibrated camera model.

Equivalent to:
    imagePoints, jacobian = cv2.projectPoints(
        objectPoints, rvec, tvec, cameraMatrix,
        distCoeffs, aspectRatio=aspectRatio)

INPUT PINS
----------
object_pts   ARRAY  (required)
    3D object points.  Accepted shapes:
      (N, 3)     — N points, each [X, Y, Z]
      (N, 1, 3)  — OpenCV convention
      (N*3,)     — flat 1D array, reshaped automatically
    dtype is coerced to float64 automatically.
    If the total number of elements is not a multiple of 3,
    an error is reported and no output is produced.

rvec         ARRAY  (required)
    Rotation vector (Rodrigues), shape (3,) or (3,1) or (1,3).
    Coerced to float64 and reshaped to (3,1).

tvec         ARRAY  (required)
    Translation vector, shape (3,) or (3,1) or (1,3).
    Coerced to float64 and reshaped to (3,1).

cmat         ARRAY  (required)
    3x3 camera matrix (intrinsic parameters).
    Coerced to float64 and reshaped to (3,3).

dvec         ARRAY  (required)
    Distortion coefficients, any length supported by OpenCV
    (4, 5, 8, 12, or 14 elements).
    Coerced to float64 and reshaped to (1,N).

aspect_ratio SCALAR  (optional)
    Free scaling parameter for the aspect ratio fix.
    Default 0 (no fix).  Passed as the aspectRatio
    argument to cv2.projectPoints().
    When connected, overrides the inspector spinbox value.

OUTPUT PINS
-----------
image_pts    ARRAY
    Projected 2D image coordinates, shape (N, 2), float64.
    Each row is [u, v] in pixel coordinates.

jacobian     ARRAY
    Jacobian matrix returned by cv2.projectPoints().
    Shape (N*2, 14+N*3) — see OpenCV docs for column layout.
    Useful for uncertainty propagation and sensitivity analysis.

INSPECTOR SETTINGS
------------------
Aspect ratio
    Default value for the aspectRatio parameter when the
    aspect_ratio input pin is not connected.
    Range: any float, typically 0.0 (disabled) or 1.0.

NOTES
-----
- All required inputs must be connected for any output to
  be produced.  Missing inputs are reported in the status line.
- Input shapes and dtypes are coerced automatically so you
  can connect arrays directly from calibration nodes without
  manual reshaping.
- The jacobian output is the raw OpenCV jacobian; it is not
  normalised or processed in any way.

KEYBOARD SHORTCUT
-----------------
Ctrl-H / Ctrl-h  —  show this help window.
"""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _coerce_object_points(
        arr: np.ndarray) -> np.ndarray | None:
    """
    Reshape and coerce objectPoints to (N,1,3) float64.
    Returns None if total elements are not a multiple of 3.
    """
    arr = np.asarray(arr, dtype=np.float64)
    n   = arr.size
    if n % 3 != 0:
        return None
    return arr.reshape(-1, 1, 3)


def _coerce_vec3(arr: np.ndarray) -> np.ndarray:
    """Coerce a rotation or translation vector to (3,1) float64."""
    return np.asarray(
        arr, dtype=np.float64).ravel()[:3].reshape(3, 1)


def _coerce_cmat(arr: np.ndarray) -> np.ndarray:
    """Coerce camera matrix to (3,3) float64."""
    return np.asarray(
        arr, dtype=np.float64).reshape(3, 3)


def _coerce_dvec(arr: np.ndarray) -> np.ndarray:
    """Coerce distortion coefficients to (1,N) float64."""
    return np.asarray(
        arr, dtype=np.float64).ravel().reshape(1, -1)


# ---------------------------------------------------------------------------
# ProjectPointsNode
# ---------------------------------------------------------------------------

class ProjectPointsNode(BaseNode):
    """
    Wraps cv2.projectPoints().

    Projects 3D object points to 2D image coordinates using
    a pinhole camera model with distortion.

    Input shapes are coerced automatically:
      objectPoints → (N,1,3) float64
      rvec/tvec    → (3,1)   float64
      cmat         → (3,3)   float64
      dvec         → (1,K)   float64

    Output imagePoints is reshaped to (N,2) float64.
    """

    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE      = "project_points"
    DISPLAY_NAME   = "Project Points"
    CATEGORY       = "process"
    NODE_WIDTH     = 200
    NODE_HEIGHT    = 110

    HELP_TEXT = _HELP_TEXT

    _BODY_BG  = "#f0fff0"
    _OUTLINE  = "#3a9a3a"
    _TITLE_FG = "#1a5a1a"
    _STATUS_FG = "#2a7a2a"

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[
                PinDef("object_pts",   PinType.ARRAY,
                       "objPts",  optional=False),
                PinDef("rvec",         PinType.ARRAY,
                       "rvec",    optional=False),
                PinDef("tvec",         PinType.ARRAY,
                       "tvec",    optional=False),
                PinDef("cmat",         PinType.ARRAY,
                       "cmat",    optional=False),
                PinDef("dvec",         PinType.ARRAY,
                       "dvec",    optional=False),
                PinDef("aspect_ratio", PinType.SCALAR,
                       "aspect",  optional=True),
            ],
            outputs=[
                PinDef("image_pts", PinType.ARRAY,
                       "imgPts"),
                PinDef("jacobian",  PinType.ARRAY,
                       "jacobian"),
            ]
        )

    def get_help_text(self) -> str:
        return _HELP_TEXT

    # ── state ─────────────────────────────────────────────────────

    def _init_state(self) -> None:
        if hasattr(self, "_aspect_var"):
            return
        self._aspect_var  = tk.DoubleVar(value=0.0)
        self._status_var  = tk.StringVar(
            value="waiting")
        self._info_var    = tk.StringVar(value="")
        self._help_popup: tk.Toplevel | None = None

    # ── build_body ────────────────────────────────────────────────

    def build_body(self) -> None:
        self._init_state()
        x, y, w, h = self.x, self.y, self.width, self.height

        self._body_rect = self.canvas.create_rectangle(
            x, y, x+w, y+h,
            fill=self._BODY_BG,
            outline=self._OUTLINE, width=2,
            tags=(self.node_id, "node_body"))

        self._title_item = self.canvas.create_text(
            x+w/2, y+13,
            text=self.DISPLAY_NAME,
            font=("Arial", 9, "bold"),
            fill=self._TITLE_FG,
            tags=(self.node_id,))

        info_lbl = tk.Label(
            self.canvas,
            textvariable=self._info_var,
            font=("Arial", 8, "bold"),
            bg=self._BODY_BG,
            fg=self._TITLE_FG)
        self.canvas.create_window(
            x+w/2, y+h//2+4,
            window=info_lbl,
            tags=(self.node_id,))

        status_lbl = tk.Label(
            self.canvas,
            textvariable=self._status_var,
            font=("Arial", 7),
            bg=self._BODY_BG,
            fg=self._STATUS_FG,
            wraplength=w-12,
            justify="center")
        self.canvas.create_window(
            x+w/2, y+h-12,
            window=status_lbl,
            tags=(self.node_id,))

        self._canvas_items += [
            self._body_rect, self._title_item]

    # ── build_inspector ───────────────────────────────────────────

    def build_inspector(self,
                         parent: tk.Frame) -> None:
        self._init_state()

        win = self._inspector_win
        if win is not None:
            for seq in ("<Control-h>",
                        "<Control-H>"):
                win.bind(
                    seq,
                    lambda e: self._open_help())

        pad = {"padx": 8, "pady": 6}

        # aspect ratio setting
        ar_frame = tk.LabelFrame(
            parent,
            text="Aspect ratio parameter",
            font=("Arial", 9), **pad)
        ar_frame.pack(fill="x", **pad)

        ar_row = tk.Frame(ar_frame)
        ar_row.pack(fill="x", pady=4)

        tk.Label(
            ar_row,
            text="aspectRatio:",
            font=("Arial", 9),
            width=16, anchor="w").pack(
            side="left")

        tk.Spinbox(
            ar_row,
            from_=-1e6, to=1e6,
            increment=0.1,
            textvariable=self._aspect_var,
            width=10, font=("Arial", 9),
            format="%.4f").pack(side="left")

        tk.Label(
            ar_row,
            text="  (0 = disabled)",
            font=("Arial", 8),
            fg="#666666").pack(side="left")

        tk.Label(
            ar_frame,
            text="Note: if the aspect_ratio input pin\n"
                 "is connected, it overrides this value.",
            font=("Arial", 8),
            fg="#666666",
            justify="left").pack(anchor="w")

        # pin summary
        pins_frame = tk.LabelFrame(
            parent,
            text="Input pin summary",
            font=("Arial", 9), **pad)
        pins_frame.pack(fill="x", **pad)

        summary = (
            "object_pts  ARRAY   (N,3) or flat  "
            "→ auto-reshaped\n"
            "rvec        ARRAY   (3,) or (3,1)\n"
            "tvec        ARRAY   (3,) or (3,1)\n"
            "cmat        ARRAY   (3,3)\n"
            "dvec        ARRAY   (1,K)  K=4,5,8,12,14\n"
            "aspect_ratio SCALAR  optional (default 0)"
        )
        tk.Label(
            pins_frame,
            text=summary,
            font=("Courier", 8),
            fg="#333333",
            justify="left").pack(
            anchor="w", padx=4, pady=4)

        # output pin summary
        out_frame = tk.LabelFrame(
            parent,
            text="Output pin summary",
            font=("Arial", 9), **pad)
        out_frame.pack(fill="x", **pad)

        out_summary = (
            "image_pts  ARRAY  (N,2) float64"
            "  — projected [u,v] pixels\n"
            "jacobian   ARRAY  (N*2, 14+N*3)"
            "  — OpenCV jacobian"
        )
        tk.Label(
            out_frame,
            text=out_summary,
            font=("Courier", 8),
            fg="#333333",
            justify="left").pack(
            anchor="w", padx=4, pady=4)

        # status
        tk.Label(
            parent,
            textvariable=self._status_var,
            font=("Arial", 9),
            fg="#2a7a2a",
            anchor="w",
            justify="left").pack(
            fill="x", **pad)

        # help button
        tk.Button(
            parent,
            text="Help  (Ctrl-H)",
            font=("Arial", 8),
            command=self._open_help).pack(
            anchor="e", **pad)

    # ── compute ───────────────────────────────────────────────────

    def compute(self,
                inputs: dict) -> dict:
        # ── gather required inputs ────────────────────────────────
        missing = []
        for pin in ("object_pts", "rvec",
                    "tvec", "cmat", "dvec"):
            if inputs.get(pin) is None:
                missing.append(pin)

        if missing:
            msg = "missing: " + ", ".join(missing)
            self._status_var.set(msg)
            self._info_var.set("")
            return {}

        # ── coerce inputs ─────────────────────────────────────────
        raw_obj = np.asarray(
            inputs["object_pts"])
        obj_pts = _coerce_object_points(raw_obj)
        if obj_pts is None:
            msg = (
                f"object_pts size {raw_obj.size}"
                f" not multiple of 3")
            self._status_var.set(msg)
            self._info_var.set("shape error")
            self.set_status("error", "#cc0000")
            return {}

        try:
            rvec = _coerce_vec3(
                inputs["rvec"])
            tvec = _coerce_vec3(
                inputs["tvec"])
            cmat = _coerce_cmat(
                inputs["cmat"])
            dvec = _coerce_dvec(
                inputs["dvec"])
        except Exception as e:
            self._status_var.set(
                f"coerce error: {e}")
            self._info_var.set("input error")
            self.set_status("error", "#cc0000")
            return {}

        # ── aspect ratio ─────────────────────────────────────────
        ar_pin = inputs.get("aspect_ratio")
        if ar_pin is not None:
            try:
                aspect = float(ar_pin)
            except (TypeError, ValueError):
                aspect = self._aspect_var.get()
        else:
            aspect = self._aspect_var.get()

        # ── call cv2.projectPoints ────────────────────────────────
        try:
            img_pts, jacobian = cv2.projectPoints(
                obj_pts,
                rvec, tvec,
                cmat, dvec,
                aspectRatio=aspect)
        except cv2.error as e:
            self._status_var.set(
                f"cv2 error: {e}")
            self._info_var.set("cv2 error")
            self.set_status("error", "#cc0000")
            return {}
        except Exception as e:
            self._status_var.set(
                f"error: {e}")
            self._info_var.set("error")
            self.set_status("error", "#cc0000")
            return {}

        # ── reshape outputs ───────────────────────────────────────
        img_pts_2d = img_pts.reshape(
            -1, 2).astype(np.float64)
        jacobian   = np.asarray(
            jacobian, dtype=np.float64)

        N = img_pts_2d.shape[0]
        self._info_var.set(
            f"N={N}  →  ({N},2)")
        self._status_var.set(
            f"ok  aspectRatio={aspect:.4g}")
        self.set_status("ok", "#3a9a3a")

        return {
            "image_pts": img_pts_2d,
            "jacobian":  jacobian,
        }

    # ── help popup ────────────────────────────────────────────────

    def _open_help(self) -> None:
        if (self._help_popup is not None
                and self._help_popup.winfo_exists()):
            self._help_popup.lift()
            return

        popup = tk.Toplevel()
        popup.title(
            "Project Points Node — Help")
        popup.geometry("680x560")
        popup.resizable(True, True)
        try:
            px, py = (
                self.canvas.winfo_pointerxy())
            popup.geometry(
                f"+{px+16}+{py+16}")
        except Exception:
            pass

        body = tk.Frame(
            popup, bg="#f8fff8",
            padx=10, pady=8)
        body.pack(fill="both", expand=True)

        txt = tk.Text(
            body, font=("Courier", 9),
            bg="#f8fff8", fg="#1a3a1a",
            wrap=tk.WORD, relief=tk.FLAT)
        vsb = ttk.Scrollbar(
            body, orient="vertical",
            command=txt.yview)
        txt.configure(
            yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", _HELP_TEXT)
        txt.configure(state="disabled")

        tk.Button(
            body, text="Close",
            font=("Arial", 9),
            command=popup.destroy).pack(
            anchor="e", pady=(8, 0))

        popup.bind(
            "<Escape>",
            lambda _e: popup.destroy())
        popup.protocol(
            "WM_DELETE_WINDOW",
            popup.destroy)
        self._help_popup = popup

    # ── serialization ─────────────────────────────────────────────

    def get_params(self) -> dict:
        return {
            "aspect_ratio":
                self._aspect_var.get()}

    def set_params(self,
                    params: dict) -> None:
        self._init_state()
        self._aspect_var.set(
            float(params.get(
                "aspect_ratio", 0.0)))

    def close_inspector(self) -> None:
        super().close_inspector()

    def on_destroy(self) -> None:
        if (self._help_popup is not None
                and self._help_popup.winfo_exists()):
            self._help_popup.destroy()
        super().on_destroy()