# node_editor/nodes/calib_nodes.py

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import os
import re
from pathlib import Path

import cv2
import numpy as np

from node_editor.base_node import BaseNode
from node_editor.pin_types import PinSchema, PinDef, PinType
from node_editor.execution import ExecutionMode
from node_editor.project_context import get_project_directory


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _parse_matrix_text(text: str, cols: int) -> np.ndarray | None:
    """Parse whitespace/comma/tab separated numbers into an (N, cols) array."""
    try:
        nums = [float(t) for t in text.replace(",", " ")
                                      .replace("\t", " ")
                                      .split()
                if t.strip()]
        if len(nums) == 0:
            return None
        arr = np.array(nums, dtype=np.float64).reshape(-1, cols)
        return arr
    except Exception:
        return None


def _array_to_text(arr: np.ndarray, fmt: str = "%.6g") -> str:
    rows = []
    flat = np.asarray(arr)
    if flat.ndim == 1:
        flat = flat.reshape(-1, 1)
    for row in flat:
        rows.append("  ".join(fmt % v for v in row))
    return "\n".join(rows)


def _parse_sequence_text(text: str) -> np.ndarray | None:
    """Parse numeric sequences with whitespace/comma separators or MATLAB-style colon notation."""
    if text is None:
        return None
    raw = str(text).strip()
    if not raw or raw.lower() in {"none", "null"}:
        return None

    # Treat the ghost-text placeholder as empty so it cannot be mistaken for real input.
    if raw.startswith("e.g.,"):
        return None

    tokens = [p for p in re.split(r"[\s,]+", raw) if p.strip()]
    if not tokens:
        return None

    values: list[float] = []
    for token in tokens:
        if ":" not in token:
            try:
                values.append(float(token))
            except ValueError:
                return None
            continue

        colon_parts = [part.strip() for part in token.split(":") if part.strip()]
        if not colon_parts:
            continue
        if len(colon_parts) == 1:
            try:
                values.append(float(colon_parts[0]))
            except ValueError:
                return None
            continue

        try:
            start = float(colon_parts[0])
            stop = float(colon_parts[-1])
            if len(colon_parts) == 2:
                step = 1.0 if stop >= start else -1.0
            else:
                step = float(colon_parts[1])
                if step == 0:
                    return None
        except ValueError:
            return None

        if step > 0:
            arr = np.arange(start, stop + step, step, dtype=np.float64)
        else:
            arr = np.arange(start, stop + step, step, dtype=np.float64)
        values.extend(float(v) for v in arr)

    if not values:
        return None
    return np.asarray(values, dtype=np.float64)


_DISTORTION_COEFFICIENT_NAMES = (
    "k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6",
    "s1", "s2", "s3", "s4", "tau_x", "tau_y",
)
_DISTORTION_COEFFICIENT_COUNT = len(_DISTORTION_COEFFICIENT_NAMES)


def _as_full_distortion_vector(dvec: np.ndarray) -> np.ndarray:
    """Return OpenCV's full 14-coefficient distortion vector (1x14).

    OpenCV permits shorter vectors (4, 5, 8, or 12 coefficients).  The node's
    public representation is always the full model, with unspecified terms
    explicitly set to zero: k1, k2, p1, p2, k3, k4-k6, s1-s4, tau_x, tau_y.
    """
    values = np.asarray(dvec, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("distortion vector is empty")
    if values.size > _DISTORTION_COEFFICIENT_COUNT:
        raise ValueError("distortion vector has more than 14 coefficients")
    full = np.zeros((1, _DISTORTION_COEFFICIENT_COUNT), dtype=np.float64)
    full[0, :values.size] = values
    return full


def _calibration_distortion_vector(
    full_dvec: np.ndarray, flags: int
) -> np.ndarray:
    """Adapt the canonical vector to the model enabled for calibrateCamera."""
    count = 5
    if flags & cv2.CALIB_RATIONAL_MODEL:
        count = 8
    if flags & cv2.CALIB_THIN_PRISM_MODEL:
        count = 12
    if flags & cv2.CALIB_TILTED_MODEL:
        count = 14
    return _as_full_distortion_vector(full_dvec)[:, :count].copy()


def _undistort_points_to_pixels(
    points: np.ndarray | None,
    cmat: np.ndarray,
    dvec: np.ndarray,
    new_cmat: np.ndarray | None = None,
) -> np.ndarray | None:
    """Map distorted image points to the same rectified coordinate system as cv2.undistort()."""
    if points is None:
        return None
    arr = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if arr.size == 0:
        return points
    if new_cmat is None:
        new_cmat = cmat
    # Supplying P returns pixel coordinates in new_cmat already.  Do not apply
    # fx/fy/cx/cy a second time.
    return cv2.undistortPoints(
        arr.reshape(-1, 1, 2),
        cmat,
        dvec,
        R=None,
        P=new_cmat,
    ).reshape(-1, 2)


def _undistorted_output_size(
    image_size: tuple[int, int],
    scale: float = 1.0,
) -> tuple[int, int]:
    """Return the new image size for an undistorted view based on a scale factor."""
    width, height = image_size
    scale = max(1e-6, float(scale))
    return (
        max(1, int(round(width * scale))),
        max(1, int(round(height * scale))),
    )


def _clamp_undistort_scale(scale: float | None) -> float:
    if scale is None:
        return 1.0
    value = float(scale)
    return max(1e-6, value)


def _draw_calib_result(
    bg_img: np.ndarray,
    image_points: np.ndarray | None,
    proj_points: np.ndarray | None,
    cmat: np.ndarray,
    dvec: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    grid_xs: np.ndarray | None,
    grid_ys: np.ndarray | None,
    grid_zs: np.ndarray | None,
    undistort: bool = False,
    new_cmat: np.ndarray | None = None,
    image_size_scale: float = 1.0,
    alpha: float | None = None,
) -> np.ndarray:
    """Draw image points, projected points, and 3D grid onto bg_img.

    The newer UI uses image_size_scale; the alpha parameter is retained only for
    backward compatibility with older callers/tests.
    """
    img = bg_img.copy()
    image_size_scale = _clamp_undistort_scale(image_size_scale)

    if undistort and cmat is not None and dvec is not None:
        h, w = img.shape[:2]
        if new_cmat is not None and alpha is None and image_size_scale == 1.0:
            output_size = (w, h)
        elif new_cmat is not None and alpha is not None:
            output_size = (w, h)
        else:
            output_size = _undistorted_output_size((w, h), scale=image_size_scale)
        if new_cmat is None:
            new_cmat, _ = cv2.getOptimalNewCameraMatrix(
                cmat, dvec, (w, h), alpha=0.0, newImgSize=output_size)
        map1, map2 = cv2.initUndistortRectifyMap(
            cmat, dvec, None, new_cmat, output_size, cv2.CV_16SC2)
        img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
        image_points = _undistort_points_to_pixels(image_points, cmat, dvec, new_cmat)
        proj_points = _undistort_points_to_pixels(proj_points, cmat, dvec, new_cmat)

    h, w = img.shape[:2]
    ms = max(3, h // 100)
    th = max(1, h // 400)

    # projected points — yellow squares
    if proj_points is not None:
        pts = proj_points.reshape(-1, 2)
        for pt in pts:
            x, y = int(round(pt[0])), int(round(pt[1]))
            cv2.drawMarker(img, (x, y), (0, 255, 255),
                           cv2.MARKER_SQUARE, ms, th)

    # image points — green crosses
    if image_points is not None:
        pts = image_points.reshape(-1, 2)
        for pt in pts:
            if not (np.isnan(pt[0]) or np.isnan(pt[1])):
                x, y = int(round(pt[0])), int(round(pt[1]))
                cv2.drawMarker(img, (x, y), (0, 255, 0),
                               cv2.MARKER_CROSS, ms * 2, th)

    # 3D grid
    if (grid_xs is not None and grid_ys is not None
            and grid_zs is not None
            and rvec is not None and tvec is not None
            and cmat is not None and dvec is not None):
        try:
            xs = np.asarray(grid_xs, dtype=np.float64).ravel()
            ys = np.asarray(grid_ys, dtype=np.float64).ravel()
            zs = np.asarray(grid_zs, dtype=np.float64).ravel()
            # draw grid lines along x
            for y_val in ys:
                for z_val in zs:
                    pts3 = np.array([[x, y_val, z_val]
                                     for x in xs],
                                    dtype=np.float32)
                    if len(pts3) < 2:
                        continue
                    proj, _ = cv2.projectPoints(
                        pts3, rvec, tvec, cmat, dvec)
                    proj = proj.reshape(-1, 2)
                    if undistort and cmat is not None and dvec is not None:
                        proj = _undistort_points_to_pixels(proj, cmat, dvec, new_cmat)
                    proj = proj.astype(int)
                    for k in range(len(proj) - 1):
                        cv2.line(img, tuple(proj[k]),
                                 tuple(proj[k + 1]),
                                 (255, 255, 0), th)
            # draw grid lines along y
            for x_val in xs:
                for z_val in zs:
                    pts3 = np.array([[x_val, y, z_val]
                                     for y in ys],
                                    dtype=np.float32)
                    if len(pts3) < 2:
                        continue
                    proj, _ = cv2.projectPoints(
                        pts3, rvec, tvec, cmat, dvec)
                    proj = proj.reshape(-1, 2)
                    if undistort and cmat is not None and dvec is not None:
                        proj = _undistort_points_to_pixels(proj, cmat, dvec, new_cmat)
                    proj = proj.astype(int)
                    for k in range(len(proj) - 1):
                        cv2.line(img, tuple(proj[k]),
                                 tuple(proj[k + 1]),
                                 (255, 255, 0), th)

            # draw grid lines along z
            for x_val in xs:
                for y_val in ys:
                    pts3 = np.array([[x_val, y_val, z]
                                     for z in zs],
                                    dtype=np.float32)
                    if len(pts3) < 2:
                        continue
                    proj, _ = cv2.projectPoints(
                        pts3, rvec, tvec, cmat, dvec)
                    proj = proj.reshape(-1, 2)
                    if undistort and cmat is not None and dvec is not None:
                        proj = _undistort_points_to_pixels(proj, cmat, dvec, new_cmat)
                    proj = proj.astype(int)
                    for k in range(len(proj) - 1):
                        cv2.line(img, tuple(proj[k]),
                                 tuple(proj[k + 1]),
                                 (255, 255, 0), th)
        except Exception as e:
            pass

    return img


# ---------------------------------------------------------------------------
# CameraCalibNode
# ---------------------------------------------------------------------------

class CameraCalibNode(BaseNode):
    """
    Camera calibration node wrapping cv2.calibrateCamera.

    Two modes:
      Single-image: user pastes 3D and 2D point coordinates manually.
      Chessboard:   user selects images; node runs findChessboardCorners
                    then calibrateCamera.

    All heavy computation (findChessboardCorners, calibrateCamera)
    runs in a background thread so the UI never freezes.

    Input pins (all optional — inspector UI is fully self-contained):
      images     IMAGE    — receive frames from ImageSequenceNode
      trigger    TRIGGER  — trigger calibration programmatically

    Output pins:
      cmat       ARRAY    — 3x3 camera matrix
      dvec       ARRAY    — distortion coefficients (1xN)
      rvec       ARRAY    — rotation vector of selected image (3x1)
      tvec       ARRAY    — translation vector of selected image (3x1)
      image_pts  ARRAY    — detected 2D image points
      object_pts ARRAY    — 3D object points used
      proj_pts   ARRAY    — reprojected 2D points
      proj_err   ARRAY    — reprojection errors
      rms        SCALAR   — RMS reprojection error
      vis_image  IMAGE    — visualization image (connect to VideoPlayOutputNode)
      done       TRIGGER  — fires when calibration completes
    """

    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE      = "camera_calib"
    DISPLAY_NAME   = "Camera Calibration"
    CATEGORY       = "process"
    NODE_WIDTH     = 220
    NODE_HEIGHT    = 120
    PIN_LABEL_COLOR = "#eaf4ff"
    INPUT_PIN_LABEL_COLOR = "#eaf4ff"
    OUTPUT_PIN_LABEL_COLOR = "#eaf4ff"

    # cv2 calibration flag definitions
    _CALIB_FLAGS = [
        ("CALIB_USE_INTRINSIC_GUESS",  cv2.CALIB_USE_INTRINSIC_GUESS),
        ("CALIB_FIX_ASPECT_RATIO",     cv2.CALIB_FIX_ASPECT_RATIO),
        ("CALIB_FIX_PRINCIPAL_POINT",  cv2.CALIB_FIX_PRINCIPAL_POINT),
        ("CALIB_ZERO_TANGENT_DIST",    cv2.CALIB_ZERO_TANGENT_DIST),
        ("CALIB_FIX_FOCAL_LENGTH",     cv2.CALIB_FIX_FOCAL_LENGTH),
        ("CALIB_FIX_K1",               cv2.CALIB_FIX_K1),
        ("CALIB_FIX_K2",               cv2.CALIB_FIX_K2),
        ("CALIB_FIX_K3",               cv2.CALIB_FIX_K3),
        ("CALIB_FIX_K4",               cv2.CALIB_FIX_K4),
        ("CALIB_FIX_K5",               cv2.CALIB_FIX_K5),
        ("CALIB_FIX_K6",               cv2.CALIB_FIX_K6),
        ("CALIB_RATIONAL_MODEL",       cv2.CALIB_RATIONAL_MODEL),
        ("CALIB_THIN_PRISM_MODEL",     cv2.CALIB_THIN_PRISM_MODEL),
        ("CALIB_TILTED_MODEL",         cv2.CALIB_TILTED_MODEL),
        ("CALIB_FIX_TANGENT_DIST",     cv2.CALIB_FIX_TANGENT_DIST),
        ("CALIB_USE_LU",               cv2.CALIB_USE_LU),
        ("CALIB_USE_QR",               cv2.CALIB_USE_QR),
    ]

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[
                PinDef("images",  PinType.IMAGE,   "images",
                       optional=True),
                PinDef("trigger", PinType.TRIGGER, "trigger",
                       optional=True),
            ],
            outputs=[
                PinDef("cmat",       PinType.ARRAY,  "cmat"),
                PinDef("dvec",       PinType.ARRAY,  "dvec"),
                PinDef("rvec",       PinType.ARRAY,  "rvec"),
                PinDef("tvec",       PinType.ARRAY,  "tvec"),
                PinDef("imgsize",    PinType.ARRAY,  "imgSize",
                       shape=(2,), dtype="int64"),     # [width, height]
                PinDef("image_pts",  PinType.ARRAY,  "imgPts"),
                PinDef("object_pts", PinType.ARRAY,  "objPts"),
                PinDef("proj_pts",   PinType.ARRAY,  "prjPts"),
                PinDef("proj_err",   PinType.ARRAY,  "prjErr"),
                PinDef("rms",        PinType.SCALAR, "RMS"),
                PinDef("vis_image",  PinType.IMAGE,  "visImg"),
                PinDef("done",       PinType.TRIGGER, "done"),
            ]
        )

    # ── init state ────────────────────────────────────────────────

    def _init_state(self) -> None:
        if hasattr(self, "_mode_var"):
            return

        # mode
        self._mode_var = tk.StringVar(value="chessboard")

        # image file list (chessboard mode)
        self._cb_files: list[str] = []
        self._collected_images: list[np.ndarray] = []

        # chessboard params
        self._cb_nx_var    = tk.IntVar(value=7)
        self._cb_ny_var    = tk.IntVar(value=7)
        self._cb_dx_var    = tk.DoubleVar(value=25.4)
        self._cb_dy_var    = tk.DoubleVar(value=25.4)

        # subpixel refinement
        self._subpix_var      = tk.BooleanVar(value=True)
        self._subpix_win_var  = tk.IntVar(value=11)
        self._subpix_iter_var = tk.IntVar(value=30)
        self._subpix_eps_var  = tk.DoubleVar(value=0.001)

        # image size
        self._img_w_var = tk.IntVar(value=1920)
        self._img_h_var = tk.IntVar(value=1080)

        # initial guess
        self._cmat_guess_text = (
            "5000  0     959.5\n"
            "0     5000  539.5\n"
            "0     0     1")
        self._dvec_guess_text = " ".join(
            "0" for _ in range(_DISTORTION_COEFFICIENT_COUNT))

        # single-image mode text
        self._pts3d_text = "0 0 0\n1 0 0\n1 1 0\n0 1 0"
        self._pts2d_text = "100 100\n200 100\n200 200\n100 200"

        # calibration flags (IntVar per flag)
        self._flag_vars: list[tk.IntVar] = [
            tk.IntVar(value=0) for _ in self._CALIB_FLAGS]
        # set sensible defaults
        self._flag_vars[0].set(1)   # USE_INTRINSIC_GUESS
        self._flag_vars[2].set(1)   # FIX_PRINCIPAL_POINT
        self._flag_vars[3].set(1)   # ZERO_TANGENT_DIST
        self._flag_vars[6].set(1)   # FIX_K2
        self._flag_vars[7].set(1)   # FIX_K3
        self._flag_vars[8].set(1)   # FIX_K4
        self._flag_vars[9].set(1)   # FIX_K5
        self._flag_vars[10].set(1)  # FIX_K6

        # grid visualization params
        self._grid_xs_text = ""
        self._grid_ys_text = ""
        self._grid_zs_text = "0"
        self._grid_placeholder = "e.g., -30 -20 -10 0 10 20 30 or -30:10:30"
        self._undistort_var = tk.BooleanVar(value=False)
        self._undistort_scale_var = tk.DoubleVar(value=1.0)

        # which image to visualize / output rvec+tvec (1-based)
        self._vis_idx_var = tk.IntVar(value=1)

        # calibration results
        self._result_cmat:      np.ndarray | None = None
        self._result_dvec:      np.ndarray | None = None
        self._result_rvecs:     list | None = None
        self._result_tvecs:     list | None = None
        self._result_imgpts:    list | None = None
        self._result_objpts:    list | None = None
        self._result_rms:       float | None = None
        self._result_proj_pts:  list | None = None   # per image
        self._result_proj_errs: list | None = None   # per image

        # fire counter for done trigger
        self._done_counter = 0
        self._last_trigger = None
        self._trigger_latched_high = False

        # status
        self._status_var = tk.StringVar(value="not calibrated")

        # background thread guard
        self._calib_running = False

        # inspector widget references
        self._nb:                 ttk.Notebook | None = None
        self._file_listbox:       tk.Listbox   | None = None
        self._pts3d_text_widget:  tk.Text | None = None
        self._pts2d_text_widget:  tk.Text | None = None
        self._cmat_guess_widget:  tk.Text | None = None
        self._dvec_guess_widget:  tk.Text | None = None
        self._result_text:        tk.Text | None = None
        self._result_rvec_widget: tk.Entry | None = None
        self._result_tvec_widget: tk.Entry | None = None
        self._result_cmat_widget: tk.Text | None = None
        self._result_dvec_widget: tk.Text | None = None
        self._result_update_btn:  tk.Button | None = None
        self._flag_sum_var        = tk.StringVar(value="flags: 0")
        self._vis_idx_sb:         tk.Spinbox | None = None
        # Only ever created inside build_inspector(); must exist here too so a
        # trigger-pin-driven calibration works even if the inspector was never
        # opened (_run_calibration_bg reads this before checking winfo_exists()).
        self._calib_btn:          tk.Button | None = None

    # ── build_body ────────────────────────────────────────────────

    def build_body(self) -> None:
        self._init_state()
        x, y, w, h = self.x, self.y, self.width, self.height

        self._body_rect = self.canvas.create_rectangle(
            x, y, x+w, y+h,
            fill="#1e1e2e", outline="#8855cc", width=2,
            tags=(self.node_id, "node_body"))
        self._title_item = self.canvas.create_text(
            x+w/2, y+13,
            text=self.DISPLAY_NAME,
            font=("Arial", 9, "bold"), fill="#cc99ff",
            tags=(self.node_id,))

        mode_lbl = tk.Label(
            self.canvas, textvariable=self._mode_var,
            font=("Arial", 8), bg="#1e1e2e", fg="#aa88dd")
        self.canvas.create_window(
            x+w/2, y+35, window=mode_lbl,
            tags=(self.node_id,))

        status_lbl = tk.Label(
            self.canvas, textvariable=self._status_var,
            font=("Arial", 7), bg="#1e1e2e", fg="#aaaaaa",
            wraplength=w-10, justify="center")
        self.canvas.create_window(
            x+w/2, y+h-14, window=status_lbl,
            tags=(self.node_id,))

        self._canvas_items += [self._body_rect, self._title_item]

    # ── build_inspector ───────────────────────────────────────────

    def open_inspector(self) -> None:
        super().open_inspector()
        if self._inspector_win is None or not self._inspector_win.winfo_exists():
            return

        self._inspector_win.update_idletasks()
        req_w = max(640, self._inspector_win.winfo_reqwidth())
        req_h = max(480, self._inspector_win.winfo_reqheight())
        self._inspector_win.geometry(f"{req_w * 2}x{req_h * 2}")
        self._inspector_win.update_idletasks()

    def build_inspector(self, parent: tk.Frame) -> None:
        self._init_state()

        self._nb = ttk.Notebook(parent)
        self._nb.pack(fill="both", expand=True)

        tab_input = tk.Frame(self._nb)
        self._nb.add(tab_input, text="Input & Calibrate")

        self._build_tab_input(tab_input)

    def _build_tab_input(self, parent: tk.Frame) -> None:
        # scrollable container
        canvas = tk.Canvas(parent, highlightthickness=0)
        vsb = ttk.Scrollbar(parent, orient="vertical",
                             command=canvas.yview)
        hsb = ttk.Scrollbar(parent, orient="horizontal",
                            command=canvas.xview)
        canvas.configure(yscrollcommand=vsb.set,
                         xscrollcommand=hsb.set)
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas)
        win_id = canvas.create_window(
            (0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(
                       scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfigure(
                        win_id, width=max(e.width, inner.winfo_reqwidth())))

        self._build_input_content(inner)

    def _build_input_content(self, parent: tk.Frame) -> None:
        pad = {"padx": 6, "pady": 3}

        layout = tk.Frame(parent)
        layout.pack(fill="both", expand=True)

        left_w = 360
        middle_w = 360
        right_w = 360

        layout.grid_columnconfigure(0, minsize=left_w, weight=0)
        layout.grid_columnconfigure(2, minsize=middle_w, weight=0)
        layout.grid_columnconfigure(4, minsize=right_w, weight=0)

        left = tk.Frame(layout, width=left_w, bg="#f8f8ff")
        middle = tk.Frame(layout, width=middle_w, bg="#f3f7ff")
        right = tk.Frame(layout, width=right_w, bg="#f9f9f9")

        sep1 = tk.Frame(layout, width=8, cursor="sb_h_double_arrow", bg="#c8c8d0")
        sep2 = tk.Frame(layout, width=8, cursor="sb_h_double_arrow", bg="#c8c8d0")

        def _bind_drag_separator(sep, left_panel, right_panel, axis):
            drag_state = {"active": False, "start_x": 0, "left_start": 0, "right_start": 0}

            def _on_press(event):
                drag_state["active"] = True
                drag_state["start_x"] = event.x_root
                drag_state["left_start"] = left_panel.winfo_width()
                drag_state["right_start"] = right_panel.winfo_width()

            def _on_release(event):
                drag_state["active"] = False

            def _on_drag(event):
                if not drag_state["active"]:
                    return
                delta = event.x_root - drag_state["start_x"]
                if axis == "left_middle":
                    new_left = max(220, drag_state["left_start"] + delta)
                    new_mid = max(220, drag_state["right_start"] - delta)
                    left_panel.configure(width=new_left)
                    right_panel.configure(width=new_mid)
                    layout.grid_columnconfigure(0, minsize=new_left)
                    layout.grid_columnconfigure(2, minsize=new_mid)
                else:
                    new_mid = max(220, drag_state["left_start"] + delta)
                    new_right = max(220, drag_state["right_start"] - delta)
                    middle_panel = left_panel
                    right_panel.configure(width=new_right)
                    middle_panel.configure(width=new_mid)
                    layout.grid_columnconfigure(2, minsize=new_mid)
                    layout.grid_columnconfigure(4, minsize=new_right)
                    layout.update_idletasks()

            sep.bind("<ButtonPress-1>", _on_press)
            sep.bind("<B1-Motion>", _on_drag)
            sep.bind("<ButtonRelease-1>", _on_release)

        _bind_drag_separator(sep1, left, middle, "left_middle")
        _bind_drag_separator(sep2, middle, right, "middle_right")

        left.grid(row=0, column=0, sticky="nsew")
        sep1.grid(row=0, column=1, sticky="ns")
        middle.grid(row=0, column=2, sticky="nsew")
        sep2.grid(row=0, column=3, sticky="ns")
        right.grid(row=0, column=4, sticky="nsew")

        # ── image size ────────────────────────────────────────────
        sz_frame = tk.LabelFrame(
            left, text="Image size (width x height)",
            font=("Arial", 9), **pad)
        sz_frame.pack(fill="x", **pad)

        sz_row = tk.Frame(sz_frame)
        sz_row.pack(fill="x")
        tk.Label(sz_row, text="W:", font=("Arial", 9)).pack(
            side="left")
        tk.Spinbox(sz_row, from_=1, to=100000,
                   textvariable=self._img_w_var,
                   width=7, font=("Arial", 9)).pack(side="left")
        tk.Label(sz_row, text="  H:", font=("Arial", 9)).pack(
            side="left")
        tk.Spinbox(sz_row, from_=1, to=100000,
                   textvariable=self._img_h_var,
                   width=7, font=("Arial", 9)).pack(side="left")

        # ── mode selector ─────────────────────────────────────────
        mode_frame = tk.LabelFrame(
            left, text="Calibration mode",
            font=("Arial", 9), **pad)
        mode_frame.pack(fill="x", **pad)

        tk.Radiobutton(
            mode_frame, text="Chessboard (multiple images)",
            variable=self._mode_var, value="chessboard",
            font=("Arial", 9),
            command=self._on_mode_change).pack(anchor="w")
        tk.Radiobutton(
            mode_frame, text="Single-image (manual points)",
            variable=self._mode_var, value="single",
            font=("Arial", 9),
            command=self._on_mode_change).pack(anchor="w")

        # ── chessboard section ────────────────────────────────────
        self._cb_section = tk.LabelFrame(
            left, text="Chessboard settings",
            font=("Arial", 9), **pad)
        self._cb_section.pack(fill="x", **pad)

        self._build_chessboard_section(self._cb_section)

        # ── single-image section ──────────────────────────────────
        self._single_section = tk.LabelFrame(
            left, text="Manual point coordinates",
            font=("Arial", 9), **pad)
        self._single_section.pack(fill="x", **pad)

        self._build_single_section(self._single_section)

        # ── initial guess ─────────────────────────────────────────
        guess_frame = tk.LabelFrame(
            left, text="Initial guess",
            font=("Arial", 9), **pad)
        guess_frame.pack(fill="x", **pad)

        tk.Label(guess_frame,
                 text="Camera matrix (3x3):",
                 font=("Arial", 8)).pack(anchor="w")
        self._cmat_guess_widget = tk.Text(
            guess_frame, width=40, height=3,
            font=("Courier", 8))
        self._cmat_guess_widget.pack(fill="x")
        self._cmat_guess_widget.insert(
            "1.0", self._cmat_guess_text)

        tk.Label(guess_frame,
                 text=("Distortion coeff. (k1 k2 p1 p2)(k3 "
                       "k4 k5 k6 s1 s2 s3 s4 taux tauy):"),
                 font=("Arial", 8)).pack(anchor="w")
        self._dvec_guess_widget = tk.Text(
            guess_frame, width=40, height=2,
            font=("Courier", 8))
        self._dvec_guess_widget.pack(fill="x")
        self._dvec_guess_widget.insert(
            "1.0", self._dvec_guess_text)

        # ── subpixel refinement ───────────────────────────────────
        subpix_frame = tk.LabelFrame(
            left, text="Subpixel refinement (cornerSubPix)",
            font=("Arial", 9), **pad)
        subpix_frame.pack(fill="x", **pad)

        self._subpix_checkbox = tk.Checkbutton(
            subpix_frame,
            text="Enable cv2.cornerSubPix",
            variable=self._subpix_var,
            font=("Arial", 9),
            command=self._on_subpix_toggle)
        self._subpix_checkbox.pack(anchor="w")

        self._subpix_detail = tk.Frame(subpix_frame)
        self._subpix_detail.pack(fill="x")

        def _sp_row(label, var, width=5):
            r = tk.Frame(self._subpix_detail)
            r.pack(fill="x", pady=1)
            tk.Label(r, text=label, font=("Arial", 8),
                     width=22, anchor="w").pack(side="left")
            tk.Spinbox(r, from_=0, to=999999,
                       textvariable=var,
                       width=width,
                       font=("Arial", 8),
                       increment=1).pack(side="left")

        _sp_row("Window size (half):", self._subpix_win_var)
        _sp_row("Max iterations:",     self._subpix_iter_var)

        eps_row = tk.Frame(self._subpix_detail)
        eps_row.pack(fill="x", pady=1)
        tk.Label(eps_row, text="Epsilon:", font=("Arial", 8),
                 width=22, anchor="w").pack(side="left")
        tk.Entry(eps_row,
                 textvariable=self._subpix_eps_var,
                 width=8, font=("Arial", 8)).pack(side="left")

        # ── flags summary ─────────────────────────────────────────
        tk.Label(left, textvariable=self._flag_sum_var,
                 font=("Arial", 8), fg="#446688",
                 anchor="w").pack(fill="x", **pad)

        # ── calibrate button ──────────────────────────────────────
        calib_row = tk.Frame(left)
        calib_row.pack(fill="x", **pad)

        self._calib_btn = tk.Button(
            calib_row,
            text="Calibrate",
            font=("Arial", 10, "bold"),
            bg="#443366", fg="white",
            activebackground="#554477",
            relief=tk.FLAT, padx=12, pady=4,
            command=self._on_calibrate)
        self._calib_btn.pack(side="left")

        tk.Label(calib_row,
                 textvariable=self._status_var,
                 font=("Arial", 9), fg="#665588",
                 anchor="w", justify="left").pack(
            side="left", padx=8)

        # ── flags panel ───────────────────────────────────────────
        flags_frame = tk.LabelFrame(
            middle, text="Flags",
            font=("Arial", 9), **pad)
        flags_frame.pack(fill="both", expand=True, **pad)
        self._build_tab_flags(flags_frame)

        # ── results summary ───────────────────────────────────────
        res_frame = tk.LabelFrame(
            right, text="Calibration results",
            font=("Arial", 9), **pad)
        res_frame.pack(fill="both", expand=True, **pad)

        self._result_rvec_widget = tk.Entry(
            res_frame, width=32, font=("Courier", 9))
        self._result_rvec_widget.pack(fill="x", pady=(4, 2))
        tk.Label(res_frame, text="rvec (3):", font=("Arial", 8)).pack(anchor="w")

        self._result_tvec_widget = tk.Entry(
            res_frame, width=32, font=("Courier", 9))
        self._result_tvec_widget.pack(fill="x", pady=(4, 2))
        tk.Label(res_frame, text="tvec (3):", font=("Arial", 8)).pack(anchor="w")

        self._result_cmat_widget = tk.Text(
            res_frame, width=46, height=5, font=("Courier", 8), wrap="none")
        self._result_cmat_widget.pack(fill="both", expand=True, pady=(4, 2))
        tk.Label(res_frame, text="camera matrix (3x3):", font=("Arial", 8)).pack(anchor="w")

        self._result_dvec_widget = tk.Text(
            res_frame, width=46, height=6, font=("Courier", 8), wrap="none")
        self._result_dvec_widget.pack(fill="both", expand=True, pady=(4, 2))
        tk.Label(res_frame, text="distortion coeffs (4..14):", font=("Arial", 8)).pack(anchor="w")

        btn_row = tk.Frame(res_frame)
        btn_row.pack(fill="x", pady=(6, 0))
        self._result_update_btn = tk.Button(
            btn_row,
            text="Update",
            font=("Arial", 9, "bold"),
            command=self._on_update_calibration_results)
        self._result_update_btn.pack(side="left")

        # ── visualization ─────────────────────────────────────────
        vis_frame = tk.LabelFrame(
            right, text="Visualization",
            font=("Arial", 9), **pad)
        vis_frame.pack(fill="x", **pad)

        vis_idx_row = tk.Frame(vis_frame)
        vis_idx_row.pack(fill="x", pady=2)
        tk.Label(vis_idx_row,
                 text="Image index (1-based):",
                 font=("Arial", 9)).pack(side="left")
        self._vis_idx_sb = tk.Spinbox(
            vis_idx_row,
            from_=1, to=9999,
            textvariable=self._vis_idx_var,
            width=5, font=("Arial", 9),
            command=self._on_vis_update)
        self._vis_idx_sb.pack(side="left", padx=4)

        tk.Checkbutton(
            vis_frame,
            text="Undistort image before drawing",
            variable=self._undistort_var,
            font=("Arial", 9),
            command=self._on_vis_update).pack(anchor="w")

        scale_row = tk.Frame(vis_frame)
        scale_row.pack(fill="x", pady=2)
        tk.Label(scale_row, text="Image size scale:", font=("Arial", 9)).pack(side="left")
        tk.Entry(
            scale_row,
            textvariable=self._undistort_scale_var,
            width=8,
            font=("Arial", 9)).pack(side="left", padx=4)
        tk.Label(scale_row,
                 text="(1.0 = original size, 2.0 = 2x larger)",
                 font=("Arial", 8), fg="#666666").pack(side="left")

        grid_frame = tk.Frame(vis_frame)
        grid_frame.pack(fill="x")

        def _grid_row(label, attr):
            r = tk.Frame(grid_frame)
            r.pack(fill="x", pady=1)
            tk.Label(r, text=label, font=("Arial", 8),
                     width=8, anchor="w").pack(side="left")
            var = tk.StringVar(value=getattr(self, attr))
            ent = tk.Entry(r, textvariable=var,
                           font=("Courier", 8), width=36,
                           fg="#0f0f0f")
            ent.pack(side="left", fill="x", expand=True)

            def _show_placeholder(event=None, entry=ent, value=var):
                if not value.get().strip():
                    entry.configure(fg="#7a7a7a")
                    entry.delete(0, tk.END)
                    entry.insert(0, self._grid_placeholder)

            def _clear_placeholder(event=None, entry=ent, value=var):
                current = value.get().strip()
                if current == self._grid_placeholder:
                    entry.delete(0, tk.END)
                    value.set("")
                    entry.configure(fg="#1d1d1d")

            if not getattr(self, attr, "").strip():
                ent.insert(0, self._grid_placeholder)
            ent.bind("<FocusIn>", _clear_placeholder)
            ent.bind("<FocusOut>", _show_placeholder)
            return var, ent

        self._grid_xs_var, _ = _grid_row(
            "grid xs:", "_grid_xs_text")
        self._grid_ys_var, _ = _grid_row(
            "grid ys:", "_grid_ys_text")
        self._grid_zs_var, _ = _grid_row(
            "grid zs:", "_grid_zs_text")

        tk.Button(
            vis_frame,
            text="Update visualization",
            font=("Arial", 9),
            command=self._on_vis_update).pack(
            anchor="w", pady=4)

        self._on_mode_change()
        self._on_subpix_toggle()
        self._update_flag_sum()
        if self._result_cmat is not None:
            self._refresh_result_text()

    def _build_chessboard_section(self,
                                   parent: tk.Frame) -> None:
        # corners and spacing
        params_row = tk.Frame(parent)
        params_row.pack(fill="x", pady=2)

        for label, var in [
            ("nx:", self._cb_nx_var),
            ("ny:", self._cb_ny_var),
            ("dx(mm):", self._cb_dx_var),
            ("dy(mm):", self._cb_dy_var),
        ]:
            tk.Label(params_row, text=label,
                     font=("Arial", 8)).pack(side="left")
            tk.Entry(params_row,
                     textvariable=var,
                     width=6,
                     font=("Arial", 8)).pack(
                side="left", padx=2)

        # image file list
        tk.Label(parent,
                 text="Calibration images:",
                 font=("Arial", 8)).pack(anchor="w")

        list_frame = tk.Frame(parent)
        list_frame.pack(fill="x")

        self._file_listbox = tk.Listbox(
            list_frame, height=5,
            font=("Courier", 8),
            selectmode=tk.EXTENDED)
        lb_vsb = ttk.Scrollbar(
            list_frame, orient="vertical",
            command=self._file_listbox.yview)
        lb_hsb = ttk.Scrollbar(
            list_frame, orient="horizontal",
            command=self._file_listbox.xview)
        self._file_listbox.configure(
            yscrollcommand=lb_vsb.set,
            xscrollcommand=lb_hsb.set)
        lb_hsb.pack(side="bottom", fill="x")
        lb_vsb.pack(side="right", fill="y")
        self._file_listbox.pack(
            fill="both", expand=True)

        # repopulate from saved list
        for f in self._cb_files:
            self._file_listbox.insert(tk.END,
                                      Path(f).name)

        btn_row = tk.Frame(parent)
        btn_row.pack(fill="x", pady=2)

        tk.Button(
            btn_row, text="Browse...",
            font=("Arial", 8),
            command=self._on_browse_cb).pack(
            side="left", padx=2)
        tk.Button(
            btn_row, text="Clear list",
            font=("Arial", 8),
            command=self._on_clear_cb_files).pack(
            side="left", padx=2)
        tk.Button(
            btn_row, text="Find corners",
            font=("Arial", 8),
            command=self._on_find_corners).pack(
            side="left", padx=2)

        # corner detection status
        self._corner_status_var = tk.StringVar(
            value="")
        tk.Label(
            parent,
            textvariable=self._corner_status_var,
            font=("Arial", 8), fg="#445588",
            anchor="w", justify="left",
            wraplength=380).pack(
            fill="x", pady=2)

    def _build_single_section(self,
                               parent: tk.Frame) -> None:
        tk.Label(parent,
                 text="3D coordinates (one point per line,"
                      " X Y Z):",
                 font=("Arial", 8)).pack(anchor="w")
        self._pts3d_text_widget = tk.Text(
            parent, width=40, height=5,
            font=("Courier", 8))
        self._pts3d_text_widget.pack(fill="x")
        self._pts3d_text_widget.insert(
            "1.0", self._pts3d_text)

        tk.Label(parent,
                 text="2D image coordinates (one point"
                      " per line, x y):",
                 font=("Arial", 8)).pack(anchor="w")
        self._pts2d_text_widget = tk.Text(
            parent, width=40, height=5,
            font=("Courier", 8))
        self._pts2d_text_widget.pack(fill="x")
        self._pts2d_text_widget.insert(
            "1.0", self._pts2d_text)

        tk.Button(
            parent,
            text="Validate points",
            font=("Arial", 8),
            command=self._on_validate_single).pack(
            anchor="w", pady=4)

        self._validate_var = tk.StringVar(value="")
        tk.Label(parent,
                 textvariable=self._validate_var,
                 font=("Arial", 8), fg="#446688",
                 anchor="w", justify="left").pack(
            fill="x")

    def _build_tab_flags(self,
                          parent: tk.Frame) -> None:
        canvas = tk.Canvas(parent, highlightthickness=0)
        vsb = ttk.Scrollbar(parent, orient="vertical",
                             command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas)
        win_id = canvas.create_window(
            (0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(
                       scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfigure(
                        win_id, width=e.width))

        tk.Label(
            inner,
            text="Calibration flags (cv2.calibrateCamera):",
            font=("Arial", 9, "bold")).pack(
            anchor="w", padx=6, pady=(6, 2))

        tk.Label(
            inner, textvariable=self._flag_sum_var,
            font=("Arial", 9), fg="#334466").pack(
            anchor="w", padx=6)

        for i, (name, val) in enumerate(
                self._CALIB_FLAGS):
            tk.Checkbutton(
                inner,
                text=f"{name}  ({val})",
                variable=self._flag_vars[i],
                font=("Arial", 9),
                command=self._update_flag_sum,
                anchor="w").pack(
                fill="x", padx=6, pady=1)

    # ── inspector helpers ─────────────────────────────────────────

    def _on_mode_change(self) -> None:
        mode = self._mode_var.get()
        if not hasattr(self, "_cb_section"):
            return
        if mode == "chessboard":
            self._cb_section.pack(fill="x",
                                  padx=6, pady=3)
            self._single_section.pack_forget()
        else:
            self._single_section.pack(fill="x",
                                      padx=6, pady=3)
            self._cb_section.pack_forget()
        self._on_subpix_toggle()

    def _on_subpix_toggle(self) -> None:
        if not hasattr(self, "_subpix_detail"):
            return
        chessboard_mode = self._mode_var.get() == "chessboard"
        if self._subpix_checkbox is not None and \
                self._subpix_checkbox.winfo_exists():
            self._subpix_checkbox.configure(
                state="normal" if chessboard_mode else "disabled")
        state = "normal" if chessboard_mode and self._subpix_var.get() \
            else "disabled"
        for child in self._subpix_detail.winfo_children():
            for w in child.winfo_children():
                try:
                    w.configure(state=state)
                except Exception:
                    pass

    def _update_flag_sum(self) -> None:
        total = 0
        for i, (_, val) in enumerate(self._CALIB_FLAGS):
            if self._flag_vars[i].get():
                total |= val
        self._flag_sum_var.set(
            f"flags value: {total}  (0x{total:04X})")

    def _get_flags(self) -> int:
        total = 0
        for i, (_, val) in enumerate(self._CALIB_FLAGS):
            if self._flag_vars[i].get():
                total |= val
        return total

    def _get_cmat_guess(self) -> np.ndarray | None:
        if self._cmat_guess_widget is None:
            text = self._cmat_guess_text
        else:
            text = self._cmat_guess_widget.get(
                "1.0", tk.END)
        return _parse_matrix_text(text, 3)

    def _get_dvec_guess(self) -> np.ndarray | None:
        if self._dvec_guess_widget is None:
            text = self._dvec_guess_text
        else:
            text = self._dvec_guess_widget.get(
                "1.0", tk.END)
        arr = _parse_matrix_text(text, 1)
        return None if arr is None else _as_full_distortion_vector(arr)

    def _result_widget_is_valid(self, widget: tk.Entry | tk.Text | None) -> bool:
        if widget is None:
            return False
        color = widget.cget("fg")
        return color != "red" and color != "#b31d1d"

    def _set_result_widget_validity(self, widget: tk.Entry | tk.Text | None, valid: bool) -> None:
        if widget is None:
            return
        widget.configure(fg="#1d1d1d" if valid else "red")
        if hasattr(widget, "configure"):
            widget.configure(bg="#ffffff" if valid else "#fff0f0")

    def _format_result_vector_text(self, arr: np.ndarray | None, count: int | None = None) -> str:
        if arr is None:
            return ""
        v = np.asarray(arr, dtype=np.float64).reshape(-1)
        if count is not None and v.size != count:
            v = v[:count]
            if v.size < count:
                v = np.pad(v, (0, count - v.size), constant_values=0.0)
        return "  ".join(f"{float(x):.15g}" for x in v)

    def _format_result_matrix_text(self, arr: np.ndarray | None) -> str:
        if arr is None:
            return ""
        m = np.asarray(arr, dtype=np.float64)
        if m.shape == (9,):
            m = m.reshape(3, 3)
        elif m.size == 9:
            m = m.reshape(3, 3)
        return _array_to_text(m, fmt="%.15g")

    def _format_result_dvec_text(self, arr: np.ndarray | None) -> str:
        if arr is None:
            return ""
        values = np.asarray(arr, dtype=np.float64).reshape(-1)
        if values.size < 4:
            values = np.pad(values, (0, 4 - values.size), constant_values=0.0)
        if values.size > 14:
            values = values[:14]
        values = np.pad(values, (0, 14 - values.size), constant_values=0.0)
        return "  ".join(f"{float(v):.15g}" for v in values)

    def _parse_result_rvec_text(self, text: str) -> np.ndarray | None:
        arr = _parse_matrix_text(text, 1)
        if arr is None or arr.size != 3:
            return None
        return arr.reshape(-1).astype(np.float64)

    def _parse_result_tvec_text(self, text: str) -> np.ndarray | None:
        arr = _parse_matrix_text(text, 1)
        if arr is None or arr.size != 3:
            return None
        return arr.reshape(-1).astype(np.float64)

    def _parse_result_cmat_text(self, text: str) -> np.ndarray | None:
        arr = _parse_matrix_text(text, 3)
        if arr is None or arr.size != 9:
            return None
        return arr.astype(np.float64)

    def _parse_result_dvec_text(self, text: str) -> np.ndarray | None:
        nums = [float(t) for t in text.replace(",", " ").replace("\t", " ").split() if t.strip()]
        if not nums or len(nums) < 4 or len(nums) > 14:
            return None
        vals = np.asarray(nums, dtype=np.float64).reshape(-1)
        vals = np.pad(vals, (0, 14 - vals.size), constant_values=0.0)
        return vals[:14]

    def _populate_result_edit_widgets(self) -> None:
        if self._result_rvec_widget is None or self._result_tvec_widget is None:
            return
        vis_idx = max(0, min(self._vis_idx_var.get() - 1, len(self._result_rvecs or []) - 1)) if self._result_rvecs else 0
        if self._result_rvecs and vis_idx < len(self._result_rvecs):
            self._result_rvec_widget.delete(0, tk.END)
            self._result_rvec_widget.insert(0, self._format_result_vector_text(self._result_rvecs[vis_idx], 3))
            self._set_result_widget_validity(self._result_rvec_widget, True)
        else:
            self._result_rvec_widget.delete(0, tk.END)
            self._result_rvec_widget.insert(0, "")
            self._set_result_widget_validity(self._result_rvec_widget, False)

        if self._result_tvecs and vis_idx < len(self._result_tvecs):
            self._result_tvec_widget.delete(0, tk.END)
            self._result_tvec_widget.insert(0, self._format_result_vector_text(self._result_tvecs[vis_idx], 3))
            self._set_result_widget_validity(self._result_tvec_widget, True)
        else:
            self._result_tvec_widget.delete(0, tk.END)
            self._result_tvec_widget.insert(0, "")
            self._set_result_widget_validity(self._result_tvec_widget, False)

        if self._result_cmat is not None:
            if self._result_cmat_widget is not None:
                self._result_cmat_widget.delete("1.0", tk.END)
                self._result_cmat_widget.insert("1.0", self._format_result_matrix_text(self._result_cmat))
                self._set_result_widget_validity(self._result_cmat_widget, True)
        elif self._result_cmat_widget is not None:
            self._result_cmat_widget.delete("1.0", tk.END)
            self._result_cmat_widget.insert("1.0", "")
            self._set_result_widget_validity(self._result_cmat_widget, False)

        if self._result_dvec is not None:
            if self._result_dvec_widget is not None:
                self._result_dvec_widget.delete("1.0", tk.END)
                self._result_dvec_widget.insert("1.0", self._format_result_dvec_text(self._result_dvec))
                self._set_result_widget_validity(self._result_dvec_widget, True)
        elif self._result_dvec_widget is not None:
            self._result_dvec_widget.delete("1.0", tk.END)
            self._result_dvec_widget.insert("1.0", "")
            self._set_result_widget_validity(self._result_dvec_widget, False)

    def _on_update_calibration_results(self) -> None:
        if self._result_rvec_widget is None or self._result_tvec_widget is None:
            return
        if self._result_cmat_widget is None or self._result_dvec_widget is None:
            return
        if not self._result_rvecs or not self._result_tvecs:
            return
        vis_idx = max(0, min(self._vis_idx_var.get() - 1, len(self._result_rvecs) - 1))

        rvec_txt = self._result_rvec_widget.get()
        tvec_txt = self._result_tvec_widget.get()
        cmat_txt = self._result_cmat_widget.get("1.0", tk.END)
        dvec_txt = self._result_dvec_widget.get("1.0", tk.END)

        new_rvec = self._parse_result_rvec_text(rvec_txt)
        new_tvec = self._parse_result_tvec_text(tvec_txt)
        new_cmat = self._parse_result_cmat_text(cmat_txt)
        new_dvec = self._parse_result_dvec_text(dvec_txt)

        valid = all(v is not None for v in (new_rvec, new_tvec, new_cmat, new_dvec))
        self._set_result_widget_validity(self._result_rvec_widget, new_rvec is not None)
        self._set_result_widget_validity(self._result_tvec_widget, new_tvec is not None)
        self._set_result_widget_validity(self._result_cmat_widget, new_cmat is not None)
        self._set_result_widget_validity(self._result_dvec_widget, new_dvec is not None)

        if not valid:
            return

        self._result_rvecs[vis_idx] = new_rvec.reshape(3, 1)
        self._result_tvecs[vis_idx] = new_tvec.reshape(3, 1)
        self._result_cmat = new_cmat.reshape(3, 3)
        self._result_dvec = _as_full_distortion_vector(new_dvec.reshape(1, -1))

        self._result_proj_pts = []
        self._result_proj_errs = []
        for i in range(len(self._result_rvecs)):
            pp, _ = cv2.projectPoints(
                self._result_objpts[i].reshape(-1, 3),
                self._result_rvecs[i], self._result_tvecs[i], self._result_cmat, self._result_dvec)
            pp = pp.reshape(-1, 2)
            img_pts = self._result_imgpts[i].reshape(-1, 2)
            err = pp - img_pts
            self._result_proj_pts.append(pp)
            self._result_proj_errs.append(err)

        self._on_vis_update()
        self._push_outputs()

    def _on_browse_cb(self) -> None:
        base = get_project_directory()
        initial = str(base) if base else "."
        paths = filedialog.askopenfilenames(
            title="Select chessboard images",
            initialdir=initial,
            filetypes=[
                ("Image files",
                 "*.jpg *.jpeg *.png *.bmp "
                 "*.tif *.tiff"),
                ("All files", "*.*"),
            ])
        if not paths:
            return
        for p in paths:
            if p not in self._cb_files:
                self._cb_files.append(p)
                if self._file_listbox is not None:
                    self._file_listbox.insert(
                        tk.END, Path(p).name)
        # auto-detect image size from first image
        try:
            img = cv2.imread(self._cb_files[0])
            if img is not None:
                self._img_h_var.set(img.shape[0])
                self._img_w_var.set(img.shape[1])
        except Exception:
            pass
        self._update_cmat_guess_principal_point()

    def _on_clear_cb_files(self) -> None:
        self._cb_files.clear()
        if self._file_listbox is not None:
            self._file_listbox.delete(0, tk.END)

    def _update_image_size_from_frame(self, frame: np.ndarray) -> None:
        """Sync the image-size widgets to the shape of an incoming frame."""
        if not isinstance(frame, np.ndarray):
            return
        if frame.ndim < 2:
            return
        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return
        self._img_h_var.set(int(h))
        self._img_w_var.set(int(w))
        self._update_cmat_guess_principal_point()

    def _update_cmat_guess_principal_point(self) -> None:
        """Update cx,cy in the camera matrix guess
        to match current image size."""
        if self._cmat_guess_widget is None:
            return
        w = self._img_w_var.get()
        h = self._img_h_var.get()
        m = _parse_matrix_text(
            self._cmat_guess_widget.get(
                "1.0", tk.END), 3)
        if m is not None and m.shape == (3, 3):
            m[0, 2] = w / 2.0 - 0.5
            m[1, 2] = h / 2.0 - 0.5
            self._cmat_guess_widget.delete(
                "1.0", tk.END)
            self._cmat_guess_widget.insert(
                "1.0", _array_to_text(m))

    def _on_validate_single(self) -> None:
        """Count valid (non-NaN) point pairs."""
        if self._pts3d_text_widget is None:
            return
        t3 = self._pts3d_text_widget.get(
            "1.0", tk.END)
        t2 = self._pts2d_text_widget.get(
            "1.0", tk.END)
        p3 = _parse_matrix_text(t3, 3)
        p2 = _parse_matrix_text(t2, 2)
        if p3 is None or p2 is None:
            self._validate_var.set(
                "Cannot parse coordinates.")
            return
        valid3 = ~np.any(np.isnan(p3), axis=1)
        valid2 = ~np.any(np.isnan(p2), axis=1)
        n = min(len(p3), len(p2))
        valid = valid3[:n] & valid2[:n]
        self._validate_var.set(
            f"{valid.sum()} valid pairs "
            f"(of {n} total)  "
            f"3D: {valid3.sum()}  "
            f"2D: {valid2.sum()}")

    def _on_find_corners(self) -> None:
        """Run findChessboardCorners in a background thread."""
        if not self._cb_files:
            messagebox.showwarning(
                "No images",
                "Please select calibration images first.")
            return
        if self._calib_running:
            return
        nx = self._cb_nx_var.get()
        ny = self._cb_ny_var.get()
        self._corner_status_var.set(
            "Finding corners...")
        self._status_var.set("finding corners...")
        self._calib_running = True

        subpix     = self._subpix_var.get()
        win        = self._subpix_win_var.get()
        max_iter   = self._subpix_iter_var.get()
        epsilon    = self._subpix_eps_var.get()
        files      = list(self._cb_files)

        def _worker():
            results = []
            for fpath in files:
                img = cv2.imread(
                    fpath, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    results.append((fpath, None))
                    continue
                flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
                         | cv2.CALIB_CB_NORMALIZE_IMAGE)
                found, corners = cv2.findChessboardCorners(
                    img, (nx, ny), flags)
                if found and subpix:
                    criteria = (
                        cv2.TERM_CRITERIA_EPS
                        | cv2.TERM_CRITERIA_MAX_ITER,
                        max_iter, epsilon)
                    corners = cv2.cornerSubPix(
                        img, corners,
                        (win, win), (-1, -1),
                        criteria)
                results.append((fpath,
                                 corners if found
                                 else None))
            return results

        def _done(future):
            results = future.result()
            self.canvas.after(
                0, lambda: self._on_corners_done(
                    results))

        import concurrent.futures
        ex = concurrent.futures.ThreadPoolExecutor(
            max_workers=1)
        fut = ex.submit(_worker)
        fut.add_done_callback(_done)

    def _on_corners_done(self, results) -> None:
        self._calib_running = False
        found_files = []
        found_corners = []
        failed = []
        for fpath, corners in results:
            if corners is not None:
                found_files.append(fpath)
                found_corners.append(corners)
            else:
                failed.append(Path(fpath).name)

        self._cb_files = found_files
        if self._file_listbox is not None:
            self._file_listbox.delete(0, tk.END)
            for f in found_files:
                self._file_listbox.insert(
                    tk.END, Path(f).name)

        # store detected corners for calibration
        self._detected_corners = found_corners

        msg = (f"Found corners in "
               f"{len(found_files)} / "
               f"{len(results)} images.")
        if failed:
            msg += f"  Failed: {', '.join(failed[:3])}"
            if len(failed) > 3:
                msg += f" ... +{len(failed)-3} more"
        self._corner_status_var.set(msg)
        self._status_var.set(msg)

        if self._vis_idx_sb is not None:
            self._vis_idx_sb.configure(
                to=max(1, len(found_files)))

    # ── calibration ───────────────────────────────────────────────

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

    def _on_calibrate(self) -> None:
        if self._calib_running:
            return
        mode = self._mode_var.get()
        if mode == "chessboard":
            self._calibrate_chessboard()
        else:
            self._calibrate_single()

    def _calibrate_chessboard(self) -> None:
        nx = self._cb_nx_var.get()
        ny = self._cb_ny_var.get()
        dx = self._cb_dx_var.get()
        dy = self._cb_dy_var.get()
        img_w = self._img_w_var.get()
        img_h = self._img_h_var.get()
        flags = self._get_flags()
        cmat_g = self._get_cmat_guess()
        dvec_g = self._get_dvec_guess()

        corners = getattr(self, "_detected_corners", None)
        if not corners:
            messagebox.showwarning(
                "No corners",
                "Please run 'Find corners' first.")
            return

        # generate object points
        one_board = np.zeros(
            (nx * ny, 3), dtype=np.float32)
        for r in range(ny):
            for c in range(nx):
                one_board[r*nx+c] = [c*dx, r*dy, 0]

        obj_pts = [one_board] * len(corners)
        img_pts = [c.reshape(-1, 1, 2).astype(
            np.float32) for c in corners]

        self._run_calibration_bg(
            obj_pts, img_pts,
            (img_w, img_h),
            cmat_g, dvec_g, flags)

    def _calibrate_single(self) -> None:
        if (self._pts3d_text_widget is not None
                and self._pts3d_text_widget.winfo_exists()):
            t3 = self._pts3d_text_widget.get("1.0", tk.END)
        else:
            t3 = self._pts3d_text
        if (self._pts2d_text_widget is not None
                and self._pts2d_text_widget.winfo_exists()):
            t2 = self._pts2d_text_widget.get("1.0", tk.END)
        else:
            t2 = self._pts2d_text
        p3 = _parse_matrix_text(t3, 3)
        p2 = _parse_matrix_text(t2, 2)
        if p3 is None or p2 is None:
            messagebox.showerror(
                "Parse error",
                "Cannot parse 3D or 2D coordinates.")
            return

        n = min(len(p3), len(p2))
        valid3 = ~np.any(np.isnan(p3[:n]), axis=1)
        valid2 = ~np.any(np.isnan(p2[:n]), axis=1)
        valid = valid3 & valid2
        vp3 = p3[:n][valid].astype(np.float32)
        vp2 = p2[:n][valid].astype(np.float32)

        if valid.sum() < 4:
            messagebox.showerror(
                "Too few points",
                "Need at least 4 valid point pairs.")
            return

        img_w = self._img_w_var.get()
        img_h = self._img_h_var.get()
        flags = self._get_flags()
        cmat_g = self._get_cmat_guess()
        dvec_g = self._get_dvec_guess()

        obj_pts = [vp3.reshape(-1, 1, 3)]
        img_pts = [vp2.reshape(-1, 1, 2)]

        self._run_calibration_bg(
            obj_pts, img_pts,
            (img_w, img_h),
            cmat_g, dvec_g, flags)

    def _run_calibration_bg(
        self,
        obj_pts, img_pts,
        img_size,
        cmat_g, dvec_g, flags
    ) -> None:
        self._calib_running = True
        self._status_var.set("calibrating...")
        if self._calib_btn is not None and \
                self._calib_btn.winfo_exists():
            self._calib_btn.configure(
                state="disabled")

        def _worker():
            cmat_in = (cmat_g.astype(np.float64)
                       if cmat_g is not None
                       else np.eye(3, dtype=np.float64))
            # calibrateCamera accepts a coefficient count matching its enabled
            # model flags.  Keep its input compact, then expose all 14 terms.
            full_dvec = (dvec_g.astype(np.float64)
                         if dvec_g is not None
                         else np.zeros(
                             (1, _DISTORTION_COEFFICIENT_COUNT),
                             dtype=np.float64))
            dvec_in = _calibration_distortion_vector(full_dvec, flags)
            ret, cmat, dvec, rvecs, tvecs = \
                cv2.calibrateCamera(
                    obj_pts, img_pts,
                    img_size,
                    cmat_in, dvec_in,
                    flags=flags)
            return ret, cmat, dvec, rvecs, tvecs, \
                obj_pts, img_pts

        def _done(future):
            try:
                (rms, cmat, dvec,
                 rvecs, tvecs,
                 op, ip) = future.result()
                self.canvas.after(
                    0,
                    lambda: self._on_calib_done(
                        rms, cmat, dvec,
                        rvecs, tvecs, op, ip))
            except Exception as e:
                self.canvas.after(
                    0,
                    lambda err=e:
                    self._on_calib_error(err))

        import concurrent.futures
        ex = concurrent.futures.ThreadPoolExecutor(
            max_workers=1)
        fut = ex.submit(_worker)
        fut.add_done_callback(_done)

    def _on_calib_error(self, err: Exception) -> None:
        self._calib_running = False
        msg = f"Calibration failed: {err}"
        self._status_var.set(msg)
        if (self._calib_btn is not None
                and self._calib_btn.winfo_exists()):
            self._calib_btn.configure(state="normal")

    def _on_calib_done(
        self,
        rms: float,
        cmat: np.ndarray,
        dvec: np.ndarray,
        rvecs, tvecs,
        obj_pts, img_pts
    ) -> None:
        self._calib_running = False
        if (self._calib_btn is not None
                and self._calib_btn.winfo_exists()):
            self._calib_btn.configure(state="normal")

        self._result_cmat  = np.asarray(cmat, dtype=np.float64).copy()
        self._result_dvec  = _as_full_distortion_vector(dvec).copy()
        self._result_rvecs = [np.asarray(r, dtype=np.float64).copy() for r in rvecs]
        self._result_tvecs = [np.asarray(t, dtype=np.float64).copy() for t in tvecs]
        self._result_rms   = float(rms)
        self._result_objpts = [
            np.asarray(o, dtype=np.float64).reshape(-1, 3).copy() for o in obj_pts]
        self._result_imgpts = [
            np.asarray(i, dtype=np.float64).reshape(-1, 2).copy() for i in img_pts]

        # compute per-image reprojection
        self._result_proj_pts  = []
        self._result_proj_errs = []
        for i in range(len(rvecs)):
            pp, _ = cv2.projectPoints(
                obj_pts[i].reshape(-1, 3),
                rvecs[i], tvecs[i], cmat, self._result_dvec)
            pp = np.asarray(pp, dtype=np.float64).reshape(-1, 2).copy()
            ip = np.asarray(img_pts[i], dtype=np.float64).reshape(-1, 2).copy()
            err = pp - ip
            self._result_proj_pts.append(pp)
            self._result_proj_errs.append(err)

        msg = (f"RMS={rms:.4f} px  "
               f"{len(rvecs)} image(s)")
        self._status_var.set(msg)

        self._refresh_result_text()
        self._on_vis_update()
        self._push_outputs()

    def _refresh_result_text(self) -> None:
        self._populate_result_edit_widgets()
        if self._result_text is None:
            return
        lines = []
        if self._result_rms is not None:
            lines.append(
                f"RMS reprojection error: "
                f"{self._result_rms:.6f} px")
        if self._result_cmat is not None:
            lines.append("\nCamera matrix:")
            lines.append(
                _array_to_text(self._result_cmat))
        if self._result_dvec is not None:
            lines.append("\nDistortion coefficients "
                         "(k1 k2 p1 p2 k3 k4 k5 k6 s1 s2 s3 s4 tau_x tau_y):")
            lines.extend(
                f"  {name:5s} = {value:.6g}"
                for name, value in zip(
                    _DISTORTION_COEFFICIENT_NAMES,
                    self._result_dvec.ravel()))
        if self._result_rvecs:
            n = len(self._result_rvecs)
            idx = min(
                self._vis_idx_var.get() - 1,
                n - 1)
            lines.append(
                f"\nRvec [image {idx+1}]:")
            lines.append(
                _array_to_text(
                    self._result_rvecs[idx].ravel()))
            lines.append(
                f"\nTvec [image {idx+1}]:")
            lines.append(
                _array_to_text(
                    self._result_tvecs[idx].ravel()))

        self._result_text.configure(state="normal")
        self._result_text.delete("1.0", tk.END)
        self._result_text.insert(
            "1.0", "\n".join(lines))
        self._result_text.configure(state="disabled")

    # ── visualization ─────────────────────────────────────────────

    def _get_background_image(self, vis_idx: int) -> np.ndarray | None:
        """Return the selected calibration image as the visualization background when available."""
        mode = self._mode_var.get()

        if mode == "chessboard" and vis_idx < len(self._cb_files):
            fpath = self._cb_files[vis_idx]
            raw = cv2.imread(fpath)
            if raw is not None:
                return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

        if vis_idx < len(getattr(self, "_collected_images", [])):
            raw = self._collected_images[vis_idx]
            if raw is None:
                return None
            if raw.ndim == 2:
                return cv2.cvtColor(raw, cv2.COLOR_GRAY2RGB)
            if raw.ndim == 3 and raw.shape[2] == 3:
                return raw.copy()
            if raw.ndim == 3 and raw.shape[2] == 4:
                return cv2.cvtColor(raw, cv2.COLOR_BGRA2RGB)

        return None

    def _on_vis_update(self) -> None:
        if self._result_cmat is None:
            return

        n = len(self._result_rvecs or [])
        if n == 0:
            return

        vis_idx = max(
            0, min(self._vis_idx_var.get() - 1,
                   n - 1))

        # get background image
        bg = self._get_background_image(vis_idx)
        if bg is None:
            w = self._img_w_var.get()
            h = self._img_h_var.get()
            bg = np.ones(
                (h, w, 3), dtype=np.uint8) * 240

        # gather data
        ip = (self._result_imgpts[vis_idx]
              if self._result_imgpts
              and vis_idx < len(
                  self._result_imgpts)
              else None)
        pp = (self._result_proj_pts[vis_idx]
              if self._result_proj_pts
              and vis_idx < len(
                  self._result_proj_pts)
              else None)
        rvec = self._result_rvecs[vis_idx]
        tvec = self._result_tvecs[vis_idx]

        # parse grid
        def _parse_1d(text_attr, var_attr):
            text = ""
            if hasattr(self, var_attr):
                text = getattr(self, var_attr).get()
            elif hasattr(self, text_attr):
                text = getattr(self, text_attr)
            if not isinstance(text, str):
                text = str(text or "")
            if text.strip() == self._grid_placeholder:
                return None
            arr = _parse_sequence_text(text)
            if arr is None:
                return None
            return arr.ravel()

        gx = _parse_1d(
            "_grid_xs_text", "_grid_xs_var")
        gy = _parse_1d(
            "_grid_ys_text", "_grid_ys_var")
        gz = _parse_1d(
            "_grid_zs_text", "_grid_zs_var")

        undist = self._undistort_var.get()
        scale = _clamp_undistort_scale(self._undistort_scale_var.get())

        def _draw_worker():
            return _draw_calib_result(
                bg, ip, pp,
                self._result_cmat,
                self._result_dvec,
                rvec, tvec,
                gx, gy, gz,
                undistort=undist,
                new_cmat=None,
                image_size_scale=scale)

        def _draw_done(future):
            try:
                vis_img = future.result()
                self.canvas.after(
                    0,
                    lambda img=vis_img:
                    self._push_vis_image(img))
            except Exception:
                pass

        import concurrent.futures
        ex = concurrent.futures.ThreadPoolExecutor(
            max_workers=1)
        fut = ex.submit(_draw_worker)
        fut.add_done_callback(_draw_done)

        self._refresh_result_text()

    def _push_vis_image(self,
                         vis_img: np.ndarray) -> None:
        self._last_vis_image = vis_img
        self._push_outputs()

    # ── output pins ───────────────────────────────────────────────

    def _image_size_array(self) -> np.ndarray:
        """'imgsize' output: [width, height] from the image-size widgets
        (kept in sync with incoming frames)."""
        return np.array([int(self._img_w_var.get()),
                         int(self._img_h_var.get())], dtype=np.int64)

    def _push_outputs(self) -> None:
        if self._result_cmat is None:
            return

        n = len(self._result_rvecs or [])
        vis_idx = max(
            0,
            min(self._vis_idx_var.get() - 1,
                n - 1)) if n > 0 else 0

        rvec = (self._result_rvecs[vis_idx].ravel()
                if self._result_rvecs else None)
        tvec = (self._result_tvecs[vis_idx].ravel()
                if self._result_tvecs else None)
        ip   = (self._result_imgpts[vis_idx]
                if self._result_imgpts
                and vis_idx < len(
                    self._result_imgpts)
                else None)
        op   = (self._result_objpts[vis_idx]
                if self._result_objpts
                and vis_idx < len(
                    self._result_objpts)
                else None)
        pp   = (self._result_proj_pts[vis_idx]
                if self._result_proj_pts
                and vis_idx < len(
                    self._result_proj_pts)
                else None)
        pe   = (self._result_proj_errs[vis_idx]
                if self._result_proj_errs
                and vis_idx < len(
                    self._result_proj_errs)
                else None)

        self._done_counter += 1
        outputs = {
            "cmat":       self._result_cmat,
            "dvec":       self._result_dvec,
            "rvec":       rvec,
            "tvec":       tvec,
            "imgsize":    self._image_size_array(),
            "image_pts":  ip,
            "object_pts": op,
            "proj_pts":   pp,
            "proj_err":   pe,
            "rms":        float(
                self._result_rms or 0.0),
            "done":       self._done_counter,
        }
        if hasattr(self, "_last_vis_image") and \
                self._last_vis_image is not None:
            outputs["vis_image"] = \
                self._last_vis_image

        if self._on_output_ready:
            self._on_output_ready(
                self.node_id, outputs)

    # ── compute ───────────────────────────────────────────────────

    def compute(self, inputs: dict) -> dict:
        # collect images from upstream ImageSequenceNode
        frame = inputs.get("images")
        if frame is not None and \
                isinstance(frame, np.ndarray):
            self._update_image_size_from_frame(frame)
            self._collected_images.append(
                frame.copy())

        # handle trigger pin: fire once per rising edge only, same as
        # clicking the Calibrate button once (a pulse also resets back to
        # 0/low afterwards, which must not re-trigger a second calibration).
        trigger = inputs.get("trigger")
        trigger_high = trigger is not None and self._coerce_bool_like(trigger)
        if trigger_high and not self._trigger_latched_high:
            self.canvas.after(
                0, self._on_calibrate)
        self._trigger_latched_high = trigger_high
        self._last_trigger = trigger

        # return last results if available
        if self._result_cmat is None:
            return {}

        n = len(self._result_rvecs or [])
        vis_idx = max(
            0,
            min(self._vis_idx_var.get() - 1,
                n - 1)) if n > 0 else 0

        result: dict = {
            "cmat": self._result_cmat.copy(),
            "dvec": self._result_dvec.copy(),
            "imgsize": self._image_size_array(),
            "rms":  float(
                self._result_rms or 0.0),
            "done": self._done_counter,
        }
        if self._result_rvecs:
            result["rvec"] = self._result_rvecs[vis_idx].ravel().copy()
            result["tvec"] = self._result_tvecs[vis_idx].ravel().copy()
        if self._result_imgpts and vis_idx < len(self._result_imgpts):
            result["image_pts"] = self._result_imgpts[vis_idx].copy()
        if self._result_objpts and vis_idx < len(self._result_objpts):
            result["object_pts"] = self._result_objpts[vis_idx].copy()
        if self._result_proj_pts and vis_idx < len(self._result_proj_pts):
            result["proj_pts"] = self._result_proj_pts[vis_idx].copy()
        if self._result_proj_errs and vis_idx < len(self._result_proj_errs):
            result["proj_err"] = self._result_proj_errs[vis_idx].copy()
        if hasattr(self, "_last_vis_image") and self._last_vis_image is not None:
            result["vis_image"] = np.asarray(self._last_vis_image).copy()
        return result

    # ── serialization ─────────────────────────────────────────────

    def _sync_text_widgets(self) -> None:
        if (self._pts3d_text_widget is not None
                and self._pts3d_text_widget.winfo_exists()):
            self._pts3d_text = \
                self._pts3d_text_widget.get(
                    "1.0", tk.END).rstrip()
        if (self._pts2d_text_widget is not None
                and self._pts2d_text_widget.winfo_exists()):
            self._pts2d_text = \
                self._pts2d_text_widget.get(
                    "1.0", tk.END).rstrip()
        if (self._cmat_guess_widget is not None
                and self._cmat_guess_widget.winfo_exists()):
            self._cmat_guess_text = \
                self._cmat_guess_widget.get(
                    "1.0", tk.END).rstrip()
        if (self._dvec_guess_widget is not None
                and self._dvec_guess_widget.winfo_exists()):
            self._dvec_guess_text = \
                self._dvec_guess_widget.get(
                    "1.0", tk.END).rstrip()
        if hasattr(self, "_grid_xs_var"):
            self._grid_xs_text = \
                self._grid_xs_var.get()
            self._grid_ys_text = \
                self._grid_ys_var.get()
            self._grid_zs_text = \
                self._grid_zs_var.get()

    def get_params(self) -> dict:
        self._sync_text_widgets()
        flag_values = [v.get()
                       for v in self._flag_vars]
        return {
            "mode":           self._mode_var.get(),
            "cb_nx":          self._cb_nx_var.get(),
            "cb_ny":          self._cb_ny_var.get(),
            "cb_dx":          self._cb_dx_var.get(),
            "cb_dy":          self._cb_dy_var.get(),
            "cb_files":       "\n".join(
                self._cb_files),
            "img_w":          self._img_w_var.get(),
            "img_h":          self._img_h_var.get(),
            "pts3d":          self._pts3d_text,
            "pts2d":          self._pts2d_text,
            "cmat_guess":     self._cmat_guess_text,
            "dvec_guess":     self._dvec_guess_text,
            "subpix":         self._subpix_var.get(),
            "subpix_win":     self._subpix_win_var.get(),
            "subpix_iter":    self._subpix_iter_var.get(),
            "subpix_eps":     self._subpix_eps_var.get(),
            "flag_values":    flag_values,
            "grid_xs":        self._grid_xs_text,
            "grid_ys":        self._grid_ys_text,
            "grid_zs":        self._grid_zs_text,
            "undistort":      self._undistort_var.get(),
            "undistort_scale": float(self._undistort_scale_var.get()),
            "vis_idx":        self._vis_idx_var.get(),
        }

    def set_params(self, params: dict) -> None:
        self._init_state()
        self._mode_var.set(
            params.get("mode", "chessboard"))
        self._cb_nx_var.set(
            int(params.get("cb_nx", 7)))
        self._cb_ny_var.set(
            int(params.get("cb_ny", 7)))
        self._cb_dx_var.set(
            float(params.get("cb_dx", 25.4)))
        self._cb_dy_var.set(
            float(params.get("cb_dy", 25.4)))
        self._img_w_var.set(
            int(params.get("img_w", 1920)))
        self._img_h_var.set(
            int(params.get("img_h", 1080)))
        self._pts3d_text = str(
            params.get("pts3d",
                       self._pts3d_text))
        self._pts2d_text = str(
            params.get("pts2d",
                       self._pts2d_text))
        self._cmat_guess_text = str(
            params.get("cmat_guess",
                       self._cmat_guess_text))
        self._dvec_guess_text = str(
            params.get("dvec_guess",
                       self._dvec_guess_text))
        self._subpix_var.set(
            bool(params.get("subpix", True)))
        self._subpix_win_var.set(
            int(params.get("subpix_win", 11)))
        self._subpix_iter_var.set(
            int(params.get("subpix_iter", 30)))
        self._subpix_eps_var.set(
            float(params.get("subpix_eps", 0.001)))
        flag_values = params.get(
            "flag_values",
            [v.get() for v in self._flag_vars])
        for i, v in enumerate(flag_values):
            if i < len(self._flag_vars):
                self._flag_vars[i].set(int(v))
        self._grid_xs_text = str(
            params.get("grid_xs", ""))
        self._grid_ys_text = str(
            params.get("grid_ys", ""))
        self._grid_zs_text = str(
            params.get("grid_zs", "0"))
        self._undistort_var.set(
            bool(params.get("undistort", False)))
        scale_value = params.get("undistort_scale")
        if scale_value is None:
            scale_value = params.get("undistort_alpha", 1.0)
        self._undistort_scale_var.set(
            float(scale_value))
        self._vis_idx_var.set(
            int(params.get("vis_idx", 1)))
        # restore file list
        files_str = str(
            params.get("cb_files", ""))
        self._cb_files = [
            f.strip()
            for f in files_str.splitlines()
            if f.strip()]

    def close_inspector(self) -> None:
        self._sync_text_widgets()
        super().close_inspector()
        self._nb                 = None
        self._file_listbox       = None
        self._pts3d_text_widget  = None
        self._pts2d_text_widget  = None
        self._cmat_guess_widget  = None
        self._dvec_guess_widget  = None
        self._result_text        = None
        self._vis_idx_sb         = None
        self._calib_btn          = None

    def on_destroy(self) -> None:
        self._sync_text_widgets()
        super().on_destroy()
