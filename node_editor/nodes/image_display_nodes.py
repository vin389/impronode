# node_editor/nodes/image_display_nodes.py

import time
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from node_editor.base_node import BaseNode
from node_editor.pin_types import PinSchema, PinDef, PinType
from node_editor.execution import ExecutionMode


_IMAGE_DISPLAY_HELP = """\
Image Display
=============
Shows the images arriving on the 'image' pin, lets you pick points on them,
and runs a few OpenCV analyses on the shown frame when you ask for it.

ON THE CANVAS
  The node shows only a short summary: image size and type, frame rate,
  "frozen" when frozen, and the number of picked points.
  Double-click the node to open the inspector, where the image is.

PINS
  image  [in]   frame to display (RGB colour or grey; float images are
                scaled to their min..max for display).
  points [in]   optional N x 2 / N x 3 / N x 4 array drawn on the image:
                x, y [, window size] or [, window width, height] -- e.g.
                tracked points. Style: Display tab.
  points [out]  the PICKED points, N x 2 float32 (x, y), in list order.
                (Replaces the old single-point 'coords' output; projects that
                used it are reconnected to 'points' automatically.)

QUICK START (first time here?)
  1. Move the mouse over the image. The MAGNIFIER (top right) shows the
     area under the cursor, strongly zoomed; its red cross is exactly at
     the cursor position, which is written next to it (x, y, pixel value).
  2. Want a precise point? Tick "Corners" (Alt+C) in the magnifier, move
     close to the corner, then press S: the detected corner nearest to the
     cross is added to the output points (no clicking, no hand shake).
  3. Mode "Pick points" (Alt+I): click to add points. Mode "Pan" (Alt+P):
     drag to move the image. Every button has an Alt+key (underlined letter).

IMAGE VIEW (left side of the inspector)
  Wheel or + / -   zoom at the cursor         Left-drag   pan
  Fit / 1:1        fit the window / 1 image pixel = 1 screen pixel
  Double-click     fit (Pan mode)             Right-click menu
  Freeze           keep showing the current frame; new frames are not shown
                   (upstream keeps running). Analyses use the SHOWN frame,
                   so freeze a live video before analysing it.
  Grid             pixel grid once zoomed in far enough.
  Interpolation    used when zoomed in (zooming out always averages).
  The status bar shows the cursor position in image pixels and the pixel
  value. Coordinates follow OpenCV: (0, 0) is the CENTRE of the top-left
  pixel. The dashed orange square on the image is the magnifier's area.

MAGNIFIER (top right) -- a zoomed look around the cursor
  What you see
    The image around the "probe" point, enlarged 1X..64X (16X: every image
    pixel is 16 x 16 screen pixels), drawn with the same Interp. method as
    the main view (NEAREST = sharp blocks, LINEAR / CUBIC / LANCZOS4 =
    smooth; AREA is shown as LINEAR here). The thin red CROSS never moves:
    it marks the probe exactly; the zoomed pixels slide under it -- as many
    times faster than your mouse as the zoom, so you see sub-pixel positions.
  The probe
    = the cursor while it is over the image. When the mouse LEAVES the
    image, the probe stays where it was (a small orange cross marks it), so
    you can look at the magnifier and use the keys without touching the
    mouse. Arrow keys move the probe by 1 pixel, Shift+arrows by 0.1 pixel
    (click on the image or the magnifier first, so it has keyboard focus).
  Zoom
    Buttons 1X 2X 4X 8X 16X 32X 64X, or Alt+0 ... Alt+6 (Alt+0 = 1X,
    Alt+4 = 16X, Alt+6 = 64X), or the mouse wheel over the magnifier.
  Detect (instant, re-run every time the magnified area changes)
    Corners   cv2.cornerHarris + local maxima; "threshold" = fraction of
              the strongest corner response (0.05 = keep corners at least
              5 % as strong as the best one); "sub-pixel" refines them with
              cv2.cornerSubPix. Drawn as CYAN circles.
    ArUco     cv2.aruco marker detector with the chosen dictionary (it MUST
              match the printed markers, e.g. DICT_4X4_50); "sub-pixel"
              refines the marker corners. Searched in the WHOLE frame
              (once per frame), so a marker is found even when only part
              of it is inside the magnifier. Drawn in YELLOW with the
              id; corner 0 has the thicker square.
    Features  cv2.goodFeaturesToTrack: "max" points, "quality" = like the
              corner threshold. It finds corners ON THE PIXEL GRID (whole
              numbers); "sub-pixel" refines them with cv2.cornerSubPix.
              Drawn as GREEN plus signs.
    The detection nearest to the cross is circled in PINK and its position
    and distance are shown; press S to add it to the output points.
    Tip: if nothing is found, zoom OUT (Alt+1 or Alt+0): a lower zoom
    shows (and searches) a bigger area.
  Drag the bar between the magnifier and the tabs to give either more room;
  each of the two panels gets a scroll bar when its content does not fit.

POINTS TAB -- two boxes (drag the bar between them to resize)
  Input points (read-only)
    The points arriving on the INPUT pin 'points' (all columns). Select and
    copy with Ctrl+C / Ctrl+A, or "Copy all". Refreshed at most 4 times a
    second; updates pause while you have text selected (so a selection you
    are copying is not replaced) or when "Live" is off.
    "-> Add to output" appends their x, y to the Output list and sends it.
  Output points (editable) -- what the OUTPUT pin 'points' sends
    The header says what is sent ("Sent: N points"). Typing in the box does
    NOT send anything: the box turns yellow ("Edited - NOT sent yet") until
    you press "Apply (send)" (or Ctrl+Enter); "Revert" discards the edits.
    One "x, y" per line (also "x y", "(x, y)", "[x, y]"; # = comment).
  Image actions send at once:
    Mode "Pick":  a click adds a point; drag an output point to move it;
                  dragging anywhere else still pans.
    Key C         (any mode) adds the probe point (= under the cursor / at
                  the magnifier's cross) and copies "x, y" to the clipboard.
    Key S         adds the detection nearest to the magnifier's cross
                  (snap; needs Corners / ArUco / Features ticked).
    Ctrl+Z        removes the last point.   Right-click: add / remove / copy.
    If the Output box has typed edits that are not sent yet, an image action
    first APPLIES (sends) them, then adds / moves / removes the point -- your
    typing is never lost. Only if a typed line is not a valid "x, y" is the
    action refused; a message on the image says so (fix the line or Revert).
  Saving the project stores the SENT list.

ANALYSIS TAB -- runs ONCE on the shown frame when you press Apply
  Good features  cv2.goodFeaturesToTrack (corners), optional sub-pixel refine
  Chessboard     cv2.findChessboardCorners (inner corners: cols x rows),
                 optional cornerSubPix
  ArUco markers  cv2.aruco detector (dictionary must match the markers).
                 FIVE points per marker: corners 0, 1, 2, 3 (clockwise from
                 the marker's own top-left corner) and the centre. Centre =
                 intersection of the diagonals (true centre of the square,
                 also in perspective) or the plain mean of the 4 corners.
                 Markers sorted by id (or as detected). Corner refinement:
                 subpix (recommended), contour, apriltag or none.
  Hough lines    Canny edges, then cv2.HoughLinesP (segments) or
                 cv2.HoughLines (infinite lines given as rho, theta)
  Sobel          |d/dx|, |d/dy| or gradient magnitude, shown as an image
  Canny          edge image
  The result is drawn on the image -- for Sobel / Canny it is shown INSTEAD
  of the image while "Show result image" is on -- and listed in the result
  box (selectable / copyable). "Add result points to picked list" copies
  point results (or line end points) into the Points list, so they go to
  the output pin.

DISPLAY TAB
  Cursor read-out and frame rate in the status bar; marker, colour and line
  width of the input 'points' overlay.

KEYBOARD SHORTCUTS
  Alt+key works anywhere in the inspector (the underlined letter in a label
  is its key). Plain keys work while the image or the magnifier has the
  keyboard focus (move the mouse over it or click it).
  View            Alt+T  fit             Alt+O  1:1
                  Alt+E  freeze on/off   Alt+G  grid on/off
                  Alt+L  next interpolation method
                  + / -  zoom in / out   wheel  zoom at the cursor
  Mode            Alt+P  Pan             Alt+I  Pick points
  Magnifier       Alt+0 .. Alt+6   zoom 1X .. 64X  (wheel over it too)
                  Alt+C  Corners     Alt+A  ArUco     Alt+F  Features
                  arrows  move the probe 1 px    Shift+arrows  0.1 px
  Points          C  add the probe point (and copy it)
                  S  add the nearest detection (snap)
                  Ctrl+Z  remove the last point
                  Ctrl+Enter  Apply (send) the edited Output box
  Analysis        Ctrl+R  Apply the selected analysis
  Tabs            Ctrl+Tab / Ctrl+Shift+Tab   next / previous tab
  Help            Ctrl+H
"""


ARUCO_DICT_NAMES = (
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
    "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
    "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
    "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL", "DICT_APRILTAG_16h5", "DICT_APRILTAG_25h9",
    "DICT_APRILTAG_36h10", "DICT_APRILTAG_36h11", "DICT_ARUCO_MIP_36h12")


def _aruco_center(quad: np.ndarray, mode: str = "diagonal intersection") -> np.ndarray:
    """Centre of a marker's 4 corners. 'diagonal intersection' (corner 0-2
    crossed with corner 1-3) is the true centre of the square even under
    perspective; 'mean of 4 corners' is shifted towards the far side."""
    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    if mode.startswith("mean"):
        return q.mean(axis=0)
    h = lambda a: np.array([a[0], a[1], 1.0])                     # noqa: E731
    pt = np.cross(np.cross(h(q[0]), h(q[2])), np.cross(h(q[1]), h(q[3])))
    if abs(pt[2]) < 1e-12:                                        # degenerate quad
        return q.mean(axis=0)
    return pt[:2] / pt[2]


def _parse_point_lines(text: str):
    """Parse the Picked points text. Returns (points, errors); errors is a
    list of 'line N: ...' messages (points is None when there are errors)."""
    points, errors = [], []
    for n, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        cleaned = line.strip("()[]{} ").replace(",", " ").replace(";", " ")
        parts = cleaned.split()
        try:
            if len(parts) != 2:
                raise ValueError
            x, y = float(parts[0]), float(parts[1])
            if not (np.isfinite(x) and np.isfinite(y)):
                raise ValueError
        except ValueError:
            errors.append(f"line {n}: '{line}' is not 'x, y'")
            continue
        points.append((x, y))
    return (None if errors else points), errors


class ImageDisplayNode(BaseNode):
    """Display images, pick points and run one-shot OpenCV analyses (all in the inspector)."""

    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE = "image_display_output"
    DISPLAY_NAME = "Image Display"
    CATEGORY = "visualize"
    NODE_WIDTH = 220
    NODE_HEIGHT = 110
    HELP_TEXT = _IMAGE_DISPLAY_HELP
    # Projects saved before the redesign linked the picked-point output under
    # its old name; the editor maps it to the new pin when loading.
    OUTPUT_PIN_ALIASES = {"image_coords": "points"}

    _INTERPOLATIONS = {
        "NEAREST": cv2.INTER_NEAREST, "LINEAR": cv2.INTER_LINEAR, "CUBIC": cv2.INTER_CUBIC,
        "AREA": cv2.INTER_AREA, "LANCZOS4": cv2.INTER_LANCZOS4,
    }
    _MARKERS = ("+", "*", "o")
    _GRID_MIN_SCALE = 6.0          # draw the pixel grid at >= 6 screen px per image px
    _MAX_GRID_LINES = 400
    _ZOOM_STEP = 1.15
    _MIN_SCALE, _MAX_SCALE = 0.01, 200.0
    _CLICK_PX = 4                  # movement below this is a click, not a drag
    _GRAB_PX = 7                   # grab radius for moving a picked point
    _PICK_COLOR = "#00e5ff"
    _RESULT_POINT_COLOR = "#ff4dd2"
    _RESULT_LINE_COLOR = "#ffd23f"
    _FACE_REFRESH_S = 0.25
    _MAX_LISTED = 500              # result lines listed in the result box

    # Magnifier (loupe): a fixed-size canvas showing the image around the
    # PROBE point (the cursor, or where it was last / was moved with keys).
    _LOUPE_SIZE = 216
    _LOUPE_ZOOMS = (1, 2, 4, 8, 16, 32, 64)    # radio buttons; Alt+0..6 = 1X..64X
    _LOUPE_CORNER_COLOR = "#00e5ff"
    _LOUPE_FEATURE_COLOR = "#7cfc00"
    _LOUPE_ARUCO_COLOR = "#ffd23f"
    _LOUPE_NEAREST_COLOR = "#ff4dd2"
    _ARUCO_DICTS = ARUCO_DICT_NAMES

    # Analysis tools: key -> (label, [(param, label, default, kind)]);
    # kind: "int" | "float" | "bool" | "choice:a|b|c".
    _TOOLS = {
        "gftt": ("Good features to track (corners)", [
            ("max_corners", "Max corners", 100, "int"),
            ("quality", "Quality level (0-1)", 0.01, "float"),
            ("min_distance", "Min distance (px)", 10.0, "float"),
            ("block_size", "Block size (px)", 3, "int"),
            ("use_harris", "Use Harris detector", False, "bool"),
            ("k", "Harris k", 0.04, "float"),
            ("subpix", "Sub-pixel refine", True, "bool"),
        ]),
        "chessboard": ("Chessboard corners", [
            ("cols", "Inner corners per row (cols)", 9, "int"),
            ("rows", "Inner corners per column (rows)", 6, "int"),
            ("adaptive", "Adaptive threshold", True, "bool"),
            ("normalize", "Normalize image", True, "bool"),
            ("fast", "Fast check", False, "bool"),
            ("subpix", "Sub-pixel refine", True, "bool"),
            ("subpix_win", "Sub-pixel half-window (px)", 11, "int"),
        ]),
        "hough": ("Hough lines", [
            ("variant", "Variant", "segments (HoughLinesP)",
             "choice:segments (HoughLinesP)|lines (HoughLines)"),
            ("canny1", "Canny low threshold", 50.0, "float"),
            ("canny2", "Canny high threshold", 150.0, "float"),
            ("rho", "rho resolution (px)", 1.0, "float"),
            ("theta_deg", "theta resolution (deg)", 1.0, "float"),
            ("threshold", "Accumulator threshold", 80, "int"),
            ("min_length", "Min line length (px, segments)", 30.0, "float"),
            ("max_gap", "Max line gap (px, segments)", 10.0, "float"),
        ]),
        "aruco": ("ArUco markers (4 corners + centre)", [
            ("dictionary", "Dictionary", "DICT_4X4_50", "choice:" + "|".join(ARUCO_DICT_NAMES)),
            ("refine", "Corner refinement", "subpix", "choice:subpix|contour|apriltag|none"),
            ("center", "5th point (centre)", "diagonal intersection",
             "choice:diagonal intersection|mean of 4 corners"),
            ("order", "Marker order", "by id", "choice:by id|as detected"),
            ("win_min", "Adaptive threshold window min (px)", 3, "int"),
            ("win_max", "Adaptive threshold window max (px)", 23, "int"),
            ("min_perimeter", "Min marker perimeter (x image size)", 0.03, "float"),
        ]),
        "sobel": ("Sobel gradient", [
            ("output", "Output", "magnitude", "choice:magnitude|d/dx|d/dy"),
            ("ksize", "Kernel size (1, 3, 5, 7)", 3, "int"),
        ]),
        "canny": ("Canny edges", [
            ("threshold1", "Low threshold", 50.0, "float"),
            ("threshold2", "High threshold", 150.0, "float"),
            ("aperture", "Aperture size (3, 5, 7)", 3, "int"),
            ("l2", "L2 gradient", False, "bool"),
        ]),
    }

    def __init__(self, node_id: str, canvas: tk.Canvas):
        super().__init__(node_id, canvas)
        # Plain Python state only; Tk variables are created lazily in
        # _init_state() (the toolbox builds throw-away nodes while dragging).
        self._frame = None                       # last received frame
        self._shown = None                       # frame shown in the view (= _frame unless frozen)
        self._input_points = np.empty((0, 2), np.float32)
        self._picked: list = []                  # applied picked points [(x, y), ...]
        self._frame_times: list = []
        self._last_face_update = 0.0
        self._status_item = None
        self._hint_item = None
        self._result = None                      # last analysis result (dict)
        self._view = None                        # (ox, oy, s): canvas = o + (px + 0.5) * s
        self._user_view = False
        self._view_key = None                    # (image size, canvas size) the fit was made for
        self._drag = None                        # current mouse gesture on the view
        self._render_pending = False
        self._photo = None
        self._photo_key = None
        self._display_cache = (None, None)       # (source key, display-ready uint8 RGB)
        self._frame_serial = 0                   # +1 per received frame (cache key; ids can be reused)
        self._result_serial = 0
        self._last_pick = (0.0, None)            # (time, (x, y)) -- ignore double-click duplicates
        self._cursor_xy = None                   # cursor position over the view, canvas px
        self._probe = None                       # (x, y) image px shown in the magnifier
        self._loupe_pending = False
        self._loupe_photo = None
        self._loupe_nearest = None               # (kind, x, y) nearest detection to the probe
        self._aruco_cache = (None, None)         # (settings key, cv2.aruco.ArucoDetector)
        self._aruco_frame_cache = (None, [])     # (frame + settings key, [(id, 4x2 corners)])
        self._last_input_refresh = 0.0
        self._help_popup = None
        self._reset_widget_refs()

    # ── lazily created Tk state ────────────────────────────────────────
    def _init_state(self) -> None:
        if hasattr(self, "_freeze_var"):
            return
        self._freeze_var = tk.BooleanVar(value=False)
        self._show_fps_var = tk.BooleanVar(value=True)
        self._show_coords_var = tk.BooleanVar(value=True)
        self._show_grid_var = tk.BooleanVar(value=False)
        self._interp_var = tk.StringVar(value="NEAREST")
        self._mode_var = tk.StringVar(value="pan")
        self._marker_var = tk.StringVar(value="+")
        self._marker_red_var = tk.IntVar(value=0)
        self._marker_green_var = tk.IntVar(value=255)
        self._marker_blue_var = tk.IntVar(value=0)
        self._marker_width_var = tk.IntVar(value=2)
        self._show_input_points_var = tk.BooleanVar(value=True)
        self._show_picked_var = tk.BooleanVar(value=True)
        self._show_numbers_var = tk.BooleanVar(value=True)
        self._input_live_var = tk.BooleanVar(value=True)
        # Magnifier.
        self._loupe_zoom_var = tk.IntVar(value=16)
        self._det_corners_var = tk.BooleanVar(value=False)
        self._det_aruco_var = tk.BooleanVar(value=False)
        self._det_features_var = tk.BooleanVar(value=False)
        self._corner_subpix_var = tk.BooleanVar(value=True)
        self._corner_thresh_var = tk.StringVar(value="0.05")
        self._aruco_dict_var = tk.StringVar(value="DICT_4X4_50")
        self._aruco_refine_var = tk.BooleanVar(value=True)
        self._feat_max_var = tk.StringVar(value="20")
        self._feat_quality_var = tk.StringVar(value="0.01")
        self._feat_subpix_var = tk.BooleanVar(value=True)
        for var in (self._feat_subpix_var, self._loupe_zoom_var, self._det_corners_var, self._det_aruco_var, self._det_features_var,
                    self._corner_subpix_var, self._corner_thresh_var, self._aruco_dict_var,
                    self._aruco_refine_var, self._feat_max_var, self._feat_quality_var):
            var.trace_add("write", lambda *_a: (self._request_loupe(), self._request_render()))
        self._input_live_var.trace_add("write", lambda *_a: self._refresh_input_text(force=True))
        self._tool_var = tk.StringVar(value="gftt")
        self._show_result_var = tk.BooleanVar(value=True)
        self._show_result_image_var = tk.BooleanVar(value=True)
        self._tool_vars = {}
        for tool, (_label, params) in self._TOOLS.items():
            vs = {}
            for key, _plabel, default, kind in params:
                vs[key] = tk.BooleanVar(value=default) if kind == "bool" else tk.StringVar(value=str(default))
            self._tool_vars[tool] = vs
        for var in (self._show_grid_var, self._interp_var, self._marker_var, self._marker_red_var,
                    self._marker_green_var, self._marker_blue_var, self._marker_width_var,
                    self._show_input_points_var, self._show_picked_var, self._show_numbers_var,
                    self._show_result_var, self._show_result_image_var):
            var.trace_add("write", lambda *_a: self._request_render())
        self._freeze_var.trace_add("write", lambda *_a: self._on_freeze_changed())
        # The tool box and parameter fields follow the selected tool, however it
        # was set (combobox, loaded settings, ...).
        self._tool_var.trace_add("write", lambda *_a: self._on_tool_var_changed())

    def _reset_widget_refs(self) -> None:
        self._c_view = None
        self._status_bar_var = None
        self._points_text = None
        self._input_text = None
        self._input_info_var = None
        self._input_shown = None
        self._out_state_var = None
        self._out_state = None
        self._points_msg = None
        self._points_msg_var = None
        self._result_text = None
        self._param_frame = None
        self._tool_combo = None
        self._add_points_btn = None
        self._result_image_cb = None
        self._text_dirty = False
        self._c_loupe = None
        self._loupe_info_var = None
        self._loupe_nearest_var = None
        self._loupe_zoom_label_var = None

    # ── pins ──────────────────────────────────────────────────────────
    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[
                PinDef("image", PinType.IMAGE, "frame"),
                PinDef("points", PinType.ARRAY, "points", optional=True),
            ],
            outputs=[
                PinDef("points", PinType.ARRAY, "points", shape=(-1, 2), dtype="float32"),
            ],
        )

    # ── node face on the canvas ───────────────────────────────────────
    _PIN_LABEL_IN, _PIN_LABEL_OUT = 46, 48       # px taken by the pin labels

    def _text_column(self, x, w):
        left, right = x + self._PIN_LABEL_IN, x + w - self._PIN_LABEL_OUT
        return (left + right) / 2, max(60, right - left)

    def build_body(self) -> None:
        self._init_state()
        x, y, w, h = self.x, self.y, self.width, self.height
        self._body_rect = self.canvas.create_rectangle(
            x, y, x + w, y + h, fill="#eef3fb", outline="#4488cc", width=2,
            tags=(self.node_id, "node_body"))
        self._title_item = self.canvas.create_text(
            x + w / 2, y + 13, text=self.get_canvas_title(), font=("Arial", 9, "bold"),
            fill="#1c3a6b", tags=(self.node_id,))
        cx, tw = self._text_column(x, w)
        self._hint_item = self.canvas.create_text(
            cx, y + 33, text="double-click to view", font=("Arial", 8), fill="#35507d",
            width=tw, justify="center", tags=(self.node_id,))
        self._status_item = self.canvas.create_text(
            cx, y + h - 5, text=self._face_text(), font=("Arial", 7), fill="#456",
            width=tw, anchor="s", justify="center", tags=(self.node_id,))
        self._canvas_items += [self._body_rect, self._title_item, self._hint_item, self._status_item]

    def on_resize(self, old_width, old_height, new_width, new_height) -> None:
        cx, tw = self._text_column(self.x, new_width)
        for item, yy in ((self._hint_item, self.y + 33), (self._status_item, self.y + new_height - 5)):
            if item is not None:
                self.canvas.coords(item, cx, yy)
                self.canvas.itemconfigure(item, width=tw)

    def _fps(self) -> float:
        now = time.perf_counter()
        recent = [t for t in self._frame_times if now - t < 2.0]
        return len(recent) / 2.0

    def _face_text(self) -> str:
        frame = self._frame
        if frame is None:
            line1 = "waiting for frames"
        else:
            h, w = frame.shape[:2]
            line1 = f"{w} x {h} {frame.dtype}"
        line2 = f"{self._fps():.1f} fps" + ("  (frozen)" if self._is_frozen() else "")
        n = len(self._picked)
        line3 = f"{n} picked point{'s' if n != 1 else ''}"
        return f"{line1}\n{line2}\n{line3}"

    def _update_face(self, force: bool = False) -> None:
        now = time.perf_counter()
        if not force and now - self._last_face_update < self._FACE_REFRESH_S:
            return                                   # cheap at 30 fps: refresh ~4x/s
        self._last_face_update = now
        if self._status_item is not None:
            try:
                self.canvas.itemconfigure(self._status_item, text=self._face_text())
            except tk.TclError:
                pass

    def _is_frozen(self) -> bool:
        return hasattr(self, "_freeze_var") and bool(self._freeze_var.get())

    # ── compute ───────────────────────────────────────────────────────
    def compute(self, inputs: dict) -> dict:
        self._init_state()
        frame = inputs.get("image")
        self._input_points = self._normalise_points(inputs.get("points"))
        if isinstance(frame, np.ndarray) and frame.ndim in (2, 3) and frame.size:
            now = time.perf_counter()
            self._frame_times = [t for t in self._frame_times if now - t < 2.0] + [now]
            self._frame = frame
            if not self._freeze_var.get():
                self._shown = frame
                self._frame_serial += 1
        self._update_face()
        self._request_render()
        self._refresh_input_text()
        return {"points": self._picked_array()} if self._picked else {}

    @staticmethod
    def _normalise_points(points) -> np.ndarray:
        """Finite N x (2..4) image-coordinate rows of the 'points' input."""
        if points is None:
            return np.empty((0, 2), dtype=np.float32)
        try:
            array = np.asarray(points, dtype=np.float32)
        except (TypeError, ValueError):
            return np.empty((0, 2), dtype=np.float32)
        if array.ndim == 3 and array.shape[1] == 1:       # OpenCV (N, 1, 2) layout
            array = array[:, 0, :]
        if array.ndim != 2 or array.shape[1] not in (2, 3, 4):
            return np.empty((0, 2), dtype=np.float32)
        return array[np.isfinite(array).all(axis=1)]

    def _picked_array(self) -> np.ndarray:
        return np.asarray(self._picked, dtype=np.float32).reshape(-1, 2)

    def _emit_points(self) -> None:
        """Send the picked points downstream now (not only on the next frame)."""
        self.push_output({"points": self._picked_array()} if self._picked else {})
        self._update_face(force=True)

    def _on_freeze_changed(self) -> None:
        if not self._freeze_var.get() and self._frame is not None:
            self._shown = self._frame
            self._frame_serial += 1
        self._update_face(force=True)
        self._request_render()

    # ══ Inspector ═════════════════════════════════════════════════════
    def build_inspector(self, parent: tk.Frame) -> None:
        self._init_state()
        self._reset_widget_refs()
        win = self._inspector_win
        if win is not None:
            sw, sh = win.winfo_screenwidth(), win.winfo_screenheight()
            win.geometry(f"{min(1300, sw - 80)}x{min(900, sh - 110)}")
            win.minsize(min(900, sw - 80), min(520, sh - 120))
            for seq in ("<Control-h>", "<Control-H>"):
                win.bind(seq, lambda _e: self._open_help())

        panes = tk.PanedWindow(parent, orient=tk.HORIZONTAL, sashwidth=6, sashrelief=tk.RAISED)
        panes.pack(fill="both", expand=True)
        left, right = tk.Frame(panes), tk.Frame(panes)
        panes.add(left, minsize=420, stretch="always")
        panes.add(right, width=465, minsize=380, stretch="never")
        self._build_view(left)

        # Right side: the magnifier on top, the tabs below (drag the bar between them).
        rpanes = tk.PanedWindow(right, orient=tk.VERTICAL, sashwidth=6, sashrelief=tk.RAISED)
        rpanes.pack(fill="both", expand=True)
        mag = tk.Frame(rpanes)
        tabs = tk.Frame(rpanes)
        rpanes.add(mag, minsize=200, stretch="never")
        rpanes.add(tabs, minsize=200, stretch="always")
        mag_inner = self._scroll_area(mag)
        self._build_magnifier(mag_inner)
        mag_inner.update_idletasks()
        rpanes.paneconfigure(mag, height=mag_inner.winfo_reqheight() + 6)
        nb = ttk.Notebook(tabs)
        nb.pack(fill="both", expand=True)
        nb.enable_traversal()                        # Ctrl+Tab / Ctrl+Shift+Tab switch tabs
        for title, builder in (("Points", self._build_points_tab),
                               ("Analysis", self._build_analysis_tab),
                               ("Display", self._build_display_tab)):
            page = tk.Frame(nb)
            nb.add(page, text=title)
            # Each tab scrolls on its own (the notebook is as tall as its
            # TALLEST tab, so one scroll area around it made short tabs scroll).
            tab = tk.Frame(self._scroll_area(page), padx=6, pady=6)
            tab.pack(fill="both", expand=True)
            builder(tab)
        self._sync_text_from_points()
        self._show_result_text()
        if win is not None:
            self._bind_shortcuts(win)
        self._request_render()
        self._request_loupe()

    def close_inspector(self) -> None:
        self._close_help()
        super().close_inspector()
        self._reset_widget_refs()

    # ── image view ────────────────────────────────────────────────────
    def _build_view(self, parent) -> None:
        bar = tk.Frame(parent)
        bar.pack(fill="x", pady=(0, 3))
        # Underlined letter = Alt+letter shortcut (see Help / Keys).
        tk.Checkbutton(bar, text="Freeze", underline=2, variable=self._freeze_var,
                       font=("Arial", 9, "bold")).pack(side="left")
        tk.Button(bar, text="Fit", underline=2, width=4, command=self._fit_view).pack(side="left", padx=(8, 0))
        tk.Button(bar, text="1:1 (O)", width=6, command=self._one_to_one).pack(side="left", padx=(2, 0))
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=8)
        tk.Label(bar, text="Mode:").pack(side="left")
        for value, label, ul in (("pan", "Pan", 0), ("pick", "Pick points", 1)):
            tk.Radiobutton(bar, text=label, underline=ul, value=value, variable=self._mode_var,
                           command=self._on_mode_changed).pack(side="left")
        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=8)
        tk.Checkbutton(bar, text="Grid", underline=0, variable=self._show_grid_var).pack(side="left")
        tk.Label(bar, text="Interp. (L)").pack(side="left", padx=(8, 2))
        ttk.Combobox(bar, textvariable=self._interp_var, values=tuple(self._INTERPOLATIONS),
                     state="readonly", width=9).pack(side="left")
        tk.Button(bar, text="Help (Ctrl+H)", command=self._open_help).pack(side="right")

        c = tk.Canvas(parent, bg="#1b1b24", highlightthickness=1, highlightbackground="#33384a",
                      cursor="fleur")
        c.pack(fill="both", expand=True)
        self._c_view = c
        c.bind("<Configure>", lambda _e: self._request_render())
        c.bind("<Enter>", lambda _e: c.focus_set())
        c.bind("<Motion>", self._on_motion)
        c.bind("<Leave>", self._on_leave)
        c.bind("<ButtonPress-1>", self._on_press)
        c.bind("<B1-Motion>", self._on_drag)
        c.bind("<ButtonRelease-1>", self._on_release)
        c.bind("<Double-Button-1>", self._on_double_click)
        c.bind("<ButtonPress-3>", self._on_right_click)
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            c.bind(seq, self._on_wheel)
        for seq, fn in (("<KeyPress-plus>", lambda e: self._zoom_at_cursor(self._ZOOM_STEP)),
                        ("<KeyPress-KP_Add>", lambda e: self._zoom_at_cursor(self._ZOOM_STEP)),
                        ("<KeyPress-minus>", lambda e: self._zoom_at_cursor(1 / self._ZOOM_STEP)),
                        ("<KeyPress-KP_Subtract>", lambda e: self._zoom_at_cursor(1 / self._ZOOM_STEP)),
                        ("<KeyPress-c>", self._on_key_c), ("<KeyPress-C>", self._on_key_c),
                        ("<KeyPress-s>", self._on_key_s), ("<KeyPress-S>", self._on_key_s),
                        ("<Control-z>", lambda e: self._undo_last()), ("<Control-Z>", lambda e: self._undo_last())):
            c.bind(seq, fn)
        self._bind_probe_keys(c)

        self._status_bar_var = tk.StringVar(value="")
        tk.Label(parent, textvariable=self._status_bar_var, anchor="w", relief=tk.SUNKEN,
                 font=("Consolas", 9), padx=4).pack(fill="x", pady=(3, 0))
        self._on_mode_changed()

    def _on_mode_changed(self) -> None:
        if self._c_view is not None:
            self._c_view.configure(cursor="crosshair" if self._mode_var.get() == "pick" else "fleur")

    def _display_source(self):
        """Image the view shows: the analysis result image, or the shown frame."""
        return self._display_source_keyed()[1]

    def _display_source_keyed(self):
        """(cache key, image). Keys use counters, not id(): a freed frame's id
        can be reused by the next one, which would show a stale image."""
        res = self._result
        if (res is not None and res.get("image") is not None and self._show_result_image_var.get()):
            return ("result", self._result_serial), res["image"]
        return ("frame", self._frame_serial), self._shown

    def _fit_view(self) -> None:
        self._user_view = False
        self._view = None
        self._request_render()

    def _one_to_one(self) -> None:
        c, img = self._c_view, self._display_source()
        if c is None or img is None:
            return
        cw, ch = c.winfo_width(), c.winfo_height()
        h, w = img.shape[:2]
        if self._view is not None:                       # keep the centre pixel
            ox, oy, s = self._view
            cx_img, cy_img = (cw / 2 - ox) / s - 0.5, (ch / 2 - oy) / s - 0.5
        else:
            cx_img, cy_img = (w - 1) / 2, (h - 1) / 2
        self._view = (cw / 2 - (cx_img + 0.5), ch / 2 - (cy_img + 0.5), 1.0)
        self._user_view = True
        self._request_render()

    def _request_render(self) -> None:
        """Coalesce redraws: at most one pending render (frames can arrive
        faster than they can be drawn)."""
        c = self._c_view
        if c is None or self._render_pending:
            return
        self._render_pending = True
        try:
            c.after_idle(self._render)
        except tk.TclError:
            self._render_pending = False

    @staticmethod
    def _to_display_rgb(img: np.ndarray) -> np.ndarray:
        """uint8 RGB version of any frame: grey -> RGB, alpha dropped, other
        dtypes scaled from their min..max to 0..255."""
        a = img
        if a.dtype != np.uint8:
            a = a.astype(np.float32)
            finite = np.isfinite(a)
            lo, hi = (float(a[finite].min()), float(a[finite].max())) if finite.any() else (0.0, 1.0)
            a = np.clip((np.nan_to_num(a, nan=lo) - lo) * (255.0 / max(hi - lo, 1e-12)), 0, 255).astype(np.uint8)
        if a.ndim == 2:
            a = np.stack([a, a, a], axis=-1)
        elif a.shape[2] == 1:
            a = np.repeat(a, 3, axis=2)
        elif a.shape[2] > 3:
            a = a[:, :, :3]
        return np.ascontiguousarray(a)

    def _render(self) -> None:
        self._render_pending = False
        c = self._c_view
        if c is None:
            return
        try:
            cw, ch = c.winfo_width(), c.winfo_height()
        except tk.TclError:
            return
        if cw < 20 or ch < 20:
            return
        src_key, src = self._display_source_keyed()
        if src is None:
            c.delete("all")
            self._photo_key = None
            c.create_text(cw / 2, ch / 2, fill="#8a92a6", font=("Arial", 11), justify="center",
                          text="Waiting for frames on the 'image' pin ...")
            self._update_status_bar()
            return
        if self._display_cache[0] != src_key:
            self._display_cache = (src_key, self._to_display_rgb(src))
        rgb = self._display_cache[1]
        h, w = rgb.shape[:2]
        key = ((w, h), (cw, ch))
        if self._view is None or (not self._user_view and key != self._view_key):
            s = 0.98 * min(cw / w, ch / h)
            self._view = ((cw - w * s) / 2.0, (ch - h * s) / 2.0, s)
            self._view_key = key
        ox, oy, s = self._view

        # Crop to the visible pixels, then scale only that part.
        x0 = max(0, int(np.floor(-ox / s)))
        y0 = max(0, int(np.floor(-oy / s)))
        x1 = min(w, int(np.ceil((cw - ox) / s)))
        y1 = min(h, int(np.ceil((ch - oy) / s)))
        c.delete("all")
        if x1 > x0 and y1 > y0:
            dw = max(1, int(round((x1 - x0) * s)))
            dh = max(1, int(round((y1 - y0) * s)))
            pkey = (src_key, x0, y0, x1, y1, dw, dh, self._interp_var.get())
            if pkey != self._photo_key:
                interp = (self._INTERPOLATIONS.get(self._interp_var.get(), cv2.INTER_NEAREST)
                          if s >= 1.0 else cv2.INTER_AREA)
                disp = cv2.resize(rgb[y0:y1, x0:x1], (dw, dh), interpolation=interp)
                self._photo = ImageTk.PhotoImage(Image.fromarray(disp), master=c)
                self._photo_key = pkey
            c.create_image(ox + x0 * s, oy + y0 * s, image=self._photo, anchor="nw")
        self._draw_overlays(c, w, h, ox, oy, s, (x0, y0, x1, y1))
        self._draw_probe_marker(c)
        self._update_status_bar()
        self._request_loupe()

    def _to_canvas(self, x, y):
        ox, oy, s = self._view
        return ox + (x + 0.5) * s, oy + (y + 0.5) * s

    def _to_image(self, X, Y):
        ox, oy, s = self._view
        return (X - ox) / s - 0.5, (Y - oy) / s - 0.5

    def _draw_overlays(self, c, w, h, ox, oy, s, visible) -> None:
        x0, y0, x1, y1 = visible
        # Pixel grid (lines on pixel EDGES).
        if self._show_grid_var.get() and s >= self._GRID_MIN_SCALE and x1 > x0 and y1 > y0:
            for k in range(x0, min(x1, x0 + self._MAX_GRID_LINES) + 1):
                X = ox + k * s
                c.create_line(X, oy + y0 * s, X, oy + y1 * s, fill="#ffffff", stipple="gray50")
            for k in range(y0, min(y1, y0 + self._MAX_GRID_LINES) + 1):
                Y = oy + k * s
                c.create_line(ox + x0 * s, Y, ox + x1 * s, Y, fill="#ffffff", stipple="gray50")

        # Analysis result.
        res = self._result
        if res is not None and self._show_result_var.get():
            for x1_, y1_, x2_, y2_ in res.get("lines", ()):
                c.create_line(*self._to_canvas(x1_, y1_), *self._to_canvas(x2_, y2_),
                              fill=self._RESULT_LINE_COLOR, width=2)
            pts = res.get("points")
            if res.get("aruco_ids") is not None and pts is not None and len(pts):
                for m, mid in enumerate(res["aruco_ids"]):
                    quad = [self._to_canvas(*q) for q in pts[5 * m:5 * m + 4]]
                    c.create_polygon(*[v for q in quad for v in q], outline=self._RESULT_LINE_COLOR,
                                     fill="", width=2)
                    c.create_rectangle(quad[0][0] - 4, quad[0][1] - 4, quad[0][0] + 4, quad[0][1] + 4,
                                       outline=self._RESULT_LINE_COLOR, width=2)    # corner 0
                    cx, cy = self._to_canvas(*pts[5 * m + 4])
                    c.create_text(cx + 6, cy - 6, text=str(mid), anchor="sw", fill=self._RESULT_LINE_COLOR,
                                  font=("Arial", 10, "bold"))
            if pts is not None and len(pts):
                if res.get("ordered"):                   # chessboard: one line per corner row
                    cols = int(res.get("cols") or len(pts))
                    for r0 in range(0, len(pts), cols):
                        row = pts[r0:r0 + cols]
                        if len(row) > 1:
                            c.create_line(*[v for p in row for v in self._to_canvas(*p)],
                                          fill=self._RESULT_POINT_COLOR, width=1)
                for i, (px, py) in enumerate(pts):
                    X, Y = self._to_canvas(px, py)
                    r = 6 if (res.get("ordered") and i == 0) else 4
                    c.create_line(X - r, Y, X + r, Y, fill=self._RESULT_POINT_COLOR, width=2)
                    c.create_line(X, Y - r, X, Y + r, fill=self._RESULT_POINT_COLOR, width=2)

        # Input 'points' overlay.
        if self._show_input_points_var.get() and self._input_points.size:
            color, width, marker = self._marker_color(), self._marker_width(), self._marker_var.get()
            for p in self._input_points:
                X, Y = self._to_canvas(float(p[0]), float(p[1]))
                if p.shape[0] >= 3:
                    ww = abs(float(p[2])) * s
                    hh = abs(float(p[3])) * s if p.shape[0] == 4 else ww
                    c.create_rectangle(X - ww / 2, Y - hh / 2, X + ww / 2, Y + hh / 2,
                                       outline=color, width=width)
                r = 5
                if marker == "o":
                    c.create_oval(X - r, Y - r, X + r, Y + r, outline=color, width=width)
                else:
                    segs = [(-r, 0, r, 0), (0, -r, 0, r)]
                    if marker == "*":
                        d = r * 0.72
                        segs += [(-d, -d, d, d), (-d, d, d, -d)]
                    for a, b, cc, dd in segs:
                        c.create_line(X + a, Y + b, X + cc, Y + dd, fill=color, width=width)

        # Picked points (numbered in list order).
        if self._show_picked_var.get():
            numbers = self._show_numbers_var.get()
            for i, (px, py) in enumerate(self._picked, start=1):
                X, Y = self._to_canvas(px, py)
                c.create_oval(X - 5, Y - 5, X + 5, Y + 5, outline=self._PICK_COLOR, width=2)
                c.create_line(X - 2, Y, X + 2, Y, fill=self._PICK_COLOR)
                c.create_line(X, Y - 2, X, Y + 2, fill=self._PICK_COLOR)
                if numbers:
                    c.create_text(X + 8, Y - 8, text=str(i), anchor="sw", fill=self._PICK_COLOR,
                                  font=("Arial", 9, "bold"))

    def _marker_color(self) -> str:
        try:
            vals = (self._marker_red_var.get(), self._marker_green_var.get(), self._marker_blue_var.get())
            r, g, b = (max(0, min(255, int(v))) for v in vals)
        except (tk.TclError, ValueError):
            r, g, b = 0, 255, 0
        return f"#{r:02x}{g:02x}{b:02x}"

    def _marker_width(self) -> int:
        try:
            return max(1, min(20, int(self._marker_width_var.get())))
        except (tk.TclError, ValueError):
            return 2

    def _update_status_bar(self) -> None:
        var = self._status_bar_var
        if var is None:
            return
        parts = []
        src = self._display_source()
        if self._show_coords_var.get() and self._probe is not None and src is not None:
            x, y = self._probe
            val = self._pixel_value_text(src, x, y)
            where = "" if self._cursor_xy is not None else "  (probe)"
            parts.append(f"x={x:9.3f}  y={y:9.3f}  value=({val}){where}" if val is not None
                         else "cursor outside the image")
        if self._shown is not None:
            h, w = self._shown.shape[:2]
            parts.append(f"{w} x {h} {self._shown.dtype}")
        if self._view is not None:
            parts.append(f"zoom {self._view[2] * 100:.0f}%")
        if self._show_fps_var.get():
            parts.append(f"{self._fps():.1f} fps")
        if self._is_frozen():
            parts.append("FROZEN")
        var.set("   |   ".join(parts))

    # ── mouse / keys on the view ──────────────────────────────────────
    def _on_motion(self, event) -> None:
        self._cursor_xy = (event.x, event.y)
        if self._view is not None:
            self._set_probe(*self._to_image(event.x, event.y), from_mouse=True)
        self._update_status_bar()

    def _on_leave(self, _event) -> None:
        # The probe (and so the magnifier) STAYS where the cursor was, so you
        # can move the mouse to other widgets while looking at the magnifier.
        self._cursor_xy = None
        self._update_status_bar()
        self._request_render()

    def _nearest_picked(self, X, Y, radius):
        best, best_d = None, radius * radius
        for i, (px, py) in enumerate(self._picked):
            cx, cy = self._to_canvas(px, py)
            d = (cx - X) ** 2 + (cy - Y) ** 2
            if d <= best_d:
                best, best_d = i, d
        return best

    def _on_press(self, event) -> None:
        if self._view is None:
            return
        self._c_view.focus_set()
        grab = None
        if self._mode_var.get() == "pick" and self._show_picked_var.get():
            grab = self._nearest_picked(event.x, event.y, self._GRAB_PX)
        self._drag = {"x": event.x, "y": event.y, "view": self._view, "moved": False, "grab": grab}

    def _on_drag(self, event) -> None:
        d = self._drag
        if d is None or self._view is None:
            return
        dx, dy = event.x - d["x"], event.y - d["y"]
        if not d["moved"] and abs(dx) + abs(dy) < self._CLICK_PX:
            return
        d["moved"] = True
        if d["grab"] is not None:                    # move a picked point
            if not self._check_no_unsent_edits():
                d["grab"] = None
                return
            self._picked[d["grab"]] = self._to_image(event.x, event.y)
        else:                                        # pan (fixed offset from the press)
            ox, oy, s = d["view"]
            self._view = (ox + dx, oy + dy, s)
            self._user_view = True
        self._cursor_xy = (event.x, event.y)
        self._set_probe(*self._to_image(event.x, event.y), from_mouse=True)
        self._request_render()

    def _on_release(self, event) -> None:
        d, self._drag = self._drag, None
        if d is None:
            return
        if d["moved"] and d["grab"] is not None:
            self._picked_changed()
        elif not d["moved"] and self._mode_var.get() == "pick":
            self._add_point(*self._to_image(event.x, event.y))

    def _on_double_click(self, _event) -> str:
        if self._mode_var.get() == "pan":
            self._fit_view()
        return "break"

    def _on_wheel(self, event) -> str:
        if getattr(event, "num", None) == 4 or getattr(event, "delta", 0) > 0:
            factor = self._ZOOM_STEP
        else:
            factor = 1 / self._ZOOM_STEP
        self._zoom_at(event.x, event.y, factor)
        return "break"

    def _zoom_at(self, X, Y, factor) -> None:
        if self._view is None:
            return
        ox, oy, s = self._view
        new_s = float(min(max(s * factor, self._MIN_SCALE), self._MAX_SCALE))
        k = new_s / s
        self._view = (X - (X - ox) * k, Y - (Y - oy) * k, new_s)
        self._user_view = True
        self._request_render()

    def _zoom_at_cursor(self, factor) -> str:
        c = self._c_view
        if c is not None:
            X, Y = self._cursor_xy or (c.winfo_width() / 2, c.winfo_height() / 2)
            self._zoom_at(X, Y, factor)
        return "break"

    def _on_key_c(self, _event=None) -> str:
        """C: add the probe point (cursor / magnifier centre) and copy 'x, y'."""
        if self._probe is None or self._shown is None:
            return "break"
        x, y = self._probe
        if self._add_point(x, y):
            self._copy_to_clipboard(f"{x:.3f}, {y:.3f}")
        return "break"

    def _on_key_s(self, _event=None) -> str:
        """S: add the detection nearest to the magnifier's cross (snap)."""
        near = self._loupe_nearest
        if near is None:
            if self._loupe_nearest_var is not None:
                self._loupe_nearest_var.set("S: nothing detected to snap to -- tick Corners / ArUco / "
                                            "Features first.")
            return "break"
        _kind, x, y = near
        if self._add_point(x, y):
            self._copy_to_clipboard(f"{x:.3f}, {y:.3f}")
        return "break"

    def _on_right_click(self, event) -> None:
        if self._view is None:
            return
        x, y = self._to_image(event.x, event.y)
        near = self._nearest_picked(event.x, event.y, 10)
        m = tk.Menu(self._c_view, tearoff=0)
        m.add_command(label=f"Add point here  ({x:.2f}, {y:.2f})", command=lambda: self._add_point(x, y))
        m.add_command(label=f"Remove point {near + 1}" if near is not None else "Remove point (none near)",
                      state=tk.NORMAL if near is not None else tk.DISABLED,
                      command=lambda: self._remove_point(near))
        m.add_command(label="Copy coordinates", command=lambda: self._copy_to_clipboard(f"{x:.3f}, {y:.3f}"))
        m.add_separator()
        m.add_command(label="Fit view", command=self._fit_view)
        m.add_command(label="1:1 (one image pixel per screen pixel)", command=self._one_to_one)
        sub = tk.Menu(m, tearoff=0)
        for name in self._INTERPOLATIONS:
            sub.add_radiobutton(label=name, value=name, variable=self._interp_var)
        m.add_cascade(label="Interpolation", menu=sub)
        m.tk_popup(event.x_root, event.y_root)
        m.grab_release()

    def _copy_to_clipboard(self, text: str) -> None:
        try:
            self.canvas.clipboard_clear()
            self.canvas.clipboard_append(text)
        except tk.TclError:
            pass

    # ── vertically scrollable panel ───────────────────────────────────
    _NO_WHEEL_SCROLL = ("Text", "Listbox", "TCombobox", "Scrollbar", "TScrollbar")

    def _scroll_area(self, parent) -> tk.Frame:
        """A frame that fills `parent` and gets a vertical scroll bar when its
        content is taller than the space (when there is room, the content
        still stretches to fill it). The mouse wheel scrolls it, except over
        text boxes / lists / the magnifier, which use the wheel themselves."""
        outer = tk.Frame(parent)
        outer.pack(fill="both", expand=True)
        cv = tk.Canvas(outer, highlightthickness=0, borderwidth=0, width=20, height=20)
        sb = ttk.Scrollbar(outer, orient="vertical", command=cv.yview)
        cv.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        cv.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(cv)
        item = cv.create_window(0, 0, window=inner, anchor="nw")
        state = {"size": None, "bound": set()}

        def wheel(event):
            if inner.winfo_reqheight() > cv.winfo_height() + 1:
                cv.yview_scroll(-1 if event.delta > 0 else 1, "units")
            return "break"

        def bind_wheel(w):
            for child in w.winfo_children():
                name = str(child)
                if name not in state["bound"]:
                    state["bound"].add(name)
                    if (child.winfo_class() not in self._NO_WHEEL_SCROLL and child is not self._c_loupe):
                        child.bind("<MouseWheel>", wheel, add="+")
                bind_wheel(child)

        def layout(_e=None):
            try:
                w, h, need = cv.winfo_width(), cv.winfo_height(), inner.winfo_reqheight()
            except tk.TclError:
                return
            size = (w, max(h, need))
            if size != state["size"]:
                state["size"] = size
                cv.itemconfigure(item, width=w, height=size[1])
                cv.configure(scrollregion=(0, 0, w, size[1]))
            if need <= h:
                cv.yview_moveto(0)

        def poll():                                  # content height changes (tabs, tool fields)
            try:
                if not cv.winfo_exists():
                    return
            except tk.TclError:
                return
            layout()
            bind_wheel(inner)
            cv.after(400, poll)

        cv.bind("<Configure>", layout)
        cv.bind("<MouseWheel>", wheel)
        inner.bind("<MouseWheel>", wheel)
        cv.after(50, poll)
        return inner

    # ── magnifier ─────────────────────────────────────────────────────
    def _build_magnifier(self, parent) -> None:
        box = tk.LabelFrame(parent, text="Magnifier  (follows the cursor; stays when the mouse leaves)",
                            padx=4, pady=3)
        box.pack(fill="both", expand=True)
        top = tk.Frame(box)
        top.pack(fill="x")
        L = self._LOUPE_SIZE
        c = tk.Canvas(top, width=L, height=L, bg="#15151d", highlightthickness=1,
                      highlightbackground="#33384a", cursor="crosshair")
        c.pack(side="left")
        c.bind("<Enter>", lambda _e: c.focus_set())
        c.bind("<KeyPress-c>", self._on_key_c)
        c.bind("<KeyPress-C>", self._on_key_c)
        c.bind("<KeyPress-s>", self._on_key_s)
        c.bind("<KeyPress-S>", self._on_key_s)
        self._bind_probe_keys(c)
        c.bind("<MouseWheel>", self._on_loupe_wheel)
        self._c_loupe = c

        side = tk.Frame(top, padx=6)
        side.pack(side="left", fill="both", expand=True)
        self._loupe_zoom_label_var = tk.StringVar(value="")
        tk.Label(side, textvariable=self._loupe_zoom_label_var, font=("Arial", 8, "bold"),
                 anchor="w", justify="left").pack(fill="x")
        zr = tk.Frame(side)
        zr.pack(fill="x")
        for i, z in enumerate(self._LOUPE_ZOOMS):
            tk.Radiobutton(zr, text=f"{z}X", value=z, variable=self._loupe_zoom_var, font=("Arial", 8),
                           padx=0, pady=0).grid(row=i // 4, column=i % 4, sticky="w")
        tk.Label(side, text="Alt+0..6 = 1X..64X, or the wheel here", font=("Arial", 7), fg="#666666",
                 anchor="w").pack(fill="x")
        self._loupe_info_var = tk.StringVar(value="")
        tk.Label(side, textvariable=self._loupe_info_var, font=("Consolas", 8), anchor="w",
                 justify="left").pack(fill="x", pady=(4, 0))
        self._loupe_nearest_var = tk.StringVar(value="")
        tk.Label(side, textvariable=self._loupe_nearest_var, font=("Arial", 8), anchor="w",
                 justify="left", wraplength=190, fg="#7a1f63").pack(fill="x", pady=(4, 0))

        # Detectors (run on the magnifier area every time it changes).
        det = tk.Frame(box)
        det.pack(fill="x", pady=(4, 0))
        tk.Label(det, text="Detect:", font=("Arial", 8, "bold")).grid(row=0, column=0, sticky="w")
        tk.Checkbutton(det, text="Corners", underline=0, variable=self._det_corners_var,
                       fg="#00799a").grid(row=0, column=1, sticky="w")
        tk.Checkbutton(det, text="ArUco", underline=0, variable=self._det_aruco_var,
                       fg="#8a6d00").grid(row=0, column=2, sticky="w")
        tk.Checkbutton(det, text="Features", underline=0, variable=self._det_features_var,
                       fg="#3b7d00").grid(row=0, column=3, sticky="w")
        opts = tk.Frame(box)
        opts.pack(fill="x")
        r1 = tk.Frame(opts)
        r1.pack(fill="x")
        tk.Label(r1, text="Corners (Harris):", font=("Arial", 8), width=15, anchor="w").pack(side="left")
        tk.Label(r1, text="threshold", font=("Arial", 8)).pack(side="left")
        tk.Entry(r1, textvariable=self._corner_thresh_var, width=5, font=("Arial", 8)).pack(side="left", padx=(2, 6))
        tk.Checkbutton(r1, text="sub-pixel", variable=self._corner_subpix_var, font=("Arial", 8)).pack(side="left")
        r2 = tk.Frame(opts)
        r2.pack(fill="x")
        tk.Label(r2, text="ArUco:", font=("Arial", 8), width=15, anchor="w").pack(side="left")
        ttk.Combobox(r2, textvariable=self._aruco_dict_var, values=self._ARUCO_DICTS, state="readonly",
                     width=19, font=("Arial", 8)).pack(side="left", padx=(0, 6))
        tk.Checkbutton(r2, text="sub-pixel", variable=self._aruco_refine_var, font=("Arial", 8)).pack(side="left")
        r3 = tk.Frame(opts)
        r3.pack(fill="x")
        tk.Label(r3, text="Features (GFTT):", font=("Arial", 8), width=15, anchor="w").pack(side="left")
        tk.Label(r3, text="max", font=("Arial", 8)).pack(side="left")
        tk.Entry(r3, textvariable=self._feat_max_var, width=4, font=("Arial", 8)).pack(side="left", padx=(2, 6))
        tk.Label(r3, text="quality", font=("Arial", 8)).pack(side="left")
        tk.Entry(r3, textvariable=self._feat_quality_var, width=5, font=("Arial", 8)).pack(side="left", padx=(2, 6))
        tk.Checkbutton(r3, text="sub-pixel", variable=self._feat_subpix_var, font=("Arial", 8)).pack(side="left")
        tk.Label(box, font=("Arial", 7), fg="#555555", anchor="w", justify="left", wraplength=400, text=(
            "Arrows: move cross (Shift: 0.1 px)   C: add cross point   S: add pink-circled   Ctrl+H: all keys"
            )).pack(
            fill="x", pady=(3, 0))

    def _set_probe(self, x: float, y: float, from_mouse: bool = False) -> None:
        self._probe = (float(x), float(y))
        self._request_loupe()
        if not from_mouse:
            self._update_status_bar()
            self._request_render()                   # move the marker on the main view

    def _on_loupe_wheel(self, event) -> str:
        zs = self._LOUPE_ZOOMS
        cur = self._loupe_zoom_var.get()
        i = min(range(len(zs)), key=lambda k: abs(zs[k] - cur))
        i = min(len(zs) - 1, i + 1) if event.delta > 0 else max(0, i - 1)
        self._loupe_zoom_var.set(zs[i])
        return "break"

    def _bind_probe_keys(self, widget) -> None:
        # A plain <KeyPress-c> binding ALSO catches Alt+C / Ctrl+C (Tk matches
        # extra modifiers) and its "break" stopped the window's Alt+C
        # (Corners). These more specific do-nothing bindings win at this
        # widget and let the event go on to the window bindings.
        for letter in "cCsS":
            for mod in ("Alt", "Control"):
                widget.bind(f"<{mod}-KeyPress-{letter}>", lambda _e: None)
        for key, dx, dy in (("Left", -1, 0), ("Right", 1, 0), ("Up", 0, -1), ("Down", 0, 1)):
            widget.bind(f"<KeyPress-{key}>", lambda e, a=dx, b=dy: self._nudge_probe(a, b))
            widget.bind(f"<Shift-KeyPress-{key}>", lambda e, a=dx, b=dy: self._nudge_probe(a * 0.1, b * 0.1))

    def _nudge_probe(self, dx: float, dy: float) -> str:
        if self._probe is None:
            src = self._display_source()
            if src is None:
                return "break"
            h, w = src.shape[:2]
            self._probe = ((w - 1) / 2.0, (h - 1) / 2.0)
        self._set_probe(self._probe[0] + dx, self._probe[1] + dy)
        return "break"

    @staticmethod
    def _pixel_value_text(src, x, y):
        h, w = src.shape[:2]
        if not (-0.5 <= x < w - 0.5 and -0.5 <= y < h - 0.5):
            return None
        value = src[int(round(y)), int(round(x))]
        return (", ".join(f"{v:g}" for v in np.ravel(value)) if np.ndim(value) else f"{float(value):g}")

    def _draw_probe_marker(self, c) -> None:
        """Main view: outline of the magnifier area around the probe."""
        if self._probe is None or self._view is None or self._c_loupe is None:
            return
        z = max(1, int(self._loupe_zoom_var.get() or 16))
        half = self._LOUPE_SIZE / (2.0 * z)
        px, py = self._probe
        ox, oy, s = self._view
        X0, Y0 = ox + (px + 0.5 - half) * s, oy + (py + 0.5 - half) * s
        X1, Y1 = ox + (px + 0.5 + half) * s, oy + (py + 0.5 + half) * s
        c.create_rectangle(X0, Y0, X1, Y1, outline="#ff8c00", dash=(3, 2))
        if self._cursor_xy is None:                  # probe moved by keys / mouse elsewhere
            X, Y = ox + (px + 0.5) * s, oy + (py + 0.5) * s
            c.create_line(X - 7, Y, X + 7, Y, fill="#ff8c00")
            c.create_line(X, Y - 7, X, Y + 7, fill="#ff8c00")

    def _request_loupe(self) -> None:
        c = self._c_loupe
        if c is None or self._loupe_pending:
            return
        self._loupe_pending = True
        try:
            c.after_idle(self._render_loupe)
        except tk.TclError:
            self._loupe_pending = False

    def _render_loupe(self) -> None:
        """Draw the zoomed area around the probe. The cross is FIXED at the
        centre; the zoomed pixels move under it (sub-pixel accurate)."""
        self._loupe_pending = False
        c = self._c_loupe
        if c is None:
            return
        L = self._LOUPE_SIZE
        try:
            z = max(1, int(self._loupe_zoom_var.get()))
        except (tk.TclError, ValueError):
            z = 16
        if self._loupe_zoom_label_var is not None:
            self._loupe_zoom_label_var.set(f"Zoom {z}X\n1 image px = {z}x{z} screen px")
        c.delete("all")
        src_key, src = self._display_source_keyed()
        if src is None or self._probe is None:
            c.create_text(L / 2, L / 2, fill="#8a92a6", justify="center", width=L - 20,
                          text="Move the mouse over the image")
            self._loupe_nearest = None
            if self._loupe_info_var is not None:
                self._loupe_info_var.set("")
                self._loupe_nearest_var.set("")
            return
        if self._display_cache[0] != src_key:
            self._display_cache = (src_key, self._to_display_rgb(src))
        rgb = self._display_cache[1]
        px, py = self._probe
        # Destination pixel u (centre at u + 0.5) samples image x = px + (u + 0.5 - L/2) / z,
        # with image pixel centres at integer x (OpenCV convention).
        M = np.array([[1.0 / z, 0.0, px + (0.5 - L / 2.0) / z],
                      [0.0, 1.0 / z, py + (0.5 - L / 2.0) / z]], dtype=np.float64)
        interp = self._INTERPOLATIONS.get(self._interp_var.get(), cv2.INTER_NEAREST)
        if interp == cv2.INTER_AREA:
            interp = cv2.INTER_LINEAR                 # warpAffine has no AREA mode
        patch = cv2.warpAffine(rgb, M, (L, L), flags=interp | cv2.WARP_INVERSE_MAP,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(21, 21, 29))
        self._loupe_photo = ImageTk.PhotoImage(Image.fromarray(patch), master=c)
        c.create_image(0, 0, image=self._loupe_photo, anchor="nw")

        def to_l(x, y):                              # image px -> magnifier canvas
            return L / 2.0 + (x - px) * z, L / 2.0 + (y - py) * z

        dets, err = self._loupe_detect(src, px, py, z)
        nearest, best = None, None
        for kind, color, pts in (("corner", self._LOUPE_CORNER_COLOR, dets.get("corners")),
                                 ("feature", self._LOUPE_FEATURE_COLOR, dets.get("features"))):
            for x, y in (pts if pts is not None else ()):
                X, Y = to_l(x, y)
                r = 5 if kind == "corner" else 4
                if kind == "corner":
                    c.create_oval(X - r, Y - r, X + r, Y + r, outline=color, width=2)
                else:
                    c.create_line(X - r, Y, X + r, Y, fill=color, width=2)
                    c.create_line(X, Y - r, X, Y + r, fill=color, width=2)
                d = (x - px) ** 2 + (y - py) ** 2
                if best is None or d < best:
                    best, nearest = d, (kind, float(x), float(y))
        for mid, quad in dets.get("markers", ()):
            pts = [to_l(x, y) for x, y in quad]
            c.create_polygon(*[v for p in pts for v in p], outline=self._LOUPE_ARUCO_COLOR, fill="", width=2)
            cx = sum(p[0] for p in pts) / 4.0
            cy = sum(p[1] for p in pts) / 4.0
            c.create_text(cx, cy, text=str(mid), fill=self._LOUPE_ARUCO_COLOR, font=("Arial", 10, "bold"))
            for k, (x, y) in enumerate(quad):
                X, Y = to_l(x, y)
                c.create_rectangle(X - 3, Y - 3, X + 3, Y + 3, outline=self._LOUPE_ARUCO_COLOR,
                                   width=2 if k == 0 else 1)
                d = (x - px) ** 2 + (y - py) ** 2
                if best is None or d < best:
                    best, nearest = d, (f"ArUco id {mid} corner {k}", float(x), float(y))
        if nearest is not None:                      # highlight the snap target
            X, Y = to_l(nearest[1], nearest[2])
            c.create_oval(X - 9, Y - 9, X + 9, Y + 9, outline=self._LOUPE_NEAREST_COLOR, width=2)
        self._loupe_nearest = nearest

        # Picked (output) points in view.
        for i, (x, y) in enumerate(self._picked, start=1):
            X, Y = to_l(x, y)
            if -10 <= X <= L + 10 and -10 <= Y <= L + 10:
                c.create_oval(X - 6, Y - 6, X + 6, Y + 6, outline=self._PICK_COLOR, width=1)
                c.create_text(X + 8, Y - 8, text=str(i), anchor="sw", fill=self._PICK_COLOR,
                              font=("Arial", 8, "bold"))

        # The fixed thin cross (with a small gap so the centre pixel stays visible).
        m, g = L / 2.0, 4
        for x0, y0, x1, y1 in ((0, m, m - g, m), (m + g, m, L, m), (m, 0, m, m - g), (m, m + g, m, L)):
            c.create_line(x0, y0, x1, y1, fill="#000000", width=3)
            c.create_line(x0, y0, x1, y1, fill="#ff3b3b", width=1)

        val = self._pixel_value_text(src, px, py)
        if self._loupe_info_var is not None:
            self._loupe_info_var.set(f"x = {px:10.3f}\ny = {py:10.3f}\nvalue = ({val})" if val is not None
                                     else f"x = {px:10.3f}\ny = {py:10.3f}\n(outside the image)")
            if err:
                text = err
            elif nearest is not None:
                d = best ** 0.5
                text = (f"Nearest {nearest[0]}:\n({nearest[1]:.3f}, {nearest[2]:.3f})  "
                        f"d = {d:.2f} px\nS adds it to the output points")
            elif self._det_corners_var.get() or self._det_aruco_var.get() or self._det_features_var.get():
                text = "Nothing detected in the magnifier area."
            else:
                text = "Tick Corners / ArUco / Features to detect in this area."
            self._loupe_nearest_var.set(text)

    def _loupe_detect(self, src, px, py, z):
        """Run the ticked detectors on the image around the probe.
        Corners / features: the magnifier area (+ a small margin). ArUco: the
        whole frame once (cached), so a marker only partly inside the
        magnifier is still found. Returns ({corners, features, markers}, error text)."""
        dets, errors = {}, []
        if not (self._det_corners_var.get() or self._det_aruco_var.get() or self._det_features_var.get()):
            return dets, ""
        h, w = src.shape[:2]

        def region(half):
            x0, y0 = max(0, int(np.floor(px - half))), max(0, int(np.floor(py - half)))
            x1, y1 = min(w, int(np.ceil(px + half)) + 1), min(h, int(np.ceil(py + half)) + 1)
            return x0, y0, x1, y1

        half = self._LOUPE_SIZE / (2.0 * z) + 6
        x0, y0, x1, y1 = region(half)
        if x1 - x0 >= 8 and y1 - y0 >= 8:
            gray = self._gray_u8(src[y0:y1, x0:x1])
            origin = np.array([x0, y0], dtype=np.float64)
            if self._det_corners_var.get():
                try:
                    thr = float(self._corner_thresh_var.get())
                    resp = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
                    peak = float(resp.max())
                    pts = np.empty((0, 2))
                    if peak > 0:
                        local = resp == cv2.dilate(resp, np.ones((3, 3), np.uint8))
                        ys, xs = np.nonzero(local & (resp > thr * peak))
                        order = np.argsort(-resp[ys, xs])[:200]
                        pts = np.stack([xs[order], ys[order]], axis=1).astype(np.float32)
                        if self._corner_subpix_var.get() and len(pts):
                            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
                            pts = cv2.cornerSubPix(gray, pts.reshape(-1, 1, 2), (3, 3), (-1, -1), crit).reshape(-1, 2)
                    dets["corners"] = pts.astype(np.float64) + origin
                except (ValueError, cv2.error) as e:
                    errors.append(f"Corners: {e}")
            if self._det_features_var.get():
                try:
                    found = cv2.goodFeaturesToTrack(gray, maxCorners=max(1, int(self._feat_max_var.get())),
                                                    qualityLevel=float(self._feat_quality_var.get()),
                                                    minDistance=3)
                    # goodFeaturesToTrack picks maxima on the PIXEL GRID (whole
                    # numbers); cornerSubPix moves them to the sub-pixel corner.
                    if found is not None and len(found) and self._feat_subpix_var.get():
                        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
                        found = cv2.cornerSubPix(gray, found.astype(np.float32), (3, 3), (-1, -1), crit)
                    dets["features"] = (np.empty((0, 2)) if found is None
                                        else found.reshape(-1, 2).astype(np.float64) + origin)
                except (ValueError, cv2.error) as e:
                    errors.append(f"Features: {e}")
        if self._det_aruco_var.get():
            try:
                key = (self._display_source_keyed()[0], self._aruco_dict_var.get(),
                       bool(self._aruco_refine_var.get()))
                if self._aruco_frame_cache[0] != key:      # whole frame, once per frame / setting
                    corners, ids, _rej = self._aruco_detector().detectMarkers(self._gray_u8(src))
                    markers = ([] if ids is None else
                               [(int(mid), quad.reshape(4, 2).astype(np.float64))
                                for quad, mid in zip(corners, ids.ravel())])
                    self._aruco_frame_cache = (key, markers)
                reach = 2 * half                           # markers touching the magnifier area
                dets["markers"] = [(mid, q) for mid, q in self._aruco_frame_cache[1]
                                   if q[:, 0].min() - reach <= px <= q[:, 0].max() + reach
                                   and q[:, 1].min() - reach <= py <= q[:, 1].max() + reach]
            except (ValueError, cv2.error, AttributeError) as e:
                errors.append(f"ArUco: {e}")
        return dets, "; ".join(errors)

    def _aruco_detector(self):
        key = (self._aruco_dict_var.get(), bool(self._aruco_refine_var.get()))
        if self._aruco_cache[0] != key:
            dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, key[0]))
            params = cv2.aruco.DetectorParameters()
            if key[1]:
                params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self._aruco_cache = (key, cv2.aruco.ArucoDetector(dictionary, params))
        return self._aruco_cache[1]

    # ── keyboard shortcuts (whole inspector window) ───────────────────
    def _bind_shortcuts(self, win) -> None:
        """Alt+letter / Alt+digit work wherever the focus is in the inspector;
        the underlined letters in the labels show them."""
        def toggle(var):
            return lambda _e: (var.set(not var.get()), "break")[1]

        def cycle_interp(_e):
            names = list(self._INTERPOLATIONS)
            cur = self._interp_var.get()
            self._interp_var.set(names[(names.index(cur) + 1) % len(names)] if cur in names else names[0])
            return "break"

        def set_mode(mode):
            def _f(_e):
                self._mode_var.set(mode)
                self._on_mode_changed()
                return "break"
            return _f

        def run(fn):
            return lambda _e: (fn(), "break")[1]

        table = {
            "e": toggle(self._freeze_var), "t": run(self._fit_view), "o": run(self._one_to_one),
            "p": set_mode("pan"), "i": set_mode("pick"), "g": toggle(self._show_grid_var),
            "l": cycle_interp, "c": toggle(self._det_corners_var), "a": toggle(self._det_aruco_var),
            "f": toggle(self._det_features_var),
        }
        for letter, fn in table.items():
            win.bind(f"<Alt-KeyPress-{letter}>", fn)
            win.bind(f"<Alt-KeyPress-{letter.upper()}>", fn)
        for digit, z in enumerate(self._LOUPE_ZOOMS):        # Alt+0 = 1X ... Alt+6 = 64X
            win.bind(f"<Alt-KeyPress-{digit}>", lambda _e, z=z: (self._loupe_zoom_var.set(z), "break")[1])
        win.bind("<Control-r>", run(self._run_analysis))
        win.bind("<Control-R>", run(self._run_analysis))

    # ── points tab: input (read-only) and output (editable) ────────────
    _INPUT_REFRESH_S = 0.25        # input box refresh at most 4x/s (live streams)
    _OUT_BG, _OUT_BG_EDITED = "#ffffff", "#fff6d6"

    def _build_points_tab(self, tab) -> None:
        tk.Label(tab, justify="left", anchor="w", wraplength=410, fg="#34405a", text=(
            "Pick mode: click = add, drag = move.  C = point at the cross, S = nearest detection, "
            "Ctrl+Z = undo.  Image actions send at once; typed edits are sent by Apply.")).pack(fill="x")

        pw = tk.PanedWindow(tab, orient=tk.VERTICAL, sashwidth=6, sashrelief=tk.RAISED)
        pw.pack(fill="both", expand=True, pady=(6, 0))

        # Input points: what arrives on the input pin -- read-only, copyable.
        box = tk.LabelFrame(pw, text="Input points  (from input pin 'points' - read-only)",
                            padx=4, pady=3, fg="#2f4f2f")
        pw.add(box, minsize=80, height=100, stretch="always")
        row = tk.Frame(box)
        row.pack(fill="x")
        self._input_info_var = tk.StringVar(value="")
        tk.Label(row, textvariable=self._input_info_var, anchor="w", fg="#2f4f2f").pack(side="left")
        tk.Button(row, text="-> Add to output", command=self._add_input_points).pack(side="right")
        tk.Button(row, text="Copy all", command=self._copy_input_points).pack(side="right", padx=(0, 4))
        # "Live": refresh the box (pauses while text is selected).
        tk.Checkbutton(row, text="Live", variable=self._input_live_var).pack(side="right", padx=(0, 2))
        frm = tk.Frame(box)
        frm.pack(fill="both", expand=True)
        tin = tk.Text(frm, height=3, width=32, font=("Consolas", 9), wrap="none", bg="#f2f5f2",
                      fg="#2f4f2f", relief=tk.SUNKEN, cursor="arrow")
        vsb = ttk.Scrollbar(frm, orient="vertical", command=tin.yview)
        tin.configure(yscrollcommand=vsb.set, state=tk.DISABLED)   # read-only; selection + Ctrl+C still work
        vsb.pack(side="right", fill="y")
        tin.pack(side="left", fill="both", expand=True)
        tin.bind("<Control-a>", lambda _e: (tin.tag_add("sel", "1.0", tk.END), "break")[1])
        tin.bind("<Button-1>", lambda _e: tin.focus_set())          # disabled Text: allow focus for Ctrl+C
        self._input_text = tin

        # Output points: what this node sends -- editable, sent only on Apply.
        box = tk.LabelFrame(pw, text="Output points  (to output pin 'points' - editable)",
                            padx=4, pady=3, fg="#1c3a6b")
        pw.add(box, minsize=160, stretch="always")
        self._out_state_var = tk.StringVar(value="")
        self._out_state = tk.Label(box, textvariable=self._out_state_var, anchor="w", justify="left",
                                   font=("Arial", 9, "bold"), wraplength=370)
        self._out_state.pack(fill="x")
        frm = tk.Frame(box)
        frm.pack(fill="both", expand=True)
        txt = tk.Text(frm, height=4, width=32, font=("Consolas", 10), undo=True, wrap="none",
                      bg=self._OUT_BG)
        vsb = ttk.Scrollbar(frm, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        txt.pack(side="left", fill="both", expand=True)
        txt.bind("<<Modified>>", self._on_text_modified)
        txt.bind("<Control-Return>", lambda _e: (self._apply_text_edits(), "break")[1])
        self._points_text = txt
        row = tk.Frame(box)                          # sending | list tools (these send at once)
        row.pack(fill="x", pady=(4, 0))
        tk.Button(row, text="Apply (send)", font=("Arial", 9, "bold"), bg="#dcebdc",
                  command=self._apply_text_edits).pack(side="left")
        tk.Button(row, text="Revert", command=self._revert_text).pack(side="left", padx=(3, 0))
        tk.Button(row, text="Copy all", command=self._copy_all).pack(side="right")
        tk.Button(row, text="Clear all", command=self._clear_points).pack(side="right", padx=(0, 3))
        tk.Button(row, text="Undo last", command=self._undo_last).pack(side="right", padx=(0, 3))
        self._points_msg_var = tk.StringVar(value="")
        self._points_msg = tk.Label(box, textvariable=self._points_msg_var, anchor="w", justify="left",
                                    wraplength=350)
        self._points_msg.pack(fill="x", pady=(2, 0))

        opts = tk.Frame(tab)
        opts.pack(fill="x", pady=(4, 0))
        tk.Checkbutton(opts, text="Show output points", variable=self._show_picked_var).pack(side="left")
        tk.Checkbutton(opts, text="Show numbers", variable=self._show_numbers_var).pack(side="left", padx=(8, 0))
        self._refresh_input_text(force=True)

    # -- input box ------------------------------------------------------
    @staticmethod
    def _format_rows(arr) -> str:
        arr = np.asarray(arr, dtype=np.float64).reshape(len(arr), -1)
        return "\n".join(", ".join(f"{v:.3f}" for v in row) for row in arr)

    def _refresh_input_text(self, force: bool = False) -> None:
        """Show the input pin's points. Throttled for live streams, and paused
        while the user has text selected (so a selection being copied is not
        wiped out) or when Live update is off."""
        tin = getattr(self, "_input_text", None)
        if tin is None:
            return
        now = time.perf_counter()
        if not force:
            if not self._input_live_var.get() or now - self._last_input_refresh < self._INPUT_REFRESH_S:
                return
            try:
                if tin.tag_ranges("sel"):
                    return
            except tk.TclError:
                return
        self._last_input_refresh = now
        pts = self._input_points
        n, cols = (pts.shape[0], pts.shape[1]) if pts.ndim == 2 else (0, 2)
        content = self._format_rows(pts) if n else ""
        if self._input_info_var is not None:
            names = {2: "x, y", 3: "x, y, size", 4: "x, y, w, h"}.get(cols, "")
            self._input_info_var.set(f"{n} point{'s' if n != 1 else ''}  ({names})" if n
                                     else "nothing on the input pin")
        if content == self._input_shown:
            return
        self._input_shown = content
        try:
            tin.configure(state=tk.NORMAL)
            tin.delete("1.0", tk.END)
            tin.insert("1.0", content)
            tin.configure(state=tk.DISABLED)
        except tk.TclError:
            pass

    def _copy_input_points(self) -> None:
        pts = self._input_points
        self._copy_to_clipboard(self._format_rows(pts) if len(pts) else "")
        self._set_points_msg(f"Copied {len(pts)} input point{'s' if len(pts) != 1 else ''} to the clipboard.")

    def _add_input_points(self) -> None:
        pts = self._input_points
        if not len(pts) or not self._check_no_unsent_edits():
            return
        self._picked += [(float(p[0]), float(p[1])) for p in pts]
        self._picked_changed()

    # -- output box -----------------------------------------------------
    def _set_points_msg(self, text: str, color: str = "#334477") -> None:
        if self._points_msg_var is not None:
            self._points_msg_var.set(text)
            self._points_msg.configure(fg=color)

    def _show_out_state(self) -> None:
        """Header of the output box: what is currently sent, or 'edited'."""
        var = getattr(self, "_out_state_var", None)
        if var is None:
            return
        n = len(self._picked)
        if self._text_dirty:
            var.set("Edited - NOT sent yet. Apply (Ctrl+Enter) sends, Revert discards.")
            self._out_state.configure(fg="#b36b00")
            color = self._OUT_BG_EDITED
        else:
            var.set(f"Sent: {n} point{'s' if n != 1 else ''}  ({n} x 2)" if n
                    else "Sent: nothing (the output pin sends no points)")
            self._out_state.configure(fg="#2e7d32" if n else "#555555")
            color = self._OUT_BG
        if self._points_text is not None:
            self._points_text.configure(bg=color)

    def _points_summary(self) -> None:
        self._show_out_state()
        self._set_points_msg("")

    def _sync_text_from_points(self) -> None:
        txt = self._points_text
        if txt is not None:
            txt.delete("1.0", tk.END)
            txt.insert("1.0", "\n".join(f"{x:.3f}, {y:.3f}" for x, y in self._picked))
            txt.see(tk.END)
            txt.edit_modified(False)
        self._text_dirty = False
        self._points_summary()

    def _on_text_modified(self, _event=None) -> None:
        txt = self._points_text
        if txt is None or not txt.edit_modified():
            return
        if not self._text_dirty:                     # typing only marks the box; nothing is sent
            self._text_dirty = True
            self._show_out_state()

    def _apply_text_edits(self) -> bool:
        txt = self._points_text
        if txt is None:
            return True
        points, errors = _parse_point_lines(txt.get("1.0", tk.END))
        if errors:
            more = f"\n... and {len(errors) - 5} more" if len(errors) > 5 else ""
            self._set_points_msg("Not sent -- fix these lines:\n" + "\n".join(errors[:5]) + more, "#c62828")
            return False
        self._picked = points
        txt.edit_modified(False)
        self._text_dirty = False
        self._picked_changed(rewrite_text=False)
        return True

    def _revert_text(self) -> None:
        self._sync_text_from_points()
        self._set_points_msg("Edits discarded; the list shows what is sent.")

    def _check_no_unsent_edits(self) -> bool:
        """Image actions (pick, drag, C, S, undo, ...) change and SEND the list.
        If the Output box has typed edits that were not sent yet, they are
        applied (sent) FIRST, so the action works on what you typed and the
        typing is never lost. Only when a typed line is not valid is the
        action refused -- with a message ON the image, where you are looking."""
        if not self._text_dirty:
            return True
        if self._apply_text_edits():
            self._set_points_msg("Your typed edits were applied (sent) first, then the image action.",
                                 "#2e7d32")
            self._flash_view("Typed edits in the Output box were applied first", "#2e7d32")
            return True
        self._flash_view("Not done: the Output box has typed lines that are not 'x, y'.\n"
                         "Fix them (Points tab) or press Revert.", "#c62828")
        return False

    def _flash_view(self, text: str, color: str = "#334477", ms: int = 3500) -> None:
        """A short message over the top of the image view."""
        c = self._c_view
        if c is None:
            return
        old = getattr(self, "_flash_label", None)
        if old is not None:
            try:
                old.destroy()
            except tk.TclError:
                pass
        lbl = tk.Label(c, text=text, bg="#fffbe6", fg=color, font=("Arial", 9, "bold"),
                       relief="solid", borderwidth=1, padx=8, pady=3, justify="center")
        lbl.place(relx=0.5, y=8, anchor="n")
        self._flash_label = lbl
        lbl.after(ms, lambda: lbl.destroy() if lbl.winfo_exists() else None)

    def _picked_changed(self, rewrite_text: bool = True) -> None:
        if rewrite_text:
            self._sync_text_from_points()
        else:
            self._points_summary()
        self._emit_points()
        self._request_render()

    def _add_point(self, x: float, y: float) -> bool:
        if not self._check_no_unsent_edits():
            return False
        now = time.perf_counter()
        t_last, p_last = self._last_pick
        if p_last is not None and now - t_last < 0.4 and abs(p_last[0] - x) < 0.5 and abs(p_last[1] - y) < 0.5:
            return False                                 # second click of a double-click
        self._last_pick = (now, (x, y))
        self._picked.append((float(x), float(y)))
        self._picked_changed()
        return True

    def _remove_point(self, index) -> None:
        if index is None or not self._check_no_unsent_edits() or not (0 <= index < len(self._picked)):
            return
        del self._picked[index]
        self._picked_changed()

    def _undo_last(self) -> str:
        if self._check_no_unsent_edits() and self._picked:
            self._picked.pop()
            self._picked_changed()
        return "break"

    def _clear_points(self) -> None:
        self._picked = []
        self._picked_changed()

    def _copy_all(self) -> None:
        """Copy what is in the output box (sent or not)."""
        if self._points_text is not None:
            text = self._points_text.get("1.0", "end-1c")
        else:
            text = "\n".join(f"{x:.3f}, {y:.3f}" for x, y in self._picked)
        self._copy_to_clipboard(text)
        n = len([l for l in text.splitlines() if l.strip()])
        self._set_points_msg(f"Copied {n} line{'s' if n != 1 else ''} to the clipboard.")

    # ── analysis ──────────────────────────────────────────────────────
    def _build_analysis_tab(self, tab) -> None:
        tk.Label(tab, justify="left", anchor="w", wraplength=360, fg="#34405a", text=(
            "Runs ONCE on the frame shown in the view when you press Apply "
            "(freeze a live video first).")).pack(fill="x")
        row = tk.Frame(tab)
        row.pack(fill="x", pady=(6, 0))
        tk.Label(row, text="Tool:").pack(side="left")
        labels = [label for label, _p in self._TOOLS.values()]
        self._tool_combo = ttk.Combobox(row, values=labels, state="readonly", width=34)
        self._tool_combo.set(self._TOOLS[self._tool_var.get()][0])
        self._tool_combo.pack(side="left", padx=(4, 0))
        self._tool_combo.bind("<<ComboboxSelected>>", self._on_tool_selected)
        self._param_frame = tk.LabelFrame(tab, text="Parameters", padx=6, pady=4)
        self._param_frame.pack(fill="x", pady=(6, 0))
        self._build_param_fields()

        row = tk.Frame(tab)
        row.pack(fill="x", pady=(6, 0))
        tk.Button(row, text="Apply to shown frame", font=("Arial", 9, "bold"), bg="#e8dcf5",
                  command=self._run_analysis).pack(side="left")
        tk.Button(row, text="Clear result", command=self._clear_result).pack(side="left", padx=(6, 0))
        row = tk.Frame(tab)
        row.pack(fill="x", pady=(4, 0))
        tk.Checkbutton(row, text="Show result overlay", variable=self._show_result_var).pack(side="left")
        self._result_image_cb = tk.Checkbutton(row, text="Show result image",
                                               variable=self._show_result_image_var)
        self._result_image_cb.pack(side="left", padx=(8, 0))
        self._add_points_btn = tk.Button(tab, text="Add result points to picked list",
                                         command=self._add_result_points)
        self._add_points_btn.pack(anchor="w", pady=(4, 0))

        tk.Label(tab, text="Result", font=("Arial", 9, "bold"), anchor="w").pack(fill="x", pady=(6, 2))
        frm = tk.Frame(tab)
        frm.pack(fill="both", expand=True)
        txt = tk.Text(frm, height=6, width=32, font=("Consolas", 9), wrap="none", bg="#f6f6f6")
        vsb = ttk.Scrollbar(frm, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=vsb.set, state=tk.DISABLED)
        vsb.pack(side="right", fill="y")
        txt.pack(side="left", fill="both", expand=True)
        self._result_text = txt

    def _on_tool_selected(self, _event=None) -> None:
        label = self._tool_combo.get()
        for key, (tlabel, _p) in self._TOOLS.items():
            if tlabel == label and key != self._tool_var.get():
                self._tool_var.set(key)                  # -> _on_tool_var_changed

    def _on_tool_var_changed(self) -> None:
        combo = self._tool_combo
        if combo is None:
            return
        try:
            label = self._TOOLS[self._tool_var.get()][0]
            if combo.get() != label:
                combo.set(label)
        except (KeyError, tk.TclError):
            return
        self._build_param_fields()

    def _build_param_fields(self) -> None:
        f = self._param_frame
        if f is None:
            return
        for child in f.winfo_children():
            child.destroy()
        tool = self._tool_var.get()
        for r, (key, label, _default, kind) in enumerate(self._TOOLS[tool][1]):
            var = self._tool_vars[tool][key]
            if kind == "bool":
                tk.Checkbutton(f, text=label, variable=var).grid(row=r, column=0, columnspan=2, sticky="w")
                continue
            tk.Label(f, text=label).grid(row=r, column=0, sticky="w", pady=1)
            if kind.startswith("choice:"):
                ttk.Combobox(f, textvariable=var, values=kind[7:].split("|"), state="readonly",
                             width=22).grid(row=r, column=1, sticky="w", padx=(6, 0))
            else:
                tk.Entry(f, textvariable=var, width=10).grid(row=r, column=1, sticky="w", padx=(6, 0))
        f.columnconfigure(1, weight=1)

    def _read_params(self, tool: str):
        out, errors = {}, []
        for key, label, _default, kind in self._TOOLS[tool][1]:
            var = self._tool_vars[tool][key]
            if kind == "bool":
                out[key] = bool(var.get())
            elif kind.startswith("choice:"):
                out[key] = var.get()
            else:
                try:
                    out[key] = int(var.get()) if kind == "int" else float(var.get())
                except ValueError:
                    errors.append(f"{label}: '{var.get()}' is not a{'n integer' if kind == 'int' else ' number'}")
        return out, errors

    @staticmethod
    def _gray_u8(img: np.ndarray) -> np.ndarray:
        a = ImageDisplayNode._to_display_rgb(img) if img.dtype != np.uint8 or img.ndim == 3 else img
        if a.ndim == 3:
            a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        return np.ascontiguousarray(a)

    def _run_analysis(self) -> None:
        frame = self._shown
        if frame is None:
            self._set_result(None, "No frame to analyse yet.")
            return
        tool = self._tool_var.get()
        params, errors = self._read_params(tool)
        if errors:
            self._set_result(self._result, "Parameters not valid:\n" + "\n".join(errors), keep=True)
            return
        gray = self._gray_u8(frame)
        t0 = time.perf_counter()
        try:
            res = getattr(self, f"_run_{tool}")(gray, params)
        except (cv2.error, ValueError) as e:
            self._set_result(None, f"{self._TOOLS[tool][0]} failed:\n{e}")
            return
        res["tool"] = tool
        res["header"] = (f"{self._TOOLS[tool][0]}  on {gray.shape[1]} x {gray.shape[0]} frame, "
                         f"{(time.perf_counter() - t0) * 1000:.1f} ms\n" + res.get("header", ""))
        if res.get("image") is not None:
            self._show_result_image_var.set(True)
        self._set_result(res)

    def _run_gftt(self, gray, p) -> dict:
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=max(0, p["max_corners"]),
                                          qualityLevel=p["quality"], minDistance=p["min_distance"],
                                          blockSize=max(1, p["block_size"]),
                                          useHarrisDetector=p["use_harris"], k=p["k"])
        if corners is None:
            return {"points": np.empty((0, 2)), "header": "0 corners found\n"}
        if p["subpix"] and len(corners):
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
            corners = cv2.cornerSubPix(gray, corners.astype(np.float32), (5, 5), (-1, -1), crit)
        pts = corners.reshape(-1, 2).astype(np.float64)
        return {"points": pts, "header": f"{len(pts)} corners found\n"}

    def _run_chessboard(self, gray, p) -> dict:
        if p["cols"] < 2 or p["rows"] < 2:
            raise ValueError("cols and rows (inner corners) must be at least 2")
        flags = ((cv2.CALIB_CB_ADAPTIVE_THRESH if p["adaptive"] else 0)
                 | (cv2.CALIB_CB_NORMALIZE_IMAGE if p["normalize"] else 0)
                 | (cv2.CALIB_CB_FAST_CHECK if p["fast"] else 0))
        found, corners = cv2.findChessboardCorners(gray, (p["cols"], p["rows"]), flags=flags)
        if not found or corners is None:
            return {"points": np.empty((0, 2)),
                    "header": f"board NOT found (pattern {p['cols']} x {p['rows']} inner corners)\n"}
        if p["subpix"]:
            win = max(2, p["subpix_win"])
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
            corners = cv2.cornerSubPix(gray, corners.astype(np.float32), (win, win), (-1, -1), crit)
        pts = corners.reshape(-1, 2).astype(np.float64)
        return {"points": pts, "ordered": True, "cols": p["cols"],
                "header": f"board found: {len(pts)} corners ({p['cols']} x {p['rows']}), "
                          f"row by row from the larger mark\n"}

    def _run_aruco(self, gray, p) -> dict:
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, p["dictionary"]))
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = {
            "subpix": cv2.aruco.CORNER_REFINE_SUBPIX, "contour": cv2.aruco.CORNER_REFINE_CONTOUR,
            "apriltag": cv2.aruco.CORNER_REFINE_APRILTAG}.get(p["refine"], cv2.aruco.CORNER_REFINE_NONE)
        if not 3 <= p["win_min"] <= p["win_max"]:
            raise ValueError("adaptive threshold windows: need 3 <= min <= max")
        params.adaptiveThreshWinSizeMin = p["win_min"]
        params.adaptiveThreshWinSizeMax = p["win_max"]
        params.minMarkerPerimeterRate = max(0.001, p["min_perimeter"])
        corners, ids, _rejected = cv2.aruco.ArucoDetector(dictionary, params).detectMarkers(gray)
        if ids is None or not len(ids):
            return {"points": np.empty((0, 2)), "aruco_ids": np.empty(0, int),
                    "header": f"no {p['dictionary']} marker found\n"}
        ids = ids.ravel().astype(int)
        quads = [q.reshape(4, 2).astype(np.float64) for q in corners]
        order = np.argsort(ids, kind="stable") if p["order"] == "by id" else np.arange(len(ids))
        pts = []
        for k in order:
            pts.extend(quads[k])
            pts.append(_aruco_center(quads[k], p["center"]))
        ids = ids[order]
        return {"points": np.array(pts, dtype=np.float64), "aruco_ids": ids, "group": 5,
                "header": (f"{len(ids)} marker(s), ids {', '.join(str(i) for i in ids[:30])}"
                           f"{' ...' if len(ids) > 30 else ''}\n"
                           f"5 points each: corners 0-3 (clockwise from the marker's top-left), "
                           f"then the centre ({p['center']})\n")}

    def _run_hough(self, gray, p) -> dict:
        edges = cv2.Canny(gray, p["canny1"], p["canny2"])
        theta = np.deg2rad(p["theta_deg"])
        h, w = gray.shape[:2]
        if p["variant"].startswith("segments"):
            found = cv2.HoughLinesP(edges, p["rho"], theta, max(1, p["threshold"]),
                                    minLineLength=p["min_length"], maxLineGap=p["max_gap"])
            segs = np.empty((0, 4)) if found is None else found.reshape(-1, 4).astype(np.float64)
            return {"lines": segs, "header": f"{len(segs)} line segments (x1, y1, x2, y2)\n"}
        found = cv2.HoughLines(edges, p["rho"], theta, max(1, p["threshold"]))
        polar = np.empty((0, 2)) if found is None else found.reshape(-1, 2).astype(np.float64)
        segs, big = [], 2.0 * (w + h)
        for rho, th in polar:                        # clip each infinite line to the image
            a, b = np.cos(th), np.sin(th)
            x0, y0 = a * rho, b * rho
            p1 = (int(round(x0 - big * b)), int(round(y0 + big * a)))
            p2 = (int(round(x0 + big * b)), int(round(y0 - big * a)))
            ok, q1, q2 = cv2.clipLine((0, 0, w, h), p1, p2)
            segs.append((*q1, *q2) if ok else (np.nan,) * 4)
        return {"lines": np.array([s for s in segs if np.isfinite(s).all()], dtype=np.float64).reshape(-1, 4),
                "polar": polar, "header": f"{len(polar)} lines (rho px, theta deg)\n"}

    def _run_sobel(self, gray, p) -> dict:
        k = p["ksize"]
        if k not in (1, 3, 5, 7):
            raise ValueError("kernel size must be 1, 3, 5 or 7")
        g = gray.astype(np.float64)
        gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=k)
        gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=k)
        out = {"d/dx": np.abs(gx), "d/dy": np.abs(gy)}.get(p["output"], np.hypot(gx, gy))
        return {"image": out, "header": f"{p['output']}: min {out.min():.4g}, max {out.max():.4g}, "
                                        f"mean {out.mean():.4g}\n"}

    def _run_canny(self, gray, p) -> dict:
        if p["aperture"] not in (3, 5, 7):
            raise ValueError("aperture size must be 3, 5 or 7")
        edges = cv2.Canny(gray, p["threshold1"], p["threshold2"],
                          apertureSize=p["aperture"], L2gradient=p["l2"])
        n = int(np.count_nonzero(edges))
        return {"image": edges, "header": f"{n} edge pixels ({100.0 * n / edges.size:.2f}% of the image)\n"}

    def _set_result(self, res, message: str = "", keep: bool = False) -> None:
        if not keep:
            self._result = res
            self._result_serial += 1
        self._show_result_text(message)
        self._request_render()

    def _clear_result(self) -> None:
        self._set_result(None, "")

    def _show_result_text(self, message: str = "") -> None:
        txt = self._result_text
        res = self._result
        if self._add_points_btn is not None:
            has_pts = res is not None and (len(res.get("points", ())) > 0 or len(res.get("lines", ())) > 0)
            self._add_points_btn.configure(state=tk.NORMAL if has_pts else tk.DISABLED)
        if self._result_image_cb is not None:
            self._result_image_cb.configure(
                state=tk.NORMAL if (res is not None and res.get("image") is not None) else tk.DISABLED)
        if txt is None:
            return
        lines = []
        if message:
            lines.append(message)
        if res is not None:
            lines.append(res.get("header", "").rstrip())
            pts, segs = res.get("points"), res.get("lines")
            if res.get("aruco_ids") is not None and pts is not None and len(pts):
                names = ("corner 0", "corner 1", "corner 2", "corner 3", "centre  ")
                n = 0
                for m, mid in enumerate(res["aruco_ids"]):
                    if n >= self._MAX_LISTED:
                        break
                    for j in range(5):
                        n += 1
                        x, y = pts[5 * m + j]
                        lines.append(f"{n:4d}  id {mid:<4d} {names[j]}  {x:10.3f}, {y:10.3f}")
            elif res.get("polar") is not None:
                for i, (rho, th) in enumerate(res["polar"][: self._MAX_LISTED], start=1):
                    lines.append(f"{i:4d}  rho {rho:9.2f}  theta {np.degrees(th):8.3f}")
            elif segs is not None and len(segs):
                for i, s in enumerate(segs[: self._MAX_LISTED], start=1):
                    lines.append(f"{i:4d}  {s[0]:8.1f} {s[1]:8.1f}  ->  {s[2]:8.1f} {s[3]:8.1f}")
            elif pts is not None and len(pts):
                for i, (x, y) in enumerate(pts[: self._MAX_LISTED], start=1):
                    lines.append(f"{i:4d}  {x:10.3f}, {y:10.3f}")
            count = len(segs) if segs is not None and len(segs) else (len(pts) if pts is not None else 0)
            if count > self._MAX_LISTED:
                lines.append(f"... {count - self._MAX_LISTED} more not listed")
        if not lines:
            lines = ["No result yet. Choose a tool and press 'Apply to shown frame'."]
        txt.configure(state=tk.NORMAL)
        txt.delete("1.0", tk.END)
        txt.insert("1.0", "\n".join(lines))
        txt.configure(state=tk.DISABLED)            # read-only, still selectable / copyable

    def _add_result_points(self) -> None:
        res = self._result
        if res is None or not self._check_no_unsent_edits():
            return
        pts = res.get("points")
        if pts is None or not len(pts):
            segs = res.get("lines")
            pts = (np.asarray(segs).reshape(-1, 2) if segs is not None and len(segs) else np.empty((0, 2)))
        self._picked += [(float(x), float(y)) for x, y in pts]
        self._picked_changed()

    # ── display tab ───────────────────────────────────────────────────
    def _build_display_tab(self, tab) -> None:
        box = tk.LabelFrame(tab, text="Status bar", padx=6, pady=4)
        box.pack(fill="x")
        tk.Checkbutton(box, text="Show cursor position and pixel value",
                       variable=self._show_coords_var).pack(anchor="w")
        tk.Checkbutton(box, text="Show frame rate", variable=self._show_fps_var).pack(anchor="w")

        box = tk.LabelFrame(tab, text="Input 'points' overlay", padx=6, pady=4)
        box.pack(fill="x", pady=(8, 0))
        tk.Checkbutton(box, text="Show input points", variable=self._show_input_points_var).grid(
            row=0, column=0, columnspan=2, sticky="w")
        tk.Label(box, text="Marker").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Combobox(box, textvariable=self._marker_var, values=self._MARKERS, state="readonly",
                     width=6).grid(row=1, column=1, sticky="w")
        tk.Label(box, text="Colour (R, G, B)").grid(row=2, column=0, sticky="w", pady=2)
        crow = tk.Frame(box)
        crow.grid(row=2, column=1, sticky="w")
        swatch = tk.Label(crow, width=3, relief=tk.SUNKEN, bg=self._marker_color())
        for var in (self._marker_red_var, self._marker_green_var, self._marker_blue_var):
            tk.Spinbox(crow, from_=0, to=255, textvariable=var, width=4, justify="center").pack(
                side="left", padx=(0, 3))
            var.trace_add("write", lambda *_a: self._update_swatch(swatch))
        swatch.pack(side="left", padx=(4, 0))
        tk.Label(box, text="Line width").grid(row=3, column=0, sticky="w", pady=2)
        tk.Spinbox(box, from_=1, to=20, textvariable=self._marker_width_var, width=4,
                   justify="center").grid(row=3, column=1, sticky="w")
        tk.Label(box, justify="left", anchor="w", fg="#555555", wraplength=330, text=(
            "Input points: N x 2 (x, y), N x 3 (x, y, window size) or N x 4 "
            "(x, y, window width, height); a window is drawn as a rectangle.")).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))

        tk.Label(tab, justify="left", anchor="w", fg="#555555", wraplength=360, text=(
            "Colours in the view: picked points cyan (numbered), analysis points magenta, "
            "analysis lines yellow.")).pack(fill="x", pady=(10, 0))

    def _update_swatch(self, swatch) -> None:
        try:
            swatch.configure(bg=self._marker_color())
        except tk.TclError:
            pass

    # ── help ──────────────────────────────────────────────────────────
    def _open_help(self) -> None:
        if self._help_popup is not None and self._help_popup.winfo_exists():
            self._help_popup.lift()
            return
        parent = self._inspector_win or self.canvas.winfo_toplevel()
        popup = tk.Toplevel(parent)
        popup.title("Image Display - Help")
        popup.geometry("720x620")
        body = tk.Frame(popup, padx=8, pady=8)
        body.pack(fill="both", expand=True)
        txt = tk.Text(body, font=("Courier", 9), wrap=tk.WORD, relief=tk.FLAT)
        vsb = ttk.Scrollbar(body, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", _IMAGE_DISPLAY_HELP)
        txt.configure(state="disabled")
        popup.bind("<Escape>", lambda _e: popup.destroy())
        self._help_popup = popup

    def _close_help(self) -> None:
        if self._help_popup is not None:
            try:
                self._help_popup.destroy()
            except tk.TclError:
                pass
        self._help_popup = None

    def on_destroy(self) -> None:
        self._close_help()
        super().on_destroy()

    # ── serialization ─────────────────────────────────────────────────
    def get_params(self) -> dict:
        self._init_state()
        # Saves what is SENT (self._picked); edits not applied are not sent and not saved.
        return {
            "show_fps": bool(self._show_fps_var.get()),
            "show_coords": bool(self._show_coords_var.get()),
            "show_grid": bool(self._show_grid_var.get()),
            "interpolation": self._interp_var.get(),
            "marker": self._marker_var.get(),
            "marker_color": [self._marker_red_var.get(), self._marker_green_var.get(),
                             self._marker_blue_var.get()],
            "marker_width": self._marker_width(),
            "show_input_points": bool(self._show_input_points_var.get()),
            "show_picked": bool(self._show_picked_var.get()),
            "show_numbers": bool(self._show_numbers_var.get()),
            "mode": self._mode_var.get(),
            "picked_points": [[round(x, 4), round(y, 4)] for x, y in self._picked],
            "magnifier": {"zoom": int(self._loupe_zoom_var.get()),
                          "corners": bool(self._det_corners_var.get()),
                          "aruco": bool(self._det_aruco_var.get()),
                          "features": bool(self._det_features_var.get()),
                          "corner_subpix": bool(self._corner_subpix_var.get()),
                          "corner_threshold": self._corner_thresh_var.get(),
                          "aruco_dict": self._aruco_dict_var.get(),
                          "aruco_subpix": bool(self._aruco_refine_var.get()),
                          "feature_max": self._feat_max_var.get(),
                          "feature_quality": self._feat_quality_var.get(),
                          "feature_subpix": bool(self._feat_subpix_var.get())},
            "analysis_tool": self._tool_var.get(),
            "analysis_params": {tool: {k: v.get() for k, v in vs.items()}
                                for tool, vs in self._tool_vars.items()},
        }

    def set_params(self, params: dict) -> None:
        self._init_state()
        self._show_fps_var.set(bool(params.get("show_fps", True)))
        self._show_coords_var.set(bool(params.get("show_coords", True)))
        self._show_grid_var.set(bool(params.get("show_grid", False)))
        interp = str(params.get("interpolation", "NEAREST")).upper()
        self._interp_var.set(interp if interp in self._INTERPOLATIONS else "NEAREST")
        marker = str(params.get("marker", "+"))
        self._marker_var.set(marker if marker in self._MARKERS else "+")
        try:
            r, g, b = (max(0, min(255, int(v))) for v in params.get("marker_color", (0, 255, 0)))
        except (TypeError, ValueError):
            r, g, b = 0, 255, 0
        self._marker_red_var.set(r)
        self._marker_green_var.set(g)
        self._marker_blue_var.set(b)
        try:
            self._marker_width_var.set(max(1, min(20, int(params.get("marker_width", 2)))))
        except (TypeError, ValueError):
            self._marker_width_var.set(2)
        self._show_input_points_var.set(bool(params.get("show_input_points", True)))
        self._show_picked_var.set(bool(params.get("show_picked", True)))
        self._show_numbers_var.set(bool(params.get("show_numbers", True)))
        mode = params.get("mode", "pan")
        self._mode_var.set(mode if mode in ("pan", "pick") else "pan")
        picked = []
        for p in params.get("picked_points", []) or []:
            try:
                x, y = float(p[0]), float(p[1])
            except (TypeError, ValueError, IndexError):
                continue
            if np.isfinite(x) and np.isfinite(y):
                picked.append((x, y))
        self._picked = picked
        mag = params.get("magnifier") or {}
        try:                                             # snap (older projects: 10X..90X)
            z = float(mag.get("zoom", 16))
            self._loupe_zoom_var.set(min(self._LOUPE_ZOOMS, key=lambda v: abs(np.log2(v) - np.log2(max(z, 1)))))
        except (TypeError, ValueError):
            self._loupe_zoom_var.set(16)
        for key, var in (("corners", self._det_corners_var), ("aruco", self._det_aruco_var),
                         ("features", self._det_features_var), ("corner_subpix", self._corner_subpix_var),
                         ("aruco_subpix", self._aruco_refine_var), ("feature_subpix", self._feat_subpix_var)):
            if key in mag:
                var.set(bool(mag[key]))
        for key, var in (("corner_threshold", self._corner_thresh_var), ("feature_max", self._feat_max_var),
                         ("feature_quality", self._feat_quality_var)):
            if key in mag:
                var.set(str(mag[key]))
        if mag.get("aruco_dict") in self._ARUCO_DICTS:
            self._aruco_dict_var.set(mag["aruco_dict"])
        tool = params.get("analysis_tool", "gftt")
        self._tool_var.set(tool if tool in self._TOOLS else "gftt")
        for t, saved in (params.get("analysis_params") or {}).items():
            for k, v in (saved or {}).items():
                var = self._tool_vars.get(t, {}).get(k)
                if var is not None:
                    try:
                        var.set(bool(v) if isinstance(var, tk.BooleanVar) else str(v))
                    except tk.TclError:
                        pass
        self._sync_text_from_points()
        self._update_face(force=True)


class GaussianBlurNode(BaseNode):
    """
    Applies cv2.GaussianBlur to an incoming image.

    Pin layout:
      inputs:
        image      — IMAGE  (required)
        ksize      — SCALAR — kernel size (odd integer, default 5)
        sigma_x    — SCALAR — sigmaX (default 1.0)
        sigma_y    — SCALAR — sigmaY (default 0.0, means same as sigmaX)
        border     — SCALAR — borderType as int (default 4 = BORDER_REFLECT_101)
      outputs:
        image      — IMAGE

    All scalar inputs are optional: if not connected, the value
    shown in the node body's entry widgets is used as default.
    When a pin IS connected, the connected value overrides the widget.
    """
    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE      = "gaussian_blur"
    DISPLAY_NAME   = "Gaussian Blur"
    CATEGORY       = "process"
    NODE_WIDTH     = 200
    NODE_HEIGHT    = 190

    # cv2 border type options shown in the node body
    _BORDER_TYPES = {
        "REFLECT_101 (4)": cv2.BORDER_REFLECT_101,
        "REFLECT (2)":     cv2.BORDER_REFLECT,
        "REPLICATE (1)":   cv2.BORDER_REPLICATE,
        "CONSTANT (0)":    cv2.BORDER_CONSTANT,
        "WRAP (3)":        cv2.BORDER_WRAP,
    }

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[
                PinDef("image",   PinType.IMAGE,  "src",    optional=False),
                PinDef("ksize",   PinType.SCALAR, "ksize",  optional=True),
                PinDef("sigma_x", PinType.SCALAR, "sigmaX", optional=True),
                PinDef("sigma_y", PinType.SCALAR, "sigmaY", optional=True),
                PinDef("border",  PinType.SCALAR, "border", optional=True),
            ],
            outputs=[
                PinDef("image", PinType.IMAGE, "out"),
            ]
        )

    def _init_param_state(self) -> None:
        if hasattr(self, "_param_ksize"):
            return
        self._param_ksize = "5"
        self._param_sigma_x = "1.0"
        self._param_sigma_y = "0.0"
        self._param_border = "REFLECT_101 (4)"

        # Optional inspector widgets (exist only while popup is open)
        self._ksize_entry = None
        self._sigma_x_entry = None
        self._sigma_y_entry = None
        self._border_var = None

    def build_body(self) -> None:
        self._init_param_state()
        x, y, w, h = self.x, self.y, self.width, self.height

        # Compact canvas body (Phase 2): full controls live in popup inspector.
        self._body_rect = self.canvas.create_rectangle(
            x, y, x+w, y+h,
            fill="#eaf6ea", outline="#55aa55", width=2,
            tags=(self.node_id, "node_body"))
        self._title_item = self.canvas.create_text(
            x+w/2, y+13,
            text=self.DISPLAY_NAME,
            font=("Arial", 9, "bold"), fill="#2f6b2f",
            tags=(self.node_id,))

        # ── status ────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="")
        status_lbl = tk.Label(
            self.canvas, textvariable=self._status_var,
            font=("Arial", 7), bg="#eaf6ea", fg="#3f6f3f")
        self.canvas.create_window(
            x+w/2, y+h-14, window=status_lbl,
            tags=(self.node_id,))

        self._canvas_items += [self._body_rect, self._title_item]

    def build_inspector(self, parent: tk.Frame) -> None:
        self._init_param_state()

        title = tk.Label(
            parent,
            text="Gaussian Blur Parameters",
            font=("Arial", 10, "bold"),
            anchor="w",
        )
        title.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        def _row(label: str, value: str, row: int) -> tk.Entry:
            tk.Label(parent, text=label, anchor="w", font=("Arial", 9)).grid(
                row=row, column=0, sticky="w", padx=(0, 8), pady=2
            )
            ent = tk.Entry(parent, width=14, font=("Arial", 9), justify="center")
            ent.insert(0, value)
            ent.grid(row=row, column=1, sticky="ew", pady=2)
            return ent

        self._ksize_entry = _row("ksize", self._param_ksize, 1)
        self._sigma_x_entry = _row("sigmaX", self._param_sigma_x, 2)
        self._sigma_y_entry = _row("sigmaY", self._param_sigma_y, 3)

        tk.Label(parent, text="border", anchor="w", font=("Arial", 9)).grid(
            row=4, column=0, sticky="w", padx=(0, 8), pady=2
        )
        self._border_var = tk.StringVar(value=self._param_border)
        border_cb = ttk.Combobox(
            parent,
            textvariable=self._border_var,
            values=list(self._BORDER_TYPES.keys()),
            state="readonly",
            width=18,
            font=("Arial", 9),
        )
        border_cb.grid(row=4, column=1, sticky="ew", pady=2)

        parent.grid_columnconfigure(1, weight=1)

        def _commit_and_trigger(_event=None):
            self._sync_params_from_widgets()
            self._trigger_recompute_from_ui()

        for ent in (self._ksize_entry, self._sigma_x_entry, self._sigma_y_entry):
            ent.bind("<FocusOut>", _commit_and_trigger)
            ent.bind("<Return>", _commit_and_trigger)

        border_cb.bind("<<ComboboxSelected>>", _commit_and_trigger)

    def _sync_params_from_widgets(self) -> None:
        if self._ksize_entry is not None and self._ksize_entry.winfo_exists():
            self._param_ksize = self._ksize_entry.get().strip() or "5"
        if self._sigma_x_entry is not None and self._sigma_x_entry.winfo_exists():
            self._param_sigma_x = self._sigma_x_entry.get().strip() or "1.0"
        if self._sigma_y_entry is not None and self._sigma_y_entry.winfo_exists():
            self._param_sigma_y = self._sigma_y_entry.get().strip() or "0.0"
        if self._border_var is not None:
            self._param_border = self._border_var.get().strip() or "REFLECT_101 (4)"

    def _trigger_recompute_from_ui(self) -> None:
        if self._request_downstream:
            self._request_downstream(self.node_id)

    # ── helpers ───────────────────────────────────────────────────

    def _get_ksize(self, inputs: dict) -> int:
        """
        ksize must be a positive odd integer.
        If connected value is even, round up to next odd number.
        """
        raw = inputs.get("ksize")
        if raw is not None:
            val = int(round(float(raw)))
        else:
            try:
                val = int(float(self._param_ksize))
            except ValueError:
                val = 5
        val = max(1, val)
        if val % 2 == 0:
            val += 1
        return val

    def _get_float(self, inputs: dict,
                   pin: str, widget: tk.Entry,
                   default: float) -> float:
        raw = inputs.get(pin)
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                return default
        try:
            return float(widget)
        except ValueError:
            return default

    def _get_border(self, inputs: dict) -> int:
        raw = inputs.get("border")
        if raw is not None:
            return int(round(float(raw)))
        return self._BORDER_TYPES.get(
            self._param_border, cv2.BORDER_REFLECT_101)

    # ── compute ───────────────────────────────────────────────────

    def compute(self, inputs: dict) -> dict:
        frame = inputs.get("image")
        if frame is None or not isinstance(frame, np.ndarray):
            self._status_var.set("no image")
            return {}

        try:
            ksize   = self._get_ksize(inputs)
            sigma_x = self._get_float(
                inputs, "sigma_x", self._param_sigma_x, 1.0)
            sigma_y = self._get_float(
                inputs, "sigma_y", self._param_sigma_y, 0.0)
            border  = self._get_border(inputs)

            result = cv2.GaussianBlur(
                frame,
                ksize=(ksize, ksize),
                sigmaX=sigma_x,
                sigmaY=sigma_y,
                borderType=border)

            self._status_var.set(
                f"k={ksize}  σx={sigma_x:.2g}"
                f"  σy={sigma_y:.2g}")
            self.set_status("ok", "#55aa55")
            return {"image": result}

        except Exception as e:
            self._status_var.set(f"error: {e}")
            self.set_status("error", "#cc0000")
            return {}

    # ── serialization ─────────────────────────────────────────────

    def get_params(self) -> dict:
        self._sync_params_from_widgets()
        return {
            "ksize":   self._param_ksize,
            "sigma_x": self._param_sigma_x,
            "sigma_y": self._param_sigma_y,
            "border":  self._param_border,
        }

    def set_params(self, params: dict) -> None:
        self._init_param_state()
        self._param_ksize = str(params.get("ksize", "5"))
        self._param_sigma_x = str(params.get("sigma_x", "1.0"))
        self._param_sigma_y = str(params.get("sigma_y", "0.0"))
        self._param_border = str(params.get("border", "REFLECT_101 (4)"))

        if self._ksize_entry is not None and self._ksize_entry.winfo_exists():
            self._ksize_entry.delete(0, tk.END)
            self._ksize_entry.insert(0, self._param_ksize)
        if self._sigma_x_entry is not None and self._sigma_x_entry.winfo_exists():
            self._sigma_x_entry.delete(0, tk.END)
            self._sigma_x_entry.insert(0, self._param_sigma_x)
        if self._sigma_y_entry is not None and self._sigma_y_entry.winfo_exists():
            self._sigma_y_entry.delete(0, tk.END)
            self._sigma_y_entry.insert(0, self._param_sigma_y)
        if self._border_var is not None:
            self._border_var.set(self._param_border)

    def close_inspector(self) -> None:
        self._sync_params_from_widgets()
        super().close_inspector()
        self._ksize_entry = None
        self._sigma_x_entry = None
        self._sigma_y_entry = None
        self._border_var = None
