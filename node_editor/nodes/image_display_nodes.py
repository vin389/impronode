# node_editor/nodes/image_display_nodes.py

import tkinter as tk
from tkinter import ttk
import threading
import queue
import time
import platform
import weakref
import cv2
import numpy as np
from PIL import Image, ImageTk

from node_editor.base_node import BaseNode
from node_editor.pin_types import PinSchema, PinDef, PinType
from node_editor.execution import ExecutionMode

class ImageDisplayNode(BaseNode):
    """
    Receives IMAGE frames and displays them inside the node body.
    Automatically updates at the rate frames arrive.

    Features:
      - Resize-aware display area with preserved image aspect ratio
      - Hover highlight and optional cursor image-coordinate readout
      - Mouse-centered wheel zoom and left-drag panning
      - Optional FPS display overlay
      - Freeze button to pause display without stopping upstream
    """
    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE      = "image_display_output"
    DISPLAY_NAME   = "Image Display"
    CATEGORY       = "visualize"
    NODE_WIDTH     = 220
    NODE_HEIGHT    = 200
    OPEN_INSPECTOR_ON_IMAGE_DOUBLE_CLICK = True
    # Pin labels are rendered over the dark preview body.
    PIN_LABEL_COLOR = "#d8efff"
    _GRID_PIXEL_THRESHOLD = 5.0
    _MAX_GRID_LINES = 400

    _INTERPOLATIONS = {
        "NEAREST": cv2.INTER_NEAREST,
        "LINEAR": cv2.INTER_LINEAR,
        "CUBIC": cv2.INTER_CUBIC,
        "AREA": cv2.INTER_AREA,
        "LANCZOS4": cv2.INTER_LANCZOS4,
    }
    _BOUND_CANVASES: "weakref.WeakSet[tk.Canvas]" = weakref.WeakSet()
    _HOVERED_BY_CANVAS: "weakref.WeakKeyDictionary[tk.Canvas, ImageDisplayNode]" = weakref.WeakKeyDictionary()

    def __init__(self, node_id: str, canvas: tk.Canvas):
        super().__init__(node_id, canvas)

        # Initialize callback-referenced state early so events can never
        # observe missing attributes.
        self._photo = None
        self._frame_times = []
        self._last_frame = None
        self._points = np.empty((0, 2), dtype=np.float32)
        self._point_items = []
        self._first_frame_received = False

        self._img_hover = False
        self._is_panning = False
        self._pan_anchor_canvas = None
        self._pan_anchor_center = None

        self._zoom = 1.0
        self._view_cx = None
        self._view_cy = None
        self._min_zoom = 0.01
        self._max_zoom = 500.0

        self._grid_items = []
        self._interp_var = tk.StringVar(value="NEAREST")
        self._interp_menu = None
        self._last_coords_value: str | None = None
        self._coords_copied = False
        self._help_popup: tk.Toplevel | None = None
        self._pan_redraw_scheduled = False
        self._pending_pan_overlay: tuple[float, float] | None = None
        self._coords_update_scheduled = False
        self._pending_coords_overlay: tuple[float | None, float | None] | None = None
        self._coords_overlay_text = ""

        self._marker_var = tk.StringVar(value="+")
        self._marker_red_var = tk.IntVar(value=0)
        self._marker_green_var = tk.IntVar(value=255)
        self._marker_blue_var = tk.IntVar(value=0)
        self._marker_width_var = tk.IntVar(value=2)

    @classmethod
    def _bind_shared_canvas_events(cls, canvas: tk.Canvas) -> None:
        if canvas in cls._BOUND_CANVASES:
            return
        cls._BOUND_CANVASES.add(canvas)
        canvas.bind(
            "<MouseWheel>",
            lambda event, c=canvas: cls._dispatch_canvas_mousewheel(c, event),
            add="+",
        )
        canvas.bind(
            "<Button-4>",
            lambda event, c=canvas: cls._dispatch_canvas_wheel_linux(c, event),
            add="+",
        )
        canvas.bind(
            "<Button-5>",
            lambda event, c=canvas: cls._dispatch_canvas_wheel_linux(c, event),
            add="+",
        )
        canvas.bind(
            "<KeyPress-plus>",
            lambda event, c=canvas: cls._dispatch_hover_hotkey(c, event, "_on_zoom_in_hotkey"),
            add="+",
        )
        canvas.bind(
            "<KeyPress-minus>",
            lambda event, c=canvas: cls._dispatch_hover_hotkey(c, event, "_on_zoom_out_hotkey"),
            add="+",
        )
        canvas.bind(
            "<KeyPress-KP_Add>",
            lambda event, c=canvas: cls._dispatch_hover_hotkey(c, event, "_on_zoom_in_hotkey"),
            add="+",
        )
        canvas.bind(
            "<KeyPress-KP_Subtract>",
            lambda event, c=canvas: cls._dispatch_hover_hotkey(c, event, "_on_zoom_out_hotkey"),
            add="+",
        )
        canvas.bind(
            "<KeyPress-c>",
            lambda event, c=canvas: cls._dispatch_hover_hotkey(c, event, "_on_copy_coords_hotkey"),
            add="+",
        )
        canvas.bind(
            "<KeyPress-C>",
            lambda event, c=canvas: cls._dispatch_hover_hotkey(c, event, "_on_copy_coords_hotkey"),
            add="+",
        )
        canvas.bind(
            "<Control-h>",
            lambda event, c=canvas: cls._dispatch_hover_hotkey(c, event, "_on_help_hotkey"),
            add="+",
        )
        canvas.bind(
            "<Control-H>",
            lambda event, c=canvas: cls._dispatch_hover_hotkey(c, event, "_on_help_hotkey"),
            add="+",
        )

    @classmethod
    def _hover_target_for_canvas(cls, canvas: tk.Canvas):
        node = cls._HOVERED_BY_CANVAS.get(canvas)
        if node is not None and getattr(node, "_img_hover", False):
            return node
        if node is not None:
            cls._HOVERED_BY_CANVAS.pop(canvas, None)
        return None

    @classmethod
    def _dispatch_canvas_mousewheel(cls, canvas: tk.Canvas, event):
        node = cls._hover_target_for_canvas(canvas)
        if node is None:
            return None
        return node._on_img_mousewheel(event)

    @classmethod
    def _dispatch_canvas_wheel_linux(cls, canvas: tk.Canvas, event):
        node = cls._hover_target_for_canvas(canvas)
        if node is None:
            return None
        return node._on_img_wheel_linux(event)

    @classmethod
    def _dispatch_hover_hotkey(cls, canvas: tk.Canvas, event, handler_name: str):
        node = cls._hover_target_for_canvas(canvas)
        if node is None:
            return None
        return getattr(node, handler_name)(event)

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[
                PinDef("image", PinType.IMAGE, "frame",
                       shape=(-1, -1, 3), dtype="uint8"),
                PinDef("points", PinType.ARRAY, "points", shape=None,
                       optional=True),
            ],
            outputs=[
                PinDef("image_coords", PinType.ARRAY, "coords", shape=(1, 2), dtype="float32"),
            ]
        )

    def build_body(self) -> None:
        x, y, w, h = self.x, self.y, self.width, self.height

        # ── body & title ─────────────────────────────────────────
        self._body_rect = self.canvas.create_rectangle(
            x, y, x+w, y+h,
            fill="#0d0d1a", outline="#4488cc", width=2,
            tags=(self.node_id, "node_body"))
        self._title_item = self.canvas.create_text(
            x+w/2, y+13,
            text=self.DISPLAY_NAME,
            font=("Arial", 9, "bold"), fill="#88ccff",
            tags=(self.node_id,))

        self._img_area_tag = f"img_area_{self.node_id}"

        # ── image display area ────────────────────────────────────
        self._img_w = w - 8
        self._img_h = h - 50

        self._img_border = self.canvas.create_rectangle(
            x+4, y+22, x+w-4, y+22+self._img_h,
            fill="#111133", outline="#334466", width=1,
            tags=(self.node_id, self._img_area_tag))

        self._img_canvas_item = self.canvas.create_image(
            x + w//2, y + 22 + self._img_h//2,
            anchor="center",
            tags=(self.node_id, self._img_area_tag))

        # placeholder rectangle (shown before first frame)
        self._placeholder = self.canvas.create_rectangle(
            x+4, y+22, x+w-4, y+22+self._img_h,
            fill="#111133", outline="#334466",
            tags=(self.node_id, self._img_area_tag))
        self._placeholder_text = self.canvas.create_text(
            x+w/2, y+22+self._img_h//2,
            text="awaiting frames...",
            font=("Arial", 8), fill="#446688",
            tags=(self.node_id, self._img_area_tag))

        # ── bottom controls ───────────────────────────────────────
        ctrl_y = y + h - 26

        self._freeze_var = tk.BooleanVar(value=False)
        freeze_cb = tk.Checkbutton(
            self.canvas, text="Freeze",
            variable=self._freeze_var,
            font=("Arial", 8),
            bg="#0d0d1a", fg="#aaccff",
            selectcolor="#223344",
            activebackground="#0d0d1a",
            activeforeground="#ffffff")
        self.canvas.create_window(
            x+10, ctrl_y, window=freeze_cb, anchor="nw",
            tags=(self.node_id,))

        self._show_fps_var = tk.BooleanVar(value=True)
        fps_cb = tk.Checkbutton(
            self.canvas, text="FPS",
            variable=self._show_fps_var,
            font=("Arial", 8),
            bg="#0d0d1a", fg="#aaccff",
            selectcolor="#223344",
            activebackground="#0d0d1a",
            activeforeground="#ffffff")
        self.canvas.create_window(
            x+w-55, ctrl_y, window=fps_cb, anchor="nw",
            tags=(self.node_id,))

        self._show_coords_var = tk.BooleanVar(value=True)
        coords_cb = tk.Checkbutton(
            self.canvas, text="Coords",
            variable=self._show_coords_var,
            font=("Arial", 8),
            bg="#0d0d1a", fg="#aaccff",
            selectcolor="#223344",
            activebackground="#0d0d1a",
            activeforeground="#ffffff")
        self.canvas.create_window(
            x+w-120, ctrl_y, window=coords_cb, anchor="nw",
            tags=(self.node_id,))

        self._show_grid_var = tk.BooleanVar(value=False)
        grid_cb = tk.Checkbutton(
            self.canvas, text="Grid",
            variable=self._show_grid_var,
            font=("Arial", 8),
            bg="#0d0d1a", fg="#aaccff",
            selectcolor="#223344",
            activebackground="#0d0d1a",
            activeforeground="#ffffff")
        self.canvas.create_window(
            x+66, ctrl_y, window=grid_cb, anchor="nw",
            tags=(self.node_id,))

        # ── FPS overlay text on canvas ────────────────────────────
        self._fps_text = self.canvas.create_text(
            x+w-10, y+26,
            text="", anchor="ne",
            font=("Arial", 8, "bold"),
            fill="#00ff88",
            tags=(self.node_id,))

        self._coords_text = self.canvas.create_text(
            x+8, y+26,
            text="", anchor="nw",
            font=("Arial", 8, "bold"),
            fill="#ffd060",
            tags=(self.node_id,))

        # ── status ────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="")
        status_lbl = tk.Label(
            self.canvas, textvariable=self._status_var,
            font=("Arial", 7), bg="#0d0d1a", fg="#aaaaaa")
        self.canvas.create_window(
            x+w/2, y+h-10, window=status_lbl,
            tags=(self.node_id,))

        # Double-click is handled by the editor and opens this node's inspector.
        self.canvas.tag_bind(self._img_area_tag, "<Enter>", self._on_img_enter)
        self.canvas.tag_bind(self._img_area_tag, "<Leave>", self._on_img_leave)
        self.canvas.tag_bind(self._img_area_tag, "<Motion>", self._on_img_motion)
        self.canvas.tag_bind(self._img_area_tag, "<ButtonPress-1>", self._on_img_press)
        self.canvas.tag_bind(self._img_area_tag, "<B1-Motion>", self._on_img_drag)
        self.canvas.tag_bind(self._img_area_tag, "<ButtonRelease-1>", self._on_img_release)
        self.canvas.tag_bind(self._img_area_tag, "<Button-3>", self._on_img_right_click)

        # Mouse wheel cannot be bound with tag_bind on canvas items; bind at canvas level
        # and gate by this node's hover/focus state.
        self._bind_shared_canvas_events(self.canvas)

        self._canvas_items += [
            self._body_rect, self._title_item,
            self._img_border, self._img_canvas_item, self._placeholder,
            self._placeholder_text, self._fps_text, self._coords_text]

        # ── internal state ────────────────────────────────────────
        self._photo:         ImageTk.PhotoImage | None = None
        self._frame_times:   list[float]               = []
        self._last_frame:    np.ndarray | None         = None
        self._first_frame_received = False

        self._img_hover = False
        self._is_panning = False
        self._pan_anchor_canvas: tuple[float, float] | None = None
        self._pan_anchor_center: tuple[float, float] | None = None
        self._pan_redraw_scheduled = False
        self._pending_pan_overlay: tuple[float, float] | None = None
        self._coords_update_scheduled = False
        self._pending_coords_overlay: tuple[float | None, float | None] | None = None
        self._coords_overlay_text = ""

        # Viewport state in image coordinates: displayed center and zoom multiplier.
        self._zoom = 1.0
        self._view_cx: float | None = None
        self._view_cy: float | None = None
        self._min_zoom = 0.05
        self._max_zoom = 500.0

    # ── compute ───────────────────────────────────────────────────

    def compute(self, inputs: dict) -> dict:
        frame = inputs.get("image")
        self._points = self._normalise_points(inputs.get("points"))
        if frame is not None and isinstance(frame, np.ndarray) and not self._freeze_var.get():
            self._last_frame = frame
            self._display_frame(frame)
        elif self._last_frame is not None:
            self._display_frame(self._last_frame)
        return {}

    @staticmethod
    def _normalise_points(points) -> np.ndarray:
        """Return finite N-by-(2..4) image-coordinate point data."""
        if points is None:
            return np.empty((0, 2), dtype=np.float32)
        try:
            array = np.asarray(points, dtype=np.float32)
        except (TypeError, ValueError):
            return np.empty((0, 2), dtype=np.float32)
        if array.ndim != 2 or array.shape[1] not in (2, 3, 4):
            return np.empty((0, 2), dtype=np.float32)
        return array[np.isfinite(array).all(axis=1)]

    def _display_frame(self, frame: np.ndarray) -> None:
        # remove placeholder on first frame
        if not self._first_frame_received:
            self.canvas.itemconfigure(
                self._placeholder,      state="hidden")
            self.canvas.itemconfigure(
                self._placeholder_text, state="hidden")
            self._first_frame_received = True

        # track FPS
        now = time.perf_counter()
        self._frame_times.append(now)
        self._frame_times = [
            t for t in self._frame_times if now - t < 2.0]
        fps = len(self._frame_times) / 2.0

        if self._show_fps_var.get():
            self.canvas.itemconfigure(
                self._fps_text, text=f"{fps:.1f} fps")
        else:
            self.canvas.itemconfigure(self._fps_text, text="")

        self._status_var.set(
            f"{frame.shape[1]} x {frame.shape[0]}  "
            f"{frame.dtype}")

        img_h, img_w = frame.shape[:2]
        self._ensure_view_center(img_w, img_h)

        area = self._image_area_rect()
        aw = max(1, int(area[2] - area[0]))
        ah = max(1, int(area[3] - area[1]))

        fit_scale = min(aw / max(1, img_w), ah / max(1, img_h))
        scale = max(1e-6, fit_scale * self._zoom)

        self._view_cx = float(np.clip(self._view_cx, 0.0, img_w - 1.0))
        self._view_cy = float(np.clip(self._view_cy, 0.0, img_h - 1.0))

        tx = (aw / 2.0) - self._view_cx * scale
        ty = (ah / 2.0) - self._view_cy * scale

        transformed = cv2.warpAffine(
            frame,
            np.array([[scale, 0.0, tx], [0.0, scale, ty]], dtype=np.float32),
            (aw, ah),
            flags=self._current_interpolation_flag(),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(17, 17, 51),
        )

        img_pil = Image.fromarray(transformed)

        # ── update canvas image ───────────────────────────────────
        self._photo = ImageTk.PhotoImage(img_pil)
        self.canvas.itemconfigure(
            self._img_canvas_item, image=self._photo)

        area_cx = (area[0] + area[2]) / 2.0
        area_cy = (area[1] + area[3]) / 2.0
        self.canvas.coords(self._img_canvas_item, area_cx, area_cy)
        self.canvas.coords(self._placeholder_text, area_cx, area_cy)

        self._draw_grid_overlay(img_w, img_h, aw, ah, scale)

        # Draw markers after the image and pixel grid so they remain visible.
        self._draw_point_overlay(aw, ah, scale)

    def _current_interpolation_flag(self) -> int:
        return self._INTERPOLATIONS.get(self._interp_var.get(), cv2.INTER_NEAREST)

    def _clear_point_overlay(self) -> None:
        for item in self._point_items:
            self.canvas.delete(item)
        self._point_items = []

    def _marker_color(self) -> str:
        try:
            values = (
                self._marker_red_var.get(),
                self._marker_green_var.get(),
                self._marker_blue_var.get(),
            )
        except tk.TclError:
            values = (0, 255, 0)
        red, green, blue = (max(0, min(255, int(value))) for value in values)
        return f"#{red:02x}{green:02x}{blue:02x}"

    def _draw_point_overlay(self, aw: int, ah: int, scale: float) -> None:
        self._clear_point_overlay()
        if self._points.size == 0 or self._view_cx is None or self._view_cy is None:
            return

        x1, y1, x2, y2 = self._image_area_rect()
        color = self._marker_color()
        try:
            width = max(1, int(self._marker_width_var.get()))
        except tk.TclError:
            width = 2
        marker = self._marker_var.get()
        radius = 5.0

        for point in self._points:
            cx = x1 + (float(point[0]) - self._view_cx) * scale + aw / 2.0
            cy = y1 + (float(point[1]) - self._view_cy) * scale + ah / 2.0
            if cx < x1 - radius or cx > x2 + radius or cy < y1 - radius or cy > y2 + radius:
                continue

            if point.shape[0] >= 3:
                window_w = abs(float(point[2]))
                window_h = abs(float(point[3])) if point.shape[0] == 4 else window_w
                self._point_items.append(self.canvas.create_rectangle(
                    cx - window_w * scale / 2.0, cy - window_h * scale / 2.0,
                    cx + window_w * scale / 2.0, cy + window_h * scale / 2.0,
                    outline=color, width=width,
                    tags=(self.node_id, self._img_area_tag, "point_overlay"),
                ))

            if marker == "o":
                self._point_items.append(self.canvas.create_oval(
                    cx - radius, cy - radius, cx + radius, cy + radius,
                    outline=color, width=width,
                    tags=(self.node_id, self._img_area_tag, "point_overlay"),
                ))
            else:
                segments = [(-radius, 0.0, radius, 0.0), (0.0, -radius, 0.0, radius)]
                if marker == "*":
                    diagonal = radius * 0.72
                    segments += [(-diagonal, -diagonal, diagonal, diagonal),
                                 (-diagonal, diagonal, diagonal, -diagonal)]
                for dx1, dy1, dx2, dy2 in segments:
                    self._point_items.append(self.canvas.create_line(
                        cx + dx1, cy + dy1, cx + dx2, cy + dy2,
                        fill=color, width=width,
                        tags=(self.node_id, self._img_area_tag, "point_overlay"),
                    ))

        self.canvas.tag_raise(self._coords_text)
        self.canvas.tag_raise(self._fps_text)

    def _clear_grid_overlay(self) -> None:
        for item in self._grid_items:
            self.canvas.delete(item)
        self._grid_items = []

    def _draw_grid_overlay(self, img_w: int, img_h: int,
                           aw: int, ah: int, scale: float) -> None:
        self._clear_grid_overlay()

        if not self._show_grid_var.get() or scale <= self._GRID_PIXEL_THRESHOLD:
            return

        x1, y1, x2, y2 = self._image_area_rect()
        view_cx = float(self._view_cx)
        view_cy = float(self._view_cy)

        left_x = view_cx - aw / (2.0 * scale)
        right_x = view_cx + aw / (2.0 * scale)
        top_y = view_cy - ah / (2.0 * scale)
        bottom_y = view_cy + ah / (2.0 * scale)

        # Pixel boundaries are at k - 0.5
        kx_min = max(0, int(np.ceil(left_x + 0.5)))
        kx_max = min(img_w, int(np.floor(right_x + 0.5)))
        ky_min = max(0, int(np.ceil(top_y + 0.5)))
        ky_max = min(img_h, int(np.floor(bottom_y + 0.5)))

        # Safety cap to avoid UI stalls when line count gets too high.
        if (kx_max - kx_min + 1) > self._MAX_GRID_LINES:
            mid = (kx_min + kx_max) // 2
            half = self._MAX_GRID_LINES // 2
            kx_min = max(0, mid - half)
            kx_max = min(img_w, kx_min + self._MAX_GRID_LINES - 1)
        if (ky_max - ky_min + 1) > self._MAX_GRID_LINES:
            mid = (ky_min + ky_max) // 2
            half = self._MAX_GRID_LINES // 2
            ky_min = max(0, mid - half)
            ky_max = min(img_h, ky_min + self._MAX_GRID_LINES - 1)

        line_color = "#ffffff"
        for k in range(kx_min, kx_max + 1):
            bx = k - 0.5
            cx = x1 + (bx - view_cx) * scale + aw / 2.0
            line = self.canvas.create_line(
                cx, y1, cx, y2,
                fill=line_color,
                width=1,
                stipple="gray50",
                tags=(self.node_id, self._img_area_tag, "img_grid"),
            )
            self._grid_items.append(line)

        for k in range(ky_min, ky_max + 1):
            by = k - 0.5
            cy = y1 + (by - view_cy) * scale + ah / 2.0
            line = self.canvas.create_line(
                x1, cy, x2, cy,
                fill=line_color,
                width=1,
                stipple="gray50",
                tags=(self.node_id, self._img_area_tag, "img_grid"),
            )
            self._grid_items.append(line)

        self.canvas.tag_raise(self._coords_text)
        self.canvas.tag_raise(self._fps_text)

    def _show_interpolation_menu(self, x_root: int, y_root: int) -> None:
        if self._interp_menu is None:
            self._interp_menu = tk.Menu(self.canvas, tearoff=0)
            self._interp_menu.add_command(label="Interpolation", state="disabled")
            self._interp_menu.add_separator()
            for label in ("NEAREST", "LINEAR", "CUBIC", "AREA", "LANCZOS4"):
                self._interp_menu.add_radiobutton(
                    label=label,
                    value=label,
                    variable=self._interp_var,
                    command=self._on_interpolation_changed,
                )

        self._interp_menu.tk_popup(int(x_root), int(y_root))
        self._interp_menu.grab_release()

    def _on_interpolation_changed(self) -> None:
        if self._last_frame is not None:
            self._display_frame(self._last_frame)

    def _image_area_rect(self) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.canvas.coords(self._img_border)
        return x1, y1, x2, y2

    def _ensure_view_center(self, img_w: int, img_h: int) -> None:
        if self._view_cx is None or self._view_cy is None:
            self._view_cx = (img_w - 1.0) / 2.0
            self._view_cy = (img_h - 1.0) / 2.0

    @staticmethod
    def _clamp_view_center(cx: float, cy: float,
                           img_w: int, img_h: int) -> tuple[float, float]:
        return (
            float(np.clip(cx, 0.0, img_w - 1.0)),
            float(np.clip(cy, 0.0, img_h - 1.0)),
        )

    def _focus_border(self, focused: bool) -> None:
        self.canvas.itemconfigure(
            self._img_border,
            outline="#ffd060" if focused else "#334466",
            width=2 if focused else 1,
        )

    def _canvas_to_image_coords(self, canvas_x: float, canvas_y: float) -> tuple[float, float] | None:
        if self._last_frame is None:
            return None

        img_h, img_w = self._last_frame.shape[:2]
        self._ensure_view_center(img_w, img_h)

        x1, y1, x2, y2 = self._image_area_rect()
        aw = max(1.0, x2 - x1)
        ah = max(1.0, y2 - y1)
        fit_scale = min(aw / max(1.0, img_w), ah / max(1.0, img_h))
        scale = max(1e-6, fit_scale * self._zoom)

        local_x = canvas_x - x1
        local_y = canvas_y - y1

        img_x = self._view_cx + (local_x - aw / 2.0) / scale
        img_y = self._view_cy + (local_y - ah / 2.0) / scale
        return float(img_x), float(img_y)

    def _update_coords_overlay(self, canvas_x: float | None = None,
                               canvas_y: float | None = None) -> None:
        if not self._show_coords_var.get() or not self._img_hover:
            self._set_coords_overlay_text("")
            self._last_coords_value = None
            self._coords_copied = False
            return

        if canvas_x is None or canvas_y is None:
            self._set_coords_overlay_text("")
            self._last_coords_value = None
            self._coords_copied = False
            return

        mapped = self._canvas_to_image_coords(canvas_x, canvas_y)
        if mapped is None:
            self._set_coords_overlay_text("")
            self._last_coords_value = None
            self._coords_copied = False
            return

        img_x, img_y = mapped
        if self._last_frame is not None:
            img_h, img_w = self._last_frame.shape[:2]
            if (img_x < -0.5 or img_x > (img_w - 0.5)
                    or img_y < -0.5 or img_y > (img_h - 0.5)):
                self._set_coords_overlay_text("")
                self._last_coords_value = None
                self._coords_copied = False
                return

        self._last_coords_value = f"{img_x:.3f}, {img_y:.3f}"
        copied_suffix = " (copied)" if self._coords_copied else ""
        self._set_coords_overlay_text(
            f"({self._last_coords_value}){copied_suffix}",
        )

    def _set_coords_overlay_text(self, text: str) -> None:
        if text == self._coords_overlay_text:
            return
        self._coords_overlay_text = text
        self.canvas.itemconfigure(self._coords_text, text=text)

    def _schedule_coords_overlay_update(self, canvas_x: float | None,
                                        canvas_y: float | None) -> None:
        self._pending_coords_overlay = (canvas_x, canvas_y)
        if self._coords_update_scheduled:
            return
        self._coords_update_scheduled = True

        def _flush() -> None:
            self._coords_update_scheduled = False
            pending = self._pending_coords_overlay
            self._pending_coords_overlay = None
            if pending is None:
                return
            try:
                self._update_coords_overlay(*pending)
            except tk.TclError:
                pass

        self.canvas.after_idle(_flush)

    def _on_copy_coords_hotkey(self, _event) -> str | None:
        if not self._show_coords_var.get() or not self._img_hover:
            return None
        if not self._last_coords_value:
            return None

        try:
            self.canvas.clipboard_clear()
            self.canvas.clipboard_append(self._last_coords_value)
        except tk.TclError:
            return None

        self._coords_copied = True
        self._set_coords_overlay_text(
            f"({self._last_coords_value}) (copied)",
        )

        # Also publish picked image coordinates for downstream array processing.
        try:
            sx, sy = (part.strip() for part in self._last_coords_value.split(",", 1))
            coord = np.array([[float(sx), float(sy)]], dtype=np.float32)
            self.push_output({"image_coords": coord})
        except Exception:
            pass

        return "break"

    def _pointer_on_this_node(self) -> bool:
        px, py = self.canvas.winfo_pointerxy()
        cx = self.canvas.canvasx(px - self.canvas.winfo_rootx())
        cy = self.canvas.canvasy(py - self.canvas.winfo_rooty())
        hit = self.canvas.find_overlapping(cx - 1, cy - 1, cx + 1, cy + 1)
        return any(self.node_id in self.canvas.gettags(item) for item in hit)

    def _close_help_popup(self) -> None:
        if self._help_popup is not None and self._help_popup.winfo_exists():
            self._help_popup.destroy()
        self._help_popup = None

    def _show_help_popup(self) -> None:
        self._close_help_popup()

        popup = tk.Toplevel(self.canvas)
        popup.title("Image Display Help")
        popup.transient(self.canvas.winfo_toplevel())
        popup.resizable(False, False)

        px, py = self.canvas.winfo_pointerxy()
        popup.geometry(f"+{px + 14}+{py + 14}")

        help_text = (
            "Image Display Node Usage\n\n"
            "- Wheel: zoom in/out at cursor\n"
            "- '+' / '-': zoom in/out (same as wheel up/down)\n"
            "- Left-drag: pan image\n"
            "- Right-click: interpolation menu\n"
            "- Toggle FPS/Coords/Grid checkboxes for overlays\n"
            "- Press C or c (while cursor is over image and Coords is ON) to copy coordinates\n"
            "- Press Ctrl-H to show this help"
        )

        body = tk.Frame(popup, bg="#111111", padx=10, pady=8)
        body.pack(fill="both", expand=True)

        tk.Label(
            body,
            text=help_text,
            justify="left",
            anchor="w",
            bg="#111111",
            fg="#dddddd",
            font=("Arial", 9),
        ).pack(fill="both", expand=True)

        tk.Button(body, text="Close", command=self._close_help_popup).pack(anchor="e", pady=(8, 0))

        popup.bind("<Escape>", lambda _e: self._close_help_popup())
        popup.protocol("WM_DELETE_WINDOW", self._close_help_popup)

        self._help_popup = popup

    def _on_help_hotkey(self, _event) -> str | None:
        if not self._pointer_on_this_node():
            return None
        self._show_help_popup()
        return "break"

    def _zoom_about_canvas_point(self, canvas_x: float, canvas_y: float,
                                 zoom_factor: float) -> None:
        if self._last_frame is None:
            return

        # Canvas mouse events report viewport-relative coordinates.
        # Convert them to the scrolled canvas coordinate space so the
        # image point under the cursor remains fixed while zooming.
        canvas_x = float(self.canvas.canvasx(canvas_x))
        canvas_y = float(self.canvas.canvasy(canvas_y))

        img_h, img_w = self._last_frame.shape[:2]
        self._ensure_view_center(img_w, img_h)

        x1, y1, x2, y2 = self._image_area_rect()
        aw = max(1.0, x2 - x1)
        ah = max(1.0, y2 - y1)
        fit_scale = min(aw / max(1.0, img_w), ah / max(1.0, img_h))

        old_zoom = self._zoom
        new_zoom = float(np.clip(old_zoom * zoom_factor,
                                 self._min_zoom, self._max_zoom))
        if abs(new_zoom - old_zoom) < 1e-9:
            return

        old_scale = max(1e-6, fit_scale * old_zoom)
        new_scale = max(1e-6, fit_scale * new_zoom)

        local_x = canvas_x - x1
        local_y = canvas_y - y1

        img_x = self._view_cx + (local_x - aw / 2.0) / old_scale
        img_y = self._view_cy + (local_y - ah / 2.0) / old_scale

        self._zoom = new_zoom
        self._view_cx, self._view_cy = self._clamp_view_center(
            img_x - (local_x - aw / 2.0) / new_scale,
            img_y - (local_y - ah / 2.0) / new_scale,
            img_w,
            img_h,
        )

        self._display_frame(self._last_frame)
        self._update_coords_overlay(canvas_x, canvas_y)

    def _on_img_enter(self, event) -> str:
        self.canvas.focus_set()
        self._img_hover = True
        self._HOVERED_BY_CANVAS[self.canvas] = self
        self._focus_border(True)
        self._schedule_coords_overlay_update(event.x, event.y)
        return "break"

    def _on_img_leave(self, _event) -> str:
        self._img_hover = False
        if self._HOVERED_BY_CANVAS.get(self.canvas) is self:
            self._HOVERED_BY_CANVAS.pop(self.canvas, None)
        self._is_panning = False
        self._pan_anchor_canvas = None
        self._pan_anchor_center = None
        self._focus_border(False)
        self._schedule_coords_overlay_update(None, None)
        return "break"

    def _on_img_motion(self, event) -> str:
        self._coords_copied = False
        self._schedule_coords_overlay_update(event.x, event.y)
        return "break"

    def _on_img_mousewheel(self, event) -> str:
        if not self._img_hover:
            return "break"
        factor = 1.12 if event.delta > 0 else 1 / 1.12
        self._zoom_about_canvas_point(event.x, event.y, factor)
        return "break"

    def _on_img_wheel_linux(self, event) -> str:
        if not self._img_hover:
            return "break"
        factor = 1.12 if event.num == 4 else 1 / 1.12
        self._zoom_about_canvas_point(event.x, event.y, factor)
        return "break"

    def _on_canvas_mousewheel(self, event):
        if not self._img_hover:
            return
        return self._on_img_mousewheel(event)

    def _on_canvas_wheel_linux(self, event):
        if not self._img_hover:
            return
        return self._on_img_wheel_linux(event)

    def _zoom_by_factor_at_pointer(self, zoom_factor: float) -> bool:
        if not self._img_hover or self._last_frame is None:
            return False
        px, py = self.canvas.winfo_pointerxy()
        ex = px - self.canvas.winfo_rootx()
        ey = py - self.canvas.winfo_rooty()
        self._zoom_about_canvas_point(ex, ey, zoom_factor)
        return True

    def _on_zoom_in_hotkey(self, _event) -> str | None:
        if not self._zoom_by_factor_at_pointer(1.12):
            return None
        return "break"

    def _on_zoom_out_hotkey(self, _event) -> str | None:
        if not self._zoom_by_factor_at_pointer(1 / 1.12):
            return None
        return "break"

    def _on_img_press(self, event) -> str:
        self._is_panning = self._img_hover and self._last_frame is not None
        self._HOVERED_BY_CANVAS[self.canvas] = self
        if self._is_panning:
            self._pan_anchor_canvas = (event.x, event.y)
            self._pan_anchor_center = (float(self._view_cx), float(self._view_cy))
        self._schedule_coords_overlay_update(event.x, event.y)
        return "break"

    def _schedule_pan_redraw(self, canvas_x: float, canvas_y: float) -> None:
        self._pending_pan_overlay = (canvas_x, canvas_y)
        if self._pan_redraw_scheduled:
            return
        self._pan_redraw_scheduled = True

        def _flush() -> None:
            self._pan_redraw_scheduled = False
            overlay = self._pending_pan_overlay
            self._pending_pan_overlay = None
            if self._last_frame is None:
                return
            try:
                self._display_frame(self._last_frame)
                if overlay is not None:
                    self._update_coords_overlay(*overlay)
            except tk.TclError:
                pass

        self.canvas.after_idle(_flush)

    def _on_img_drag(self, event) -> str:
        self._coords_copied = False
        if not self._is_panning or self._last_frame is None:
            self._update_coords_overlay(event.x, event.y)
            return "break"

        img_h, img_w = self._last_frame.shape[:2]
        self._ensure_view_center(img_w, img_h)

        x1, y1, x2, y2 = self._image_area_rect()
        aw = max(1.0, x2 - x1)
        ah = max(1.0, y2 - y1)
        fit_scale = min(aw / max(1.0, img_w), ah / max(1.0, img_h))
        scale = max(1e-6, fit_scale * self._zoom)

        if self._pan_anchor_canvas is None or self._pan_anchor_center is None:
            self._is_panning = False
            return "break"

        ax, ay = self._pan_anchor_canvas
        base_cx, base_cy = self._pan_anchor_center

        dx = event.x - ax
        dy = event.y - ay

        next_cx, next_cy = self._clamp_view_center(
            base_cx - dx / scale,
            base_cy - dy / scale,
            img_w,
            img_h,
        )

        # If we are already clamped at the boundary, skip expensive redraw.
        if (self._view_cx is not None and self._view_cy is not None
                and abs(next_cx - self._view_cx) < 1e-6
                and abs(next_cy - self._view_cy) < 1e-6):
            self._update_coords_overlay(event.x, event.y)
            return "break"

        self._view_cx = next_cx
        self._view_cy = next_cy

        self._schedule_pan_redraw(event.x, event.y)
        return "break"

    def _on_img_release(self, event) -> str:
        self._is_panning = False
        self._pan_anchor_canvas = None
        self._pan_anchor_center = None
        if self._pan_redraw_scheduled:
            self._pending_pan_overlay = (event.x, event.y)
        else:
            self._schedule_coords_overlay_update(event.x, event.y)
        return "break"

    def _on_img_right_click(self, event) -> str:
        self._show_interpolation_menu(event.x_root, event.y_root)
        return "break"

    def on_resize(self, old_width: int, old_height: int,
                  new_width: int, new_height: int) -> None:
        super().on_resize(old_width, old_height, new_width, new_height)

        x, y, w, h = self.x, self.y, self.width, self.height
        self._img_w = max(1, w - 8)
        self._img_h = max(1, h - 50)

        self.canvas.coords(self._img_border, x+4, y+22, x+w-4, y+22+self._img_h)
        self.canvas.coords(self._placeholder, x+4, y+22, x+w-4, y+22+self._img_h)
        self.canvas.coords(self._placeholder_text, x+w/2, y+22+self._img_h/2)

        self.canvas.coords(self._fps_text, x+w-10, y+26)
        self.canvas.coords(self._coords_text, x+8, y+26)

        if self._last_frame is not None:
            self._display_frame(self._last_frame)

    def on_destroy(self) -> None:
        if self._HOVERED_BY_CANVAS.get(self.canvas) is self:
            self._HOVERED_BY_CANVAS.pop(self.canvas, None)
        self._close_help_popup()
        self._clear_grid_overlay()
        self._clear_point_overlay()
        super().on_destroy()

    # ── serialization ─────────────────────────────────────────────

    def build_inspector(self, parent: tk.Frame) -> None:
        tk.Label(parent, text="Point symbols", font=("Arial", 10, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )
        tk.Label(parent, text="Marker", font=("Arial", 9)).grid(
            row=1, column=0, sticky="w", pady=2
        )
        marker_menu = ttk.Combobox(
            parent, textvariable=self._marker_var, values=("+", "*", "o"),
            state="readonly", width=8, font=("Arial", 9),
        )
        marker_menu.grid(row=1, column=1, sticky="w", pady=2)

        tk.Label(parent, text="Color (R, G, B)", font=("Arial", 9)).grid(
            row=2, column=0, sticky="w", pady=2
        )
        color_row = tk.Frame(parent)
        color_row.grid(row=2, column=1, sticky="w", pady=2)
        for variable in (self._marker_red_var, self._marker_green_var, self._marker_blue_var):
            tk.Spinbox(
                color_row, from_=0, to=255, textvariable=variable,
                width=4, font=("Arial", 9), justify="center",
            ).pack(side="left", padx=(0, 3))

        tk.Label(parent, text="Line width", font=("Arial", 9)).grid(
            row=3, column=0, sticky="w", pady=2
        )
        tk.Spinbox(
            parent, from_=1, to=20, textvariable=self._marker_width_var,
            width=5, font=("Arial", 9), justify="center",
        ).grid(row=3, column=1, sticky="w", pady=2)

        tk.Label(
            parent,
            text="Points must be an N x 2, N x 3, or N x 4 array.\n"
                 "Columns: x, y, [window size] or [window width, height].",
            justify="left", anchor="w", font=("Arial", 8), fg="#555555",
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(10, 0))

        marker_menu.bind("<<ComboboxSelected>>", self._on_marker_style_changed)
        for variable in (
            self._marker_red_var, self._marker_green_var,
            self._marker_blue_var, self._marker_width_var,
        ):
            variable.trace_add("write", self._on_marker_style_changed)

    def _on_marker_style_changed(self, *_args) -> None:
        if self._last_frame is not None:
            self._display_frame(self._last_frame)

    def get_params(self) -> dict:
        return {
            "show_fps": self._show_fps_var.get(),
            "show_coords": self._show_coords_var.get(),
            "show_grid": self._show_grid_var.get(),
            "interpolation": self._interp_var.get(),
            "marker": self._marker_var.get(),
            "marker_color": [
                self._marker_red_var.get(), self._marker_green_var.get(),
                self._marker_blue_var.get(),
            ],
            "marker_width": self._marker_width_var.get(),
        }

    def set_params(self, params: dict) -> None:
        self._show_fps_var.set(params.get("show_fps", True))
        self._show_coords_var.set(params.get("show_coords", False))
        self._show_grid_var.set(params.get("show_grid", False))
        interp = str(params.get("interpolation", "NEAREST")).upper()
        self._interp_var.set(interp if interp in self._INTERPOLATIONS else "NEAREST")
        marker = str(params.get("marker", "+"))
        self._marker_var.set(marker if marker in ("+", "*", "o") else "+")
        color = params.get("marker_color", (0, 255, 0))
        try:
            red, green, blue = (max(0, min(255, int(value))) for value in color)
        except (TypeError, ValueError):
            red, green, blue = 0, 255, 0
        self._marker_red_var.set(red)
        self._marker_green_var.set(green)
        self._marker_blue_var.set(blue)
        try:
            width = int(params.get("marker_width", 2))
        except (TypeError, ValueError):
            width = 2
        self._marker_width_var.set(max(1, min(20, width)))



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
