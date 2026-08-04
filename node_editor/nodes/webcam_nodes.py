# node_editor/nodes/webcam_nodes.py

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


class WebcamNode(BaseNode):
    """
    Captures frames from a camera device (cv2.VideoCapture).
    Runs in STREAMING mode: a dedicated thread reads frames
    and pushes them downstream via push_output().

    Supports:
      - Camera index selector (0, 1, 2 ...)
      - Resolution selector
      - Target FPS control
      - Start / Stop button
      - Live FPS display
    """
    EXECUTION_MODE = ExecutionMode.STREAMING
    NODE_TYPE      = "webcam"
    DISPLAY_NAME   = "Webcam"
    CATEGORY       = "source"
    NODE_WIDTH     = 250
    NODE_HEIGHT    = 155
    # Pin labels are drawn by NodeEditorApp. Use a light color because this
    # node's dark body makes the default dark output-label text unreadable.
    OUTPUT_PIN_LABEL_COLOR = "#d8efff"

    _RESOLUTIONS = [
        ("320 x 240",   320,  240),
        ("640 x 480",   640,  480),
        ("1280 x 720",  1280, 720),
        ("1920 x 1080", 1920, 1080),
        ("3840 x 2160", 3840, 2160),
    ]
    _DEFAULT_RES_IDX = 1   # 640 x 480

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[],
            outputs=[PinDef("image", PinType.IMAGE, "frame",
                            shape=(-1, -1, 3), dtype="uint8")]
        )

    def build_body(self) -> None:
        x, y, w, h = self.x, self.y, self.width, self.height

        if not hasattr(self, "_cam_idx_var"):
            self._cam_idx_var = tk.IntVar(value=0)
            self._res_var = tk.StringVar(value=self._RESOLUTIONS[self._DEFAULT_RES_IDX][0])
            self._fps_var = tk.IntVar(value=30)
            self._btn_var = tk.StringVar(value="Start")
            self._status_var = tk.StringVar(value="stopped")
            self._btn = None

        # Compact canvas shell (Phase 2).
        self._body_rect = self.canvas.create_rectangle(
            x, y, x+w, y+h,
            fill="#1a1a2e", outline="#4488cc", width=2,
            tags=(self.node_id, "node_body"))
        self._title_item = self.canvas.create_text(
            x+w/2, y+13,
            text=self.DISPLAY_NAME,
            font=("Arial", 9, "bold"), fill="#88ccff",
            tags=(self.node_id,))

        status_lbl = tk.Label(
            self.canvas, textvariable=self._status_var,
            font=("Arial", 8), bg="#1a1a2e", fg="#55ff88")
        self.canvas.create_window(
            x+w/2, y+h-12, window=status_lbl,
            tags=(self.node_id,))

        self._canvas_items += [self._body_rect, self._title_item]

        # ── internal state ────────────────────────────────────────
        self._cap:        cv2.VideoCapture | None = None
        self._thread:     threading.Thread | None = None
        self._stop_event: threading.Event         = threading.Event()
        self._is_running: bool                    = False
        self._frame_times: list[float]            = []

    def build_inspector(self, parent: tk.Frame) -> None:
        tk.Label(parent, text="Camera:", font=("Arial", 9)).grid(row=0, column=0, sticky="w", padx=(0, 8), pady=2)
        cam_sb = tk.Spinbox(parent, from_=0, to=9, textvariable=self._cam_idx_var, width=4, font=("Arial", 9))
        cam_sb.grid(row=0, column=1, sticky="w", pady=2)

        tk.Label(parent, text="Resolution:", font=("Arial", 9)).grid(row=1, column=0, sticky="w", padx=(0, 8), pady=2)
        res_cb = ttk.Combobox(parent, textvariable=self._res_var, values=[r[0] for r in self._RESOLUTIONS],
                              width=14, state="readonly", font=("Arial", 9))
        res_cb.grid(row=1, column=1, sticky="ew", pady=2)

        tk.Label(parent, text="Target FPS:", font=("Arial", 9)).grid(row=2, column=0, sticky="w", padx=(0, 8), pady=2)
        fps_sb = tk.Spinbox(parent, from_=1, to=60, textvariable=self._fps_var, width=5, font=("Arial", 9))
        fps_sb.grid(row=2, column=1, sticky="w", pady=2)

        self._btn = tk.Button(parent, textvariable=self._btn_var, font=("Arial", 9, "bold"),
                              bg="#226688", fg="white", activebackground="#338899",
                              command=self._toggle_stream, relief=tk.FLAT, padx=8)
        self._btn.grid(row=3, column=0, columnspan=2, pady=(8, 4), sticky="w")

        status_lbl = tk.Label(parent, textvariable=self._status_var, font=("Arial", 9), fg="#226622")
        status_lbl.grid(row=4, column=0, columnspan=2, sticky="w")

        parent.grid_columnconfigure(1, weight=1)

    def close_inspector(self) -> None:
        super().close_inspector()
        self._btn = None

    # ── stream control ────────────────────────────────────────────

    def _toggle_stream(self) -> None:
        if self._is_running:
            self.stop_stream()
        else:
            self.start_stream()

    def start_stream(self) -> None:
        if self._is_running:
            return

        idx    = self._cam_idx_var.get()
        res    = self._res_var.get()
        rentry = next((r for r in self._RESOLUTIONS
                       if r[0] == res),
                      self._RESOLUTIONS[self._DEFAULT_RES_IDX])

        system_name = platform.system().lower()
        if system_name.startswith("win"):
            # Windows: DirectShow backend is often needed for BRIO 4K.
            self._cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        elif system_name.startswith("linux"):
            # Linux: V4L2 backend is typically required for high-res webcam modes.
            self._cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        else:
            # macOS and other platforms.
            self._cap = cv2.VideoCapture(idx)

        if not self._cap.isOpened():
            self._status_var.set(f"cannot open cam {idx}")
            return

        # Request MJPG first; many webcams need this for 4K/30 capture modes.
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self._cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  rentry[1])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, rentry[2])
        self._cap.set(cv2.CAP_PROP_FPS, self._fps_var.get())

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)

        req_w, req_h = int(rentry[1]), int(rentry[2])
        if actual_w != req_w or actual_h != req_h:
            self._status_var.set(
                f"requested {req_w}x{req_h}, got {actual_w}x{actual_h} @ {actual_fps:.1f}fps"
            )
        else:
            self._status_var.set(
                f"{actual_w}x{actual_h} @ {actual_fps:.1f}fps"
            )

        self._stop_event.clear()
        self._is_running = True
        self._btn_var.set("Stop")
        if self._btn is not None and self._btn.winfo_exists():
            self._btn.config(bg="#882222")
        if "requested" not in self._status_var.get():
            self._status_var.set("starting...")
        self._frame_times = []

        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop_stream(self) -> None:
        self._stop_event.set()
        self._is_running = False
        self._btn_var.set("Start")
        if self._btn is not None and self._btn.winfo_exists():
            self._btn.config(bg="#226688")
        self._status_var.set("stopped")
        if self._cap:
            self._cap.release()
            self._cap = None

    def _capture_loop(self) -> None:
        """
        Runs in a worker thread.
        Reads frames from the camera and calls push_output().
        Throttles to the target FPS using a sleep.
        """
        target_fps      = max(1, self._fps_var.get())
        frame_interval  = 1.0 / target_fps

        while not self._stop_event.is_set():
            t0 = time.perf_counter()

            if self._cap is None or not self._cap.isOpened():
                break

            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            # BGR → RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # track real FPS
            now = time.perf_counter()
            self._frame_times.append(now)
            self._frame_times = [
                t for t in self._frame_times if now - t < 2.0]
            real_fps = len(self._frame_times) / 2.0

            # push to downstream (thread-safe via Engine queue)
            self.push_output({
                "image": frame_rgb,
                "_fps":  real_fps,       # metadata, not a pin
            })

            # throttle
            elapsed = time.perf_counter() - t0
            sleep   = frame_interval - elapsed
            if sleep > 0:
                time.sleep(sleep)

        # update UI from main thread
        self.canvas.after(
            0, lambda: self._status_var.set("stopped"))

    def _update_fps_display(self, fps: float) -> None:
        """Called from main thread by VideoPlayOutputNode or Engine."""
        self._status_var.set(f"{fps:.1f} fps")

    # ── compute (not used in STREAMING) ──────────────────────────

    def compute(self, inputs: dict) -> dict:
        return {}

    # ── serialization ─────────────────────────────────────────────

    def get_params(self) -> dict:
        return {
            "cam_idx":    self._cam_idx_var.get(),
            "resolution": self._res_var.get(),
            "fps":        self._fps_var.get(),
        }

    def set_params(self, params: dict) -> None:
        self._cam_idx_var.set(params.get("cam_idx",    0))
        self._res_var.set(    params.get("resolution",
                              self._RESOLUTIONS[self._DEFAULT_RES_IDX][0]))
        self._fps_var.set(    params.get("fps", 30))

    def on_destroy(self) -> None:
        self.stop_stream()
        super().on_destroy()


# ─────────────────────────────────────────────────────────────────────────────


