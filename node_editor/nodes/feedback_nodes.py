import tkinter as tk

import numpy as np

from node_editor.base_node import BaseNode
from node_editor.execution import ExecutionMode
from node_editor.pin_types import PinDef, PinSchema, PinType


class ArrayFeedbackNode(BaseNode):
    """
    Breaks a graph cycle by treating the 'current' input as delayed state.

    The node outputs:
      - the init array on the first pass after reset
      - the previous iteration's current array on later passes

    This allows iterative pipelines like:
      init points -> Feedback.init
      OpticalFlow.nextPts -> Feedback.current
      Feedback.next -> OpticalFlow.nextPts

    The link into 'current' is read from the engine's cached upstream outputs,
    so it represents the previous pass instead of a same-pass dependency.
    """

    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE = "array_feedback"
    DISPLAY_NAME = "Array Feedback"
    CATEGORY = "process"
    SEARCH_KEYWORDS = ("feedback", "loop", "iter", "previous", "delay")
    DELAYED_INPUT_PINS = ("current",)
    NODE_WIDTH = 220
    NODE_HEIGHT = 120

    HELP_TEXT = (
        "Array Feedback Node\n\n"
        "Purpose:\n"
        "- Feed an ARRAY result from one iteration into the next iteration\n"
        "  without creating a hard graph cycle.\n\n"
        "Pins:\n"
        "- init [ARRAY]: initial array used on the first pass after reset.\n"
        "- current [ARRAY]: delayed feedback input, usually connected from the\n"
        "  previous node's output (for example Optical Flow nextPts).\n"
        "- reset [TRIGGER]: rising edge forces the next pass to use init again.\n"
        "- next [ARRAY]: array to use on the current iteration.\n\n"
        "Notes:\n"
        "- The current input is delayed by one pass.\n"
        "- Click Reset or send a reset pulse to force the next pass to use init again.\n"
        "- If init changes, the node auto-resets to the new init array."
    )

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[
                PinDef("init", PinType.ARRAY, "init"),
                PinDef("current", PinType.ARRAY, "cur", optional=True),
                PinDef("reset", PinType.TRIGGER, "rst", optional=True),
            ],
            outputs=[
                PinDef("next", PinType.ARRAY, "next"),
            ],
        )

    def _init_state(self) -> None:
        if hasattr(self, "_status_var"):
            return
        self._status_var = tk.StringVar(value="reset -> init")
        self._next_source = "init"
        self._reset_requested = True
        self._reset_latched_high = False
        self._last_output: np.ndarray | None = None
        self._last_init_snapshot: np.ndarray | None = None

    def build_body(self) -> None:
        self._init_state()
        x, y, w, h = self.x, self.y, self.width, self.height

        self._body_rect = self.canvas.create_rectangle(
            x, y, x + w, y + h,
            fill="#f3efe4", outline="#8d6f32", width=2,
            tags=(self.node_id, "node_body"),
        )
        self._title_item = self.canvas.create_text(
            x + w / 2, y + 13,
            text=self.DISPLAY_NAME,
            font=("Arial", 9, "bold"), fill="#6e531f",
            tags=(self.node_id,),
        )

        reset_btn = tk.Button(
            self.canvas,
            text="Reset",
            font=("Arial", 8),
            command=self._request_reset,
            padx=8,
            pady=1,
        )
        self.canvas.create_window(
            x + w / 2, y + 44,
            window=reset_btn,
            tags=(self.node_id,),
        )

        hint_lbl = tk.Label(
            self.canvas,
            text="init first, then delayed current",
            font=("Arial", 7), bg="#f3efe4", fg="#7a6330",
        )
        self.canvas.create_window(
            x + w / 2, y + 72,
            window=hint_lbl,
            tags=(self.node_id,),
        )

        status_lbl = tk.Label(
            self.canvas,
            textvariable=self._status_var,
            font=("Arial", 7), bg="#f3efe4", fg="#7a6330",
        )
        self.canvas.create_window(
            x + w / 2, y + h - 12,
            window=status_lbl,
            tags=(self.node_id,),
        )

        self._canvas_items += [self._body_rect, self._title_item]

    def build_inspector(self, parent: tk.Frame) -> None:
        self._init_state()

        tk.Label(
            parent,
            text="Use init on the next pass, then continue with delayed current values.",
            font=("Arial", 9),
            justify="left",
            anchor="w",
            wraplength=320,
        ).pack(fill="x")

        tk.Button(parent, text="Reset To Init", command=self._request_reset).pack(
            anchor="w", pady=(10, 6)
        )

        tk.Label(
            parent,
            textvariable=self._status_var,
            font=("Arial", 9),
            fg="#6e531f",
            anchor="w",
            justify="left",
        ).pack(fill="x")

    def get_help_text(self) -> str:
        return self.HELP_TEXT

    def _request_reset(self) -> None:
        self._init_state()
        self._reset_requested = True
        self._next_source = "init"
        self._status_var.set("reset requested -> init")
        self.set_status("reset", "#8d6f32")

    def on_upstream_changed(self) -> None:
        self._request_reset()

    @staticmethod
    def _coerce_array(value, pin_name: str) -> np.ndarray:
        arr = np.asarray(value)
        if arr.ndim == 0:
            raise ValueError(f"{pin_name} must be an array")
        return arr

    @staticmethod
    def _arrays_equal(left: np.ndarray | None, right: np.ndarray | None) -> bool:
        if left is None or right is None:
            return left is right
        return left.shape == right.shape and left.dtype == right.dtype and np.array_equal(left, right)

    @staticmethod
    def _shape_text(arr: np.ndarray) -> str:
        return "x".join(str(dim) for dim in arr.shape)

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

    def compute(self, inputs: dict) -> dict:
        self._init_state()

        raw_init = inputs.get("init")
        raw_current = inputs.get("current")
        raw_reset = inputs.get("reset")

        init_arr = None if raw_init is None else self._coerce_array(raw_init, "init")
        current_arr = None if raw_current is None else self._coerce_array(raw_current, "current")
        reset_now = False if raw_reset is None else self._coerce_bool_like(raw_reset)

        if reset_now and not self._reset_latched_high:
            self._request_reset()
            self._reset_latched_high = True
        elif not reset_now:
            self._reset_latched_high = False

        if init_arr is not None:
            init_snapshot = np.array(init_arr, copy=True)
            if not self._arrays_equal(self._last_init_snapshot, init_snapshot):
                self._last_init_snapshot = init_snapshot
                self._reset_requested = True

        if self._reset_requested:
            if init_arr is None:
                self._status_var.set("waiting for init")
                self.set_status("waiting", "#cc6666")
                return {
                    "_skip_downstream": True,
                    "_preserve_cache": True,
                }
            out = np.array(init_arr, copy=True)
            self._reset_requested = False
            self._next_source = "current"
            source = "init"
        elif current_arr is not None:
            out = np.array(current_arr, copy=True)
            source = "current"
        elif self._last_output is not None:
            out = np.array(self._last_output, copy=True)
            source = "latched"
        elif init_arr is not None:
            out = np.array(init_arr, copy=True)
            source = "init"
            self._next_source = "current"
        else:
            self._status_var.set("waiting for init/current")
            self.set_status("waiting", "#cc6666")
            return {
                "_skip_downstream": True,
                "_preserve_cache": True,
            }

        self._last_output = np.array(out, copy=True)
        shape_text = self._shape_text(out)
        self._status_var.set(f"{source}: ({shape_text}) {out.dtype}")
        self.set_status(source, "#4c8a4c" if source != "init" else "#8d6f32")
        return {"next": out}

    def get_params(self) -> dict:
        self._init_state()
        return {
            "reset_requested": bool(self._reset_requested),
        }

    def set_params(self, params: dict) -> None:
        self._init_state()
        self._reset_requested = bool(params.get("reset_requested", True))
        self._reset_latched_high = False
        self._next_source = "init" if self._reset_requested else "current"
        self._status_var.set("reset -> init" if self._reset_requested else "ready")