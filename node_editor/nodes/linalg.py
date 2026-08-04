import tkinter as tk

import numpy as np

from node_editor.base_node import BaseNode
from node_editor.pin_types import PinSchema, PinDef, PinType
from node_editor.execution import ExecutionMode


class MatMulNode(BaseNode):
    """
    Matrix multiplication node.

    Computes: out = a @ b
    """

    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE = "matmul"
    DISPLAY_NAME = "matmul"
    CATEGORY = "process"
    NODE_WIDTH = 180
    NODE_HEIGHT = 90

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[
                PinDef("a", PinType.ARRAY, "a"),
                PinDef("b", PinType.ARRAY, "b"),
            ],
            outputs=[
                PinDef("out", PinType.ARRAY, "out"),
            ],
        )

    def build_body(self) -> None:
        x, y, w, h = self.x, self.y, self.width, self.height

        self._body_rect = self.canvas.create_rectangle(
            x, y, x + w, y + h,
            fill="#f8f3e6", outline="#b39b4a", width=2,
            tags=(self.node_id, "node_body"),
        )
        self._title_item = self.canvas.create_text(
            x + w / 2, y + 13,
            text=self.DISPLAY_NAME,
            font=("Arial", 9, "bold"), fill="#6b5a26",
            tags=(self.node_id,),
        )

        self._status_var = tk.StringVar(value="waiting for a,b")
        status_lbl = tk.Label(
            self.canvas,
            textvariable=self._status_var,
            font=("Arial", 7), bg="#f8f3e6", fg="#6b5a26",
        )
        self.canvas.create_window(
            x + w / 2, y + h - 12, window=status_lbl,
            tags=(self.node_id,),
        )

        self._canvas_items += [self._body_rect, self._title_item]

    def compute(self, inputs: dict) -> dict:
        a = inputs.get("a")
        b = inputs.get("b")
        if a is None or b is None:
            self._status_var.set("missing a or b")
            self.set_status("missing", "#cc6666")
            return {}

        try:
            aa = np.asarray(a)
            bb = np.asarray(b)
            out = aa @ bb
            self._status_var.set(f"ok: {tuple(out.shape)}")
            self.set_status("ok", "#55aa55")
            return {"out": out}
        except Exception as e:
            self._status_var.set(f"error: {e}")
            self.set_status("error", "#cc0000")
            return {}


class InvNode(BaseNode):
    """
    Matrix inversion node.

    Computes: out = inv(a)
    """

    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE = "inv"
    DISPLAY_NAME = "inv"
    CATEGORY = "process"
    NODE_WIDTH = 160
    NODE_HEIGHT = 90

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[
                PinDef("a", PinType.ARRAY, "a"),
            ],
            outputs=[
                PinDef("out", PinType.ARRAY, "out"),
            ],
        )

    def build_body(self) -> None:
        x, y, w, h = self.x, self.y, self.width, self.height

        self._body_rect = self.canvas.create_rectangle(
            x, y, x + w, y + h,
            fill="#eaf2ff", outline="#4d84d4", width=2,
            tags=(self.node_id, "node_body"),
        )
        self._title_item = self.canvas.create_text(
            x + w / 2, y + 13,
            text=self.DISPLAY_NAME,
            font=("Arial", 9, "bold"), fill="#2e5c93",
            tags=(self.node_id,),
        )

        self._status_var = tk.StringVar(value="waiting for a")
        status_lbl = tk.Label(
            self.canvas,
            textvariable=self._status_var,
            font=("Arial", 7), bg="#eaf2ff", fg="#3d678f",
        )
        self.canvas.create_window(
            x + w / 2, y + h - 12, window=status_lbl,
            tags=(self.node_id,),
        )

        self._canvas_items += [self._body_rect, self._title_item]

    def compute(self, inputs: dict) -> dict:
        a = inputs.get("a")
        if a is None:
            self._status_var.set("missing a")
            self.set_status("missing", "#cc6666")
            return {}

        try:
            aa = np.asarray(a, dtype=np.float64)
            out = np.linalg.inv(aa)
            self._status_var.set(f"ok: {tuple(out.shape)}")
            self.set_status("ok", "#55aa55")
            return {"out": out}
        except Exception as e:
            self._status_var.set(f"error: {e}")
            self.set_status("error", "#cc0000")
            return {}
