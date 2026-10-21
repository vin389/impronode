# node_editor/nodes/demo_addition_3_scalars.py
"""Demo: add three scalars (A + B + C) -- the Addition node with a third input."""

from node_editor.base_node import BaseNode
from node_editor.execution import ExecutionMode
from node_editor.pin_types import PinDef, PinSchema, PinType


class AddThreeScalarsNode(BaseNode):
    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE      = "demo_add_3_scalars"
    DISPLAY_NAME   = "Addition of 3 (A+B+C)"
    CATEGORY       = "process"
    SEARCH_KEYWORDS = ("add", "sum", "plus", "three", "3", "scalar", "demo")

    HELP_TEXT = (
        "Addition of 3 (A+B+C)\n\n"
        "Outputs A + B + C. While any of the three inputs is missing, the\n"
        "previous result is kept and nothing downstream is recomputed.\n"
    )

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[PinDef("in1", PinType.SCALAR, "A"),
                    PinDef("in2", PinType.SCALAR, "B"),
                    PinDef("in3", PinType.SCALAR, "C")],
            outputs=[PinDef("result", PinType.SCALAR, "A+B+C")]
        )

    def _init_state(self) -> None:
        if hasattr(self, "_last_result"):
            return
        self._last_result = 0.0

    def build_body(self) -> None:
        self._init_state()
        x, y, w, h = self.x, self.y, self.width, self.height
        self._body_rect = self.canvas.create_rectangle(
            x, y, x+w, y+h, fill="#ddeeff",
            outline="#6699cc", tags=(self.node_id, "node_body"))
        self._title_item = self.canvas.create_text(
            x+w/2, y+14, text=self.DISPLAY_NAME,
            font=("Arial", 9, "bold"), tags=(self.node_id,))
        self._val_text = self.canvas.create_text(
            x+w/2, y+50, text=f"{self._last_result:.4f}",
            fill="#336699", font=("Arial", 11), tags=(self.node_id,))
        self._canvas_items += [self._body_rect, self._title_item, self._val_text]

    def compute(self, inputs: dict) -> dict:
        self._init_state()

        a = inputs.get("in1")
        b = inputs.get("in2")
        c = inputs.get("in3")
        if a is None or b is None or c is None:
            # Hold previous output and do not trigger downstream recompute.
            return {
                "_skip_downstream": True,
                "_preserve_cache": True,
            }

        result = float(a) + float(b) + float(c)
        self._last_result = result
        self.canvas.itemconfig(self._val_text, text=f"{result:.4f}")
        return {"result": result}

    def get_params(self) -> dict:
        self._init_state()
        return {"last_result": self._last_result}

    def set_params(self, params: dict) -> None:
        self._init_state()
        try:
            self._last_result = float(params.get("last_result", 0.0))
        except Exception:
            self._last_result = 0.0
