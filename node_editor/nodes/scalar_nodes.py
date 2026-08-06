# nodes/scalar_nodes.py

import tkinter as tk
from node_editor.base_node import BaseNode
from node_editor.pin_types import PinSchema, PinDef, PinType
from node_editor.execution import ExecutionMode


class ScalarInputNode(BaseNode):
    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE      = "scalar_input"
    DISPLAY_NAME   = "Numeric Input"
    CATEGORY       = "source"
#    NODE_HEIGHT    = 80

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[],
            outputs=[PinDef(name="value", type=PinType.SCALAR, label="Value")]
        )

    def _init_state(self) -> None:
        if hasattr(self, "_value_var"):
            return
        self._value_var = tk.StringVar(value="1.0")
        self._entry = None
        self._status_var = tk.StringVar(value=self._value_var.get())

    def build_body(self) -> None:
        self._init_state()
        x, y, w, h = self.x, self.y, self.width, self.height
        self._body_rect = self.canvas.create_rectangle(
            x, y, x+w, y+h, fill=self.BODY_COLOR,
            outline="#999", tags=(self.node_id, "node_body"))
        self._title_item = self.canvas.create_text(
            x+w/2, y+14, text=self.DISPLAY_NAME,
            font=("Arial", 9, "bold"), fill=self.TITLE_COLOR,
            tags=(self.node_id,))

        status_lbl = tk.Label(self.canvas, textvariable=self._status_var,
                              font=("Arial", 8), bg=self.BODY_COLOR, fg="#336699")
        self.canvas.create_window(x+w/2, y+h-14, window=status_lbl, tags=(self.node_id,))

        self._canvas_items += [self._body_rect, self._title_item]

    def build_inspector(self, parent: tk.Frame) -> None:
        self._init_state()
        tk.Label(parent, text="Numeric value", font=("Arial", 9)).pack(anchor="w")
        self._entry = tk.Entry(parent, textvariable=self._value_var, width=16, justify="center")
        self._entry.pack(anchor="w", pady=(4, 0))

        def _on_entry_change(_e=None):
            self._status_var.set(self._value_var.get().strip() or "0")
            if self._request_downstream:
                self._request_downstream(self.node_id)

        self._entry.bind("<KeyRelease>", _on_entry_change)
        self._entry.bind("<Return>", _on_entry_change)
        self._status_var.set(self._value_var.get().strip() or "0")

    def close_inspector(self) -> None:
        super().close_inspector()
        self._entry = None

    def compute(self, inputs: dict) -> dict:
        try:
            return {"value": float(self._value_var.get())}
        except ValueError:
            return {"value": 0.0}

    def get_params(self) -> dict:
        return {"value": self._value_var.get()}

    def set_params(self, params: dict) -> None:
        self._init_state()
        self._value_var.set(str(params.get("value", "1.0")))
        self._status_var.set(self._value_var.get().strip() or "0")

#    def on_move(self, dx: int, dy: int) -> None:
#        for item in self.canvas.find_withtag(self.node_id):
#            if self.canvas.type(item) == "window":
#                x, y = self.canvas.coords(item)
#                self.canvas.coords(item, x + dx, y + dy)


class AddNode(BaseNode):
    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE      = "add"
    DISPLAY_NAME   = "Addition (+)"
    CATEGORY       = "process"
#    NODE_HEIGHT    = 80

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[PinDef("in1", PinType.SCALAR, "A"),
                    PinDef("in2", PinType.SCALAR, "B")],
            outputs=[PinDef("result", PinType.SCALAR, "A+B")]
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
        if a is None or b is None:
            # Hold previous output and do not trigger downstream recompute.
            return {
                "_skip_downstream": True,
                "_preserve_cache": True,
            }

        result = float(a) + float(b)
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

        if hasattr(self, "_val_text"):
            self.canvas.itemconfig(self._val_text, text=f"{self._last_result:.4f}")


class ScalarOutputNode(BaseNode):
    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE      = "scalar_output"
    DISPLAY_NAME   = "Scalar Display"
    CATEGORY       = "visualize"
#    NODE_HEIGHT    = 80

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[PinDef("value", PinType.SCALAR, "Value")],
            outputs=[]
        )

    def build_body(self) -> None:
        x, y, w, h = self.x, self.y, self.width, self.height
        self._body_rect = self.canvas.create_rectangle(
            x, y, x+w, y+h, fill="#eeffee",
            outline="#66aa66", tags=(self.node_id, "node_body"))
        self._title_item = self.canvas.create_text(
            x+w/2, y+14, text=self.DISPLAY_NAME,
            font=("Arial", 9, "bold"), tags=(self.node_id,))
        self._lbl = tk.Label(self.canvas, text="---",
                             font=("Arial", 13, "bold"),
                             bg="#eeffee", fg="#226622")
        self.canvas.create_window(x+w/2, y+52, window=self._lbl,
                                  tags=(self.node_id,))
        self._canvas_items.append(self._body_rect)

    def compute(self, inputs: dict) -> dict:
        val = inputs.get("value")
        self._lbl.config(text=f"{val:.4f}" if val is not None else "---")
        return {}


class ScalarAccumulatorNode(BaseNode):
    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE = "scalar_accumulator"
    DISPLAY_NAME = "Scalar Accumulator"
    CATEGORY = "process"
    NODE_WIDTH = 190
    NODE_HEIGHT = 115

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[
                PinDef("value", PinType.SCALAR, "value"),
                PinDef("trigger", PinType.TRIGGER, "trig", optional=True),
                PinDef("reset", PinType.TRIGGER, "reset", optional=True),
            ],
            outputs=[
                PinDef("sum", PinType.SCALAR, "sum"),
                PinDef("count", PinType.SCALAR, "count"),
                PinDef("last", PinType.SCALAR, "last"),
            ],
        )

    def _init_state(self) -> None:
        if hasattr(self, "_sum"):
            return
        self._sum = 0.0
        self._count = 0
        self._last = 0.0
        self._history: list[float] = []
        self._history_text = None
        self._status_var = tk.StringVar(value="sum=0.0  n=0")

    def build_body(self) -> None:
        self._init_state()
        x, y, w, h = self.x, self.y, self.width, self.height

        self._body_rect = self.canvas.create_rectangle(
            x, y, x + w, y + h,
            fill="#fff6d9", outline="#c8a847",
            tags=(self.node_id, "node_body"),
        )
        self._title_item = self.canvas.create_text(
            x + w / 2, y + 14,
            text=self.DISPLAY_NAME,
            font=("Arial", 9, "bold"),
            fill="#6b5316",
            tags=(self.node_id,),
        )

        status_lbl = tk.Label(
            self.canvas,
            textvariable=self._status_var,
            font=("Arial", 8),
            bg="#fff6d9",
            fg="#6b5316",
        )
        self.canvas.create_window(
            x + w / 2,
            y + h - 14,
            window=status_lbl,
            tags=(self.node_id,),
        )

        self._canvas_items += [self._body_rect, self._title_item]

    @staticmethod
    def _truthy(value) -> bool:
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

    def _refresh_status(self) -> None:
        self._status_var.set(f"sum={self._sum:.4f}  n={self._count}")

    def _refresh_history_widget(self) -> None:
        if self._history_text is None or not self._history_text.winfo_exists():
            return
        self._history_text.configure(state=tk.NORMAL)
        self._history_text.delete("1.0", tk.END)
        running = 0.0
        for i, val in enumerate(self._history, start=1):
            running += val
            self._history_text.insert(tk.END, f"{i:04d}: +{val:.6g}  =>  {running:.6g}\n")
        self._history_text.configure(state=tk.DISABLED)

    def _emit_current_state(self) -> None:
        if self._on_output_ready:
            self.push_output({
                "sum": self._sum,
                "count": float(self._count),
                "last": self._last,
            })

    def _clear_history(self) -> None:
        self._sum = 0.0
        self._count = 0
        self._last = 0.0
        self._history.clear()
        self._refresh_status()
        self._refresh_history_widget()
        self.set_status("cleared", "#666666")
        self._emit_current_state()

    def build_inspector(self, parent: tk.Frame) -> None:
        self._init_state()

        tk.Label(parent, text="Accumulation history", font=("Arial", 9, "bold")).pack(anchor="w")

        frame = tk.Frame(parent, bd=1, relief=tk.SUNKEN)
        frame.pack(fill="both", expand=True, pady=(6, 4))

        self._history_text = tk.Text(frame, width=46, height=12, font=("Courier", 9), wrap=tk.NONE)
        vsb = tk.Scrollbar(frame, orient="vertical", command=self._history_text.yview)
        self._history_text.configure(yscrollcommand=vsb.set)

        self._history_text.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        row = tk.Frame(parent)
        row.pack(fill="x", pady=(2, 0))
        tk.Button(row, text="Clear History", command=self._clear_history).pack(side="right")
        tk.Label(row, textvariable=self._status_var, font=("Arial", 9), fg="#6b5316").pack(side="left")

        self._refresh_history_widget()

    def close_inspector(self) -> None:
        super().close_inspector()
        self._history_text = None

    def compute(self, inputs: dict) -> dict:
        self._init_state()

        if self._truthy(inputs.get("reset")):
            self._clear_history()
            return {
                "sum": self._sum,
                "count": float(self._count),
                "last": self._last,
            }

        value = inputs.get("value")
        use_trigger = "trigger" in inputs
        should_accumulate = (value is not None) and (
            (not use_trigger) or self._truthy(inputs.get("trigger"))
        )

        if should_accumulate:
            try:
                v = float(value)
                self._sum += v
                self._count += 1
                self._last = v
                self._history.append(v)
                self._refresh_status()
                self._refresh_history_widget()
                self.set_status("ok", "#339966")
            except Exception:
                self.set_status("bad input", "#cc0000")

        return {
            "sum": self._sum,
            "count": float(self._count),
            "last": self._last,
        }

    def get_params(self) -> dict:
        return {
            "sum": self._sum,
            "count": self._count,
            "last": self._last,
            "history": list(self._history),
        }

    def set_params(self, params: dict) -> None:
        self._init_state()
        try:
            self._sum = float(params.get("sum", 0.0))
        except Exception:
            self._sum = 0.0
        try:
            self._count = int(params.get("count", 0))
        except Exception:
            self._count = 0
        try:
            self._last = float(params.get("last", 0.0))
        except Exception:
            self._last = 0.0

        raw_history = params.get("history", [])
        parsed_history: list[float] = []
        if isinstance(raw_history, list):
            for item in raw_history:
                try:
                    parsed_history.append(float(item))
                except Exception:
                    continue
        self._history = parsed_history

        # Keep derived fields consistent when loading legacy/inconsistent payloads.
        if self._history:
            self._count = len(self._history)
            self._sum = float(sum(self._history))
            self._last = float(self._history[-1])

        self._refresh_status()
        self._refresh_history_widget()