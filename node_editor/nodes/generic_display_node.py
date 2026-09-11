import ast
import math
import re
import tkinter as tk
import traceback

import numpy as np

from node_editor.base_node import BaseNode
from node_editor.pin_types import PinDef, PinSchema, PinType
from node_editor.execution import ExecutionMode


class GenericDisplayNode(BaseNode):
    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE = "generic_display"
    DISPLAY_NAME = "Generic Display"
    CATEGORY = "visualize"
    NODE_WIDTH = 210
    NODE_HEIGHT = 110

    def __init__(self, node_id: str, canvas: tk.Canvas):
        super().__init__(node_id, canvas)
        self._value = None
        self._last_value = None
        self._status_var = tk.StringVar(value="waiting for input")
        self._data_var = tk.StringVar(value="")
        self._index_var = tk.StringVar(value="")
        self._key_var = tk.StringVar(value="")
        self._expr_var = tk.StringVar(value="")
        self._index_text = self._index_var
        self._key_text = self._key_var
        self._expr_text = self._expr_var
        self._expr_error_var = tk.StringVar(value="")
        self._data_label_var = tk.StringVar(value="Data")
        self._has_body = False

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[PinDef("value", PinType.ANY, "value", optional=True)],
            outputs=[PinDef("result", PinType.ANY, "result")],
        )

    def _init_state(self) -> None:
        if self._has_body:
            return
        self._has_body = True

    def _coerce_python_value(self, value):
        if value is None:
            return None
        if isinstance(value, (str, bytes, bytearray)):
            return value
        try:
            return np.asarray(value)
        except Exception:
            return value

    @staticmethod
    def _safe_repr(value):
        try:
            return repr(value)
        except Exception:
            try:
                return str(value)
            except Exception:
                return "<unprintable object>"

    @staticmethod
    def _is_array_like(value):
        if isinstance(value, np.ndarray):
            return True
        if isinstance(value, (list, tuple)):
            return True
        try:
            return hasattr(value, "__array__")
        except Exception:
            return False

    def _format_value(self, value):
        if value is None:
            return "None"
        if isinstance(value, np.ndarray):
            return repr(value)
        if isinstance(value, (list, tuple, set, dict)):
            return self._safe_repr(value)
        if isinstance(value, str):
            return value
        if np.isscalar(value):
            return str(value)
        try:
            return self._safe_repr(value)
        except Exception:
            return str(value)

    def _status_summary(self, value, state="ready"):
        bits = [state]
        if value is None:
            bits.append("type=None")
            return " | ".join(bits)

        bits.append(f"type={type(value).__name__}")

        try:
            bits.append(f"len={len(value)}")
        except Exception:
            pass

        try:
            shape = getattr(value, "shape", None)
            if shape is not None:
                bits.append(f"shape={tuple(shape)}")
        except Exception:
            pass

        return " | ".join(bits)

    def _data_label_text(self, value):
        if value is None:
            value_type = "None"
        else:
            value_type = type(value).__name__

        details = [f"type: {value_type}"]
        try:
            details.append(f"len: {len(value)}")
        except Exception:
            # Scalars and custom objects without a meaningful length should
            # retain the concise type-only label.
            pass

        if isinstance(value, np.ndarray):
            details.append(f"shape: {value.shape}")

        return f"Data ({', '.join(details)})"

    def _parse_index(self, text):
        text = (text or "").strip()
        if text == "":
            return None
        try:
            return int(text)
        except ValueError:
            raise ValueError("index must be an integer")

    def _parse_key(self, text):
        text = (text or "").strip()
        if text == "":
            return None
        return text

    def _lookup_value(self, x):
        idx_text = self._index_text.get() if hasattr(self, "_index_text") else self._index_var.get()
        key_text = self._key_text.get() if hasattr(self, "_key_text") else self._key_var.get()
        expr_text = self._expr_text.get() if hasattr(self, "_expr_text") else self._expr_var.get()
        idx = self._parse_index(idx_text) if idx_text.strip() else None
        key = self._parse_key(key_text) if key_text.strip() else None
        expr = (expr_text or "").strip()

        if expr:
            return self._evaluate_expression(x, expr)

        if idx is not None:
            if isinstance(x, (list, tuple)):
                if 0 <= idx < len(x):
                    return x[idx]
                raise IndexError(f"index {idx} out of range for length {len(x)}")
            if isinstance(x, np.ndarray):
                if x.ndim == 0:
                    raise IndexError("scalar array has no index")
                if 0 <= idx < x.size:
                    return x.reshape(-1)[idx]
                raise IndexError(f"index {idx} out of range for size {x.size}")
            raise TypeError(f"indexing is unsupported for {type(x).__name__}")

        if key is not None:
            if isinstance(x, dict):
                if key not in x:
                    raise KeyError(f"key {key!r} not found")
                return x[key]
            raise TypeError(f"key lookup is unsupported for {type(x).__name__}")

        return x

    def _evaluate_expression(self, x, expr):
        try:
            code = compile(expr, "<generic_display_expr>", "eval")
            env = {
                "x": x,
                "np": np,
                "math": math,
                "__builtins__": {},
            }
            result = eval(code, env, {})
            return result
        except Exception as exc:
            self._expr_error_var.set(f"error: {exc}")
            raise

    def _current_display_value(self):
        base = self._value
        if base is None:
            return None
        try:
            return self._lookup_value(base)
        except Exception:
            raise

    def _resolved_display_value(self):
        value = self._value
        if value is None:
            return None
        return self._current_display_value()

    def _refresh_display(self):
        try:
            value = self._current_display_value()
            self._last_value = value
            self._data_var.set(self._format_value(value))
            self._data_label_var.set(self._data_label_text(value))
            self._status_var.set(self._status_summary(value, "ready"))
            self._expr_error_var.set("")
        except Exception as exc:
            self._status_var.set(self._status_summary(self._value, "error"))
            self._expr_error_var.set(str(exc))
            self._data_var.set(self._format_value(self._value))
            self._data_label_var.set(self._data_label_text(self._value))

    def build_body(self) -> None:
        self._init_state()
        x, y, w, h = self.x, self.y, self.width, self.height
        self._body_rect = self.canvas.create_rectangle(
            x, y, x + w, y + h,
            fill="#eef2ff", outline="#4d5ea8", width=2,
            tags=(self.node_id, "node_body"),
        )
        self._title_item = self.canvas.create_text(
            x + w / 2, y + 11,
            text=self.DISPLAY_NAME,
            font=("Arial", 9, "bold"), fill="#2d3b73",
            tags=(self.node_id,),
        )
        status = tk.Label(self.canvas, textvariable=self._status_var,
                          font=("Arial", 8), bg="#eef2ff", fg="#3b4a7a")
        self.canvas.create_window(x + 6, y + 24, window=status, anchor="nw", tags=(self.node_id,))

        self._canvas_items += [self._body_rect, self._title_item]

    def build_inspector(self, parent: tk.Frame) -> None:
        self._init_state()

        tk.Label(parent, text="Index", font=("Arial", 9)).pack(anchor="w")
        idx_entry = tk.Entry(parent, textvariable=self._index_var, width=20)
        idx_entry.pack(fill="x", pady=(0, 6))

        tk.Label(parent, text="Key", font=("Arial", 9)).pack(anchor="w")
        key_entry = tk.Entry(parent, textvariable=self._key_var, width=30)
        key_entry.pack(fill="x", pady=(0, 6))

        # set the label "Expression ..." something that user can understand that x is the input variable
        expr_label = "Expression (x as input object)"
        tk.Label(parent, text=expr_label, font=("Arial", 9)).pack(anchor="w")
        expr_entry = tk.Entry(parent, textvariable=self._expr_var, width=40)
        expr_entry.pack(fill="x", pady=(0, 6))
        # make expr_entry having a placeholder text "e.g. x[0] or x['key'] or np.mean(x)"
        # the placeholder text will be cleared when the user focuses on the entry, and restored if the entry is empty when focus is lost
        # set placeholder text color to gray, and normal text color to black
        expr_entry.config(fg="gray")
        def _clear_placeholder(event):
            if expr_entry.get() == "e.g. x[0] or x['key'] or np.mean(x) or np.linalg.svd(x.T@x)":
                expr_entry.delete(0, tk.END)
                expr_entry.config(fg="black")

        def _restore_placeholder(event):
            if not expr_entry.get():
                expr_entry.insert(0, "e.g. x[0] or x['key'] or np.mean(x) or np.linalg.svd(x.T@x)")
                expr_entry.config(fg="gray")
        expr_entry.bind("<FocusIn>", _clear_placeholder)
        expr_entry.bind("<FocusOut>", _restore_placeholder)
        expr_entry.insert(0, "e.g. x[0] or x['key'] or np.mean(x) or np.linalg.svd(x.T@x)") 

        tk.Label(parent, textvariable=self._data_label_var, font=("Arial", 9), anchor="w").pack(anchor="w")
        data_box = tk.Text(parent, height=12, width=42, state="normal")
        data_box.pack(fill="both", expand=True, pady=(0, 4))

        err_lbl = tk.Label(parent, textvariable=self._expr_error_var, font=("Arial", 8), fg="#aa0000", anchor="w", justify="left")
        err_lbl.pack(fill="x")

        def _sync_display(*_args):
            try:
                value = self._current_display_value()
                self._data_var.set(self._format_value(value))
                self._data_label_var.set(self._data_label_text(value))
                self._expr_error_var.set("")
            except Exception as exc:
                self._data_var.set(self._format_value(self._value))
                self._data_label_var.set(self._data_label_text(self._value))
                self._expr_error_var.set(str(exc))
            data_box.delete("1.0", tk.END)
            data_box.insert("1.0", self._data_var.get())

        self._index_var.trace_add("write", _sync_display)
        self._key_var.trace_add("write", _sync_display)
        self._expr_var.trace_add("write", _sync_display)

        _sync_display()

    def compute(self, inputs: dict) -> dict:
        value = inputs.get("value")
        self._value = value
        try:
            display = self._resolved_display_value()
            self._last_value = display
            self._data_var.set(self._format_value(display))
            self._data_label_var.set(self._data_label_text(display))
            self._status_var.set(self._status_summary(display, "ready"))
            self._expr_error_var.set("")
            if self.is_inspector_open():
                self._refresh_display()
            return {"result": display}
        except Exception as exc:
            display = value
            self._status_var.set(self._status_summary(value, "error"))
            self._expr_error_var.set(str(exc))
            self._data_var.set(self._format_value(value))
            self._data_label_var.set(self._data_label_text(value))
            return {"result": display}

    def get_params(self) -> dict:
        return {
            "index": self._index_var.get(),
            "key": self._key_var.get(),
            "expr": self._expr_var.get(),
        }

    def set_params(self, params: dict) -> None:
        self._index_var.set(str(params.get("index", "")))
        self._key_var.set(str(params.get("key", "")))
        self._expr_var.set(str(params.get("expr", "")))
        self._refresh_display()
