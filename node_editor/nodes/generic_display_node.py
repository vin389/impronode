import ast
import math
import re
import tkinter as tk
import traceback

import numpy as np

from node_editor.base_node import BaseNode
from node_editor.pin_types import PinDef, PinSchema, PinType
from node_editor.execution import ExecutionMode


_EXPR_PLACEHOLDER = "e.g. x[0] or x['key'] or np.mean(x) or np.linalg.svd(x.T@x)"

# Builtins available inside the expression box, in addition to x, np, math.
# Deliberately a small read-only set (no open/eval/exec/__import__ ...).
_EXPR_BUILTINS = {
    "type": type, "len": len, "isinstance": isinstance,
    "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
    "sorted": sorted, "range": range,
    "float": float, "int": int, "str": str, "bool": bool,
    "list": list, "tuple": tuple, "dict": dict, "set": set,
}


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
        self._expr_var = tk.StringVar(value="")       # text being edited in the inspector
        self._applied_expr = ""                       # expression the OUTPUT uses (set by Apply)
        self._apply_status_var = tk.StringVar(value="")
        self._expr_trace_id = None
        self._data_box = None
        self._apply_btn = None
        self._expr_error_var = tk.StringVar(value="")
        self._data_label_var = tk.StringVar(value="Data")
        self._has_body = False

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[PinDef("x", PinType.ANY, "x", optional=True)],
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

    # ── expression evaluation ─────────────────────────────────────

    def _evaluate_expression(self, x, expr):
        """Evaluate a non-empty expression with x, np, math (raises on error)."""
        code = compile(expr, "<generic_display_expr>", "eval")
        env = {
            "x": x,
            "np": np,
            "math": math,
            "__builtins__": _EXPR_BUILTINS,
        }
        return eval(code, env, {})

    def _lookup_value(self, x, expr):
        expr = (expr or "").strip()
        return self._evaluate_expression(x, expr) if expr else x

    def _try_expression(self, expr):
        """Check an expression against the current input, evaluating it once.
        Returns (ok, message, value). ok is False on a syntax error, or when
        it fails on the last received input; with no input yet only the
        syntax can be checked."""
        expr = (expr or "").strip()
        if not expr:
            return True, "", self._value
        try:
            compile(expr, "<generic_display_expr>", "eval")
        except (SyntaxError, ValueError) as exc:
            return False, f"syntax error: {getattr(exc, 'msg', None) or exc}", None
        if self._value is None:
            return True, "", None
        try:
            return True, "", self._evaluate_expression(self._value, expr)
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}", None

    def _refresh_display(self):
        """Refresh the inspector from the expression currently typed (which
        may not be applied yet): preview, error text and Apply-button state.
        Never changes what the node outputs."""
        expr = self._expr_var.get()
        ok, message, value = self._try_expression(expr)
        if ok:
            shown = value
            self._expr_error_var.set("")
        else:
            shown = self._value
            self._expr_error_var.set(message)
        self._data_var.set(self._format_value(shown))
        self._data_label_var.set(self._data_label_text(shown))

        box = self._data_box
        if box is not None and box.winfo_exists():
            box.delete("1.0", tk.END)
            box.insert("1.0", self._data_var.get())

        btn = self._apply_btn
        if btn is not None and btn.winfo_exists():
            btn.config(state=tk.NORMAL if ok else tk.DISABLED)

        applied = self._applied_expr
        text = f"Output uses: {applied or 'x (pass-through)'}"
        if expr.strip() != applied:
            text += "   (edited - click Apply)"
        self._apply_status_var.set(text)

    def _on_apply(self) -> None:
        """Make the typed expression the one the output uses, and push the
        new result downstream. Ignored while the expression is invalid."""
        expr = self._expr_var.get().strip()
        ok, _message, _value = self._try_expression(expr)
        if not ok:
            self._refresh_display()
            return
        self._applied_expr = expr
        self._refresh_display()
        request = getattr(self, "_request_downstream", None)
        if request is not None:
            request(self.node_id)          # engine re-runs compute() and downstream nodes
        else:
            self.compute({"x": self._value})

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

    def _remove_expr_trace(self) -> None:
        """Detach the write-trace of a previous inspector so it cannot fire
        against widgets that no longer exist."""
        if self._expr_trace_id is not None:
            try:
                self._expr_var.trace_remove("write", self._expr_trace_id)
            except tk.TclError:
                pass
            self._expr_trace_id = None

    def close_inspector(self) -> None:
        self._remove_expr_trace()
        self._data_box = None
        self._apply_btn = None
        super().close_inspector()

    def build_inspector(self, parent: tk.Frame) -> None:
        self._init_state()
        self._remove_expr_trace()

        tk.Label(parent, text="Expression (x as input object)",
                 font=("Arial", 9)).pack(anchor="w")
        expr_entry = tk.Entry(parent, textvariable=self._expr_var, width=40)
        expr_entry.pack(fill="x", pady=(0, 6))

        # Placeholder: a gray label laid over the entry, shown only while the
        # entry is empty AND does not have keyboard focus. It is not text in
        # the entry, so it never reaches _expr_var.
        placeholder = tk.Label(expr_entry, text=_EXPR_PLACEHOLDER, fg="gray",
                               bg=expr_entry.cget("bg"), font=expr_entry.cget("font"),
                               anchor="w", cursor="xterm")
        has_focus = {"value": False}

        def _update_placeholder(*_args):
            if not placeholder.winfo_exists():
                return
            if self._expr_var.get() == "" and not has_focus["value"]:
                placeholder.place(x=3, rely=0.5, anchor="w")
            else:
                placeholder.place_forget()

        def _on_focus_in(_event):
            has_focus["value"] = True
            _update_placeholder()

        def _on_focus_out(_event):
            has_focus["value"] = False
            _update_placeholder()

        placeholder.bind("<Button-1>", lambda _e: expr_entry.focus_set())
        expr_entry.bind("<FocusIn>", _on_focus_in)
        expr_entry.bind("<FocusOut>", _on_focus_out)

        # Apply: enabled only while the typed expression is valid.
        apply_row = tk.Frame(parent)
        apply_row.pack(fill="x", pady=(0, 6))
        self._apply_btn = tk.Button(apply_row, text="Apply", width=8, state=tk.DISABLED,
                                    command=self._on_apply)
        self._apply_btn.pack(side="left")
        tk.Label(apply_row, textvariable=self._apply_status_var, font=("Arial", 8),
                 fg="#555555", anchor="w").pack(side="left", padx=(8, 0))
        expr_entry.bind("<Return>", lambda _e: self._on_apply())

        tk.Label(parent, textvariable=self._data_label_var, font=("Arial", 9), anchor="w").pack(anchor="w")
        self._data_box = tk.Text(parent, height=12, width=42, state="normal")
        self._data_box.pack(fill="both", expand=True, pady=(0, 4))

        err_lbl = tk.Label(parent, textvariable=self._expr_error_var, font=("Arial", 8), fg="#aa0000", anchor="w", justify="left")
        err_lbl.pack(fill="x")

        def _on_expr_changed(*_args):
            self._refresh_display()
            _update_placeholder()

        self._expr_trace_id = self._expr_var.trace_add("write", _on_expr_changed)

        self._refresh_display()
        _update_placeholder()

    def compute(self, inputs: dict) -> dict:
        value = inputs.get("x")
        self._value = value
        try:
            # The output always uses the APPLIED expression, not whatever is
            # currently being typed in the inspector.
            result = None if value is None else self._lookup_value(value, self._applied_expr)
            self._last_value = result
            self._status_var.set(self._status_summary(result, "ready"))
        except Exception as exc:
            result = value
            self._status_var.set(self._status_summary(value, "error"))
            self._expr_error_var.set(f"{type(exc).__name__}: {exc}")
        if self.is_inspector_open():
            self._refresh_display()
        return {"result": result}

    def get_params(self) -> dict:
        # The applied expression is what defines the node's output.
        return {"expr": self._applied_expr}

    def set_params(self, params: dict) -> None:
        expr = str(params.get("expr", "") or "").strip()
        if not expr:
            # Files saved before Index/Key were removed: express the old
            # setting as an equivalent expression (old priority: index, key).
            idx = str(params.get("index", "") or "").strip()
            key = str(params.get("key", "") or "").strip()
            if idx:
                try:
                    expr = f"x[{int(idx)}]"
                except ValueError:
                    pass
            elif key:
                expr = f"x[{key!r}]"
        self._applied_expr = expr
        self._expr_var.set(expr)
        self._refresh_display()