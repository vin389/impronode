# node_editor/nodes/array_algebra_node.py

import tkinter as tk
from tkinter import ttk

import numpy as np

from node_editor.base_node import BaseNode
from node_editor.execution import ExecutionMode
from node_editor.pin_types import PinDef, PinSchema, PinType


# ---------------------------------------------------------------------------
# Operation catalog
# ---------------------------------------------------------------------------

_OPS = [
    ("inv",         "Matrix: Inverse (inv)"),
    ("matmul",      "Matrix: Multiply (matmul)"),
    ("concatenate", "Assembly: Concatenate"),
    ("stack",       "Assembly: Stack"),
    ("hstack",      "Assembly: Hstack"),
    ("vstack",      "Assembly: Vstack"),
    ("dstack",      "Assembly: Dstack"),
    ("transpose",   "Assembly: Transpose"),
    ("det",         "Decompose: Determinant (det)"),
    ("eig",         "Decompose: Eigenvalues/Eigenvectors (eig)"),
    ("solve",       "Decompose: Solve Linear System (solve)"),
    ("lstsq",       "Decompose: Least Squares (lstsq)"),
    ("svd",         "Decompose: SVD"),
    ("qr",          "Decompose: QR"),
    ("cholesky",    "Decompose: Cholesky"),
    ("norm",        "Norm: Norm"),
    ("cond",        "Norm: Condition Number (cond)"),
]
_OP_LABEL_TO_KEY = {label: key for key, label in _OPS}
_OP_KEY_TO_LABEL = {key: label for key, label in _OPS}

_OPS_REQUIRING_B = {"matmul", "solve", "lstsq"}
_OPS_ASSEMBLY    = {"concatenate", "stack", "hstack", "vstack", "dstack"}
_OPS_USING_AXIS_FIELD   = {"concatenate", "stack", "norm"}
_OPS_USING_AXES_FIELD   = {"transpose"}
_OPS_USING_ORD_FIELD    = {"norm", "cond"}
_OPS_USING_MODE_FIELD   = {"qr"}
_OPS_USING_FULLMAT_FIELD = {"svd"}
_OPS_USING_RCOND_FIELD  = {"lstsq"}


_HELP_TEXT = (
    "Array Algebra Node\n\n"
    "Purpose:\n"
    "- A single node exposing a curated set of NumPy array and linear\n"
    "  algebra operations, selectable from the inspector, so you do not\n"
    "  need a separate node type for every numpy function.\n\n"
    "Input Pins:\n"
    "- a [ARRAY]: primary array (required for every operation).\n"
    "- b, c, d, e [ARRAY, optional]: additional arrays.\n"
    "    * matmul / solve / lstsq require 'b' as the second operand.\n"
    "    * concatenate / stack / hstack / vstack / dstack use every\n"
    "      connected array among a, b, c, d, e, in that order.\n"
    "    * All other operations use only 'a'; b..e are ignored.\n"
    "  This node supports up to 5 arrays. For more, chain a second\n"
    "  Array Algebra node whose 'a' is the first node's 'result'.\n"
    "- axis [SCALAR, optional]: overrides the inspector's Axis field\n"
    "  with a single integer. Only meaningful for concatenate/stack.\n\n"
    "Output Pins:\n"
    "- result [ARRAY]: the primary array result (inverse, product,\n"
    "  concatenated/stacked array, transposed array, eigenvectors,\n"
    "  solve's x, lstsq's x, SVD's U, QR's Q, Cholesky's L). Only\n"
    "  present when the selected operation produces a primary array.\n"
    "- scalar [SCALAR]: the primary scalar result (determinant, norm,\n"
    "  condition number). Only present when the operation produces a\n"
    "  scalar. lstsq additionally reports its numerical rank here.\n"
    "- extra [ANY]: a Python list of (name, value) tuples holding every\n"
    "  secondary result that does not fit 'result' or 'scalar', for\n"
    "  example eig's eigenvalues, SVD's S and Vh, QR's R, and lstsq's\n"
    "  residuals/rank/singular_values. Because this pin is typed ANY,\n"
    "  it can only be connected to another ANY-typed input pin -- it is\n"
    "  meant for inspection/logging, not for chaining into a typed\n"
    "  ARRAY/SCALAR pin on a downstream node.\n\n"
    "Dtype policy:\n"
    "- Linear-algebra operations (inv, matmul, det, eig, solve, lstsq,\n"
    "  svd, qr, cholesky, norm, cond) coerce their inputs to float64,\n"
    "  since NumPy's linear algebra routines expect floating point.\n"
    "- Assembly operations (concatenate, stack, hstack, vstack, dstack,\n"
    "  transpose) preserve the original dtype of each input array, so\n"
    "  they remain usable on non-numeric-algebra data such as images.\n\n"
    "Inspector Settings (shown only when relevant to the selected op):\n"
    "- Axis: single integer, used by Concatenate/Stack (default 0) and\n"
    "  by Norm (empty = None; a single int is also accepted). The axis\n"
    "  pin, when connected, overrides this field.\n"
    "- Axes (transpose only): comma-separated integers giving the new\n"
    "  axis order, e.g. \"1,0,2\". Empty = reverse all axes (NumPy's\n"
    "  default transpose behaviour).\n"
    "- ord / p (Norm, Condition Number): empty = None (default 2-norm\n"
    "  behaviour); or a number; or one of fro, nuc, inf, -inf.\n"
    "- Mode (QR only): reduced (default), complete, r, or raw.\n"
    "- Full matrices (SVD only): checkbox, default checked.\n"
    "- rcond (Least Squares only): empty = None (NumPy's recommended\n"
    "  default); or a number.\n\n"
    "Notes:\n"
    "- Any missing required input, or any error raised by NumPy (a\n"
    "  singular matrix for inv/solve/cholesky, mismatched shapes for\n"
    "  concatenate, etc.) is caught and shown in the status line rather\n"
    "  than crashing the data flow.\n"
    "- This node recomputes on every upstream change, like a plain SYNC\n"
    "  node (no trigger-gating). For very large matrices where this is\n"
    "  undesirable, a trig-gated variant could be added later following\n"
    "  the same buffered/edge-detected pattern used by the Optical Flow\n"
    "  and Template Match nodes.\n\n"
    "Keyboard Shortcut:\n"
    "Ctrl-H / Ctrl-h - show this help window.\n"
)


class ArrayAlgebraNode(BaseNode):
    """
    A single node exposing a curated set of NumPy array/linear-algebra
    operations, selectable from the inspector.

    See _HELP_TEXT (Ctrl-H) for the full pin/inspector reference.
    """

    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE      = "array_algebra"
    DISPLAY_NAME   = "Array Algebra"
    CATEGORY       = "process"
    NODE_WIDTH     = 210
    NODE_HEIGHT    = 130

    HELP_TEXT = _HELP_TEXT

    _BODY_BG   = "#eef0ff"
    _OUTLINE   = "#5555aa"
    _TITLE_FG  = "#333377"
    _STATUS_FG = "#555599"

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[
                PinDef("a",    PinType.ARRAY,  "a", optional=False),
                PinDef("b",    PinType.ARRAY,  "b", optional=True),
                PinDef("c",    PinType.ARRAY,  "c", optional=True),
                PinDef("d",    PinType.ARRAY,  "d", optional=True),
                PinDef("e",    PinType.ARRAY,  "e", optional=True),
                PinDef("axis", PinType.SCALAR, "axis", optional=True),
            ],
            outputs=[
                PinDef("result", PinType.ARRAY,  "result"),
                PinDef("scalar", PinType.SCALAR, "scalar"),
                PinDef("extra",  PinType.ANY,    "extra"),
            ],
        )

    def get_help_text(self) -> str:
        return _HELP_TEXT

    # ── state ─────────────────────────────────────────────────────

    def _init_state(self) -> None:
        if hasattr(self, "_op_var"):
            return
        self._op_var           = tk.StringVar(value=_OP_KEY_TO_LABEL["matmul"])
        self._axis_var         = tk.StringVar(value="0")
        self._axes_var         = tk.StringVar(value="")
        self._ord_var          = tk.StringVar(value="")
        self._mode_var         = tk.StringVar(value="reduced")
        self._full_matrices_var = tk.BooleanVar(value=True)
        self._rcond_var        = tk.StringVar(value="")

        self._status_var    = tk.StringVar(value="waiting for input a")
        self._op_label_var  = tk.StringVar(value=self._op_var.get())

        self._help_popup: tk.Toplevel | None = None

        # inspector-only widget refs, grouped rows for show/hide toggling
        self._row_axis  = None
        self._row_axes  = None
        self._row_ord   = None
        self._row_mode  = None
        self._row_full  = None
        self._row_rcond = None

    # ── body ──────────────────────────────────────────────────────

    def build_body(self) -> None:
        self._init_state()
        x, y, w, h = self.x, self.y, self.width, self.height

        self._body_rect = self.canvas.create_rectangle(
            x, y, x + w, y + h,
            fill=self._BODY_BG, outline=self._OUTLINE, width=2,
            tags=(self.node_id, "node_body"),
        )
        self._title_item = self.canvas.create_text(
            x + w / 2, y + 13,
            text=self.DISPLAY_NAME,
            font=("Arial", 9, "bold"), fill=self._TITLE_FG,
            tags=(self.node_id,),
        )

        op_lbl = tk.Label(
            self.canvas, textvariable=self._op_label_var,
            font=("Arial", 8, "bold"), bg=self._BODY_BG, fg=self._TITLE_FG,
            wraplength=w - 10, justify="center",
        )
        self.canvas.create_window(
            x + w / 2, y + h // 2, window=op_lbl, tags=(self.node_id,),
        )

        status_lbl = tk.Label(
            self.canvas, textvariable=self._status_var,
            font=("Arial", 7), bg=self._BODY_BG, fg=self._STATUS_FG,
            wraplength=w - 10, justify="center",
        )
        self.canvas.create_window(
            x + w / 2, y + h - 14, window=status_lbl, tags=(self.node_id,),
        )

        self._canvas_items += [self._body_rect, self._title_item]

    # ── inspector ─────────────────────────────────────────────────

    def build_inspector(self, parent: tk.Frame) -> None:
        self._init_state()

        win = self._inspector_win
        if win is not None:
            for seq in ("<Control-h>", "<Control-H>"):
                win.bind(seq, lambda e: self._open_help())

        pad = {"padx": 6, "pady": 4}

        # ── operation selector ──────────────────────────────────────
        op_frame = tk.LabelFrame(parent, text="Operation", font=("Arial", 9), **pad)
        op_frame.pack(fill="x", **pad)

        op_combo = ttk.Combobox(
            op_frame, textvariable=self._op_var,
            values=[label for _key, label in _OPS],
            state="readonly", width=32, font=("Arial", 9),
        )
        op_combo.pack(fill="x")
        op_combo.bind("<<ComboboxSelected>>", self._on_op_change)

        # ── pin usage hint ──────────────────────────────────────────
        self._pin_hint_var = tk.StringVar(value="")
        tk.Label(
            op_frame, textvariable=self._pin_hint_var,
            font=("Arial", 8), fg="#666666", anchor="w", justify="left",
        ).pack(fill="x", pady=(4, 0))

        # ── parameters (rows toggled per-operation) ─────────────────
        param_frame = tk.LabelFrame(parent, text="Parameters", font=("Arial", 9), **pad)
        param_frame.pack(fill="x", **pad)

        def _text_row(label_text, var, hint):
            row = tk.Frame(param_frame)
            tk.Label(row, text=label_text, font=("Arial", 9), width=16, anchor="w").pack(side="left")
            entry = tk.Entry(row, textvariable=var, width=16, font=("Arial", 9))
            entry.pack(side="left")
            entry.bind("<FocusOut>", self._on_param_committed)
            entry.bind("<Return>", self._on_param_committed)
            tk.Label(row, text=hint, font=("Arial", 8), fg="#888888").pack(side="left", padx=(6, 0))
            return row

        self._row_axis = _text_row(
            "Axis:", self._axis_var, "int, e.g. 0 (concatenate/stack), or empty (norm)")
        self._row_axes = _text_row(
            "Axes (transpose):", self._axes_var, "comma ints, empty = reverse all axes")
        self._row_ord = _text_row(
            "ord / p:", self._ord_var, "empty=None, number, or fro/nuc/inf/-inf")
        self._row_rcond = _text_row(
            "rcond:", self._rcond_var, "empty=None (recommended), or a number")

        row_mode = tk.Frame(param_frame)
        tk.Label(row_mode, text="Mode (qr):", font=("Arial", 9), width=16, anchor="w").pack(side="left")
        mode_combo = ttk.Combobox(
            row_mode, textvariable=self._mode_var,
            values=["reduced", "complete", "r", "raw"],
            state="readonly", width=14, font=("Arial", 9),
        )
        mode_combo.pack(side="left")
        mode_combo.bind("<<ComboboxSelected>>", self._on_param_committed)
        self._row_mode = row_mode

        row_full = tk.Frame(param_frame)
        full_cb = tk.Checkbutton(
            row_full, text="Full matrices (svd)",
            variable=self._full_matrices_var, font=("Arial", 9),
            command=self._on_param_committed,
        )
        full_cb.pack(side="left")
        self._row_full = row_full

        # ── status / last result summary ────────────────────────────
        status_frame = tk.LabelFrame(parent, text="Status", font=("Arial", 9), **pad)
        status_frame.pack(fill="x", **pad)
        tk.Label(
            status_frame, textvariable=self._status_var, font=("Arial", 9),
            fg="#334477", anchor="w", justify="left", wraplength=380,
        ).pack(fill="x")

        tk.Button(
            parent, text="Help (Ctrl-H)", font=("Arial", 9),
            command=self._open_help,
        ).pack(anchor="e", **pad)

        self._on_op_change()

    def close_inspector(self) -> None:
        super().close_inspector()
        self._row_axis = None
        self._row_axes = None
        self._row_ord = None
        self._row_mode = None
        self._row_full = None
        self._row_rcond = None

    # ── inspector behaviour ───────────────────────────────────────

    def _current_op_key(self) -> str:
        return _OP_LABEL_TO_KEY.get(self._op_var.get(), "matmul")

    def _on_op_change(self, _event=None) -> None:
        op = self._current_op_key()
        self._op_label_var.set(self._op_var.get())

        def _set_visible(row, visible):
            if row is None:
                return
            if visible:
                row.pack(fill="x", pady=2)
            else:
                row.pack_forget()

        _set_visible(self._row_axis,  op in _OPS_USING_AXIS_FIELD)
        _set_visible(self._row_axes,  op in _OPS_USING_AXES_FIELD)
        _set_visible(self._row_ord,   op in _OPS_USING_ORD_FIELD)
        _set_visible(self._row_mode,  op in _OPS_USING_MODE_FIELD)
        _set_visible(self._row_full,  op in _OPS_USING_FULLMAT_FIELD)
        _set_visible(self._row_rcond, op in _OPS_USING_RCOND_FIELD)

        if op in _OPS_ASSEMBLY:
            hint = "Uses every connected array among a, b, c, d, e, in order."
        elif op in _OPS_REQUIRING_B:
            hint = "Requires a and b."
        else:
            hint = "Uses only a; b, c, d, e are ignored."
        self._pin_hint_var.set(hint)

        self._trigger_recompute()

    def _on_param_committed(self, _event=None) -> None:
        self._trigger_recompute()

    def _trigger_recompute(self) -> None:
        if self._request_downstream:
            self._request_downstream(self.node_id)

    def _open_help(self) -> None:
        if self._help_popup is not None and self._help_popup.winfo_exists():
            self._help_popup.lift()
            return

        popup = tk.Toplevel()
        popup.title("Array Algebra - Help")
        popup.geometry("700x600")
        popup.resizable(True, True)
        try:
            px, py = self.canvas.winfo_pointerxy()
            popup.geometry(f"+{px + 16}+{py + 16}")
        except Exception:
            pass

        body = tk.Frame(popup, bg="#f8f8f8", padx=10, pady=8)
        body.pack(fill="both", expand=True)

        txt = tk.Text(
            body, font=("Courier", 9), bg="#f8f8f8", fg="#222222",
            wrap=tk.WORD, relief=tk.FLAT,
        )
        vsb = ttk.Scrollbar(body, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", _HELP_TEXT)
        txt.configure(state="disabled")

        tk.Button(body, text="Close", font=("Arial", 9), command=popup.destroy).pack(
            anchor="e", pady=(8, 0))
        popup.bind("<Escape>", lambda _e: popup.destroy())
        popup.protocol("WM_DELETE_WINDOW", popup.destroy)
        self._help_popup = popup

    # ── parsing helpers ───────────────────────────────────────────

    @staticmethod
    def _parse_axis_generic(text: str):
        """Return None, a single int, or a tuple of ints from comma-separated text."""
        text = (text or "").strip()
        if not text:
            return None
        parts = [p.strip() for p in text.split(",") if p.strip()]
        try:
            ints = [int(p) for p in parts]
        except ValueError as e:
            raise ValueError(f"invalid axis value: {text!r}") from e
        return ints[0] if len(ints) == 1 else tuple(ints)

    @staticmethod
    def _parse_axes_tuple(text: str):
        """Return None (reverse all axes) or a tuple of ints for np.transpose."""
        text = (text or "").strip()
        if not text:
            return None
        try:
            return tuple(int(p.strip()) for p in text.split(",") if p.strip())
        except ValueError as e:
            raise ValueError(f"invalid axes value: {text!r}") from e

    @staticmethod
    def _parse_ord(text: str):
        """Return None, 'fro', 'nuc', +-inf, or a number for norm's ord / cond's p."""
        text = (text or "").strip()
        if not text or text.lower() == "none":
            return None
        low = text.lower()
        if low in ("fro", "nuc"):
            return low
        if low in ("inf", "+inf"):
            return np.inf
        if low == "-inf":
            return -np.inf
        try:
            return float(text) if "." in text else int(text)
        except ValueError as e:
            raise ValueError(f"invalid ord/p value: {text!r}") from e

    @staticmethod
    def _parse_rcond(text: str):
        text = (text or "").strip()
        if not text or text.lower() == "none":
            return None
        try:
            return float(text)
        except ValueError as e:
            raise ValueError(f"invalid rcond value: {text!r}") from e

    def _resolve_axis(self, inputs: dict):
        """Pin overrides the inspector text field, as elsewhere in this project."""
        raw = inputs.get("axis")
        if raw is not None:
            try:
                return int(round(float(raw)))
            except (TypeError, ValueError):
                pass
        return self._parse_axis_generic(self._axis_var.get())

    # ── operation implementations ─────────────────────────────────

    def _gather_assembly_arrays(self, inputs: dict, a_raw) -> list[np.ndarray]:
        arrays = [np.asarray(a_raw)]
        for name in ("b", "c", "d", "e"):
            v = inputs.get(name)
            if v is not None:
                arrays.append(np.asarray(v))
        return arrays

    def _run_assembly_op(self, op: str, arrays: list[np.ndarray], inputs: dict) -> dict:
        if op == "concatenate":
            axis = self._resolve_axis(inputs)
            result = np.concatenate(arrays, axis=0 if axis is None else axis)
        elif op == "stack":
            axis = self._resolve_axis(inputs)
            result = np.stack(arrays, axis=0 if axis is None else axis)
        elif op == "hstack":
            result = np.hstack(arrays)
        elif op == "vstack":
            result = np.vstack(arrays)
        elif op == "dstack":
            result = np.dstack(arrays)
        else:
            raise ValueError(f"unknown assembly op: {op}")
        return {"result": result, "extra": []}

    def _run_transpose(self, a: np.ndarray) -> dict:
        axes = self._parse_axes_tuple(self._axes_var.get())
        result = np.transpose(a, axes)
        return {"result": result, "extra": []}

    def _run_binary_op(self, op: str, a: np.ndarray, b: np.ndarray) -> dict:
        if op == "matmul":
            return {"result": np.matmul(a, b), "extra": []}
        if op == "solve":
            return {"result": np.linalg.solve(a, b), "extra": []}
        if op == "lstsq":
            rcond = self._parse_rcond(self._rcond_var.get())
            x, residuals, rank, s = np.linalg.lstsq(a, b, rcond=rcond)
            extra = [
                ("residuals", residuals),
                ("rank", rank),
                ("singular_values", s),
            ]
            return {"result": x, "scalar": float(rank), "extra": extra}
        raise ValueError(f"unknown binary op: {op}")

    def _run_unary_linalg_op(self, op: str, a: np.ndarray) -> dict:
        if op == "inv":
            return {"result": np.linalg.inv(a), "extra": []}
        if op == "det":
            return {"scalar": float(np.linalg.det(a)), "extra": []}
        if op == "eig":
            w, v = np.linalg.eig(a)
            return {"result": v, "extra": [("eigenvalues", w)]}
        if op == "svd":
            full_matrices = bool(self._full_matrices_var.get())
            u, s, vh = np.linalg.svd(a, full_matrices=full_matrices)
            return {"result": u, "extra": [("S", s), ("Vh", vh)]}
        if op == "qr":
            mode = self._mode_var.get() or "reduced"
            res = np.linalg.qr(a, mode=mode)
            if mode == "r":
                return {"result": res, "extra": []}
            q, r = res
            return {"result": q, "extra": [("R", r)]}
        if op == "cholesky":
            return {"result": np.linalg.cholesky(a), "extra": []}
        if op == "norm":
            ord_ = self._parse_ord(self._ord_var.get())
            axis = self._parse_axis_generic(self._axis_var.get())
            n = np.linalg.norm(a, ord=ord_, axis=axis)
            if np.isscalar(n) or (isinstance(n, np.ndarray) and n.ndim == 0):
                return {"scalar": float(n), "extra": []}
            return {"result": np.asarray(n), "extra": []}
        if op == "cond":
            p = self._parse_ord(self._ord_var.get())
            return {"scalar": float(np.linalg.cond(a, p=p)), "extra": []}
        raise ValueError(f"unknown unary op: {op}")

    # ── status summary ────────────────────────────────────────────

    @staticmethod
    def _describe(value) -> str:
        if isinstance(value, np.ndarray):
            return f"{value.shape} {value.dtype}"
        if isinstance(value, (int, float, np.integer, np.floating)):
            return f"{float(value):.6g}"
        return type(value).__name__

    def _summarize(self, op: str, outputs: dict) -> str:
        parts = [f"ok: {_OP_KEY_TO_LABEL[op]}"]
        if "result" in outputs:
            parts.append(f"result {self._describe(outputs['result'])}")
        if "scalar" in outputs:
            parts.append(f"scalar {self._describe(outputs['scalar'])}")
        extra = outputs.get("extra") or []
        if extra:
            extra_desc = ", ".join(f"{name}:{self._describe(v)}" for name, v in extra)
            parts.append(f"extra [{extra_desc}]")
        return "  ".join(parts)

    # ── compute ───────────────────────────────────────────────────

    def compute(self, inputs: dict) -> dict:
        self._init_state()
        op = self._current_op_key()

        a_raw = inputs.get("a")
        if a_raw is None:
            self._status_var.set("waiting for input a")
            self.set_status("waiting", "#888888")
            return {}

        try:
            if op in _OPS_ASSEMBLY:
                arrays = self._gather_assembly_arrays(inputs, a_raw)
                outputs = self._run_assembly_op(op, arrays, inputs)
            elif op == "transpose":
                outputs = self._run_transpose(np.asarray(a_raw))
            elif op in _OPS_REQUIRING_B:
                b_raw = inputs.get("b")
                if b_raw is None:
                    self._status_var.set(f"waiting for input b (required for {op})")
                    self.set_status("waiting", "#888888")
                    return {}
                a = np.asarray(a_raw, dtype=np.float64)
                b = np.asarray(b_raw, dtype=np.float64)
                outputs = self._run_binary_op(op, a, b)
            else:
                a = np.asarray(a_raw, dtype=np.float64)
                outputs = self._run_unary_linalg_op(op, a)
        except Exception as e:
            self._status_var.set(f"error: {e}")
            self.set_status("error", "#cc0000")
            return {}

        self._status_var.set(self._summarize(op, outputs))
        self.set_status("ok", "#5555aa")
        return outputs

    # ── serialization ────────────────────────────────────────────

    def get_params(self) -> dict:
        self._init_state()
        return {
            "op":            self._op_var.get(),
            "axis":          self._axis_var.get(),
            "axes":          self._axes_var.get(),
            "ord":           self._ord_var.get(),
            "mode":          self._mode_var.get(),
            "full_matrices": bool(self._full_matrices_var.get()),
            "rcond":         self._rcond_var.get(),
        }

    def set_params(self, params: dict) -> None:
        self._init_state()
        self._op_var.set(str(params.get("op", _OP_KEY_TO_LABEL["matmul"])))
        self._axis_var.set(str(params.get("axis", "0")))
        self._axes_var.set(str(params.get("axes", "")))
        self._ord_var.set(str(params.get("ord", "")))
        self._mode_var.set(str(params.get("mode", "reduced")))
        self._full_matrices_var.set(bool(params.get("full_matrices", True)))
        self._rcond_var.set(str(params.get("rcond", "")))
        self._op_label_var.set(self._op_var.get())

    def on_destroy(self) -> None:
        if self._help_popup is not None and self._help_popup.winfo_exists():
            self._help_popup.destroy()
        super().on_destroy()