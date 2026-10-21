import ast
import math
import os
import pickle
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import numpy as np

from node_editor.base_node import BaseNode
from node_editor.pin_types import PinDef, PinSchema, PinType
from node_editor.execution import ExecutionMode
from node_editor.project_context import get_project_directory
from node_editor.tooltip import add_tooltip


_EXPR_PLACEHOLDER = "e.g. x[0] or x['key'] or np.mean(x) or np.linalg.svd(x.T@x)"
_EXPR_PLACEHOLDER_SHORT = "(empty = no output, None)"

# Builtins available inside the expression box, in addition to x, np, math.
# Deliberately a small read-only set (no open/eval/exec/__import__ ...).
_EXPR_BUILTINS = {
    "type": type, "len": len, "isinstance": isinstance,
    "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
    "sorted": sorted, "range": range,
    "float": float, "int": int, "str": str, "bool": bool,
    "list": list, "tuple": tuple, "dict": dict, "set": set,
}

N_OUTPUTS = 6
# Output pin names. The first keeps the name of the node's former single
# output, so links in older projects stay connected.
OUTPUT_PINS = ["result"] + [f"result_{k}" for k in range(2, N_OUTPUTS + 1)]

_OBJECT_FILETYPES = [
    ("Supported files", "*.pkl *.pickle *.npy *.npz *.csv *.txt"),
    ("Pickle", "*.pkl *.pickle"), ("NumPy", "*.npy *.npz"), ("CSV / text", "*.csv *.txt"),
    ("All files", "*.*"),
]


def load_object_file(path: str):
    """Load a Python object from a file, by extension:
      .pkl / .pickle  pickle.load  (runs code from the file: trusted files only)
      .npy            a NumPy array
      .npz            dict {name: array}
      .csv / .txt     a numeric array; with a header line of names, a dict
                      {column name: column array}
    Raises on failure."""
    ext = Path(path).suffix.lower()
    if ext in (".pkl", ".pickle"):
        with open(path, "rb") as f:
            return pickle.load(f)
    if ext == ".npy":
        return np.load(path, allow_pickle=False)
    if ext == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return {key: np.array(data[key]) for key in data.files}
    if ext in (".csv", ".txt"):
        return _load_csv(path)
    raise ValueError(f"unsupported file type '{ext}' (use .pkl, .npy, .npz, .csv or .txt)")


def _load_csv(path: str):
    with open(path, "r", encoding="utf-8-sig") as f:
        first = f.readline()
    delim = next((d for d in (",", ";", "\t") if d in first), None)     # None = whitespace
    try:
        return np.loadtxt(path, delimiter=delim, ndmin=1, encoding="utf-8-sig")
    except ValueError:
        pass
    # A header line of column names: {name: column}.
    table = np.genfromtxt(path, delimiter=delim, names=True, dtype=None, encoding="utf-8-sig")
    if table.dtype.names is None:
        raise ValueError("cannot read the CSV file as numbers")
    return {name: np.asarray(table[name]) for name in table.dtype.names}


class GenericDisplayNode(BaseNode):
    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE = "generic_display"
    DISPLAY_NAME = "Generic Display"
    CATEGORY = "visualize"
    NODE_WIDTH = 210
    NODE_HEIGHT = 170

    HELP_TEXT = (
        "Generic Display\n\n"
        "Shows any object and computes up to 6 outputs from it with Python\n"
        "expressions. The object is called x (np and math are available).\n\n"
        "Data source:\n"
        "  From input pin   x = what arrives at the input pin.\n"
        "  From file        x = an object loaded with 'Load ...'; the input pin\n"
        "                   is then ignored. Files: .pkl / .pickle (pickle --\n"
        "                   trusted files only), .npy (array), .npz (dict of\n"
        "                   arrays), .csv / .txt (array, or a dict of columns\n"
        "                   when the first line holds column names).\n"
        "  'Load ...' selects 'From file'; an array arriving at the pin while\n"
        "  no file is loaded selects 'From input pin'. The project stores the\n"
        "  file path (relative to the project file) and loads it again.\n\n"
        "Outputs out 1 ... out 6:\n"
        "  each has its own expression, e.g. x[0], x['key'], np.mean(x),\n"
        "  x.shape. Empty: out 1 = x itself, out 2 ... out 6 = None (no\n"
        "  output). An expression that does not use x\n"
        "  (e.g. np.linspace(0, 1, 5)) works without any input. Each row shows\n"
        "  the result's type / shape or its error; click 'out k' to preview\n"
        "  that output in the box below. Apply makes the typed expressions the\n"
        "  outputs (enabled while all of them are valid).\n"
    )

    def __init__(self, node_id: str, canvas: tk.Canvas):
        super().__init__(node_id, canvas)
        self._value = None                            # x: the input or the loaded file object
        self._status_var = tk.StringVar(value="waiting for input")
        self._expr_vars = [tk.StringVar(value="") for _ in range(N_OUTPUTS)]   # typed
        self._applied = [""] * N_OUTPUTS             # what the OUTPUTS use (set by Apply)
        self._row_status_vars = [tk.StringVar(value="") for _ in range(N_OUTPUTS)]
        self._row_status_labels = [None] * N_OUTPUTS
        self._preview_var = tk.IntVar(value=0)        # which output the data box previews
        self._apply_status_var = tk.StringVar(value="")
        self._expr_error_var = tk.StringVar(value="")
        self._data_label_var = tk.StringVar(value="Data")
        self._expr_trace_ids = []
        self._data_box = None
        self._apply_btn = None
        self._has_body = False
        # Data source: "pin" or "file" (object loaded with Load ...; the pin
        # is ignored then).
        self._source_var = tk.StringVar(value="pin")
        self._file_obj = None
        self._file_path = ""                          # absolute path of the loaded file
        self._file_info_var = tk.StringVar(value="")
        self._last_input = None                       # latest object from the pin

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[PinDef("x", PinType.ANY, "x", optional=True)],
            outputs=[PinDef(name, PinType.ANY, f"out {k + 1}") for k, name in enumerate(OUTPUT_PINS)],
        )

    def _init_state(self) -> None:
        if self._has_body:
            return
        self._has_body = True

    @staticmethod
    def _safe_repr(value):
        try:
            return repr(value)
        except Exception:
            try:
                return str(value)
            except Exception:
                return "<unprintable object>"

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

    @staticmethod
    def _short_summary(value) -> str:
        """One-line type / size of a value, for the per-output rows."""
        if value is None:
            return "None"
        text = type(value).__name__
        shape = getattr(value, "shape", None)
        if shape is not None:
            text += f" {tuple(shape)}"
            dtype = getattr(value, "dtype", None)
            if dtype is not None:
                text += f" {dtype}"
        else:
            try:
                text += f" len {len(value)}"
            except Exception:
                if np.isscalar(value):
                    text += f" = {value!r}"[:40]
        return text

    def _data_label_text(self, value, k: int):
        value_type = "None" if value is None else type(value).__name__
        details = [f"type: {value_type}"]
        try:
            details.append(f"len: {len(value)}")
        except Exception:
            pass
        if isinstance(value, np.ndarray):
            details.append(f"shape: {value.shape}")
        return f"Preview of out {k + 1} ({', '.join(details)})"

    # ── expression evaluation ─────────────────────────────────────

    def _evaluate_expression(self, x, expr):
        """Evaluate a non-empty expression with x, np, math (raises on error)."""
        code = compile(expr, "<generic_display_expr>", "eval")
        env = {"x": x, "np": np, "math": math, "__builtins__": _EXPR_BUILTINS}
        return eval(code, env, {})

    @staticmethod
    def _uses_x(expr: str) -> bool:
        """True if the expression refers to the input x. One that does not
        (e.g. np.linspace(0, 1, 5)) is evaluated even with no input."""
        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError:
            return True
        return any(isinstance(node, ast.Name) and node.id == "x" for node in ast.walk(tree))

    @staticmethod
    def _empty_result(k: int, value):
        """What an EMPTY expression gives: out 1 passes x through (as the
        node's former single output did); out 2 ... out 6 give None."""
        return value if k == 0 else None

    def _output_for(self, value, expr, k: int = 0):
        """Output k for x = `value`: the expression's result (empty: see
        _empty_result), or None while an expression that needs x has no x."""
        expr = (expr or "").strip()
        if not expr:
            return self._empty_result(k, value)
        if value is None and self._uses_x(expr):
            return None
        return self._evaluate_expression(value, expr)

    def _try_expression(self, expr, k: int = 0):
        """Check expression k against the current x, evaluating it once.
        Returns (ok, message, value). ok is False on a syntax error, or when
        it fails on the current x. With no x, an expression that does not
        use x is evaluated; one that does is only syntax-checked."""
        expr = (expr or "").strip()
        if not expr:
            return True, "", self._empty_result(k, self._value)
        try:
            compile(expr, "<generic_display_expr>", "eval")
        except (SyntaxError, ValueError) as exc:
            return False, f"syntax error: {getattr(exc, 'msg', None) or exc}", None
        if self._value is None and self._uses_x(expr):
            return True, "", None
        try:
            return True, "", self._evaluate_expression(self._value, expr)
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}", None

    def _refresh_display(self):
        """Refresh the inspector from the TYPED expressions (which may not be
        applied yet): per-output summaries / errors, the preview box and the
        Apply button. Never changes what the node outputs."""
        results = [self._try_expression(var.get(), k) for k, var in enumerate(self._expr_vars)]
        for k, (ok, message, value) in enumerate(results):
            self._row_status_vars[k].set(self._short_summary(value) if ok else message)
            lbl = self._row_status_labels[k]
            if lbl is not None and lbl.winfo_exists():
                lbl.config(fg="#3b4a7a" if ok else "#aa0000")

        k = self._preview_var.get()
        ok, message, value = results[k]
        shown = value if ok else None
        self._expr_error_var.set("" if ok else f"out {k + 1}: {message}")
        self._data_label_var.set(self._data_label_text(shown, k))
        box = self._data_box
        if box is not None and box.winfo_exists():
            box.delete("1.0", tk.END)
            box.insert("1.0", self._format_value(shown))

        all_ok = all(r[0] for r in results)
        btn = self._apply_btn
        if btn is not None and btn.winfo_exists():
            btn.config(state=tk.NORMAL if all_ok else tk.DISABLED)
        edited = [str(j + 1) for j in range(N_OUTPUTS)
                  if self._expr_vars[j].get().strip() != self._applied[j]]
        self._apply_status_var.set(
            ("Edited: out " + ", ".join(edited) + " -- click Apply" if edited else "Outputs are up to date")
            + ("" if all_ok else "   (fix the red rows first)"))

    def _on_apply(self) -> None:
        """Make the typed expressions what the outputs use, and push the new
        results downstream. Ignored while any expression is invalid."""
        if not all(self._try_expression(var.get(), k)[0] for k, var in enumerate(self._expr_vars)):
            self._refresh_display()
            return
        self._applied = [var.get().strip() for var in self._expr_vars]
        self._refresh_display()
        self._push()

    def _push(self) -> None:
        request = getattr(self, "_request_downstream", None)
        if request is not None:
            request(self.node_id)          # engine re-runs compute() and downstream nodes
        else:
            self.compute({"x": self._last_input})

    # ── data source: input pin or a loaded file ───────────────────

    def _dlg_parent(self):
        win = getattr(self, "_inspector_win", None)
        try:
            if win is not None and win.winfo_exists():
                return win
        except tk.TclError:
            pass
        return self.canvas.winfo_toplevel()

    def _on_load_file(self) -> bool:
        """Load ... : read an object from a file, select 'From file' and send
        it on. Returns False if nothing was loaded."""
        path = self.ask_open_filename(parent=self._dlg_parent(), title="Load an object",
                                      filetypes=_OBJECT_FILETYPES)
        if not path:
            return False
        if not self._load_path(path, show_errors=True):
            return False
        self._source_var.set("file")
        self._value = self._file_obj
        self._refresh_display()
        self._push()
        return True

    def _load_path(self, path: str, show_errors: bool) -> bool:
        try:
            obj = load_object_file(path)
        except Exception as e:
            msg = f"Cannot load {Path(path).name}: {type(e).__name__}: {e}"
            self._file_info_var.set(msg)
            if show_errors:
                messagebox.showerror("Load", msg, parent=self._dlg_parent())
            return False
        self._file_obj, self._file_path = obj, os.path.abspath(path)
        self._file_info_var.set(f"From file: {Path(path).name}   ({self._short_summary(obj)}; "
                                f"input pin ignored)")
        return True

    def _on_source_selected(self) -> None:
        """Radio button 'From input pin' / 'From file' clicked."""
        if self._source_var.get() == "file":
            if self._file_obj is None and not self._on_load_file():
                self._source_var.set("pin")           # no file chosen: stay on the pin
                return
            self._value = self._file_obj
            self._file_info_var.set(f"From file: {Path(self._file_path).name}   (input pin ignored)")
        else:
            self._value = self._last_input
            self._file_info_var.set("From input pin" + ("" if self._last_input is not None
                                                          else "   (nothing received yet)"))
        self._refresh_display()
        self._push()

    # ── node face ─────────────────────────────────────────────────

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
        status = tk.Label(self.canvas, textvariable=self._status_var, wraplength=w - 50,
                          justify="left", font=("Arial", 8), bg="#eef2ff", fg="#3b4a7a")
        self.canvas.create_window(x + 6, y + 24, window=status, anchor="nw", tags=(self.node_id,))
        self._canvas_items += [self._body_rect, self._title_item]

    # ── inspector ─────────────────────────────────────────────────

    def _remove_expr_traces(self) -> None:
        """Detach the write-traces of a previous inspector so they cannot
        fire against widgets that no longer exist."""
        for var, tid in self._expr_trace_ids:
            try:
                var.trace_remove("write", tid)
            except tk.TclError:
                pass
        self._expr_trace_ids = []

    def close_inspector(self) -> None:
        self._remove_expr_traces()
        self._data_box = None
        self._apply_btn = None
        self._row_status_labels = [None] * N_OUTPUTS
        super().close_inspector()

    def build_inspector(self, parent: tk.Frame) -> None:
        self._init_state()
        self._remove_expr_traces()

        # Data source.
        src = tk.Frame(parent)
        src.pack(fill="x")
        src_lbl = tk.Label(src, text="Data source (x):", font=("Arial", 9, "bold"))
        src_lbl.pack(side="left")
        pin_rb = tk.Radiobutton(src, text="From input pin", value="pin", variable=self._source_var,
                                command=self._on_source_selected)
        pin_rb.pack(side="left", padx=(6, 0))
        file_rb = tk.Radiobutton(src, text="From file", value="file", variable=self._source_var,
                                 command=self._on_source_selected)
        file_rb.pack(side="left")
        load_btn = tk.Button(src, text="Load ...", font=("Arial", 8), command=self._on_load_file)
        load_btn.pack(side="left", padx=(8, 0))
        add_tooltip(src_lbl, "Where the object x comes from: the input pin, or a file loaded "
                             "with 'Load ...'.")
        add_tooltip(pin_rb, "x = the object arriving at the input pin (the usual use). Selected "
                            "automatically when something arrives and no file has been loaded.")
        add_tooltip(file_rb, "x = the object loaded from a file. While this is selected the input "
                             "pin's data is IGNORED. Selected automatically by 'Load ...'; choosing "
                             "it with no file loaded yet opens the file dialog.")
        add_tooltip(load_btn, "Load an object from a file and select 'From file': .pkl / .pickle "
                              "(pickle -- this runs code stored in the file, so open only files you "
                              "trust), .npy (array), .npz (dict of arrays), .csv / .txt (array, or "
                              "a dict of columns if the first line has column names). The project "
                              "remembers the file and loads it again.")
        tk.Label(parent, textvariable=self._file_info_var, font=("Arial", 8), fg="#334455",
                 anchor="w", justify="left").pack(fill="x")

        # One row per output: [out k] [expression] [type / shape or error].
        grid = tk.Frame(parent)
        grid.pack(fill="x", pady=(6, 4))
        hdr_tip = ("A Python expression computed from x (np and math are available), e.g. x[0], "
                   "x['key'], np.mean(x), x.shape. Empty: out 1 = x itself, out 2 - 6 = None "
                   "(no output). An expression that does not "
                   "use x (e.g. np.linspace(0, 1, 5)) works without any input. Apply makes the "
                   "typed expressions the outputs.\nWhy: to pick elements, shapes or statistics out "
                   "of an object and pass them on, without building extra nodes.")
        tk.Label(grid, text="Output", font=("Arial", 8, "bold")).grid(row=0, column=0, sticky="w")
        expr_hdr = tk.Label(grid, text="Expression (x = the object)", font=("Arial", 8, "bold"))
        expr_hdr.grid(row=0, column=1, sticky="w")
        tk.Label(grid, text="Result", font=("Arial", 8, "bold")).grid(row=0, column=2, sticky="w", padx=(6, 0))
        add_tooltip(expr_hdr, hdr_tip)
        for k in range(N_OUTPUTS):
            rb = tk.Radiobutton(grid, text=f"out {k + 1}", value=k, variable=self._preview_var,
                                command=self._refresh_display, font=("Arial", 9))
            rb.grid(row=k + 1, column=0, sticky="w")
            add_tooltip(rb, f"Preview out {k + 1} in the box below.")
            entry = tk.Entry(grid, textvariable=self._expr_vars[k], width=38, font=("Courier", 9))
            entry.grid(row=k + 1, column=1, sticky="ew", pady=1)
            entry.bind("<Return>", lambda _e: self._on_apply())
            entry.bind("<FocusIn>", lambda _e, kk=k: (self._preview_var.set(kk), self._refresh_display()))
            add_tooltip(entry, hdr_tip)
            self._add_placeholder(entry, self._expr_vars[k],
                                  _EXPR_PLACEHOLDER if k == 0 else _EXPR_PLACEHOLDER_SHORT)
            lbl = tk.Label(grid, textvariable=self._row_status_vars[k], font=("Arial", 8),
                           fg="#3b4a7a", anchor="w", width=26)
            lbl.grid(row=k + 1, column=2, sticky="w", padx=(6, 0))
            self._row_status_labels[k] = lbl
            tid = self._expr_vars[k].trace_add("write", lambda *_a: self._refresh_display())
            self._expr_trace_ids.append((self._expr_vars[k], tid))
        grid.grid_columnconfigure(1, weight=1)

        apply_row = tk.Frame(parent)
        apply_row.pack(fill="x", pady=(0, 6))
        self._apply_btn = tk.Button(apply_row, text="Apply", width=8, state=tk.DISABLED,
                                    command=self._on_apply)
        self._apply_btn.pack(side="left")
        add_tooltip(self._apply_btn, "Make the typed expressions the node's outputs (Enter in a "
                                     "row does the same). Enabled while all rows are valid.")
        tk.Label(apply_row, textvariable=self._apply_status_var, font=("Arial", 8),
                 fg="#555555", anchor="w").pack(side="left", padx=(8, 0))

        tk.Label(parent, textvariable=self._data_label_var, font=("Arial", 9), anchor="w").pack(anchor="w")
        self._data_box = tk.Text(parent, height=12, width=60, state="normal")
        self._data_box.pack(fill="both", expand=True, pady=(0, 4))
        tk.Label(parent, textvariable=self._expr_error_var, font=("Arial", 8), fg="#aa0000",
                 anchor="w", justify="left").pack(fill="x")
        self._refresh_display()

    def _add_placeholder(self, entry: tk.Entry, var: tk.StringVar, text: str) -> None:
        """Grey hint laid over an empty entry that has no focus. It is not
        text in the entry, so it never reaches the variable. Follows the
        variable (also when it is set by code), via a trace removed with the
        inspector."""
        placeholder = tk.Label(entry, text=text, fg="gray", bg=entry.cget("bg"),
                               font=entry.cget("font"), anchor="w", cursor="xterm")
        focus = {"in": False}

        def update(*_a):
            try:
                if var.get() == "" and not focus["in"]:
                    placeholder.place(x=3, rely=0.5, anchor="w")
                else:
                    placeholder.place_forget()
            except tk.TclError:
                pass

        def on_focus(state):
            focus["in"] = state
            update()
        placeholder.bind("<Button-1>", lambda _e: entry.focus_set())
        entry.bind("<FocusIn>", lambda _e: on_focus(True), add="+")
        entry.bind("<FocusOut>", lambda _e: on_focus(False), add="+")
        self._expr_trace_ids.append((var, var.trace_add("write", update)))
        update()

    # ── engine ────────────────────────────────────────────────────

    def compute(self, inputs: dict) -> dict:
        raw = inputs.get("x")
        if raw is not None:
            self._last_input = raw
            if self._source_var.get() == "file" and self._file_obj is None:
                self._source_var.set("pin")           # nothing loaded: follow the input
        # 'From file': the pin is ignored. 'From input pin': exactly what the
        # pin delivers now (None when disconnected).
        value = self._file_obj if self._source_var.get() == "file" else raw
        self._value = value
        outputs, errors = {}, []
        for k, name in enumerate(OUTPUT_PINS):
            try:
                outputs[name] = self._output_for(value, self._applied[k], k)
            except Exception as exc:
                outputs[name] = None
                errors.append(f"out {k + 1}: {type(exc).__name__}: {exc}")
        first = outputs[OUTPUT_PINS[0]]
        if errors:
            self._status_var.set("error -- " + errors[0])
        elif value is None and first is None:
            self._status_var.set("waiting for input")
        else:
            self._status_var.set(self._status_summary(first, "ready"))
        if self.is_inspector_open():
            self._refresh_display()
        return outputs

    # ── project ───────────────────────────────────────────────────

    def get_params(self) -> dict:
        params = {"exprs": list(self._applied), "expr": self._applied[0],   # 'expr': older versions
                  "source": self._source_var.get()}
        if self._file_path:
            base = get_project_directory()
            rel = self._file_path
            if base is not None:
                try:
                    rel = os.path.relpath(self._file_path, str(base))
                except ValueError:                    # other drive: keep absolute
                    rel = self._file_path
            params["file"] = rel
        return params

    def set_params(self, params: dict) -> None:
        exprs = params.get("exprs")
        if isinstance(exprs, list):
            applied = [str(e or "").strip() for e in exprs[:N_OUTPUTS]]
        else:
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
            applied = [expr]
        self._applied = (applied + [""] * N_OUTPUTS)[:N_OUTPUTS]
        for var, expr in zip(self._expr_vars, self._applied):
            var.set(expr)

        file_ref = str(params.get("file", "") or "")
        self._file_obj, self._file_path = None, ""
        self._source_var.set("pin")
        if file_ref:
            base = get_project_directory()
            path = file_ref if os.path.isabs(file_ref) or base is None else str(Path(base) / file_ref)
            if self._load_path(path, show_errors=False):
                if str(params.get("source", "pin")) == "file":
                    self._source_var.set("file")
                    self._value = self._file_obj
            else:
                self._file_path = os.path.abspath(path)   # keep the reference for the next save
        self._refresh_display()
