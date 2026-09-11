# node_editor/nodes/array_nodes.py

import tkinter as tk
from tkinter import ttk
from pathlib import Path
import numpy as np

from node_editor.base_node import BaseNode
from node_editor.pin_types import PinSchema, PinDef, PinType
from node_editor.execution import ExecutionMode
from node_editor.project_context import get_project_directory, get_project_file_path


class ArrayInputNode(BaseNode):
    """
    Lets the user manually enter a NumPy array via CSV text.
    Shape is inferred from the pasted content.
    An explicit dtype selector is provided.
    Output is triggered only when the user clicks Apply.
    """
    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE      = "array_input"
    DISPLAY_NAME   = "Array Input"
    CATEGORY       = "source"
    NODE_WIDTH     = 220
    NODE_HEIGHT    = 200

    _DTYPES = ["float64", "float32", "int32", "int16", "uint8"]

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[],
            outputs=[PinDef("array", PinType.ARRAY, "out",
                            shape=None)]   # any shape
        )

    def _init_state(self) -> None:
        if hasattr(self, "_dtype_var"):
            return
        self._dtype_var = tk.StringVar(value="float64")
        self._shape_mode_var = tk.StringVar(value="auto")
        self._shape_entry_var = tk.StringVar(value="")
        self._shape_display_var = tk.StringVar(value="Detected shape: ?")
        self._status_var = tk.StringVar(value="shape: ?")
        self._csv_text = "1, 2, 3\n4, 5, 6"
        self._text = None
        self._shape_entry = None
        self._last_array: np.ndarray | None = None

    def build_body(self) -> None:
        self._init_state()
        x, y, w, h = self.x, self.y, self.width, self.height

        # Compact canvas shell (Phase 2).
        self._body_rect = self.canvas.create_rectangle(
            x, y, x+w, y+h,
            fill="#f5f0e8", outline="#aa8855", width=1,
            tags=(self.node_id, "node_body"))
        self._title_item = self.canvas.create_text(
            x+w/2, y+12,
            text=self.DISPLAY_NAME,
            font=("Arial", 9, "bold"), fill="#553300",
            tags=(self.node_id,))

        status_lbl = tk.Label(
            self.canvas,
            textvariable=self._status_var,
            font=("Arial", 8), bg="#f5f0e8", fg="#553300")
        self.canvas.create_window(
            x+w/2, y+h-14, window=status_lbl,
            tags=(self.node_id,))

        self._canvas_items += [self._body_rect, self._title_item]

    def build_inspector(self, parent: tk.Frame) -> None:
        self._init_state()

        tk.Label(parent, text="dtype:", font=("Arial", 9)).grid(row=0, column=0, sticky="w")
        dtype_cb = ttk.Combobox(parent, textvariable=self._dtype_var,
                                values=self._DTYPES, width=10, state="readonly")
        dtype_cb.grid(row=0, column=1, sticky="w", padx=(6, 0))

        shape_frame = tk.LabelFrame(parent, text="Array shape", font=("Arial", 9), padx=6, pady=6)
        shape_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 4))

        self._shape_mode_var.set("auto")
        tk.Radiobutton(shape_frame, text="Auto reshape", variable=self._shape_mode_var,
                       value="auto", command=self._refresh_shape_state, anchor="w").pack(anchor="w")
        tk.Radiobutton(shape_frame, text="User defined shape", variable=self._shape_mode_var,
                       value="manual", command=self._refresh_shape_state, anchor="w").pack(anchor="w")

        shape_row = tk.Frame(shape_frame)
        shape_row.pack(fill="x", pady=(4, 0))
        tk.Label(shape_row, text="shape:", font=("Arial", 9)).pack(side="left")
        self._shape_entry = tk.Entry(shape_row, textvariable=self._shape_entry_var, width=18)
        self._shape_entry.pack(side="left", padx=(6, 0))
        self._shape_entry.bind("<KeyRelease>", lambda _e: self._refresh_shape_state())

        tk.Label(shape_frame, textvariable=self._shape_display_var, font=("Arial", 8), fg="#334455").pack(anchor="w", pady=(4, 0))

        tk.Label(parent, text="Rows (comma or space/tab separated):", font=("Arial", 9)).grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(8, 2)
        )

        frame = tk.Frame(parent, bd=1, relief=tk.SUNKEN)
        frame.grid(row=3, column=0, columnspan=2, sticky="nsew")
        self._text = tk.Text(frame, width=40, height=10, font=("Courier", 9), wrap=tk.NONE)
        vsb = tk.Scrollbar(frame, orient="vertical", command=self._text.yview)
        self._text.configure(yscrollcommand=vsb.set)
        self._text.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self._text.insert("1.0", self._csv_text)
        self._text.bind("<KeyRelease>", lambda _e: self._refresh_shape_state())

        apply_btn = tk.Button(parent, text="Apply", font=("Arial", 9), command=self._on_apply)
        apply_btn.grid(row=4, column=1, sticky="e", pady=(8, 2))

        status_lbl = tk.Label(parent, textvariable=self._status_var, font=("Arial", 9), fg="#553300")
        status_lbl.grid(row=4, column=0, sticky="w", pady=(8, 2))

        parent.grid_rowconfigure(3, weight=1)
        parent.grid_columnconfigure(1, weight=1)
        self._refresh_shape_state()

    def close_inspector(self) -> None:
        if self._text is not None and self._text.winfo_exists():
            self._csv_text = self._text.get("1.0", tk.END)
        super().close_inspector()
        self._text = None
        self._shape_entry = None

    # ── parsing ──────────────────────────────────────────────────

    def _get_csv_text(self) -> str:
        if self._text is not None and self._text.winfo_exists():
            return self._text.get("1.0", tk.END)
        return self._csv_text

    def _parse_numeric_rows(self, raw: str):
        raw = raw.strip()
        if not raw:
            return None, "empty"

        rows = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            tokens = []
            # accept comma, semicolon, space, and tab as separators;
            # runs of whitespace collapse via str.split() with no args
            for token in line.replace(";", ",").replace(",", " ").split():
                token = token.strip()
                if not token:
                    continue
                try:
                    tokens.append(float(token))
                except ValueError:
                    return None, "invalid number"
            if not tokens:
                continue
            rows.append(tokens)

        if not rows:
            return None, "empty"

        widths = {len(row) for row in rows}
        if len(widths) > 1:
            return None, "ragged rows"
        return rows, None

    def _parse_shape_text(self, text: str):
        text = text.strip()
        if not text:
            return None
        parts = [p.strip() for p in text.replace(";", ",").split(",")]
        if not parts or any(not p for p in parts):
            return None
        try:
            shape = tuple(int(p) for p in parts)
        except ValueError:
            return None
        if not shape:
            return None
        if any(dim == 0 for dim in shape):
            return None
        if sum(1 for dim in shape if dim < 0) > 1:
            return None
        # allow a single -1 for NumPy-style inference, but reject invalid negatives
        if any(dim < -1 for dim in shape):
            return None
        return shape

    def _refresh_shape_state(self) -> None:
        if self._shape_entry is None:
            return

        csv_text = self._get_csv_text()
        mode = self._shape_mode_var.get()
        if mode == "manual":
            self._shape_entry.configure(state="normal")
            desired_shape = self._parse_shape_text(self._shape_entry_var.get())
            rows, error = self._parse_numeric_rows(csv_text)
            if rows is None:
                self._shape_display_var.set("Need valid CSV rows")
                self._shape_entry.configure(bg="#ffe6e6")
                if self._text is not None and self._text.winfo_exists():
                    self._text.configure(bg="#ffe6e6")
                return

            flat = [value for row in rows for value in row]
            if desired_shape is None:
                self._shape_display_var.set("Enter a shape like 2,3 or 2,-1")
                self._shape_entry.configure(bg="#ffe6e6")
                if self._text is not None and self._text.winfo_exists():
                    self._text.configure(bg="#ffffff")
                return

            resolved_shape = desired_shape
            if -1 in desired_shape:
                known_dims = [dim for dim in desired_shape if dim != -1]
                known_product = int(np.prod(known_dims)) if known_dims else 1
                if known_product == 0:
                    self._shape_display_var.set("Shape mismatch: inferred dimension is invalid")
                    self._shape_entry.configure(bg="#ffe6e6")
                    if self._text is not None and self._text.winfo_exists():
                        self._text.configure(bg="#ffe6e6")
                    return
                inferred = len(flat) // known_product
                if len(flat) % known_product != 0:
                    self._shape_display_var.set(f"Shape mismatch: {desired_shape} does not fit {len(flat)} values")
                    self._shape_entry.configure(bg="#ffe6e6")
                    if self._text is not None and self._text.winfo_exists():
                        self._text.configure(bg="#ffe6e6")
                    return
                if inferred <= 0:
                    self._shape_display_var.set("Shape mismatch: inferred dimension must be positive")
                    self._shape_entry.configure(bg="#ffe6e6")
                    if self._text is not None and self._text.winfo_exists():
                        self._text.configure(bg="#ffe6e6")
                    return
                resolved_shape = tuple(inferred if dim == -1 else dim for dim in desired_shape)
            elif int(np.prod(desired_shape)) != len(flat):
                self._shape_display_var.set(f"Shape mismatch: {desired_shape} does not fit {len(flat)} values")
                self._shape_entry.configure(bg="#ffe6e6")
                if self._text is not None and self._text.winfo_exists():
                    self._text.configure(bg="#ffe6e6")
                return

            self._shape_display_var.set(f"Valid reshape: {resolved_shape}")
            self._shape_entry.configure(bg="#ffffff")
            if self._text is not None and self._text.winfo_exists():
                self._text.configure(bg="#ffffff")
            return

        self._shape_entry.configure(state="disabled", bg="#f2f2f2")
        rows, error = self._parse_numeric_rows(csv_text)
        if rows is None:
            self._shape_display_var.set("Detected shape: invalid")
            if self._text is not None and self._text.winfo_exists():
                self._text.configure(bg="#ffe6e6")
            return

        flat = [value for row in rows for value in row]
        shape = (len(rows), len(rows[0]))
        self._shape_display_var.set(f"Detected shape: {shape}")
        if self._text is not None and self._text.winfo_exists():
            self._text.configure(bg="#ffffff")

    def _parse_csv(self) -> np.ndarray | None:
        """
        Parse the text box content into a numpy array and honor the selected reshape strategy.
        Auto mode infers the natural 2D row-major shape from the CSV. Manual mode reshapes to
        the requested dimensions if the total count matches exactly.
        """
        csv_text = self._get_csv_text()
        rows, error = self._parse_numeric_rows(csv_text)
        if rows is None:
            return None

        dtype = np.dtype(self._dtype_var.get())
        flat = np.array([value for row in rows for value in row], dtype=dtype)

        if self._shape_mode_var.get() == "manual":
            shape = self._parse_shape_text(self._shape_entry_var.get())
            if shape is None:
                return None
            if -1 in shape:
                known_dims = [dim for dim in shape if dim != -1]
                known_product = int(np.prod(known_dims)) if known_dims else 1
                if known_product == 0 or flat.size % known_product != 0:
                    return None
                inferred = flat.size // known_product
                shape = tuple(inferred if dim == -1 else dim for dim in shape)
            elif int(np.prod(shape)) != flat.size:
                return None
            return flat.reshape(shape)

        if len(rows) == 1 and len(rows[0]) > 1:
            return np.array([rows[0]], dtype=dtype)
        if len(rows) == 1 and len(rows[0]) == 1:
            return np.array([[rows[0][0]]], dtype=dtype)
        return np.array(rows, dtype=dtype)

    def _on_apply(self) -> None:
        arr = self._parse_csv()
        if arr is None:
            self._status_var.set("parse error")
            self.set_status("error", "#cc0000")
            return

        self._last_array = arr
        shape_str = " x ".join(str(d) for d in arr.shape)
        self._status_var.set(
#            f"shape: ({shape_str})  {arr.dtype}")
            f"({shape_str})\n{arr.dtype}")
        self.set_status("ok", "#339966")

        if self._request_downstream:
            self._request_downstream(self.node_id)

    # ── compute / serialization ───────────────────────────────────

    def compute(self, inputs: dict) -> dict:
        if self._last_array is None:
            self._last_array = self._parse_csv()
        if self._last_array is None:
            return {}
        return {"array": self._last_array}

    def get_params(self) -> dict:
        if self._text is not None and self._text.winfo_exists():
            self._csv_text = self._text.get("1.0", tk.END)
        return {
            "csv": self._csv_text,
            "dtype": self._dtype_var.get(),
            "shape_mode": self._shape_mode_var.get(),
            "shape_text": self._shape_entry_var.get(),
        }

    def set_params(self, params: dict) -> None:
        self._init_state()
        self._csv_text = params.get("csv", self._csv_text)
        self._dtype_var.set(params.get("dtype", "float64"))
        self._shape_mode_var.set(params.get("shape_mode", "auto"))
        self._shape_entry_var.set(params.get("shape_text", ""))
        if self._text is not None and self._text.winfo_exists():
            self._text.delete("1.0", tk.END)
            self._text.insert("1.0", self._csv_text)
        if self._shape_entry is not None and self._shape_entry.winfo_exists():
            self._shape_entry.delete(0, tk.END)
            self._shape_entry.insert(0, self._shape_entry_var.get())
        self._refresh_shape_state()


class ArrayViewerNode(BaseNode):
    """
    Compact node body shows only the array shape on canvas.
    Double-click opens the detailed inspector with:
            - NumPy-style row/column sub-array selectors
            - Optional NumPy-style reshape for the output array
            - Paginated table with selected cells highlighted
      - Statistics panel (mean, std, min, max, sum)
      - Copy visible slice to clipboard as CSV
    """
    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE      = "array_viewer"
    DISPLAY_NAME   = "Array Viewer"
    CATEGORY       = "visualize"
    NODE_WIDTH     = 210
    NODE_HEIGHT    = 105

    MAX_ROWS = 200
    MAX_COLS = 50

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[PinDef("array", PinType.ARRAY, "in", shape=None)],
            outputs=[PinDef("array", PinType.ARRAY, "out", shape=None)]
        )

    def build_body(self) -> None:
        x, y, w, h = self.x, self.y, self.width, self.height

        self._body_rect = self.canvas.create_rectangle(
            x, y, x+w, y+h,
            fill="#e8f0f8", outline="#4477aa", width=1,
            tags=(self.node_id, "node_body"))
        self._title_item = self.canvas.create_text(
            x+w/2, y+12,
            text=self.DISPLAY_NAME,
            font=("Arial", 9, "bold"), fill="#223355",
            tags=(self.node_id,))

        self._shape_var  = tk.StringVar(value="no data")
        self._minmax_var = tk.StringVar(value="")

        shape_lbl = tk.Label(
            self.canvas, textvariable=self._shape_var,
            font=("Arial", 8, "bold"),
            bg="#e8f0f8", fg="#223355")
        self.canvas.create_window(
            x+4, y+h-16, window=shape_lbl, anchor="sw",
            tags=(self.node_id,))

        self.canvas.tag_bind(
            self.node_id, "<Double-Button-1>",
            lambda e: self._open_popup())

        self._canvas_items  += [self._body_rect, self._title_item]
        self._current_array : np.ndarray | None = None
        self._skip_next_input_update = False
        self._popup         : tk.Toplevel | None = None

        # Inspector detail widgets (created in build_inspector).
        self._popup_info = None
        self._rows_var = tk.StringVar(value=":")
        self._cols_var = tk.StringVar(value=":")
        self._reshape_var = tk.StringVar(value="")
        self._rows_entry = None
        self._cols_entry = None
        self._reshape_entry = None
        self._stats_var = tk.StringVar(value="")
        self._page_var = tk.IntVar(value=0)
        self._page_lbl = None
        self._table_canvas = None
        for variable in (self._rows_var, self._cols_var, self._reshape_var):
            variable.trace_add("write", self._on_selector_text_changed)

    def build_inspector(self, parent: tk.Frame) -> None:
        # Use the existing detailed array viewer as the node inspector.
        self._build_popup_ui(parent)
        self._refresh_popup()

    def close_inspector(self) -> None:
        super().close_inspector()
        self._popup_info = None
        self._rows_entry = None
        self._cols_entry = None
        self._reshape_entry = None
        self._page_lbl = None
        self._table_canvas = None

    # ── compute ───────────────────────────────────────────────────

    @staticmethod
    def _coerce_scalar_to_array(value):
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value.copy() if value.flags.writeable else value
        try:
            arr = np.asarray(value)
        except Exception:
            return None
        if arr.shape == ():
            return arr.reshape(1)
        return arr

    @staticmethod
    def _coerce_trigger_or_scalar_to_array(value):
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            arr = value.copy() if value.flags.writeable else value
            return arr

        try:
            arr = np.asarray(value)
        except Exception:
            return None

        if arr.shape == ():
            return arr.reshape(1)
        return arr

    def compute(self, inputs: dict) -> dict:
        if self._skip_next_input_update:
            self._skip_next_input_update = False
            raw = None
        else:
            raw = inputs.get("array")
        if raw is None:
            arr = self._current_array
            if arr is None:
                self._shape_var.set("no data")
                self._minmax_var.set("")
                self._current_array = None
                return {}
            self._current_array = arr
            self._update_summary(arr)
            if self.is_inspector_open():
                self._refresh_popup()
            try:
                if arr.ndim <= 1:
                    return {"array": arr.copy() if hasattr(arr, "copy") else arr}
                return {"array": self._get_selected_array(arr)}
            except Exception as e:
                self.set_status(f"error: {e}", color="#cc0000")
                return {}

        if isinstance(raw, np.ndarray):
            arr = raw.copy() if raw.flags.writeable else raw
        else:
            arr = self._coerce_trigger_or_scalar_to_array(raw)
        if arr is None:
            self._shape_var.set("no data")
            self._minmax_var.set("")
            self._current_array = None
            return {}

        self._current_array = arr
        self._update_summary(arr)
        if self.is_inspector_open():
            self._refresh_popup()
        try:
            if arr.ndim <= 1:
                return {"array": arr.copy() if hasattr(arr, "copy") else arr}
            return {"array": self._get_selected_array(arr)}
        except Exception as e:
            self.set_status(f"error: {e}", color="#cc0000")
            return {}

    def _update_summary(self, arr: np.ndarray) -> None:
        shape_str = "×".join(str(d) for d in arr.shape)
        self._shape_var.set(f"shape: ({shape_str})\n{arr.dtype}")
        self._minmax_var.set("")

    # ── popup ─────────────────────────────────────────────────────

    def _open_popup(self) -> None:
        # Keep backward-compatible method name, but route to inspector.
        self.open_inspector()

    def _build_popup_ui(self, win: tk.Misc) -> None:

        # ── top bar ───────────────────────────────────────────────
        top = tk.Frame(win, bg="#dde8f0", pady=4)
        top.pack(fill="x")

        self._popup_info = tk.Label(
            top, text="", bg="#dde8f0",
            font=("Arial", 9, "bold"))
        self._popup_info.pack(side="left", padx=8)

        # ── sub-array selector controls ───────────────────────────
        ctrl = tk.Frame(win, bg="#eef4f8", pady=3)
        ctrl.pack(fill="x", padx=4)

        tk.Label(ctrl, text="rows:", bg="#eef4f8",
                 font=("Arial", 8)).pack(side="left")
        self._rows_entry = tk.Entry(
            ctrl, textvariable=self._rows_var, width=16, font=("Arial", 8))
        self._rows_entry.pack(side="left", padx=(2, 8))
        tk.Label(ctrl, text="cols:", bg="#eef4f8",
                 font=("Arial", 8)).pack(side="left")
        self._cols_entry = tk.Entry(
            ctrl, textvariable=self._cols_var, width=16, font=("Arial", 8))
        self._cols_entry.pack(side="left", padx=(2, 8))
        tk.Label(ctrl, text="reshape:", bg="#eef4f8",
                 font=("Arial", 8)).pack(side="left")
        self._reshape_entry = tk.Entry(
            ctrl, textvariable=self._reshape_var, width=14, font=("Arial", 8))
        self._reshape_entry.pack(side="left", padx=(2, 8))
        tk.Button(ctrl, text="Apply",
                  font=("Arial", 8),
                  command=self._on_apply_selector).pack(side="left")

        # ── stats bar ─────────────────────────────────────────────
        self._stats_var = tk.StringVar(value="")
        tk.Label(win, textvariable=self._stats_var,
                 font=("Courier", 8),
                 bg="#f8f8f8", anchor="w").pack(
            fill="x", padx=4)

        # ── pagination ────────────────────────────────────────────
        pg = tk.Frame(win, bg="#eef4f8", pady=2)
        pg.pack(fill="x")
        tk.Button(pg, text="Prev",
                  command=lambda: self._change_page(-1),
                  font=("Arial", 8)).pack(side="left", padx=4)
        self._page_lbl = tk.Label(
            pg, text="", bg="#eef4f8",
            font=("Arial", 8))
        self._page_lbl.pack(side="left")
        tk.Button(pg, text="Next",
                  command=lambda: self._change_page(1),
                  font=("Arial", 8)).pack(side="left", padx=4)

        # copy actions
        tk.Button(pg, text="Copy full CSV",
                  font=("Arial", 8),
                  command=self._copy_full_csv).pack(
            side="right", padx=(0, 4))
        tk.Button(pg, text="Copy selected CSV",
                  font=("Arial", 8),
                  command=self._copy_selected_csv).pack(
            side="right", padx=4)

        # ── table ─────────────────────────────────────────────────
        tbl = tk.Frame(win)
        tbl.pack(fill="both", expand=True, padx=4, pady=4)
        self._table_canvas = tk.Canvas(tbl, bg="#ffffff", highlightthickness=0)
        vsb = ttk.Scrollbar(
            tbl, orient="vertical",   command=self._table_canvas.yview)
        hsb = ttk.Scrollbar(
            tbl, orient="horizontal", command=self._table_canvas.xview)
        self._table_canvas.configure(
            yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        hsb.pack(side="bottom", fill="x")
        vsb.pack(side="right",  fill="y")
        self._table_canvas.pack(fill="both", expand=True)

        self._page_var = tk.IntVar(value=0)

    # ── data extraction ───────────────────────────────────────────

    @staticmethod
    def _as_2d_view(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 0:
            return arr.reshape(1, 1)
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        if arr.ndim == 2:
            return arr
        return arr.reshape(-1, arr.shape[-1])

    @staticmethod
    def _parse_text_indices(text: str, size: int, name: str) -> np.ndarray:
        raw = text.strip()
        if raw == "" or raw == ":":
            return np.arange(size, dtype=np.int64)

        values: list[int] = []
        saw_token = False
        try:
            for token in raw.replace(";", ",").split(","):
                token = token.strip()
                if token == "":
                    continue
                saw_token = True
                if ":" not in token:
                    idx = int(token)
                    values.append(idx + size if idx < 0 else idx)
                    continue

                parts = token.split(":")
                if len(parts) > 3:
                    raise ValueError
                start = int(parts[0]) if parts[0] else None
                stop = int(parts[1]) if len(parts) > 1 and parts[1] else None
                step = int(parts[2]) if len(parts) > 2 and parts[2] else None
                start, stop, step = slice(start, stop, step).indices(size)
                values.extend(range(start, stop, step))
        except (TypeError, ValueError):
            raise ValueError(f"{name} text parse error") from None

        if not saw_token:
            raise ValueError(f"{name} is empty")
        indices = np.array(values, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= size):
            raise ValueError(f"{name} out of range for length {size}")
        return indices

    @staticmethod
    def _parse_reshape(text: str) -> tuple[int, ...] | None:
        raw = text.strip()
        if raw == "":
            return None
        try:
            shape = tuple(int(token.strip()) for token in raw.split(",") if token.strip())
        except ValueError:
            raise ValueError("reshape text parse error") from None
        if not shape:
            return None
        if shape.count(-1) > 1 or any(dim < -1 or dim == 0 for dim in shape):
            raise ValueError("reshape dimensions are invalid")
        return shape

    @staticmethod
    def _selector_is_explicit(text: str) -> bool:
        return text.strip() not in ("", ":")

    def _selector_state(self, arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mat = self._as_2d_view(arr)
        rows = self._parse_text_indices(self._rows_var.get(), mat.shape[0], "rows")
        cols = self._parse_text_indices(self._cols_var.get(), mat.shape[1], "cols")
        selected = mat[np.ix_(rows, cols)]
        reshape = self._parse_reshape(self._reshape_var.get())
        if reshape is not None:
            selected = selected.reshape(reshape)
        return mat, rows, cols, selected

    def _get_selected_array(self, arr: np.ndarray) -> np.ndarray:
        return self._selector_state(arr)[3]

    def _set_entry_validity(self, entry: tk.Entry | None, valid: bool) -> None:
        if entry is not None and entry.winfo_exists():
            entry.configure(fg="#000000" if valid else "#cc0000")

    def _validate_selector_text(self) -> bool:
        arr = self._current_array
        if arr is None:
            for entry in (self._rows_entry, self._cols_entry, self._reshape_entry):
                self._set_entry_validity(entry, True)
            return True
        mat = self._as_2d_view(arr)
        valid = True
        rows = cols = None
        try:
            rows = self._parse_text_indices(self._rows_var.get(), mat.shape[0], "rows")
            self._set_entry_validity(self._rows_entry, True)
        except ValueError:
            valid = False
            self._set_entry_validity(self._rows_entry, False)
        try:
            cols = self._parse_text_indices(self._cols_var.get(), mat.shape[1], "cols")
            self._set_entry_validity(self._cols_entry, True)
        except ValueError:
            valid = False
            self._set_entry_validity(self._cols_entry, False)
        try:
            reshape = self._parse_reshape(self._reshape_var.get())
            if reshape is not None and rows is not None and cols is not None:
                mat[np.ix_(rows, cols)].reshape(reshape)
            self._set_entry_validity(self._reshape_entry, True)
        except ValueError:
            valid = False
            self._set_entry_validity(self._reshape_entry, False)
        return valid

    def _on_selector_text_changed(self, *_args) -> None:
        self._validate_selector_text()
        if self.is_inspector_open():
            self._refresh_popup()

    def _on_apply_selector(self) -> None:
        if self._request_downstream:
            self._request_downstream(self.node_id)

    def _refresh_popup(self) -> None:
        arr = self._current_array
        if arr is None or self._popup_info is None or not self._popup_info.winfo_exists():
            return

        shape_str = " x ".join(str(d) for d in arr.shape)
        self._popup_info.config(
            text=f"shape: ({shape_str}) dtype: {arr.dtype}"
                 f" ndim: {arr.ndim}")

        # update stats
        try:
            mat, _rows, _cols, selected = self._selector_state(arr)
            flat = selected.ravel()
            self._stats_var.set(
                f"  view rows:{mat.shape[0]}  "
                f"cols:{mat.shape[1]}  |  "
                f"out:{selected.shape}  |  "
                f"mean:{flat.mean():.6g}  "
                f"std:{flat.std():.6g}  "
                f"min:{flat.min():.6g}  "
                f"max:{flat.max():.6g}  "
                f"sum:{flat.sum():.6g}")
        except Exception:
            self._stats_var.set("selector error")

        self._page_var.set(0)
        self._populate_table(arr)

    def _populate_table(self, arr: np.ndarray) -> None:
        if (self._table_canvas is None or not self._table_canvas.winfo_exists()
                or self._page_lbl is None):
            return
        try:
            mat, rows, cols, _selected = self._selector_state(arr)
        except Exception:
            self._table_canvas.delete("all")
            return

        nrows, ncols = mat.shape
        page    = self._page_var.get()
        n_pages = max(1, (nrows + self.MAX_ROWS - 1) // self.MAX_ROWS)
        page    = max(0, min(page, n_pages - 1))
        self._page_var.set(page)
        self._page_lbl.config(
            text=f"  page {page+1}/{n_pages}  "
                 f"(rows {page*self.MAX_ROWS}-"
                 f"{min((page+1)*self.MAX_ROWS, nrows)-1}"
                 f" of {nrows})  ")

        row_start = page * self.MAX_ROWS
        row_end   = min(row_start + self.MAX_ROWS, nrows)
        col_end   = min(ncols, self.MAX_COLS)
        mat_page  = mat[row_start:row_end, :col_end]

        self._draw_canvas_table(mat_page, row_start, rows, cols, mat.dtype)

    @staticmethod
    def _format_cell_value(value, is_float: bool) -> str:
        return f"{value:.6g}" if is_float else str(value)

    def _draw_canvas_table(self, mat_page: np.ndarray, row_start: int,
                           rows: np.ndarray, cols: np.ndarray,
                           dtype: np.dtype) -> None:
        canvas = self._table_canvas
        if canvas is None or not canvas.winfo_exists():
            return

        canvas.delete("all")
        row_header_w = 58
        cell_w = 88
        cell_h = 22
        header_h = 24
        n_page_rows, n_page_cols = mat_page.shape
        total_w = row_header_w + n_page_cols * cell_w
        total_h = header_h + n_page_rows * cell_h
        canvas.configure(scrollregion=(0, 0, total_w, total_h))

        selected_rows = set(int(v) for v in rows.tolist())
        selected_cols = set(int(v) for v in cols.tolist())
        explicit_rows = self._selector_is_explicit(self._rows_var.get())
        explicit_cols = self._selector_is_explicit(self._cols_var.get())
        highlight = explicit_rows or explicit_cols

        canvas.create_rectangle(0, 0, row_header_w, header_h,
                                fill="#dfe7f0", outline="#c0c8d0")
        canvas.create_text(row_header_w - 6, header_h / 2,
                           text="row", anchor="e", font=("Arial", 8, "bold"))
        for c in range(n_page_cols):
            x0 = row_header_w + c * cell_w
            canvas.create_rectangle(x0, 0, x0 + cell_w, header_h,
                                    fill="#dfe7f0", outline="#c0c8d0")
            canvas.create_text(x0 + cell_w - 6, header_h / 2,
                               text=str(c), anchor="e", font=("Arial", 8, "bold"))

        is_float = np.issubdtype(dtype, np.floating)
        for r, row in enumerate(mat_page):
            source_row = row_start + r
            y0 = header_h + r * cell_h
            base_colour = "#f0f5ff" if r % 2 == 0 else "#ffffff"
            canvas.create_rectangle(0, y0, row_header_w, y0 + cell_h,
                                    fill=base_colour, outline="#d6dde5")
            canvas.create_text(row_header_w - 6, y0 + cell_h / 2,
                               text=str(source_row), anchor="e", font=("Arial", 8))
            for c, value in enumerate(row):
                selected = (source_row in selected_rows and c in selected_cols)
                fill = "#aee8ff" if highlight and selected else base_colour
                x0 = row_header_w + c * cell_w
                canvas.create_rectangle(x0, y0, x0 + cell_w, y0 + cell_h,
                                        fill=fill, outline="#d6dde5")
                canvas.create_text(x0 + cell_w - 6, y0 + cell_h / 2,
                                   text=self._format_cell_value(value, is_float),
                                   anchor="e", font=("Courier", 8))

    def _change_page(self, delta: int) -> None:
        arr = self._current_array
        if arr is None:
            return
        try:
            mat = self._as_2d_view(arr)
            nrows   = mat.shape[0]
            n_pages = max(1, (nrows + self.MAX_ROWS - 1)
                          // self.MAX_ROWS)
            new_page = max(0, min(
                self._page_var.get() + delta, n_pages - 1))
            self._page_var.set(new_page)
            self._populate_table(arr)
        except Exception:
            pass

    @staticmethod
    def _array_to_csv_text(arr: np.ndarray) -> str:
        """Serialize a numpy array to CSV text without losing effective float precision."""
        if arr is None:
            return ""
        mat = np.asarray(arr)
        if mat.ndim == 0:
            mat = mat.reshape(1, 1)
        elif mat.ndim == 1:
            mat = mat.reshape(1, -1)
        elif mat.ndim > 2:
            mat = mat.reshape(-1, mat.shape[-1])

        rows: list[str] = []
        if np.issubdtype(mat.dtype, np.floating):
            float_precision = 17 if np.dtype(mat.dtype).itemsize >= 8 else 9
            for row in mat:
                values = []
                for v in row:
                    if np.isnan(v):
                        values.append("nan")
                    elif np.isinf(v):
                        values.append("inf" if v > 0 else "-inf")
                    else:
                        values.append(f"{float(v):.{float_precision}g}")
                rows.append(",".join(values))
        else:
            for row in mat:
                rows.append(",".join(str(v) for v in row))
        return "\n".join(rows)

    def _copy_array_to_clipboard(self, arr: np.ndarray | None) -> None:
        if arr is None:
            return
        try:
            csv_text = self._array_to_csv_text(arr)
            host = self._popup_info if (self._popup_info is not None and self._popup_info.winfo_exists()) else self.canvas
            host.clipboard_clear()
            host.clipboard_append(csv_text)
        except Exception:
            pass

    def _copy_full_csv(self) -> None:
        """Copy the full array view to the clipboard as CSV."""
        arr = self._current_array
        if arr is None:
            return
        self._copy_array_to_clipboard(self._as_2d_view(arr))

    def _copy_selected_csv(self) -> None:
        """Copy the currently selected output array as CSV to clipboard."""
        arr = self._current_array
        if arr is None:
            return
        self._copy_array_to_clipboard(self._get_selected_array(arr))

    def _close_popup(self) -> None:
        self.close_inspector()

    def on_destroy(self) -> None:
        self._close_popup()
        super().on_destroy()
    def _project_sidecar_filename(self) -> str | None:
        project_path = get_project_file_path()
        if project_path is None:
            return None
        safe_node_id = "".join(
            character if character.isalnum() or character in ("-", "_") else "_"
            for character in self.node_id
        )
        return f"{project_path.stem}.{safe_node_id}.array_viewer.npy"

    def _write_project_sidecar(self, filename: str) -> None:
        project_dir = get_project_directory()
        if project_dir is None:
            raise ValueError("project path is required to save array viewer data")

        path = project_dir / filename
        if self._current_array is None:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            return

        temporary_path = path.with_name(path.name + ".tmp")
        try:
            with open(temporary_path, "wb") as file:
                np.save(file, np.asarray(self._current_array), allow_pickle=False)
            temporary_path.replace(path)
        except Exception:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            raise

    def _load_project_sidecar(self, filename: str) -> np.ndarray | None:
        project_dir = get_project_directory()
        if project_dir is None or Path(filename).name != filename:
            return None
        try:
            with open(project_dir / filename, "rb") as file:
                return np.load(file, allow_pickle=False)
        except Exception:
            return None

    def get_params(self) -> dict:
        arr = self._current_array if self._current_array is not None else None
        params = {
            "rows_text": self._rows_var.get(),
            "cols_text": self._cols_var.get(),
            "reshape_text": self._reshape_var.get(),
        }
        sidecar_filename = self._project_sidecar_filename()
        if sidecar_filename is not None:
            self._write_project_sidecar(sidecar_filename)
            if arr is not None:
                params["array_viewer_file"] = sidecar_filename
        else:
            params["data"] = None if arr is None else np.asarray(arr).tolist()
        return params

    def set_params(self, params: dict) -> None:
        self._rows_var.set(str(params.get("rows_text", ":")))
        self._cols_var.set(str(params.get("cols_text", ":")))
        self._reshape_var.set(str(params.get("reshape_text", "")))

        sidecar_filename = str(params.get("array_viewer_file", "")).strip()
        raw_data = self._load_project_sidecar(sidecar_filename) if sidecar_filename else params.get("data")
        if raw_data is None:
            self._current_array = None
            self._shape_var.set("no data")
            self._minmax_var.set("")
            return

        try:
            arr = np.asarray(raw_data)
            if arr.size == 0:
                self._current_array = arr.reshape((0,))
            else:
                self._current_array = arr
        except Exception:
            self._current_array = None
            self._shape_var.set("no data")
            self._minmax_var.set("")
            return

        self._skip_next_input_update = bool(sidecar_filename)
        self._update_summary(self._current_array)
        if self.is_inspector_open():
            self._refresh_popup()

# node_editor/nodes/array_nodes.py — add after ArrayViewerNode

import os
from tkinter import filedialog


class NpyFileInputNode(BaseNode):
    """
    Loads a .npy or .npz file from disk.
    For .npz, a key selector combobox appears after loading.
    Triggers downstream automatically on file load / key change.
    """
    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE      = "npy_file_input"
    DISPLAY_NAME   = "Npy File Input"
    CATEGORY       = "source"
    NODE_WIDTH     = 220
    NODE_HEIGHT    = 110

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[],
            outputs=[PinDef("array", PinType.ARRAY, "out", shape=None)]
        )

    def _init_state(self) -> None:
        if hasattr(self, "_path_var"):
            return
        self._path_var = tk.StringVar(value="(no file)")
        self._key_var = tk.StringVar(value="")
        self._status_var = tk.StringVar(value="")
        self._key_cb = None
        self._filepath: str = ""
        self._npz_data: object = None
        self._last_array: np.ndarray | None = None

    def build_body(self) -> None:
        self._init_state()
        x, y, w, h = self.x, self.y, self.width, self.height

        # Compact canvas shell (Phase 2).
        self._body_rect = self.canvas.create_rectangle(
            x, y, x+w, y+h,
            fill="#f0f8e8", outline="#558833", width=1,
            tags=(self.node_id, "node_body"))
        self._title_item = self.canvas.create_text(
            x+w/2, y+12,
            text=self.DISPLAY_NAME,
            font=("Arial", 9, "bold"), fill="#224400",
            tags=(self.node_id,))

        status_lbl = tk.Label(
            self.canvas, textvariable=self._status_var,
            font=("Arial", 8), bg="#f0f8e8", fg="#335500")
        self.canvas.create_window(
            x+w/2, y+h-14, window=status_lbl,
            tags=(self.node_id,))

        self._canvas_items += [self._body_rect, self._title_item]

    def build_inspector(self, parent: tk.Frame) -> None:
        self._init_state()

        tk.Label(parent, textvariable=self._path_var, anchor="w", justify="left", wraplength=360,
                 font=("Arial", 8)).pack(fill="x")

        tk.Button(parent, text="Browse...", font=("Arial", 9), command=self._on_browse).pack(anchor="w", pady=(6, 6))

        self._key_cb = ttk.Combobox(parent, textvariable=self._key_var, values=[], width=20, state="readonly")
        self._key_cb.bind("<<ComboboxSelected>>", lambda e: self._load_array())
        self._sync_key_widget_visibility()
        if self._filepath.endswith(".npz"):
            keys = list(self._npz_data.keys()) if self._npz_data is not None else []
            self._key_cb.configure(values=keys)

        tk.Label(parent, textvariable=self._status_var, anchor="w", justify="left", font=("Arial", 9)).pack(fill="x", pady=(6, 0))

    def close_inspector(self) -> None:
        super().close_inspector()
        self._key_cb = None

    def _sync_key_widget_visibility(self) -> None:
        if self._key_cb is None or not self._key_cb.winfo_exists():
            return
        if self._filepath.endswith(".npz"):
            if self._key_cb.winfo_manager() == "":
                self._key_cb.pack(anchor="w")
        else:
            if self._key_cb.winfo_manager() != "":
                self._key_cb.pack_forget()

    # ── file loading ──────────────────────────────────────────────

    def _on_browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Select NumPy file",
            filetypes=[
                ("NumPy files", "*.npy *.npz"),
                ("All files",   "*.*"),
            ])
        if not path:
            return
        self._filepath = path
        self._path_var.set(os.path.basename(path))
        self._load_file()

    def _load_file(self) -> None:
        try:
            if self._filepath.endswith(".npz"):
                self._npz_data = np.load(
                    self._filepath, allow_pickle=False)
                keys = list(self._npz_data.keys())
                if self._key_cb is not None and self._key_cb.winfo_exists():
                    self._key_cb.configure(values=keys)
                self._key_var.set(keys[0] if keys else "")
                self._sync_key_widget_visibility()
                self._load_array()
            else:
                self._npz_data = None
                self._sync_key_widget_visibility()
                self._load_array()
        except Exception as e:
            self._status_var.set(f"error: {e}")
            self.set_status("error", "#cc0000")

    def _load_array(self) -> None:
        try:
            if self._filepath.endswith(".npz"):
                key = self._key_var.get()
                if not key:
                    return
                arr = self._npz_data[key]
            else:
                arr = np.load(
                    self._filepath, allow_pickle=False)

            self._last_array = arr
            shape_str = " x ".join(str(d) for d in arr.shape)
            self._status_var.set(
                f"({shape_str})  {arr.dtype}")
            self.set_status("ok", "#339966")

            if self._request_downstream:
                self._request_downstream(self.node_id)

        except Exception as e:
            self._status_var.set(f"error: {e}")
            self.set_status("error", "#cc0000")

    # ── compute / serialization ───────────────────────────────────

    def compute(self, inputs: dict) -> dict:
        if self._last_array is None:
            return {}
        return {"array": self._last_array}

    def get_params(self) -> dict:
        return {
            "filepath": self._filepath,
            "key":      self._key_var.get(),
        }

    def set_params(self, params: dict) -> None:
        self._init_state()
        self._filepath = params.get("filepath", "")
        if self._filepath:
            self._path_var.set(os.path.basename(self._filepath))
            self._load_file()
            key = params.get("key", "")
            if key:
                self._key_var.set(key)
                self._load_array()

    def on_destroy(self) -> None:
        if self._npz_data is not None:
            self._npz_data.close()
        super().on_destroy()

