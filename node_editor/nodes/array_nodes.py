# node_editor/nodes/array_nodes.py

import tkinter as tk
from tkinter import ttk
import numpy as np

from node_editor.base_node import BaseNode
from node_editor.pin_types import PinSchema, PinDef, PinType
from node_editor.execution import ExecutionMode


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
        self._status_var = tk.StringVar(value="shape: ?")
        self._csv_text = "1, 2, 3\n4, 5, 6"
        self._text = None
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

        tk.Label(parent, text="CSV rows (comma-separated):", font=("Arial", 9)).grid(
            row=1, column=0, columnspan=2, sticky="w", pady=(8, 2)
        )

        frame = tk.Frame(parent, bd=1, relief=tk.SUNKEN)
        frame.grid(row=2, column=0, columnspan=2, sticky="nsew")
        self._text = tk.Text(frame, width=40, height=10, font=("Courier", 9), wrap=tk.NONE)
        vsb = tk.Scrollbar(frame, orient="vertical", command=self._text.yview)
        self._text.configure(yscrollcommand=vsb.set)
        self._text.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        self._text.insert("1.0", self._csv_text)

        apply_btn = tk.Button(parent, text="Apply", font=("Arial", 9), command=self._on_apply)
        apply_btn.grid(row=3, column=1, sticky="e", pady=(8, 2))

        status_lbl = tk.Label(parent, textvariable=self._status_var, font=("Arial", 9), fg="#553300")
        status_lbl.grid(row=3, column=0, sticky="w", pady=(8, 2))

        parent.grid_rowconfigure(2, weight=1)
        parent.grid_columnconfigure(1, weight=1)

    def close_inspector(self) -> None:
        if self._text is not None and self._text.winfo_exists():
            self._csv_text = self._text.get("1.0", tk.END)
        super().close_inspector()
        self._text = None

    # ── parsing ──────────────────────────────────────────────────

    def _parse_csv(self) -> np.ndarray | None:
        """
        Parse the text box content into a numpy array.
        Each line is a row; values within a line are comma-separated.
        Handles both 1D (single line or single column) and 2D cases.
        """
        if self._text is not None and self._text.winfo_exists():
            self._csv_text = self._text.get("1.0", tk.END)
        raw = self._csv_text.strip()
        if not raw:
            return None
        try:
            dtype = np.dtype(self._dtype_var.get())
            rows  = []
            for line in raw.splitlines():
                line = line.strip()
                if not line:
                    continue
                rows.append([float(v) for v in line.split(",")])

            if len(rows) == 1:
                arr = np.array(rows[0], dtype=dtype)       # 1D
            else:
                arr = np.array(rows, dtype=dtype)           # 2D
            return arr
        except Exception as e:
            return None

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
            f"({shape_str})  {arr.dtype}")
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
            "csv":   self._csv_text,
            "dtype": self._dtype_var.get(),
        }

    def set_params(self, params: dict) -> None:
        self._init_state()
        self._csv_text = params.get("csv", self._csv_text)
        self._dtype_var.set(params.get("dtype", "float64"))
        if self._text is not None and self._text.winfo_exists():
            self._text.delete("1.0", tk.END)
            self._text.insert("1.0", self._csv_text)


class ArrayViewerNode(BaseNode):
    """
    Compact node body shows shape, min/max, and a corner preview.
    Double-click opens a full popup with:
      - Slice selectors for each leading axis (ndim > 2)
      - Axis transpose controls (swap row/col axes)
      - Paginated table with alternating row colours
      - Statistics panel (mean, std, min, max, sum)
      - Copy visible slice to clipboard as CSV
      - Optional heatmap cell colouring
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
            outputs=[]
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

        self._shape_var   = tk.StringVar(value="no data")
        self._minmax_var  = tk.StringVar(value="")
        self._preview_var = tk.StringVar(value="")

        shape_lbl = tk.Label(
            self.canvas, textvariable=self._shape_var,
            font=("Arial", 8, "bold"),
            bg="#e8f0f8", fg="#223355")
        self.canvas.create_window(
            x+4, y+28, window=shape_lbl, anchor="nw",
            tags=(self.node_id,))

        minmax_lbl = tk.Label(
            self.canvas, textvariable=self._minmax_var,
            font=("Arial", 8),
            bg="#e8f0f8", fg="#446688")
        self.canvas.create_window(
            x+4, y+44, window=minmax_lbl, anchor="nw",
            tags=(self.node_id,))

        preview_lbl = tk.Label(
            self.canvas, textvariable=self._preview_var,
            font=("Courier", 7),
            bg="#e8f0f8", fg="#334466",
            justify="left")
        self.canvas.create_window(
            x+4, y+60, window=preview_lbl, anchor="nw",
            tags=(self.node_id,))

        self.canvas.tag_bind(
            self.node_id, "<Double-Button-1>",
            lambda e: self._open_popup())

        self._canvas_items  += [self._body_rect, self._title_item]
        self._current_array : np.ndarray | None = None
        self._popup         : tk.Toplevel | None = None

        # Inspector detail widgets (created in build_inspector).
        self._popup_info = None
        self._slice_frame = None
        self._slice_vars = []
        self._row_axis_var = tk.IntVar(value=-2)
        self._col_axis_var = tk.IntVar(value=-1)
        self._row_axis_sb = None
        self._col_axis_sb = None
        self._heatmap_var = tk.BooleanVar(value=False)
        self._stats_var = tk.StringVar(value="")
        self._page_var = tk.IntVar(value=0)
        self._page_lbl = None
        self._tree = None

    def build_inspector(self, parent: tk.Frame) -> None:
        # Use the existing detailed array viewer as the node inspector.
        self._build_popup_ui(parent)
        self._refresh_popup()

    def close_inspector(self) -> None:
        super().close_inspector()
        self._popup_info = None
        self._slice_frame = None
        self._slice_vars = []
        self._row_axis_sb = None
        self._col_axis_sb = None
        self._page_lbl = None
        self._tree = None

    # ── compute ───────────────────────────────────────────────────

    def compute(self, inputs: dict) -> dict:
        arr = inputs.get("array")
        if arr is None or not isinstance(arr, np.ndarray):
            self._shape_var.set("no data")
            self._minmax_var.set("")
            self._preview_var.set("")
            self._current_array = None
            return {}

        self._current_array = arr
        self._update_summary(arr)
        if self._popup and self._popup.winfo_exists():
            self._refresh_popup()
        return {}

    def _update_summary(self, arr: np.ndarray) -> None:
        shape_str = "×".join(str(d) for d in arr.shape)
        self._shape_var.set(f"shape: ({shape_str})  {arr.dtype}")
        try:
            self._minmax_var.set(
                f"min: {arr.min():.4g}   max: {arr.max():.4g}"
                f"   mean: {arr.mean():.4g}")
        except Exception:
            self._minmax_var.set("")

        # corner preview
        try:
            lines = []
            if arr.ndim == 1:
                vals = "  ".join(f"{v:.3g}" for v in arr[:6])
                lines.append(vals + (" ..." if len(arr) > 6 else ""))
            else:
                src = arr if arr.ndim == 2 else arr.reshape(
                    -1, arr.shape[-1])
                for row in src[:3]:
                    cells = "  ".join(
                        f"{v:.3g}" for v in row[:4])
                    lines.append(
                        cells + (" ..." if arr.shape[-1] > 4 else ""))
                if src.shape[0] > 3:
                    lines.append("  ...")
            self._preview_var.set("\n".join(lines))
        except Exception:
            self._preview_var.set("(preview unavailable)")

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

        # ── slice & transpose controls ────────────────────────────
        ctrl = tk.Frame(win, bg="#eef4f8", pady=3)
        ctrl.pack(fill="x", padx=4)

        # leading-axis slice spinboxes (for ndim > 2)
        self._slice_frame = tk.Frame(ctrl, bg="#eef4f8")
        self._slice_frame.pack(side="left")
        self._slice_vars: list[tk.IntVar] = []

        # row/col axis selectors (for ndim >= 2)
        ax_frame = tk.Frame(ctrl, bg="#eef4f8")
        ax_frame.pack(side="left", padx=12)
        tk.Label(ax_frame, text="row axis:",
                 bg="#eef4f8",
                 font=("Arial", 8)).pack(side="left")
        self._row_axis_var = tk.IntVar(value=-2)
        self._row_axis_sb  = tk.Spinbox(
            ax_frame, from_=-10, to=10,
            textvariable=self._row_axis_var,
            width=3, command=self._refresh_popup)
        self._row_axis_sb.pack(side="left", padx=2)

        tk.Label(ax_frame, text="  col axis:",
                 bg="#eef4f8",
                 font=("Arial", 8)).pack(side="left")
        self._col_axis_var = tk.IntVar(value=-1)
        self._col_axis_sb  = tk.Spinbox(
            ax_frame, from_=-10, to=10,
            textvariable=self._col_axis_var,
            width=3, command=self._refresh_popup)
        self._col_axis_sb.pack(side="left", padx=2)

        # heatmap toggle
        self._heatmap_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            ctrl, text="heatmap",
            variable=self._heatmap_var,
            bg="#eef4f8", font=("Arial", 8),
            command=self._refresh_popup).pack(
            side="left", padx=8)

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

        # copy button
        tk.Button(pg, text="Copy CSV",
                  font=("Arial", 8),
                  command=self._copy_csv).pack(
            side="right", padx=8)

        # ── table ─────────────────────────────────────────────────
        tbl = tk.Frame(win)
        tbl.pack(fill="both", expand=True, padx=4, pady=4)
        self._tree = ttk.Treeview(
            tbl, show="headings", selectmode="browse")
        vsb = ttk.Scrollbar(
            tbl, orient="vertical",   command=self._tree.yview)
        hsb = ttk.Scrollbar(
            tbl, orient="horizontal", command=self._tree.xview)
        self._tree.configure(
            yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        hsb.pack(side="bottom", fill="x")
        vsb.pack(side="right",  fill="y")
        self._tree.pack(fill="both", expand=True)

        self._page_var = tk.IntVar(value=0)

    # ── data extraction ───────────────────────────────────────────

    def _get_2d_slice(self, arr: np.ndarray) -> np.ndarray:
        """
        Extract a 2D slice from an N-dimensional array.

        For ndim > 2: apply slice_vars on all axes except
        the chosen row_axis and col_axis.
        For ndim == 2: apply row/col axis directly.
        For ndim == 1: return as column vector.
        """
        if arr.ndim == 1:
            return arr.reshape(-1, 1)

        ndim     = arr.ndim
        row_axis = self._row_axis_var.get() % ndim
        col_axis = self._col_axis_var.get() % ndim

        if row_axis == col_axis:
            col_axis = (row_axis + 1) % ndim

        # Build index tuple: slice for row/col axes,
        # integer index for all other (leading) axes
        idx        = [0] * ndim
        idx[row_axis] = slice(None)
        idx[col_axis] = slice(None)

        # fill leading axis indices from spinboxes
        slice_ax = 0
        for ax in range(ndim):
            if ax == row_axis or ax == col_axis:
                continue
            if slice_ax < len(self._slice_vars):
                idx[ax] = min(
                    self._slice_vars[slice_ax].get(),
                    arr.shape[ax] - 1)
            slice_ax += 1

        result = arr[tuple(idx)]

        # ensure rows correspond to row_axis
        # after fancy indexing the two surviving axes may be
        # in the wrong order
        if result.ndim == 2:
            surviving = [ax for ax in range(ndim)
                         if ax in (row_axis, col_axis)]
            if surviving[0] == col_axis:
                result = result.T
        elif result.ndim != 2:
            result = result.reshape(-1, 1)

        return result

    def _refresh_popup(self) -> None:
        arr = self._current_array
        if arr is None or self._popup_info is None or not self._popup_info.winfo_exists():
            return

        shape_str = " x ".join(str(d) for d in arr.shape)
        self._popup_info.config(
            text=f"shape: ({shape_str})   dtype: {arr.dtype}"
                 f"   ndim: {arr.ndim}")

        # rebuild leading-axis slice spinboxes
        for w in self._slice_frame.winfo_children():
            w.destroy()
        self._slice_vars = []

        ndim     = arr.ndim
        row_axis = self._row_axis_var.get() % ndim
        col_axis = self._col_axis_var.get() % ndim
        if row_axis == col_axis:
            col_axis = (row_axis + 1) % ndim

        if ndim > 2:
            tk.Label(self._slice_frame,
                     text="slice:  ",
                     bg="#eef4f8",
                     font=("Arial", 8)).pack(side="left")
            for ax in range(ndim):
                if ax in (row_axis, col_axis):
                    continue
                var = tk.IntVar(value=0)
                self._slice_vars.append(var)
                tk.Label(self._slice_frame,
                         text=f"ax{ax}:",
                         bg="#eef4f8",
                         font=("Arial", 8)).pack(side="left")
                tk.Spinbox(
                    self._slice_frame,
                    from_=0, to=arr.shape[ax]-1,
                    textvariable=var, width=4,
                    command=self._refresh_popup).pack(
                    side="left", padx=2)

        # update stats
        try:
            mat = self._get_2d_slice(arr)
            flat = mat.ravel()
            self._stats_var.set(
                f"  slice  rows:{mat.shape[0]}  "
                f"cols:{mat.shape[1]}  |  "
                f"mean:{flat.mean():.6g}  "
                f"std:{flat.std():.6g}  "
                f"min:{flat.min():.6g}  "
                f"max:{flat.max():.6g}  "
                f"sum:{flat.sum():.6g}")
        except Exception:
            self._stats_var.set("")

        self._page_var.set(0)
        self._populate_table(arr)

    def _populate_table(self, arr: np.ndarray) -> None:
        if self._tree is None or not self._tree.winfo_exists() or self._page_lbl is None:
            return
        try:
            mat = self._get_2d_slice(arr)
        except Exception:
            return

        nrows, ncols = mat.shape
        page    = self._page_var.get()
        n_pages = max(1, (nrows + self.MAX_ROWS - 1) // self.MAX_ROWS)
        page    = max(0, min(page, n_pages - 1))
        self._page_var.set(page)
        self._page_lbl.config(
            text=f"  page {page+1}/{n_pages}  "
                 f"(rows {page*self.MAX_ROWS}–"
                 f"{min((page+1)*self.MAX_ROWS, nrows)-1}"
                 f" of {nrows})  ")

        row_start = page * self.MAX_ROWS
        row_end   = min(row_start + self.MAX_ROWS, nrows)
        col_end   = min(ncols, self.MAX_COLS)
        mat_page  = mat[row_start:row_end, :col_end]

        # heatmap colour scale for current page
        do_heat = self._heatmap_var.get()
        if do_heat:
            try:
                vmin = float(mat_page.min())
                vmax = float(mat_page.max())
                vrange = vmax - vmin if vmax != vmin else 1.0
            except Exception:
                do_heat = False

        # rebuild columns
        col_ids = ["#idx"] + [str(c) for c in range(col_end)]
        self._tree.configure(columns=col_ids)
        self._tree.heading("#idx", text="row")
        self._tree.column("#idx", width=52, anchor="e")
        col_w = max(55, min(90, 700 // max(col_end, 1)))
        for c in range(col_end):
            self._tree.heading(str(c), text=str(c))
            self._tree.column(str(c), width=col_w, anchor="e")

        for item in self._tree.get_children():
            self._tree.delete(item)

        is_float = np.issubdtype(arr.dtype, np.floating)
        for r, row in enumerate(mat_page):
            vals = [str(row_start + r)]
            for v in row:
                vals.append(
                    f"{v:.6g}" if is_float else str(v))
            tag = f"row_{r}"
            self._tree.insert(
                "", "end", values=vals, tags=(tag,))

            if do_heat:
                # map each row's mean to a blue→red colour
                row_mean = float(row.mean())
                t = (row_mean - vmin) / vrange   # 0..1
                r_col = int(180 * t)
                b_col = int(180 * (1 - t))
                colour = f"#{r_col:02x}e0{b_col:02x}"
            else:
                colour = "#f0f5ff" if r % 2 == 0 else "#ffffff"
            self._tree.tag_configure(tag, background=colour)

    def _change_page(self, delta: int) -> None:
        arr = self._current_array
        if arr is None:
            return
        try:
            mat     = self._get_2d_slice(arr)
            nrows   = mat.shape[0]
            n_pages = max(1, (nrows + self.MAX_ROWS - 1)
                          // self.MAX_ROWS)
            new_page = max(0, min(
                self._page_var.get() + delta, n_pages - 1))
            self._page_var.set(new_page)
            self._populate_table(arr)
        except Exception:
            pass

    def _copy_csv(self) -> None:
        """Copy the currently visible 2D slice as CSV to clipboard."""
        arr = self._current_array
        if arr is None:
            return
        try:
            mat  = self._get_2d_slice(arr)
            rows = []
            for row in mat:
                rows.append(",".join(
                    f"{v:.8g}" if np.issubdtype(
                        arr.dtype, np.floating)
                    else str(v)
                    for v in row))
            csv_text = "\n".join(rows)
            host = self._popup_info if (self._popup_info is not None and self._popup_info.winfo_exists()) else self.canvas
            host.clipboard_clear()
            host.clipboard_append(csv_text)
        except Exception as e:
            pass

    def _close_popup(self) -> None:
        self.close_inspector()

    def on_destroy(self) -> None:
        self._close_popup()
        super().on_destroy()

    def get_params(self) -> dict:
        return {}

    def set_params(self, params: dict) -> None:
        pass

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


class SubArraySelectorNode(BaseNode):
    """
    Select a 2D sub-array using 0-based row/column indices.

    Output is equivalent to:
      data[np.ix_(rows, cols)]

    rows/cols can come from input pins or from the node text boxes.
    Pin inputs take priority over text boxes.
    """

    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE = "sub_array_selector"
    DISPLAY_NAME = "Sub-Array Selector"
    CATEGORY = "process"
    NODE_WIDTH = 230
    NODE_HEIGHT = 120

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[
                PinDef("data", PinType.ARRAY, "data", optional=False),
                PinDef("rows", PinType.ARRAY, "rows", optional=True),
                PinDef("cols", PinType.ARRAY, "cols", optional=True),
            ],
            outputs=[
                PinDef("array", PinType.ARRAY, "out", shape=None),
            ],
        )

    def _init_state(self) -> None:
        if hasattr(self, "_rows_var"):
            return
        self._rows_var = tk.StringVar(value="0,1")
        self._cols_var = tk.StringVar(value="0,1")
        self._reshape_var = tk.StringVar(value="")
        self._status_var = tk.StringVar(value="waiting for data")
        self._rows_entry = None
        self._cols_entry = None
        self._reshape_entry = None
        self._current_data: np.ndarray | None = None
        for variable in (self._rows_var, self._cols_var, self._reshape_var):
            variable.trace_add("write", self._on_selector_text_changed)

    def build_body(self) -> None:
        self._init_state()
        x, y, w, h = self.x, self.y, self.width, self.height

        self._body_rect = self.canvas.create_rectangle(
            x, y, x + w, y + h,
            fill="#f2f6ec", outline="#6f9a52", width=1,
            tags=(self.node_id, "node_body"),
        )
        self._title_item = self.canvas.create_text(
            x + w / 2, y + 12,
            text=self.DISPLAY_NAME,
            font=("Arial", 9, "bold"), fill="#2f4f1f",
            tags=(self.node_id,),
        )

        status_lbl = tk.Label(
            self.canvas, textvariable=self._status_var,
            font=("Arial", 8), bg="#f2f6ec", fg="#2f4f1f",
        )
        self.canvas.create_window(
            x + w / 2, y + h - 14, window=status_lbl,
            tags=(self.node_id,),
        )

        self._canvas_items += [self._body_rect, self._title_item]

    def build_inspector(self, parent: tk.Frame) -> None:
        self._init_state()

        tk.Label(parent, text="rows (0-based):", font=("Arial", 9)).grid(row=0, column=0, sticky="w", pady=2)
        self._rows_entry = tk.Entry(parent, textvariable=self._rows_var, width=24, font=("Arial", 9))
        self._rows_entry.grid(row=0, column=1, sticky="ew", pady=2)

        tk.Label(parent, text="cols (0-based):", font=("Arial", 9)).grid(row=1, column=0, sticky="w", pady=2)
        self._cols_entry = tk.Entry(parent, textvariable=self._cols_var, width=24, font=("Arial", 9))
        self._cols_entry.grid(row=1, column=1, sticky="ew", pady=2)

        tk.Label(parent, text="reshape:", font=("Arial", 9)).grid(row=2, column=0, sticky="w", pady=2)
        self._reshape_entry = tk.Entry(parent, textvariable=self._reshape_var, width=24, font=("Arial", 9))
        self._reshape_entry.grid(row=2, column=1, sticky="ew", pady=2)

        tk.Label(
            parent,
            text="Indices are 0-based: 0 is first, -1 is last. Ranges are inclusive (0:2 selects 0, 1, 2).",
            justify="left", anchor="w", wraplength=360, font=("Arial", 8), fg="#555555",
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(6, 2))
        tk.Button(parent, text="Apply", font=("Arial", 9), command=self._on_apply).grid(row=4, column=1, sticky="e", pady=(6, 2))
        tk.Label(parent, textvariable=self._status_var, font=("Arial", 9)).grid(row=4, column=0, sticky="w", pady=(6, 2))

        parent.grid_columnconfigure(1, weight=1)
        self._validate_selector_text()

    def close_inspector(self) -> None:
        super().close_inspector()
        self._rows_entry = None
        self._cols_entry = None
        self._reshape_entry = None

    def _on_apply(self) -> None:
        if self._request_downstream:
            self._request_downstream(self.node_id)

    @staticmethod
    def _coerce_1d_int_indices(value, name: str) -> np.ndarray:
        arr = np.asarray(value)
        if arr.size == 0:
            raise ValueError(f"{name} is empty")
        flat = arr.reshape(-1)
        ints = flat.astype(np.int64)
        if np.any(np.abs(flat.astype(np.float64) - ints.astype(np.float64)) > 1e-9):
            raise ValueError(f"{name} must contain integers")
        return ints

    @staticmethod
    def _normalise_indices(indices: np.ndarray, size: int, name: str) -> np.ndarray:
        normalised = np.where(indices < 0, indices + size, indices)
        if np.any(normalised < 0) or np.any(normalised >= size):
            raise ValueError(f"{name} out of range for axis length {size}")
        return normalised.astype(np.int64)

    @classmethod
    def _parse_text_indices(cls, text: str, name: str, size: int) -> np.ndarray:
        raw = text.strip()
        if not raw:
            raise ValueError(f"{name} text is empty")
        tokens = raw.replace(",", " ").replace(";", " ").split()
        if not tokens:
            raise ValueError(f"{name} text is empty")
        values: list[int] = []
        try:
            for token in tokens:
                if ":" not in token:
                    values.append(int(token))
                    continue
                parts = token.split(":")
                if len(parts) != 2 or not parts[0] or not parts[1]:
                    raise ValueError
                start, end = (int(part) for part in parts)
                start = start + size if start < 0 else start
                end = end + size if end < 0 else end
                if not (0 <= start < size and 0 <= end < size):
                    raise ValueError
                step = 1 if end >= start else -1
                values.extend(range(start, end + step, step))
        except (TypeError, ValueError):
            raise ValueError(f"{name} text parse error") from None
        if not values:
            raise ValueError(f"{name} is empty")
        return cls._normalise_indices(np.array(values, dtype=np.int64), size, name)

    @staticmethod
    def _parse_reshape(text: str) -> tuple[int, ...] | None:
        raw = text.strip()
        if not raw:
            return None
        tokens = raw.replace(",", " ").replace(";", " ").split()
        if not tokens:
            return None
        try:
            shape = tuple(int(token) for token in tokens)
        except ValueError:
            raise ValueError("reshape text parse error") from None
        if shape.count(-1) > 1 or any(size < -1 for size in shape):
            raise ValueError("reshape dimensions are invalid")
        return shape

    def _set_entry_validity(self, entry: tk.Entry | None, valid: bool) -> None:
        if entry is not None and entry.winfo_exists():
            entry.configure(fg="#000000" if valid else "#cc0000")

    def _validate_selector_text(self) -> None:
        mat = self._current_data
        if mat is None or mat.ndim != 2:
            self._set_entry_validity(self._rows_entry, True)
            self._set_entry_validity(self._cols_entry, True)
            self._set_entry_validity(self._reshape_entry, True)
            return
        try:
            rows = self._parse_text_indices(self._rows_var.get(), "rows", mat.shape[0])
            rows_valid = True
        except ValueError:
            rows = None
            rows_valid = False
        try:
            cols = self._parse_text_indices(self._cols_var.get(), "cols", mat.shape[1])
            cols_valid = True
        except ValueError:
            cols = None
            cols_valid = False
        self._set_entry_validity(self._rows_entry, rows_valid)
        self._set_entry_validity(self._cols_entry, cols_valid)

        reshape_valid = True
        try:
            reshape = self._parse_reshape(self._reshape_var.get())
            if reshape is not None and rows is not None and cols is not None:
                mat[np.ix_(rows, cols)].reshape(reshape)
        except ValueError:
            reshape_valid = False
        self._set_entry_validity(self._reshape_entry, reshape_valid)

    def _on_selector_text_changed(self, *_args) -> None:
        self._validate_selector_text()

    def compute(self, inputs: dict) -> dict:
        data = inputs.get("data")
        if data is None:
            self._status_var.set("missing data")
            self.set_status("missing", "#cc0000")
            return {}

        try:
            mat = np.asarray(data)
            if mat.ndim != 2:
                raise ValueError("data must be a 2D array")
            self._current_data = mat
            self._validate_selector_text()

            rows_raw = inputs.get("rows")
            cols_raw = inputs.get("cols")
            if rows_raw is None:
                rows = self._parse_text_indices(self._rows_var.get(), "rows", mat.shape[0])
            else:
                rows = self._normalise_indices(
                    self._coerce_1d_int_indices(rows_raw, "rows"), mat.shape[0], "rows")
            if cols_raw is None:
                cols = self._parse_text_indices(self._cols_var.get(), "cols", mat.shape[1])
            else:
                cols = self._normalise_indices(
                    self._coerce_1d_int_indices(cols_raw, "cols"), mat.shape[1], "cols")

            out = mat[np.ix_(rows, cols)]
            reshape = self._parse_reshape(self._reshape_var.get())
            if reshape is not None:
                out = out.reshape(reshape)

            self._status_var.set(f"ok: {out.shape}")
            self.set_status("ok", "#339966")
            return {"array": out}
        except Exception as e:
            self._status_var.set(f"error: {e}")
            self.set_status("error", "#cc0000")
            return {}

    def get_params(self) -> dict:
        return {
            "rows_text": self._rows_var.get(),
            "cols_text": self._cols_var.get(),
            "reshape_text": self._reshape_var.get(),
        }

    def set_params(self, params: dict) -> None:
        self._init_state()
        self._rows_var.set(str(params.get("rows_text", "0,1")))
        self._cols_var.set(str(params.get("cols_text", "0,1")))
        self._reshape_var.set(str(params.get("reshape_text", "")))
        self._validate_selector_text()
