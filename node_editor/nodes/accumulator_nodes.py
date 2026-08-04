# node_editor/nodes/accumulator_nodes.py

import tkinter as tk
from tkinter import ttk, filedialog
import threading
import csv
from pathlib import Path

import numpy as np

from node_editor.base_node import BaseNode
from node_editor.pin_types import PinSchema, PinDef, PinType
from node_editor.execution import ExecutionMode
from node_editor.project_context import get_project_directory


class ArrayAccumulatorNode(BaseNode):
    """
    Accumulates incoming arrays row by row into a growing 2D table.

    Each compute() call receives one array on the 'data' input pin.
    The array is flattened to 1D and appended as one row.

    Example: optical flow outputs nextPts of shape (N, 1, 2) or (N, 2).
    This node flattens it to (N*2,) and appends it as one row.
    After 1000 calls the table is (1000, N*2).

    Features:
      - Column names specified as comma-separated text
      - CSV file output with column name header row
      - Configurable save interval (save every N rows)
      - Inspector shows live table preview
      - Reset clears the table
      - Sends row_done TRIGGER output after each row (for BatchRunnerNode)

    Input pins:
      data     ARRAY    one array per step, any shape (will be flattened)
      index    SCALAR   optional row label (e.g. loop index)
      reset    TRIGGER  clears the accumulated table

    Output pins:
      table    ARRAY    accumulated 2D table so far (N_rows x N_cols)
      count    SCALAR   number of rows collected so far
      row_done TRIGGER  fires after each row is appended
    """

    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE      = "array_accumulator"
    DISPLAY_NAME   = "Array Accumulator"
    CATEGORY       = "process"
    NODE_WIDTH     = 200
    NODE_HEIGHT    = 120

    MAX_PREVIEW_ROWS = 500
    MAX_PREVIEW_COLS = 50

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[
                PinDef("data",  PinType.ARRAY,   "data",
                       optional=False),
                PinDef("index", PinType.SCALAR,  "index",
                       optional=True),
                PinDef("reset", PinType.TRIGGER, "reset",
                       optional=True),
            ],
            outputs=[
                PinDef("table",    PinType.ARRAY,   "table"),
                PinDef("count",    PinType.SCALAR,  "count"),
                PinDef("row_done", PinType.TRIGGER, "done"),
            ]
        )

    # ── init state ────────────────────────────────────────────────

    def _init_state(self) -> None:
        if hasattr(self, "_col_names_var"):
            return
        self._col_names_var    = tk.StringVar(value="")
        self._save_path_var    = tk.StringVar(value="(no file)")
        self._save_interval_var = tk.IntVar(value=1)
        self._status_var       = tk.StringVar(value="0 rows")

        # runtime accumulation state
        self._rows:         list[np.ndarray] = []
        self._indices:      list[float]      = []
        self._table:        np.ndarray | None = None
        self._fire_counter: int  = 0
        self._last_reset          = None
        self._save_path:    str  = ""
        self._save_lock          = threading.Lock()
        self._n_cols:       int  = 0   # inferred from first row

        # inspector widgets (exist only while open)
        self._col_entry:      tk.Entry    | None = None
        self._interval_sb:    tk.Spinbox  | None = None
        self._tree:           ttk.Treeview | None = None
        self._tree_scroll_v:  ttk.Scrollbar | None = None
        self._tree_scroll_h:  ttk.Scrollbar | None = None
        self._inspector_status_lbl: tk.Label | None = None

    # ── build_body ────────────────────────────────────────────────

    def build_body(self) -> None:
        self._init_state()
        x, y, w, h = self.x, self.y, self.width, self.height

        self._body_rect = self.canvas.create_rectangle(
            x, y, x+w, y+h,
            fill="#1e2a1e", outline="#55aa55", width=2,
            tags=(self.node_id, "node_body"))
        self._title_item = self.canvas.create_text(
            x+w/2, y+13,
            text=self.DISPLAY_NAME,
            font=("Arial", 9, "bold"), fill="#88ff88",
            tags=(self.node_id,))

        # row count (large)
        self._count_item = self.canvas.create_text(
            x+w/2, y+h//2+6,
            text="0 rows",
            font=("Arial", 14, "bold"), fill="#88ff88",
            tags=(self.node_id,))

        # status line
        status_lbl = tk.Label(
            self.canvas, textvariable=self._status_var,
            font=("Arial", 7), bg="#1e2a1e", fg="#aaccaa")
        self.canvas.create_window(
            x+w/2, y+h-10, window=status_lbl,
            tags=(self.node_id,))

        self._canvas_items += [self._body_rect, self._title_item,
                               self._count_item]

    # ── build_inspector ───────────────────────────────────────────

    def build_inspector(self, parent: tk.Frame) -> None:
        self._init_state()

        # ── column names ─────────────────────────────────────────
        col_frame = tk.LabelFrame(
            parent, text="Column names (comma-separated)",
            font=("Arial", 9), padx=6, pady=4)
        col_frame.pack(fill="x", pady=(0, 6))

        self._col_entry = tk.Entry(
            col_frame,
            textvariable=self._col_names_var,
            font=("Courier", 9))
        self._col_entry.pack(fill="x")

        tk.Label(
            col_frame,
            text='e.g.  "xi_P1, yi_P1, xi_P2, yi_P2"',
            font=("Arial", 8), fg="#666666",
            anchor="w").pack(fill="x")

        # ── save settings ─────────────────────────────────────────
        save_frame = tk.LabelFrame(
            parent, text="CSV output",
            font=("Arial", 9), padx=6, pady=4)
        save_frame.pack(fill="x", pady=(0, 6))

        path_row = tk.Frame(save_frame)
        path_row.pack(fill="x", pady=(0, 4))

        tk.Label(
            path_row, text="File:",
            font=("Arial", 9)).pack(side="left")
        tk.Label(
            path_row, textvariable=self._save_path_var,
            font=("Arial", 8), fg="#224488",
            anchor="w", justify="left",
            wraplength=320).pack(
            side="left", padx=(6, 0), fill="x", expand=True)
        tk.Button(
            path_row, text="Browse…",
            font=("Arial", 8),
            command=self._on_browse_save_path).pack(
            side="right")

        interval_row = tk.Frame(save_frame)
        interval_row.pack(fill="x")
        tk.Label(
            interval_row,
            text="Save every N rows:",
            font=("Arial", 9)).pack(side="left")
        self._interval_sb = tk.Spinbox(
            interval_row,
            from_=1, to=100000,
            textvariable=self._save_interval_var,
            width=7, font=("Arial", 9))
        self._interval_sb.pack(side="left", padx=(6, 0))
        tk.Label(
            interval_row,
            text="(1 = save after every row)",
            font=("Arial", 8), fg="#666666").pack(
            side="left", padx=(8, 0))

        # ── action buttons ────────────────────────────────────────
        btn_row = tk.Frame(parent)
        btn_row.pack(fill="x", pady=(0, 6))

        tk.Button(
            btn_row, text="Clear table",
            font=("Arial", 9),
            command=self._clear).pack(side="left", padx=(0, 6))
        tk.Button(
            btn_row, text="Save now",
            font=("Arial", 9),
            command=self._save_now).pack(side="left", padx=(0, 6))
        tk.Button(
            btn_row, text="Copy to clipboard",
            font=("Arial", 9),
            command=self._copy_table_to_clipboard).pack(side="left")

        # ── status ────────────────────────────────────────────────
        self._inspector_status_lbl = tk.Label(
            parent, textvariable=self._status_var,
            font=("Arial", 9), anchor="w",
            justify="left", fg="#226622")
        self._inspector_status_lbl.pack(fill="x", pady=(0, 6))

        # ── table preview ─────────────────────────────────────────
        preview_frame = tk.LabelFrame(
            parent, text="Table preview",
            font=("Arial", 9), padx=4, pady=4)
        preview_frame.pack(fill="both", expand=True)

        self._tree = ttk.Treeview(
            preview_frame, show="headings",
            selectmode="browse")
        self._tree_scroll_v = ttk.Scrollbar(
            preview_frame, orient="vertical",
            command=self._tree.yview)
        self._tree_scroll_h = ttk.Scrollbar(
            preview_frame, orient="horizontal",
            command=self._tree.xview)
        self._tree.configure(
            yscrollcommand=self._tree_scroll_v.set,
            xscrollcommand=self._tree_scroll_h.set)
        self._tree_scroll_h.pack(side="bottom", fill="x")
        self._tree_scroll_v.pack(side="right",  fill="y")
        self._tree.pack(fill="both", expand=True)

        # populate with current data if any
        self._refresh_inspector_table()

    def close_inspector(self) -> None:
        super().close_inspector()
        self._col_entry       = None
        self._interval_sb     = None
        self._tree            = None
        self._tree_scroll_v   = None
        self._tree_scroll_h   = None
        self._inspector_status_lbl = None

    # ── inspector helpers ─────────────────────────────────────────

    def _on_browse_save_path(self) -> None:
        base = get_project_directory()
        initial_dir = str(base) if base else "."
        path = filedialog.asksaveasfilename(
            title="Save CSV output file",
            initialdir=initial_dir,
            filetypes=[
                ("CSV files", "*.csv"),
                ("All files", "*.*"),
            ],
            defaultextension=".csv")
        if not path:
            return
        self._save_path = path
        self._save_path_var.set(Path(path).name)

    def _col_names_list(self) -> list[str]:
        """Parse comma-separated column name string into a list."""
        raw = self._col_names_var.get().strip()
        if not raw:
            return []
        return [c.strip() for c in raw.split(",") if c.strip()]

    def _refresh_inspector_table(self) -> None:
        if self._tree is None or not self._tree.winfo_exists():
            return
        if self._table is None or len(self._rows) == 0:
            self._tree.configure(columns=[])
            for item in self._tree.get_children():
                self._tree.delete(item)
            return

        n_rows, n_cols = self._table.shape
        col_names      = self._col_names_list()

        # build column headers
        col_ids = ["#idx"] + [str(c) for c in range(n_cols)]
        self._tree.configure(columns=col_ids)
        self._tree.heading("#idx", text="row")
        self._tree.column("#idx", width=52, anchor="e")

        col_w = max(55, min(90, 700 // max(n_cols, 1)))
        for c in range(n_cols):
            label = (col_names[c]
                     if c < len(col_names)
                     else str(c))
            self._tree.heading(str(c), text=label)
            self._tree.column(str(c), width=col_w, anchor="e")

        # repopulate rows (show last MAX_PREVIEW_ROWS)
        for item in self._tree.get_children():
            self._tree.delete(item)

        start = max(0, n_rows - self.MAX_PREVIEW_ROWS)
        is_float = np.issubdtype(self._table.dtype, np.floating)

        for r in range(start, n_rows):
            row = self._table[r]
            vals = [f"{(self._indices[r] + 1):.0f}"] + [
                f"{v:.6g}" if is_float else str(v)
                for v in row[:self.MAX_PREVIEW_COLS]]
            tag = "even" if (r - start) % 2 == 0 else "odd"
            self._tree.insert("", "end", values=vals,
                              tags=(tag,))

        self._tree.tag_configure("even", background="#eef8ee")
        self._tree.tag_configure("odd",  background="#ffffff")

        # scroll to bottom to show latest row
        children = self._tree.get_children()
        if children:
            self._tree.see(children[-1])

    # ── clear / save ──────────────────────────────────────────────

    def _clear(self) -> None:
        self._rows.clear()
        self._indices.clear()
        self._table  = None
        self._n_cols = 0
        self._update_status(0)
        self._refresh_inspector_table()

    def _save_now(self) -> None:
        """Force an immediate CSV save regardless of interval."""
        if not self._save_path:
            self._status_var.set(
                "no save path set — use Browse…")
            return
        self._write_csv()

    def _copy_table_to_clipboard(self) -> None:
        """Copy current table as CSV text for pasting into Array Input."""
        if self._table is None or self._table.size == 0:
            self._status_var.set("no data to copy")
            return

        table = np.atleast_2d(self._table)
        dtype = table.dtype

        if np.issubdtype(dtype, np.integer):
            def _fmt(v) -> str:
                return "%d" % int(v)
        elif dtype == np.float32:
            def _fmt(v) -> str:
                return "%.9g" % float(v)
        elif dtype == np.float64:
            def _fmt(v) -> str:
                return "%.17g" % float(v)
        else:
            def _fmt(v) -> str:
                if isinstance(v, (np.integer, int)):
                    return "%d" % int(v)
                if isinstance(v, np.float32):
                    return "%.9g" % float(v)
                if isinstance(v, (np.floating, float)):
                    return "%.17g" % float(v)
                return str(v)

        lines: list[str] = []
        for row in table:
            lines.append(",".join(_fmt(v) for v in row))
        text = "\n".join(lines)

        top = self.canvas.winfo_toplevel()
        top.clipboard_clear()
        top.clipboard_append(text)
        top.update_idletasks()

        n_rows, n_cols = table.shape
        self._status_var.set(
            f"copied {n_rows}x{n_cols} CSV to clipboard")

    def _write_csv(self) -> None:
        """
        Write the current table to CSV in a background thread
        so the main thread is not blocked by file I/O.
        """
        if self._table is None or not self._save_path:
            return

        table_snapshot = self._table.copy()
        indices_snapshot = list(self._indices)
        col_names = self._col_names_list()
        path = self._save_path

        def _do_write():
            try:
                with open(path, "w", newline="",
                          encoding="utf-8") as f:
                    writer = csv.writer(f)
                    # header row
                    n_cols = table_snapshot.shape[1]
                    if col_names:
                        # pad or trim to match actual column count
                        header = col_names[:n_cols]
                        while len(header) < n_cols:
                            header.append(f"col_{len(header)}")
                    else:
                        header = [f"col_{c}"
                                  for c in range(n_cols)]
                    writer.writerow(["index"] + header)
                    # data rows
                    is_float = np.issubdtype(
                        table_snapshot.dtype, np.floating)
                    for r, row in enumerate(table_snapshot):
                        idx_str = f"{indices_snapshot[r]:.0f}"
                        if is_float:
                            vals = [f"{v:.8g}" for v in row]
                        else:
                            vals = [str(v) for v in row]
                        writer.writerow([idx_str] + vals)
            except Exception as e:
                # schedule UI update back on main thread
                self.canvas.after(
                    0,
                    lambda err=e: self._status_var.set(
                        f"save error: {err}"))
                return
            n = table_snapshot.shape[0]
            self.canvas.after(
                0,
                lambda: self._status_var.set(
                    f"{n} rows  saved → "
                    f"{Path(path).name}"))

        t = threading.Thread(target=_do_write, daemon=True)
        t.start()

    def _update_status(self, n_rows: int) -> None:
        shape_str = (
            f"{n_rows} × {self._n_cols}"
            if self._n_cols > 0 else f"{n_rows} rows")
        self._status_var.set(shape_str)
        self.canvas.itemconfig(
            self._count_item, text=f"{n_rows} rows")

    # ── compute ───────────────────────────────────────────────────

    def compute(self, inputs: dict) -> dict:
        # ── handle reset trigger ──────────────────────────────────
        reset = inputs.get("reset")
        if reset is not None and reset != self._last_reset:
            self._last_reset = reset
            self._clear()

        # ── incoming data ─────────────────────────────────────────
        data = inputs.get("data")
        if data is None or not isinstance(data, np.ndarray):
            return self._current_outputs()

        # flatten to 1D row
        row = np.asarray(data).ravel()
        if row.size == 0:
            return self._current_outputs()

        # enforce consistent column count
        if self._n_cols == 0:
            self._n_cols = row.size
        elif row.size != self._n_cols:
            self._status_var.set(
                f"shape mismatch: expected {self._n_cols} "
                f"cols, got {row.size} — row skipped")
            return self._current_outputs()

        # append row
        self._rows.append(row.copy())
        idx = inputs.get("index")
        self._indices.append(
            float(idx) if idx is not None
            else float(len(self._rows) - 1))

        # rebuild table
        self._table = np.vstack(self._rows)
        n_rows      = len(self._rows)
        self._update_status(n_rows)

        # refresh inspector table view if open
        if self.is_inspector_open():
            self._refresh_inspector_table()

        # periodic CSV save
        if self._save_path:
            interval = max(1, self._save_interval_var.get())
            if n_rows % interval == 0:
                self._write_csv()

        # fire row_done trigger
        self._fire_counter += 1
        return self._current_outputs()

    def _current_outputs(self) -> dict:
        if self._table is None:
            return {"count": 0.0,
                    "row_done": self._fire_counter}
        return {
            "table":    self._table,
            "count":    float(len(self._rows)),
            "row_done": self._fire_counter,
        }

    # ── array data for XLSX project save ─────────────────────────

    def get_array_data(self) -> dict:
        if self._table is None:
            return {}
        return {"accumulated_table": self._table}

    # ── serialization ─────────────────────────────────────────────

    def get_params(self) -> dict:
        return {
            "col_names":     self._col_names_var.get(),
            "save_path":     self._save_path,
            "save_interval": self._save_interval_var.get(),
        }

    def set_params(self, params: dict) -> None:
        self._init_state()
        self._col_names_var.set(
            str(params.get("col_names", "")))
        path = str(params.get("save_path", ""))
        self._save_path = path
        self._save_path_var.set(
            Path(path).name if path else "(no file)")
        try:
            self._save_interval_var.set(
                int(params.get("save_interval", 1)))
        except Exception:
            self._save_interval_var.set(1)

    def on_destroy(self) -> None:
        super().on_destroy()