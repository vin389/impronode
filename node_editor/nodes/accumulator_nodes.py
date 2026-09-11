# node_editor/nodes/array_accumulator_node.py

import tkinter as tk
from tkinter import ttk, filedialog
import threading
import csv
from pathlib import Path

import numpy as np

from node_editor.base_node import BaseNode
from node_editor.pin_types import PinSchema, PinDef, PinType
from node_editor.execution import ExecutionMode
from node_editor.project_context import get_project_directory, get_project_file_path


_HELP_TEXT = (
    "Array Accumulator Node\n\n"
    "Purpose:\n"
    "- Accumulate incoming 1D/2D arrays into a growing 2D table, one row\n"
    "  per compute() call by default. Optionally, an explicit row number\n"
    "  can be supplied to UPDATE an existing row or to grow the table to\n"
    "  fit a row that is further ahead than anything received so far.\n\n"
    "Input Pins:\n"
    "- data [ARRAY]: one array per call (any shape, will be flattened to\n"
    "  a single row). Required.\n"
    "- row [SCALAR, optional]: 1-based row number to write this data to.\n"
    "    * Not connected -> APPEND mode: data is written to the next new\n"
    "      row (current row count + 1), exactly like before.\n"
    "    * row <= current row count -> UPDATE mode: the existing row at\n"
    "      that position is overwritten with the new data.\n"
    "    * row > current row count -> GROW mode: the table is extended so\n"
    "      row `row` exists. Any newly created rows strictly between the\n"
    "      old last row and `row` that receive no data are left as NaN\n"
    "      (\"gap\" rows) until something writes to them later.\n"
    "- reset [TRIGGER, optional]: clears the accumulated table completely.\n\n"
    "Output Pins:\n"
    "- table [ARRAY]: the accumulated table so far, shape (n_rows, n_cols).\n"
    "  Gap rows (never written) appear as all-NaN.\n"
    "- count [SCALAR]: current table height (n_rows), including any gap\n"
    "  rows created by a GROW-mode write.\n"
    "- row_done [TRIGGER]: fires once after every successful write\n"
    "  (append, update, or gap-fill). Useful to drive a LoopNode's\n"
    "  advance pin or any other trigger consumer.\n\n"
    "Inspector Settings:\n"
    "- Column names: comma-separated list, e.g. \"xi_P1, yi_P1, xi_P2\".\n"
    "  Used as the header row for CSV export and as column headers in the\n"
    "  table preview below.\n"
    "- CSV output file: path to save the table to. Browse... picks a file.\n"
    "- Save every N writes: write the CSV automatically every N successful\n"
    "  writes (append, update, or gap-fill all count). 1 = every write.\n"
    "  Larger values reduce file-I/O overhead for many points / many steps.\n"
    "- Clear table: erase all accumulated data immediately.\n"
    "- Save now: force an immediate CSV write regardless of the interval.\n\n"
    "- Load CSV: replace the current table with rows read from a CSV file.\n\n"
    "Project Persistence:\n"
    "- Saving a project automatically writes this accumulator's current\n"
    "  data to a uniquely named CSV beside the project XLSX file. The XLSX\n"
    "  stores only the relative CSV filename; loading the project restores\n"
    "  the data automatically. Keep the XLSX and its generated CSV files\n"
    "  together when moving or copying a project.\n\n"
    "Table Preview:\n"
    "- Rows that have never received data (pure gap rows created by a\n"
    "  GROW-mode write that jumped ahead) are highlighted so they are easy\n"
    "  to tell apart from rows that were actually written (even if that\n"
    "  written data happens to contain NaN values itself).\n\n"
    "Notes:\n"
    "- All rows must have the same number of columns as the first row\n"
    "  ever received. A row with a different length is rejected with a\n"
    "  status message; existing data is left unchanged.\n"
    "- `row` must be a positive integer (1-based). A `row` value beyond\n"
    "  an internal safety cap is rejected to avoid accidentally allocating\n"
    "  an enormous table from a bad input value.\n"
    "- CSV writes run on a background thread so the main UI is never\n"
    "  blocked, even for large tables.\n\n"
    "Keyboard Shortcut:\n"
    "Ctrl-H / Ctrl-h - show this help window.\n"
)


class ArrayAccumulatorNode(BaseNode):
    """
    Accumulates arrays row by row into a growing 2D table.

    Each compute() call receives one array on the 'data' input pin.
    The array is flattened to 1D and written into the table at a row
    position controlled by the optional 'row' input pin:

      - row not connected  -> append at the end (row = current height + 1)
      - row <= current height -> update/overwrite that existing row
      - row >  current height -> grow the table; any skipped rows in
                                  between are left as NaN "gap" rows

    Example: optical flow outputs nextPts of shape (N, 1, 2) or (N, 2).
    This node flattens it to (N*2,) and writes it as one row. Connecting
    the tracker's own frame index to 'row' lets you go back and correct
    a single frame's row later without disturbing any other row.

    See _HELP_TEXT (Ctrl-H) for the full pin/inspector reference.
    """

    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE      = "array_accumulator"
    DISPLAY_NAME   = "Array Accumulator"
    CATEGORY       = "process"
    NODE_WIDTH     = 200
    NODE_HEIGHT    = 120

    HELP_TEXT = _HELP_TEXT

    MAX_PREVIEW_ROWS = 500
    MAX_PREVIEW_COLS = 50

    # Safety cap on the 'row' pin value, to avoid accidentally allocating
    # an enormous table from a bad/garbage input value (e.g. a stray NaN
    # or a wildly out-of-range scalar arriving on the row pin).
    _MAX_ROWS_HARD_CAP = 2_000_000

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[
                PinDef("data",  PinType.ARRAY,   "data",
                       optional=False),
                PinDef("row",   PinType.SCALAR,  "row",
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

    def get_help_text(self) -> str:
        return _HELP_TEXT

    # ── init state ────────────────────────────────────────────────

    def _init_state(self) -> None:
        if hasattr(self, "_col_names_var"):
            return
        self._col_names_var     = tk.StringVar(value="")
        self._save_path_var     = tk.StringVar(value="(no file)")
        self._save_interval_var = tk.IntVar(value=1)
        self._status_var        = tk.StringVar(value="0 rows")

        # runtime accumulation state
        # self._rows maps a 1-based row number to that row's flattened
        # data. Using a dict (rather than a plain list) lets a GROW-mode
        # write jump ahead of the current table height without needing to
        # pre-fill every intermediate row up front -- those intermediate
        # rows simply have no entry in the dict until something writes
        # to them, and _build_table() fills any missing entry with NaN.
        self._rows: dict[int, np.ndarray] = {}
        self._max_row:      int  = 0    # current table height (n_rows)
        self._n_cols:       int  = 0    # inferred from the first row received
        self._fire_counter: int  = 0    # row_done trigger value
        self._write_count:  int  = 0    # successful writes, for save-interval
        self._skip_next_data_write: bool = False

        self._save_path:    str  = ""
        self._save_lock            = threading.Lock()

        # --- reset-pin edge-detection state -----------------------------
        # Same edge-detection design used by OpticalFlowNode/TemplateMatchNode:
        # fires on a classic rising edge (falsy -> truthy) for boolean
        # pulse-style trigger sources, AND on any value change while already
        # truthy, for ever-incrementing counter-style trigger sources.
        self._last_reset_value = None
        self._reset_was_truthy = False

        # inspector widgets (exist only while inspector is open)
        self._col_entry:      tk.Entry     | None = None
        self._interval_sb:    tk.Spinbox   | None = None
        self._tree:           ttk.Treeview | None = None
        self._tree_scroll_v:  ttk.Scrollbar | None = None
        self._tree_scroll_h:  ttk.Scrollbar | None = None
        self._help_popup:     tk.Toplevel  | None = None

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
            font=("Arial", 7), bg="#1e2a1e", fg="#aaccaa",
            wraplength=w-10, justify="center")
        self.canvas.create_window(
            x+w/2, y+h-10, window=status_lbl,
            tags=(self.node_id,))

        self._canvas_items += [self._body_rect, self._title_item,
                               self._count_item]

    # ── build_inspector ───────────────────────────────────────────

    def build_inspector(self, parent: tk.Frame) -> None:
        self._init_state()

        # Bind Ctrl-H inside the inspector window, same pattern used by
        # the other nodes in this project (UndistortNode, MeshgridNode,
        # PointPredictorNode, CameraCalibNode).
        win = self._inspector_win
        if win is not None:
            for seq in ("<Control-h>", "<Control-H>"):
                win.bind(seq, lambda e: self._open_help())

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
            path_row, text="Browse...",
            font=("Arial", 8),
            command=self._on_browse_save_path).pack(
            side="right")

        interval_row = tk.Frame(save_frame)
        interval_row.pack(fill="x")
        tk.Label(
            interval_row,
            text="Save every N writes:",
            font=("Arial", 9)).pack(side="left")
        self._interval_sb = tk.Spinbox(
            interval_row,
            from_=1, to=100000,
            textvariable=self._save_interval_var,
            width=7, font=("Arial", 9))
        self._interval_sb.pack(side="left", padx=(6, 0))
        tk.Label(
            interval_row,
            text="(1 = save after every write)",
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
            btn_row, text="Load CSV...",
            font=("Arial", 9),
            command=self._load_csv).pack(side="left", padx=(0, 6))
        tk.Button(
            btn_row, text="Help (Ctrl-H)",
            font=("Arial", 9),
            command=self._open_help).pack(side="right")

        # ── status ────────────────────────────────────────────────
        tk.Label(
            parent, textvariable=self._status_var,
            font=("Arial", 9), anchor="w",
            justify="left", fg="#226622").pack(fill="x", pady=(0, 6))

        # ── legend ────────────────────────────────────────────────
        legend_row = tk.Frame(parent)
        legend_row.pack(fill="x", pady=(0, 4))
        tk.Label(
            legend_row, text="  ", bg="#ffdddd", width=2).pack(side="left")
        tk.Label(
            legend_row, text="gap row (never written)",
            font=("Arial", 8), fg="#666666").pack(side="left", padx=(4, 0))

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

    # ── help popup ────────────────────────────────────────────────

    def _open_help(self) -> None:
        if (self._help_popup is not None
                and self._help_popup.winfo_exists()):
            self._help_popup.lift()
            return

        popup = tk.Toplevel()
        popup.title("Array Accumulator - Help")
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
            body, font=("Courier", 9),
            bg="#f8f8f8", fg="#222222",
            wrap=tk.WORD, relief=tk.FLAT)
        vsb = ttk.Scrollbar(body, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", _HELP_TEXT)
        txt.configure(state="disabled")

        tk.Button(
            body, text="Close", font=("Arial", 9),
            command=popup.destroy).pack(anchor="e", pady=(8, 0))

        popup.bind("<Escape>", lambda _e: popup.destroy())
        popup.protocol("WM_DELETE_WINDOW", popup.destroy)
        self._help_popup = popup

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

    def _build_table(self) -> np.ndarray | None:
        """
        Build a fresh (n_rows, n_cols) table from self._rows.

        Rows with no entry in self._rows (gaps created by a GROW-mode
        write that jumped ahead of the previous table height) are filled
        with NaN. A brand-new array is allocated on every call, so the
        returned table never aliases anything stored in self._rows --
        callers (including the CSV writer thread) can use it freely
        without needing an extra .copy().
        """
        if self._max_row == 0 or self._n_cols == 0:
            return None
        table = np.full((self._max_row, self._n_cols), np.nan, dtype=np.float64)
        for row_number, row_data in self._rows.items():
            table[row_number - 1] = row_data
        return table

    def _refresh_inspector_table(self) -> None:
        if self._tree is None or not self._tree.winfo_exists():
            return

        table = self._build_table()
        if table is None:
            self._tree.configure(columns=[])
            for item in self._tree.get_children():
                self._tree.delete(item)
            return

        n_rows, n_cols = table.shape
        col_names      = self._col_names_list()

        # build column headers
        col_ids = ["#row"] + [str(c) for c in range(n_cols)]
        self._tree.configure(columns=col_ids)
        self._tree.heading("#row", text="row")
        self._tree.column("#row", width=52, anchor="e")

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
        is_float = np.issubdtype(table.dtype, np.floating)

        for r in range(start, n_rows):
            row_number = r + 1   # 1-based, matches the 'row' pin convention
            row_data   = table[r]
            vals = [str(row_number)] + [
                f"{v:.6g}" if is_float else str(v)
                for v in row_data[:self.MAX_PREVIEW_COLS]]

            was_written = row_number in self._rows
            if not was_written:
                tag = "gap"
            else:
                tag = "even" if (r - start) % 2 == 0 else "odd"

            self._tree.insert("", "end", values=vals, tags=(tag,))

        self._tree.tag_configure("even", background="#eef8ee")
        self._tree.tag_configure("odd",  background="#ffffff")
        self._tree.tag_configure("gap",  background="#ffdddd")

        # scroll to bottom to show latest activity
        children = self._tree.get_children()
        if children:
            self._tree.see(children[-1])

    # ── clear / save ──────────────────────────────────────────────

    def _clear(self) -> None:
        self._rows.clear()
        self._max_row  = 0
        self._n_cols    = 0
        self._write_count = 0
        self._update_display_counts()
        self._status_var.set("cleared")
        self._refresh_inspector_table()

    def _save_now(self) -> None:
        """Force an immediate CSV save regardless of the save interval."""
        if not self._save_path:
            self._status_var.set("no save path set - use Browse...")
            return
        self._write_csv()

    def _load_csv(self) -> None:
        base = get_project_directory()
        initial_dir = str(Path(self._save_path).parent) if self._save_path else str(base or ".")
        path = filedialog.askopenfilename(
            title="Load CSV data",
            initialdir=initial_dir,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            with open(path, "r", newline="", encoding="utf-8-sig") as file:
                rows, max_row, n_cols, col_names = self._parse_csv_rows(csv.reader(file))
        except Exception as exc:
            self._status_var.set(f"load error: {exc}")
            return

        self._restore_table(rows, max_row, n_cols)
        if col_names:
            self._col_names_var.set(", ".join(col_names))
        self._save_path = path
        self._save_path_var.set(Path(path).name)
        self._status_var.set(f"{max_row} rows loaded <- {Path(path).name}")
        self.push_output(self._current_outputs())

    @classmethod
    def _parse_csv_rows(cls, csv_rows) -> tuple[dict[int, np.ndarray], int, int, list[str]]:
        records = [list(row) for row in csv_rows if any(cell.strip() for cell in row)]
        if not records:
            raise ValueError("CSV file is empty")

        first = records[0]
        has_row_column = bool(first) and first[0].strip().lower() in {"row", "index"}
        has_header = has_row_column
        if not has_header:
            try:
                [float(cell) if cell.strip() else np.nan for cell in first]
            except ValueError:
                has_header = True

        if has_header:
            col_names = [cell.strip() for cell in first[1 if has_row_column else 0:]]
            data_records = records[1:]
        else:
            col_names = []
            data_records = records

        restored_rows: dict[int, np.ndarray] = {}
        n_cols = 0
        for sequential_row, record in enumerate(data_records, start=1):
            if has_row_column:
                if not record:
                    continue
                raw_row = float(record[0])
                row_number = int(raw_row)
                if not np.isfinite(raw_row) or raw_row != row_number:
                    raise ValueError(f"row number must be an integer: {record[0]!r}")
                values = record[1:]
            else:
                row_number = sequential_row
                values = record

            if row_number < 1 or row_number > cls._MAX_ROWS_HARD_CAP:
                raise ValueError(f"row number out of range: {row_number}")
            if row_number in restored_rows:
                raise ValueError(f"duplicate row number: {row_number}")
            if not values:
                raise ValueError(f"row {row_number} contains no data")

            try:
                row_data = np.asarray(
                    [float(value) if value.strip() else np.nan for value in values],
                    dtype=np.float64,
                )
            except ValueError as exc:
                raise ValueError(f"row {row_number} contains non-numeric data") from exc

            if n_cols == 0:
                n_cols = row_data.size
            elif row_data.size != n_cols:
                raise ValueError(
                    f"row {row_number} has {row_data.size} columns; expected {n_cols}")
            restored_rows[row_number] = row_data

        if not restored_rows:
            raise ValueError("CSV file contains no data rows")
        if col_names and len(col_names) != n_cols:
            raise ValueError(f"header has {len(col_names)} columns; expected {n_cols}")
        return restored_rows, max(restored_rows), n_cols, col_names

    def _restore_table(self, rows: dict[int, np.ndarray],
                       max_row: int, n_cols: int) -> None:
        self._rows = {row: np.asarray(data, dtype=np.float64).copy()
                      for row, data in rows.items()}
        self._max_row = max_row
        self._n_cols = n_cols
        self._write_count = 0
        self._update_display_counts()
        self._refresh_inspector_table()

    def _project_sidecar_filename(self) -> str | None:
        project_path = get_project_file_path()
        if project_path is None:
            return None
        safe_node_id = "".join(
            character if character.isalnum() or character in ("-", "_") else "_"
            for character in self.node_id
        )
        return f"{project_path.stem}.{safe_node_id}.array_accumulator.csv"

    def _write_project_sidecar(self, filename: str) -> None:
        project_dir = get_project_directory()
        if project_dir is None:
            raise ValueError("project path is required to save accumulator data")

        path = project_dir / filename
        if not self._rows:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            return

        temporary_path = path.with_name(path.name + ".tmp")
        col_names = self._col_names_list()
        header = col_names[:self._n_cols]
        while len(header) < self._n_cols:
            header.append(f"col_{len(header)}")

        try:
            with open(temporary_path, "w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["row"] + header)
                for row_number in sorted(self._rows):
                    values = [format(float(value), ".17g")
                              for value in self._rows[row_number]]
                    writer.writerow([row_number] + values)
            temporary_path.replace(path)
        except Exception:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
            raise

    def _load_project_sidecar(self, filename: str) -> bool:
        project_dir = get_project_directory()
        if project_dir is None:
            self._status_var.set("accumulator CSV cannot be resolved without a project path")
            return False
        if Path(filename).name != filename:
            self._status_var.set("project accumulator CSV filename is invalid")
            return False

        path = project_dir / filename
        try:
            with open(path, "r", newline="", encoding="utf-8-sig") as file:
                rows, max_row, n_cols, _col_names = self._parse_csv_rows(csv.reader(file))
        except Exception as exc:
            self._status_var.set(f"project accumulator CSV load error: {exc}")
            return False

        self._restore_table(rows, max_row, n_cols)
        return True

    def _write_csv(self) -> None:
        """
        Write the current table to CSV in a background thread so the
        main thread is never blocked by file I/O, even for large tables.
        """
        table = self._build_table()
        if table is None or not self._save_path:
            return

        # _build_table() always allocates a brand-new array (see its
        # docstring), so this snapshot cannot be mutated by any later
        # write to self._rows while the background thread is running.
        table_snapshot = table
        col_names = self._col_names_list()
        path = self._save_path

        def _do_write():
            try:
                with open(path, "w", newline="",
                          encoding="utf-8") as f:
                    writer = csv.writer(f)
                    n_cols = table_snapshot.shape[1]
                    if col_names:
                        header = col_names[:n_cols]
                        while len(header) < n_cols:
                            header.append(f"col_{len(header)}")
                    else:
                        header = [f"col_{c}" for c in range(n_cols)]
                    writer.writerow(["row"] + header)

                    is_float = np.issubdtype(
                        table_snapshot.dtype, np.floating)
                    for r, row in enumerate(table_snapshot):
                        row_number = r + 1
                        if is_float:
                            vals = [f"{v:.8g}" for v in row]
                        else:
                            vals = [str(v) for v in row]
                        writer.writerow([str(row_number)] + vals)
            except Exception as e:
                self.canvas.after(
                    0,
                    lambda err=e: self._status_var.set(
                        f"save error: {err}"))
                return
            n = table_snapshot.shape[0]
            self.canvas.after(
                0,
                lambda: self._status_var.set(
                    f"{n} rows saved -> {Path(path).name}"))

        t = threading.Thread(target=_do_write, daemon=True)
        t.start()

    def _update_display_counts(self) -> None:
        self.canvas.itemconfig(
            self._count_item, text=f"{self._max_row} rows")

    @staticmethod
    def _coerce_bool_like(value) -> bool:
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

    def on_upstream_changed(self) -> None:
        # Only the ephemeral reset-pin edge-detection state is cleared
        # here. The accumulated table itself is durable data the user
        # explicitly wants to keep, so it must survive a rewiring event
        # (adding/removing an unrelated link elsewhere in the graph).
        self._last_reset_value = None
        self._reset_was_truthy = False

    # ── compute ───────────────────────────────────────────────────

    def compute(self, inputs: dict) -> dict:
        self._init_state()

        # ── reset trigger (edge-detected) ───────────────────────────
        raw_reset = inputs.get("reset")
        reset_truthy_now = raw_reset is not None and self._coerce_bool_like(raw_reset)
        reset_value_changed = raw_reset is not None and raw_reset != self._last_reset_value
        if reset_truthy_now and (not self._reset_was_truthy or reset_value_changed):
            self._clear()
        self._reset_was_truthy = reset_truthy_now
        self._last_reset_value = raw_reset

        if self._skip_next_data_write:
            self._skip_next_data_write = False
            return self._current_outputs()

        # ── incoming data ────────────────────────────────────────────
        data = inputs.get("data")
        if data is None or not isinstance(data, np.ndarray):
            return self._current_outputs()

        # asarray(..., dtype=float64) copies only when the dtype actually
        # changes; ravel() may return a VIEW rather than a copy when the
        # array is already contiguous. Either way, row_flat could still
        # alias memory owned by the upstream node. Since this value is
        # about to be stored in self._rows and read back on every future
        # compute() call, it MUST be an independent copy -- see the
        # explicit .copy() a few lines below where it is actually stored.
        row_flat = np.asarray(data, dtype=np.float64).ravel()
        if row_flat.size == 0:
            return self._current_outputs()

        # ── determine target row (1-based) ──────────────────────────
        raw_row = inputs.get("row")
        if raw_row is not None:
            try:
                target_row = int(round(float(raw_row)))
            except (TypeError, ValueError):
                self._status_var.set(f"invalid row value: {raw_row!r}")
                return self._current_outputs()
        else:
            # No row pin connected: append mode, same behavior as before
            # this feature was added.
            target_row = self._max_row + 1

        if target_row < 1:
            self._status_var.set(
                f"row must be >= 1 (1-based), got {target_row} - write skipped")
            return self._current_outputs()

        if target_row > self._MAX_ROWS_HARD_CAP:
            self._status_var.set(
                f"row {target_row} exceeds safety cap "
                f"({self._MAX_ROWS_HARD_CAP}) - write skipped")
            return self._current_outputs()

        # ── enforce consistent column count ──────────────────────────
        if self._n_cols == 0:
            self._n_cols = row_flat.size
        elif row_flat.size != self._n_cols:
            self._status_var.set(
                f"shape mismatch: expected {self._n_cols} cols, "
                f"got {row_flat.size} - row skipped")
            return self._current_outputs()

        # ── classify the write for status/logging purposes ───────────
        previous_max_row = self._max_row
        was_already_written = target_row in self._rows

        if target_row > previous_max_row:
            if target_row == previous_max_row + 1:
                write_kind = "append"
            else:
                write_kind = "append_with_gap"
        elif was_already_written:
            write_kind = "update"
        else:
            write_kind = "fill_gap"

        # ── perform the write ─────────────────────────────────────────
        self._rows[target_row] = row_flat.copy()
        if target_row > self._max_row:
            self._max_row = target_row

        table = self._build_table()
        self._update_display_counts()

        if write_kind == "append":
            self._status_var.set(
                f"row {target_row} appended ({self._max_row} rows total)")
        elif write_kind == "append_with_gap":
            self._status_var.set(
                f"row {target_row} appended; rows "
                f"{previous_max_row + 1}..{target_row - 1} left as gaps (NaN)")
        elif write_kind == "update":
            self._status_var.set(
                f"row {target_row} updated (overwritten)")
        else:  # fill_gap
            self._status_var.set(
                f"row {target_row} filled (was previously empty)")

        if self.is_inspector_open():
            self._refresh_inspector_table()

        # ── periodic CSV save ────────────────────────────────────────
        self._write_count += 1
        if self._save_path:
            interval = max(1, self._save_interval_var.get())
            if self._write_count % interval == 0:
                self._write_csv()

        # ── fire row_done trigger ─────────────────────────────────────
        self._fire_counter += 1
        return {
            "table":    table,
            "count":    float(self._max_row),
            "row_done": self._fire_counter,
        }

    def _current_outputs(self) -> dict:
        table = self._build_table()
        if table is None:
            return {"count": 0.0, "row_done": self._fire_counter}
        return {
            "table":    table,
            "count":    float(self._max_row),
            "row_done": self._fire_counter,
        }

    # ── array data for XLSX project export (optional convenience hook) ──

    def get_array_data(self) -> dict:
        table = self._build_table()
        if table is None:
            return {}
        return {"accumulated_table": table}

    # ── serialization ─────────────────────────────────────────────

    def get_params(self) -> dict:
        params = {
            "col_names":     self._col_names_var.get(),
            "save_path":     self._save_path,
            "save_interval": self._save_interval_var.get(),
            "max_row":       self._max_row,
            "n_cols":        self._n_cols,
            "fire_counter":  self._fire_counter,
        }
        sidecar_filename = self._project_sidecar_filename()
        if sidecar_filename is not None:
            self._write_project_sidecar(sidecar_filename)
            if self._rows:
                params["accumulator_csv"] = sidecar_filename
        elif self._rows:
            params["rows"] = {str(row): data.tolist()
                              for row, data in self._rows.items()}
        return params

    def set_params(self, params: dict) -> None:
        self._init_state()
        self._col_names_var.set(str(params.get("col_names", "")))
        path = str(params.get("save_path", ""))
        self._save_path = path
        self._save_path_var.set(Path(path).name if path else "(no file)")
        try:
            self._save_interval_var.set(int(params.get("save_interval", 1)))
        except Exception:
            self._save_interval_var.set(1)

        sidecar_filename = str(params.get("accumulator_csv", "")).strip()
        if sidecar_filename:
            if self._load_project_sidecar(sidecar_filename):
                self._skip_next_data_write = True
                self._fire_counter = max(0, int(params.get("fire_counter", 0)))
                self._status_var.set(f"{self._max_row} rows restored from project CSV")
            return

        raw_rows = params.get("rows", {})
        if not isinstance(raw_rows, dict):
            return
        try:
            n_cols = int(params.get("n_cols", 0))
            max_row = int(params.get("max_row", 0))
            restored_rows: dict[int, np.ndarray] = {}
            for raw_row, raw_data in raw_rows.items():
                row_number = int(raw_row)
                row_data = np.asarray(raw_data, dtype=np.float64).ravel()
                if row_number < 1 or row_number > self._MAX_ROWS_HARD_CAP:
                    raise ValueError("stored row number is out of range")
                if n_cols == 0:
                    n_cols = row_data.size
                if row_data.size != n_cols:
                    raise ValueError("stored rows have inconsistent column counts")
                restored_rows[row_number] = row_data
            if restored_rows:
                max_row = max(max_row, max(restored_rows))
            if max_row < 0 or max_row > self._MAX_ROWS_HARD_CAP:
                raise ValueError("stored table height is out of range")
            if max_row and n_cols:
                self._restore_table(restored_rows, max_row, n_cols)
                self._skip_next_data_write = True
                self._status_var.set(f"{max_row} rows restored from project")
            self._fire_counter = max(0, int(params.get("fire_counter", 0)))
        except (TypeError, ValueError):
            self._status_var.set("stored accumulator data is invalid")

    def on_destroy(self) -> None:
        if self._help_popup is not None and self._help_popup.winfo_exists():
            self._help_popup.destroy()
        super().on_destroy()