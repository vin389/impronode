# node_editor/nodes/shared_array_node.py

import csv
import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path

import numpy as np

from node_editor.base_node import BaseNode
from node_editor.pin_types import PinSchema, PinDef, PinType
from node_editor.execution import ExecutionMode
from node_editor.project_context import get_project_directory
from node_editor import shared_array_registry as registry


_HELP_TEXT = (
    "Shared Array Node\n\n"
    "Purpose:\n"
    "- Holds a single named 2D numpy array (a table) as a project-global\n"
    "  variable. Any node in the project can read or write it by name,\n"
    "  either through this node's pins or, from Python node code, via\n"
    "  node_editor.shared_array_registry (get_array/set_array/set_row).\n\n"
    "Inspector Settings:\n"
    "- Global name: unique across the project. Only letters, digits and\n"
    "  underscore are accepted. The box turns red while the name is empty,\n"
    "  contains invalid characters, or collides with another Shared Array.\n"
    "- Header: comma-separated column names, e.g. \"step, x, y\". Used as\n"
    "  the CSV header and as the spreadsheet's column headings.\n"
    "- Header query: enter a column name below the Header box to check\n"
    "  whether it exists. The read-only box to its right shows either\n"
    "  \"<name>: column <N>\" (1-based) if it exists, or \"<name> does not\n"
    "  exist\" if it does not. The check runs immediately as you type\n"
    "  (after stripping surrounding whitespace); an empty query clears\n"
    "  the output. The [Add] button is enabled only when the query text\n"
    "  is non-empty AND the name does not already exist in the header --\n"
    "  click it to append that column name to the Header box.\n"
    "- Load CSV / Save CSV: if the first row of the loaded file is all\n"
    "  numeric it is treated as data (no header); otherwise it is used as\n"
    "  the header row and data starts on the second file row.\n"
    "- Insert a new row after row N: adds a blank (NaN) row right after\n"
    "  1-based row N; leave the box blank to append at the end.\n"
    "- Delete row: removes the 1-based row number entered.\n"
    "- Insert a new column after column N: adds a blank (NaN) column right\n"
    "  after 1-based column N; leave the box blank to append at the end.\n"
    "- Delete column: removes the column whose header name matches the\n"
    "  text entered (looked up in the Header box, not by index).\n"
    "- Spreadsheet: scrollable preview/editor. Double-click a cell to edit\n"
    "  its value. Drag a column border to resize it.\n"
    "- Write log: read-only, shows time + node name + row for every write.\n"
    "  Not saved to the project file (header + data are).\n\n"
    "Pins:\n"
    "- data [ARRAY, optional]: values to write.\n"
    "- row [SCALAR, optional]: 1-based row number; append if not connected.\n"
    "- columns [STRING, optional]: comma-separated column names matching\n"
    "  data positionally, e.g. \"step, xi, xi_err\". When connected, data\n"
    "  is written by NAME instead of by flattened position: each named\n"
    "  column that does not already exist in the header is appended (as\n"
    "  an all-NaN column first), the table is grown with all-NaN rows up\n"
    "  to the target row if needed, and only the named cells in that row\n"
    "  are written -- every other cell in a newly created row or column\n"
    "  stays NaN. When NOT connected, data is written the previous way:\n"
    "  flattened positionally into the target row starting at column 1.\n"
    "- writer [STRING, optional]: label recorded in the write log for pin\n"
    "  writes; defaults to \"(pin input)\".\n"
    "- array [ARRAY out]: current full table.\n"
    "- count [SCALAR out]: current row count.\n\n"
    "Keyboard Shortcut:\n"
    "Ctrl-H / Ctrl-h - show this help window.\n"
)


class SharedArrayNode(BaseNode):
    """
    A project-global named 2D array ("global variable"), editable through
    a spreadsheet-like inspector and reachable by name from any node.
    """

    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE = "shared_array"
    DISPLAY_NAME = "Shared Array"
    CATEGORY = "process"
    SEARCH_KEYWORDS = ("global", "variable", "shared", "table", "blackboard")
    NODE_WIDTH = 200
    NODE_HEIGHT = 110

    HELP_TEXT = _HELP_TEXT

    MAX_PREVIEW_ROWS = 1000
    MAX_PREVIEW_COLS = 200

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[
                PinDef("data", PinType.ARRAY, "data", optional=True),
                PinDef("row", PinType.SCALAR, "row", optional=True),
                PinDef("columns", PinType.STRING, "columns", optional=True),
                PinDef("writer", PinType.STRING, "writer", optional=True),
            ],
            outputs=[
                PinDef("array", PinType.ARRAY, "array"),
                PinDef("count", PinType.SCALAR, "count"),
            ],
        )

    def get_help_text(self) -> str:
        return _HELP_TEXT

    # ── init state ────────────────────────────────────────────────

    def _init_state(self) -> None:
        if hasattr(self, "_name_var"):
            return

        default_name = registry.unique_name(f"arr_{self.node_id}", exclude_owner_id=self.node_id)
        registry.register(default_name, self.node_id,
                           array=np.zeros((0, 0), dtype=np.float64), header=[])
        self._committed_name = default_name

        self._name_var = tk.StringVar(value=default_name)
        self._name_var.trace_add("write", self._on_name_var_changed)
        self._header_var = tk.StringVar(value="")
        self._csv_path_var = tk.StringVar(value="(no file)")
        self._status_var = tk.StringVar(value="0 rows")
        self._last_csv_path: str = ""
        self._insert_row_var = tk.StringVar(value="")
        self._delete_row_var = tk.StringVar(value="")
        self._insert_col_var = tk.StringVar(value="")
        self._delete_col_var = tk.StringVar(value="")

        # header-query row
        self._header_query_var = tk.StringVar(value="")
        self._header_query_var.trace_add("write", self._on_header_query_changed)
        self._header_query_output_var = tk.StringVar(value="")

        # canvas item ids
        self._name_item = None
        self._count_item = None

        # inspector widgets (only exist while inspector is open)
        self._name_entry: tk.Entry | None = None
        self._header_entry: tk.Entry | None = None
        self._header_query_entry: tk.Entry | None = None
        self._header_query_output_entry: tk.Entry | None = None
        self._header_query_add_btn: tk.Button | None = None
        self._tree: ttk.Treeview | None = None
        self._log_text: tk.Text | None = None
        self._help_popup: tk.Toplevel | None = None
        self._poll_after_id = None

    # ── array access helpers ──────────────────────────────────────

    def _get_array(self) -> np.ndarray:
        arr = registry.get_array(self._committed_name)
        return arr if arr is not None else np.zeros((0, 0), dtype=np.float64)

    def _col_names_list(self) -> list[str]:
        raw = self._header_var.get().strip()
        if not raw:
            return []
        return [c.strip() for c in raw.split(",")]

    # ── build_body ────────────────────────────────────────────────

    def build_body(self) -> None:
        self._init_state()
        x, y, w, h = self.x, self.y, self.width, self.height

        self._body_rect = self.canvas.create_rectangle(
            x, y, x + w, y + h,
            fill="#eef0f8", outline="#5f6ba8", width=2,
            tags=(self.node_id, "node_body"))
        self._title_item = self.canvas.create_text(
            x + w / 2, y + 13,
            text=self.DISPLAY_NAME,
            font=("Arial", 9, "bold"), fill="#33396b",
            tags=(self.node_id,))
        self._name_item = self.canvas.create_text(
            x + w / 2, y + h / 2 - 8,
            text=self._committed_name,
            font=("Courier", 10, "bold"), fill="#33396b",
            tags=(self.node_id,))
        self._count_item = self.canvas.create_text(
            x + w / 2, y + h / 2 + 12,
            text="0 rows",
            font=("Arial", 9), fill="#5f6ba8",
            tags=(self.node_id,))

        self._canvas_items += [self._body_rect, self._title_item,
                                self._name_item, self._count_item]
        self._update_canvas_labels()

    def _update_canvas_labels(self) -> None:
        if self._name_item is not None:
            self.canvas.itemconfig(self._name_item, text=self._committed_name)
        if self._count_item is not None:
            n = self._get_array().shape[0]
            self.canvas.itemconfig(self._count_item, text=f"{n} rows")

    # ── build_inspector ───────────────────────────────────────────

    def build_inspector(self, parent: tk.Frame) -> None:
        self._init_state()

        win = self._inspector_win
        if win is not None:
            try:
                win.geometry("640x560")
                win.minsize(480, 320)
            except Exception:
                pass
            for seq in ("<Control-h>", "<Control-H>"):
                win.bind(seq, lambda e: self._open_help())

        # ── scrollable container: the inspector UI is tall ────────────
        outer_canvas = tk.Canvas(parent, highlightthickness=0)
        outer_vsb = ttk.Scrollbar(parent, orient="vertical", command=outer_canvas.yview)
        outer_canvas.configure(yscrollcommand=outer_vsb.set)
        outer_vsb.pack(side="right", fill="y")
        outer_canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(outer_canvas)
        inner_window = outer_canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_configure(_e=None):
            outer_canvas.configure(scrollregion=outer_canvas.bbox("all"))
        inner.bind("<Configure>", _on_inner_configure)

        def _on_canvas_configure(e):
            outer_canvas.itemconfig(inner_window, width=e.width)
        outer_canvas.bind("<Configure>", _on_canvas_configure)

        def _on_mousewheel(e):
            outer_canvas.yview_scroll(-1 if e.delta > 0 else 1, "units")
        outer_canvas.bind("<MouseWheel>", _on_mousewheel)
        inner.bind("<MouseWheel>", _on_mousewheel)

        parent = inner   # everything below is built inside the scrollable area

        # ── global name ───────────────────────────────────────────
        name_frame = tk.LabelFrame(
            parent, text="Global name (unique; letters, digits, _ only)",
            font=("Arial", 9), padx=6, pady=4)
        name_frame.pack(fill="x", pady=(0, 6))

        self._name_entry = tk.Entry(
            name_frame, textvariable=self._name_var, font=("Courier", 10))
        self._name_entry.pack(fill="x")
        self._on_name_var_changed()

        # ── header ────────────────────────────────────────────────
        header_frame = tk.LabelFrame(
            parent, text="Header (comma-separated column names)",
            font=("Arial", 9), padx=6, pady=4)
        header_frame.pack(fill="x", pady=(0, 6))

        self._header_entry = tk.Entry(
            header_frame, textvariable=self._header_var, font=("Courier", 9))
        self._header_entry.pack(fill="x")
        self._header_entry.bind("<Return>", lambda e: self._on_header_committed())
        self._header_entry.bind("<FocusOut>", lambda e: self._on_header_committed())

        tk.Label(
            header_frame,
            text='e.g.  "step, prev idx, next idx, prev pts x, prev pts y"',
            font=("Arial", 8), fg="#666666", anchor="w").pack(fill="x")

        # ── header query row ─────────────────────────────────────
        query_row = tk.Frame(header_frame)
        query_row.pack(fill="x", pady=(6, 0))

        tk.Label(query_row, text="Query column:", font=("Arial", 8)).pack(side="left")
        self._header_query_entry = tk.Entry(
            query_row, textvariable=self._header_query_var, font=("Courier", 9), width=14)
        self._header_query_entry.pack(side="left", padx=(4, 6))

        self._header_query_output_entry = tk.Entry(
            query_row, textvariable=self._header_query_output_var, font=("Courier", 9),
            state="readonly", readonlybackground="#f4f4f4", fg="#333333")
        self._header_query_output_entry.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self._header_query_add_btn = tk.Button(
            query_row, text="Add", font=("Arial", 8), state="disabled",
            command=self._on_header_query_add)
        self._header_query_add_btn.pack(side="left")

        self._refresh_header_query()

        # ── CSV ───────────────────────────────────────────────────
        csv_frame = tk.LabelFrame(parent, text="CSV", font=("Arial", 9), padx=6, pady=4)
        csv_frame.pack(fill="x", pady=(0, 6))

        btn_row = tk.Frame(csv_frame)
        btn_row.pack(fill="x")
        tk.Button(btn_row, text="Load CSV...", font=("Arial", 9),
                  command=self._on_load_csv).pack(side="left")
        tk.Button(btn_row, text="Save CSV...", font=("Arial", 9),
                  command=self._on_save_csv).pack(side="left", padx=(6, 0))
        tk.Label(btn_row, textvariable=self._csv_path_var, font=("Arial", 8),
                 fg="#224488", anchor="w").pack(side="left", padx=(8, 0), fill="x", expand=True)
        tk.Button(btn_row, text="Help (Ctrl-H)", font=("Arial", 8),
                  command=self._open_help).pack(side="right")

        # ── row operations ──────────────────────────────────────────
        row_ops_frame = tk.LabelFrame(parent, text="Row operations",
                                       font=("Arial", 9), padx=6, pady=4)
        row_ops_frame.pack(fill="x", pady=(0, 6))

        insert_row = tk.Frame(row_ops_frame)
        insert_row.pack(fill="x")
        tk.Button(insert_row, text="Insert a new row after row", font=("Arial", 9),
                  command=self._on_insert_row).pack(side="left")
        tk.Entry(insert_row, textvariable=self._insert_row_var, font=("Arial", 9),
                 width=8).pack(side="left", padx=(6, 0))
        tk.Label(insert_row, text="(blank = append at end)", font=("Arial", 8),
                 fg="#666666").pack(side="left", padx=(8, 0))

        delete_row = tk.Frame(row_ops_frame)
        delete_row.pack(fill="x", pady=(4, 0))
        tk.Button(delete_row, text="Delete row", font=("Arial", 9),
                  command=self._on_delete_row).pack(side="left")
        tk.Entry(delete_row, textvariable=self._delete_row_var, font=("Arial", 9),
                 width=8).pack(side="left", padx=(6, 0))

        # ── column operations ──────────────────────────────────
        col_ops_frame = tk.LabelFrame(parent, text="Column operations",
                                       font=("Arial", 9), padx=6, pady=4)
        col_ops_frame.pack(fill="x", pady=(0, 6))

        insert_col = tk.Frame(col_ops_frame)
        insert_col.pack(fill="x")
        tk.Button(insert_col, text="Insert a new column after column", font=("Arial", 9),
                  command=self._on_insert_column).pack(side="left")
        tk.Entry(insert_col, textvariable=self._insert_col_var, font=("Arial", 9),
                 width=8).pack(side="left", padx=(6, 0))
        tk.Label(insert_col, text="(1-based; blank = append at end)", font=("Arial", 8),
                 fg="#666666").pack(side="left", padx=(8, 0))

        delete_col = tk.Frame(col_ops_frame)
        delete_col.pack(fill="x", pady=(4, 0))
        tk.Button(delete_col, text="Delete column", font=("Arial", 9),
                  command=self._on_delete_column).pack(side="left")
        tk.Entry(delete_col, textvariable=self._delete_col_var, font=("Arial", 9),
                 width=14).pack(side="left", padx=(6, 0))
        tk.Label(delete_col, text="(header name)", font=("Arial", 8),
                 fg="#666666").pack(side="left", padx=(8, 0))

        # ── status ────────────────────────────────────────────────
        tk.Label(parent, textvariable=self._status_var, font=("Arial", 9),
                 anchor="w", justify="left", fg="#333366").pack(fill="x", pady=(0, 6))

        # ── spreadsheet + write log, in a mouse-resizable vertical split ──
        paned = tk.PanedWindow(parent, orient=tk.VERTICAL, sashrelief=tk.RAISED, sashwidth=6)
        paned.pack(fill="both", expand=True, pady=(0, 6))

        sheet_frame = tk.LabelFrame(paned, text="Data (double-click a cell to edit)",
                                     font=("Arial", 9), padx=4, pady=4)

        self._tree = ttk.Treeview(sheet_frame, show="headings", selectmode="browse", height=10)
        vsb = ttk.Scrollbar(sheet_frame, orient="vertical", command=self._tree.yview)
        hsb = ttk.Scrollbar(sheet_frame, orient="horizontal", command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        hsb.pack(side="bottom", fill="x")
        vsb.pack(side="right", fill="y")
        self._tree.pack(fill="both", expand=True)
        self._tree.bind("<Double-1>", self._on_cell_double_click)
        paned.add(sheet_frame, height=260, minsize=120, stretch="always")

        # ── write log ─────────────────────────────────────────────
        log_frame = tk.LabelFrame(paned, text="Write log (not saved to project file)",
                                   font=("Arial", 9), padx=4, pady=4)

        self._log_text = tk.Text(log_frame, height=6, font=("Courier", 8),
                                  state="disabled", wrap="none")
        log_vsb = ttk.Scrollbar(log_frame, orient="vertical", command=self._log_text.yview)
        self._log_text.configure(yscrollcommand=log_vsb.set)
        log_vsb.pack(side="right", fill="y")
        self._log_text.pack(fill="both", expand=True)
        paned.add(log_frame, height=110, minsize=60, stretch="always")

        self._refresh_spreadsheet()
        self._refresh_log()
        self._schedule_poll()

    def close_inspector(self) -> None:
        if self._poll_after_id is not None and self._inspector_win is not None:
            try:
                self._inspector_win.after_cancel(self._poll_after_id)
            except Exception:
                pass
        self._poll_after_id = None
        super().close_inspector()
        self._name_entry = None
        self._header_entry = None
        self._header_query_entry = None
        self._header_query_output_entry = None
        self._header_query_add_btn = None
        self._tree = None
        self._log_text = None

    def _schedule_poll(self) -> None:
        win = self._inspector_win
        if win is None or not win.winfo_exists():
            return
        self._refresh_spreadsheet()
        self._refresh_log()
        self._poll_after_id = win.after(500, self._schedule_poll)

    # ── help popup ────────────────────────────────────────────────

    def _open_help(self) -> None:
        if self._help_popup is not None and self._help_popup.winfo_exists():
            self._help_popup.lift()
            return
        popup = tk.Toplevel()
        popup.title("Shared Array - Help")
        popup.geometry("700x560")
        popup.resizable(True, True)
        try:
            px, py = self.canvas.winfo_pointerxy()
            popup.geometry(f"+{px + 16}+{py + 16}")
        except Exception:
            pass
        body = tk.Frame(popup, bg="#f8f8f8", padx=10, pady=8)
        body.pack(fill="both", expand=True)
        txt = tk.Text(body, font=("Courier", 9), bg="#f8f8f8", fg="#222222",
                      wrap=tk.WORD, relief=tk.FLAT)
        vsb = ttk.Scrollbar(body, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", _HELP_TEXT)
        txt.configure(state="disabled")
        tk.Button(body, text="Close", font=("Arial", 9),
                  command=popup.destroy).pack(anchor="e", pady=(8, 0))
        popup.bind("<Escape>", lambda _e: popup.destroy())
        popup.protocol("WM_DELETE_WINDOW", popup.destroy)
        self._help_popup = popup

    # ── name validation / commit ──────────────────────────────────

    def _on_name_var_changed(self, *_args) -> None:
        if self._name_entry is None or not self._name_entry.winfo_exists():
            return
        text = self._name_var.get()

        if not registry.is_valid_name(text):
            self._name_entry.configure(fg="#cc0000")
            self._status_var.set("invalid name: use only letters, digits and _")
            return

        if registry.is_name_taken(text, exclude_owner_id=self.node_id):
            self._name_entry.configure(fg="#cc0000")
            self._status_var.set(f"name '{text}' is already used by another Shared Array")
            return

        self._name_entry.configure(fg="#000000")
        if text != self._committed_name:
            if registry.rename(self._committed_name, text, self.node_id):
                self._committed_name = text
                self._status_var.set(f"renamed to '{text}'")
                self._update_canvas_labels()

    def _apply_requested_name(self, requested_name: str) -> None:
        """Used during set_params(); auto-uniquifies on conflict instead of rejecting."""
        if not requested_name or requested_name == self._committed_name:
            return
        if registry.is_valid_name(requested_name) and not registry.is_name_taken(
                requested_name, exclude_owner_id=self.node_id):
            target = requested_name
        else:
            target = registry.unique_name(requested_name, exclude_owner_id=self.node_id)
        if registry.rename(self._committed_name, target, self.node_id):
            self._committed_name = target
        self._name_var.set(self._committed_name)

    def _on_header_committed(self) -> None:
        registry.register(self._committed_name, self.node_id, header=self._col_names_list())
        self._refresh_spreadsheet()
        self._refresh_header_query()

    # ── header query row ──────────────────────────────────────────

    def _on_header_query_changed(self, *_args) -> None:
        self._refresh_header_query()

    def _refresh_header_query(self) -> None:
        """
        Re-evaluate the header-query output text and the [Add] button's
        enabled state from the current query text and the current Header
        box contents. Runs on every keystroke in the query box (via the
        StringVar trace) and also whenever the Header box itself changes,
        since the header the query is checked against may have moved.
        """
        if (self._header_query_output_entry is None
                or not self._header_query_output_entry.winfo_exists()):
            return

        query = self._header_query_var.get().strip()
        if not query:
            self._header_query_output_var.set("")
            if self._header_query_add_btn is not None and self._header_query_add_btn.winfo_exists():
                self._header_query_add_btn.configure(state="disabled")
            return

        header = self._col_names_list()
        if query in header:
            col_1based = header.index(query) + 1
            self._header_query_output_var.set(f"{query}: column {col_1based}")
            add_state = "disabled"   # already exists -- cannot add again
        else:
            self._header_query_output_var.set(f"{query} does not exist")
            add_state = "normal"     # does not exist -- user may add it

        if self._header_query_add_btn is not None and self._header_query_add_btn.winfo_exists():
            self._header_query_add_btn.configure(state=add_state)

    def _on_header_query_add(self) -> None:
        query = self._header_query_var.get().strip()
        if not query:
            return
        header = self._col_names_list()
        if query in header:
            # Should not normally happen since [Add] is disabled in this
            # case, but guard anyway in case the header changed elsewhere
            # between keystroke validation and the button click.
            self._refresh_header_query()
            return

        header.append(query)
        self._header_var.set(", ".join(header))
        self._on_header_committed()
        self._status_var.set(f"added column '{query}' to header")

    # ── formatting helpers ────────────────────────────────────────

    @staticmethod
    def _format_display(v) -> str:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return str(v)
        if np.isnan(fv):
            return ""
        if fv.is_integer() and abs(fv) < 1e15:
            return str(int(fv))
        return f"{fv:.10g}"

    @staticmethod
    def _format_edit(v) -> str:
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return ""
        if np.isnan(fv):
            return ""
        return repr(fv)

    # ── spreadsheet ───────────────────────────────────────────────

    def _refresh_spreadsheet(self) -> None:
        if self._tree is None or not self._tree.winfo_exists():
            return
        arr = self._get_array()
        n_rows, n_cols = (arr.shape if arr.ndim == 2 else (0, 0))
        n_cols = min(n_cols, self.MAX_PREVIEW_COLS)
        header = self._col_names_list()

        col_ids = ["#row"] + [str(c) for c in range(n_cols)]
        self._tree.configure(columns=col_ids)
        self._tree.heading("#row", text="row")
        self._tree.column("#row", width=55, anchor="e", stretch=False)
        for c in range(n_cols):
            label = header[c] if c < len(header) else f"col_{c}"
            self._tree.heading(str(c), text=label)
            self._tree.column(str(c), width=90, anchor="e", stretch=False)

        selected = self._tree.selection()
        for item in self._tree.get_children():
            self._tree.delete(item)

        start = max(0, n_rows - self.MAX_PREVIEW_ROWS)
        for r in range(start, n_rows):
            row_vals = [str(r + 1)] + [
                self._format_display(v) for v in arr[r, :n_cols]]
            tag = "even" if (r - start) % 2 == 0 else "odd"
            self._tree.insert("", "end", iid=str(r + 1), values=row_vals, tags=(tag,))
        self._tree.tag_configure("even", background="#eef1fa")
        self._tree.tag_configure("odd", background="#ffffff")
        if selected and self._tree.exists(selected[0]):
            self._tree.selection_set(selected)

    def _on_cell_double_click(self, event) -> None:
        tree = self._tree
        if tree is None:
            return
        region = tree.identify_region(event.x, event.y)
        if region != "cell":
            return
        row_iid = tree.identify_row(event.y)
        col_id = tree.identify_column(event.x)
        if not row_iid or not col_id or col_id == "#1":
            return  # "#1" is the read-only row-number column

        columns = tree["columns"]
        col_index = int(col_id.replace("#", "")) - 1
        if col_index < 0 or col_index >= len(columns):
            return
        data_col = int(columns[col_index])
        row_number = int(row_iid)

        arr = self._get_array()
        if arr.ndim != 2 or row_number - 1 >= arr.shape[0] or data_col >= arr.shape[1]:
            return

        bbox = tree.bbox(row_iid, col_id)
        if not bbox:
            return
        x, y, w, h = bbox

        current_val = arr[row_number - 1, data_col]
        edit_var = tk.StringVar(value=self._format_edit(current_val))
        entry = tk.Entry(tree, textvariable=edit_var, font=("Courier", 9))
        entry.place(x=x, y=y, width=w, height=h)
        entry.focus_set()
        entry.select_range(0, "end")

        def commit(_event=None) -> None:
            text = edit_var.get().strip()
            new_val = float("nan") if text == "" else None
            if new_val is None:
                try:
                    new_val = float(text)
                except ValueError:
                    entry.destroy()
                    return
            registry.set_cell(self._committed_name, row_number - 1, data_col, new_val,
                               writer_name=self.get_inspector_title())
            entry.destroy()
            self._refresh_spreadsheet()
            self._refresh_log()

        def cancel(_event=None) -> None:
            entry.destroy()

        entry.bind("<Return>", commit)
        entry.bind("<FocusOut>", commit)
        entry.bind("<Escape>", cancel)

    # ── row insert / delete ───────────────────────────────────────

    def _on_insert_row(self) -> None:
        text = self._insert_row_var.get().strip()
        arr = self._get_array()
        n_rows, n_cols = arr.shape if arr.ndim == 2 else (0, 0)
        if n_cols == 0:
            header = self._col_names_list()
            n_cols = len(header) if header else 1
            arr = arr.reshape(0, n_cols) if arr.size == 0 else arr

        if text == "":
            after = n_rows
        else:
            try:
                after = int(round(float(text)))
            except ValueError:
                self._status_var.set(f"invalid row number: {text!r}")
                return
        if after < 0:
            self._status_var.set("row number must be >= 0")
            return
        after = min(after, n_rows)

        new_row = np.full((1, n_cols), np.nan, dtype=np.float64)
        new_arr = np.vstack([arr[:after], new_row, arr[after:]])
        registry.set_array(self._committed_name, new_arr, writer_name=self.get_inspector_title())
        self._status_var.set(f"inserted new row after row {after} (now row {after + 1})")
        self._update_canvas_labels()
        self._refresh_spreadsheet()
        self._refresh_log()

    def _on_delete_row(self) -> None:
        text = self._delete_row_var.get().strip()
        if text == "":
            self._status_var.set("enter a row number to delete")
            return
        try:
            row_number = int(round(float(text)))
        except ValueError:
            self._status_var.set(f"invalid row number: {text!r}")
            return

        arr = self._get_array()
        n_rows = arr.shape[0] if arr.ndim == 2 else 0
        if row_number < 1 or row_number > n_rows:
            self._status_var.set(f"row {row_number} out of range (1..{n_rows})")
            return

        new_arr = np.delete(arr, row_number - 1, axis=0)
        registry.set_array(self._committed_name, new_arr, writer_name=self.get_inspector_title())
        self._status_var.set(f"deleted row {row_number}")
        self._update_canvas_labels()
        self._refresh_spreadsheet()
        self._refresh_log()

    # ── column insert / delete ─────────────────────────────────────

    def _on_insert_column(self) -> None:
        text = self._insert_col_var.get().strip()
        arr = self._get_array()
        n_rows, n_cols = arr.shape if arr.ndim == 2 else (0, 0)
        header = self._col_names_list()

        if text == "":
            after = n_cols
        else:
            try:
                after = int(round(float(text)))
            except ValueError:
                self._status_var.set(f"invalid column number: {text!r}")
                return
        if after < 0:
            self._status_var.set("column number must be >= 0")
            return
        after = min(after, n_cols)

        new_col = np.full((n_rows, 1), np.nan, dtype=np.float64)
        if n_cols == 0:
            new_arr = new_col
        else:
            new_arr = np.hstack([arr[:, :after], new_col, arr[:, after:]])

        if header:
            new_header = header[:after] + [""] + header[after:]
            self._header_var.set(", ".join(new_header))
        else:
            new_header = header

        registry.set_array(self._committed_name, new_arr, header=new_header,
                            writer_name=self.get_inspector_title())
        self._status_var.set(f"inserted new column after column {after} (now column {after + 1})")
        self._update_canvas_labels()
        self._refresh_spreadsheet()
        self._refresh_log()

    def _on_delete_column(self) -> None:
        name = self._delete_col_var.get().strip()
        if name == "":
            self._status_var.set("enter the header name of the column to delete")
            return

        header = self._col_names_list()
        if name not in header:
            self._status_var.set(f"column '{name}' not found in header")
            return
        idx = header.index(name)

        arr = self._get_array()
        if arr.ndim != 2 or idx >= arr.shape[1]:
            self._status_var.set(f"column '{name}' has no matching data column")
            return

        new_arr = np.delete(arr, idx, axis=1)
        new_header = header[:idx] + header[idx + 1:]
        self._header_var.set(", ".join(new_header))
        registry.set_array(self._committed_name, new_arr, header=new_header,
                            writer_name=self.get_inspector_title())
        self._status_var.set(f"deleted column '{name}'")
        self._update_canvas_labels()
        self._refresh_spreadsheet()
        self._refresh_log()

    # ── log ───────────────────────────────────────────────────────

    def _refresh_log(self) -> None:
        if self._log_text is None or not self._log_text.winfo_exists():
            return
        lines = registry.get_log(self._committed_name)
        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.insert("end", "\n".join(lines[-300:]))
        self._log_text.see("end")
        self._log_text.configure(state="disabled")

    # ── CSV load / save ───────────────────────────────────────────

    @staticmethod
    def _is_numeric_cell(text: str) -> bool:
        try:
            float(text)
            return True
        except (TypeError, ValueError):
            return False

    def _on_load_csv(self) -> None:
        base = get_project_directory()
        path = filedialog.askopenfilename(
            title="Load CSV",
            initialdir=str(base) if base else ".",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            with open(path, "r", newline="", encoding="utf-8") as f:
                rows = [r for r in csv.reader(f) if r]
            if not rows:
                self._status_var.set("CSV file is empty")
                return

            first_row = rows[0]
            has_header = not all(self._is_numeric_cell(c) for c in first_row)
            if has_header:
                header = [c.strip() for c in first_row]
                data_rows = rows[1:]
            else:
                header = self._col_names_list()
                data_rows = rows

            if data_rows:
                n_cols = len(data_rows[0])
                arr = np.full((len(data_rows), n_cols), np.nan, dtype=np.float64)
                for i, r in enumerate(data_rows):
                    for j, c in enumerate(r[:n_cols]):
                        try:
                            arr[i, j] = float(c)
                        except ValueError:
                            arr[i, j] = np.nan
            else:
                arr = np.zeros((0, len(header)), dtype=np.float64)

            if has_header:
                self._header_var.set(", ".join(header))

            registry.set_array(self._committed_name, arr, header=self._col_names_list(),
                                writer_name=self.get_inspector_title())
            self._last_csv_path = path
            self._csv_path_var.set(Path(path).name)
            self._status_var.set(f"loaded {arr.shape[0]} rows x {arr.shape[1]} cols from CSV")
            self._update_canvas_labels()
            self._refresh_spreadsheet()
            self._refresh_log()
            self._refresh_header_query()
        except Exception as e:
            self._status_var.set(f"load error: {e}")

    def _on_save_csv(self) -> None:
        base = get_project_directory()
        initial_dir = str(Path(self._last_csv_path).parent) if self._last_csv_path else (
            str(base) if base else ".")
        path = filedialog.asksaveasfilename(
            title="Save CSV",
            initialdir=initial_dir,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            defaultextension=".csv")
        if not path:
            return
        arr = self._get_array()
        header = self._col_names_list()
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                n_cols = arr.shape[1] if arr.ndim == 2 else 0
                if header:
                    hdr = header[:n_cols]
                    while len(hdr) < n_cols:
                        hdr.append(f"col_{len(hdr)}")
                    writer.writerow(hdr)
                for row in arr:
                    writer.writerow([self._format_edit(v) for v in row])
            self._last_csv_path = path
            self._csv_path_var.set(Path(path).name)
            self._status_var.set(f"saved {arr.shape[0]} rows -> {Path(path).name}")
        except Exception as e:
            self._status_var.set(f"save error: {e}")

    # ── named-column row write (columns pin) ────────────────────────

    def _write_named_row(self, row_1based: int, column_names: list[str],
                         values: np.ndarray, writer_name: str) -> tuple[bool, str]:
        """
        Write `values` into row `row_1based` (1-based) at the columns
        named by `column_names` (positionally matched to `values`).

        Any name in `column_names` not already present in the header is
        appended as a new all-NaN column (extending every existing row).
        If `row_1based` is beyond the current row count, the table is
        grown with all-NaN rows up to (and including) that row first.
        Every cell not explicitly named/targeted stays NaN, exactly like
        the existing "Insert row"/"Insert column" operations.

        Returns (success, message).
        """
        if row_1based < 1:
            return False, f"row must be >= 1 (1-based), got {row_1based}"
        if len(column_names) != values.size:
            return False, (f"columns count ({len(column_names)}) does not match "
                           f"data length ({values.size})")

        arr = self._get_array()
        n_rows, n_cols = arr.shape if arr.ndim == 2 else (0, 0)
        header = self._col_names_list()

        # Grow the header/table with new all-NaN columns for any name
        # not already present, same NaN-fill convention as "Insert column".
        for name in column_names:
            if name not in header:
                header.append(name)
                new_col = np.full((n_rows, 1), np.nan, dtype=np.float64)
                arr = new_col if n_cols == 0 else np.hstack([arr, new_col])
                n_cols += 1

        # Grow the table with new all-NaN rows up to row_1based if needed,
        # same NaN-fill convention as "Insert row".
        if row_1based > n_rows:
            extra = row_1based - n_rows
            pad = np.full((extra, n_cols), np.nan, dtype=np.float64)
            arr = pad if n_rows == 0 else np.vstack([arr, pad])
            n_rows = row_1based

        # Write only the named cells; everything else in the row/columns
        # created above remains NaN.
        row_idx = row_1based - 1
        for name, value in zip(column_names, values):
            col_idx = header.index(name)
            arr[row_idx, col_idx] = float(value)

        self._header_var.set(", ".join(header))
        registry.set_array(self._committed_name, arr, header=header, writer_name=writer_name)
        return True, f"row {row_1based} written by {writer_name} (columns: {', '.join(column_names)})"

    # ── compute ───────────────────────────────────────────────────

    def compute(self, inputs: dict) -> dict:
        self._init_state()

        data = inputs.get("data")
        if data is not None:
            values = np.asarray(data, dtype=np.float64).ravel()
            if values.size > 0:
                writer_raw = inputs.get("writer")
                writer_name = str(writer_raw).strip() if writer_raw else "(pin input)"

                raw_columns = inputs.get("columns")
                if raw_columns is not None and str(raw_columns).strip():
                    # Named-column write: row and columns pins together
                    # decide exactly which cells are touched; everything
                    # else stays NaN. See _write_named_row() docstring.
                    raw_row = inputs.get("row")
                    if raw_row is None:
                        self._status_var.set(
                            "columns pin connected but row pin is missing -- write skipped")
                    else:
                        try:
                            row_1based = int(round(float(raw_row)))
                        except (TypeError, ValueError):
                            self._status_var.set(f"invalid row value: {raw_row!r}")
                            row_1based = None

                        if row_1based is not None:
                            column_names = [c.strip() for c in str(raw_columns).split(",")
                                           if c.strip()]
                            ok, msg = self._write_named_row(
                                row_1based, column_names, values, writer_name)
                            self._status_var.set(msg)
                            if ok:
                                self._update_canvas_labels()
                                if self.is_inspector_open():
                                    self._refresh_spreadsheet()
                                    self._refresh_log()
                                    self._refresh_header_query()
                else:
                    # Previous behavior: flattened positional write into
                    # the target row starting at column 1.
                    raw_row = inputs.get("row")
                    if raw_row is not None:
                        try:
                            target_row = int(round(float(raw_row))) - 1
                        except (TypeError, ValueError):
                            target_row = -1
                    else:
                        target_row = self._get_array().shape[0]

                    if target_row >= 0:
                        ok = registry.set_row(self._committed_name, target_row, values,
                                               writer_name=writer_name)
                        if ok:
                            self._status_var.set(f"row {target_row + 1} written by {writer_name}")
                            self._update_canvas_labels()
                            if self.is_inspector_open():
                                self._refresh_spreadsheet()
                                self._refresh_log()
                        else:
                            self._status_var.set(
                                f"write failed: row {target_row + 1} has wrong column count")

        arr = self._get_array()
        return {
            "array": arr,
            "count": float(arr.shape[0]),
        }

    # ── serialization ─────────────────────────────────────────────

    def get_params(self) -> dict:
        arr = self._get_array()
        return {
            "name": self._committed_name,
            "header": self._header_var.get(),
            "data": arr.tolist(),
        }

    def set_params(self, params: dict) -> None:
        self._init_state()

        requested_name = str(params.get("name", "")).strip()
        header_str = str(params.get("header", ""))
        raw_data = params.get("data", [])

        try:
            arr = np.array(raw_data, dtype=np.float64)
            if arr.ndim == 1:
                arr = arr.reshape(0, 0) if arr.size == 0 else arr.reshape(1, -1)
            elif arr.ndim != 2:
                arr = np.zeros((0, 0), dtype=np.float64)
        except Exception:
            arr = np.zeros((0, 0), dtype=np.float64)

        self._apply_requested_name(requested_name)

        header_list = [c.strip() for c in header_str.split(",")] if header_str.strip() else []
        registry.set_array(self._committed_name, arr, header=header_list,
                            writer_name="(project load)")
        self._header_var.set(header_str)
        self._update_canvas_labels()
        if self.is_inspector_open():
            self._refresh_spreadsheet()
            self._refresh_header_query()

    def on_destroy(self) -> None:
        if self._help_popup is not None and self._help_popup.winfo_exists():
            self._help_popup.destroy()
        registry.unregister(self._committed_name, self.node_id)
        super().on_destroy()