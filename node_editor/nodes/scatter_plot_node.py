# node_editor/nodes/scatter_plot_node.py

import io
import re
import tkinter as tk
from tkinter import ttk

import numpy as np

import matplotlib
matplotlib.use("Agg")  # backend for the embedded FigureCanvasTkAgg path below
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from node_editor.base_node import BaseNode
from node_editor.pin_types import PinSchema, PinDef, PinType
from node_editor.execution import ExecutionMode


_HELP_TEXT = (
    "Scatter Plot 2D Node\n\n"
    "Purpose:\n"
    "- Plot up to 4 independent curves, each a set of x/y points.\n"
    "  Curves may have different lengths. Renders with real\n"
    "  matplotlib, both embedded in the inspector and via a standalone\n"
    "  popup window, and generates the equivalent matplotlib source code\n"
    "  live as you change settings.\n\n"
    "Input Pins:\n"
    "- curve1 .. curve4 [ARRAY, optional]: any numpy array. What is\n"
    "  plotted is decided by the x / y expressions of each curve (see\n"
    "  below), so the arrays do not have to be (n,2).\n\n"
    "Output Pins:\n"
    "- (none) -- this is a terminal/visualization node.\n\n"
    "Inspector -- per-curve settings (left panel, one tab per curve):\n"
    "- Enabled: include this curve in the plot.\n"
    "- Data source (radio button):\n"
    "    Data from pins cN   - x and y are extracted from the pin arrays\n"
    "                          by two numpy expressions (x [ ] and y [ ]).\n"
    "                          The text box below is disabled.\n"
    "    Data from text box  - one point per line, \"x y\" or \"x,y\".\n"
    "                          The x / y expression boxes are disabled.\n"
    "- x / y expressions: ordinary Python/numpy expressions. The pin\n"
    "  arrays are available as c1, c2, c3, c4 and numpy as np. Defaults\n"
    "  for curve N are cN[:,0] and cN[:,1] (cN is an (n,2) array).\n"
    "  Examples:\n"
    "    c1[:,1] - c1[0,1]            displacement relative to first sample\n"
    "    (c1[:,1] - c1[0,1]) * 25.4   ... converted from inch to mm\n"
    "    np.arange(len(c1))           sample index as x\n"
    "    c2[:,1]                      y of curve 1 taken from pin c2\n"
    "- A label under the data source shows the number of points, or\n"
    "  \"Invalid data\" (with a short reason) if an expression fails, x and y\n"
    "  differ in length, or nothing is left to plot.\n"
    "- Legend name: label shown in the plot legend.\n"
    "- Marker: type, size, color.\n"
    "- Line: width, style, color. Set line width to 0 to hide the\n"
    "  connecting line and show markers only (a pure scatter plot).\n\n"
    "Inspector -- axes/grid settings:\n"
    "- X label / Y label / Title.\n"
    "- Axis scale: x linear/log, y linear/log. On a log axis,\n"
    "  non-positive values are not drawn.\n"
    "- Grid on/off, grid line style, grid line color.\n"
    "- Legend on/off, legend location.\n\n"
    "Inspector -- right panel:\n"
    "- Live embedded matplotlib plot, redrawn on every Apply/Compute.\n"
    "  Mouse interaction on the embedded plot:\n"
    "    Wheel        - zoom both axes about the cursor\n"
    "    Ctrl+wheel   - zoom the x axis only\n"
    "    Shift+wheel  - zoom the y axis only\n"
    "    Left-drag    - pan\n"
    "    Double-click - reset view to auto-fit the data\n"
    "  A zoomed/panned view is kept when settings change or new data\n"
    "  arrives, until you double-click to reset it (changing an axis\n"
    "  scale also resets it).\n"
    "- Generated code box: a complete, runnable matplotlib script\n"
    "  reproducing the current plot exactly (data arrays are inlined\n"
    "  as literals), regenerated live whenever the plot is redrawn.\n"
    "- \"Open in matplotlib window\" button: pops up a real, independent\n"
    "  matplotlib figure window (plt.show(block=False)) built fresh from\n"
    "  the current settings.\n\n"
    "Keyboard Shortcut:\n"
    "Ctrl-H / Ctrl-h - show this help window.\n"
)

_COMMON_COLORS = [
    "red", "green", "blue", "black", "orange",
    "purple", "brown", "magenta", "cyan", "gray",
]

_MATPLOTLIB_MARKERS = [
    ("o", "circle"), ("s", "square"), ("^", "triangle up"),
    ("v", "triangle down"), ("D", "diamond"), ("*", "star"),
    ("+", "plus"), ("x", "x"), (".", "point"), ("None", "none"),
]

_LINE_STYLES = [
    ("-", "solid"), ("--", "dashed"), ("-.", "dash-dot"),
    (":", "dotted"), ("None", "none"),
]

_LEGEND_LOCATIONS = [
    "best", "upper right", "upper left", "lower left", "lower right",
    "right", "center left", "center right", "lower center",
    "upper center", "center",
]

_DEFAULT_POINTS_TEXT = "0 0\n1 1\n2 0\n1 -1\n0 -1\n-1 0"

_DEFAULT_CURVE_COLORS = ["red", "blue", "green", "orange"]
_DEFAULT_CURVE_MARKERS = ["o", "s", "^", "D"]


def _parse_point_block(text: str) -> tuple[np.ndarray | None, int]:
    """Parse an (n,2) array from pasted text: one point per line,
    space/comma/tab separated. Blank lines and '#' comments are
    skipped. Malformed lines are skipped individually. Returns
    (array_or_None, skipped_line_count)."""
    rows: list[list[float]] = []
    skipped = 0
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            skipped += 1
            continue
        if len(vals) != 2:
            skipped += 1
            continue
        rows.append(vals)
    if not rows:
        return None, skipped
    return np.array(rows, dtype=np.float64), skipped


# Names usable inside the x / y expressions, besides np and c1..c4.
_EVAL_BUILTINS = {
    "len": len, "min": min, "max": max, "abs": abs, "sum": sum,
    "range": range, "float": float, "int": int, "round": round,
    "pow": pow, "slice": slice, "zip": zip, "list": list,
}

_PIN_NAME_RE = re.compile(r"c[1-4]$")


def _coerce_pin_array(raw) -> np.ndarray | None:
    """Pin value -> float64 ndarray for use in expressions, or None."""
    if raw is None:
        return None
    try:
        a = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    return a if a.size > 0 else None


def _eval_pin_expr(expr: str, pins: dict) -> np.ndarray:
    """Evaluate a numpy expression with c1..c4 bound to the pin arrays.
    Returns a 1-D float64 array; raises on anything else."""
    expr = expr.strip()
    if not expr:
        raise ValueError("empty expression")
    env = {"__builtins__": _EVAL_BUILTINS, "np": np}
    for k, arr in pins.items():
        if arr is not None:
            env[f"c{k}"] = arr
    try:
        with np.errstate(all="ignore"):
            val = eval(expr, env)
    except NameError as e:
        name = getattr(e, "name", None)
        if not name:
            m = re.search(r"'(\w+)'", str(e))
            name = m.group(1) if m else ""
        if _PIN_NAME_RE.match(name):
            raise ValueError(f"pin {name} has no data") from None
        raise
    a = np.asarray(val, dtype=np.float64)
    if a.ndim != 1:
        raise ValueError(f"expected a 1-D array, got shape {a.shape}")
    return a


def _short(e: Exception) -> str:
    msg = str(e).strip() or type(e).__name__
    return msg if len(msg) <= 70 else msg[:67] + "..."


def _points_msg(n: int, dropped: int = 0, skipped: int = 0) -> str:
    msg = f"{n} point" + ("" if n == 1 else "s")
    notes = []
    if dropped:
        notes.append(f"{dropped} non-finite dropped")
    if skipped:
        notes.append(f"{skipped} bad line" + ("" if skipped == 1 else "s") + " skipped")
    return msg + (f"  ({', '.join(notes)})" if notes else "")


def _py_repr_array(arr: np.ndarray) -> str:
    """Render a (n,2) array as a compact Python literal for generated code."""
    rows = ", ".join(f"[{x:.10g}, {y:.10g}]" for x, y in arr)
    return f"[{rows}]"


class ScatterPlotNode(BaseNode):
    """
    2D scatter/line plot node for up to 4 independently-styled curves.
    Each curve's x/y data comes either from numpy expressions applied
    to the pin arrays c1..c4, or from a manual text box of points. Renders with a real embedded matplotlib figure, generates
    equivalent matplotlib source code live, and can pop open a
    standalone matplotlib window built fresh from current settings.
    """

    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE = "scatter_plot"
    DISPLAY_NAME = "Scatter Plot 2D"
    CATEGORY = "process"
    SEARCH_KEYWORDS = ("plot", "scatter", "chart", "curve", "matplotlib",
                       "graph", "line plot", "xy plot")
    NODE_WIDTH = 210
    NODE_HEIGHT = 120

    HELP_TEXT = _HELP_TEXT

    MAX_CURVES = 4

    def get_pin_schema(self) -> PinSchema:
        inputs = [
            PinDef(f"curve{i}", PinType.ARRAY, f"c{i}", optional=True)
            for i in range(1, self.MAX_CURVES + 1)
        ]
        return PinSchema(inputs=inputs, outputs=[])

    def get_help_text(self) -> str:
        return _HELP_TEXT

    # ── state ─────────────────────────────────────────────────────

    def _init_state(self) -> None:
        if hasattr(self, "_curves"):
            return

        self._xlabel_var = tk.StringVar(value="x")
        self._ylabel_var = tk.StringVar(value="y")
        self._title_var = tk.StringVar(value="")
        self._grid_on_var = tk.BooleanVar(value=True)
        self._grid_style_var = tk.StringVar(value="--")
        self._grid_color_var = tk.StringVar(value="gray")
        self._legend_on_var = tk.BooleanVar(value=True)
        self._legend_loc_var = tk.StringVar(value="best")
        self._xscale_var = tk.StringVar(value="linear")   # "linear" or "log"
        self._yscale_var = tk.StringVar(value="linear")

        # Last array received on each curve pin (None = nothing connected).
        # Kept so x/y expressions can be re-evaluated while editing in the
        # inspector, without waiting for the next graph execution.
        self._pin_arrays: dict[int, np.ndarray | None] = {
            i: None for i in range(1, self.MAX_CURVES + 1)}

        self._status_var = tk.StringVar(value="not plotted yet")

        self._curves: dict[int, dict] = {}
        for i in range(1, self.MAX_CURVES + 1):
            self._curves[i] = {
                "enabled_var": tk.BooleanVar(value=(i == 1)),
                "legend_var": tk.StringVar(value=f"Curve {i}"),
                "marker_var": tk.StringVar(value=_DEFAULT_CURVE_MARKERS[i - 1]),
                "marker_size_var": tk.DoubleVar(value=6.0),
                "marker_color_var": tk.StringVar(value=_DEFAULT_CURVE_COLORS[i - 1]),
                "line_width_var": tk.DoubleVar(value=1.5),
                "line_style_var": tk.StringVar(value="-"),
                "line_color_var": tk.StringVar(value=_DEFAULT_CURVE_COLORS[i - 1]),
                "source_var": tk.StringVar(value="pin"),   # "pin" or "text"
                "x_expr_var": tk.StringVar(value=f"c{i}[:,0]"),
                "y_expr_var": tk.StringVar(value=f"c{i}[:,1]"),
                "points_var": tk.StringVar(value="no data"),
                "text_cache": _DEFAULT_POINTS_TEXT,
                "text_widget": None,
                "x_entry": None,
                "y_entry": None,
                "pin_indicator": None,
                "points_label": None,
                "data": np.empty((0, 2), dtype=np.float64),
                "valid": False,
            }

        self._fig = None
        self._ax = None
        self._fig_canvas_widget: FigureCanvasTkAgg | None = None
        self._code_text_widget: tk.Text | None = None
        self._help_popup: tk.Toplevel | None = None

        # Interactive view state for the embedded plot. _view_locked is
        # True once the user has zoomed/panned; _redraw() then restores
        # the user's limits instead of auto-fitting the data.
        self._view_locked: bool = False
        self._pan_state: dict = {
            "active": False, "inv": None, "p0": None,
            "xlim0": None, "ylim0": None,
        }

    # ── body ──────────────────────────────────────────────────────

    def build_body(self) -> None:
        self._init_state()
        x, y, w, h = self.x, self.y, self.width, self.height

        self._body_rect = self.canvas.create_rectangle(
            x, y, x + w, y + h,
            fill="#f0f4ff", outline="#5566bb", width=2,
            tags=(self.node_id, "node_body"))
        self._title_item = self.canvas.create_text(
            x + w / 2, y + 13, text=self.DISPLAY_NAME,
            font=("Arial", 9, "bold"), fill="#223377",
            tags=(self.node_id,))

        status_lbl = tk.Label(
            self.canvas, textvariable=self._status_var,
            font=("Arial", 8), bg="#f0f4ff", fg="#334477",
            wraplength=w - 12, justify="center")
        self.canvas.create_window(
            x + w / 2, y + h // 2 + 8, window=status_lbl, tags=(self.node_id,))

        self._canvas_items += [self._body_rect, self._title_item]
        self.canvas.tag_bind(self.node_id, "<Double-Button-1>", lambda e: self.open_inspector())

    # ── inspector ─────────────────────────────────────────────────

    def build_inspector(self, parent: tk.Frame) -> None:
        self._init_state()
        for child in parent.winfo_children():
            child.destroy()

        win = self._inspector_win
        if win is not None:
            for seq in ("<Control-h>", "<Control-H>"):
                win.bind(seq, lambda e: self._open_help())

        pad = {"padx": 8, "pady": 4}

        panes = tk.PanedWindow(parent, orient=tk.HORIZONTAL, sashwidth=6, showhandle=True)
        panes.pack(fill="both", expand=True)

        left = tk.Frame(panes, width=420, bg="#f7f7f7")
        right = tk.Frame(panes, width=520, bg="#ffffff")
        for f in (left, right):
            f.pack_propagate(False)
        panes.add(left, stretch="always")
        panes.add(right, stretch="always")

        self._build_left_panel(left, pad)
        self._build_right_panel(right, pad)

        if win is not None:
            win.update_idletasks()
            min_w = max(1080, win.winfo_reqwidth())
            min_h = max(660, win.winfo_reqheight())
            win.minsize(min_w, min_h)
            win.geometry(f"{min_w}x{min_h}")

        self._resolve_all_curves()
        self._redraw()

    def _make_scrollable(self, parent: tk.Frame) -> tk.Frame:
        outer = tk.Frame(parent)
        outer.pack(fill="both", expand=True)
        canvas = tk.Canvas(outer, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        inner = tk.Frame(canvas)
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win_id, width=e.width))
        return inner

    def _build_left_panel(self, parent: tk.Frame, pad: dict) -> None:
        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True)

        for i in range(1, self.MAX_CURVES + 1):
            tab = tk.Frame(nb)
            nb.add(tab, text=f"Curve {i}")
            self._build_curve_tab(tab, i, pad)

        axes_tab = tk.Frame(nb)
        nb.add(axes_tab, text="Axes / Grid / Legend")
        self._build_axes_tab(axes_tab, pad)

    def _build_curve_tab(self, parent: tk.Frame, i: int, pad: dict) -> None:
        inner = self._make_scrollable(parent)
        cfg = self._curves[i]

        top_row = tk.Frame(inner)
        top_row.pack(fill="x", **pad)
        tk.Checkbutton(top_row, text="Enabled", variable=cfg["enabled_var"],
                      font=("Arial", 9), command=self._on_apply).pack(side="left")
        tk.Label(top_row, text="Legend:", font=("Arial", 9)).pack(side="left", padx=(12, 2))
        tk.Entry(top_row, textvariable=cfg["legend_var"], width=16,
                 font=("Arial", 9)).pack(side="left")

        # ── data source: pins (numpy expressions) or manual text box ──
        data_frame = tk.LabelFrame(inner, text="Data", font=("Arial", 9), **pad)
        data_frame.pack(fill="x", **pad)

        pin_row = tk.Frame(data_frame)
        pin_row.pack(fill="x")
        tk.Radiobutton(pin_row, text=f"Data from pins c{i}", variable=cfg["source_var"],
                      value="pin", font=("Arial", 9),
                      command=lambda i=i: self._on_source_changed(i)).pack(side="left")
        pin_ind = tk.Label(pin_row, text="○ pin empty", font=("Arial", 8), fg="#aaaaaa")
        pin_ind.pack(side="left", padx=(8, 0))
        cfg["pin_indicator"] = pin_ind

        expr_row = tk.Frame(data_frame)
        expr_row.pack(fill="x", padx=(24, 0), pady=(2, 0))
        tk.Label(expr_row, text="x", font=("Arial", 9)).pack(side="left")
        x_entry = tk.Entry(expr_row, textvariable=cfg["x_expr_var"], width=14,
                           font=("Courier", 9))
        x_entry.pack(side="left", fill="x", expand=True, padx=(3, 8))
        tk.Label(expr_row, text="y", font=("Arial", 9)).pack(side="left")
        y_entry = tk.Entry(expr_row, textvariable=cfg["y_expr_var"], width=14,
                           font=("Courier", 9))
        y_entry.pack(side="left", fill="x", expand=True, padx=(3, 0))
        for entry in (x_entry, y_entry):
            entry.bind("<KeyRelease>", lambda e: self._refresh_local())
        cfg["x_entry"] = x_entry
        cfg["y_entry"] = y_entry
        tk.Label(data_frame,
                text="numpy expressions using c1..c4 and np, e.g. (c1[:,1]-c1[0,1])*25.4",
                font=("Arial", 7), fg="#888888", anchor="w").pack(fill="x", padx=(24, 0))

        tk.Radiobutton(data_frame, text="Data from text box", variable=cfg["source_var"],
                      value="text", font=("Arial", 9),
                      command=lambda i=i: self._on_source_changed(i)
                      ).pack(anchor="w", pady=(6, 0))
        tk.Label(data_frame,
                text=f"Curve {i}: one point per line \"x y\" or \"x,y\" (e.g. 0 0\\n  1,2\\n  2,3)",
                font=("Arial", 8), fg="#555555", anchor="w").pack(fill="x", padx=(24, 0))

        text_widget = tk.Text(data_frame, height=6, font=("Courier", 8), wrap=tk.NONE)
        text_widget.insert("1.0", cfg["text_cache"])
        text_widget.pack(fill="x", padx=(24, 0), pady=(0, 2))
        text_widget.bind("<KeyRelease>", lambda e: self._refresh_local())
        cfg["text_widget"] = text_widget

        points_lbl = tk.Label(data_frame, textvariable=cfg["points_var"],
                              font=("Arial", 9, "bold"), anchor="w")
        points_lbl.pack(fill="x", pady=(4, 0))
        cfg["points_label"] = points_lbl

        self._apply_source_ui(i)
        self._update_curve_entry_ui(i)

        # ── marker ────────────────────────────────────────────────
        marker_frame = tk.LabelFrame(inner, text="Marker", font=("Arial", 9), **pad)
        marker_frame.pack(fill="x", **pad)
        mrow = tk.Frame(marker_frame)
        mrow.pack(fill="x")
        tk.Label(mrow, text="Type:", font=("Arial", 8)).pack(side="left")
        ttk.Combobox(mrow, textvariable=cfg["marker_var"],
                    values=[m for m, _ in _MATPLOTLIB_MARKERS],
                    width=6, state="readonly").pack(side="left", padx=(2, 8))
        tk.Label(mrow, text="Size:", font=("Arial", 8)).pack(side="left")
        tk.Spinbox(mrow, from_=1, to=30, increment=0.5, textvariable=cfg["marker_size_var"],
                  width=5, font=("Arial", 8), command=self._redraw).pack(side="left", padx=(2, 8))
        tk.Label(mrow, text="Color:", font=("Arial", 8)).pack(side="left")
        color_combo = ttk.Combobox(mrow, textvariable=cfg["marker_color_var"],
                                   values=_COMMON_COLORS, width=8)
        color_combo.pack(side="left", padx=(2, 0))
        color_combo.bind("<<ComboboxSelected>>", lambda e: self._redraw())
        color_combo.bind("<FocusOut>", lambda e: self._redraw())
        color_combo.bind("<Return>", lambda e: self._redraw())

        # ── line ──────────────────────────────────────────────────
        line_frame = tk.LabelFrame(inner, text="Line", font=("Arial", 9), **pad)
        line_frame.pack(fill="x", **pad)
        lrow = tk.Frame(line_frame)
        lrow.pack(fill="x")
        tk.Label(lrow, text="Width:", font=("Arial", 8)).pack(side="left")
        tk.Spinbox(lrow, from_=0.0, to=10.0, increment=0.5, textvariable=cfg["line_width_var"],
                  width=5, font=("Arial", 8), command=self._redraw).pack(side="left", padx=(2, 8))
        tk.Label(lrow, text="Style:", font=("Arial", 8)).pack(side="left")
        ttk.Combobox(lrow, textvariable=cfg["line_style_var"],
                    values=[s for s, _ in _LINE_STYLES], width=5,
                    state="readonly").pack(side="left", padx=(2, 8))
        tk.Label(lrow, text="Color:", font=("Arial", 8)).pack(side="left")
        line_color_combo = ttk.Combobox(lrow, textvariable=cfg["line_color_var"],
                                        values=_COMMON_COLORS, width=8)
        line_color_combo.pack(side="left", padx=(2, 0))
        line_color_combo.bind("<<ComboboxSelected>>", lambda e: self._redraw())
        line_color_combo.bind("<FocusOut>", lambda e: self._redraw())
        line_color_combo.bind("<Return>", lambda e: self._redraw())
        tk.Label(line_frame, text="(width 0 = markers only, no connecting line)",
                font=("Arial", 7), fg="#888888").pack(anchor="w")

        for var in (cfg["marker_var"], cfg["line_style_var"]):
            var.trace_add("write", lambda *_a: self._redraw())

        tk.Button(inner, text="Apply (parse points / redraw)", font=("Arial", 9, "bold"),
                  bg="#446622", fg="white", activebackground="#557733", relief=tk.FLAT,
                  padx=10, pady=3, command=self._on_apply).pack(anchor="w", padx=8, pady=8)

    def _build_axes_tab(self, parent: tk.Frame, pad: dict) -> None:
        inner = self._make_scrollable(parent)

        labels_frame = tk.LabelFrame(inner, text="Labels", font=("Arial", 9), **pad)
        labels_frame.pack(fill="x", **pad)
        for label, var in [("X label:", self._xlabel_var),
                           ("Y label:", self._ylabel_var),
                           ("Title:", self._title_var)]:
            row = tk.Frame(labels_frame)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label, font=("Arial", 9), width=10, anchor="w").pack(side="left")
            entry = tk.Entry(row, textvariable=var, font=("Arial", 9))
            entry.pack(side="left", fill="x", expand=True)
            entry.bind("<KeyRelease>", lambda e: self._redraw())

        scale_frame = tk.LabelFrame(inner, text="Axis scale", font=("Arial", 9), **pad)
        scale_frame.pack(fill="x", **pad)
        for label, var in [("X axis:", self._xscale_var), ("Y axis:", self._yscale_var)]:
            row = tk.Frame(scale_frame)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=label, font=("Arial", 9), width=10, anchor="w").pack(side="left")
            for text, value in (("linear", "linear"), ("log", "log")):
                tk.Radiobutton(row, text=text, variable=var, value=value,
                              font=("Arial", 9), command=self._on_scale_changed
                              ).pack(side="left", padx=(0, 10))
        tk.Label(scale_frame, text="(log scale: non-positive values are not drawn)",
                font=("Arial", 7), fg="#888888").pack(anchor="w")

        grid_frame = tk.LabelFrame(inner, text="Grid", font=("Arial", 9), **pad)
        grid_frame.pack(fill="x", **pad)
        tk.Checkbutton(grid_frame, text="Grid on", variable=self._grid_on_var,
                      font=("Arial", 9), command=self._redraw).pack(anchor="w")
        grow = tk.Frame(grid_frame)
        grow.pack(fill="x", pady=2)
        tk.Label(grow, text="Style:", font=("Arial", 8)).pack(side="left")
        ttk.Combobox(grow, textvariable=self._grid_style_var,
                    values=[s for s, _ in _LINE_STYLES], width=5,
                    state="readonly").pack(side="left", padx=(2, 8))
        tk.Label(grow, text="Color:", font=("Arial", 8)).pack(side="left")
        grid_color_combo = ttk.Combobox(grow, textvariable=self._grid_color_var,
                                        values=_COMMON_COLORS, width=8)
        grid_color_combo.pack(side="left", padx=(2, 0))
        grid_color_combo.bind("<<ComboboxSelected>>", lambda e: self._redraw())
        grid_color_combo.bind("<FocusOut>", lambda e: self._redraw())
        self._grid_style_var.trace_add("write", lambda *_a: self._redraw())

        legend_frame = tk.LabelFrame(inner, text="Legend", font=("Arial", 9), **pad)
        legend_frame.pack(fill="x", **pad)
        tk.Checkbutton(legend_frame, text="Legend on", variable=self._legend_on_var,
                      font=("Arial", 9), command=self._redraw).pack(anchor="w")
        lrow = tk.Frame(legend_frame)
        lrow.pack(fill="x", pady=2)
        tk.Label(lrow, text="Location:", font=("Arial", 8)).pack(side="left")
        ttk.Combobox(lrow, textvariable=self._legend_loc_var, values=_LEGEND_LOCATIONS,
                    width=14, state="readonly").pack(side="left", padx=(2, 0))
        self._legend_loc_var.trace_add("write", lambda *_a: self._redraw())

        tk.Button(inner, text="Help (Ctrl-H)", font=("Arial", 8),
                  command=self._open_help).pack(anchor="e", padx=8, pady=8)

    def _build_right_panel(self, parent: tk.Frame, pad: dict) -> None:
        plot_frame = tk.LabelFrame(parent, text="Plot", font=("Arial", 9), **pad)
        plot_frame.pack(fill="both", expand=True, **pad)

        self._fig = plt.Figure(figsize=(5.5, 4.2), dpi=100)
        self._ax = self._fig.add_subplot(111)
        self._fig_canvas_widget = FigureCanvasTkAgg(self._fig, master=plot_frame)
        self._fig_canvas_widget.get_tk_widget().pack(fill="both", expand=True)

        self._view_locked = False
        self._pan_state.update(active=False, inv=None, p0=None, xlim0=None, ylim0=None)
        self._fig_canvas_widget.mpl_connect("scroll_event", self._on_plot_scroll)
        self._fig_canvas_widget.mpl_connect("button_press_event", self._on_plot_press)
        self._fig_canvas_widget.mpl_connect("motion_notify_event", self._on_plot_drag)
        self._fig_canvas_widget.mpl_connect("button_release_event", self._on_plot_release)

        btn_row = tk.Frame(parent)
        btn_row.pack(fill="x", **pad)
        tk.Button(btn_row, text="Open in matplotlib window", font=("Arial", 9, "bold"),
                  bg="#334477", fg="white", activebackground="#445588", relief=tk.FLAT,
                  padx=10, pady=3, command=self._on_open_popup).pack(side="left")

        code_frame = tk.LabelFrame(parent, text="Equivalent matplotlib code",
                                    font=("Arial", 9), **pad)
        code_frame.pack(fill="both", expand=False, **pad)

        self._code_text_widget = tk.Text(code_frame, height=14, font=("Courier", 8), wrap=tk.NONE)
        code_vsb = ttk.Scrollbar(code_frame, orient="vertical",
                                 command=self._code_text_widget.yview)
        code_hsb = ttk.Scrollbar(code_frame, orient="horizontal",
                                 command=self._code_text_widget.xview)
        self._code_text_widget.configure(yscrollcommand=code_vsb.set,
                                         xscrollcommand=code_hsb.set)
        code_hsb.pack(side="bottom", fill="x")
        code_vsb.pack(side="right", fill="y")
        self._code_text_widget.pack(fill="both", expand=True)

    # ── inspector <-> data sync helpers ──────────────────────────

    def _get_curve_text(self, i: int) -> str:
        cfg = self._curves[i]
        w = cfg.get("text_widget")
        if w is not None and w.winfo_exists():
            # get() works on a disabled Text too; "end-1c" drops Tk's
            # implicit trailing newline so the text does not grow.
            cfg["text_cache"] = w.get("1.0", "end-1c")
        return cfg["text_cache"]

    def _set_curve_text(self, i: int, text: str) -> None:
        cfg = self._curves[i]
        cfg["text_cache"] = text
        w = cfg.get("text_widget")
        if w is not None and w.winfo_exists():
            state_before = str(w.cget("state"))
            w.configure(state="normal")
            w.delete("1.0", tk.END)
            w.insert("1.0", text)
            w.configure(state=state_before)

    def _apply_source_ui(self, i: int) -> None:
        """Enable/disable the x/y expression boxes and the points text box
        according to the curve's data-source radio button."""
        cfg = self._curves[i]
        use_pin = cfg["source_var"].get() == "pin"
        for key in ("x_entry", "y_entry"):
            w = cfg.get(key)
            if w is not None and w.winfo_exists():
                w.configure(state="normal" if use_pin else "disabled")
        tw = cfg.get("text_widget")
        if tw is not None and tw.winfo_exists():
            tw.configure(state="disabled" if use_pin else "normal")

    def _update_curve_entry_ui(self, i: int) -> None:
        cfg = self._curves[i]
        has_pin = self._pin_arrays.get(i) is not None
        pin_lbl = cfg.get("pin_indicator")
        if pin_lbl is not None and pin_lbl.winfo_exists():
            pin_lbl.configure(text="● pin has data" if has_pin else "○ pin empty",
                             fg="#226600" if has_pin else "#aaaaaa")
        pts_lbl = cfg.get("points_label")
        if pts_lbl is not None and pts_lbl.winfo_exists():
            pts_lbl.configure(fg="#333333" if cfg["valid"] else "#cc0000")

    def _on_source_changed(self, i: int) -> None:
        self._apply_source_ui(i)
        self._refresh_local()

    def _on_scale_changed(self) -> None:
        # Limits from a previous scale can be invalid on the new one
        # (e.g. <= 0 on a log axis), so go back to auto-fit.
        self._view_locked = False
        self._redraw()

    def _refresh_local(self) -> None:
        """Re-resolve data from the current inspector settings and the last
        pin values, then redraw. No graph execution is triggered."""
        self._resolve_all_curves()
        self._redraw()

    def _on_apply(self) -> None:
        if self._request_downstream:
            self._request_downstream(self.node_id)
        else:
            self._redraw()

    # ── help popup ────────────────────────────────────────────────

    def _open_help(self) -> None:
        if self._help_popup is not None and self._help_popup.winfo_exists():
            self._help_popup.lift()
            return
        popup = tk.Toplevel()
        popup.title("Scatter Plot 2D - Help")
        popup.geometry("700x600")
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
        tk.Button(body, text="Close", font=("Arial", 9), command=popup.destroy
                 ).pack(anchor="e", pady=(8, 0))
        popup.bind("<Escape>", lambda _e: popup.destroy())
        popup.protocol("WM_DELETE_WINDOW", popup.destroy)
        self._help_popup = popup

    # ── plotted-data resolution (shared by compute / redraw / popup / code) ──

    def _resolve_all_curves(self, inputs: dict | None = None) -> None:
        """Refresh cached pin arrays (if inputs given), then rebuild every
        curve's (n,2) data, points label and validity flag from its data
        source (pin expressions or text box)."""
        if inputs is not None:
            for i in range(1, self.MAX_CURVES + 1):
                self._pin_arrays[i] = _coerce_pin_array(inputs.get(f"curve{i}"))

        for i in range(1, self.MAX_CURVES + 1):
            cfg = self._curves[i]
            arr, msg, valid = self._resolve_curve(i)
            cfg["data"] = arr
            cfg["valid"] = valid
            cfg["points_var"].set(msg)
            self._update_curve_entry_ui(i)

    def _resolve_curve(self, i: int) -> tuple[np.ndarray, str, bool]:
        """Returns (data (n,2) float64, message for the points label, valid)."""
        cfg = self._curves[i]
        empty = np.empty((0, 2), dtype=np.float64)

        if cfg["source_var"].get() == "text":
            parsed, skipped = _parse_point_block(self._get_curve_text(i))
            if parsed is None:
                extra = f" ({skipped} bad line{'s' if skipped != 1 else ''})" if skipped else ""
                return empty, f"Invalid data: no valid points{extra}", False
            good = np.isfinite(parsed).all(axis=1)
            arr = parsed[good]
            if arr.shape[0] == 0:
                return empty, "Invalid data: no finite points", False
            return arr, _points_msg(arr.shape[0], skipped=skipped), True

        try:
            x = _eval_pin_expr(cfg["x_expr_var"].get(), self._pin_arrays)
        except Exception as e:
            return empty, f"Invalid data: x: {_short(e)}", False
        try:
            y = _eval_pin_expr(cfg["y_expr_var"].get(), self._pin_arrays)
        except Exception as e:
            return empty, f"Invalid data: y: {_short(e)}", False

        if x.size == 0 or y.size == 0:
            return empty, "Invalid data: empty", False
        if x.size != y.size:
            return empty, f"Invalid data: x has {x.size} points, y has {y.size}", False

        good = np.isfinite(x) & np.isfinite(y)
        if not good.any():
            return empty, "Invalid data: no finite points", False
        arr = np.column_stack([x[good], y[good]])
        return arr, _points_msg(arr.shape[0], dropped=int((~good).sum())), True

    def _plotted_xy(self, i: int) -> tuple[np.ndarray, np.ndarray] | None:
        data = self._curves[i]["data"]
        if data.shape[0] == 0:
            return None
        return data[:, 0], data[:, 1]

    # ── drawing (shared by embedded canvas + popup window) ───────────

    def _style_axes(self, ax) -> None:
        if self._xscale_var.get() == "log":
            ax.set_xscale("log", nonpositive="mask")
        else:
            ax.set_xscale("linear")
        if self._yscale_var.get() == "log":
            ax.set_yscale("log", nonpositive="mask")
        else:
            ax.set_yscale("linear")
        ax.set_xlabel(self._xlabel_var.get())
        ax.set_ylabel(self._ylabel_var.get())
        title = self._title_var.get().strip()
        if title:
            ax.set_title(title)
        if self._grid_on_var.get():
            ax.grid(True, linestyle=self._grid_style_var.get(),
                   color=self._grid_color_var.get())
        else:
            ax.grid(False)

    def _plot_all_curves(self, ax) -> bool:
        """Returns True if at least one curve was actually drawn."""
        any_drawn = False
        for i in range(1, self.MAX_CURVES + 1):
            cfg = self._curves[i]
            if not cfg["enabled_var"].get():
                continue
            xy = self._plotted_xy(i)
            if xy is None:
                continue
            x, y = xy
            marker = cfg["marker_var"].get()
            line_width = float(cfg["line_width_var"].get())
            line_style = cfg["line_style_var"].get() if line_width > 0 else "None"
            ax.plot(
                x, y,
                marker=marker if marker != "None" else None,
                markersize=float(cfg["marker_size_var"].get()),
                markerfacecolor=cfg["marker_color_var"].get(),
                markeredgecolor=cfg["marker_color_var"].get(),
                linewidth=line_width,
                linestyle=line_style,
                color=cfg["line_color_var"].get(),
                label=cfg["legend_var"].get() or f"Curve {i}",
            )
            any_drawn = True
        if any_drawn and self._legend_on_var.get():
            ax.legend(loc=self._legend_loc_var.get())
        return any_drawn

    # ── mouse wheel zoom / drag pan (embedded plot) ───────────────

    @staticmethod
    def _zoom_limits(axis, lim, center: float, factor: float) -> tuple[float, float]:
        """Scale lim about `center` by `factor`, in the axis' own scale
        (linear zooms additively, log multiplicatively, so limits stay > 0)."""
        tr = axis.get_transform()
        c = float(tr.transform(center))
        lo, hi = (float(v) for v in tr.transform(np.asarray(lim, dtype=float)))
        new = np.array([c - (c - lo) * factor, c - (c - hi) * factor])
        new_lo, new_hi = tr.inverted().transform(new)
        if not (np.isfinite(new_lo) and np.isfinite(new_hi)):
            return tuple(lim)
        return float(new_lo), float(new_hi)

    @staticmethod
    def _shift_limits(axis, lim0, a0: float, a1: float) -> tuple[float, float]:
        """Shift lim0 so data point a1 lands where a0 was, in the axis' own
        scale (linear shifts additively, log multiplicatively)."""
        tr = axis.get_transform()
        shift = float(tr.transform(a1)) - float(tr.transform(a0))
        lo, hi = tr.transform(np.asarray(lim0, dtype=float)) - shift
        new_lo, new_hi = tr.inverted().transform(np.array([lo, hi]))
        if not (np.isfinite(new_lo) and np.isfinite(new_hi)):
            return tuple(lim0)
        return float(new_lo), float(new_hi)

    def _on_plot_scroll(self, event) -> None:
        ax = self._ax
        if ax is None or event.inaxes is not ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        factor = 0.9 if event.button == "up" else 1.1
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        key = getattr(event, "key", None) or ""
        # event.key can be a combination such as "ctrl+shift"
        mods = set(key.split("+"))

        # Ctrl+wheel: zoom x only. Shift+wheel: zoom y only.
        # No modifier: zoom both axes together.
        if mods & {"control", "ctrl"}:
            ax.set_xlim(self._zoom_limits(ax.xaxis, xlim, event.xdata, factor))
        elif "shift" in mods:
            ax.set_ylim(self._zoom_limits(ax.yaxis, ylim, event.ydata, factor))
        else:
            ax.set_xlim(self._zoom_limits(ax.xaxis, xlim, event.xdata, factor))
            ax.set_ylim(self._zoom_limits(ax.yaxis, ylim, event.ydata, factor))

        self._view_locked = True
        self._fig_canvas_widget.draw_idle()

    def _on_plot_press(self, event) -> None:
        ax = self._ax
        if ax is None or event.inaxes is not ax or event.button != 1:
            return
        if getattr(event, "dblclick", False):
            # Double-click: drop the user's view and auto-fit the data again.
            self._view_locked = False
            self._pan_state["active"] = False
            self._redraw()
            return
        # Pan is computed in the data coordinates of the view at press
        # time (fixed inverse transform), so the plot does not jitter as
        # the limits change during the drag.
        inv = ax.transData.inverted()
        self._pan_state.update(
            active=True, inv=inv,
            p0=inv.transform((event.x, event.y)),
            xlim0=ax.get_xlim(), ylim0=ax.get_ylim(),
        )

    def _on_plot_drag(self, event) -> None:
        st = self._pan_state
        ax = self._ax
        if not st["active"] or ax is None or event.x is None or event.y is None:
            return
        p1 = st["inv"].transform((event.x, event.y))
        p0 = st["p0"]
        ax.set_xlim(self._shift_limits(ax.xaxis, st["xlim0"], p0[0], p1[0]))
        ax.set_ylim(self._shift_limits(ax.yaxis, st["ylim0"], p0[1], p1[1]))
        self._view_locked = True
        self._fig_canvas_widget.draw_idle()

    def _on_plot_release(self, _event) -> None:
        self._pan_state["active"] = False

    def _redraw(self) -> None:
        if self._ax is None or self._fig_canvas_widget is None:
            return
        # Keep the user's zoom/pan across redraws (setting changes, new
        # pin data); ax.clear() would otherwise reset it to auto-fit.
        saved_view = ((self._ax.get_xlim(), self._ax.get_ylim())
                      if self._view_locked else None)
        self._ax.clear()
        any_drawn = self._plot_all_curves(self._ax)
        self._style_axes(self._ax)
        if not any_drawn:
            self._ax.text(0.5, 0.5, "No enabled curve has data",
                         ha="center", va="center", transform=self._ax.transAxes,
                         color="#888888")
        if saved_view is not None and any_drawn:
            self._ax.set_xlim(saved_view[0])
            self._ax.set_ylim(saved_view[1])
        self._fig.tight_layout()
        self._fig_canvas_widget.draw_idle()
        self._status_var.set("plotted" if any_drawn else "no data to plot")
        self._update_generated_code()

    def _on_open_popup(self) -> None:
        """Opens an independent matplotlib window, built fresh from
        current settings. Does not reuse the embedded figure/canvas."""
        popup_fig = plt.figure(figsize=(7, 5.5))
        popup_ax = popup_fig.add_subplot(111)
        self._plot_all_curves(popup_ax)
        self._style_axes(popup_ax)
        popup_fig.tight_layout()
        plt.figure(popup_fig.number)
        plt.show(block=False)

    # ── generated matplotlib code ─────────────────────────────────

    def _update_generated_code(self) -> None:
        if self._code_text_widget is None or not self._code_text_widget.winfo_exists():
            return
        code = self._generate_code()
        self._code_text_widget.configure(state="normal")
        self._code_text_widget.delete("1.0", tk.END)
        self._code_text_widget.insert("1.0", code)

    def _generate_code(self) -> str:
        lines = [
            "import matplotlib.pyplot as plt",
            "",
            "fig, ax = plt.subplots(figsize=(7, 5.5))",
            "",
        ]
        for i in range(1, self.MAX_CURVES + 1):
            cfg = self._curves[i]
            if not cfg["enabled_var"].get():
                continue
            xy = self._plotted_xy(i)
            if xy is None:
                continue
            x, y = xy
            arr = np.column_stack([x, y])
            var_name = f"curve{i}"
            if cfg["source_var"].get() == "pin":
                lines.append(f"# curve {i}: x = {cfg['x_expr_var'].get().strip()}"
                             f"   y = {cfg['y_expr_var'].get().strip()}")
            lines.append(f"{var_name} = {_py_repr_array(arr)}")
            lines.append(f"{var_name}_x = [p[0] for p in {var_name}]")
            lines.append(f"{var_name}_y = [p[1] for p in {var_name}]")

            marker = cfg["marker_var"].get()
            marker_arg = "None" if marker == "None" else repr(marker)
            line_width = float(cfg["line_width_var"].get())
            line_style = cfg["line_style_var"].get() if line_width > 0 else "None"

            lines.append(
                f"ax.plot({var_name}_x, {var_name}_y, "
                f"marker={marker_arg}, "
                f"markersize={cfg['marker_size_var'].get():g}, "
                f"markerfacecolor={cfg['marker_color_var'].get()!r}, "
                f"markeredgecolor={cfg['marker_color_var'].get()!r}, "
                f"linewidth={line_width:g}, "
                f"linestyle={line_style!r}, "
                f"color={cfg['line_color_var'].get()!r}, "
                f"label={(cfg['legend_var'].get() or f'Curve {i}')!r})"
            )
            lines.append("")

        if self._xscale_var.get() == "log":
            lines.append("ax.set_xscale('log', nonpositive='mask')")
        if self._yscale_var.get() == "log":
            lines.append("ax.set_yscale('log', nonpositive='mask')")
        lines.append(f"ax.set_xlabel({self._xlabel_var.get()!r})")
        lines.append(f"ax.set_ylabel({self._ylabel_var.get()!r})")
        title = self._title_var.get().strip()
        if title:
            lines.append(f"ax.set_title({title!r})")
        if self._grid_on_var.get():
            lines.append(
                f"ax.grid(True, linestyle={self._grid_style_var.get()!r}, "
                f"color={self._grid_color_var.get()!r})"
            )
        if self._legend_on_var.get():
            lines.append(f"ax.legend(loc={self._legend_loc_var.get()!r})")
        lines.append("")
        lines.append("fig.tight_layout()")
        lines.append("plt.show()")
        return "\n".join(lines)

    # ── compute ───────────────────────────────────────────────────

    def compute(self, inputs: dict) -> dict:
        self._init_state()
        self._resolve_all_curves(inputs)
        if self.is_inspector_open():
            self._redraw()
        else:
            n_enabled = sum(
                1 for i in range(1, self.MAX_CURVES + 1)
                if self._curves[i]["enabled_var"].get() and self._curves[i]["data"].shape[0] > 0)
            self._status_var.set(f"{n_enabled}/{self.MAX_CURVES} curves ready "
                                 f"(open inspector to view plot)")
        return {}

    # ── serialization ─────────────────────────────────────────────

    def get_params(self) -> dict:
        curves_params = []
        for i in range(1, self.MAX_CURVES + 1):
            cfg = self._curves[i]
            self._get_curve_text(i)  # flush widget -> cache if applicable
            curves_params.append({
                "enabled": bool(cfg["enabled_var"].get()),
                "legend": cfg["legend_var"].get(),
                "marker": cfg["marker_var"].get(),
                "marker_size": float(cfg["marker_size_var"].get()),
                "marker_color": cfg["marker_color_var"].get(),
                "line_width": float(cfg["line_width_var"].get()),
                "line_style": cfg["line_style_var"].get(),
                "line_color": cfg["line_color_var"].get(),
                "source": cfg["source_var"].get(),
                "x_expr": cfg["x_expr_var"].get(),
                "y_expr": cfg["y_expr_var"].get(),
                "text": cfg["text_cache"],
            })
        return {
            "xlabel": self._xlabel_var.get(),
            "ylabel": self._ylabel_var.get(),
            "title": self._title_var.get(),
            "grid_on": bool(self._grid_on_var.get()),
            "grid_style": self._grid_style_var.get(),
            "grid_color": self._grid_color_var.get(),
            "legend_on": bool(self._legend_on_var.get()),
            "legend_loc": self._legend_loc_var.get(),
            "xscale": self._xscale_var.get(),
            "yscale": self._yscale_var.get(),
            "curves": curves_params,
        }

    def set_params(self, params: dict) -> None:
        self._init_state()
        self._xlabel_var.set(str(params.get("xlabel", "x")))
        self._ylabel_var.set(str(params.get("ylabel", "y")))
        self._title_var.set(str(params.get("title", "")))
        self._grid_on_var.set(bool(params.get("grid_on", True)))
        self._grid_style_var.set(str(params.get("grid_style", "--")))
        self._grid_color_var.set(str(params.get("grid_color", "gray")))
        self._legend_on_var.set(bool(params.get("legend_on", True)))
        self._legend_loc_var.set(str(params.get("legend_loc", "best")))
        xs = str(params.get("xscale", "linear"))
        ys = str(params.get("yscale", "linear"))
        self._xscale_var.set(xs if xs in ("linear", "log") else "linear")
        self._yscale_var.set(ys if ys in ("linear", "log") else "linear")

        saved_curves = params.get("curves", [])
        for i in range(1, self.MAX_CURVES + 1):
            cfg = self._curves[i]
            saved = saved_curves[i - 1] if i - 1 < len(saved_curves) else {}
            cfg["enabled_var"].set(bool(saved.get("enabled", i == 1)))
            cfg["legend_var"].set(str(saved.get("legend", f"Curve {i}")))
            cfg["marker_var"].set(str(saved.get("marker", _DEFAULT_CURVE_MARKERS[i - 1])))
            cfg["marker_size_var"].set(float(saved.get("marker_size", 6.0)))
            cfg["marker_color_var"].set(str(saved.get("marker_color", _DEFAULT_CURVE_COLORS[i - 1])))
            cfg["line_width_var"].set(float(saved.get("line_width", 1.5)))
            cfg["line_style_var"].set(str(saved.get("line_style", "-")))
            cfg["line_color_var"].set(str(saved.get("line_color", _DEFAULT_CURVE_COLORS[i - 1])))
            src = str(saved.get("source", "pin"))
            cfg["source_var"].set(src if src in ("pin", "text") else "pin")
            cfg["x_expr_var"].set(str(saved.get("x_expr", f"c{i}[:,0]")))
            if "y_expr" in saved:
                cfg["y_expr_var"].set(str(saved["y_expr"]))
            elif saved.get("mode") == "rel":
                # Older files had an "Rel." mode (y - y[0]); keep that result.
                cfg["y_expr_var"].set(f"c{i}[:,1]-c{i}[0,1]")
            else:
                cfg["y_expr_var"].set(f"c{i}[:,1]")
            self._set_curve_text(i, str(saved.get("text", _DEFAULT_POINTS_TEXT)))
            self._apply_source_ui(i)

    def close_inspector(self) -> None:
        for i in range(1, self.MAX_CURVES + 1):
            self._get_curve_text(i)  # flush before losing the widget
            for key in ("text_widget", "x_entry", "y_entry",
                        "pin_indicator", "points_label"):
                self._curves[i][key] = None
        if self._fig is not None:
            plt.close(self._fig)
        self._fig = None
        self._ax = None
        self._fig_canvas_widget = None
        self._code_text_widget = None
        super().close_inspector()

    def on_destroy(self) -> None:
        if self._help_popup is not None and self._help_popup.winfo_exists():
            self._help_popup.destroy()
        if self._fig is not None:
            plt.close(self._fig)
        super().on_destroy()