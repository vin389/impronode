# node_editor/nodes/mesh_scatter_viewer_node.py

import math
import tkinter as tk
from tkinter import ttk

import numpy as np

from node_editor.base_node import BaseNode
from node_editor.pin_types import PinSchema, PinDef, PinType
from node_editor.execution import ExecutionMode


# ---------------------------------------------------------------------------
# Help text
# ---------------------------------------------------------------------------

_HELP_TEXT = """\
Mesh & Scatter Viewer
======================

PURPOSE
-------
Extends the Meshgrid concept with support for overlaying up to 4
independent sets of arbitrary (unstructured) 3D points -- commonly
called "scatter series" -- alongside an optional structured grid.
Typical use: compare "before deformation" vs "after deformation"
point clouds on the same 3D plot, optionally with a reference grid.

The grid is now OPTIONAL: if x or y is not provided, no grid is
generated and only the enabled point sets are shown.

Each point set can ALSO optionally be drawn with its own 3D grid
lines, by telling this node the (nx, ny, nz) dimensions the point
set is secretly a flattened grid of -- e.g. a tracked-point set of
800 points that is really a 20x10x4 grid can be given "20 10 4" to
draw the connecting grid lines between neighbouring tracked points,
which is very useful for visually inspecting deformation patterns.

INPUT PINS
----------
x, y, z   ARRAY (optional)
     Same as the Meshgrid node. Grid is only generated when BOTH x
     and y resolve to non-empty arrays. z empty/absent -> 2D grid.
     Pin input overrides the inspector text box, same as before.

points1 .. points4   ARRAY (optional)
     Each is one point set, shape (N,3) or (N,2) (z padded with 0),
     or a flat array whose length is a multiple of 3 (or 2).
     Non-finite rows (NaN/Inf) are dropped automatically.
     Pin input overrides that set's inspector text box.

OUTPUT PINS
-----------
grid_coords   ARRAY   Same as Meshgrid; empty (0,3) if no grid.
grid_shape    ARRAY   Same as Meshgrid; empty (0,) if no grid.
points_all    ARRAY   (M,3) -- every ENABLED point set with data,
                        concatenated in set order (1,2,3,4).
points_set_id ARRAY   (M,) float64 -- which set (1-based) each row
                        of points_all came from. Useful downstream
                        to re-split or color-code points_all.

INSPECTOR -- "Grid" TAB
------------------------
Same X/Y/Z fields, axis order, and Compute button as Meshgrid.
Leave X or Y empty (and disconnected) to skip the grid entirely.

INSPECTOR -- "Point Sets" TAB
------------------------------
Each of the 4 sets has:
  Enabled     checkbox -- include this set in the preview/outputs.
              Only Set 1 is enabled by default.
  Label       a free-text note to help you remember what this set
              represents (e.g. "reference", "deformed frame 42").
  points_N    text box -- one point per line: "x y z" or "x,y,z"
              (comma/tab/space separated). "x y" (2 values) is
              padded with z=0. Lines starting with '#' are ignored.
              Malformed lines are skipped individually rather than
              aborting the whole parse.
  Color / Marker / Size -- this set's scatter-point rendering
              style. Changing these updates the preview
              IMMEDIATELY (no need to click Apply).
  Draw 3D grid lines  checkbox -- when checked, and the point count
              exactly equals nx*ny*nz below, this set's points are
              reshaped to (nx,ny,nz,3) -- assumed stored in the same
              "x slowest ... z fastest" order the Grid tab's own
              output uses -- and connecting grid lines are drawn
              between neighbouring points along each of the three
              axes, exactly like the structured Grid tab's grid
              lines, but using this set's own points and its own
              independent line color/width below.
  Grid dims (nx ny nz) -- three integers whose product must equal
              this set's point count exactly, or the grid-line
              overlay is skipped (with a message in the status line)
              while the scatter markers still draw normally.
  Grid line color / width -- this set's grid-line style, separate
              from every other set's and from the main Grid tab's.
A green "●" / gray "○" indicator shows whether each set is
currently pin-driven; the text box is read-only while pin-driven
and shows a preview of the incoming data.
Text box edits require clicking "Compute" (Grid tab) or "Apply"
(this tab) to take effect -- exactly like the X/Y/Z fields.

INSPECTOR -- "Visualization" (middle pane)
-------------------------------------------
Same grid-style controls as Meshgrid (show points/lines per axis,
point size/color/marker, line width/color/style). This governs
ONLY the structured grid from the Grid tab, not the point sets.

INSPECTOR -- "3D preview" (right pane)
-----------------------------------------
Same interactive orbit viewer as Meshgrid:
  Left-drag    rotate (orbits around the current pan target)
  Right-drag   pan
  Mouse wheel  zoom
"Enable 3D preview" can be left off for very large point counts,
since rendering thousands of points (and grid lines) as Tk canvas
items is slow.

KEYBOARD SHORTCUT
-----------------
Ctrl-H / Ctrl-h -- show this help window.
"""

_AXIS_LABELS = {
    "ij": "(nx, ny, nz)  ('ij')",
    "xy": "(ny, nx, nz)  ('xy' / Matlab style)",
    "xzy": "(nx, nz, ny)",
    "zxy": "(nz, nx, ny)",
    "yzx": "(ny, nz, nx)",
    "zyx": "(nz, ny, nx)",
}

_AXIS_ORDER = {
    "ij": ("x", "y", "z"),
    "xy": ("y", "x", "z"),
    "xzy": ("x", "z", "y"),
    "zxy": ("z", "x", "y"),
    "yzx": ("y", "z", "x"),
    "zyx": ("z", "y", "x"),
}

_COMMON_COLORS = [
    "red", "green", "blue", "black", "yellow",
    "orange", "purple", "white", "gray", "brown", "pink",
]

# All marker shapes are actually implemented by _draw_marker() below --
# fixing a gap in the original Meshgrid node, where the combobox listed
# "s", "D", "^", "v", "<", ">" but only "o"/"+"/"*"/"x" were drawn.
_MARKERS = ["o", "+", "*", "x", "s", "D", "^", "v", "<", ">"]

_MIN_GRID_DIM = 1


def _normalize_indexing(value: str | None) -> str:
    if value is None:
        return "ij"
    value = str(value).strip().lower()
    mapping = {
        "ij": "ij", "xy": "xy", "xyz": "ij", "yxz": "xy",
        "xzy": "xzy", "zxy": "zxy", "yzx": "yzx", "zyx": "zyx",
    }
    return mapping.get(value, "ij")


# ---------------------------------------------------------------------------
# Sequence parser (grid axes) -- unchanged from Meshgrid
# ---------------------------------------------------------------------------

def _parse_sequence(text: str) -> np.ndarray | None:
    """
    Parse a 1D array from a text string.

    Supported formats:
      "v0 v1 v2 ..."         space/comma separated floats
      "start:step:end"       MATLAB-style, end is inclusive
      "start:end"            step defaults to 1.0, end inclusive
    """
    text = text.strip()
    if not text:
        return None

    if ":" in text:
        parts = text.split(":")
        try:
            if len(parts) == 2:
                start = float(parts[0]); end = float(parts[1]); step = 1.0
            elif len(parts) == 3:
                start = float(parts[0]); step = float(parts[1]); end = float(parts[2])
            else:
                return None
            if step == 0:
                return None
            epsilon = abs(step) * 1e-9
            arr = np.arange(start, end + np.sign(step) * epsilon, step, dtype=np.float64)
            return arr if arr.size > 0 else None
        except ValueError:
            return None

    try:
        nums = [float(t) for t in text.replace(",", " ").split() if t.strip()]
        if not nums:
            return None
        return np.array(nums, dtype=np.float64)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Point-set parser and coercion
# ---------------------------------------------------------------------------

def _parse_point_block(text: str) -> tuple[np.ndarray | None, int]:
    """
    Parse an N-by-3 array from pasted text: one point per line,
    values separated by spaces, tabs, or commas. Lines that are
    blank or start with '#' are ignored. Lines with exactly 2
    values are padded with z=0. Malformed lines are skipped
    individually (not treated as a fatal parse error).

    Returns (array_or_None, skipped_line_count).
    """
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
        if len(vals) == 2:
            vals.append(0.0)
        if len(vals) < 3:
            skipped += 1
            continue
        rows.append(vals[:3])
    if not rows:
        return None, skipped
    return np.array(rows, dtype=np.float64), skipped


def _coerce_point_array(arr: np.ndarray, label: str) -> tuple[np.ndarray, int]:
    """
    Coerce an array to (M,3) float64, padding (M,2) with z=0 and
    reshaping flat arrays whose length is a multiple of 2 or 3.
    Drops non-finite rows. Returns (array, dropped_row_count).
    Raises ValueError if the shape cannot be interpreted at all.
    """
    a = np.asarray(arr, dtype=np.float64)

    if a.ndim == 3 and a.shape[1:] == (1, 3):
        a = a.reshape(-1, 3)
    elif a.ndim == 1:
        if a.size > 0 and a.size % 3 == 0:
            a = a.reshape(-1, 3)
        elif a.size > 0 and a.size % 2 == 0:
            a = a.reshape(-1, 2)
        else:
            raise ValueError(
                f"{label}: flat array of size {a.size} is not a "
                f"multiple of 2 or 3")

    if a.ndim != 2 or a.shape[1] not in (2, 3):
        raise ValueError(f"{label}: expected shape (N,2) or (N,3), got {a.shape}")

    if a.shape[1] == 2:
        a = np.column_stack([a, np.zeros(a.shape[0], dtype=np.float64)])

    finite_mask = np.isfinite(a).all(axis=1)
    dropped = int((~finite_mask).sum())
    return a[finite_mask], dropped


def _parse_grid_dims(text: str) -> tuple[int, int, int] | None:
    """
    Parse "nx ny nz" (space/comma/tab separated) into a tuple of 3
    positive integers. Returns None if the text cannot be parsed as
    exactly 3 positive integers.
    """
    parts = text.replace(",", " ").split()
    if len(parts) != 3:
        return None
    try:
        vals = [int(float(p)) for p in parts]
    except ValueError:
        return None
    if any(v < _MIN_GRID_DIM for v in vals):
        return None
    return vals[0], vals[1], vals[2]


# ---------------------------------------------------------------------------
# MeshScatterViewerNode
# ---------------------------------------------------------------------------

class MeshScatterViewerNode(BaseNode):
    """
    Extends MeshgridNode: an optional structured grid, plus up to
    4 independent unstructured "point sets" (scatter series), each
    with its own color/marker/size, for visual comparison (e.g.
    before/after deformation) on a shared interactive 3D preview.

    Each point set can also optionally be re-interpreted as a
    structured (nx,ny,nz) grid purely for drawing connecting grid
    lines between neighbouring points, independent of the main
    Grid tab's own structured grid.

    See _HELP_TEXT (Ctrl-H) for the full pin/inspector reference.
    """

    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE      = "mesh_scatter_viewer"
    DISPLAY_NAME   = "Mesh & Scatter Viewer"
    CATEGORY       = "process"
    SEARCH_KEYWORDS = ("meshgrid", "scatter", "point set", "point cloud",
                       "3d", "viewer", "compare", "before after", "grid lines")
    NODE_WIDTH     = 210
    NODE_HEIGHT    = 140

    HELP_TEXT = _HELP_TEXT

    _BODY_BG   = "#fffde7"
    _OUTLINE   = "#ccaa00"
    _TITLE_FG  = "#554400"
    _STATUS_FG = "#776600"
    _PIN_COLOR = "#334400"

    MAX_POINT_SETS = 4
    _DEFAULT_SET_COLORS       = ["red", "blue", "orange", "purple"]
    _DEFAULT_SET_MARKERS      = ["o", "s", "^", "D"]
    _DEFAULT_SET_LINE_COLORS  = ["#aa4444", "#4444aa", "#aa8800", "#7a4499"]

    def get_pin_schema(self) -> PinSchema:
        inputs = [
            PinDef("x", PinType.ARRAY, "x", optional=True),
            PinDef("y", PinType.ARRAY, "y", optional=True),
            PinDef("z", PinType.ARRAY, "z", optional=True),
        ]
        for i in range(1, self.MAX_POINT_SETS + 1):
            inputs.append(
                PinDef(f"points{i}", PinType.ARRAY, f"pts{i}", optional=True))

        return PinSchema(
            inputs=inputs,
            outputs=[
                PinDef("grid_coords",   PinType.ARRAY, "coords"),
                PinDef("grid_shape",    PinType.ARRAY, "shape"),
                PinDef("points_all",    PinType.ARRAY, "ptsAll"),
                PinDef("points_set_id", PinType.ARRAY, "setId"),
            ]
        )

    def get_help_text(self) -> str:
        return _HELP_TEXT

    # ── state init ────────────────────────────────────────────────

    def _init_state(self) -> None:
        if hasattr(self, "_x_var"):
            return

        # grid state (same defaults/shape as Meshgrid)
        self._x_var = tk.StringVar(value="-8:2:8")
        self._y_var = tk.StringVar(value="-6:2:6")
        self._z_var = tk.StringVar(value="")
        self._indexing_var = tk.StringVar(value="ij")

        self._point_size_var  = tk.DoubleVar(value=2.0)
        self._point_color_var = tk.StringVar(value="blue")
        self._line_width_var  = tk.DoubleVar(value=1.0)
        self._line_color_var  = tk.StringVar(value="black")
        self._marker_var      = tk.StringVar(value="o")
        self._line_style_var  = tk.StringVar(value="solid")
        self._show_points_var   = tk.BooleanVar(value=True)
        self._show_lines_var    = tk.BooleanVar(value=True)
        self._show_x_lines_var  = tk.BooleanVar(value=True)
        self._show_y_lines_var  = tk.BooleanVar(value=True)
        self._show_z_lines_var  = tk.BooleanVar(value=True)
        self._preview_enabled_var = tk.BooleanVar(value=False)

        self._status_var = tk.StringVar(value="not computed")
        self._shape_var  = tk.StringVar(value="no grid")
        self._sets_summary_var = tk.StringVar(value="sets: 0/4 enabled")

        self._x_entry: tk.Entry | None = None
        self._y_entry: tk.Entry | None = None
        self._z_entry: tk.Entry | None = None
        self._preview_canvas: tk.Canvas | None = None
        self._help_popup: tk.Toplevel | None = None

        self._last_grid_coords = np.empty((0, 3), dtype=np.float64)
        self._last_grid_shape  = np.empty((0,), dtype=np.int64)

        self._x_from_pin = False
        self._y_from_pin = False
        self._z_from_pin = False

        # 3D preview camera state. Rotation always pivots around
        # camera_target; panning moves camera_target in world space.
        self._preview_state = {
            "yaw": -0.9,
            "pitch": 0.7,
            "zoom": 1.0,
            "camera_target": [0.0, 0.0, 0.0],
            "dragging": False,
            "last_x": 0,
            "last_y": 0,
            "button": None,
        }

        # ── point sets ───────────────────────────────────────────
        self._point_sets: dict[int, dict] = {}
        for i in range(1, self.MAX_POINT_SETS + 1):
            self._point_sets[i] = {
                "enabled_var":     tk.BooleanVar(value=(i == 1)),
                "label_var":       tk.StringVar(value=f"Set {i}"),
                "color_var":       tk.StringVar(value=self._DEFAULT_SET_COLORS[i - 1]),
                "marker_var":      tk.StringVar(value=self._DEFAULT_SET_MARKERS[i - 1]),
                "size_var":        tk.DoubleVar(value=4.0),
                "status_var":      tk.StringVar(value="no data"),
                # 3D grid-lines-for-this-point-set feature
                "grid_lines_var":  tk.BooleanVar(value=False),
                "grid_dims_var":   tk.StringVar(value=""),
                "grid_line_color_var": tk.StringVar(value=self._DEFAULT_SET_LINE_COLORS[i - 1]),
                "grid_line_width_var": tk.DoubleVar(value=1.0),
                "grid_status_var": tk.StringVar(value=""),
                "text_cache":  "",
                "text_widget": None,
                "pin_indicator": None,
                "data":     np.empty((0, 3), dtype=np.float64),
                "from_pin": False,
            }

    # ── build_body ────────────────────────────────────────────────

    def build_body(self) -> None:
        self._init_state()
        x, y, w, h = self.x, self.y, self.width, self.height

        self._body_rect = self.canvas.create_rectangle(
            x, y, x + w, y + h,
            fill=self._BODY_BG, outline=self._OUTLINE, width=2,
            tags=(self.node_id, "node_body"))
        self._title_item = self.canvas.create_text(
            x + w / 2, y + 13, text=self.DISPLAY_NAME,
            font=("Arial", 9, "bold"), fill=self._TITLE_FG,
            tags=(self.node_id,))

        shape_lbl = tk.Label(
            self.canvas, textvariable=self._shape_var,
            font=("Arial", 8, "bold"), bg=self._BODY_BG, fg=self._PIN_COLOR,
            wraplength=w - 12, justify="center")
        self.canvas.create_window(
            x + w / 2, y + h * 0.42, window=shape_lbl, tags=(self.node_id,))

        sets_lbl = tk.Label(
            self.canvas, textvariable=self._sets_summary_var,
            font=("Arial", 8), bg=self._BODY_BG, fg=self._PIN_COLOR,
            wraplength=w - 12, justify="center")
        self.canvas.create_window(
            x + w / 2, y + h * 0.62, window=sets_lbl, tags=(self.node_id,))

        status_lbl = tk.Label(
            self.canvas, textvariable=self._status_var,
            font=("Arial", 7), bg=self._BODY_BG, fg=self._STATUS_FG,
            wraplength=w - 12, justify="center")
        self.canvas.create_window(
            x + w / 2, y + h - 12, window=status_lbl, tags=(self.node_id,))

        self._canvas_items += [self._body_rect, self._title_item]
        self.canvas.tag_bind(self.node_id, "<Double-Button-1>", self._on_double_click)

    def _on_double_click(self, _event=None) -> str | None:
        self.open_inspector()
        return "break"

    # ── build_inspector ───────────────────────────────────────────

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

        left   = tk.Frame(panes, width=420, bg="#f7f7f7")
        middle = tk.Frame(panes, width=260, bg="#f2f2f2")
        right  = tk.Frame(panes, width=420, bg="#ffffff")
        for f in (left, middle, right):
            f.pack_propagate(False)
        panes.add(left, stretch="always")
        panes.add(middle, stretch="always")
        panes.add(right, stretch="always")

        left_nb = ttk.Notebook(left)
        left_nb.pack(fill="both", expand=True)
        grid_tab = tk.Frame(left_nb)
        sets_tab = tk.Frame(left_nb)
        left_nb.add(grid_tab, text="Grid")
        left_nb.add(sets_tab, text="Point Sets")

        self._build_grid_tab(grid_tab, pad)
        self._build_point_sets_tab(sets_tab, pad)
        self._build_visualization_pane(middle, pad)
        self._build_preview_pane(right, pad)

        if win is not None:
            win.update_idletasks()
            min_w = max(1060, win.winfo_reqwidth())
            min_h = max(660, win.winfo_reqheight())
            win.minsize(min_w, min_h)
            win.geometry(f"{min_w}x{min_h}")

        self._refresh_preview()

    def _make_scrollable(self, parent: tk.Frame) -> tk.Frame:
        """Wrap tall content in a scrollable canvas, return the inner frame."""
        outer = tk.Frame(parent)
        outer.pack(fill="both", expand=True)
        canvas = tk.Canvas(outer, highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas)
        win_id = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfigure(win_id, width=e.width))
        return inner

    def _build_grid_tab(self, parent: tk.Frame, pad: dict) -> None:
        inner = self._make_scrollable(parent)

        def _coord_row(frame, label, var, from_pin, hint=""):
            row = tk.Frame(frame)
            row.pack(fill="x", pady=2)
            tk.Label(row, text="●" if from_pin else "○", font=("Arial", 9),
                     fg="#226600" if from_pin else "#aaaaaa", width=2).pack(side="left")
            tk.Label(row, text=label, font=("Arial", 9, "bold"), width=3,
                     anchor="w").pack(side="left")
            ent = tk.Entry(row, textvariable=var, font=("Courier", 9),
                           state="disabled" if from_pin else "normal",
                           disabledforeground="#888888", disabledbackground="#eeeeee")
            ent.pack(side="left", fill="x", expand=True, padx=(4, 0))
            if hint:
                tk.Label(row, text=hint, font=("Arial", 7), fg="#888888").pack(side="right")
            return ent

        coord_frame = tk.LabelFrame(
            inner, text="Coordinate arrays (pin overrides text; optional)",
            font=("Arial", 9), **pad)
        coord_frame.pack(fill="x", **pad)
        tk.Label(
            coord_frame,
            text='Syntax: "v0 v1 v2 ..." or "start:step:end" or "start:end". '
                 'Leave X or Y empty (and disconnected) to skip the grid.',
            font=("Arial", 8), fg="#555555", anchor="w", justify="left",
            wraplength=360).pack(fill="x", pady=(0, 4))
        self._x_entry = _coord_row(coord_frame, "x:", self._x_var, self._x_from_pin, "(grid)")
        self._y_entry = _coord_row(coord_frame, "y:", self._y_var, self._y_from_pin, "(grid)")
        self._z_entry = _coord_row(coord_frame, "z:", self._z_var, self._z_from_pin, "(empty → 2D)")

        idx_frame = tk.LabelFrame(inner, text="Axis order", font=("Arial", 9), **pad)
        idx_frame.pack(fill="x", **pad)
        for mode, label in _AXIS_LABELS.items():
            tk.Radiobutton(idx_frame, text=label, variable=self._indexing_var,
                          value=mode, font=("Arial", 9), anchor="w").pack(anchor="w", pady=1)

        out_frame = tk.LabelFrame(inner, text="Output", font=("Arial", 9), **pad)
        out_frame.pack(fill="x", **pad)
        tk.Label(out_frame, textvariable=self._shape_var, font=("Arial", 9, "bold"),
                 fg="#224400", anchor="w").pack(fill="x")
        tk.Label(out_frame, textvariable=self._sets_summary_var, font=("Arial", 9),
                 fg="#224400", anchor="w").pack(fill="x")
        tk.Label(out_frame, textvariable=self._status_var, font=("Arial", 9),
                 fg="#444444", anchor="w").pack(fill="x")

        btn_row = tk.Frame(inner)
        btn_row.pack(fill="x", **pad)
        tk.Button(btn_row, text="Compute", font=("Arial", 9, "bold"), bg="#446622",
                  fg="white", activebackground="#557733", relief=tk.FLAT, padx=10,
                  pady=3, command=self._on_apply_btn).pack(side="left")
        tk.Button(btn_row, text="Help (Ctrl-H)", font=("Arial", 8),
                  command=self._open_help).pack(side="right")

    def _build_point_sets_tab(self, parent: tk.Frame, pad: dict) -> None:
        inner = self._make_scrollable(parent)

        for i in range(1, self.MAX_POINT_SETS + 1):
            cfg = self._point_sets[i]
            set_frame = tk.LabelFrame(inner, text=f"Point Set {i}", font=("Arial", 9), **pad)
            set_frame.pack(fill="x", pady=(0, 8))

            top_row = tk.Frame(set_frame)
            top_row.pack(fill="x")
            tk.Checkbutton(top_row, text="Enabled", variable=cfg["enabled_var"],
                          font=("Arial", 9), command=self._refresh_preview).pack(side="left")
            tk.Label(top_row, text="Label:", font=("Arial", 9)).pack(side="left", padx=(12, 2))
            tk.Entry(top_row, textvariable=cfg["label_var"], width=18,
                     font=("Arial", 9)).pack(side="left")

            pin_row = tk.Frame(set_frame)
            pin_row.pack(fill="x", pady=(4, 0))
            pin_ind = tk.Label(pin_row, text="●" if cfg["from_pin"] else "○",
                              font=("Arial", 9),
                              fg="#226600" if cfg["from_pin"] else "#aaaaaa", width=2)
            pin_ind.pack(side="left")
            cfg["pin_indicator"] = pin_ind
            tk.Label(pin_row, text=f"points{i}:  one point per line  \"x y z\"  or  \"x,y,z\"",
                    font=("Arial", 8), fg="#555555").pack(side="left")

            text_widget = tk.Text(set_frame, height=4, font=("Courier", 8), wrap=tk.NONE)
            text_widget.pack(fill="x")
            text_widget.insert("1.0", cfg["text_cache"])
            if cfg["from_pin"]:
                text_widget.configure(state="disabled")
            cfg["text_widget"] = text_widget

            style_row = tk.Frame(set_frame)
            style_row.pack(fill="x", pady=(4, 0))
            tk.Label(style_row, text="Color:", font=("Arial", 8)).pack(side="left")
            color_combo = ttk.Combobox(style_row, textvariable=cfg["color_var"],
                                       values=_COMMON_COLORS, width=8, state="readonly")
            color_combo.pack(side="left", padx=(2, 8))
            color_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_preview())

            tk.Label(style_row, text="Marker:", font=("Arial", 8)).pack(side="left")
            marker_combo = ttk.Combobox(style_row, textvariable=cfg["marker_var"],
                                        values=_MARKERS, width=4, state="readonly")
            marker_combo.pack(side="left", padx=(2, 8))
            marker_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_preview())

            tk.Label(style_row, text="Size:", font=("Arial", 8)).pack(side="left")
            tk.Spinbox(style_row, from_=1, to=30, increment=0.5, textvariable=cfg["size_var"],
                      width=5, font=("Arial", 8), command=self._refresh_preview
                      ).pack(side="left")

            # ── 3D grid lines for this point set ─────────────────────
            grid_row = tk.Frame(set_frame)
            grid_row.pack(fill="x", pady=(6, 0))
            tk.Checkbutton(
                grid_row, text="Draw 3D grid lines, dims (nx ny nz):",
                variable=cfg["grid_lines_var"], font=("Arial", 8),
                command=lambda i=i: self._on_grid_lines_toggle(i)
            ).pack(side="left")
            dims_entry = tk.Entry(grid_row, textvariable=cfg["grid_dims_var"],
                                  width=10, font=("Courier", 8))
            dims_entry.pack(side="left", padx=(4, 0))
            dims_entry.bind("<FocusOut>", lambda e, i=i: self._on_apply_btn())
            dims_entry.bind("<Return>", lambda e, i=i: self._on_apply_btn())
            tk.Label(grid_row, text='e.g. "20 10 4" for 800 pts',
                    font=("Arial", 7), fg="#888888").pack(side="left", padx=(6, 0))

            grid_style_row = tk.Frame(set_frame)
            grid_style_row.pack(fill="x", pady=(2, 0))
            tk.Label(grid_style_row, text="Grid line color:", font=("Arial", 8)).pack(side="left")
            gl_color_combo = ttk.Combobox(grid_style_row, textvariable=cfg["grid_line_color_var"],
                                          values=_COMMON_COLORS + [self._DEFAULT_SET_LINE_COLORS[i - 1]],
                                          width=10, state="normal")
            gl_color_combo.pack(side="left", padx=(2, 8))
            gl_color_combo.bind("<<ComboboxSelected>>", lambda e: self._refresh_preview())
            gl_color_combo.bind("<FocusOut>", lambda e: self._refresh_preview())
            gl_color_combo.bind("<Return>", lambda e: self._refresh_preview())

            tk.Label(grid_style_row, text="width:", font=("Arial", 8)).pack(side="left")
            tk.Spinbox(grid_style_row, from_=0.5, to=8.0, increment=0.5,
                      textvariable=cfg["grid_line_width_var"], width=5, font=("Arial", 8),
                      command=self._refresh_preview).pack(side="left")

            tk.Label(set_frame, textvariable=cfg["grid_status_var"], font=("Arial", 7),
                    fg="#996600", anchor="w").pack(fill="x", pady=(2, 0))
            tk.Label(set_frame, textvariable=cfg["status_var"], font=("Arial", 8),
                    fg="#666666", anchor="w").pack(fill="x", pady=(2, 0))

        tk.Button(inner, text="Apply (parse text boxes / grid dims)", font=("Arial", 9, "bold"),
                  bg="#446622", fg="white", activebackground="#557733", relief=tk.FLAT,
                  padx=10, pady=3, command=self._on_apply_btn).pack(anchor="w", pady=(4, 8))

    def _on_grid_lines_toggle(self, _i: int) -> None:
        # Checkbox toggle needs a recompute (to validate dims against the
        # current point count and populate grid_status_var), not just a
        # cheap preview refresh.
        self._on_apply_btn()

    def _build_visualization_pane(self, parent: tk.Frame, pad: dict) -> None:
        vis_frame = tk.LabelFrame(parent, text="Grid Visualization", font=("Arial", 9), **pad)
        vis_frame.pack(fill="both", expand=True, **pad)

        tk.Checkbutton(vis_frame, text="Show grid points", variable=self._show_points_var,
                      command=self._refresh_preview).pack(anchor="w")
        tk.Checkbutton(vis_frame, text="Show grid lines", variable=self._show_lines_var,
                      command=self._refresh_preview).pack(anchor="w")
        tk.Checkbutton(vis_frame, text="x lines", variable=self._show_x_lines_var,
                      command=self._refresh_preview).pack(anchor="w")
        tk.Checkbutton(vis_frame, text="y lines", variable=self._show_y_lines_var,
                      command=self._refresh_preview).pack(anchor="w")
        tk.Checkbutton(vis_frame, text="z lines", variable=self._show_z_lines_var,
                      command=self._refresh_preview).pack(anchor="w")

        ttk.Label(vis_frame, text="Grid point size:").pack(anchor="w", padx=6, pady=(8, 0))
        ttk.Spinbox(vis_frame, from_=0.5, to=10.0, increment=0.5,
                    textvariable=self._point_size_var, width=8).pack(anchor="w", padx=6)
        ttk.Label(vis_frame, text="Grid point color:").pack(anchor="w", padx=6, pady=(8, 0))
        ttk.Combobox(vis_frame, textvariable=self._point_color_var, values=_COMMON_COLORS,
                    state="readonly", width=12).pack(anchor="w", padx=6)
        ttk.Label(vis_frame, text="Grid marker:").pack(anchor="w", padx=6, pady=(8, 0))
        ttk.Combobox(vis_frame, textvariable=self._marker_var, values=_MARKERS,
                    state="readonly", width=10).pack(anchor="w", padx=6)
        ttk.Label(vis_frame, text="Grid line width:").pack(anchor="w", padx=6, pady=(8, 0))
        ttk.Spinbox(vis_frame, from_=0.5, to=6.0, increment=0.5,
                    textvariable=self._line_width_var, width=8).pack(anchor="w", padx=6)
        ttk.Label(vis_frame, text="Grid line color:").pack(anchor="w", padx=6, pady=(8, 0))
        ttk.Combobox(vis_frame, textvariable=self._line_color_var, values=_COMMON_COLORS,
                    state="readonly", width=12).pack(anchor="w", padx=6)
        ttk.Label(vis_frame, text="Grid line style:").pack(anchor="w", padx=6, pady=(8, 0))
        ttk.Combobox(vis_frame, textvariable=self._line_style_var,
                    values=["solid", "dash", "dot", "dashdot"],
                    state="readonly", width=10).pack(anchor="w", padx=6)

        for var in (self._point_size_var, self._point_color_var, self._marker_var,
                    self._line_width_var, self._line_color_var, self._line_style_var):
            var.trace_add("write", lambda *_a: self._refresh_preview())

    def _build_preview_pane(self, parent: tk.Frame, pad: dict) -> None:
        preview_frame = tk.LabelFrame(parent, text="3D preview", font=("Arial", 9), **pad)
        preview_frame.pack(fill="both", expand=True, **pad)
        tk.Checkbutton(preview_frame, text="Enable 3D preview (slow with many points)",
                      variable=self._preview_enabled_var,
                      command=self._refresh_preview).pack(anchor="w")

        self._preview_canvas = tk.Canvas(preview_frame, bg="#ffffff", highlightthickness=0)
        self._preview_canvas.pack(fill="both", expand=True)
        self._preview_canvas.bind("<ButtonPress-1>", self._on_preview_drag_start)
        self._preview_canvas.bind("<B1-Motion>", self._on_preview_drag_motion)
        self._preview_canvas.bind("<ButtonRelease-1>", self._on_preview_drag_release)
        self._preview_canvas.bind("<ButtonPress-3>", self._on_preview_pan_start)
        self._preview_canvas.bind("<B3-Motion>", self._on_preview_pan_motion)
        self._preview_canvas.bind("<ButtonRelease-3>", self._on_preview_pan_release)
        self._preview_canvas.bind("<MouseWheel>", self._on_preview_mousewheel)
        self._preview_canvas.bind("<Configure>", lambda _e: self._refresh_preview())

    # ── inspector helpers: point-set text sync ──────────────────────

    def _get_point_set_text(self, i: int) -> str:
        """
        Read the text box into text_cache -- but ONLY when this set is
        not currently pin-driven, since a pin-driven text box shows a
        synthetic preview of the incoming data, not user-authored text.
        Reading that preview into text_cache would silently discard the
        user's real manual entry the next time the pin is disconnected.
        """
        cfg = self._point_sets[i]
        w = cfg.get("text_widget")
        if w is not None and w.winfo_exists() and not cfg.get("from_pin", False):
            cfg["text_cache"] = w.get("1.0", tk.END)
        return cfg["text_cache"]

    def _set_point_set_text(self, i: int, text: str) -> None:
        cfg = self._point_sets[i]
        cfg["text_cache"] = text
        w = cfg.get("text_widget")
        if w is not None and w.winfo_exists():
            state_before = str(w.cget("state"))
            w.configure(state="normal")
            w.delete("1.0", tk.END)
            w.insert("1.0", text)
            w.configure(state=state_before)

    def _update_point_set_entry_ui(self, i: int) -> None:
        cfg = self._point_sets[i]
        pin_lbl = cfg.get("pin_indicator")
        if pin_lbl is not None and pin_lbl.winfo_exists():
            if cfg["from_pin"]:
                pin_lbl.configure(text="●", fg="#226600")
            else:
                pin_lbl.configure(text="○", fg="#aaaaaa")

        w = cfg.get("text_widget")
        if w is None or not w.winfo_exists():
            return

        if cfg["from_pin"] and cfg["data"].shape[0] > 0:
            preview_lines = ["  ".join(f"{v:.4g}" for v in row)
                              for row in cfg["data"][:10]]
            if cfg["data"].shape[0] > 10:
                preview_lines.append("...")
            w.configure(state="normal")
            w.delete("1.0", tk.END)
            w.insert("1.0", "\n".join(preview_lines))
            w.configure(state="disabled")
        else:
            w.configure(state="normal")
            current = w.get("1.0", tk.END)
            if current.strip() != cfg["text_cache"].strip():
                w.delete("1.0", tk.END)
                w.insert("1.0", cfg["text_cache"])

    def _on_apply_btn(self) -> None:
        """Re-run with current inspector values (grid text + point set text)."""
        if self._request_downstream:
            self._request_downstream(self.node_id)

    # ── help popup ────────────────────────────────────────────────

    def _open_help(self) -> None:
        if self._help_popup is not None and self._help_popup.winfo_exists():
            self._help_popup.lift()
            return
        popup = tk.Toplevel()
        popup.title("Mesh & Scatter Viewer — Help")
        popup.geometry("720x620")
        popup.resizable(True, True)
        try:
            px, py = self.canvas.winfo_pointerxy()
            popup.geometry(f"+{px+16}+{py+16}")
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

    # ── core computation: grid axes ──────────────────────────────

    def _resolve_axis(self, pin_value, text_var, label, required):
        if pin_value is not None and isinstance(pin_value, np.ndarray) and pin_value.size > 0:
            return pin_value.ravel().astype(np.float64), True

        text = text_var.get().strip()
        arr = _parse_sequence(text)
        if arr is None or arr.size == 0:
            if required:
                raise ValueError(f"Axis '{label}' is empty or invalid.")
            return None, False
        return arr, False

    def _update_inspector_entries(self, x, y, z) -> None:
        def _sync(entry, var, arr, from_pin):
            if entry is None or not entry.winfo_exists():
                return
            if from_pin and arr is not None:
                text = "  ".join(f"{v:.6g}" for v in arr.ravel()[:20])
                if arr.size > 20:
                    text += "  ..."
                var.set(text)
                entry.configure(state="disabled")
            else:
                entry.configure(state="normal")
        _sync(self._x_entry, self._x_var, x, self._x_from_pin)
        _sync(self._y_entry, self._y_var, y, self._y_from_pin)
        _sync(self._z_entry, self._z_var, z, self._z_from_pin)

    def _resolve_point_set(self, pin_value, text: str, label: str) -> tuple[np.ndarray, bool, int]:
        if pin_value is not None and isinstance(pin_value, np.ndarray) and pin_value.size > 0:
            arr, dropped = _coerce_point_array(pin_value, label)
            return arr, True, dropped

        parsed, skipped = _parse_point_block(text)
        if parsed is None:
            return np.empty((0, 3), dtype=np.float64), False, skipped
        arr, dropped = _coerce_point_array(parsed, label)
        return arr, False, skipped + dropped

    def compute(self, inputs: dict) -> dict:
        # ── grid (optional) ────────────────────────────────────────
        pin_x, pin_y, pin_z = inputs.get("x"), inputs.get("y"), inputs.get("z")
        x_arr, self._x_from_pin = self._resolve_axis(pin_x, self._x_var, "x", required=False)
        y_arr, self._y_from_pin = self._resolve_axis(pin_y, self._y_var, "y", required=False)
        z_arr, self._z_from_pin = self._resolve_axis(pin_z, self._z_var, "z", required=False)
        self._update_inspector_entries(x_arr, y_arr, z_arr)

        have_grid = x_arr is not None and x_arr.size > 0 and y_arr is not None and y_arr.size > 0

        if have_grid:
            indexing = _normalize_indexing(self._indexing_var.get())
            axis_order = _AXIS_ORDER[indexing]
            mode_3d = z_arr is not None and z_arr.size > 0
            try:
                if mode_3d:
                    arrays = [x_arr if a == "x" else y_arr if a == "y" else z_arr
                              for a in axis_order]
                    mesh = np.meshgrid(*arrays, indexing="ij")
                    grid_coords = np.column_stack([p.ravel() for p in mesh]).astype(np.float64)
                    grid_shape = np.array([arr.size for arr in arrays], dtype=np.int64)
                    n = int(np.prod(grid_shape))
                    self._shape_var.set(
                        f"grid ({grid_shape[0]}×{grid_shape[1]}×{grid_shape[2]}) -> ({n},3)")
                else:
                    active = [a for a in axis_order if a != "z"]
                    arrays = [x_arr if a == "x" else y_arr for a in active]
                    mesh = np.meshgrid(*arrays, indexing="ij")
                    grid_coords = np.column_stack([p.ravel() for p in mesh]).astype(np.float64)
                    grid_shape = np.array([arr.size for arr in arrays], dtype=np.int64)
                    n = int(np.prod(grid_shape))
                    self._shape_var.set(f"grid ({grid_shape[0]}×{grid_shape[1]}) -> ({n},2)")
                self._last_grid_coords = grid_coords
                self._last_grid_shape = grid_shape
            except Exception as e:
                self._last_grid_coords = np.empty((0, 3), dtype=np.float64)
                self._last_grid_shape = np.empty((0,), dtype=np.int64)
                self._shape_var.set(f"grid error: {e}")
        else:
            self._last_grid_coords = np.empty((0, 3), dtype=np.float64)
            self._last_grid_shape = np.empty((0,), dtype=np.int64)
            self._shape_var.set("no grid (connect/enter x and y to add one)")

        # ── point sets ────────────────────────────────────────────
        all_parts, all_ids = [], []
        n_enabled_with_data = 0
        total_pts = 0

        for i in range(1, self.MAX_POINT_SETS + 1):
            cfg = self._point_sets[i]
            raw = inputs.get(f"points{i}")
            text = self._get_point_set_text(i)
            try:
                arr, from_pin, skipped = self._resolve_point_set(raw, text, f"points{i}")
                cfg["data"] = arr
                cfg["from_pin"] = from_pin
                if arr.shape[0] > 0:
                    cfg["status_var"].set(
                        f"{arr.shape[0]} pts" + (f"  ({skipped} skipped)" if skipped else ""))
                else:
                    cfg["status_var"].set(
                        "no data" if skipped == 0 else f"0 pts ({skipped} skipped)")
            except Exception as e:
                cfg["data"] = np.empty((0, 3), dtype=np.float64)
                cfg["from_pin"] = False
                cfg["status_var"].set(f"error: {e}")

            self._update_point_set_entry_ui(i)
            self._validate_point_set_grid_dims(i)

            if cfg["enabled_var"].get() and cfg["data"].shape[0] > 0:
                n_enabled_with_data += 1
                total_pts += cfg["data"].shape[0]
                all_parts.append(cfg["data"])
                all_ids.append(np.full(cfg["data"].shape[0], float(i), dtype=np.float64))

        points_all = (np.concatenate(all_parts, axis=0) if all_parts
                      else np.empty((0, 3), dtype=np.float64))
        points_set_id = (np.concatenate(all_ids, axis=0) if all_ids
                          else np.empty((0,), dtype=np.float64))

        self._sets_summary_var.set(
            f"sets: {n_enabled_with_data}/{self.MAX_POINT_SETS} enabled  |  {total_pts} pts")
        self._status_var.set("ok")
        self.set_status("ok", "#446622" if (have_grid or total_pts) else "#888888")

        self._refresh_preview()

        return {
            "grid_coords":   self._last_grid_coords,
            "grid_shape":    self._last_grid_shape,
            "points_all":    points_all,
            "points_set_id": points_set_id,
        }

    def _validate_point_set_grid_dims(self, i: int) -> None:
        """
        Check (but do not raise) whether this set's grid-lines checkbox
        is on and, if so, whether nx*ny*nz matches the current point
        count exactly. Populates grid_status_var either way so the
        inspector always explains why grid lines are/aren't drawn.
        """
        cfg = self._point_sets[i]
        if not cfg["grid_lines_var"].get():
            cfg["grid_status_var"].set("")
            return

        dims = _parse_grid_dims(cfg["grid_dims_var"].get())
        n_pts = cfg["data"].shape[0]
        if dims is None:
            cfg["grid_status_var"].set(
                'grid lines: enter 3 positive integers, e.g. "20 10 4"')
            return

        nx, ny, nz = dims
        expected = nx * ny * nz
        if n_pts == 0:
            cfg["grid_status_var"].set(
                f"grid lines: waiting for point data ({nx}×{ny}×{nz}={expected} expected)")
        elif expected != n_pts:
            cfg["grid_status_var"].set(
                f"grid lines: {nx}×{ny}×{nz}={expected} does not match "
                f"{n_pts} points -- grid lines skipped")
        else:
            cfg["grid_status_var"].set(
                f"grid lines: ok ({nx}×{ny}×{nz})")

    # ── unified projection pipeline (verbatim from Meshgrid's fixed version) ─

    def _combined_points_for_scale(self) -> np.ndarray:
        """Grid + every enabled, non-empty point set -- used for auto-fit
        scaling and rotation pivot estimation, so the view frames
        everything currently visible, not just the grid."""
        parts = []
        if self._last_grid_coords.size:
            pts = self._last_grid_coords
            if pts.shape[1] == 2:
                pts = np.column_stack([pts, np.zeros(pts.shape[0])])
            parts.append(pts)
        for i in range(1, self.MAX_POINT_SETS + 1):
            cfg = self._point_sets[i]
            if cfg["enabled_var"].get() and cfg["data"].size:
                parts.append(cfg["data"])
        if not parts:
            return np.empty((0, 3), dtype=np.float64)
        return np.concatenate(parts, axis=0)

    def _get_scale(self) -> float:
        pts = self._combined_points_for_scale()
        if pts.size == 0:
            return 1.0
        radius = max(float(np.max(np.linalg.norm(pts, axis=1))), 1e-6)
        canvas_min = 200.0
        if self._preview_canvas is not None and self._preview_canvas.winfo_exists():
            canvas_min = min(float(self._preview_canvas.winfo_width()),
                              float(self._preview_canvas.winfo_height()))
        return canvas_min * 0.38 * self._preview_state["zoom"] / radius

    def _rotation_matrix(self) -> np.ndarray:
        yaw, pitch = self._preview_state["yaw"], self._preview_state["pitch"]
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        Ryaw = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
        Rpitch = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float64)
        return Rpitch @ Ryaw

    def _project_points(self, pts: np.ndarray) -> np.ndarray:
        if self._preview_canvas is None or not self._preview_canvas.winfo_exists():
            return np.zeros((len(pts), 2))
        pts = np.asarray(pts, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        if pts.shape[1] == 2:
            pts = np.column_stack([pts, np.zeros(pts.shape[0])])

        target = np.asarray(self._preview_state["camera_target"], dtype=np.float64)
        rel = pts - target
        R = self._rotation_matrix()
        rot = (R @ rel.T).T

        scale = self._get_scale()
        sx = rot[:, 0] * scale
        sy = -rot[:, 2] * scale
        cw = float(self._preview_canvas.winfo_width())
        ch = float(self._preview_canvas.winfo_height())
        sx += cw / 2.0
        sy += ch / 2.0
        return np.column_stack([sx, sy])

    def _screen_to_world_delta(self, dx_screen: float, dy_screen: float) -> np.ndarray:
        scale = self._get_scale()
        if scale < 1e-12:
            return np.zeros(3)
        R = self._rotation_matrix()
        Rt = R.T
        return (Rt[:, 0] * dx_screen - Rt[:, 2] * dy_screen) / scale

    # ── drag / pan / zoom ─────────────────────────────────────────
    # NOTE: rotation always pivots around camera_target (set by
    # right-drag panning) via _project_points above. The older
    # per-click "rotation anchor" estimation logic from an earlier
    # version has been removed -- it referenced
    # self._preview_state["pan_x"/"pan_y"], which no longer exist
    # after the pan/rotation coordinate-frame fix, so it would have
    # raised KeyError the first time a user rotated the view. This
    # simpler model (pivot = current pan target) needs no anchor
    # estimation at all.

    def _on_preview_drag_start(self, event) -> None:
        self._preview_state["dragging"] = True
        self._preview_state["last_x"] = event.x
        self._preview_state["last_y"] = event.y
        self._preview_state["button"] = 1

    def _on_preview_drag_motion(self, event) -> None:
        if not self._preview_state["dragging"] or self._preview_state["button"] != 1:
            return
        dx = event.x - self._preview_state["last_x"]
        dy = event.y - self._preview_state["last_y"]
        self._preview_state["yaw"] += dx * 0.01
        self._preview_state["pitch"] += dy * 0.01
        self._preview_state["last_x"] = event.x
        self._preview_state["last_y"] = event.y
        self._refresh_preview()

    def _on_preview_drag_release(self, _event) -> None:
        self._preview_state["dragging"] = False
        self._preview_state["button"] = None

    def _on_preview_pan_start(self, event) -> None:
        self._preview_state["dragging"] = True
        self._preview_state["last_x"] = event.x
        self._preview_state["last_y"] = event.y
        self._preview_state["button"] = 3

    def _on_preview_pan_motion(self, event) -> None:
        if not self._preview_state["dragging"] or self._preview_state["button"] != 3:
            return
        dx = event.x - self._preview_state["last_x"]
        dy = event.y - self._preview_state["last_y"]
        world_delta = self._screen_to_world_delta(dx, dy)
        target = np.asarray(self._preview_state["camera_target"], dtype=np.float64)
        self._preview_state["camera_target"] = (target - world_delta).tolist()
        self._preview_state["last_x"] = event.x
        self._preview_state["last_y"] = event.y
        self._refresh_preview()

    def _on_preview_pan_release(self, _event) -> None:
        self._preview_state["dragging"] = False
        self._preview_state["button"] = None

    def _on_preview_mousewheel(self, event) -> str:
        if self._preview_canvas is None or not self._preview_canvas.winfo_exists():
            return "break"
        delta = int(event.delta / 120)
        if delta == 0:
            return "break"
        self._preview_state["zoom"] = max(0.05, self._preview_state["zoom"] * (1.12 ** delta))
        self._refresh_preview()
        return "break"

    # ── marker rendering (shared by grid points and scatter points) ──

    @staticmethod
    def _draw_marker(canvas, x, y, marker, size, color, width, filled, tags=()):
        fill = color if filled else ""
        if marker == "o":
            canvas.create_oval(x - size, y - size, x + size, y + size,
                               fill=fill, outline=color, width=width, tags=tags)
        elif marker == "s":
            canvas.create_rectangle(x - size, y - size, x + size, y + size,
                                    fill=fill, outline=color, width=width, tags=tags)
        elif marker == "D":
            canvas.create_polygon(x, y - size, x + size, y, x, y + size, x - size, y,
                                  fill=fill, outline=color, width=width, tags=tags)
        elif marker == "^":
            canvas.create_polygon(x, y - size, x + size, y + size, x - size, y + size,
                                  fill=fill, outline=color, width=width, tags=tags)
        elif marker == "v":
            canvas.create_polygon(x, y + size, x + size, y - size, x - size, y - size,
                                  fill=fill, outline=color, width=width, tags=tags)
        elif marker == "<":
            canvas.create_polygon(x - size, y, x + size, y - size, x + size, y + size,
                                  fill=fill, outline=color, width=width, tags=tags)
        elif marker == ">":
            canvas.create_polygon(x + size, y, x - size, y - size, x - size, y + size,
                                  fill=fill, outline=color, width=width, tags=tags)
        elif marker in ("+", "*", "x"):
            w = max(1, int(width))
            if marker in ("+", "*"):
                canvas.create_line(x - size, y, x + size, y, fill=color, width=w, tags=tags)
                canvas.create_line(x, y - size, x, y + size, fill=color, width=w, tags=tags)
            if marker in ("x", "*"):
                canvas.create_line(x - size, y - size, x + size, y + size,
                                   fill=color, width=w, tags=tags)
                canvas.create_line(x - size, y + size, x + size, y - size,
                                   fill=color, width=w, tags=tags)
        else:
            canvas.create_oval(x - size, y - size, x + size, y + size,
                               fill=fill, outline=color, width=width, tags=tags)

    # ── axis triad (unchanged from Meshgrid) ─────────────────────

    _CUBE_VERTS = np.array([
        [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
    ], dtype=np.float64)

    _CUBE_FACES = [
        {"normal": np.array([1.0, 0.0, 0.0]),  "verts": [1, 2, 6, 5], "color": "#ff9999"},
        {"normal": np.array([-1.0, 0.0, 0.0]), "verts": [0, 3, 7, 4], "color": "#ffcccc"},
        {"normal": np.array([0.0, 1.0, 0.0]),  "verts": [3, 2, 6, 7], "color": "#99dd99"},
        {"normal": np.array([0.0, -1.0, 0.0]), "verts": [0, 1, 5, 4], "color": "#cceecc"},
        {"normal": np.array([0.0, 0.0, 1.0]),  "verts": [4, 5, 6, 7], "color": "#9999ff"},
        {"normal": np.array([0.0, 0.0, -1.0]), "verts": [0, 1, 2, 3], "color": "#ccccff"},
    ]

    def _draw_axis_triad(self, canvas: tk.Canvas) -> None:
        if not canvas.winfo_exists() or canvas.winfo_width() < 2:
            return
        R = self._rotation_matrix()
        corner = np.array([32.0, float(canvas.winfo_height()) - 32.0])
        axis_len = 22.0

        for axis, color, label in [
            (np.array([1.0, 0.0, 0.0]), "#cc0000", "X"),
            (np.array([0.0, 1.0, 0.0]), "#00aa00", "Y"),
            (np.array([0.0, 0.0, 1.0]), "#0000cc", "Z"),
        ]:
            rot = R @ axis
            end = corner + np.array([rot[0], -rot[2]]) * axis_len
            canvas.create_line(corner[0], corner[1], end[0], end[1], fill=color, width=2)
            canvas.create_text(
                end[0] + (end[0] - corner[0]) * 0.2, end[1] + (end[1] - corner[1]) * 0.2,
                text=label, fill=color, font=("Arial", 8, "bold"))

        cube_half = axis_len * 0.32
        verts = self._CUBE_VERTS * cube_half
        faces_by_depth = sorted(self._CUBE_FACES, key=lambda f: float((R @ f["normal"])[1]))
        for face in faces_by_depth[:3]:
            pts = [(corner[0] + (R @ verts[vi])[0], corner[1] - (R @ verts[vi])[2])
                   for vi in face["verts"]]
            canvas.create_polygon(pts, fill=face["color"], outline="#333333", width=1.3)

    # ── grid + point-set drawing ──────────────────────────────────

    def _draw_grid(self, canvas: tk.Canvas, shape: tuple[int, ...]) -> None:
        pts = self._last_grid_coords
        size = float(self._point_size_var.get())
        color = self._point_color_var.get()
        marker = self._marker_var.get() or "o"

        if len(shape) == 2:
            if self._show_points_var.get():
                for (x, y) in self._project_points(pts):
                    self._draw_marker(canvas, x, y, marker, size, color, 1, filled=False)
            return

        if len(shape) != 3:
            return

        grid = pts.reshape(shape[0], shape[1], shape[2], 3)
        order = _AXIS_ORDER[_normalize_indexing(self._indexing_var.get())]
        axis_axes = {"x": order.index("x"), "y": order.index("y"), "z": order.index("z")}

        if self._show_lines_var.get():
            line_color = self._line_color_var.get() or "#444444"
            line_width = max(1.0, float(self._line_width_var.get()))
            dash_map = {"solid": (), "dash": (8, 4), "dot": (2, 4), "dashdot": (6, 3, 2, 3)}
            dash = dash_map.get(self._line_style_var.get(), ())
            self._draw_structured_grid_lines(
                canvas, grid, shape, axis_axes,
                x_enabled=self._show_x_lines_var.get(),
                y_enabled=self._show_y_lines_var.get(),
                z_enabled=self._show_z_lines_var.get(),
                color=line_color, width=line_width, dash=dash,
            )

        if self._show_points_var.get():
            for (x, y) in self._project_points(pts):
                self._draw_marker(canvas, x, y, marker, size, color, 1, filled=False)

    def _draw_structured_grid_lines(
        self, canvas: tk.Canvas, grid: np.ndarray, shape: tuple[int, int, int],
        axis_axes: dict, x_enabled: bool, y_enabled: bool, z_enabled: bool,
        color: str, width: float, dash: tuple,
    ) -> None:
        """
        Draw connecting lines between neighbouring grid points along
        each enabled axis. `grid` must already be reshaped to
        (shape[0], shape[1], shape[2], 3). `axis_axes` maps the
        logical axis name ("x"/"y"/"z") to its position (0, 1, or 2)
        within `shape`/`grid`'s first three dimensions.

        Shared by both the main structured Grid tab and each point
        set's optional 3D grid-lines overlay, so both features stay
        visually and behaviourally consistent.
        """
        for axis_name, enabled in (("x", x_enabled), ("y", y_enabled), ("z", z_enabled)):
            if not enabled:
                continue
            axis_idx = axis_axes[axis_name]
            other = [d for d in range(3) if d != axis_idx]
            for fixed in np.ndindex(*(shape[d] for d in other)):
                idx = [slice(None)] * 3
                for local_pos, real_pos in enumerate(other):
                    idx[real_pos] = fixed[local_pos]
                seg = grid[tuple(idx)]
                if seg.shape[0] < 2:
                    continue
                proj = self._project_points(seg)
                for i in range(len(proj) - 1):
                    p0, p1 = proj[i], proj[i + 1]
                    canvas.create_line(p0[0], p0[1], p1[0], p1[1],
                                      fill=color, width=width, dash=dash)

    def _draw_point_sets(self, canvas: tk.Canvas) -> None:
        for i in range(1, self.MAX_POINT_SETS + 1):
            cfg = self._point_sets[i]
            if not cfg["enabled_var"].get() or cfg["data"].shape[0] == 0:
                continue

            # ── optional 3D grid lines for this point set ────────────
            if cfg["grid_lines_var"].get():
                dims = _parse_grid_dims(cfg["grid_dims_var"].get())
                if dims is not None:
                    nx, ny, nz = dims
                    if nx * ny * nz == cfg["data"].shape[0]:
                        try:
                            grid = cfg["data"].reshape(nx, ny, nz, 3)
                            line_color = cfg["grid_line_color_var"].get() or "#888888"
                            try:
                                line_width = max(0.5, float(cfg["grid_line_width_var"].get()))
                            except (tk.TclError, ValueError):
                                line_width = 1.0
                            self._draw_structured_grid_lines(
                                canvas, grid, (nx, ny, nz),
                                axis_axes={"x": 0, "y": 1, "z": 2},
                                x_enabled=True, y_enabled=True, z_enabled=True,
                                color=line_color, width=line_width, dash=(),
                            )
                        except Exception:
                            pass  # reshape can't actually fail given the size check above

            # ── scatter markers ───────────────────────────────────────
            color = cfg["color_var"].get() or "red"
            marker = cfg["marker_var"].get() or "o"
            try:
                size = float(cfg["size_var"].get())
            except (tk.TclError, ValueError):
                size = 4.0
            for (x, y) in self._project_points(cfg["data"]):
                self._draw_marker(canvas, x, y, marker, size, color, 1, filled=True)

    def _refresh_preview(self) -> None:
        canvas = self._preview_canvas
        if canvas is None or not canvas.winfo_exists():
            return
        canvas.delete("all")

        if not self._preview_enabled_var.get():
            canvas.create_text(canvas.winfo_width() / 2, canvas.winfo_height() / 2,
                               text="3D preview disabled", fill="#999999")
            return

        combined = self._combined_points_for_scale()
        if combined.size == 0:
            canvas.create_text(canvas.winfo_width() / 2, canvas.winfo_height() / 2,
                               text="Compute a grid or enable a point set to preview",
                               fill="#777777")
            return

        self._draw_axis_triad(canvas)

        if self._last_grid_coords.size:
            shape = tuple(int(v) for v in self._last_grid_shape)
            if len(shape) in (2, 3) and int(np.prod(shape, dtype=np.int64)) == self._last_grid_coords.shape[0]:
                self._draw_grid(canvas, shape)
            else:
                canvas.create_text(canvas.winfo_width() / 2, canvas.winfo_height() / 2,
                                   text="Invalid grid shape — recompute",
                                   fill="#aa0000", font=("Arial", 9, "bold"))

        self._draw_point_sets(canvas)

    # ── serialization ─────────────────────────────────────────────

    def get_params(self) -> dict:
        point_sets_params = []
        for i in range(1, self.MAX_POINT_SETS + 1):
            cfg = self._point_sets[i]
            self._get_point_set_text(i)  # flush widget -> cache if applicable
            point_sets_params.append({
                "enabled": bool(cfg["enabled_var"].get()),
                "label":   cfg["label_var"].get(),
                "color":   cfg["color_var"].get(),
                "marker":  cfg["marker_var"].get(),
                "size":    float(cfg["size_var"].get()),
                "text":    cfg["text_cache"],
                "grid_lines": bool(cfg["grid_lines_var"].get()),
                "grid_dims": cfg["grid_dims_var"].get(),
                "grid_line_color": cfg["grid_line_color_var"].get(),
                "grid_line_width": float(cfg["grid_line_width_var"].get()),
            })
        return {
            "x": self._x_var.get(),
            "y": self._y_var.get(),
            "z": self._z_var.get(),
            "indexing": _normalize_indexing(self._indexing_var.get()),
            "point_size": float(self._point_size_var.get()),
            "point_color": self._point_color_var.get(),
            "line_width": float(self._line_width_var.get()),
            "line_color": self._line_color_var.get(),
            "marker": self._marker_var.get(),
            "line_style": self._line_style_var.get(),
            "show_points": bool(self._show_points_var.get()),
            "show_lines": bool(self._show_lines_var.get()),
            "show_x_lines": bool(self._show_x_lines_var.get()),
            "show_y_lines": bool(self._show_y_lines_var.get()),
            "show_z_lines": bool(self._show_z_lines_var.get()),
            "preview_enabled": bool(self._preview_enabled_var.get()),
            "point_sets": point_sets_params,
        }

    def set_params(self, params: dict) -> None:
        self._init_state()
        self._x_var.set(str(params.get("x", "-8:2:8")))
        self._y_var.set(str(params.get("y", "-6:2:6")))
        self._z_var.set(str(params.get("z", "")))
        self._indexing_var.set(_normalize_indexing(params.get("indexing", "ij")))

        self._point_size_var.set(float(params.get("point_size", 2.0)))
        self._point_color_var.set(str(params.get("point_color", "blue")))
        self._line_width_var.set(float(params.get("line_width", 1.0)))
        self._line_color_var.set(str(params.get("line_color", "black")))
        self._marker_var.set(str(params.get("marker", "o")))
        self._line_style_var.set(str(params.get("line_style", "solid")))
        self._show_points_var.set(bool(params.get("show_points", True)))
        self._show_lines_var.set(bool(params.get("show_lines", True)))
        self._show_x_lines_var.set(bool(params.get("show_x_lines", True)))
        self._show_y_lines_var.set(bool(params.get("show_y_lines", True)))
        self._show_z_lines_var.set(bool(params.get("show_z_lines", True)))
        self._preview_enabled_var.set(bool(params.get("preview_enabled", False)))

        saved_sets = params.get("point_sets", [])
        for i in range(1, self.MAX_POINT_SETS + 1):
            cfg = self._point_sets[i]
            saved = saved_sets[i - 1] if i - 1 < len(saved_sets) else {}
            cfg["enabled_var"].set(bool(saved.get("enabled", i == 1)))
            cfg["label_var"].set(str(saved.get("label", f"Set {i}")))
            cfg["color_var"].set(str(saved.get("color", self._DEFAULT_SET_COLORS[i - 1])))
            cfg["marker_var"].set(str(saved.get("marker", self._DEFAULT_SET_MARKERS[i - 1])))
            cfg["size_var"].set(float(saved.get("size", 4.0)))
            self._set_point_set_text(i, str(saved.get("text", "")))
            cfg["grid_lines_var"].set(bool(saved.get("grid_lines", False)))
            cfg["grid_dims_var"].set(str(saved.get("grid_dims", "")))
            cfg["grid_line_color_var"].set(
                str(saved.get("grid_line_color", self._DEFAULT_SET_LINE_COLORS[i - 1])))
            cfg["grid_line_width_var"].set(float(saved.get("grid_line_width", 1.0)))

    def close_inspector(self) -> None:
        for i in range(1, self.MAX_POINT_SETS + 1):
            self._get_point_set_text(i)  # flush before losing the widget
            self._point_sets[i]["text_widget"] = None
            self._point_sets[i]["pin_indicator"] = None
        super().close_inspector()
        self._x_entry = None
        self._y_entry = None
        self._z_entry = None

    def on_destroy(self) -> None:
        if self._help_popup is not None and self._help_popup.winfo_exists():
            self._help_popup.destroy()
        super().on_destroy()