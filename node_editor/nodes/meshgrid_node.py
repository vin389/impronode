# node_editor/nodes/meshgrid_node.py

import math
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

from node_editor.base_node import BaseNode
from node_editor.pin_types import PinSchema, PinDef, PinType
from node_editor.execution import ExecutionMode


# ---------------------------------------------------------------------------
# Help text
# ---------------------------------------------------------------------------

_HELP_TEXT = """\
Meshgrid Node
=============

PURPOSE
-------
Generates a grid of 3D (or 2D) coordinates aligned with the
coordinate axes, equivalent to:

    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    grid_coords = np.column_stack([xx.ravel(),
                                   yy.ravel(),
                                   zz.ravel()])

If z is empty or not connected, the node operates in 2D mode
and returns an (N, 2) array instead of (N, 3).

INPUT PINS
----------
x    ARRAY (optional)
     1D float array of x-axis sample positions.
     If connected, the pin value overrides the inspector
     text box.

y    ARRAY (optional)
     1D float array of y-axis sample positions.
     Same pin-first priority.

z    ARRAY (optional)
     1D float array of z-axis sample positions.
     Leave disconnected or empty for 2D mode.

OUTPUT PINS
-----------
grid_coords  ARRAY
     Shape (N, 3) in 3D mode, or (N, 2) in 2D mode.
     N = x.size * y.size            (2D)
     N = x.size * y.size * z.size   (3D)
     Row order follows the selected axis order:
       default:  (x, y, z)  — x slowest, z fastest
       Matlab/xy style:  (y, x, z)

grid_shape   ARRAY
     Integer array matching the chosen axis order, e.g.
     [nx, ny, nz] or [ny, nx, nz].
     Useful for reshaping grid_coords back to the grid
     structure in downstream nodes.

INSPECTOR SETTINGS
------------------
X / Y / Z text boxes
     Accepts two syntaxes:
       Direct values:  -30  -20  -10  0  10  20  30
       Sequence:       start:step:end   (e.g.  -30:10:30)
                       start:end        (step defaults to 1.0)
     Leave Z empty for 2D mode.

Axis order
     (nx, ny, nz) ('ij')  — default, scientific / engineering order
     (ny, nx, nz) ('xy' / Matlab style)  — image / plot order
     (nx, nz, ny)
     (nz, nx, ny)
     (ny, nz, nx)
     (nz, ny, nx)

KEYBOARD SHORTCUT
-----------------
Ctrl-H / Ctrl-h  — show this help window.
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
    "orange", "purple", "white", "gray", "brown",
    "pink",
]


def _normalize_indexing(value: str | None) -> str:
    if value is None:
        return "ij"
    value = str(value).strip().lower()
    mapping = {
        "ij": "ij",
        "xy": "xy",
        "xyz": "ij",
        "yxz": "xy",
        "xzy": "xzy",
        "zxy": "zxy",
        "yzx": "yzx",
        "zyx": "zyx",
    }
    return mapping.get(value, "ij")


# ---------------------------------------------------------------------------
# Sequence parser
# ---------------------------------------------------------------------------

def _parse_sequence(text: str) -> np.ndarray | None:
    """
    Parse a 1D array from a text string.

    Supported formats:
      "v0 v1 v2 ..."         space/comma separated floats
      "start:step:end"       MATLAB-style, end is inclusive
      "start:end"            step defaults to 1.0, end inclusive

    Returns None if the text is empty or cannot be parsed.
    """
    text = text.strip()
    if not text:
        return None

    # MATLAB sequence syntax
    if ":" in text:
        parts = text.split(":")
        try:
            if len(parts) == 2:
                start = float(parts[0])
                end   = float(parts[1])
                step  = 1.0
            elif len(parts) == 3:
                start = float(parts[0])
                step  = float(parts[1])
                end   = float(parts[2])
            else:
                return None
            if step == 0:
                return None
            # inclusive end — add a small epsilon in the
            # direction of step to ensure end is included
            # when it falls exactly on a multiple of step
            epsilon = abs(step) * 1e-9
            arr = np.arange(
                start,
                end + np.sign(step) * epsilon,
                step,
                dtype=np.float64)
            return arr if arr.size > 0 else None
        except ValueError:
            return None

    # space / comma separated floats
    try:
        nums = [float(t)
                for t in text.replace(",", " ").split()
                if t.strip()]
        if not nums:
            return None
        return np.array(nums, dtype=np.float64)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# MeshgridNode
# ---------------------------------------------------------------------------

class MeshgridNode(BaseNode):
    """
    Generates a grid of N*2 or N*3 coordinates from 1D x, y, (z) arrays.

    Equivalent to np.meshgrid with a column_stack to produce (N, 2/3).
    Row order with indexing='ij': x slowest, z fastest.
    """

    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE      = "meshgrid"
    DISPLAY_NAME   = "Meshgrid"
    CATEGORY       = "process"
    NODE_WIDTH     = 200
    NODE_HEIGHT    = 115

    HELP_TEXT = _HELP_TEXT   # picked up by NodeEditorApp Ctrl-H handler

    # body colors — bright background, dark text
    _BODY_BG    = "#fffde7"   # very light yellow
    _BODY_FG    = "#333300"
    _OUTLINE    = "#ccaa00"
    _TITLE_FG   = "#554400"
    _STATUS_FG  = "#776600"
    _PIN_COLOR  = "#334400"   # dark color for pin labels

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[
                PinDef("x", PinType.ARRAY, "x",
                       optional=True),
                PinDef("y", PinType.ARRAY, "y",
                       optional=True),
                PinDef("z", PinType.ARRAY, "z",
                       optional=True),
            ],
            outputs=[
                PinDef("grid_coords", PinType.ARRAY,
                       "coords"),
                PinDef("grid_shape",  PinType.ARRAY,
                       "shape"),
            ]
        )

    # ── state init ────────────────────────────────────────────────

    def _init_state(self) -> None:
        if not hasattr(self, "_x_var"):
            self._x_var = tk.StringVar(value="-8:2:8")
            self._y_var = tk.StringVar(value="-6:2:6")
            self._z_var = tk.StringVar(value="-4:2:4")
            self._indexing_var = tk.StringVar(value="ij")

            self._point_size_var = tk.DoubleVar(value=2.0)
            self._point_color_var = tk.StringVar(value="blue")
            self._line_width_var = tk.DoubleVar(value=1.0)
            self._line_color_var = tk.StringVar(value="black")
            self._marker_var = tk.StringVar(value="o")
            self._line_style_var = tk.StringVar(value="solid")
            self._show_points_var = tk.BooleanVar(value=True)
            self._show_lines_var = tk.BooleanVar(value=True)
            self._show_x_lines_var = tk.BooleanVar(value=True)
            self._show_y_lines_var = tk.BooleanVar(value=True)
            self._show_z_lines_var = tk.BooleanVar(value=True)
            self._preview_enabled_var = tk.BooleanVar(value=False)

            self._status_var = tk.StringVar(value="not computed")
            self._shape_var = tk.StringVar(value="")

            self._x_entry: tk.Entry | None = None
            self._y_entry: tk.Entry | None = None
            self._z_entry: tk.Entry | None = None
            self._preview_canvas: tk.Canvas | None = None
            self._help_popup: tk.Toplevel | None = None
            self._last_grid_coords = np.empty((0, 3), dtype=np.float64)
            self._last_grid_shape = np.empty((0,), dtype=np.int64)
            self._preview_state = {
                "yaw":           -0.9,
                "pitch":          0.7,
                "zoom":           1.0,
                "camera_target": [0.0, 0.0, 0.0],  # world-space pan
                "dragging":       False,
                "last_x":         0,
                "last_y":         0,
                "button":         None,
            }
            self._x_from_pin = False
            self._y_from_pin = False
            self._z_from_pin = False

        # Backfill compatibility for older partial state created before all
        # visualization attributes were added.
        for name, factory in {
            "_point_size_var": lambda: tk.DoubleVar(value=2.0),
            "_point_color_var": lambda: tk.StringVar(value="blue"),
            "_line_width_var": lambda: tk.DoubleVar(value=1.0),
            "_line_color_var": lambda: tk.StringVar(value="black"),
            "_marker_var": lambda: tk.StringVar(value="o"),
            "_line_style_var": lambda: tk.StringVar(value="solid"),
            "_show_points_var": lambda: tk.BooleanVar(value=True),
            "_show_lines_var": lambda: tk.BooleanVar(value=True),
            "_show_x_lines_var": lambda: tk.BooleanVar(value=True),
            "_show_y_lines_var": lambda: tk.BooleanVar(value=True),
            "_show_z_lines_var": lambda: tk.BooleanVar(value=True),
            "_preview_enabled_var": lambda: tk.BooleanVar(value=False),
            "_status_var": lambda: tk.StringVar(value="not computed"),
            "_shape_var": lambda: tk.StringVar(value=""),
            "_x_entry": lambda: None,
            "_y_entry": lambda: None,
            "_z_entry": lambda: None,
            "_preview_canvas": lambda: None,
            "_help_popup": lambda: None,
            "_last_grid_coords": lambda: np.empty((0, 3), dtype=np.float64),
            "_last_grid_shape": lambda: np.empty((0,), dtype=np.int64),
            "_preview_state": lambda: {
                "yaw": -0.9,
                "pitch": 0.7,
                "zoom": 1.0,
                "camera_target": [0.0, 0.0, 0.0],
                "dragging": False,
                "last_x": 0,
                "last_y": 0,
                "button": None,
                "rotation_anchor": np.zeros(3, dtype=np.float64),
            },
            "_x_from_pin": lambda: False,
            "_y_from_pin": lambda: False,
            "_z_from_pin": lambda: False,
        }.items():
            if not hasattr(self, name):
                setattr(self, name, factory())

    def get_help_text(self) -> str:
        return _HELP_TEXT

    # ── build_body ────────────────────────────────────────────────

    def build_body(self) -> None:
        self._init_state()
        x, y, w, h = self.x, self.y, self.width, self.height

        self._body_rect = self.canvas.create_rectangle(
            x, y, x+w, y+h,
            fill=self._BODY_BG,
            outline=self._OUTLINE, width=2,
            tags=(self.node_id, "node_body"))

        self._title_item = self.canvas.create_text(
            x+w/2, y+13,
            text=self.DISPLAY_NAME,
            font=("Arial", 9, "bold"),
            fill=self._TITLE_FG,
            tags=(self.node_id,))

        # shape summary
        shape_lbl = tk.Label(
            self.canvas,
            textvariable=self._shape_var,
            font=("Arial", 8, "bold"),
            bg=self._BODY_BG, fg=self._PIN_COLOR)
        self.canvas.create_window(
            x+w/2, y+h//2+4,
            window=shape_lbl,
            tags=(self.node_id,))

        # status
        status_lbl = tk.Label(
            self.canvas,
            textvariable=self._status_var,
            font=("Arial", 7),
            bg=self._BODY_BG, fg=self._STATUS_FG,
            wraplength=w-12, justify="center")
        self.canvas.create_window(
            x+w/2, y+h-12,
            window=status_lbl,
            tags=(self.node_id,))

        self._canvas_items += [
            self._body_rect, self._title_item]

        self.canvas.tag_bind(
            self.node_id,
            "<Double-Button-1>",
            self._on_double_click,
        )

    def _on_double_click(self, _event=None) -> str | None:
        self.open_inspector()
        return "break"

    # ── build_inspector ───────────────────────────────────────────

    def build_inspector(self, parent: tk.Frame) -> None:
        self._init_state()

        for child in parent.winfo_children():
            child.destroy()

        # bind Ctrl-H inside inspector window
        win = self._inspector_win
        if win is not None:
            for seq in ("<Control-h>", "<Control-H>"):
                win.bind(seq,
                         lambda e: self._open_help())

        pad = {"padx": 8, "pady": 4}

        panes = tk.PanedWindow(parent, orient=tk.HORIZONTAL, sashwidth=6, showhandle=True)
        panes.pack(fill="both", expand=True)

        left = tk.Frame(panes, width=380, bg="#f7f7f7")
        left.pack_propagate(False)
        middle = tk.Frame(panes, width=260, bg="#f2f2f2")
        middle.pack_propagate(False)
        right = tk.Frame(panes, width=420, bg="#ffffff")
        right.pack_propagate(False)
        panes.add(left, stretch="always")
        panes.add(middle, stretch="always")
        panes.add(right, stretch="always")

        def _coord_row(frame: tk.Frame, label: str, var: tk.StringVar, from_pin: bool, hint: str = "") -> tk.Entry:
            row = tk.Frame(frame)
            row.pack(fill="x", pady=2)
            pin_ind = tk.Label(row, text="●" if from_pin else "○", font=("Arial", 9), fg="#226600" if from_pin else "#aaaaaa", width=2)
            pin_ind.pack(side="left")
            tk.Label(row, text=label, font=("Arial", 9, "bold"), width=3, anchor="w").pack(side="left")
            ent = tk.Entry(row, textvariable=var, font=("Courier", 9), state="disabled" if from_pin else "normal", disabledforeground="#888888", disabledbackground="#eeeeee")
            ent.pack(side="left", fill="x", expand=True, padx=(4, 0))
            if hint:
                tk.Label(row, text=hint, font=("Arial", 7), fg="#888888").pack(side="right")
            return ent

        coord_frame = tk.LabelFrame(left, text="Coordinate arrays  (pin input overrides text box)", font=("Arial", 9), **pad)
        coord_frame.pack(fill="x", **pad)
        syntax_lbl = tk.Label(coord_frame, text='Syntax:  "v0 v1 v2 ..."  or  "start:step:end"  or  "start:end"  (step=1)', font=("Arial", 8), fg="#555555", anchor="w", justify="left")
        syntax_lbl.pack(fill="x", pady=(0, 4))
        self._x_entry = _coord_row(coord_frame, "x:", self._x_var, self._x_from_pin, "(required)")
        self._y_entry = _coord_row(coord_frame, "y:", self._y_var, self._y_from_pin, "(required)")
        self._z_entry = _coord_row(coord_frame, "z:", self._z_var, self._z_from_pin, "(empty → 2D)")

        idx_frame = tk.LabelFrame(left, text="Axis order", font=("Arial", 9), **pad)
        idx_frame.pack(fill="x", **pad)
        for mode, label in _AXIS_LABELS.items():
            tk.Radiobutton(idx_frame, text=label, variable=self._indexing_var, value=mode, font=("Arial", 9), justify=tk.LEFT, anchor="w").pack(anchor="w", pady=1)

        out_frame = tk.LabelFrame(left, text="Output", font=("Arial", 9), **pad)
        out_frame.pack(fill="x", **pad)
        tk.Label(out_frame, textvariable=self._shape_var, font=("Arial", 9, "bold"), fg="#224400", anchor="w").pack(fill="x")
        tk.Label(out_frame, textvariable=self._status_var, font=("Arial", 9), fg="#444444", anchor="w").pack(fill="x")

        btn_row = tk.Frame(left)
        btn_row.pack(fill="x", **pad)
        tk.Button(btn_row, text="Compute", font=("Arial", 9, "bold"), bg="#446622", fg="white", activebackground="#557733", relief=tk.FLAT, padx=10, pady=3, command=self._on_compute_btn).pack(side="left")
        tk.Button(btn_row, text="Help  (Ctrl-H)", font=("Arial", 8), command=self._open_help).pack(side="right")

        vis_frame = tk.LabelFrame(middle, text="Visualization", font=("Arial", 9), **pad)
        vis_frame.pack(fill="both", expand=True, **pad)

        tk.Checkbutton(vis_frame, text="Show points", variable=self._show_points_var, command=self._refresh_preview).pack(anchor="w")
        tk.Checkbutton(vis_frame, text="Show grid lines", variable=self._show_lines_var, command=self._refresh_preview).pack(anchor="w")
        tk.Checkbutton(vis_frame, text="x lines", variable=self._show_x_lines_var, command=self._refresh_preview).pack(anchor="w")
        tk.Checkbutton(vis_frame, text="y lines", variable=self._show_y_lines_var, command=self._refresh_preview).pack(anchor="w")
        tk.Checkbutton(vis_frame, text="z lines", variable=self._show_z_lines_var, command=self._refresh_preview).pack(anchor="w")

        ttk.Label(vis_frame, text="Point size:").pack(anchor="w", padx=6, pady=(8, 0))
        ttk.Spinbox(vis_frame, from_=0.5, to=10.0, increment=0.5, textvariable=self._point_size_var, width=8).pack(anchor="w", padx=6)
        ttk.Label(vis_frame, text="Point color:").pack(anchor="w", padx=6, pady=(8, 0))
        self._point_color_combo = ttk.Combobox(
            vis_frame,
            textvariable=self._point_color_var,
            values=_COMMON_COLORS,
            state="readonly",
            width=12,
        )
        self._point_color_combo.pack(anchor="w", padx=6)
        ttk.Label(vis_frame, text="Marker:").pack(anchor="w", padx=6, pady=(8, 0))
        ttk.Combobox(vis_frame, textvariable=self._marker_var, values=["o", "+", "*", "x", "s", "D", "^", "v", "<", ">"], state="readonly", width=10).pack(anchor="w", padx=6)
        ttk.Label(vis_frame, text="Grid line width:").pack(anchor="w", padx=6, pady=(8, 0))
        ttk.Spinbox(vis_frame, from_=0.5, to=6.0, increment=0.5, textvariable=self._line_width_var, width=8).pack(anchor="w", padx=6)
        ttk.Label(vis_frame, text="Grid line color:").pack(anchor="w", padx=6, pady=(8, 0))
        self._line_color_combo = ttk.Combobox(
            vis_frame,
            textvariable=self._line_color_var,
            values=_COMMON_COLORS,
            state="readonly",
            width=12,
        )
        self._line_color_combo.pack(anchor="w", padx=6)
        ttk.Label(vis_frame, text="Line style:").pack(anchor="w", padx=6, pady=(8, 0))
        ttk.Combobox(vis_frame, textvariable=self._line_style_var, values=["solid", "dash", "dot", "dashdot"], state="readonly", width=10).pack(anchor="w", padx=6)

        preview_frame = tk.LabelFrame(right, text="3D preview", font=("Arial", 9), **pad)
        preview_frame.pack(fill="both", expand=True, **pad)
        tk.Checkbutton(
            preview_frame, text="Enable 3D preview (slow with many points)",
            variable=self._preview_enabled_var,
            command=self._refresh_preview,
        ).pack(anchor="w")
        self._preview_canvas = tk.Canvas(preview_frame, bg="#ffffff", highlightthickness=0)
        self._preview_canvas.pack(fill="both", expand=True)
        self._preview_canvas.bind("<ButtonPress-1>", self._on_preview_drag_start)
        self._preview_canvas.bind("<B1-Motion>", self._on_preview_drag_motion)
        self._preview_canvas.bind("<ButtonRelease-1>", self._on_preview_drag_release)
        self._preview_canvas.bind("<ButtonPress-3>", self._on_preview_pan_start)
        self._preview_canvas.bind("<B3-Motion>", self._on_preview_pan_motion)
        self._preview_canvas.bind("<ButtonRelease-3>", self._on_preview_pan_release)
        self._preview_canvas.bind("<MouseWheel>", self._on_preview_mousewheel)
        self._preview_canvas.bind("<Shift-MouseWheel>", self._on_preview_mousewheel)
        self._preview_canvas.bind("<Configure>", lambda _e: self._refresh_preview())

        if win is not None:
            win.update_idletasks()
            min_w = max(980, win.winfo_reqwidth())
            min_h = max(620, win.winfo_reqheight())
            win.minsize(min_w, min_h)
            win.geometry(f"{min_w}x{min_h}")

        self._refresh_preview()

    # ── inspector helpers ─────────────────────────────────────────

    def _update_inspector_entries(
        self,
        x: np.ndarray | None,
        y: np.ndarray | None,
        z: np.ndarray | None,
    ) -> None:
        """
        When pins supply values, show them in the text boxes
        (disabled) so the user can see what data arrived.
        """
        def _sync(entry, var, arr, from_pin):
            if entry is None or \
                    not entry.winfo_exists():
                return
            if from_pin and arr is not None:
                text = "  ".join(
                    f"{v:.6g}" for v in
                    arr.ravel()[:20])
                if arr.size > 20:
                    text += "  ..."
                var.set(text)
                entry.configure(state="disabled")
            else:
                entry.configure(state="normal")

        _sync(self._x_entry, self._x_var,
              x, self._x_from_pin)
        _sync(self._y_entry, self._y_var,
              y, self._y_from_pin)
        _sync(self._z_entry, self._z_var,
              z, self._z_from_pin)

    def _on_compute_btn(self) -> None:
        """Re-run with current inspector values."""
        if self._request_downstream:
            self._request_downstream(self.node_id)

    # ── help popup ────────────────────────────────────────────────

    def _open_help(self) -> None:
        if (self._help_popup is not None
                and self._help_popup.winfo_exists()):
            self._help_popup.lift()
            return

        popup = tk.Toplevel()
        popup.title("Meshgrid Node — Help")
        popup.geometry("660x540")
        popup.resizable(True, True)

        try:
            px, py = self.canvas.winfo_pointerxy()
            popup.geometry(f"+{px+16}+{py+16}")
        except Exception:
            pass

        body = tk.Frame(
            popup, bg="#f8f8f8",
            padx=10, pady=8)
        body.pack(fill="both", expand=True)

        txt = tk.Text(
            body,
            font=("Courier", 9),
            bg="#f8f8f8", fg="#222222",
            wrap=tk.WORD,
            relief=tk.FLAT)
        vsb = ttk.Scrollbar(
            body, orient="vertical",
            command=txt.yview)
        txt.configure(
            yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", _HELP_TEXT)
        txt.configure(state="disabled")

        tk.Button(
            body, text="Close",
            font=("Arial", 9),
            command=popup.destroy).pack(
            anchor="e", pady=(8, 0))

        popup.bind(
            "<Escape>",
            lambda _e: popup.destroy())
        popup.protocol(
            "WM_DELETE_WINDOW",
            popup.destroy)
        self._help_popup = popup

    # ── core computation ──────────────────────────────────────────

    def _resolve_axis(
        self,
        pin_value: np.ndarray | None,
        text_var: tk.StringVar,
        label: str,
        required: bool,
    ) -> tuple[np.ndarray | None, bool]:
        """
        Resolve one axis array from pin (priority) or text box.
        Returns (array_or_None, from_pin).
        Raises ValueError for required axes that cannot be resolved.
        """
        if pin_value is not None and \
                isinstance(pin_value, np.ndarray) \
                and pin_value.size > 0:
            arr = pin_value.ravel().astype(
                np.float64)
            return arr, True

        text = text_var.get().strip()
        arr = _parse_sequence(text)

        if arr is None or arr.size == 0:
            if required:
                raise ValueError(
                    f"Axis '{label}' is empty or "
                    f"invalid.  Please connect a pin "
                    f"or enter values in the text box.")
            return None, False

        return arr, False

    def compute(self, inputs: dict) -> dict:
        pin_x = inputs.get("x")
        pin_y = inputs.get("y")
        pin_z = inputs.get("z")

        try:
            x_arr, self._x_from_pin = \
                self._resolve_axis(
                    pin_x, self._x_var, "x",
                    required=True)
            y_arr, self._y_from_pin = \
                self._resolve_axis(
                    pin_y, self._y_var, "y",
                    required=True)
            z_arr, self._z_from_pin = \
                self._resolve_axis(
                    pin_z, self._z_var, "z",
                    required=False)
        except ValueError as e:
            msg = str(e)
            self._status_var.set(f"⚠  {msg}")
            self._shape_var.set("")
            self.set_status("error", "#cc0000")
            # show warning dialog on main thread
            self.canvas.after(
                0,
                lambda m=msg: messagebox.showwarning(
                    "Meshgrid — Invalid Input", m))
            return {}

        # update inspector entry display
        self._update_inspector_entries(
            x_arr, y_arr, z_arr)

        indexing = _normalize_indexing(self._indexing_var.get())
        axis_order = _AXIS_ORDER[indexing]
        mode_3d  = (z_arr is not None
                    and z_arr.size > 0)

        try:
            if mode_3d:
                arrays = [
                    x_arr if axis == "x" else y_arr if axis == "y" else z_arr
                    for axis in axis_order
                ]
                mesh = np.meshgrid(*arrays, indexing="ij")
                grid_coords = np.column_stack([
                    part.ravel() for part in mesh
                ]).astype(np.float64)
                grid_shape = np.array(
                    [arr.size for arr in arrays],
                    dtype=np.int64)
                n = int(np.prod(grid_shape))
                shape_str = (
                    f"({grid_shape[0]} × "
                    f"{grid_shape[1]} × "
                    f"{grid_shape[2]})  →  "
                    f"({n}, 3)")
            else:
                active_axes = [axis for axis in axis_order if axis != "z"]
                arrays = [
                    x_arr if axis == "x" else y_arr
                    for axis in active_axes
                ]
                mesh = np.meshgrid(*arrays, indexing="ij")
                grid_coords = np.column_stack([
                    part.ravel() for part in mesh
                ]).astype(np.float64)
                grid_shape = np.array(
                    [arr.size for arr in arrays],
                    dtype=np.int64)
                n = int(np.prod(grid_shape))
                shape_str = (
                    f"({grid_shape[0]} × "
                    f"{grid_shape[1]})  →  "
                    f"({n}, 2)")

        except Exception as e:
            self._status_var.set(
                f"compute error: {e}")
            self._shape_var.set("")
            self.set_status("error", "#cc0000")
            return {}

        dim_str = "3D" if mode_3d else "2D"
        self._status_var.set(
            f"ok  {dim_str}  order={indexing}")
        self._shape_var.set(shape_str)
        self.set_status("ok", "#446622")

        if int(np.prod(grid_shape, dtype=np.int64)) != int(grid_coords.shape[0]):
            raise ValueError(
                f"Grid shape {tuple(int(v) for v in grid_shape)} does not match "
                f"the flattened coordinate count {int(grid_coords.shape[0])}."
            )

        self._last_grid_coords = np.asarray(grid_coords, dtype=np.float64)
        self._last_grid_shape = np.asarray(grid_shape, dtype=np.int64)
        self._refresh_preview()

        return {
            "grid_coords": grid_coords,
            "grid_shape":  grid_shape,
        }

    def _rotate_vector(self, vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float64).reshape(3)
        yaw = self._preview_state["yaw"]
        pitch = self._preview_state["pitch"]
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)

        x, y, z = vec
        x1 = x * cy - y * sy
        y1 = x * sy + y * cy
        z1 = z
        y2 = y1 * cp - z1 * sp
        z2 = y1 * sp + z1 * cp
        return np.array([x1, y2, z2], dtype=np.float64)

    def _project_world_point(self, point: np.ndarray, anchor: np.ndarray | None = None) -> np.ndarray:
        point = np.asarray(point, dtype=np.float64).reshape(3)
        if anchor is None:
            anchor = np.zeros(3, dtype=np.float64)
        anchored = point - anchor
        rotated = self._rotate_vector(anchored)
        canvas_w = float(self._preview_canvas.winfo_width()) if self._preview_canvas is not None and self._preview_canvas.winfo_exists() else 200.0
        canvas_h = float(self._preview_canvas.winfo_height()) if self._preview_canvas is not None and self._preview_canvas.winfo_exists() else 200.0
        scale = 320.0 * self._preview_state["zoom"] / max(1.0, float(np.max(np.linalg.norm(np.asarray(self._last_grid_coords, dtype=np.float64), axis=1))) if self._last_grid_coords.size else 1.0)
        x_screen = rotated[0] * scale + canvas_w / 2.0 + self._preview_state["pan_x"]
        y_screen = -rotated[2] * scale + canvas_h / 2.0 + self._preview_state["pan_y"]
        return np.array([x_screen, y_screen], dtype=np.float64)

    def _project_point_to_canvas(self, point: np.ndarray, zoom: float | None = None, anchor: np.ndarray | None = None) -> np.ndarray:
        if self._preview_canvas is None or not self._preview_canvas.winfo_exists():
            return np.zeros(2, dtype=np.float64)

        p = np.asarray(point, dtype=np.float64).reshape(3)
        if anchor is None:
            anchor = self._get_rotation_anchor_for_projection()

        rel = p - anchor
        q = self._rotate_vector(rel) + anchor
        radius = max(1.0, float(np.max(np.linalg.norm(self._last_grid_coords, axis=1))) if self._last_grid_coords.size else 1.0)
        scale = 320.0 * (self._preview_state["zoom"] if zoom is None else zoom) / radius
        canvas_w = float(self._preview_canvas.winfo_width())
        canvas_h = float(self._preview_canvas.winfo_height())
        x_screen = q[0] * scale + canvas_w / 2.0 + self._preview_state["pan_x"]
        y_screen = -q[2] * scale + canvas_h / 2.0 + self._preview_state["pan_y"]
        return np.array([x_screen, y_screen], dtype=np.float64)

    def _preview_project(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points

        points = np.asarray(points, dtype=np.float64)
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.shape[1] == 2:
            points_3d = np.zeros((points.shape[0], 3), dtype=np.float64)
            points_3d[:, :2] = points
            points = points_3d

        anchor = self._get_rotation_anchor_for_projection()
        projected = []
        for p in points:
            projected.append(self._project_point_to_canvas(p, anchor=anchor))
        return np.asarray(projected, dtype=np.float64)

    def _estimate_rotation_anchor(self, screen_x: float, screen_y: float) -> np.ndarray:
        if self._last_grid_coords.size == 0:
            return np.zeros(3, dtype=np.float64)
        points = np.asarray(self._last_grid_coords, dtype=np.float64)
        if points.shape[1] == 2:
            points = np.column_stack([points, np.zeros(points.shape[0], dtype=np.float64)])

        if self._preview_canvas is None or not self._preview_canvas.winfo_exists():
            return points[0].copy()

        center_x = self._preview_canvas.winfo_width() / 2.0
        center_y = self._preview_canvas.winfo_height() / 2.0
        projected = self._preview_project(points)
        if projected.size == 0:
            return np.zeros(3, dtype=np.float64)

        dist = np.hypot(projected[:, 0] - center_x, projected[:, 1] - center_y)
        idx = int(np.argmin(dist))
        return points[idx].copy()

    def _get_rotation_anchor_for_projection(self) -> np.ndarray:
        if (
            self._preview_state.get("dragging")
            and self._preview_state.get("button") == 1
        ):
            return np.asarray(self._preview_state.get("rotation_anchor", np.zeros(3, dtype=np.float64)), dtype=np.float64)
        return np.zeros(3, dtype=np.float64)

    # vertices and face definitions for the small orientation cube,
    # shared by _draw_axis_triad. Faces list opposite-normal pairs
    # consecutively so exactly one of each pair is front-facing.
    _CUBE_VERTS = np.array([
        [-1.0, -1.0, -1.0],
        [ 1.0, -1.0, -1.0],
        [ 1.0,  1.0, -1.0],
        [-1.0,  1.0, -1.0],
        [-1.0, -1.0,  1.0],
        [ 1.0, -1.0,  1.0],
        [ 1.0,  1.0,  1.0],
        [-1.0,  1.0,  1.0],
    ], dtype=np.float64)

    _CUBE_FACES = [
        {"normal": np.array([ 1.0,  0.0,  0.0]), "verts": [1, 2, 6, 5], "color": "#ff9999"},
        {"normal": np.array([-1.0,  0.0,  0.0]), "verts": [0, 3, 7, 4], "color": "#ffcccc"},
        {"normal": np.array([ 0.0,  1.0,  0.0]), "verts": [3, 2, 6, 7], "color": "#99dd99"},
        {"normal": np.array([ 0.0, -1.0,  0.0]), "verts": [0, 1, 5, 4], "color": "#cceecc"},
        {"normal": np.array([ 0.0,  0.0,  1.0]), "verts": [4, 5, 6, 7], "color": "#9999ff"},
        {"normal": np.array([ 0.0,  0.0, -1.0]), "verts": [0, 1, 2, 3], "color": "#ccccff"},
    ]

    def _draw_axis_triad(self, canvas: tk.Canvas) -> None:
        if (not canvas.winfo_exists()
                or canvas.winfo_width() < 2):
            return

        R = self._rotation_matrix()
        corner = np.array(
            [32.0, float(canvas.winfo_height()) - 32.0])
        axis_len = 22.0

        # 3-line axis triad, drawn once from a single origin.
        for axis, color, label in [
            (np.array([1.0, 0.0, 0.0]), "#cc0000", "X"),
            (np.array([0.0, 1.0, 0.0]), "#00aa00", "Y"),
            (np.array([0.0, 0.0, 1.0]), "#0000cc", "Z"),
        ]:
            rot = R @ axis
            end = corner + np.array([rot[0], -rot[2]]) * axis_len
            canvas.create_line(
                corner[0], corner[1], end[0], end[1],
                fill=color, width=2)
            canvas.create_text(
                end[0] + (end[0] - corner[0]) * 0.2,
                end[1] + (end[1] - corner[1]) * 0.2,
                text=label, fill=color,
                font=("Arial", 8, "bold"))

        # Small orientation cube centered on the same origin, sized
        # smaller than the axis lines so it never covers them.
        cube_half = axis_len * 0.32
        verts = self._CUBE_VERTS * cube_half

        # The projection pipeline drops the rotated Y axis as depth
        # (see _project_points); a face is front-facing (visible to
        # the user) when its rotated normal points back toward the
        # camera, i.e. has a negative Y component. Exactly one face
        # of each opposite-normal pair satisfies this, so this always
        # yields 3 mutually perpendicular, front-facing faces.
        faces_by_depth = sorted(
            self._CUBE_FACES,
            key=lambda f: float((R @ f["normal"])[1]))
        visible_faces = faces_by_depth[:3]

        for face in visible_faces:
            pts = []
            for vi in face["verts"]:
                r = R @ verts[vi]
                pts.append((corner[0] + r[0], corner[1] - r[2]))
            canvas.create_polygon(
                pts, fill=face["color"],
                outline="#333333", width=1.3)

    def _refresh_preview(self) -> None:
        canvas = self._preview_canvas
        if canvas is None or not canvas.winfo_exists():
            return
        canvas.delete("all")
        if not self._preview_enabled_var.get():
            canvas.create_text(canvas.winfo_width() / 2, canvas.winfo_height() / 2, text="3D preview disabled", fill="#999999")
            return
        if self._last_grid_coords.size == 0:
            canvas.create_text(canvas.winfo_width() / 2, canvas.winfo_height() / 2, text="Compute mesh to preview", fill="#777777")
            return

        pts = self._last_grid_coords
        shape = tuple(int(v) for v in self._last_grid_shape)
        if int(np.prod(shape, dtype=np.int64)) != pts.shape[0]:
            canvas.create_text(
                canvas.winfo_width() / 2,
                canvas.winfo_height() / 2,
                text="Invalid grid shape — recompute mesh",
                fill="#aa0000",
                font=("Arial", 9, "bold"),
            )
            return

        self._draw_axis_triad(canvas)

        if len(shape) == 2:
            proj = self._project_points(pts)
            if self._show_points_var.get():
                for p, q in zip(pts, proj):
                    marker = self._marker_var.get() or "o"
                    size = float(self._point_size_var.get())
                    x, y = q
                    if marker == "o":
                        canvas.create_oval(x - size, y - size, x + size, y + size, fill=self._point_color_var.get(), outline="")
                    else:
                        canvas.create_line(x - size, y - size, x + size, y + size, fill=self._point_color_var.get(), width=max(1, int(size)))
            return

        if len(shape) != 3:
            return

        grid = pts.reshape(shape[0], shape[1], shape[2], 3)
        order = _AXIS_ORDER[_normalize_indexing(self._indexing_var.get())]
        axis_axes = {
            "x": order.index("x"),
            "y": order.index("y"),
            "z": order.index("z"),
        }

        if self._show_lines_var.get():
            line_color = self._line_color_var.get() or "#444444"
            line_width = max(1.0, float(self._line_width_var.get()))
            dash_map = {"solid": (), "dash": (8, 4), "dot": (2, 4), "dashdot": (6, 3, 2, 3)}
            dash = dash_map.get(self._line_style_var.get(), ())
            for axis_name, enabled in (("x", self._show_x_lines_var.get()), ("y", self._show_y_lines_var.get()), ("z", self._show_z_lines_var.get())):
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
                        p0 = proj[i]
                        p1 = proj[i + 1]
                        canvas.create_line(p0[0], p0[1], p1[0], p1[1], fill=line_color, width=line_width, dash=dash)

        if self._show_points_var.get():
            proj = self._project_points(pts)
            marker = self._marker_var.get() or "o"
            size = float(self._point_size_var.get())
            point_color = self._point_color_var.get() or "#1f77b4"
            for p, q in zip(pts, proj):
                x, y = q
                if marker == "o":
                    canvas.create_oval(x - size, y - size, x + size, y + size, fill=point_color, outline="")
                elif marker == "+":
                    canvas.create_line(x - size, y, x + size, y, fill=point_color, width=max(1, int(size)))
                    canvas.create_line(x, y - size, x, y + size, fill=point_color, width=max(1, int(size)))
                elif marker == "*":
                    canvas.create_line(x - size, y, x + size, y, fill=point_color, width=max(1, int(size)))
                    canvas.create_line(x, y - size, x, y + size, fill=point_color, width=max(1, int(size)))
                    canvas.create_line(x - size, y - size, x + size, y + size, fill=point_color, width=max(1, int(size)))
                    canvas.create_line(x - size, y + size, x + size, y - size, fill=point_color, width=max(1, int(size)))
                elif marker == "x":
                    canvas.create_line(x - size, y - size, x + size, y + size, fill=point_color, width=max(1, int(size)))
                    canvas.create_line(x - size, y + size, x + size, y - size, fill=point_color, width=max(1, int(size)))
                else:
                    canvas.create_oval(x - size, y - size, x + size, y + size, fill=point_color, outline="")

    # ── unified projection pipeline ───────────────────────────────

    def _get_scale(self) -> float:
        """Scale factor: maps world units to screen pixels."""
        if self._last_grid_coords.size == 0:
            return 1.0
        pts = np.asarray(
            self._last_grid_coords, dtype=np.float64)
        if pts.shape[1] == 2:
            pts = np.column_stack(
                [pts, np.zeros(pts.shape[0])])
        radius = float(
            np.max(np.linalg.norm(pts, axis=1)))
        radius = max(radius, 1e-6)
        canvas_min = 200.0
        if (self._preview_canvas is not None
                and self._preview_canvas.winfo_exists()):
            canvas_min = min(
                float(self._preview_canvas.winfo_width()),
                float(self._preview_canvas.winfo_height()))
        return (canvas_min * 0.38
                * self._preview_state["zoom"]
                / radius)

    def _rotation_matrix(self) -> np.ndarray:
        """
        3x3 rotation matrix from current yaw and pitch.
        Yaw rotates around Z axis (world up).
        Pitch rotates around the rotated X axis.
        """
        yaw   = self._preview_state["yaw"]
        pitch = self._preview_state["pitch"]
        cy, sy = np.cos(yaw),   np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        # Ryaw: rotation around world Z
        Ryaw = np.array([
            [ cy, -sy,  0],
            [ sy,  cy,  0],
            [  0,   0,  1]], dtype=np.float64)
        # Rpitch: rotation around local X (after yaw)
        Rpitch = np.array([
            [1,   0,   0],
            [0,  cp, -sp],
            [0,  sp,  cp]], dtype=np.float64)
        return Rpitch @ Ryaw

    def _project_points(
            self,
            pts: np.ndarray) -> np.ndarray:
        """
        Project (N,3) world points to (N,2) screen pixels.

        Pipeline (all in one place):
        1. subtract camera_target  (world-space pan)
        2. rotate
        3. scale
        4. map to screen:  x→right, z→up, y→depth
        5. add screen centre
        """
        if (self._preview_canvas is None
                or not self._preview_canvas.winfo_exists()):
            return np.zeros((len(pts), 2))

        pts = np.asarray(pts, dtype=np.float64)
        if pts.ndim == 1:
            pts = pts.reshape(1, -1)
        if pts.shape[1] == 2:
            pts = np.column_stack(
                [pts, np.zeros(pts.shape[0])])

        # step 1: world-space pan
        target = np.asarray(
            self._preview_state["camera_target"],
            dtype=np.float64)
        rel = pts - target

        # step 2: rotate
        R = self._rotation_matrix()
        rot = (R @ rel.T).T   # shape (N, 3)

        # step 3 & 4: scale and project
        #   screen x = rotated x
        #   screen y = -rotated z  (z is up in world)
        scale = self._get_scale()
        sx = rot[:, 0] * scale
        sy = -rot[:, 2] * scale

        # step 5: screen centre
        cw = float(self._preview_canvas.winfo_width())
        ch = float(self._preview_canvas.winfo_height())
        sx += cw / 2.0
        sy += ch / 2.0

        return np.column_stack([sx, sy])

    def _screen_to_world_delta(
            self,
            dx_screen: float,
            dy_screen: float) -> np.ndarray:
        """
        Convert a screen-space drag delta to a world-space
        translation of camera_target.

        We invert the rotation for the x and z world axes
        (the two axes that map to screen x and screen y).
        """
        scale = self._get_scale()
        if scale < 1e-12:
            return np.zeros(3)

        # screen x maps to rotated world x axis
        # screen y maps to -rotated world z axis
        # invert rotation: R^T = R^-1 for orthogonal R
        R = self._rotation_matrix()
        Rt = R.T

        # unit screen-x in world = first column of R^T
        # unit screen-y in world = -(third column of R^T)
        world_dx = (Rt[:, 0] * dx_screen
                    - Rt[:, 2] * dy_screen) / scale
        return world_dx

    # ── drag / pan / zoom interactions ───────────────────────────

    def _on_preview_drag_start(self, event) -> None:
        self._preview_state["dragging"] = True
        self._preview_state["last_x"]   = event.x
        self._preview_state["last_y"]   = event.y
        self._preview_state["button"]   = 1

    def _on_preview_drag_motion(self, event) -> None:
        if (not self._preview_state["dragging"]
                or self._preview_state["button"] != 1):
            return
        dx = event.x - self._preview_state["last_x"]
        dy = event.y - self._preview_state["last_y"]
        self._preview_state["yaw"]   += dx * 0.01
        self._preview_state["pitch"] += dy * 0.01
        self._preview_state["last_x"] = event.x
        self._preview_state["last_y"] = event.y
        self._refresh_preview()

    def _on_preview_drag_release(self, _event) -> None:
        self._preview_state["dragging"] = False
        self._preview_state["button"]   = None

    def _on_preview_pan_start(self, event) -> None:
        self._preview_state["dragging"] = True
        self._preview_state["last_x"]   = event.x
        self._preview_state["last_y"]   = event.y
        self._preview_state["button"]   = 3

    def _on_preview_pan_motion(self, event) -> None:
        if (not self._preview_state["dragging"]
                or self._preview_state["button"] != 3):
            return
        dx = event.x - self._preview_state["last_x"]
        dy = event.y - self._preview_state["last_y"]
        # convert screen delta to world-space camera_target shift
        world_delta = self._screen_to_world_delta(dx, dy)
        target = np.asarray(
            self._preview_state["camera_target"],
            dtype=np.float64)
        # panning moves target in the OPPOSITE direction
        self._preview_state["camera_target"] = \
            (target - world_delta).tolist()
        self._preview_state["last_x"] = event.x
        self._preview_state["last_y"] = event.y
        self._refresh_preview()

    def _on_preview_pan_release(self, _event) -> None:
        self._preview_state["dragging"] = False
        self._preview_state["button"]   = None

    def _on_preview_mousewheel(self, event) -> str:
        if (self._preview_canvas is None
                or not self._preview_canvas.winfo_exists()):
            return "break"
        delta = int(event.delta / 120)
        if delta == 0:
            return "break"
        factor = 1.12 ** delta
        self._preview_state["zoom"] = max(
            0.05,
            self._preview_state["zoom"] * factor)
        self._refresh_preview()
        return "break"

    # ── serialization ─────────────────────────────────────────────

    def get_params(self) -> dict:
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
        }

    def set_params(self, params: dict) -> None:
        self._init_state()
        self._x_var.set(str(params.get("x", "-8:1:8")))
        self._y_var.set(str(params.get("y", "-6:1:6")))
        self._z_var.set(str(params.get("z", "-4:1:4")))
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

    def close_inspector(self) -> None:
        super().close_inspector()
        self._x_entry = None
        self._y_entry = None
        self._z_entry = None

    def on_destroy(self) -> None:
        if (self._help_popup is not None
                and self._help_popup.winfo_exists()):
            self._help_popup.destroy()
        super().on_destroy()