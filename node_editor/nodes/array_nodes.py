# node_editor/nodes/array_nodes.py

import tkinter as tk
from tkinter import messagebox, ttk
from pathlib import Path
import numpy as np

from node_editor.base_node import BaseNode
from node_editor import recent_dirs
from node_editor.pin_types import PinSchema, PinDef, PinType
from node_editor.execution import ExecutionMode
from node_editor.tooltip import add_tooltip
from node_editor.project_context import get_project_directory
from node_editor import project_data


# ── NumPy file load / save and dtype conversion (Array Input, Array Viewer) ──

DTYPE_CHOICES = ["float64", "float32", "float16", "int64", "int32", "int16", "int8",
                 "uint64", "uint32", "uint16", "uint8", "bool"]
NUMPY_FILETYPES = [("NumPy files", "*.npy *.npz"), ("NumPy array", "*.npy"),
                   ("NumPy archive", "*.npz"), ("All files", "*.*")]

DTYPE_TIP = ("Number type of the output array -- choose another one to convert the data. "
             "float64 = normal decimal numbers; float32 = half the memory, needed by some "
             "OpenCV functions; int / uint = whole numbers (uint8 for 8-bit images); bool = "
             "0 / 1. Float to integer drops the fraction (2.7 -> 2), like NumPy; values outside "
             "the integer range are clipped to it (e.g. 300 -> 255 for uint8) and NaN becomes 0.")


def convert_dtype(arr: np.ndarray, dtype_name: str) -> tuple[np.ndarray, str]:
    """`arr` as `dtype_name`, plus a short note on what the conversion did
    (empty if nothing worth reporting). Integer targets: NaN / inf -> 0 and
    out-of-range values clipped to the type's range (NumPy would wrap or give
    undefined values); the fraction is dropped as with NumPy's astype."""
    target = np.dtype(dtype_name)
    arr = np.asarray(arr)
    if arr.dtype == target:
        return arr, ""
    if target.kind not in "iu" or arr.dtype.kind not in "fciub":
        return arr.astype(target), ""
    src = np.real(arr) if arr.dtype.kind == "c" else arr
    notes = []
    if src.dtype.kind == "f":
        bad = ~np.isfinite(src)
        if bad.any():
            notes.append(f"{int(bad.sum())} NaN/inf -> 0")
            src = np.where(bad, 0, src)
    info = np.iinfo(target)
    low, high = src < info.min, src > info.max
    n_clip = int(low.sum() + high.sum())
    if n_clip:
        notes.append(f"{n_clip} value(s) clipped to {info.min}..{info.max}")
        src = np.clip(src, info.min, info.max)
    return src.astype(target), "; ".join(notes)


def ask_load_numpy(parent, title: str = "Load a NumPy array",
                   owner=None) -> tuple[np.ndarray, str] | None:
    """File dialog + np.load for .npy / .npz. For an .npz with several arrays
    the user picks one. Returns (array, description) or None (cancelled /
    failed -- the error is shown). ``owner`` (a node) remembers the folder."""
    path = recent_dirs.ask_open_filename(owner, parent=parent, title=title, filetypes=NUMPY_FILETYPES)
    if not path:
        return None
    try:
        loaded = np.load(path, allow_pickle=False)
        if isinstance(loaded, np.lib.npyio.NpzFile):            # .npz: one or more named arrays
            with loaded as data:
                keys = list(data.files)
                if not keys:
                    raise ValueError("the .npz file contains no arrays")
                key = keys[0] if len(keys) == 1 else _ask_npz_key(parent, path, data)
                if key is None:
                    return None
                arr = np.array(data[key])
                desc = f"{Path(path).name} [{key}]"
        else:                                                   # .npy: one array
            arr, desc = np.asarray(loaded), Path(path).name
    except Exception as e:
        messagebox.showerror(title, f"Cannot load the file:\n{path}\n\n{e}", parent=parent)
        return None
    if arr.dtype.kind not in "biufc":
        messagebox.showerror(title, f"Only numeric arrays are supported (this one is {arr.dtype}).",
                             parent=parent)
        return None
    return arr, desc


def _ask_npz_key(parent, path: str, data) -> str | None:
    """Modal list of the arrays in an .npz file; returns the chosen key."""
    win = tk.Toplevel(parent)
    win.title("Choose an array")
    win.transient(parent)
    tk.Label(win, text=f"{Path(path).name} contains {len(data.files)} arrays -- choose one:",
             font=("Arial", 9), anchor="w").pack(fill="x", padx=8, pady=(8, 4))
    lb = tk.Listbox(win, width=60, height=min(12, len(data.files)), font=("Courier", 9))
    lb.pack(fill="both", expand=True, padx=8)
    for key in data.files:
        a = data[key]
        lb.insert(tk.END, f"{key:<24} {str(a.shape):<18} {a.dtype}")
    lb.selection_set(0)
    chosen = {"key": None}

    def ok(_e=None):
        sel = lb.curselection()
        if sel:
            chosen["key"] = data.files[sel[0]]
        win.destroy()
    btns = tk.Frame(win)
    btns.pack(fill="x", padx=8, pady=8)
    tk.Button(btns, text="Load", width=10, command=ok).pack(side="right")
    tk.Button(btns, text="Cancel", width=10, command=win.destroy).pack(side="right", padx=(0, 6))
    lb.bind("<Double-Button-1>", ok)
    win.bind("<Return>", ok)
    win.bind("<Escape>", lambda _e: win.destroy())
    win.grab_set()
    lb.focus_set()
    win.wait_window()
    return chosen["key"]


def ask_save_numpy(parent, arr: np.ndarray, title: str = "Save the array",
                   owner=None) -> str | None:
    """File dialog + save as .npy (or .npz, key 'array', compressed), written
    to a temporary file first so a failed save never leaves a broken file.
    Returns the path written, or None."""
    path = recent_dirs.ask_save_as_filename(owner, parent=parent, title=title, defaultextension=".npy",
                                            filetypes=NUMPY_FILETYPES[1:3] + NUMPY_FILETYPES[3:])
    if not path:
        return None
    target = Path(path)
    tmp = target.with_name(target.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            if target.suffix.lower() == ".npz":
                np.savez_compressed(f, array=np.asarray(arr))
            else:
                np.save(f, np.asarray(arr), allow_pickle=False)
        tmp.replace(target)
    except Exception as e:
        try:
            tmp.unlink()
        except OSError:
            pass
        messagebox.showerror(title, f"Cannot save the file:\n{path}\n\n{e}", parent=parent)
        return None
    return str(target)


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

    _DTYPES = DTYPE_CHOICES
    # A loaded array whose text would be longer than this stays an array
    # (read-only preview): one project cell holds at most 32767 characters,
    # and a huge text box is useless to edit anyway.
    _TEXT_MAX_CHARS = 20000

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
        self._load_info_var = tk.StringVar(value="")
        self._csv_text = "1, 2, 3\n4, 5, 6"
        self._text = None
        self._shape_entry = None
        self._discard_btn = None
        self._last_array: np.ndarray | None = None
        self._last_note = ""
        # A loaded array too large for the text box: the data source instead
        # of the text (saved inside the project file, its Data sheet).
        self._loaded: np.ndarray | None = None
        self._loaded_desc = ""

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
        add_tooltip(dtype_cb, DTYPE_TIP)

        shape_frame = tk.LabelFrame(parent, text="Array shape", font=("Arial", 9), padx=6, pady=6)
        shape_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 4))

        auto_rb = tk.Radiobutton(shape_frame, text="Auto reshape", variable=self._shape_mode_var,
                                 value="auto", command=self._refresh_shape_state, anchor="w")
        auto_rb.pack(anchor="w")
        manual_rb = tk.Radiobutton(shape_frame, text="User defined shape", variable=self._shape_mode_var,
                                   value="manual", command=self._refresh_shape_state, anchor="w")
        manual_rb.pack(anchor="w")
        add_tooltip(auto_rb, "The shape follows the text: one row per line, one column per "
                             "value; the result is always 2-D (one line = a 1 x N array).")
        add_tooltip(manual_rb, "Reshape the typed numbers to the shape below, e.g. 3,3 for a "
                               "3x3 matrix or 2,4,4 for a 3-D array. The number of values must "
                               "match; -1 lets one dimension be computed.")

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

        status_lbl = tk.Label(parent, textvariable=self._status_var, font=("Arial", 9), fg="#553300",
                              justify="left")
        status_lbl.grid(row=4, column=0, sticky="w", pady=(8, 2))

        # Load / save NumPy files (Save also works as a dtype converter).
        file_row = tk.Frame(parent)
        file_row.grid(row=5, column=0, columnspan=2, sticky="w", pady=(6, 0))
        load_btn = tk.Button(file_row, text="Load ...", font=("Arial", 9), command=self._on_load)
        load_btn.pack(side="left")
        save_btn = tk.Button(file_row, text="Save ...", font=("Arial", 9), command=self._on_save)
        save_btn.pack(side="left", padx=(6, 0))
        self._discard_btn = tk.Button(file_row, text="Discard loaded", font=("Arial", 9),
                                      command=self._on_discard_loaded)
        self._discard_btn.pack(side="left", padx=(6, 0))
        add_tooltip(load_btn, "Load an array from a NumPy file (.npy, or one array of an .npz). "
                              "Its values go into the text box, the dtype menu is set to its "
                              "dtype and its shape goes into the shape box. A very large array "
                              "is kept as an array (read-only preview).")
        add_tooltip(save_btn, "Save the array this node outputs -- the values converted to the "
                              "dtype chosen above -- as .npy or .npz. With Load, this converts "
                              "a file from one dtype to another.")
        add_tooltip(self._discard_btn, "Forget the large loaded array and edit the text box again.")
        tk.Label(parent, textvariable=self._load_info_var, font=("Arial", 8), fg="#334455",
                 anchor="w", justify="left", wraplength=360).grid(
            row=6, column=0, columnspan=2, sticky="w")

        parent.grid_rowconfigure(3, weight=1)
        parent.grid_columnconfigure(1, weight=1)
        self._show_text_or_preview()
        self._refresh_shape_state()

    def close_inspector(self) -> None:
        if self._loaded is None and self._text is not None and self._text.winfo_exists():
            self._csv_text = self._text.get("1.0", tk.END)
        super().close_inspector()
        self._text = None
        self._shape_entry = None
        self._discard_btn = None

    # ── NumPy file load / save ────────────────────────────────────

    def _dlg_parent(self):
        win = getattr(self, "_inspector_win", None)
        try:
            if win is not None and win.winfo_exists():
                return win
        except tk.TclError:
            pass
        return self.canvas.winfo_toplevel()

    @staticmethod
    def _array_to_text(arr: np.ndarray, limit: int) -> str | None:
        """Rows of the array (last axis = one line) as comma-separated text
        that parses back exactly (shortest round-trip floats); None if the
        text would be longer than `limit` characters."""
        a = np.asarray(arr)
        a = a.reshape(1) if a.ndim == 0 else a
        if a.size * 4 > limit:                        # cannot fit, skip formatting
            return None
        rows = a.reshape(1, -1) if a.ndim == 1 else a.reshape(-1, a.shape[-1])
        if a.dtype.kind == "f":
            fmt = lambda v: repr(float(v))            # noqa: E731
        elif a.dtype.kind == "c":
            fmt = lambda v: repr(float(v.real))       # noqa: E731
        else:
            fmt = lambda v: str(int(v))               # noqa: E731  (bool -> 0 / 1)
        text = "\n".join(", ".join(fmt(v) for v in row) for row in rows)
        return text if len(text) <= limit else None

    def _show_text_or_preview(self) -> None:
        """Text box: editable CSV text, or (large loaded array) a read-only
        preview of its first rows."""
        t = self._text
        if t is None or not t.winfo_exists():
            return
        t.configure(state="normal")
        t.delete("1.0", tk.END)
        if self._loaded is None:
            t.insert("1.0", self._csv_text)
            t.configure(bg="#ffffff")
        else:
            a = self._loaded.reshape(1) if self._loaded.ndim == 0 else self._loaded
            rows = a.reshape(1, -1) if a.ndim == 1 else a.reshape(-1, a.shape[-1])
            lines = [", ".join(f"{v:.6g}" for v in row[:12]) + (", ..." if row.size > 12 else "")
                     for row in rows[:40]]
            more = f"\n... ({rows.shape[0] - 40} more rows)" if rows.shape[0] > 40 else ""
            t.insert("1.0", f"# Loaded array, too large to edit as text: shape {self._loaded.shape}, "
                            f"{self._loaded.size} values.\n# Read-only preview of the first rows:\n"
                     + "\n".join(lines) + more)
            t.configure(state="disabled", bg="#f4f4f4")
        if self._discard_btn is not None and self._discard_btn.winfo_exists():
            self._discard_btn.configure(state=tk.NORMAL if self._loaded is not None else tk.DISABLED)

    def _on_load(self) -> None:
        res = ask_load_numpy(self._dlg_parent(), owner=self)
        if res is None:
            return
        arr, desc = res
        if arr.ndim == 0:
            arr = arr.reshape(1)
        note = ""
        if arr.dtype.name in self._DTYPES:
            self._dtype_var.set(arr.dtype.name)
        else:                                          # e.g. complex: offer float64
            self._dtype_var.set("float64")
            note = f" (dtype {arr.dtype} is not in the menu: float64 is selected"\
                   + (", real part only" if arr.dtype.kind == "c" else "") + ")"
        text = self._array_to_text(arr, self._TEXT_MAX_CHARS)
        if text is not None:
            self._loaded, self._loaded_desc = None, ""
            self._csv_text = text
        else:
            self._loaded, self._loaded_desc = arr, desc
        self._show_text_or_preview()
        # The loaded shape goes into the (selected) shape box.
        self._shape_mode_var.set("manual")
        self._shape_entry_var.set(", ".join(str(d) for d in arr.shape))
        if self._shape_entry is not None and self._shape_entry.winfo_exists():
            self._shape_entry.configure(state="normal")
            self._shape_entry.focus_set()
            self._shape_entry.select_range(0, tk.END)
            self._shape_entry.icursor(tk.END)
        kept = "  -- kept as an array (too large for the text box)" if text is None else ""
        self._load_info_var.set(f"Loaded {desc}: shape {arr.shape}, {arr.dtype}{kept}{note}")
        self._status_var.set("loaded -- press Apply to send")
        self._refresh_shape_state()

    def _on_discard_loaded(self) -> None:
        if self._loaded is None:
            return
        self._loaded, self._loaded_desc = None, ""
        self._load_info_var.set("Loaded array discarded; the text box is used again.")
        self._show_text_or_preview()
        self._refresh_shape_state()

    def _on_save(self) -> None:
        arr = self._parse_csv()
        if arr is None:
            messagebox.showwarning("Save array", "The array is not valid yet (see the shape / "
                                                 "text box).", parent=self._dlg_parent())
            return
        path = ask_save_numpy(self._dlg_parent(), arr, owner=self)
        if path:
            note = f"\n{self._last_note}" if self._last_note else ""
            self._load_info_var.set(f"Saved {Path(path).name}: shape {arr.shape}, {arr.dtype}{note}")

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

    @staticmethod
    def _resolve_shape(shape, n: int):
        """`shape` (may contain one -1) for n values -> (resolved, None) or
        (None, error text)."""
        if -1 in shape:
            known = int(np.prod([d for d in shape if d != -1])) if len(shape) > 1 else 1
            if known <= 0 or n % known != 0:
                return None, f"Shape mismatch: {shape} does not fit {n} values"
            return tuple(n // known if d == -1 else d for d in shape), None
        if int(np.prod(shape)) != n:
            return None, f"Shape mismatch: {shape} does not fit {n} values"
        return tuple(shape), None

    def _refresh_shape_state(self) -> None:
        if self._shape_entry is None:
            return

        if self._loaded is not None:                   # data = the loaded array
            if self._shape_mode_var.get() == "manual":
                self._shape_entry.configure(state="normal")
                desired = self._parse_shape_text(self._shape_entry_var.get())
                if desired is None:
                    self._shape_display_var.set("Enter a shape like 2,3 or 2,-1")
                    self._shape_entry.configure(bg="#ffe6e6")
                    return
                resolved, error = self._resolve_shape(desired, self._loaded.size)
                self._shape_display_var.set(error or f"Valid reshape: {resolved}")
                self._shape_entry.configure(bg="#ffe6e6" if error else "#ffffff")
            else:
                self._shape_entry.configure(state="disabled", bg="#f2f2f2")
                self._shape_display_var.set(f"Loaded shape: {self._loaded.shape}")
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
        self._last_note = ""
        if self._loaded is not None:                   # large loaded array
            data = self._loaded
        else:
            rows, error = self._parse_numeric_rows(self._get_csv_text())
            if rows is None:
                return None
            # Auto: the natural 2-D shape of the text (a single line = 1 x N).
            data = np.array(rows, dtype=np.float64)

        if self._shape_mode_var.get() == "manual":
            shape = self._parse_shape_text(self._shape_entry_var.get())
            if shape is None:
                return None
            shape, error = self._resolve_shape(shape, data.size)
            if shape is None:
                return None
            data = data.reshape(shape)

        # float64 parse first, then one conversion with the same rules as
        # Save: integer types are clipped to their range, NaN -> 0.
        out, self._last_note = convert_dtype(data, self._dtype_var.get())
        return out

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
        if self._last_note:
            self._load_info_var.set(f"Conversion to {arr.dtype}: {self._last_note}")
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
        if self._loaded is None and self._text is not None and self._text.winfo_exists():
            self._csv_text = self._text.get("1.0", tk.END)
        params = {
            "csv": self._csv_text,
            "dtype": self._dtype_var.get(),
            "shape_mode": self._shape_mode_var.get(),
            "shape_text": self._shape_entry_var.get(),
        }
        if self._loaded is not None:
            # A large loaded array is saved inside the project file (Data sheet).
            params.update(loaded_data=project_data.put_array(self, "loaded", self._loaded),
                          loaded_desc=self._loaded_desc)
        return params

    def set_params(self, params: dict) -> None:
        self._init_state()
        self._csv_text = params.get("csv", self._csv_text)
        self._dtype_var.set(params.get("dtype", "float64"))
        self._shape_mode_var.set(params.get("shape_mode", "auto"))
        self._shape_entry_var.set(params.get("shape_text", ""))
        self._loaded, self._loaded_desc = None, ""
        name = str(params.get("loaded_file", "") or "")            # projects saved before the Data sheet
        project_dir = get_project_directory()
        if params.get("loaded_data") is not None:
            self._loaded = project_data.get_array(params["loaded_data"])
            self._loaded_desc = str(params.get("loaded_desc", ""))
            if self._loaded is not None:
                self._load_info_var.set(f"Loaded array {self._loaded_desc}: shape {self._loaded.shape}")
            else:
                self._load_info_var.set("The saved array is missing from the project file; the text box is used.")
        elif name and project_dir is not None and Path(name).name == name:
            try:
                self._loaded = np.load(project_dir / name, allow_pickle=False)
                self._loaded_desc = str(params.get("loaded_desc", ""))
                self._load_info_var.set(f"Loaded array {self._loaded_desc}: shape {self._loaded.shape}")
            except Exception as e:
                self._load_info_var.set(f"The saved array {name} could not be loaded ({e}); "
                                        "the text box is used.")
        self._last_array = None                       # recompute from the restored settings
        self._show_text_or_preview()
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
        # Output dtype: "(as input)" keeps the data's own dtype; another
        # choice converts the output and what Save writes.
        self._dtype_out_var = tk.StringVar(value=self._AS_INPUT)
        self._dtype_out_var.trace_add("write", lambda *_a: self._on_dtype_changed())
        self._file_info_var = tk.StringVar(value="")
        self._last_note = ""
        # Data source: "pin" (the input pin) or "file" (an array loaded with
        # Load ...). Both arrays are kept, so switching shows the other again;
        # in "file" mode the input pin's data is ignored.
        self._source_var = tk.StringVar(value="pin")
        self._file_array: np.ndarray | None = None
        self._file_desc = ""
        self._last_input: np.ndarray | None = None

    _AS_INPUT = "(as input)"

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
        out = self._compute_view(inputs)
        if "array" in out:
            out["array"] = self._convert_output(out["array"])
        return out

    def _convert_output(self, arr: np.ndarray) -> np.ndarray:
        """Apply the chosen output dtype ('(as input)' = unchanged)."""
        name = self._dtype_out_var.get()
        self._last_note = ""
        if name == self._AS_INPUT:
            return arr
        out, self._last_note = convert_dtype(arr, name)
        if self._last_note:
            self._file_info_var.set(f"Output as {name}: {self._last_note}")
        return out

    def _output_array(self) -> np.ndarray | None:
        """What compute() outputs for the current array (for Save)."""
        arr = self._current_array
        if arr is None:
            return None
        out = arr.copy() if arr.ndim <= 1 else self._get_selected_array(arr)
        return self._convert_output(out)

    def _on_dtype_changed(self) -> None:
        if getattr(self, "_loading_params", False):
            return
        if self.is_inspector_open():
            self._refresh_popup()
        if self._request_downstream:                  # the output changed
            self._request_downstream(self.node_id)

    def _dlg_parent(self):
        win = getattr(self, "_inspector_win", None)
        try:
            if win is not None and win.winfo_exists():
                return win
        except tk.TclError:
            pass
        return self.canvas.winfo_toplevel()

    def _on_load_file(self) -> bool:
        """Load ... : read a file, select 'From file', show and send it.
        Returns False if nothing was loaded."""
        res = ask_load_numpy(self._dlg_parent(), owner=self)
        if res is None:
            return False
        arr, desc = res
        self._file_array = arr.reshape(1) if arr.ndim == 0 else arr
        self._file_desc = desc
        self._source_var.set("file")
        self._current_array = self._file_array
        # Whole array selected; its own shape in the (selected) reshape box,
        # so the output is the loaded array as it is.
        self._rows_var.set(":")
        self._cols_var.set(":")
        self._reshape_var.set(", ".join(str(d) for d in self._file_array.shape))
        entry = self._reshape_entry
        if entry is not None and entry.winfo_exists():
            entry.focus_set()
            entry.select_range(0, tk.END)
            entry.icursor(tk.END)
        self._file_info_var.set(f"From file: {desc}   shape {self._file_array.shape}, "
                                f"{self._file_array.dtype}")
        self._push_view()
        return True

    def _on_source_selected(self) -> None:
        """Radio button 'From input pin' / 'From file' clicked."""
        if self._source_var.get() == "file":
            if self._file_array is None and not self._on_load_file():
                self._source_var.set("pin")           # no file chosen: stay on the pin
                return
            self._file_info_var.set(f"From file: {self._file_desc}   (input pin ignored)")
        else:
            self._file_info_var.set("From input pin" + ("" if self._last_input is not None
                                                          else "   (no array received yet)"))
        self._push_view()

    def _push_view(self) -> None:
        """Show the array of the selected source and send it on."""
        arr = self._file_array if self._source_var.get() == "file" else self._last_input
        self._current_array = arr
        if arr is not None:
            self._update_summary(arr)
        else:
            self._shape_var.set("no data")
        self._refresh_popup()
        if self._request_downstream:
            self._request_downstream(self.node_id)

    def _on_save_file(self) -> None:
        try:
            arr = self._output_array()
        except Exception as e:
            messagebox.showwarning("Save array", f"The output is not valid: {e}", parent=self._dlg_parent())
            return
        if arr is None:
            messagebox.showinfo("Save array", "There is no array to save yet.", parent=self._dlg_parent())
            return
        path = ask_save_numpy(self._dlg_parent(), arr, owner=self)
        if path:
            note = f"   ({self._last_note})" if self._last_note else ""
            self._file_info_var.set(f"Saved {Path(path).name}: shape {arr.shape}, {arr.dtype}{note}")

    def _compute_view(self, inputs: dict) -> dict:
        if self._skip_next_input_update:
            self._skip_next_input_update = False
            raw = None
        else:
            raw = inputs.get("array")
        if raw is not None:
            # The pin's array is always remembered (shown again when "From
            # input pin" is chosen), but used only in that mode.
            arr_in = (raw.copy() if raw.flags.writeable else raw) if isinstance(raw, np.ndarray) \
                else self._coerce_trigger_or_scalar_to_array(raw)
            if arr_in is not None:
                self._last_input = arr_in
                if self._source_var.get() == "file" and self._file_array is None:
                    self._source_var.set("pin")       # nothing loaded: follow the input
        if self._source_var.get() == "file":
            arr = self._file_array
            if raw is not None:
                self._file_info_var.set(f"From file: {self._file_desc}   (input pin ignored)")
        else:
            arr = self._last_input
        if arr is None:
            self._shape_var.set("no data")
            self._minmax_var.set("")
            self._current_array = None
            if self.is_inspector_open():
                self._refresh_popup()
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

        # Load / save NumPy files and the output dtype (a dtype converter).
        save_btn = tk.Button(top, text="Save ...", font=("Arial", 8), command=self._on_save_file)
        save_btn.pack(side="right", padx=(4, 8))
        load_btn = tk.Button(top, text="Load ...", font=("Arial", 8), command=self._on_load_file)
        load_btn.pack(side="right", padx=(4, 0))
        # Data source, left of Load / Save (packed right to left).
        file_rb = tk.Radiobutton(top, text="From file", value="file", variable=self._source_var,
                                 bg="#dde8f0", font=("Arial", 8), command=self._on_source_selected)
        file_rb.pack(side="right")
        pin_rb = tk.Radiobutton(top, text="From input pin", value="pin", variable=self._source_var,
                                bg="#dde8f0", font=("Arial", 8), command=self._on_source_selected)
        pin_rb.pack(side="right")
        src_lbl = tk.Label(top, text="Data source:", bg="#dde8f0", font=("Arial", 8, "bold"))
        src_lbl.pack(side="right", padx=(12, 2))
        add_tooltip(src_lbl, "Where the array shown (and sent on) comes from: the input pin, or "
                             "a NumPy file loaded with 'Load ...'.")
        add_tooltip(pin_rb, "Show and send on the array arriving at the input pin (the usual "
                            "use). Selected automatically when an array arrives and no file has "
                            "been loaded.")
        add_tooltip(file_rb, "Show and send on the array loaded from a file. While this is "
                             "selected the input pin's data is IGNORED, so the file stays even if "
                             "the input changes. Selected automatically by 'Load ...'; choosing it "
                             "with no file loaded yet opens the file dialog.")
        tk.Frame(top, width=1, bg="#9fb3c8").pack(side="right", fill="y", padx=(6, 0), pady=2)
        dtype_cb = ttk.Combobox(top, textvariable=self._dtype_out_var, state="readonly", width=11,
                                values=[self._AS_INPUT] + DTYPE_CHOICES)
        dtype_cb.pack(side="right", padx=(2, 4))
        dtype_lbl = tk.Label(top, text="output dtype:", bg="#dde8f0", font=("Arial", 8))
        dtype_lbl.pack(side="right")
        add_tooltip(load_btn, "Load an array from a NumPy file (.npy, or one array of an .npz), "
                              "show it and send it on; selects 'From file'. rows / cols are reset "
                              "to ':' and its shape goes into the reshape box.")
        add_tooltip(save_btn, "Save the array this node outputs (selected rows / cols, reshaped, "
                              "converted to the output dtype) as .npy or .npz.")
        for widget in (dtype_lbl, dtype_cb):
            add_tooltip(widget, "'(as input)' keeps the data's own dtype. " + DTYPE_TIP)
        tk.Label(win, textvariable=self._file_info_var, font=("Arial", 8), fg="#334455",
                 anchor="w", justify="left").pack(fill="x", padx=8)

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
        if self._popup_info is None or not self._popup_info.winfo_exists():
            return
        if arr is None:                               # e.g. 'From input pin' before any input
            self._popup_info.config(text="no data")
            self._stats_var.set("")
            if self._table_canvas is not None and self._table_canvas.winfo_exists():
                self._table_canvas.delete("all")
            if self._page_lbl is not None and self._page_lbl.winfo_exists():
                self._page_lbl.config(text="")
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
            "output_dtype": self._dtype_out_var.get(),
            # The side file holds the array shown, i.e. the file's array when
            # "From file" is selected (restored as such by set_params).
            "source": self._source_var.get(),
            "file_desc": self._file_desc,
        }
        if arr is not None:
            # Stored inside the project file (its Data sheet), exactly.
            params["array_data"] = project_data.put_array(self, "array", np.asarray(arr))
        return params

    def set_params(self, params: dict) -> None:
        self._rows_var.set(str(params.get("rows_text", ":")))
        self._cols_var.set(str(params.get("cols_text", ":")))
        self._reshape_var.set(str(params.get("reshape_text", "")))
        dtype_out = str(params.get("output_dtype", self._AS_INPUT))
        self._loading_params = True                   # no downstream push while the project loads
        try:
            self._dtype_out_var.set(dtype_out if dtype_out in DTYPE_CHOICES else self._AS_INPUT)
        finally:
            self._loading_params = False

        sidecar_filename = str(params.get("array_viewer_file", "")).strip()    # before the Data sheet
        if params.get("array_data") is not None:
            raw_data = project_data.get_array(params["array_data"])
            sidecar_filename = "data"                     # restored data: do not take the next input over it
        elif sidecar_filename:
            raw_data = self._load_project_sidecar(sidecar_filename)
        else:
            raw_data = params.get("data")
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
        # Put the restored array back where it came from.
        if str(params.get("source", "pin")) == "file":
            self._source_var.set("file")
            self._file_array, self._file_desc = self._current_array, str(params.get("file_desc", ""))
            self._file_info_var.set(f"From file: {self._file_desc}   (input pin ignored)")
        else:
            self._source_var.set("pin")
            self._last_input = self._current_array
        self._update_summary(self._current_array)
        if self.is_inspector_open():
            self._refresh_popup()

# node_editor/nodes/array_nodes.py — add after ArrayViewerNode

import os


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
        path = self.ask_open_filename(
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

