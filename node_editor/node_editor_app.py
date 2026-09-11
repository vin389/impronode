# node_editor_app.py
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from pathlib import Path
import time
from typing import Type

from node_editor.base_node        import BaseNode
from node_editor.pin_types        import PinType, pins_compatible
from node_editor.execution        import ExecutionMode
from node_editor.data_flow_engine import DataFlowEngine
from node_editor.project_io       import load_project, save_project, ProjectFormatError
from node_editor.project_context  import set_project_file_path, get_project_directory


class NodeEditorApp:
    """
        Responsibilities:
            - Canvas / Toolbox UI
            - Mouse event routing (dragging nodes, drawing links)
            - Drawing pins and validating connection types
            - Creating / deleting nodes and delegating data-flow management to DataFlowEngine
    """

    PIN_RADIUS   = 8
    PIN_IN_COLOR  = "#336699"
    PIN_OUT_COLOR = "#339966"
    PIN_HOVER_COLOR = "#ffcc33"
    PIN_HOVER_MS = 160
    PIN_HIT_PAD = 12
    PIN_ERR_COLOR = "#cc4444"
    LINK_HOVER_COLOR = "#ff8800"
    LINK_HOVER_MS = 220
    LINK_HIT_PAD = 6
    SCROLL_UNITS  = 25
    DEFAULT_CANVAS_WIDTH = 3000
    DEFAULT_CANVAS_HEIGHT = 2000

    def __init__(self, root: tk.Tk):
        self.root   = root
        self.engine = DataFlowEngine(root)
        self._tb_drag: dict | None = None
        self._node_type_map: dict[str, Type[BaseNode]] = {}
        self._toolbox_entries_by_category: dict[str, list[tuple[str, Type[BaseNode]]]] = {}
        self._toolbox_category_order: list[str] = []
        self._toolbox_register_category: str | None = None
        self._toolbox_item_widgets: list[tk.Widget] = []
        self._toolbox_search_var = tk.StringVar(value="")
        self._canvas_width = self.DEFAULT_CANVAS_WIDTH
        self._canvas_height = self.DEFAULT_CANVAS_HEIGHT
        self._show_links = True
        self._links_visible_var = tk.BooleanVar(value=True)

        self._project_path: str | None = None
        self._dirty = False
        self._suspend_dirty = False
        self._file_load_in_progress = False
        self._file_load_last_finished_at: float | None = None
        self._project_progress_window: tk.Toplevel | None = None
        self._project_progress_text: tk.Text | None = None
        self._project_progress_start: float | None = None
        self._project_node_load_starts: dict[str, float] = {}

        self._node_counter = 0
        # canvas_nodes: node_id -> BaseNode (shared with engine.nodes)
        self.canvas_nodes: dict[str, BaseNode] = {}
        # Canvas line ids for links
        # link_items: (src_node, src_pin, dst_node, dst_pin) -> canvas line id
        self.link_items: dict[tuple, int] = {}

        self._build_ui()
        self._toolbox_search_var.trace_add("write", lambda *_args: self._refresh_toolbox_palette())
        self._bind_events()
        self._set_project_path(None)
        self._update_title()

        # Drag state
        self._drag: dict = {"mode": None, "node_id": None, "ox": 0, "oy": 0}
        self._resize: dict = {
            "active": False,
            "node_id": None,
            "handle": None,
            "x": 0,
            "y": 0,
            "w": 0,
            "h": 0,
        }
        self._selected_node_id: str | None = None
        self._selected_node_ids: set[str] = set()
        self._selection_items: list[int] = []
        self._pin_id_counter = 1
        self._pin_id_by_key: dict[tuple[str, str, str], int] = {}
        self._pin_key_by_id: dict[int, tuple[str, str, str]] = {}
        self._pin_link_popup: tk.Toplevel | None = None
        # Link-drawing state
        self._linking: dict = {"active": False, "line": None,
                               "src_node": None, "src_pin": None,
                               "src_type": None}
        self._hover_pin_item: int | None = None
        self._hover_pin_candidate: int | None = None
        self._hover_pin_after_id: str | None = None
        self._hover_link_key: tuple | None = None
        self._hover_link_candidate: tuple | None = None
        self._hover_after_id: str | None = None
        self._canvas_pan: dict = {"active": False, "x": 0, "y": 0, "moved": False}

        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ══ UI construction ═══════════════════════════════════════════

    def _build_ui(self) -> None:
        self.root.title("Node Editor")
        self.root.geometry("1100x650")

        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="New", command=self._file_new)
        file_menu.add_command(
            label="Load...", underline=0, accelerator="Ctrl+O",
            command=self._file_load,
        )
        file_menu.add_separator()
        file_menu.add_command(
            label="Save", underline=0, accelerator="Ctrl+S",
            command=self._file_save,
        )
        file_menu.add_command(
            label="Save As...", underline=5, command=self._file_save_as,
        )
        # Tk menu mnemonics use an underlined character rather than a literal
        # ampersand.  This is the native equivalent of "&File" and lets
        # Alt+F open the menu on platforms with a standard menu bar.
        menubar.add_cascade(label="File", underline=0, menu=file_menu)

        canvas_menu = tk.Menu(menubar, tearoff=0)
        canvas_menu.add_command(label="Set Canvas Size...", command=self._set_canvas_size_prompt)
        canvas_menu.add_checkbutton(
            label="Show/hide &Links",
            variable=self._links_visible_var,
            command=self._toggle_link_visibility,
            accelerator="Alt+L",
        )
        menubar.add_cascade(label="Canvas", menu=canvas_menu)
        self.root.config(menu=menubar)

        self.main_pane = tk.PanedWindow(
            self.root,
            orient=tk.HORIZONTAL,
            sashwidth=6,
            sashrelief=tk.RAISED,
            opaqueresize=True,
        )
        self.main_pane.pack(fill="both", expand=True)

        self.toolbox = tk.Frame(self.main_pane, width=180, bg="#f0f0f0",
                                bd=1, relief=tk.SOLID)
        self.toolbox.pack_propagate(False)
        self.main_pane.add(self.toolbox, minsize=120)

        self.toolbox_title = tk.Label(
            self.toolbox, text="Toolbox", font=("Arial", 11, "bold"),
            bg="#f0f0f0", pady=10,
        )
        self.toolbox_title.pack(fill="x")
        # 
#        self.toolbox_title.bind("<Enter>", self._focus_toolbox)

        self.toolbox_search_frame = tk.Frame(self.toolbox, bg="#f0f0f0", padx=6, pady=4)
        self.toolbox_search_frame.pack(fill="x")
        tk.Label(
            self.toolbox_search_frame,
            text="Search",
            font=("Arial", 8, "bold"),
            bg="#f0f0f0",
            anchor="w",
        ).pack(fill="x")
        self.toolbox_search_entry = tk.Entry(
            self.toolbox_search_frame,
            textvariable=self._toolbox_search_var,
            font=("Arial", 9),
        )
        self.toolbox_search_entry.pack(fill="x", pady=(2, 0))

        self.toolbox_body = tk.Frame(self.toolbox, bg="#f0f0f0")
        self.toolbox_body.pack(fill="both", expand=True)

        self.toolbox_canvas = tk.Canvas(
            self.toolbox_body,
            bg="#f0f0f0",
            highlightthickness=0,
            bd=0,
        )
        self.toolbox_v_scroll = tk.Scrollbar(
            self.toolbox_body,
            orient="vertical",
            command=self.toolbox_canvas.yview,
            # Keep the control visually prominent and directly usable with
            # the mouse: its arrows and track scroll, and its thumb drags.
            width=18,
            relief=tk.SUNKEN,
            bd=1,
            cursor="sb_v_double_arrow",
            takefocus=True,
        )
        self.toolbox_canvas.configure(yscrollcommand=self.toolbox_v_scroll.set)

        self.toolbox_canvas.pack(side="left", fill="both", expand=True)
        self.toolbox_v_scroll.pack(side="right", fill="y")

        self.toolbox_content = tk.Frame(self.toolbox_canvas, bg="#f0f0f0")
        self.toolbox_canvas.create_window(
            (0, 0),
            window=self.toolbox_content,
            anchor="nw",
            tags=("toolbox_content",),
        )
        self.toolbox_content.bind(
            "<Configure>",
            lambda _e: self.toolbox_canvas.configure(
                scrollregion=self.toolbox_canvas.bbox("all")
            ),
        )
        self.toolbox_canvas.bind(
            "<Configure>",
            lambda e: self.toolbox_canvas.itemconfigure(
                "toolbox_content",
                width=e.width,
            ),
        )
        for key in ("<Up>", "<Down>"):
            self.toolbox_canvas.bind(key, self._on_toolbox_arrow_key)

        self.canvas_host = tk.Frame(self.main_pane)

        self.canvas = tk.Canvas(
            self.canvas_host,
            bg="#ffffff",
            highlightthickness=0,
            xscrollincrement=1,
            yscrollincrement=1,
        )
        self.h_scroll = tk.Scrollbar(self.canvas_host, orient="horizontal", command=self.canvas.xview)
        self.v_scroll = tk.Scrollbar(self.canvas_host, orient="vertical", command=self.canvas.yview)

        self.canvas.configure(xscrollcommand=self.h_scroll.set,
                              yscrollcommand=self.v_scroll.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.v_scroll.grid(row=0, column=1, sticky="ns")
        self.h_scroll.grid(row=1, column=0, sticky="ew")

        self.canvas_host.grid_rowconfigure(0, weight=1)
        self.canvas_host.grid_columnconfigure(0, weight=1)
        self.main_pane.add(self.canvas_host, minsize=320)

        self._set_canvas_size(self._canvas_width, self._canvas_height)

        for key in ("<Up>", "<Down>", "<Left>", "<Right>"):
            self.canvas.bind(key, self._on_canvas_arrow_key)

    def _set_canvas_size(self, width: int, height: int) -> None:
        self._canvas_width = max(200, int(width))
        self._canvas_height = max(200, int(height))
        self.canvas.configure(scrollregion=(0, 0, self._canvas_width, self._canvas_height))

    def _set_links_visibility(self, visible: bool) -> None:
        self._show_links = bool(visible)
        if hasattr(self, "_links_visible_var"):
            self._links_visible_var.set(self._show_links)

        selected_ids = set(self._selected_node_ids)
        if self._selected_node_id is not None and not selected_ids:
            selected_ids = {self._selected_node_id}

        for (src_node, _src_pin, dst_node, _dst_pin), line_id in self.link_items.items():
            force_visible = (
                bool(selected_ids)
                and (src_node in selected_ids or dst_node in selected_ids)
            )
            try:
                self.canvas.itemconfigure(
                    line_id,
                    state="normal" if self._show_links or force_visible else "hidden",
                )
            except Exception:
                pass
        if self._show_links:
            try:
                self.canvas.tag_raise("graph_link")
                self.canvas.tag_raise("temp_link")
            except Exception:
                pass
        self._clear_hovered_link()
        self._hover_link_candidate = None
        self._cancel_link_hover_after()

    def _toggle_link_visibility(self) -> None:
        self._set_links_visibility(self._links_visible_var.get())

    def _restore_canvas_size(self, meta: dict) -> str | None:
        """Restore a saved canvas size, returning a load warning if invalid."""
        width = meta.get("canvas_width")
        height = meta.get("canvas_height")

        # Projects created before canvas dimensions were persisted retain the
        # application's normal canvas dimensions when opened.
        if width is None and height is None:
            self._set_canvas_size(self.DEFAULT_CANVAS_WIDTH, self.DEFAULT_CANVAS_HEIGHT)
            return None

        try:
            if isinstance(width, bool) or isinstance(height, bool):
                raise ValueError
            width = int(width)
            height = int(height)
            if width < 200 or height < 200:
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            self._set_canvas_size(self.DEFAULT_CANVAS_WIDTH, self.DEFAULT_CANVAS_HEIGHT)
            return "Invalid saved canvas size; restored the default 3000 x 2000 canvas."

        self._set_canvas_size(width, height)
        return None

    def _set_canvas_size_prompt(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Canvas Size")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.resizable(False, False)

        tk.Label(dialog, text="Canvas width (pixels):").grid(row=0, column=0, padx=10, pady=(10, 4), sticky="w")
        width_var = tk.StringVar(value=str(self._canvas_width))
        width_entry = tk.Entry(dialog, textvariable=width_var, width=18)
        width_entry.grid(row=0, column=1, padx=10, pady=(10, 4))

        tk.Label(dialog, text="Canvas height (pixels):").grid(row=1, column=0, padx=10, pady=4, sticky="w")
        height_var = tk.StringVar(value=str(self._canvas_height))
        height_entry = tk.Entry(dialog, textvariable=height_var, width=18)
        height_entry.grid(row=1, column=1, padx=10, pady=4)

        result = {"ok": False}

        def _apply() -> None:
            try:
                width = int(width_var.get().strip())
                height = int(height_var.get().strip())
            except ValueError:
                messagebox.showwarning("Canvas Size", "Width and height must be integers.", parent=dialog)
                return
            if width < 200 or height < 200:
                messagebox.showwarning("Canvas Size", "Width and height must be at least 200.", parent=dialog)
                return
            self._set_canvas_size(width, height)
            self._mark_dirty()
            result["ok"] = True
            dialog.destroy()

        def _cancel() -> None:
            dialog.destroy()

        btn_row = tk.Frame(dialog)
        btn_row.grid(row=2, column=0, columnspan=2, pady=(8, 10))
        tk.Button(btn_row, text="Cancel", width=10, command=_cancel).pack(side="right", padx=5)
        tk.Button(btn_row, text="OK", width=10, command=_apply).pack(side="right", padx=5)

        width_entry.focus_set()
        dialog.bind("<Return>", lambda _e: _apply())
        dialog.bind("<Escape>", lambda _e: _cancel())
        self.root.wait_window(dialog)

    def _widget_is_in_canvas(self, widget: tk.Misc | None) -> bool:
        while widget is not None:
            if widget is self.canvas:
                return True
            widget = getattr(widget, "master", None)
        return False

    def _widget_is_in_toolbox(self, widget: tk.Misc | None) -> bool:
        while widget is not None:
            if widget is self.toolbox:
                return True
            widget = getattr(widget, "master", None)
        return False

    def _on_canvas_mousewheel(self, event) -> None:
        if not self._widget_is_in_canvas(getattr(event, "widget", None)):
            return
        delta = getattr(event, "delta", 0)
        if delta == 0:
            return
        self.canvas.yview_scroll(int(-delta / 120) * self.SCROLL_UNITS, "units")

    def _on_canvas_shift_mousewheel(self, event) -> None:
        if not self._widget_is_in_canvas(getattr(event, "widget", None)):
            return
        delta = getattr(event, "delta", 0)
        if delta == 0:
            return
        self.canvas.xview_scroll(int(-delta / 120) * self.SCROLL_UNITS, "units")

    def _on_toolbox_mousewheel(self, event) -> str | None:
        if not self._widget_is_in_toolbox(getattr(event, "widget", None)):
            return None
        delta = getattr(event, "delta", 0)
        if delta == 0:
            return "break"
        self.toolbox_canvas.yview_scroll(int(-delta / 120) * self.SCROLL_UNITS, "units")
        return "break"

    def _focus_toolbox(self, _event=None) -> None:
        """Route arrow keys to the palette while the pointer is over it."""
        self.toolbox_canvas.focus_set()

    def _on_toolbox_arrow_key(self, event) -> str:
        direction = -1 if event.keysym == "Up" else 1
        self.toolbox_canvas.yview_scroll(direction * 3, "units")
        return "break"

    def _on_canvas_arrow_key(self, event) -> str:
        """Pan the canvas with arrow keys while it has pointer focus."""
        if event.keysym == "Up":
            self.canvas.yview_scroll(-self.SCROLL_UNITS, "units")
        elif event.keysym == "Down":
            self.canvas.yview_scroll(self.SCROLL_UNITS, "units")
        elif event.keysym == "Left":
            self.canvas.xview_scroll(-self.SCROLL_UNITS, "units")
        else:  # Right
            self.canvas.xview_scroll(self.SCROLL_UNITS, "units")
        return "break"

    def _event_canvas_xy(self, event) -> tuple[float, float]:
        return self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

    def register_node_types(self,
                            node_types: list[tuple[str, Type[BaseNode]]]) -> None:
        """
        Register available node types in the Toolbox.
        Each item uses drag-and-drop instead of a click button.
        """
        category = self._toolbox_register_category
        for display_name, cls in node_types:
            self._node_type_map[cls.NODE_TYPE] = cls
            entry_category = category or (getattr(cls, "CATEGORY", "misc") or "misc")
            self._ensure_toolbox_category(entry_category)

            entries = self._toolbox_entries_by_category[entry_category]
            exists = any(existing_cls is cls for _, existing_cls in entries)
            if not exists:
                entries.append((display_name, cls))

        self._refresh_toolbox_palette()

    def set_node_registry(self,
                          registry: dict[str, list[tuple[str, Type[BaseNode]]]]) -> None:
        for entries in registry.values():
            for _display_name, cls in entries:
                self._node_type_map[cls.NODE_TYPE] = cls

    @staticmethod
    def _normalized_search_keywords(cls: Type[BaseNode]) -> list[str]:
        raw = getattr(cls, "SEARCH_KEYWORDS", ())
        if isinstance(raw, str):
            return [raw]
        if isinstance(raw, (list, tuple, set)):
            return [str(v) for v in raw if isinstance(v, str) and str(v).strip()]
        return []

    def _toolbox_entry_matches(self, query: str, category: str,
                               display_name: str, cls: Type[BaseNode]) -> bool:
        text = query.strip().lower()
        if not text:
            return True

        tokens = [tok for tok in text.split() if tok]
        if not tokens:
            return True

        keywords = self._normalized_search_keywords(cls)
        haystack_parts = [
            str(display_name),
            str(getattr(cls, "DISPLAY_NAME", "")),
            str(getattr(cls, "NODE_TYPE", "")),
            str(category),
            *keywords,
        ]
        haystack = " ".join(haystack_parts).lower()
        return all(tok in haystack for tok in tokens)

    def _ensure_toolbox_category(self, category: str) -> None:
        if category not in self._toolbox_entries_by_category:
            self._toolbox_entries_by_category[category] = []
        if category not in self._toolbox_category_order:
            self._toolbox_category_order.append(category)

    def _add_toolbox_item_widget(self, widget: tk.Widget) -> None:
        self._toolbox_item_widgets.append(widget)
        widget.bind("<Enter>", self._focus_toolbox)

    def _refresh_toolbox_palette(self) -> None:
        for widget in self._toolbox_item_widgets:
            try:
                widget.destroy()
            except Exception:
                pass
        self._toolbox_item_widgets = []

        query = self._toolbox_search_var.get()
        has_any = False

        for category in self._toolbox_category_order:
            entries = self._toolbox_entries_by_category.get(category, [])
            matched = [
                (display_name, cls)
                for display_name, cls in entries
                if self._toolbox_entry_matches(query, category, display_name, cls)
            ]
            if not matched:
                continue

            has_any = True
            header = tk.Label(
                self.toolbox_content,
                text=category.upper(),
                font=("Arial", 7, "bold"),
                bg="#d0d0d0", fg="#666666",
                anchor="w", padx=6,
            )
            header.pack(fill="x", pady=(8, 0))
            self._add_toolbox_item_widget(header)

            for display_name, cls in matched:
                lbl = tk.Label(
                    self.toolbox_content,
                    text=display_name,
                    pady=5,
                    relief=tk.RAISED,
                    bg="#d0d0d0",
                    anchor="w",
                    padx=8,
                    cursor="fleur",
                )
                lbl.pack(fill="x", padx=4, pady=4)
                self._add_toolbox_item_widget(lbl)

                lbl.bind("<ButtonPress-1>",
                         lambda e, c=cls: self._tb_on_press(e, c))
                lbl.bind("<B1-Motion>",
                         lambda e: self._tb_on_motion(e))
                lbl.bind("<ButtonRelease-1>",
                         lambda e: self._tb_on_release(e))

        if not has_any:
            no_match = tk.Label(
                self.toolbox_content,
                text="No nodes match your search.",
                font=("Arial", 8),
                fg="#666666",
                bg="#f0f0f0",
                anchor="w",
                padx=8,
                pady=8,
            )
            no_match.pack(fill="x")
            self._add_toolbox_item_widget(no_match)

        self.toolbox_content.update_idletasks()
        self.toolbox_canvas.configure(scrollregion=self.toolbox_canvas.bbox("all"))

    # ══ Toolbox Drag-and-Drop ════════════════════════════════════════

    def _tb_on_press(self, event, cls: Type[BaseNode]) -> None:
        """Record the dragged node type and create a preview outline on the canvas."""
        self._tb_drag = {
            "cls":     cls,
            "preview": None,    # Preview rectangle id on the canvas
        }

    def _tb_on_motion(self, event) -> None:
        """
        Show a preview outline on the canvas while the mouse moves.
        event coordinates are relative to the toolbox widget and must be converted to canvas coordinates.
        """
        if not hasattr(self, "_tb_drag") or self._tb_drag is None:
            return

        # Convert screen coordinates to viewport coords, then to true canvas coords
        rel_x = event.x_root - self.canvas.winfo_rootx()
        rel_y = event.y_root - self.canvas.winfo_rooty()
        cx = self.canvas.canvasx(rel_x)
        cy = self.canvas.canvasy(rel_y)

        cls = self._tb_drag["cls"]
        preview_node = cls("tb_preview", self.canvas)
        w   = cls.NODE_WIDTH
        h   = preview_node.get_default_height()

        if self._tb_drag["preview"] is None:
            # First time entering the canvas: create the preview outline
            self._tb_drag["preview"] = self.canvas.create_rectangle(
                cx, cy, cx + w, cy + h,
                outline="#999999",
                dash=(6, 3),
                fill="",
                tags=("tb_preview",)
            )
        else:
            # Update the preview outline position
            self.canvas.coords(
                self._tb_drag["preview"],
                cx, cy, cx + w, cy + h
            )

    def _tb_on_release(self, event) -> None:
        """
        When the mouse is released:
        - If it is inside the canvas, create a node
        - If it is outside the canvas, cancel
        """
        if not hasattr(self, "_tb_drag") or self._tb_drag is None:
            return

        # Remove the preview outline
        if self._tb_drag["preview"] is not None:
            self.canvas.delete(self._tb_drag["preview"])

        # Check whether the release position is inside the canvas
        rel_x = event.x_root - self.canvas.winfo_rootx()
        rel_y = event.y_root - self.canvas.winfo_rooty()
        cx = self.canvas.canvasx(rel_x)
        cy = self.canvas.canvasy(rel_y)

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        if 0 <= rel_x <= canvas_w and 0 <= rel_y <= canvas_h:
            self._spawn_node(self._tb_drag["cls"], x=cx, y=cy)

        self._tb_drag = None

    # ══ Node creation ═════════════════════════════════════════════

    def _spawn_node(self, cls: Type[BaseNode], x: int = 120,
                    y: int = 150) -> BaseNode:
        node_id = f"node_{self._node_counter}"
        self._node_counter += 1
        preview_node = cls(node_id, self.canvas)
        default_h = preview_node.get_default_height()
        node = self._create_node_instance(cls, node_id, x, y, cls.NODE_WIDTH, default_h, {})
        self.engine.trigger_all()
        self._mark_dirty()
        return node

    def _create_node_instance(self, cls: Type[BaseNode], node_id: str,
                              x: int, y: int, width: int, height: int,
                              params: dict, node_name: str | None = None) -> BaseNode:
        self._node_counter = max(self._node_counter, self._node_index_from_id(node_id) + 1)

        node = cls(node_id, self.canvas)
        node.x = x
        node.y = y
        node.width = max(node.MIN_WIDTH, int(width))
        node.height = max(node.MIN_HEIGHT, int(height))

        # Register before drawing pins: _refresh_pin_ids() only sees nodes in canvas_nodes.
        self.canvas_nodes[node_id] = node

        # 1. Let the node draw its own body
        node.build_body()
        self._apply_node_name(node, node_name)

        # 2. The editor draws pins based on the pin schema
        self._draw_pins(node)
        self._raise_node_title(node)

        # 3. Make the whole node tag draggable
        self.engine.add_node(node)

        node_restore_start = time.perf_counter()
        try:
            if params:
                node.set_params(params)
        except Exception as e:
            messagebox.showwarning("Load Warning", f"Node '{node_id}' params could not be fully restored: {e}")
        finally:
            if getattr(self, "_project_progress_start", None) is not None and cls.NODE_TYPE in {
                "array_viewer", "array_accumulator",
            }:
                self._project_node_load_starts[node_id] = (
                    time.perf_counter() - node_restore_start
                )

        # Start STREAMING nodes immediately after creation
        if node.EXECUTION_MODE == ExecutionMode.STREAMING:
            node._is_running = True
            node.start_stream()
            # Start polling
            self.engine._on_streaming_output(node_id, {})

        return node

    @staticmethod
    def _node_index_from_id(node_id: str) -> int:
        try:
            return int(str(node_id).rsplit("_", 1)[1])
        except Exception:
            return -1

    def _raise_node_title(self, node: BaseNode) -> None:
        """Keep a node's title above its own content and above other nodes when
        the node is selected or resized."""
        if node is None:
            return

        node_id = getattr(node, "node_id", None)
        if node_id is not None:
            try:
                self.canvas.tag_raise(node_id)
            except Exception:
                pass

        title_item = getattr(node, "_title_item", None)
        if title_item:
            try:
                self.canvas.tag_raise(title_item)
            except Exception:
                pass

        if node_id is not None:
            selected_tag = f"selected_node_{node_id}"
            try:
                self.canvas.tag_raise(selected_tag)
            except Exception:
                pass

    def _refresh_pin_ids(self) -> None:
        """Assign a stable integer pin ID to every pin that does not have one yet.

        IDs must never be reassigned once given out: they are baked into each
        pin's canvas tag (f"pin_id_{pin_id}") at draw time, shown to the user
        via right-click, and typed back in by the "Add link by pin ID"
        dialog. Previously this method cleared and rebuilt the whole mapping
        from scratch (sorted by node_id) on every call, which is triggered
        once per node as it is added -- so adding a new node could silently
        renumber every other node's pins (worse once node IDs like "node_10"
        sort lexicographically before "node_2"), leaving already-drawn pin
        tags out of sync with the freshly recomputed ID map.
        """
        for node_id, node in self.canvas_nodes.items():
            try:
                schema = node.get_pin_schema()
            except Exception:
                continue
            for direction in ("in", "out"):
                pins = schema.inputs if direction == "in" else schema.outputs
                for pin_def in pins:
                    key = (node_id, direction, pin_def.name)
                    if key in self._pin_id_by_key:
                        continue
                    self._pin_id_by_key[key] = self._pin_id_counter
                    self._pin_key_by_id[self._pin_id_counter] = key
                    self._pin_id_counter += 1

    def _pin_id_for_key(self, key: tuple[str, str, str]) -> int | None:
        if key not in self._pin_id_by_key:
            self._refresh_pin_ids()
        return self._pin_id_by_key.get(key)

    def _pin_id_from_item(self, item: int) -> int | None:
        for tag in self.canvas.gettags(item):
            if tag.startswith("pin_id_"):
                try:
                    return int(tag.split("pin_id_", 1)[1])
                except ValueError:
                    return None
        return None

    def _draw_pins(self, node: BaseNode) -> None:
        self._refresh_pin_ids()
        schema = node.get_pin_schema()
        w, h   = node.width, node.height
        x, y   = node.x, node.y
        r      = self.PIN_RADIUS

        # Input pins: left side, vertically distributed
        n_in = len(schema.inputs)
        input_label_color = getattr(
            node, "INPUT_PIN_LABEL_COLOR",
            getattr(node, "PIN_LABEL_COLOR", "#1a1a1a"),
        )
        for i, pin_def in enumerate(schema.inputs):
            py = y + (i + 1) * h // (n_in + 1)
            px = x
            pin_id = self._pin_id_for_key((node.node_id, "in", pin_def.name))
            oid = self.canvas.create_oval(
                px - r, py - r, px + r, py + r,
                fill=self.PIN_IN_COLOR, tags=(node.node_id, "in_pin",
                                              f"pin_{node.node_id}_in_{pin_def.name}",
                                              f"pin_id_{pin_id}")
            )
            node.input_pin_items[pin_def.name] = oid
            # Label
            self.canvas.create_text(px + r + 3, py, text=pin_def.label or pin_def.name,
                                    anchor="w", font=("Arial", 7),
                                    fill=input_label_color,
                                    tags=(node.node_id, f"pin_label_{node.node_id}_in_{pin_def.name}"))

        # Output pins: right side
        n_out = len(schema.outputs)
        output_label_color = getattr(
            node, "OUTPUT_PIN_LABEL_COLOR",
            getattr(node, "PIN_LABEL_COLOR", "#1a1a1a"),
        )
        for i, pin_def in enumerate(schema.outputs):
            py = y + (i + 1) * h // (n_out + 1)
            px = x + w
            pin_id = self._pin_id_for_key((node.node_id, "out", pin_def.name))
            oid = self.canvas.create_oval(
                px - r, py - r, px + r, py + r,
                fill=self.PIN_OUT_COLOR, tags=(node.node_id, "out_pin",
                                               f"pin_{node.node_id}_out_{pin_def.name}",
                                               f"pin_id_{pin_id}")
            )
            node.output_pin_items[pin_def.name] = oid
            self.canvas.create_text(px - r - 3, py, text=pin_def.label or pin_def.name,
                                    anchor="e", font=("Arial", 7),
                                    fill=output_label_color,
                                    tags=(node.node_id, f"pin_label_{node.node_id}_out_{pin_def.name}"))

    def _layout_pins_for_node(self, node: BaseNode) -> None:
        """Reposition pins after geometry changes while keeping a fixed pin size."""
        schema = node.get_pin_schema()
        w, h = node.width, node.height
        x, y = node.x, node.y
        r = self.PIN_RADIUS

        n_in = len(schema.inputs)
        for i, pin_def in enumerate(schema.inputs):
            py = y + (i + 1) * h // (n_in + 1)
            px = x
            oid = node.input_pin_items.get(pin_def.name)
            if oid is not None:
                self.canvas.coords(oid, px - r, py - r, px + r, py + r)

            label_tag = f"pin_label_{node.node_id}_in_{pin_def.name}"
            for lid in self.canvas.find_withtag(label_tag):
                self.canvas.coords(lid, px + r + 3, py)

        n_out = len(schema.outputs)
        for i, pin_def in enumerate(schema.outputs):
            py = y + (i + 1) * h // (n_out + 1)
            px = x + w
            oid = node.output_pin_items.get(pin_def.name)
            if oid is not None:
                self.canvas.coords(oid, px - r, py - r, px + r, py + r)

            label_tag = f"pin_label_{node.node_id}_out_{pin_def.name}"
            for lid in self.canvas.find_withtag(label_tag):
                self.canvas.coords(lid, px - r - 3, py)

    # ══ Event binding ════════════════════════════════════════════

    def _bind_events(self) -> None:
        self.canvas.bind("<ButtonPress-1>",   self._on_canvas_press)
        self.canvas.bind("<B1-Motion>",       self._on_canvas_motion)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<Motion>",          self._on_canvas_hover_motion)
        self.canvas.bind("<Leave>",           self._on_canvas_leave)
        self.canvas.bind("<Double-Button-1>", self._on_canvas_double_click)
        self.canvas.bind("<ButtonPress-3>",   self._on_canvas_right_button_press)
        self.canvas.bind("<B3-Motion>",       self._on_canvas_pan_motion)
        self.canvas.bind("<ButtonRelease-3>", self._on_canvas_pan_release)
        self.canvas.bind("<Delete>",          self._on_delete_key)
        # NOTE: Tk ignores letter case for Control-modified key events (a
        # Control-o keypress matches BOTH "<Control-o>" and "<Control-O>"
        # patterns), so binding both here would fire the handler twice per
        # keypress. Bind the lowercase form only; it already matches either
        # case.
        self.canvas.bind("<Control-o>",       self._on_load_hotkey)
        self.canvas.bind("<Control-s>",       self._on_save_hotkey)
        self.canvas.bind("<Control-a>",       self._on_select_all_nodes_hotkey)
        self.canvas.bind("<Control-h>",       self._on_node_help_hotkey, add="+")
        self.root.bind_all("<Alt-l>", self._on_toggle_links_hotkey)
        self.root.bind_all("<Alt-L>", self._on_toggle_links_hotkey)
        self.root.bind_all("<MouseWheel>", self._on_canvas_mousewheel, add="+")
        self.root.bind_all("<Shift-MouseWheel>", self._on_canvas_shift_mousewheel, add="+")
        self.root.bind_all("<MouseWheel>", self._on_toolbox_mousewheel, add="+")

    def _on_load_hotkey(self, _event=None) -> str:
        self._file_load()
        return "break"

    def _on_save_hotkey(self, _event=None) -> str:
        self._file_save()
        return "break"

    def _on_select_all_nodes_hotkey(self, _event=None) -> str:
        """Select every graph node while the main canvas has focus."""
        if self.canvas_nodes:
            self._selected_node_ids = set(self.canvas_nodes)
            self._selected_node_id = next(reversed(self.canvas_nodes))
            self._redraw_selection_overlays()
            self._refresh_selection_visuals()
        return "break"

    def _on_toggle_links_hotkey(self, _event=None) -> str:
        self._links_visible_var.set(not self._links_visible_var.get())
        self._toggle_link_visibility()
        return "break"

    def _on_canvas_double_click(self, event) -> str | None:
        cx, cy = self._event_canvas_xy(event)
        item = self._item_at(cx, cy)
        if item is None:
            return None

        tags = self.canvas.gettags(item)
        node_id = self._node_id_from_tags(tags)
        if node_id is None:
            return None

        node = self.canvas_nodes.get(node_id)
        if node is None:
            return None

        # Image nodes may retain a custom double-click action.  Nodes that
        # opt in use the same gesture to open their inspector instead.
        if (any(tag.startswith("img_area_") for tag in tags)
                and not getattr(node, "OPEN_INSPECTOR_ON_IMAGE_DOUBLE_CLICK", False)):
            return None

        self._select_node(node_id)
        node.open_inspector()
        return "break"

    def _on_canvas_right_button_press(self, event) -> str | None:
        cx, cy = self._event_canvas_xy(event)
        pin_item = self._find_pin_item_near(cx, cy)
        if pin_item is not None:
            pin_id = self._pin_id_from_item(pin_item)
            if pin_id is not None:
                self._show_pin_link_form(pin_item, pin_id, event)
                return "break"

        link_key = self._find_link_key_near(cx, cy)
        if link_key is not None:
            self._show_link_context_menu(link_key, event)
            return "break"

        self._on_canvas_pan_start(event)
        return None

    def _on_canvas_pan_start(self, event) -> None:
        self._canvas_pan = {"active": True, "x": event.x, "y": event.y, "moved": False}
        self.canvas.scan_mark(event.x, event.y)

    def _on_canvas_pan_motion(self, event) -> None:
        if not self._canvas_pan["active"]:
            return
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self._canvas_pan["moved"] = True

    def _on_canvas_pan_release(self, event) -> None:
        if not self._canvas_pan["active"]:
            return

        try:
            if self._canvas_pan["moved"]:
                return
        finally:
            self._canvas_pan = {"active": False, "x": 0, "y": 0, "moved": False}
        self._on_canvas_right_click(event)

    def _show_link_context_menu(self, link_key: tuple, event) -> None:
        src_node_id, src_pin_name, dst_node_id, dst_pin_name = link_key
        src_id = self._pin_id_for_key((src_node_id, "out", src_pin_name))
        dst_id = self._pin_id_for_key((dst_node_id, "in", dst_pin_name))
        src_node = self.canvas_nodes.get(src_node_id)
        dst_node = self.canvas_nodes.get(dst_node_id)
        src_node_name = self._node_display_name(src_node) if src_node is not None else src_node_id
        dst_node_name = self._node_display_name(dst_node) if dst_node is not None else dst_node_id
        src_pin_label = src_pin_name
        dst_pin_label = dst_pin_name
        if src_node is not None:
            src_pin_def = next(
                (pin for pin in src_node.get_pin_schema().outputs if pin.name == src_pin_name),
                None,
            )
            if src_pin_def is not None:
                src_pin_label = src_pin_def.label or src_pin_name
        if dst_node is not None:
            dst_pin_def = next(
                (pin for pin in dst_node.get_pin_schema().inputs if pin.name == dst_pin_name),
                None,
            )
            if dst_pin_def is not None:
                dst_pin_label = dst_pin_def.label or dst_pin_name

        menu = tk.Menu(self.canvas, tearoff=0)
        menu.add_command(
            label=f"From {src_id}: {src_node_name}. {src_pin_label}",
            state=tk.DISABLED,
        )
        menu.add_command(
            label=f"To {dst_id}: {dst_node_name} {dst_pin_label}",
            state=tk.DISABLED,
        )
        menu.add_separator()
        menu.add_command(
            label="Delete link",
            command=lambda key=link_key: self._delete_link(key, trigger_recompute=True),
        )
        menu.tk_popup(event.x_root, event.y_root)
        menu.grab_release()

    def _compute_node_now(self, node_id: str) -> None:
        node = self.canvas_nodes.get(node_id)
        if node is None:
            return
        self.engine._trigger_from(node_id)

    def _on_canvas_right_click(self, event) -> None:
        cx, cy = self._event_canvas_xy(event)
        item = self._item_at(cx, cy)
        if item is None:
            return

        node_id = self._node_id_from_tags(self.canvas.gettags(item))
        if node_id is None or node_id not in self.canvas_nodes:
            return

        node = self.canvas_nodes[node_id]
        self._select_node(node_id)
        menu = tk.Menu(self.canvas, tearoff=0)
        menu.add_command(label="Rename...", command=lambda nid=node_id: self._rename_node_prompt(nid))
        menu.add_command(label="Compute", command=lambda nid=node_id: self._compute_node_now(nid))
        menu.add_separator()
        menu.add_command(label=f"Node: {self._node_display_name(node)}", state=tk.DISABLED)
        menu.add_command(label=f"Type: {getattr(node, 'NODE_TYPE', 'unknown')}", state=tk.DISABLED)
        menu.add_command(
            label=f"Last compute: {BaseNode.format_time_ago(getattr(node, '_last_compute_time', None))}",
            state=tk.DISABLED,
        )

        style_menu = tk.Menu(menu, tearoff=0)
        for style_name in BaseNode.NODE_COLOR_STYLES:
            label = style_name.replace("_", " ").title()
            style_menu.add_command(
                label=label,
                command=lambda nid=node_id, style=style_name: self.canvas_nodes[nid].set_style(style),
            )
        menu.add_cascade(label="Color style", menu=style_menu)
        menu.tk_popup(event.x_root, event.y_root)
        menu.grab_release()

    def _connected_pin_ids_for_pin(self, node_id: str, direction: str, pin_name: str) -> list[int]:
        connected: set[int] = set()
        for (src_node, src_pin, dst_node, dst_pin), _ in self.link_items.items():
            if direction == "out":
                if src_node == node_id and src_pin == pin_name:
                    linked_id = self._pin_id_for_key((dst_node, "in", dst_pin))
                    if linked_id is not None:
                        connected.add(linked_id)
            elif direction == "in":
                if dst_node == node_id and dst_pin == pin_name:
                    linked_id = self._pin_id_for_key((src_node, "out", src_pin))
                    if linked_id is not None:
                        connected.add(linked_id)
        return sorted(connected)

    def _compatible_pin_targets_for_pin(self, node_id: str, direction: str, pin_name: str) -> list[tuple[int, str, str, str, str]]:
        """Return (target_pin_id, target_node_id, target_node_name, target_pin_direction, target_pin_name) for all compatible targets."""
        source_node = self.canvas_nodes.get(node_id)
        if source_node is None:
            return []

        source_schema = source_node.get_pin_schema()
        if direction == "out":
            source_pin_def = next((p for p in source_schema.outputs if p.name == pin_name), None)
            if source_pin_def is None:
                return []
            target_direction = "in"
        else:
            source_pin_def = next((p for p in source_schema.inputs if p.name == pin_name), None)
            if source_pin_def is None:
                return []
            target_direction = "out"

        targets: list[tuple[int, str, str, str, str]] = []
        for target_pin_id, target_key in sorted(self._pin_key_by_id.items()):
            target_node_id, target_direction_name, target_pin_name = target_key
            if target_node_id == node_id:
                continue
            if target_direction_name != target_direction:
                continue

            target_node = self.canvas_nodes.get(target_node_id)
            if target_node is None:
                continue
            target_schema = target_node.get_pin_schema()
            target_defs = target_schema.inputs if target_direction_name == "in" else target_schema.outputs
            target_pin_def = next((p for p in target_defs if p.name == target_pin_name), None)
            if target_pin_def is None:
                continue

            if direction == "out":
                if not pins_compatible(source_pin_def.type, target_pin_def.type):
                    continue
            else:
                if not pins_compatible(target_pin_def.type, source_pin_def.type):
                    continue

            node_name = self._node_display_name(target_node)
            targets.append((target_pin_id, target_node_id, node_name, target_direction_name, target_pin_name))

        return targets

    def _show_pin_link_form(self, pin_item: int, pin_id: int, event) -> None:
        tags = self.canvas.gettags(pin_item)
        if "in_pin" in tags:
            direction = "in"
        elif "out_pin" in tags:
            direction = "out"
        else:
            return

        node_id = self._node_id_from_tags(tags)
        if node_id is None:
            return

        pin_name = None
        for tag in tags:
            if tag.startswith(f"pin_{node_id}_{direction}_"):
                pin_name = tag.split(f"pin_{node_id}_{direction}_", 1)[1]
                break
        if pin_name is None:
            return

        if self._pin_link_popup is not None:
            try:
                self._pin_link_popup.destroy()
            except tk.TclError:
                pass

        popup = tk.Toplevel(self.root)
        self._pin_link_popup = popup
        popup.withdraw()
        popup.transient(self.root)
        popup.overrideredirect(True)

        body = tk.Frame(popup, padx=10, pady=8, bd=1, relief=tk.SOLID, bg="#f0f0f0")
        body.pack(fill="both", expand=True)

        tk.Label(
            body, text=f"Pin ID: {pin_id}", anchor="w",
            font=("Arial", 9), bg="#f0f0f0",
        ).pack(fill="x", pady=(0, 6))
        connected_ids = self._connected_pin_ids_for_pin(node_id, direction, pin_name)
        if connected_ids:
            text_add_a_link_to = f"connects to pin {connected_ids[0]}. Replace it with pin:"
        else:
            text_add_a_link_to = "Add a link to pin:"
        tk.Label(
            body, text=text_add_a_link_to, anchor="w",
            font=("Arial", 9), bg="#f0f0f0",
        ).pack(fill="x", pady=(0, 4))

        entry_var = tk.StringVar()
        entry = tk.Entry(body, textvariable=entry_var, width=18, justify="center")
        entry.pack(fill="x")
        validation_label = tk.Label(
            body, text="", anchor="w", fg="#aa2222",
            font=("Arial", 8), bg="#f0f0f0",
        )
        validation_label.pack(fill="x", pady=(2, 6))

        def _close() -> None:
            if self._pin_link_popup is popup:
                self._pin_link_popup = None
            try:
                popup.grab_release()
            except tk.TclError:
                pass
            popup.destroy()

        def _add() -> None:
            value = (entry_var.get() or "").strip()
            try:
                target_pin_id = int(value)
            except ValueError:
                validation_label.configure(text="Pin ID must be an integer.")
                entry.configure(bg="#ffe8e8")
                entry.focus_force()
                entry.selection_range(0, tk.END)
                return
            _close()
            self._connect_by_pin_id(node_id, direction, pin_name, target_pin_id)

        def _clear_validation(*_args) -> None:
            validation_label.configure(text="")
            entry.configure(bg="white")

        entry_var.trace_add("write", _clear_validation)

        buttons = tk.Frame(body, bg="#f0f0f0")
        buttons.pack(fill="x")
        tk.Button(buttons, text="Add", width=10, command=_add).pack(side="left")
        tk.Button(buttons, text="Cancel", width=10, command=_close).pack(side="right")

        popup.bind("<Return>", lambda _e: _add())
        popup.bind("<Escape>", lambda _e: _close())
        popup.protocol("WM_DELETE_WINDOW", _close)

        popup.update_idletasks()
        pin_box = self.canvas.bbox(pin_item)
        if pin_box is not None:
            popup_x = int(self.canvas.winfo_rootx() + pin_box[2] - self.canvas.canvasx(0) + 6)
            popup_y = int(self.canvas.winfo_rooty() + pin_box[1] - self.canvas.canvasy(0))
        elif event is not None:
            popup_x, popup_y = int(event.x_root + 12), int(event.y_root + 12)
        else:
            popup_x, popup_y = self.root.winfo_pointerxy()
        popup_x = min(popup_x, popup.winfo_screenwidth() - popup.winfo_reqwidth())
        popup_y = min(popup_y, popup.winfo_screenheight() - popup.winfo_reqheight())
        popup.geometry(f"+{max(0, popup_x)}+{max(0, popup_y)}")
        popup.deiconify()
        popup.lift()
        popup.grab_set()
        entry.focus_force()

    def _prompt_pin_link_by_id(self, pin_item: int, pin_id: int) -> None:
        self._show_pin_link_form(pin_item, pin_id, None)

    def _connect_by_pin_id(self, node_id: str, direction: str, pin_name: str, target_pin_id: int) -> None:
        src_key = (node_id, direction, pin_name)
        src_pin_id = self._pin_id_for_key(src_key)
        if src_pin_id is None:
            messagebox.showwarning("Add link by pin ID", "Source pin is not mapped to a valid ID.", parent=self.root)
            return

        dst_key = self._pin_key_by_id.get(target_pin_id)
        if dst_key is None:
            messagebox.showwarning("Add link by pin ID", f"Pin {target_pin_id} does not exist.", parent=self.root)
            return

        dst_node, dst_direction, dst_pin = dst_key
        if direction == "out":
            if dst_direction != "in":
                messagebox.showwarning("Add link by pin ID", f"Pin {target_pin_id} is not an input pin.", parent=self.root)
                return
            src_node, src_pin, dest_node, dest_pin = node_id, pin_name, dst_node, dst_pin
        else:
            if dst_direction != "out":
                messagebox.showwarning("Add link by pin ID", f"Pin {target_pin_id} is not an output pin.", parent=self.root)
                return
            src_node, src_pin, dest_node, dest_pin = dst_node, dst_pin, node_id, pin_name

        if src_node == dest_node:
            messagebox.showwarning("Add link by pin ID", "A pin cannot link to itself.", parent=self.root)
            return

        existing = self.link_items.get((src_node, src_pin, dest_node, dest_pin))
        if existing is not None:
            self._delete_link((src_node, src_pin, dest_node, dest_pin), trigger_recompute=True)
            return

        src = self.canvas_nodes.get(src_node)
        dst = self.canvas_nodes.get(dest_node)
        if src is None or dst is None:
            messagebox.showwarning("Add link by pin ID", "One of the pins no longer exists in the graph.", parent=self.root)
            return

        src_schema = src.get_pin_schema()
        dst_schema = dst.get_pin_schema()
        src_pin_def = next((p for p in src_schema.outputs if p.name == src_pin), None)
        dst_pin_def = next((p for p in dst_schema.inputs if p.name == dest_pin), None)
        if src_pin_def is None or dst_pin_def is None:
            messagebox.showwarning("Add link by pin ID", "This pin pair could not be resolved.", parent=self.root)
            return
        if not pins_compatible(src_pin_def.type, dst_pin_def.type):
            messagebox.showwarning("Type Mismatch", f"Cannot connect: {src_pin_def.type.name} -> {dst_pin_def.type.name}")
            return

        self._remove_links_to(dest_node, dest_pin)
        ok = self._draw_link(src_node, src_pin, dest_node, dest_pin)
        if ok:
            self._mark_dirty()
            self.engine.trigger_all()
        else:
            messagebox.showwarning("Add link by pin ID", "The link could not be created.", parent=self.root)

    def _node_display_name(self, node: BaseNode) -> str:
        name = getattr(node, "node_name", "")
        if isinstance(name, str) and name.strip():
            return name.strip()
        return str(getattr(node, "DISPLAY_NAME", node.node_id))

    def _apply_node_name(self, node: BaseNode, node_name: str | None) -> None:
        name = (node_name or "").strip()
        if name:
            node.node_name = name
            title = f"{node.DISPLAY_NAME}: {name}"
        else:
            node.node_name = ""
            title = str(getattr(node, "DISPLAY_NAME", node.node_id))

        if getattr(node, "_title_item", None):
            self.canvas.itemconfigure(node._title_item, text=title)

    def _rename_node_prompt(self, node_id: str) -> None:
        node = self.canvas_nodes.get(node_id)
        if node is None:
            return

        current = self._node_display_name(node)
        new_name = simpledialog.askstring(
            "Rename Node",
            "Node name:",
            initialvalue=current,
            parent=self.root,
        )
        if new_name is None:
            return

        new_name = new_name.strip()
        if not new_name:
            messagebox.showwarning("Rename Node", "Node name cannot be empty.", parent=self.root)
            return

        if new_name != current:
            self._apply_node_name(node, new_name)
            self._mark_dirty()

    def _on_canvas_press(self, event) -> None:
        self.canvas.focus_set()
        cx, cy = self._event_canvas_xy(event)
        pin_item = self._find_pin_item_near(cx, cy)
        item = pin_item if pin_item is not None else self._item_at(cx, cy)
        if item is None:
            self._clear_selection()
            return
        tags = self.canvas.gettags(item)

        if "out_pin" in tags:
            self._start_linking(event, item, tags)
        elif "in_pin" in tags:
            pass   # Do nothing when pressing an input pin
        else:
            node_id = self._node_id_from_tags(tags)
            if node_id is None:
                return

            ctrl_down = bool(event.state & 0x0004)
            if ctrl_down:
                if node_id in self._selected_node_ids:
                    self._selected_node_ids.discard(node_id)
                    if self._selected_node_ids:
                        self._selected_node_id = sorted(self._selected_node_ids)[-1]
                    else:
                        self._selected_node_id = None
                    self._redraw_selection_overlays()
                    self._refresh_selection_visuals()
                    return

                self._selected_node_ids.add(node_id)
                self._selected_node_id = node_id
                self._draw_selection_overlay(node_id)
                return

            if node_id not in self._selected_node_ids:
                self._selected_node_ids = {node_id}
                self._selected_node_id = node_id
            else:
                self._selected_node_id = node_id
            self._draw_selection_overlay(node_id)

            # Let image-area interactions (pan/zoom handlers in node) consume drag gestures.
            if any(tag.startswith("img_area_") for tag in tags):
                self._drag = {"mode": None, "node_id": None, "ox": 0, "oy": 0}
                return

            handle = self._resize_handle_from_tags(tags)
            if handle is not None:
                node = self.canvas_nodes[node_id]
                self._resize = {
                    "active": True,
                    "node_id": node_id,
                    "handle": handle,
                    "x": node.x,
                    "y": node.y,
                    "w": node.width,
                    "h": node.height,
                }
                self._drag = {"mode": None, "node_id": None, "ox": 0, "oy": 0}
            else:
                selected_ids = set(self._selected_node_ids) if self._selected_node_ids else {node_id}
                self._drag = {"mode": "move", "node_id": node_id,
                              "node_ids": selected_ids,
                              "ox": cx, "oy": cy}

    def _on_canvas_motion(self, event) -> None:
        cx, cy = self._event_canvas_xy(event)
        if self._linking["active"]:
            self.canvas.coords(self._linking["line"],
                               *self.canvas.coords(self._linking["line"])[:2],
                               cx, cy)
        elif self._resize["active"]:
            self._resize_node(cx, cy)
        elif self._drag["mode"] == "move":
            self._move_nodes(cx, cy)

    def _on_canvas_hover_motion(self, event) -> None:
        if self._linking["active"] or self._resize["active"] or self._drag["mode"] == "move":
            self._clear_hovered_pin()
            self._hover_pin_candidate = None
            self._cancel_pin_hover_after()
            self._clear_hovered_link()
            self._hover_link_candidate = None
            self._cancel_link_hover_after()
            return

        cx, cy = self._event_canvas_xy(event)

        pin_candidate = self._find_pin_item_near(cx, cy)
        if pin_candidate is not None:
            self._hover_link_candidate = None
            self._cancel_link_hover_after()
            self._clear_hovered_link()

            if pin_candidate == self._hover_pin_item:
                self._hover_pin_candidate = pin_candidate
                return

            if pin_candidate != self._hover_pin_candidate:
                self._hover_pin_candidate = pin_candidate
                self._cancel_pin_hover_after()
                self._hover_pin_after_id = self.canvas.after(
                    self.PIN_HOVER_MS,
                    lambda item_id=pin_candidate: self._apply_hovered_pin_if_still(item_id),
                )
            return

        self._hover_pin_candidate = None
        self._cancel_pin_hover_after()
        self._clear_hovered_pin()

        candidate = self._find_link_key_near(cx, cy)
        if candidate is None:
            self._hover_link_candidate = None
            self._cancel_link_hover_after()
            self._clear_hovered_link()
            return

        if candidate == self._hover_link_key:
            self._hover_link_candidate = candidate
            return

        if candidate != self._hover_link_candidate:
            self._hover_link_candidate = candidate
            self._cancel_link_hover_after()
            self._hover_after_id = self.canvas.after(
                self.LINK_HOVER_MS,
                lambda key=candidate: self._apply_hovered_link_if_still(key),
            )

    def _on_canvas_leave(self, _event) -> None:
        self._hover_pin_candidate = None
        self._cancel_pin_hover_after()
        self._clear_hovered_pin()
        self._hover_link_candidate = None
        self._cancel_link_hover_after()
        self._clear_hovered_link()

    def _cancel_pin_hover_after(self) -> None:
        if self._hover_pin_after_id is None:
            return
        try:
            self.canvas.after_cancel(self._hover_pin_after_id)
        except Exception:
            pass
        self._hover_pin_after_id = None

    def _apply_hovered_pin_if_still(self, item_id: int) -> None:
        self._hover_pin_after_id = None
        if item_id != self._hover_pin_candidate:
            return
        self._set_hovered_pin(item_id)

    def _set_hovered_pin(self, item_id: int) -> None:
        if item_id == self._hover_pin_item:
            return
        self._clear_hovered_pin()
        tags = self.canvas.gettags(item_id)
        if "in_pin" not in tags and "out_pin" not in tags:
            return
        self.canvas.itemconfig(item_id, fill=self.PIN_HOVER_COLOR, width=2)
        self._hover_pin_item = item_id

    def _clear_hovered_pin(self) -> None:
        if self._hover_pin_item is None:
            return
        tags = self.canvas.gettags(self._hover_pin_item)
        base_color = self.PIN_OUT_COLOR if "out_pin" in tags else self.PIN_IN_COLOR
        self.canvas.itemconfig(self._hover_pin_item, fill=base_color, width=1)
        self._hover_pin_item = None

    def _cancel_link_hover_after(self) -> None:
        if self._hover_after_id is None:
            return
        try:
            self.canvas.after_cancel(self._hover_after_id)
        except Exception:
            pass
        self._hover_after_id = None

    def _apply_hovered_link_if_still(self, key: tuple) -> None:
        self._hover_after_id = None
        if key != self._hover_link_candidate:
            return
        self._set_hovered_link(key)

    def _set_hovered_link(self, key: tuple) -> None:
        if key == self._hover_link_key:
            return
        self._clear_hovered_link()
        line_id = self.link_items.get(key)
        if line_id is None:
            return
        self.canvas.itemconfig(line_id, fill=self.LINK_HOVER_COLOR, width=3)
        self._hover_link_key = key

    def _clear_hovered_link(self) -> None:
        if self._hover_link_key is None:
            return
        line_id = self.link_items.get(self._hover_link_key)
        if line_id is not None:
            self.canvas.itemconfig(line_id, fill=self.PIN_OUT_COLOR, width=2)
        self._hover_link_key = None

    def _on_canvas_release(self, event) -> None:
        if self._linking["active"]:
            self._try_finish_link(event)
        self._drag  = {"mode": None, "node_id": None, "ox": 0, "oy": 0}
        self._resize = {"active": False, "node_id": None, "handle": None,
                        "x": 0, "y": 0, "w": 0, "h": 0}

    # ══ Link logic ════════════════════════════════════════════════

    def _start_linking(self, event, item: int, tags: tuple) -> None:
        # Parse node_id and pin_name from the tag
        src_node, src_pin = self._parse_pin_tag(tags, "out")
        if src_node is None:
            return

        schema   = self.canvas_nodes[src_node].get_pin_schema()
        pin_def  = next((p for p in schema.outputs if p.name == src_pin), None)
        if pin_def is None:
            return

        cx_event, cy_event = self._event_canvas_xy(event)
        x1, y1, x2, y2 = self.canvas.coords(item)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        line = self.canvas.create_line(cx, cy, cx_event, cy_event,
                                       fill=self.PIN_OUT_COLOR, width=2,
                                       dash=(4, 2),
                                       tags=("temp_link",))
        self._linking = {"active": True, "line": line,
                         "src_node": src_node, "src_pin": src_pin,
                         "src_type": pin_def.type}

    def _try_finish_link(self, event) -> None:
        cx, cy = self._event_canvas_xy(event)
        line = self._linking["line"]
        item = self._find_pin_item_near(cx, cy, direction="in")

        success = False
        if item is not None:
            tags = self.canvas.gettags(item)
            dst_node, dst_pin = self._parse_pin_tag(tags, "in")
            if dst_node is not None:
                # Disallow self-connections.
                if dst_node != self._linking["src_node"]:
                    # Type compatibility check.
                    dst_schema = self.canvas_nodes[dst_node].get_pin_schema()
                    dst_pin_def = next((p for p in dst_schema.inputs if p.name == dst_pin), None)
                    if dst_pin_def is not None:
                        if not pins_compatible(self._linking["src_type"], dst_pin_def.type):
                            messagebox.showwarning(
                                "Type Mismatch",
                                f"Cannot connect: {self._linking['src_type'].name}"
                                f" -> {dst_pin_def.type.name}",
                            )
                        else:
                            # An input pin can only have one incoming link.
                            self._remove_links_to(dst_node, dst_pin)

                            # Turn the temporary dashed line into a solid connection.
                            x1, y1, x2, y2 = self.canvas.coords(item)
                            ex, ey = (x1 + x2) / 2, (y1 + y2) / 2
                            sx, sy = self.canvas.coords(line)[:2]
                            self.canvas.coords(line, sx, sy, ex, ey)
                            self.canvas.itemconfig(line, dash=())
                            self.canvas.itemconfig(line, tags=("graph_link",))
                            try:
                                self.canvas.tag_raise("graph_link")
                            except Exception:
                                pass

                            key = (
                                self._linking["src_node"],
                                self._linking["src_pin"],
                                dst_node,
                                dst_pin,
                            )
                            self.link_items[key] = line
                            self._set_links_visibility(self._show_links)

                            ok = self.engine.add_link(
                                self._linking["src_node"],
                                self._linking["src_pin"],
                                dst_node,
                                dst_pin,
                            )

                            if not ok:
                                self.canvas.delete(line)
                                del self.link_items[key]
                                messagebox.showwarning(
                                    "Cycle Dependency",
                                    "This connection would create a cycle and was rejected.",
                                )
                            else:
                                self._mark_dirty()
                            success = True

        if not success:
            self.canvas.delete(line)

        self._refresh_selection_visuals()
        self._linking = {
            "active": False,
            "line": None,
            "src_node": None,
            "src_pin": None,
            "src_type": None,
        }

    def _remove_links_to(self, dst_node: str, dst_pin: str) -> None:
        to_remove = [k for k in self.link_items
                     if k[2] == dst_node and k[3] == dst_pin]
        for k in to_remove:
            self._delete_link(k)

    def _find_link_key_near(self, x: float, y: float) -> tuple | None:
        hit = self.canvas.find_overlapping(
            x - self.LINK_HIT_PAD,
            y - self.LINK_HIT_PAD,
            x + self.LINK_HIT_PAD,
            y + self.LINK_HIT_PAD,
        )
        for item in reversed(hit):
            tags = self.canvas.gettags(item)
            if "graph_link" not in tags:
                continue
            key = self._link_key_from_line(item)
            if key is not None:
                return key
        return None

    def _find_pin_item_near(self, x: float, y: float, direction: str | None = None) -> int | None:
        hit = self.canvas.find_overlapping(
            x - self.PIN_HIT_PAD,
            y - self.PIN_HIT_PAD,
            x + self.PIN_HIT_PAD,
            y + self.PIN_HIT_PAD,
        )

        candidates: list[tuple[float, int, int]] = []
        for item in hit:
            tags = self.canvas.gettags(item)
            is_in = "in_pin" in tags
            is_out = "out_pin" in tags
            if not is_in and not is_out:
                continue
            if direction == "in" and not is_in:
                continue
            if direction == "out" and not is_out:
                continue

            x1, y1, x2, y2 = self.canvas.coords(item)
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            dist2 = (cx - x) * (cx - x) + (cy - y) * (cy - y)
            # Prefer output pin when distance ties in generic hit-test.
            pri = 0 if is_out else 1
            candidates.append((dist2, pri, item))

        if not candidates:
            return None
        candidates.sort(key=lambda it: (it[0], it[1]))
        return candidates[0][2]

    def _link_key_from_line(self, line_id: int) -> tuple | None:
        for key, lid in self.link_items.items():
            if lid == line_id:
                return key
        return None

    def _delete_link(self, key: tuple, trigger_recompute: bool = False) -> None:
        line_id = self.link_items.pop(key, None)
        if line_id is None:
            return
        if self._hover_link_key == key:
            self._hover_link_key = None
        if self._hover_link_candidate == key:
            self._hover_link_candidate = None
        self.canvas.delete(line_id)
        self.engine.remove_link(key[0], key[1], key[2], key[3])
        if trigger_recompute:
            self.engine.trigger_all()
        self._refresh_selection_visuals()
        self._mark_dirty()

    # ══ Node dragging and movement ═════════════════════════════════

    def _move_nodes(self, cur_x: float, cur_y: float) -> None:
        selected_ids = list(self._drag.get("node_ids") or [self._drag["node_id"]])
        dx = cur_x - self._drag["ox"]
        dy = cur_y - self._drag["oy"]

        for node_id in selected_ids:
            if node_id not in self.canvas_nodes:
                continue
            self.canvas.move(node_id, dx, dy)
            node = self.canvas_nodes[node_id]
            node.x += dx
            node.y += dy
            self._update_links_for_node(node_id)

        self._drag["ox"] = cur_x
        self._drag["oy"] = cur_y
        self._redraw_selection_overlays()
        self._mark_dirty()

    def _move_node(self, cur_x: float, cur_y: float) -> None:
        self._move_nodes(cur_x, cur_y)

    def _resize_node(self, x: int, y: int) -> None:
        node_id = self._resize["node_id"]
        if node_id is None:
            return

        node = self.canvas_nodes[node_id]
        start_x = self._resize["x"]
        start_y = self._resize["y"]
        start_w = max(1, self._resize["w"])
        start_h = max(1, self._resize["h"])

        min_w = max(node.MIN_WIDTH, 1)
        min_h = max(node.MIN_HEIGHT, node.get_min_resizable_height())
        right = start_x + start_w
        bottom = start_y + start_h
        handle = self._resize["handle"]

        if handle == "se":
            new_x = start_x
            new_y = start_y
            new_w = max(min_w, x - start_x)
            new_h = max(min_h, y - start_y)
        elif handle == "sw":
            new_x = min(x, right - min_w)
            new_y = start_y
            new_w = right - new_x
            new_h = max(min_h, y - start_y)
        elif handle == "ne":
            new_x = start_x
            new_y = min(y, bottom - min_h)
            new_w = max(min_w, x - start_x)
            new_h = bottom - new_y
        else:   # nw
            new_x = min(x, right - min_w)
            new_y = min(y, bottom - min_h)
            new_w = right - new_x
            new_h = bottom - new_y

        self._apply_node_geometry(node_id, new_x, new_y, new_w, new_h)

    def _apply_node_geometry(self, node_id: str, new_x: int, new_y: int,
                             new_w: int, new_h: int) -> None:
        node = self.canvas_nodes[node_id]
        old_x, old_y = node.x, node.y
        old_w = max(1, node.width)
        old_h = max(1, node.height)

        new_w = max(node.MIN_WIDTH, int(new_w))
        new_h = max(node.get_min_resizable_height(), int(new_h))

        scale_x = new_w / old_w
        scale_y = new_h / old_h

        for item in self.canvas.find_withtag(node_id):
            if item == node._body_rect:
                self.canvas.coords(item, new_x, new_y, new_x + new_w, new_y + new_h)
                continue

            if item == node._title_item:
                # Keep title anchored near the top edge instead of scaling vertically.
                self.canvas.coords(item, new_x + new_w / 2, new_y + 13)
                continue

            coords = self.canvas.coords(item)
            if not coords:
                continue

            if self.canvas.type(item) == "window":
                self.canvas.coords(
                    item,
                    new_x + (coords[0] - old_x) * scale_x,
                    new_y + (coords[1] - old_y) * scale_y,
                )
                continue

            transformed: list[float] = []
            for idx in range(0, len(coords), 2):
                transformed.extend([
                    new_x + (coords[idx] - old_x) * scale_x,
                    new_y + (coords[idx + 1] - old_y) * scale_y,
                ])
            self.canvas.coords(item, *transformed)

        node.x = new_x
        node.y = new_y
        node.set_size(new_w, new_h)
        node.on_resize(old_w, old_h, new_w, new_h)
        self._raise_node_title(node)
        self._layout_pins_for_node(node)
        self._update_links_for_node(node_id)
        if self._selected_node_id == node_id:
            self._draw_selection_overlay(node_id)
        self._mark_dirty()

    def _update_links_for_node(self, node_id: str) -> None:
        node   = self.canvas_nodes[node_id]
        schema = node.get_pin_schema()

        for pin_def in schema.outputs:
            oid = node.output_pin_items.get(pin_def.name)
            if oid is None:
                continue
            x1, y1, x2, y2 = self.canvas.coords(oid)
            sx, sy = (x1 + x2) / 2, (y1 + y2) / 2
            for k, line_id in self.link_items.items():
                if k[0] == node_id and k[1] == pin_def.name:
                    coords = self.canvas.coords(line_id)
                    self.canvas.coords(line_id, sx, sy, coords[2], coords[3])

        for pin_def in schema.inputs:
            oid = node.input_pin_items.get(pin_def.name)
            if oid is None:
                continue
            x1, y1, x2, y2 = self.canvas.coords(oid)
            ex, ey = (x1 + x2) / 2, (y1 + y2) / 2
            for k, line_id in self.link_items.items():
                if k[2] == node_id and k[3] == pin_def.name:
                    coords = self.canvas.coords(line_id)
                    self.canvas.coords(line_id, coords[0], coords[1], ex, ey)

    # ══ Helper methods ════════════════════════════════════════════

    def _item_at(self, x: int, y: int) -> int | None:
        items = self.canvas.find_overlapping(x - 4, y - 4, x + 4, y + 4)
        return items[-1] if items else None

    def _node_id_from_tags(self, tags: tuple) -> str | None:
        for tag in tags:
            if tag in self.canvas_nodes:
                return tag
            if tag.startswith("selected_node_"):
                return tag[len("selected_node_"):]
        return None

    @staticmethod
    def _resize_handle_from_tags(tags: tuple) -> str | None:
        for tag in tags:
            if tag.startswith("resize_handle_"):
                return tag[len("resize_handle_"):]
        return None

    def _clear_selection(self) -> None:
        for item in self._selection_items:
            self.canvas.delete(item)
        self._selection_items = []
        self._selected_node_ids = set()
        self._selected_node_id = None
        self._refresh_selection_visuals()

    def _refresh_selection_visuals(self) -> None:
        selected_ids = set(self._selected_node_ids)
        if self._selected_node_id is not None and not selected_ids:
            selected_ids = {self._selected_node_id}

        neighbor_ids: set[str] = set()
        for (src_node, _src_pin, dst_node, _dst_pin), line_id in self.link_items.items():
            is_neighbor = src_node in selected_ids or dst_node in selected_ids
            if is_neighbor:
                neighbor_ids.add(src_node)
                neighbor_ids.add(dst_node)
            force_visible = bool(selected_ids) and is_neighbor
            try:
                if self._show_links or force_visible:
                    self.canvas.itemconfig(
                        line_id,
                        state="normal",
                        width=4 if is_neighbor else 2,
                        fill=self.PIN_OUT_COLOR,
                    )
                else:
                    self.canvas.itemconfig(line_id, state="hidden")
            except Exception:
                pass

        for node_id, node in self.canvas_nodes.items():
            body_id = getattr(node, "_body_rect", None)
            if body_id is None:
                continue
            if node_id in selected_ids:
                width = 5
            elif node_id in neighbor_ids:
                width = 4
            else:
                width = 1
            try:
                self.canvas.itemconfig(body_id, width=width)
            except Exception:
                pass

    def _select_node(self, node_id: str) -> None:
        if self._selected_node_id == node_id and self._selected_node_ids == {node_id} and self._selection_items:
            return
        self._selected_node_ids = {node_id}
        self._selected_node_id = node_id
        node = self.canvas_nodes.get(node_id)
        if node is not None:
            self._raise_node_title(node)
        self._draw_selection_overlay(node_id)
        if node is not None:
            self._raise_node_title(node)
        try:
            self.canvas.tag_raise("graph_link")
            self.canvas.tag_raise("temp_link")
        except Exception:
            pass

    def _redraw_selection_overlays(self) -> None:
        for item in self._selection_items:
            if self.canvas.find_withtag(item):
                self.canvas.delete(item)
        self._selection_items = []

        selected_ids = set(self._selected_node_ids)
        if self._selected_node_id is not None and not selected_ids:
            selected_ids = {self._selected_node_id}
        if not selected_ids:
            return

        for node_id in sorted(selected_ids):
            if node_id not in self.canvas_nodes:
                continue
            node = self.canvas_nodes[node_id]
            pad = 4
            handle = 6
            x, y, w, h = node.x - pad, node.y - pad, node.width + pad * 2, node.height + pad * 2

            frame = self.canvas.create_rectangle(
                x, y, x + w, y + h,
                outline="#222222", width=1, dash=(4, 2),
                tags=(f"selected_node_{node_id}", "selection_overlay"),
            )
            self._selection_items.append(frame)
            self.canvas.tag_raise(f"selected_node_{node_id}")

            corner_specs = {
                "nw": (x, y),
                "ne": (x + w, y),
                "sw": (x, y + h),
                "se": (x + w, y + h),
            }
            for corner, (cx, cy) in corner_specs.items():
                square = self.canvas.create_rectangle(
                    cx - handle, cy - handle, cx + handle, cy + handle,
                    fill="#ffffff", outline="#222222", width=1,
                    tags=(f"selected_node_{node_id}", "selection_overlay",
                          f"resize_handle_{corner}"),
                )
                self._selection_items.append(square)
                self.canvas.tag_raise(f"selected_node_{node_id}")

        try:
            self.canvas.tag_raise("graph_link")
            self.canvas.tag_raise("temp_link")
        except Exception:
            pass

    def _draw_selection_overlay(self, node_id: str) -> None:
        selected_ids = set(self._selected_node_ids)
        if not selected_ids:
            selected_ids = {node_id}
        self._selected_node_ids = selected_ids
        if self._selected_node_id is None:
            self._selected_node_id = node_id
        self._redraw_selection_overlays()
        self._refresh_selection_visuals()
        node = self.canvas_nodes.get(node_id)
        if node is not None:
            self._raise_node_title(node)

    def _on_delete_key(self, _event=None) -> None:
        selected_ids = sorted(self._selected_node_ids) if self._selected_node_ids else ([] if self._selected_node_id is None else [self._selected_node_id])
        if not selected_ids:
            return

        if len(selected_ids) == 1:
            node_id = selected_ids[0]
            node = self.canvas_nodes.get(node_id)
            if node is None:
                return
            node_name = getattr(node, "DISPLAY_NAME", node_id)
            prompt = f"Delete selected node '{node_name}'?"
        else:
            prompt = f"Delete {len(selected_ids)} selected nodes?"

        confirmed = messagebox.askyesno("Delete Node", prompt, parent=self.root)
        if not confirmed:
            return

        for node_id in list(selected_ids):
            if node_id in self.canvas_nodes:
                self._delete_node(node_id)

    def _show_selected_node_help_popup(self, help_text: str) -> None:
        popup = tk.Toplevel(self.root)
        popup.title("Node Help")
        popup.transient(self.root)
        popup.resizable(False, False)

        px, py = self.root.winfo_pointerxy()
        popup.geometry(f"+{px + 14}+{py + 14}")

        body = tk.Frame(popup, bg="#111111", padx=10, pady=8)
        body.pack(fill="both", expand=True)

        tk.Label(
            body,
            text=help_text,
            justify="left",
            anchor="w",
            bg="#111111",
            fg="#dddddd",
            font=("Arial", 9),
        ).pack(fill="both", expand=True)

        tk.Button(body, text="Close", command=popup.destroy).pack(anchor="e", pady=(8, 0))

        popup.bind("<Escape>", lambda _e: popup.destroy())
        popup.protocol("WM_DELETE_WINDOW", popup.destroy)

    def _on_node_help_hotkey(self, _event=None) -> str | None:
        node_id = self._selected_node_id
        if node_id is None:
            return None

        node = self.canvas_nodes.get(node_id)
        if node is None:
            return None

        help_text = ""
        if hasattr(node, "get_help_text") and callable(getattr(node, "get_help_text")):
            try:
                help_text = str(node.get_help_text() or "")
            except Exception:
                help_text = ""

        if not help_text:
            maybe = getattr(node, "HELP_TEXT", "")
            help_text = str(maybe or "")

        if not help_text:
            return None

        self._show_selected_node_help_popup(help_text)
        return "break"

    def _delete_node(self, node_id: str) -> None:
        if node_id not in self.canvas_nodes:
            return

        related_links = [k for k in self.link_items
                         if k[0] == node_id or k[2] == node_id]
        for key in related_links:
            self._delete_link(key)

        self.canvas.delete(node_id)
        self.engine.remove_node(node_id)
        self.canvas_nodes.pop(node_id, None)

        if self._selected_node_id == node_id:
            self._clear_selection()

        self.engine.trigger_all()
        self._mark_dirty()

    def _draw_link(self, src_node: str, src_pin: str,
                   dst_node: str, dst_pin: str, trigger: bool = True) -> bool:
        src = self.canvas_nodes.get(src_node)
        dst = self.canvas_nodes.get(dst_node)
        if src is None or dst is None:
            return False

        src_oid = src.output_pin_items.get(src_pin)
        dst_oid = dst.input_pin_items.get(dst_pin)
        if src_oid is None or dst_oid is None:
            return False

        sx1, sy1, sx2, sy2 = self.canvas.coords(src_oid)
        dx1, dy1, dx2, dy2 = self.canvas.coords(dst_oid)
        sx, sy = (sx1 + sx2) / 2, (sy1 + sy2) / 2
        ex, ey = (dx1 + dx2) / 2, (dy1 + dy2) / 2
        line = self.canvas.create_line(sx, sy, ex, ey,
                                       fill=self.PIN_OUT_COLOR, width=2,
                                       tags=("graph_link",))
        try:
            self.canvas.tag_raise("graph_link")
        except Exception:
            pass

        ok = self.engine.add_link(src_node, src_pin, dst_node, dst_pin, trigger=trigger)
        if not ok:
            self.canvas.delete(line)
            return False

        self.link_items[(src_node, src_pin, dst_node, dst_pin)] = line
        self._set_links_visibility(self._show_links)
        return True

    def _serialize_graph(self) -> tuple[list[dict], list[dict], dict]:
        nodes = []
        for node in self.canvas_nodes.values():
            progress_start = time.perf_counter()
            payload = node.serialize()
            payload["node_name"] = self._node_display_name(node)
            nodes.append(payload)
            if node.NODE_TYPE in {"array_viewer", "array_accumulator"}:
                shape = self._array_node_shape(node)
                self._append_project_progress(
                    f"Saving node name {self._node_display_name(node)} "
                    f"(id: {node.node_id}) ... array shape: {shape}. "
                    f"Completed (took {time.perf_counter() - progress_start:.3f} s)"
                )

        links = []
        for src_node, src_pin, dst_node, dst_pin in self.link_items.keys():
            links.append({
                "src_node": src_node,
                "src_pin": src_pin,
                "dst_node": dst_node,
                "dst_pin": dst_pin,
            })

        meta = {
            "saved_at_unix": time.time(),
            "node_count": len(nodes),
            "link_count": len(links),
            "canvas_width": self._canvas_width,
            "canvas_height": self._canvas_height,
        }
        return nodes, links, meta

    def _on_project_node_computed(self, node: BaseNode,
                                  elapsed: float) -> None:
        if getattr(self, "_project_progress_start", None) is None:
            return
        if node.NODE_TYPE not in {"array_viewer", "array_accumulator"}:
            return
        restore_elapsed = self._project_node_load_starts.pop(node.node_id, 0.0)
        total_elapsed = restore_elapsed + elapsed
        self._append_project_progress(
            f"Loading node name {self._node_display_name(node)} "
            f"(id: {node.node_id}) ... array shape: {self._array_node_shape(node)}. "
            f"Completed (took {total_elapsed:.3f} s)"
        )

    @staticmethod
    def _array_node_shape(node: BaseNode) -> str:
        array = getattr(node, "_current_array", None)
        if array is not None and hasattr(array, "shape"):
            return str(tuple(array.shape))
        max_row = getattr(node, "_max_row", 0)
        n_cols = getattr(node, "_n_cols", 0)
        if max_row and n_cols:
            return str((max_row, n_cols))
        return "(empty)"

    def _open_project_progress(self, operation: str) -> None:
        if (self._project_progress_window is not None
                and self._project_progress_window.winfo_exists()):
            self._project_progress_window.destroy()

        window = tk.Toplevel(self.root)
        window.title(f"Project {operation} progress")
        window.geometry("760x420")
        window.transient(self.root)
        window.protocol("WM_DELETE_WINDOW", window.destroy)

        text = tk.Text(window, wrap=tk.WORD, font=("Courier", 9),
                       state=tk.DISABLED)
        scrollbar = tk.Scrollbar(window, orient=tk.VERTICAL,
                                 command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(8, 4))
        tk.Button(window, text="Close", command=window.destroy).pack(
            anchor="e", padx=8, pady=(0, 8))

        self._project_progress_window = window
        self._project_progress_text = text
        self._project_progress_start = time.perf_counter()
        self._project_node_load_starts = {}
        self.engine.node_compute_callback = (
            self._on_project_node_computed if operation == "load" else None
        )
        self._append_project_progress(f"Starting project {operation}...")

    def _append_project_progress(self, message: str) -> None:
        text = getattr(self, "_project_progress_text", None)
        window = getattr(self, "_project_progress_window", None)
        if text is None or window is None or not window.winfo_exists():
            return
        try:
            text.configure(state=tk.NORMAL)
            text.insert(tk.END, message + "\n")
            text.see(tk.END)
            text.configure(state=tk.DISABLED)
            window.update_idletasks()
            window.update()
        except tk.TclError:
            self._project_progress_text = None
            self._project_progress_window = None

    def _finish_project_progress(self, operation: str) -> None:
        progress_start = getattr(self, "_project_progress_start", None)
        if progress_start is None:
            return

        # Clear all "a run is in progress" state BEFORE appending the message.
        # _append_project_progress() pumps the Tk event loop (window.update())
        # to force the dialog to redraw immediately, and that pump can process
        # a queued after() callback that re-enters this method (e.g. a
        # straggling BACKGROUND-node completion from an earlier, now-obsolete
        # execution). Clearing state first means that re-entrant call sees
        # progress_start is already None and returns immediately instead of
        # printing a second "Completed" line for the same load/save.
        self._project_progress_start = None
        self.engine.node_compute_callback = None
        self.engine.execution_complete_callback = None

        elapsed = time.perf_counter() - progress_start
        self._append_project_progress(
            f"Completed (totally took {elapsed:.3f} s to {operation} this project.)"
        )

    def _load_graph(self, nodes_data: list[dict], links_data: list[dict]) -> list[str]:
        self._clear_graph()
        issues: list[str] = []

        for n in nodes_data:
            node_type = str(n.get("node_type", ""))
            cls = self._node_type_map.get(node_type)
            if cls is None:
                issues.append(f"Unknown node type '{node_type}' for node_id '{n.get('node_id', '')}'.")
                continue

            node_id = str(n.get("node_id", f"node_{self._node_counter}"))
            x = int(n.get("x", 120))
            y = int(n.get("y", 150))
            width = int(n.get("width", cls.NODE_WIDTH))
            height = int(n.get("height", cls.NODE_HEIGHT))
            params = n.get("params", {}) or {}
            node_name = str(n.get("node_name", "") or "")
            self._create_node_instance(cls, node_id, x, y, width, height, params, node_name=node_name)

        for lk in links_data:
            # trigger=False: restoring N links would otherwise recompute the
            # downstream subgraph N times (once per add_link call) before the
            # trigger_all() below recomputes the whole graph once more. That
            # redundant work is the main reason a load can take far longer
            # than a save, and it also multiplies "node computed" / progress
            # messages during load. We defer all computation to the single
            # trigger_all() call after every link has been restored.
            ok = self._draw_link(
                str(lk.get("src_node", "")),
                str(lk.get("src_pin", "")),
                str(lk.get("dst_node", "")),
                str(lk.get("dst_pin", "")),
                trigger=False,
            )
            if not ok:
                issues.append(
                    "Could not restore link "
                    f"{lk.get('src_node', '')}.{lk.get('src_pin', '')} -> "
                    f"{lk.get('dst_node', '')}.{lk.get('dst_pin', '')}."
                )

        if getattr(self, "_project_progress_start", None) is not None:
            self.engine.execution_complete_callback = (
                lambda: self._finish_project_progress("load")
            )
        self.engine.trigger_all()
        return issues

    def _show_load_report(self, issues: list[str]) -> None:
        if not issues:
            return
        max_lines = 25
        shown = issues[:max_lines]
        hidden_count = max(0, len(issues) - len(shown))
        msg = "Load completed with warnings:\n\n" + "\n".join(f"- {line}" for line in shown)
        if hidden_count:
            msg += f"\n\n... and {hidden_count} more warnings."
        messagebox.showwarning("Load Report", msg, parent=self.root)
        # also display msg in console for debugging
        print(msg, file=sys.stderr)

    def _clear_graph(self) -> None:
        self._clear_selection()
        self._cancel_pin_hover_after()
        self._hover_pin_item = None
        self._hover_pin_candidate = None
        self._cancel_link_hover_after()
        self._hover_link_key = None
        self._hover_link_candidate = None
        self.canvas.delete("all")

        for node_id in list(self.canvas_nodes.keys()):
            self.engine.remove_node(node_id)

        self.canvas_nodes = {}
        self.link_items = {}
        self._pin_id_by_key = {}
        self._pin_key_by_id = {}
        self._pin_id_counter = 1
        self._drag = {"mode": None, "node_id": None, "ox": 0, "oy": 0}
        self._resize = {"active": False, "node_id": None, "handle": None,
                        "x": 0, "y": 0, "w": 0, "h": 0}
        self._linking = {"active": False, "line": None,
                         "src_node": None, "src_pin": None, "src_type": None}

    def _mark_dirty(self) -> None:
        if self._suspend_dirty:
            return
        self._dirty = True
        self._update_title()

    def _set_clean(self) -> None:
        self._dirty = False
        self._update_title()

    def _set_project_path(self, path: str | None) -> None:
        if path:
            self._project_path = str(Path(path).expanduser().resolve())
        else:
            self._project_path = None
        set_project_file_path(self._project_path)

    def _update_title(self) -> None:
        if self._project_path:
            name = Path(self._project_path).name
            project_dir = get_project_directory()
            dir_text = str(project_dir) if project_dir is not None else ""
        else:
            name = "Untitled"
            dir_text = "No project path"
        dirty_mark = "*" if self._dirty else ""
        self.root.title(f"Node Editor - {name} [{dir_text}]{dirty_mark}")

    def _confirm_discard_if_dirty(self) -> bool:
        if not self._dirty:
            return True

        choice = messagebox.askyesnocancel(
            "Unsaved Changes",
            "Save changes before continuing?",
            parent=self.root,
        )
        if choice is None:
            return False
        if choice:
            return self._file_save()
        return True

    def _file_new(self) -> None:
        if not self._confirm_discard_if_dirty():
            return
        self._suspend_dirty = True
        try:
            self._clear_graph()
            self._node_counter = 0
            self._set_project_path(None)
            self._set_canvas_size(self.DEFAULT_CANVAS_WIDTH, self.DEFAULT_CANVAS_HEIGHT)
        finally:
            self._suspend_dirty = False
        self._set_clean()

    # Minimum gap required between two load triggers; guards against a
    # trailing duplicate event (e.g. an OS key-repeat echo) that gets
    # dispatched right after the previous _file_load call has already
    # returned and cleared the in-progress flag.
    _FILE_LOAD_DEBOUNCE_SECONDS = 1.0

    def _file_load(self) -> None:
        # Defense-in-depth re-entrancy guard. The flag must be set BEFORE any
        # call that can pump the Tk event loop (messagebox, file dialog,
        # window.update() inside the progress reporting) — any of those can
        # dispatch a queued/duplicate load trigger (double key-repeat event,
        # a stray after() re-fire, a future accidental double bind, etc.)
        # reentrantly while this call is still on the stack. Guarding only
        # around the heavy load step would leave that earlier window open.
        if self._file_load_in_progress:
            print("Ignored duplicate project-load trigger (a load is already in progress).")
            return
        last_finished = self._file_load_last_finished_at
        if last_finished is not None and (time.monotonic() - last_finished) < self._FILE_LOAD_DEBOUNCE_SECONDS:
            print("Ignored duplicate project-load trigger (arrived immediately after a previous load).")
            return
        self._file_load_in_progress = True
        try:
            if not self._confirm_discard_if_dirty():
                return

            path = filedialog.askopenfilename(
                parent=self.root,
                title="Load Project",
                filetypes=[("Excel files", "*.xlsx")],
                defaultextension=".xlsx",
            )
            if not path:
                return

            selected_path = str(Path(path).expanduser().resolve())
            previous_path = self._project_path

            self._open_project_progress("load")
            try:
                nodes, links, meta = load_project(selected_path)
                self._suspend_dirty = True
                try:
                    self._set_project_path(selected_path)
                    canvas_size_issue = self._restore_canvas_size(meta)
                    issues = self._load_graph(nodes, links)
                finally:
                    self._suspend_dirty = False
                self._set_clean()
                if canvas_size_issue:
                    issues.append(canvas_size_issue)
                issues.extend(list(meta.get("load_issues", [])) if isinstance(meta, dict) else [])
                self._show_load_report(issues)
                if self.engine.execution_complete_callback is None:
                    self._finish_project_progress("load")
            except (RuntimeError, ProjectFormatError, OSError, ValueError) as e:
                self._append_project_progress(f"Load failed: {e}")
                self._finish_project_progress("load")
                self._set_project_path(previous_path)
                messagebox.showerror("Load Failed", str(e), parent=self.root)
        finally:
            self._file_load_in_progress = False
            self._file_load_last_finished_at = time.monotonic()

    def _file_save(self) -> bool:
        if not self._project_path:
            return self._file_save_as()

        try:
            self._open_project_progress("save")
            nodes, links, meta = self._serialize_graph()
            save_project(self._project_path, nodes, links, meta)
            self._set_clean()
            self._finish_project_progress("save")
            return True
        except (RuntimeError, OSError, ValueError) as e:
            self._append_project_progress(f"Save failed: {e}")
            self._finish_project_progress("save")
            messagebox.showerror("Save Failed", str(e), parent=self.root)
            return False

    def _file_save_as(self) -> bool:
        path = filedialog.asksaveasfilename(
            parent=self.root,
            title="Save Project As",
            filetypes=[("Excel files", "*.xlsx")],
            defaultextension=".xlsx",
        )
        if not path:
            return False

        previous_path = self._project_path
        self._set_project_path(path)
        ok = self._file_save()
        if not ok:
            self._set_project_path(previous_path)
        return ok

    @staticmethod
    def _parse_pin_tag(tags: tuple, direction: str) -> tuple[str | None, str | None]:
        """
        Parse (node_id, pin_name) from canvas tags.
        Tag format: pin_{node_id}_{in|out}_{pin_name}
        """
        prefix = f"pin_"
        for tag in tags:
            if tag.startswith(prefix) and f"_{direction}_" in tag:
                # pin_node_0_out_result
                parts = tag.split(f"_{direction}_", 1)
                node_id  = parts[0][len(prefix):]
                pin_name = parts[1]
                return node_id, pin_name
        return None, None

    # ══ Lifecycle ═════════════════════════════════════════════════

    def _on_close(self) -> None:
        if not self._confirm_discard_if_dirty():
            return
        self.engine.shutdown()
        self.root.destroy()

    def register_category_label(self, category: str) -> None:
        """Add a section divider label in the toolbox."""
        self._toolbox_register_category = category
        self._ensure_toolbox_category(category)
        self._refresh_toolbox_palette()
    
