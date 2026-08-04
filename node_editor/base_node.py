# base_node.py

import tkinter as tk
from abc import ABC, abstractmethod
from typing import Optional, Callable, Any

from node_editor.pin_types import PinSchema, PinType
from node_editor.execution import ExecutionMode


class BaseNode(ABC):
    """
    Abstract base class for all nodes.

    Subclasses must implement:
        get_pin_schema()  - declare input / output pins
        build_body()      - create the node UI on the canvas
        compute()         - computation logic for SYNC / BACKGROUND nodes

    STREAMING subclasses should implement:
        start_stream()    - start producing data continuously
        stop_stream()     - stop streaming
    and call self.push_output() to deliver new data.
    """

    # ── Class attributes overridden by subclasses ──────────────────
    EXECUTION_MODE: ExecutionMode = ExecutionMode.SYNC
    NODE_TYPE:      str           = "base"    # Unique identifier used for serialization
    DISPLAY_NAME:   str           = "Base"    # Displayed in the Toolbox and node title
    CATEGORY:       str           = "misc"    # "source" / "process" / "visualize"
    SEARCH_KEYWORDS: tuple[str, ...] = ()      # Optional toolbox-search aliases

    # ── Default node appearance, overridable by subclasses ─────────
    NODE_WIDTH:  int = 140
    NODE_HEIGHT: int = 160
    MIN_WIDTH:   int = 80
    MIN_HEIGHT:  int = 60
    BODY_COLOR:  str = "#e1e1e1"
    TITLE_COLOR: str = "#333333"

    # ─────────────────────────────────────────────────────────────
    def __init__(self, node_id: str, canvas: tk.Canvas):
        self.node_id  = node_id          # Unique ID assigned by the editor
        self.canvas   = canvas

        # Callbacks injected by the engine (set by the engine, not called by subclasses)
        self._on_output_ready: Optional[Callable[[str, dict], None]] = None
        self._request_downstream: Optional[Callable[[str], None]]    = None

        # Execution state
        self._is_running: bool = False   # Used by STREAMING nodes

        # Canvas item IDs (filled in by build_body)
        self._canvas_items: list[int] = []   # All created canvas items
        self._body_rect:    Optional[int] = None
        self._title_item:   Optional[int] = None

        # Pin canvas item IDs (created by the editor after build_body and stored here)
        self.input_pin_items:  dict[str, int] = {}   # pin name -> canvas oval id
        self.output_pin_items: dict[str, int] = {}

        # Current node position (managed by the editor)
        self.x: int = 0
        self.y: int = 0
        self.width: int = self.NODE_WIDTH
        self.height: int = self.NODE_HEIGHT

        # Optional non-modal inspector window (Phase 1 infrastructure).
        self._inspector_win: Optional[tk.Toplevel] = None
        self._inspector_body: Optional[tk.Frame] = None

    # ══ 1. Abstract methods subclasses must implement ═════════════

    @abstractmethod
    def get_pin_schema(self) -> PinSchema:
        """
        Declare this node's input / output pins.
        The editor uses this information to draw pins and validate type compatibility.
        """
        ...

    @abstractmethod
    def build_body(self) -> None:
        """
        Create all widgets / canvas items for the node body on self.canvas.
        Positioning is based on self.x and self.y.
        Append every created canvas item id to self._canvas_items.
        """
        ...

    def build_inspector(self, parent: tk.Frame) -> None:
        """
        Build the full node UI in a non-modal popup window.
        Phase 1 default implementation shows a placeholder.
        Subclasses can override to provide rich controls.
        """
        tk.Label(
            parent,
            text="No inspector UI for this node yet.",
            anchor="w",
            justify="left",
            font=("Arial", 9),
        ).pack(fill="x")

    def get_inspector_title(self) -> str:
        node_name = getattr(self, "node_name", "")
        if isinstance(node_name, str) and node_name.strip():
            return node_name.strip()
        return self.DISPLAY_NAME

    def is_inspector_open(self) -> bool:
        return self._inspector_win is not None and self._inspector_win.winfo_exists()

    def open_inspector(self) -> None:
        """Open (or focus) a non-modal inspector popup for this node."""
        if self.is_inspector_open():
            self._inspector_win.deiconify()
            self._inspector_win.lift()
            self._inspector_win.focus_force()
            return

        top = self.canvas.winfo_toplevel()
        win = tk.Toplevel(top)
        win.title(f"{self.get_inspector_title()} - Inspector")
        win.resizable(True, True)
        try:
            # Keep inspector above the main editor window without forcing global topmost.
            win.transient(top)
        except Exception:
            pass

        try:
            px, py = self.canvas.winfo_pointerxy()
            win.geometry(f"+{px + 16}+{py + 16}")
        except Exception:
            pass

        body = tk.Frame(win, padx=8, pady=8)
        body.pack(fill="both", expand=True)

        self._inspector_win = win
        self._inspector_body = body

        try:
            self.build_inspector(body)
        except Exception as e:
            for child in body.winfo_children():
                child.destroy()
            tk.Label(
                body,
                text=f"Failed to build inspector UI: {e}",
                anchor="w",
                justify="left",
                fg="#aa0000",
                font=("Arial", 9),
            ).pack(fill="x")

        win.protocol("WM_DELETE_WINDOW", self.close_inspector)

    def close_inspector(self) -> None:
        if self._inspector_win is not None:
            try:
                if self._inspector_win.winfo_exists():
                    self._inspector_win.destroy()
            except Exception:
                pass
        self._inspector_win = None
        self._inspector_body = None

    # ══ 2. Implement this for SYNC / BACKGROUND nodes ══════════════

    def compute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Receive upstream data and return output data.
        inputs  key = PinDef.name of input pins
        outputs key = PinDef.name of output pins

        STREAMING nodes do not use this method and do not need to override it (default returns an empty dict).
        """
        return {}

    # ══ 3. STREAMING nodes implement these two methods ════════════

    def start_stream(self) -> None:
        """
        Start producing data continuously.
        Subclasses can start a thread here and call self.push_output() whenever new data is available.
        Default no-op implementation; SYNC / BACKGROUND nodes do not need to override it.
        """
        pass

    def stop_stream(self) -> None:
        """
        Stop data production and release resources (webcam, file handles, etc.).
        """
        pass

    def push_output(self, outputs: dict[str, Any]) -> None:
        """
        Call this method when a STREAMING node has new data.
        The engine takes over and safely drives downstream computation back on the main thread.
        Subclasses should not override this method.
        """
        if self._on_output_ready:
            self._on_output_ready(self.node_id, outputs)

    # ══ 4. UI status display (subclasses may override) ════════════

    def set_status(self, status: str, color: str = "#666666") -> None:
        """
        Display status text in the node body (for example "running" or "error: ...").
        The default implementation updates the title item's color; subclasses can override for richer UI.
        """
        if self._title_item:
            self.canvas.itemconfig(self._title_item, fill=color)

    def on_upstream_changed(self) -> None:
        """
        Called by the engine when upstream connections change (links added or removed).
        Subclasses can override this to update the UI, for example to show a "waiting for input" prompt.
        Default no-op implementation.
        """
        pass

    # ══ 5. Serialization interface ═════════════════════════════════

    def serialize(self) -> dict:
        """
        Return a JSON-serializable dict for saving.
        When overriding, subclasses should call super().serialize() first and then add their own fields.
        """
        return {
            "node_id":   self.node_id,
            "node_type": self.NODE_TYPE,
            "x":         self.x,
            "y":         self.y,
            "width":     self.width,
            "height":    self.height,
            "params":    self.get_params(),
        }

    def deserialize(self, data: dict) -> None:
        """
        Restore node state from a dict (called when loading saved data).
        When overriding, subclasses should call super().deserialize(data) first.
        This method is called after build_body(), so the widgets already exist.
        """
        self.x = data.get("x", 0)
        self.y = data.get("y", 0)
        self.width = int(data.get("width", self.NODE_WIDTH))
        self.height = int(data.get("height", self.NODE_HEIGHT))
        self.set_params(data.get("params", {}))

    def set_size(self, width: int, height: int) -> None:
        self.width = max(self.MIN_WIDTH, int(width))
        self.height = max(self.MIN_HEIGHT, int(height))

    def on_resize(self, old_width: int, old_height: int,
                  new_width: int, new_height: int) -> None:
        """
        Called after the editor updates the node size.
        Subclasses can override when they need to reposition custom canvas items.
        """
        pass

    def get_params(self) -> dict:
        """
        Return internal node parameters (Entry values, combobox selections, etc.).
        Used for serialization. Override in subclasses.
        """
        return {}

    def set_params(self, params: dict) -> None:
        """
        Restore internal node parameters from a dict.
        Used for deserialization. Override in subclasses.
        """
        pass

    # ══ 6. Lifecycle ═══════════════════════════════════════════════

    def on_destroy(self) -> None:
        """
        Called when the node is deleted.
        Subclasses should release resources here: stop threads, close webcams, release file handles.
        The editor calls this before deleting canvas items.
        """
        self.close_inspector()
        if self.EXECUTION_MODE == ExecutionMode.STREAMING:
            self.stop_stream()

#    def on_move(self, dx: int, dy: int) -> None:
#        """
#        Repositions all create_window() items belonging to this node.
#        Works for most nodes automatically.
#        Override only if you need custom behaviour beyond this.
#        """
#        for item in self.canvas.find_withtag(self.node_id):
#            if self.canvas.type(item) == "window":
#                x, y = self.canvas.coords(item)
#                self.canvas.coords(item, x + dx, y + dy)
