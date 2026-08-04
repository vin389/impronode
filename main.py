# main.py

import tkinter as tk
from pathlib import Path

from node_editor.node_editor_app import NodeEditorApp
from node_editor.node_registry   import discover_nodes

# Category display order in the toolbox
CATEGORY_ORDER = ["source", "process", "visualize", "misc"]

if __name__ == "__main__":
    root = tk.Tk()
    app  = NodeEditorApp(root)

    nodes_path = Path(__file__).parent / "node_editor" / "nodes"
    registry   = discover_nodes(nodes_path)
    app.set_node_registry(registry)

    # Register in category order, with a separator label per category
    for category in CATEGORY_ORDER:
        entries = registry.get(category, [])
        if not entries:
            continue
        app.register_category_label(category)   # ← new, see below
        app.register_node_types(entries)

    # catch any categories not in CATEGORY_ORDER
    for category, entries in registry.items():
        if category not in CATEGORY_ORDER:
            app.register_category_label(category)
            app.register_node_types(entries)

    root.mainloop()
