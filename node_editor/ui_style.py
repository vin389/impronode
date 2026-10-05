# node_editor/ui_style.py
"""
Shared ttk styling for the node editor and every node's inspector UI.

apply_notebook_tab_style() restyles the DEFAULT ttk.Notebook tab
("TNotebook.Tab"), so every ttk.Notebook in the application -- in any
node's inspector -- gets clearly visible tabs without per-node code.
"""

import tkinter as tk
from tkinter import ttk

TAB_BG = "#cfd6e0"           # unselected tab
TAB_BG_ACTIVE = "#e2e8f0"    # mouse over an unselected tab
TAB_BG_SELECTED = "#ffffff"  # selected tab
TAB_BORDER = "#6b7685"
TAB_LIGHT = "#e4e9f0"        # top/left bevel
TAB_DARK = "#aab3c0"         # bottom/right bevel

_TAB_ELEMENT = "Visible.tab"


def apply_notebook_tab_style(root: tk.Misc | None = None) -> None:
    """Give every ttk.Notebook visible, bordered tabs.

    The native Windows ('vista') theme draws tabs from the OS theme and
    ignores border / colour options, so unselected tabs look like plain
    text and users do not notice there are other tabs. This borrows the
    'clam' theme's tab element (which honours bordercolor / background)
    for the tab only -- buttons, entries, the notebook body etc. keep the
    native look. Safe to call more than once.
    """
    st = ttk.Style(root)
    if _TAB_ELEMENT not in st.element_names():
        try:
            st.element_create(_TAB_ELEMENT, "from", "clam", "tab")
        except tk.TclError:
            return                                   # very old Tk: keep the default look
    st.layout("TNotebook.Tab", [(_TAB_ELEMENT, {"sticky": "nswe", "children": [
        ("Notebook.padding", {"side": "top", "sticky": "nswe", "children": [
            ("Notebook.label", {"side": "top", "sticky": ""})]})]})])
    st.configure("TNotebook.Tab", background=TAB_BG, bordercolor=TAB_BORDER, lightcolor=TAB_LIGHT,
                 darkcolor=TAB_DARK, padding=(10, 3))
    st.map("TNotebook.Tab", background=[("selected", TAB_BG_SELECTED), ("active", TAB_BG_ACTIVE)],
           lightcolor=[("selected", TAB_BG_SELECTED)], expand=[("selected", (1, 2, 1, 0))])
