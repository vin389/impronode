import tkinter as tk

import pytest

from node_editor import tooltip
from node_editor.tooltip import add_tooltip, add_tooltips_by_text


class FakeNode:
    NODE_TYPE = "fake_node_type"

    def build(self, parent):
        self._copy_sel_entry = e = tk.Entry(parent)               # attribute held via a local name
        add_tooltip(e, "frames")
        self._vars = {"square": tk.StringVar(master=parent)}
        tk.Label(parent, text="Square size (mm):")
        sq = tk.Entry(parent, textvariable=self._vars["square"])
        add_tooltip(sq, "square")                                 # named from its variable
        free = tk.Frame(parent)
        tk.Label(free, text="Max width:")
        mw = tk.Entry(free)
        add_tooltip(mw, "width")                                  # named from the label before it
        b1, b2 = tk.Button(parent, text="Load..."), tk.Button(parent, text="Load...")
        add_tooltip(b1, "a")
        add_tooltip(b2, "b")                                      # same caption: numbered
        add_tooltip(tk.Button(parent, text="Run"), "run", name="my_custom_name")
        tk.Label(parent, text="Columns: inner corners")
        return e, sq, mw, b1, b2


@pytest.fixture
def root():
    r = tk.Tk()
    r.withdraw()
    yield r
    r.destroy()


def test_names(root):
    win = tk.Toplevel(root)
    node = FakeNode()
    e, sq, mw, b1, b2 = node.build(win)
    names = {w: w._impronode_tooltip.name for w in (e, sq, mw, b1, b2)}
    assert names[e] == "fake_node_type__copy_sel_entry"
    assert names[sq] == "fake_node_type__square_entry"
    assert names[mw] == "fake_node_type__max_width_entry"
    assert names[b1] == "fake_node_type__load_button" and names[b2] == "fake_node_type__load_button_2"
    custom = [w for w in win.winfo_children() if getattr(w, "_impronode_tooltip", None)
              and w._impronode_tooltip.name == "my_custom_name"]
    assert custom
    add_tooltips_by_text(win, {"Columns:": "inner corners per row"})
    label = [w for w in win.winfo_children() if isinstance(w, tk.Label) and w.cget("text").startswith("Columns")][0]
    assert label._impronode_tooltip.name == "test_tooltip_names__columns_label"
    # the same widgets in another window (e.g. the inspector reopened) get the same names
    win2 = tk.Toplevel(root)
    e2, *_ = FakeNode().build(win2)
    assert e2._impronode_tooltip.name == "fake_node_type__copy_sel_entry"


def test_tooltip_shows_the_name_last(root, monkeypatch):
    monkeypatch.setattr(tooltip, "_enabled", True)
    b = tk.Button(root, text="Go")
    t = add_tooltip(b, "Start it.", name="x__go_button")
    t._show()
    texts = [w.cget("text") for w in t._tip.winfo_children()[0].winfo_children()]
    assert texts == ["Start it.", "(widget name: x__go_button)"]
    t._hide()
