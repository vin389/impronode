# node_editor/tooltip.py
"""Small hover tooltips for Tk widgets, shared by all nodes' inspectors.

    from node_editor.tooltip import add_tooltip
    add_tooltip(button, "What it does, and why you would use it.")

The text is wrapped automatically (pass newlines to force breaks). A
tooltip also shows on a DISABLED widget -- often exactly when the user
wonders what it is for.

All tooltips can be switched off (set_tooltips_enabled; e.g. when they
cover something important); the choice is remembered per user in
~/.impronode/settings.json.
"""

import json
import os
import textwrap
import tkinter as tk
from pathlib import Path

_WRAP_CHARS = 60
_SETTINGS = Path.home() / ".impronode" / "settings.json"


def _load_enabled() -> bool:
    try:
        return bool(json.loads(_SETTINGS.read_text(encoding="utf-8")).get("tooltips", True))
    except (OSError, ValueError, AttributeError):
        return True


_enabled = _load_enabled()
_listeners: list = []


def tooltips_enabled() -> bool:
    return _enabled


def set_tooltips_enabled(enabled: bool) -> None:
    """Switch every tooltip on / off (and remember it); listeners added with
    on_tooltips_changed() are told, e.g. to update their check boxes."""
    global _enabled
    _enabled = bool(enabled)
    try:
        data = json.loads(_SETTINGS.read_text(encoding="utf-8")) if _SETTINGS.exists() else {}
        if not isinstance(data, dict):
            data = {}
        data["tooltips"] = _enabled
        _SETTINGS.parent.mkdir(parents=True, exist_ok=True)
        tmp = _SETTINGS.with_name(_SETTINGS.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, _SETTINGS)
    except (OSError, ValueError):
        pass
    for fn in list(_listeners):
        try:
            fn(_enabled)
        except (tk.TclError, RuntimeError):         # its widget / variable is gone
            _listeners.remove(fn)


def on_tooltips_changed(fn) -> None:
    _listeners.append(fn)


def tooltip_toggle_var(master) -> tk.BooleanVar:
    """A BooleanVar that follows the global switch (for a Checkbutton /
    menu checkbutton: command=lambda: set_tooltips_enabled(var.get()))."""
    var = tk.BooleanVar(master=master, value=_enabled)
    on_tooltips_changed(var.set)
    return var


class Tooltip:
    """Show `text` in a small yellow box after the pointer rests on `widget`."""

    def __init__(self, widget, text: str, delay_ms: int = 500):
        self._widget, self._delay = widget, delay_ms
        self._text = self._wrap(text)
        self._after = None
        self._tip = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")
        widget.bind("<Destroy>", self._hide, add="+")

    @staticmethod
    def _wrap(text: str) -> str:
        return "\n".join(textwrap.fill(par, _WRAP_CHARS) if par.strip() else ""
                         for par in text.split("\n"))

    def set_text(self, text: str) -> None:
        self._text = self._wrap(text)

    def _schedule(self, _e=None):
        self._hide()
        if not _enabled:
            return
        try:
            self._after = self._widget.after(self._delay, self._show)
        except tk.TclError:
            self._after = None

    def _show(self):
        self._after = None
        try:
            x = self._widget.winfo_rootx() + 10
            y = self._widget.winfo_rooty() + self._widget.winfo_height() + 4
            self._tip = tip = tk.Toplevel(self._widget)
            tip.wm_overrideredirect(True)
            label = tk.Label(tip, text=self._text, font=("Arial", 8), bg="#ffffe0", justify="left",
                             relief=tk.SOLID, borderwidth=1, padx=4, pady=2)
            label.pack()
            # Keep the box on the screen (inspectors often sit at the right edge).
            tip.update_idletasks()
            sw, sh = tip.winfo_screenwidth(), tip.winfo_screenheight()
            w, h = tip.winfo_reqwidth(), tip.winfo_reqheight()
            x = max(0, min(x, sw - w - 4))
            if y + h > sh:
                y = max(0, self._widget.winfo_rooty() - h - 4)
            tip.wm_geometry(f"+{x}+{y}")
        except tk.TclError:
            self._tip = None

    def _hide(self, _e=None):
        if self._after is not None:
            try:
                self._widget.after_cancel(self._after)
            except tk.TclError:
                pass
            self._after = None
        if self._tip is not None:
            try:
                self._tip.destroy()
            except tk.TclError:
                pass
            self._tip = None


def add_tooltip(widget, text: str, delay_ms: int = 500) -> Tooltip:
    """Attach a tooltip to `widget`; returns it (set_text() can change it).
    A widget that already has one keeps it, with the new text -- so calling
    this twice never stacks two tooltips."""
    existing = getattr(widget, "_impronode_tooltip", None)
    if existing is not None:
        existing.set_text(text)
        return existing
    tip = Tooltip(widget, text, delay_ms)
    widget._impronode_tooltip = tip
    return tip


def add_tooltips_by_text(container, tips: dict[str, str], with_inputs: bool = True) -> None:
    """Attach tooltips to the widgets inside `container` (recursively) whose
    'text' option STARTS WITH one of the keys of `tips` -- handy for long
    inspectors whose buttons / radio buttons / labels are not kept in
    variables. Each widget gets at most one tooltip (the first key that
    matches, in dict order).

    with_inputs: a matching LABEL also passes its tooltip to the input
    widgets created right after it (entries, spinboxes, combo boxes,
    sliders) -- the usual "label + box" row."""
    stack = [container]
    while stack:
        widget = stack.pop()
        stack.extend(widget.winfo_children())
        try:
            text = str(widget.cget("text"))
        except tk.TclError:
            continue
        for prefix, tip in tips.items():
            if text.startswith(prefix):
                add_tooltip(widget, tip)
                if with_inputs and widget.winfo_class() in ("Label", "TLabel"):
                    # The input boxes created right after the label (e.g. "w" and
                    # "h" entries), up to the next widget that is not an input.
                    siblings = widget.master.winfo_children()
                    for sibling in siblings[siblings.index(widget) + 1:]:
                        if not _is_input(sibling):
                            break
                        add_tooltip(sibling, tip)
                break


_INPUT_CLASSES = ("Entry", "TEntry", "Spinbox", "TSpinbox", "TCombobox", "Scale", "TScale")


def _is_input(widget) -> bool:
    return widget.winfo_class() in _INPUT_CLASSES
