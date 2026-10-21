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

WIDGET NAMES. The last line of every tooltip is "(widget name: X)": a
name a user / developer can quote to refer to the widget exactly (e.g.
when asking an AI coder to change it). X = "<node type>__<widget>", made
when the tooltip is attached, e.g. mesh_template_track__copy_sel_entry:
  <node type>  NODE_TYPE of the node whose code attached the tooltip (else
               the module's file name without "_node(s)");
  <widget>     the attribute the widget is kept in (self._copy_sel_entry ->
               copy_sel_entry), else the widget's own text, else the text
               of the label in front of it, plus its kind (button, entry,
               ...);
  a number is appended when two widgets of one window would get the
  same name. Search the code for the attribute / the text to find it.
Pass name="..." to add_tooltip to choose a name yourself.
"""

import json
import linecache
import os
import re
import sys
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


# ── widget names ──────────────────────────────────────────────────────────

_KINDS = {"Button": "button", "TButton": "button", "Checkbutton": "check", "TCheckbutton": "check",
          "Radiobutton": "radio", "TRadiobutton": "radio", "Label": "label", "TLabel": "label",
          "Entry": "entry", "TEntry": "entry", "TCombobox": "combo", "Spinbox": "spin", "TSpinbox": "spin",
          "Scale": "slider", "TScale": "slider", "Text": "text", "Treeview": "table", "Canvas": "canvas",
          "Listbox": "list", "Frame": "frame", "TFrame": "frame", "Labelframe": "group",
          "TLabelframe": "group", "Menubutton": "menu", "TMenubutton": "menu", "TProgressbar": "progress",
          "TNotebook": "tabs"}
_CALL = re.compile(r"add_tooltip\(\s*(self\.[A-Za-z_][\w.]*|[A-Za-z_]\w*)\s*(\[)?")
_names_by_window: dict = {}          # toplevel path -> {name: widget}


def _slug(text: str, limit: int = 40) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(text)).strip("_").lower()
    return s[:limit].rstrip("_")


def _owner(frame) -> str:
    """Node type of the code that attaches the tooltip, else its module."""
    slf = frame.f_locals.get("self") if frame is not None else None
    node_type = getattr(type(slf), "NODE_TYPE", None) if slf is not None else None
    if isinstance(node_type, str) and node_type and node_type != "base":
        return node_type
    stem = Path(frame.f_code.co_filename).stem if frame is not None else "app"
    return re.sub(r"_nodes?$", "", stem)


def _kind(widget) -> str:
    try:
        return _KINDS.get(widget.winfo_class(), _slug(widget.winfo_class()) or "widget")
    except tk.TclError:
        return "widget"


def _own_text(widget) -> str:
    """The widget's caption -- only a real 'text' option (an Entry has none;
    Tk would read cget('text') as an abbreviation of 'textvariable')."""
    try:
        return str(widget.cget("text")).strip() if "text" in widget.keys() else ""
    except (tk.TclError, AttributeError):
        return ""


def _self_attr(slf, match) -> str:
    """Name of the attribute of `slf` (or 'attr_key' of a dict attribute)
    whose value satisfies match(value), or ''."""
    if slf is None:
        return ""
    try:
        items = list(vars(slf).items())
    except TypeError:
        return ""
    for attr, value in items:
        try:
            if match(value):
                return _slug(attr)
        except Exception:
            continue
    for attr, value in items:
        if isinstance(value, dict) and len(value) < 200:
            for key, v in list(value.items()):
                try:
                    if match(v):
                        return _slug(f"{attr}_{key}")
                except Exception:
                    continue
    return ""


def _var_attr(widget, slf) -> str:
    """Attribute name of the Tk variable the widget shows ('copy_frames_var'
    -> 'copy_frames'), or ''."""
    try:
        if "textvariable" in widget.keys():
            var_name = str(widget.cget("textvariable"))
        elif "variable" in widget.keys():
            var_name = str(widget.cget("variable"))
        else:
            return ""
    except tk.TclError:
        return ""
    if not var_name:
        return ""
    attr = _self_attr(slf, lambda v: isinstance(v, tk.Variable) and str(v) == var_name)
    return re.sub(r"_?vars?(?=_|$)", "", attr).strip("_") if attr else ""


def _label_before(widget) -> str:
    """Text of the nearest label created before `widget` in its parent (the
    usual 'label + entry' pair)."""
    try:
        siblings = widget.master.winfo_children()
        k = siblings.index(widget)
    except (tk.TclError, ValueError, AttributeError):
        return ""
    for w in reversed(siblings[:k]):
        if w.winfo_class() in ("Label", "TLabel"):
            return _own_text(w)
        if _own_text(w):                              # another captioned widget in between
            return ""
    return ""


def _attr_from_call(frame) -> tuple[str, bool]:
    """('copy_sel_entry', subscripted) from a call 'add_tooltip(self._copy_sel_entry, ...'."""
    if frame is None:
        return "", False
    filename, line = frame.f_code.co_filename, frame.f_lineno
    for ln in range(line, max(0, line - 6), -1):        # a call may span several lines
        m = _CALL.search(linecache.getline(filename, ln))
        if m:
            expr = m.group(1)
            if expr.startswith("self."):
                return _slug(expr.rsplit(".", 1)[1]), bool(m.group(2))
            return "", False
    return "", False


def auto_name(widget, frame=None, key_text: str = "") -> str:
    """The widget's name (see the module docstring), unique in its window."""
    owner = _owner(frame)
    slf = frame.f_locals.get("self") if frame is not None else None
    kind = _kind(widget)
    caption = key_text or _own_text(widget)
    held = "" if key_text else _self_attr(slf, lambda v: v is widget)      # self._copy_sel_entry = e = ...
    attr, subscripted = _attr_from_call(frame) if not (key_text or held) else ("", False)
    if held:
        part = held
    elif attr and not subscripted:
        part = attr
    elif attr:                                         # self._btn_pick[which]: add the button's text
        part = attr + ("_" + _slug(caption) if caption else "")
    elif caption:
        part = f"{_slug(caption)}_{kind}"
    else:
        var = _var_attr(widget, slf)
        label = "" if var else _label_before(widget)
        part = f"{var}_{kind}" if var else (f"{_slug(label)}_{kind}" if label else kind)
    name = f"{owner}__{part}"
    try:
        window = str(widget.winfo_toplevel())
    except tk.TclError:
        return name
    used = _names_by_window.setdefault(window, {})
    for other_name, other in list(used.items()):      # forget destroyed widgets
        try:
            if not other.winfo_exists():
                del used[other_name]
        except tk.TclError:
            del used[other_name]
    unique, k = name, 2
    while unique in used and used[unique] is not widget:
        unique, k = f"{name}_{k}", k + 1
    used[unique] = widget
    return unique


class Tooltip:
    """Show `text` in a small yellow box after the pointer rests on `widget`;
    the last line names the widget (see 'WIDGET NAMES' above)."""

    def __init__(self, widget, text: str, delay_ms: int = 500, name: str = ""):
        self._widget, self._delay = widget, delay_ms
        self.name = name
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
            box = tk.Frame(tip, bg="#ffffe0", relief=tk.SOLID, borderwidth=1)
            box.pack()
            tk.Label(box, text=self._text, font=("Arial", 8), bg="#ffffe0", justify="left",
                     padx=4, pady=2).pack(anchor="w")
            if self.name:
                tk.Label(box, text=f"(widget name: {self.name})", font=("Consolas", 7), fg="#7a7a7a",
                         bg="#ffffe0", justify="left", padx=4, pady=0).pack(anchor="w", pady=(0, 2))
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


def add_tooltip(widget, text: str, delay_ms: int = 500, name: str | None = None,
                _frame=None, _key_text: str = "") -> Tooltip:
    """Attach a tooltip to `widget`; returns it (set_text() can change it).
    A widget that already has one keeps it, with the new text -- so calling
    this twice never stacks two tooltips. `name`: the widget name shown on
    the last line (default: made automatically, see 'WIDGET NAMES')."""
    existing = getattr(widget, "_impronode_tooltip", None)
    if existing is not None:
        existing.set_text(text)
        if name:
            existing.name = name
        return existing
    if not name:
        try:
            name = auto_name(widget, _frame if _frame is not None else sys._getframe(1), _key_text)
        except Exception:                                # a name is a help, never an error
            name = ""
    tip = Tooltip(widget, text, delay_ms, name)
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
    caller = sys._getframe(1)
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
                add_tooltip(widget, tip, _frame=caller, _key_text=prefix)
                if with_inputs and widget.winfo_class() in ("Label", "TLabel"):
                    # The input boxes created right after the label (e.g. "w" and
                    # "h" entries), up to the next widget that is not an input.
                    siblings = widget.master.winfo_children()
                    for sibling in siblings[siblings.index(widget) + 1:]:
                        if not _is_input(sibling):
                            break
                        add_tooltip(sibling, tip, _frame=caller, _key_text=prefix)
                break


_INPUT_CLASSES = ("Entry", "TEntry", "Spinbox", "TSpinbox", "TCombobox", "Scale", "TScale")


def _is_input(widget) -> bool:
    return widget.winfo_class() in _INPUT_CLASSES
