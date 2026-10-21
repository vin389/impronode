"""
Remembered folders for file dialogs.

Every owner (normally a node) keeps the last folder it used per dialog kind
("load" / "save") in ``owner._recent_dialog_dirs`` as absolute paths.  The
next dialog of the same kind opens in that folder.  When a project is saved
the folders are written into the node's params under ``PARAM_KEY``, relative
to the project folder when possible (absolute when on another drive), and
resolved against the project folder again when the project is loaded.

Use the ``ask_*`` wrappers instead of ``tkinter.filedialog.ask*`` directly:

    path = ask_open_filename(self, title="Load ...", filetypes=[...])
"""

import os
from tkinter import filedialog

from node_editor.project_context import get_project_directory

PARAM_KEY = "_recent_dirs"
_ATTR = "_recent_dialog_dirs"


def _dirs(owner) -> dict[str, str]:
    d = getattr(owner, _ATTR, None)
    if not isinstance(d, dict):
        d = {}
        setattr(owner, _ATTR, d)
    return d


def get_recent_dir(owner, kind: str) -> str | None:
    """The remembered folder for ``kind`` if it still exists, else None."""
    if owner is None:
        return None
    path = _dirs(owner).get(kind)
    return path if path and os.path.isdir(path) else None


def remember_dir(owner, kind: str, folder: str | None) -> None:
    if owner is None or not folder:
        return
    _dirs(owner)[kind] = os.path.abspath(folder)


def _with_initialdir(owner, kind: str, kwargs: dict) -> dict:
    """Start in the remembered folder; otherwise keep the caller's initialdir,
    otherwise the project folder."""
    recent = get_recent_dir(owner, kind)
    if recent:
        kwargs["initialdir"] = recent
    elif not kwargs.get("initialdir"):
        base = get_project_directory()
        if base is not None:
            kwargs["initialdir"] = str(base)
    return kwargs


# ── dialog wrappers ──────────────────────────────────────────────────

def ask_open_filename(owner, kind: str = "load", **kwargs) -> str:
    path = filedialog.askopenfilename(**_with_initialdir(owner, kind, kwargs))
    if path:
        remember_dir(owner, kind, os.path.dirname(path))
    return path


def ask_open_filenames(owner, kind: str = "load", **kwargs):
    paths = filedialog.askopenfilenames(**_with_initialdir(owner, kind, kwargs))
    if paths:
        first = paths if isinstance(paths, str) else paths[0]
        remember_dir(owner, kind, os.path.dirname(first))
    return paths


def ask_save_as_filename(owner, kind: str = "save", **kwargs) -> str:
    path = filedialog.asksaveasfilename(**_with_initialdir(owner, kind, kwargs))
    if path:
        remember_dir(owner, kind, os.path.dirname(path))
    return path


def ask_directory(owner, kind: str = "load", **kwargs) -> str:
    folder = filedialog.askdirectory(**_with_initialdir(owner, kind, kwargs))
    if folder:
        remember_dir(owner, kind, folder)
    return folder


# ── project persistence ──────────────────────────────────────────────

def dirs_to_params(owner) -> dict[str, str]:
    """Remembered folders for saving, relative to the project folder when
    possible."""
    base = get_project_directory()
    out: dict[str, str] = {}
    for kind, path in _dirs(owner).items():
        if not path:
            continue
        text = path
        if base is not None:
            try:
                text = os.path.relpath(path, str(base))
            except ValueError:          # another drive: keep absolute
                pass
        out[kind] = text
    return out


def dirs_from_params(owner, value) -> None:
    """Restore folders written by ``dirs_to_params``."""
    if not isinstance(value, dict):
        return
    base = get_project_directory()
    d = _dirs(owner)
    for kind, text in value.items():
        if not isinstance(text, str) or not text:
            continue
        if not os.path.isabs(text) and base is not None:
            text = os.path.join(str(base), text)
        d[str(kind)] = os.path.normpath(os.path.abspath(text))
