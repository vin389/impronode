# node_editor/shared_array_registry.py
"""
Global, in-memory registry of named "Shared Array" global variables.

A SharedArrayNode owns exactly one entry (name -> array + header), but any
node in the project can read or write that entry directly by name, without
needing a wired connection, by importing this module. This mirrors the
"global variable" pattern found in most visual-scripting tools.

Entries persist only for the lifetime of the running app / open project;
the owning SharedArrayNode is responsible for saving/loading the actual
array + header to/from the project file. The write log kept here is purely
informational (shown in the node's inspector) and is never persisted.
"""

import re
import threading
import time

import numpy as np

NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")

_MAX_LOG_LINES = 500

_lock = threading.RLock()

# name -> {"owner_id": str, "array": np.ndarray, "header": list[str]}
_entries: dict[str, dict] = {}
# name -> list[str] (most recent last), never persisted
_logs: dict[str, list] = {}


def is_valid_name(name: str) -> bool:
    return bool(name) and bool(NAME_RE.match(name))


def is_name_taken(name: str, exclude_owner_id: str | None = None) -> bool:
    """True if `name` is already registered to a different owner."""
    with _lock:
        entry = _entries.get(name)
        if entry is None:
            return False
        return entry["owner_id"] != exclude_owner_id


def register(name: str, owner_id: str, array=None, header=None) -> bool:
    """
    Claim `name` for `owner_id`. Returns False if `name` is already owned
    by a different node. If `owner_id` already owns `name`, updates array
    and/or header in place (whichever is not None).
    """
    with _lock:
        if is_name_taken(name, exclude_owner_id=owner_id):
            return False
        entry = _entries.get(name)
        if entry is not None and entry["owner_id"] == owner_id:
            if array is not None:
                entry["array"] = array
            if header is not None:
                entry["header"] = list(header)
            return True
        _entries[name] = {
            "owner_id": owner_id,
            "array": array if array is not None else np.zeros((0, 0), dtype=np.float64),
            "header": list(header) if header else [],
        }
        _logs.setdefault(name, [])
        return True


def unregister(name: str, owner_id: str) -> None:
    with _lock:
        entry = _entries.get(name)
        if entry is not None and entry["owner_id"] == owner_id:
            _entries.pop(name, None)
            _logs.pop(name, None)


def rename(old_name: str, new_name: str, owner_id: str) -> bool:
    """Move an existing entry owned by `owner_id` from old_name to new_name."""
    with _lock:
        if old_name == new_name:
            return True
        if is_name_taken(new_name, exclude_owner_id=owner_id):
            return False
        entry = _entries.get(old_name)
        if entry is None or entry["owner_id"] != owner_id:
            return False
        del _entries[old_name]
        _entries[new_name] = entry
        _logs[new_name] = _logs.pop(old_name, [])
        return True


def unique_name(preferred: str, exclude_owner_id: str | None = None) -> str:
    """Return `preferred` if free, otherwise `preferred_2`, `preferred_3`, ..."""
    base = preferred if is_valid_name(preferred) else "shared_array"
    if not is_name_taken(base, exclude_owner_id):
        return base
    i = 2
    while is_name_taken(f"{base}_{i}", exclude_owner_id):
        i += 1
    return f"{base}_{i}"


def get_array(name: str) -> np.ndarray | None:
    with _lock:
        entry = _entries.get(name)
        return None if entry is None else entry["array"]


def get_header(name: str) -> list[str] | None:
    with _lock:
        entry = _entries.get(name)
        return None if entry is None else list(entry["header"])


def set_array(name: str, array, header=None, writer_name: str = "external") -> bool:
    """Replace the whole array for `name`. Returns False if `name` is unknown."""
    with _lock:
        entry = _entries.get(name)
        if entry is None:
            return False
        arr = np.asarray(array, dtype=np.float64)
        entry["array"] = arr
        if header is not None:
            entry["header"] = list(header)
        _append_log(name, writer_name, f"replaced full array, shape={arr.shape}")
        return True


def set_row(name: str, row_index: int, row_data, writer_name: str = "external") -> bool:
    """
    Write one row (0-based row_index) into the named array, growing it
    (with NaN-filled gap rows) if row_index is beyond the current height.
    Returns False if `name` is unknown, row_index < 0, or the column count
    does not match the existing array.
    """
    with _lock:
        entry = _entries.get(name)
        if entry is None or row_index < 0:
            return False
        row = np.asarray(row_data, dtype=np.float64).reshape(-1)
        n_cols = row.shape[0]
        arr = entry["array"]
        if arr.size == 0:
            arr = np.full((0, n_cols), np.nan, dtype=np.float64)
        elif arr.shape[1] != n_cols:
            return False
        if row_index >= arr.shape[0]:
            pad = np.full((row_index + 1 - arr.shape[0], n_cols), np.nan, dtype=np.float64)
            arr = np.vstack([arr, pad])
        arr[row_index] = row
        entry["array"] = arr
        _append_log(name, writer_name, f"row {row_index + 1}")
        return True


def _append_log(name: str, writer_name: str, message: str) -> None:
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {writer_name}: {message}"
    log = _logs.setdefault(name, [])
    log.append(line)
    if len(log) > _MAX_LOG_LINES:
        del log[: len(log) - _MAX_LOG_LINES]


def set_cell(name: str, row_index: int, col_index: int, value: float,
             writer_name: str = "external") -> bool:
    """Edit a single cell in place (used by the inspector's spreadsheet editor)."""
    with _lock:
        entry = _entries.get(name)
        if entry is None:
            return False
        arr = entry["array"]
        if arr.ndim != 2 or not (0 <= row_index < arr.shape[0]) or not (0 <= col_index < arr.shape[1]):
            return False
        arr[row_index, col_index] = value
        _append_log(name, writer_name, f"cell (row {row_index + 1}, col {col_index + 1})")
        return True


def get_log(name: str) -> list[str]:
    with _lock:
        return list(_logs.get(name, []))


def clear_log(name: str) -> None:
    with _lock:
        _logs[name] = []


def all_names() -> list[str]:
    with _lock:
        return list(_entries.keys())
