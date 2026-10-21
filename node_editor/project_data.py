"""
Binary data of nodes (numpy arrays) stored INSIDE the project .xlsx.

Excel cells hold text (at most 32,767 characters each), so an array is
stored as text: np.savez bytes -> zlib (only when a sample shows it pays:
measured noise hardly compresses, and compressing it costs most of the save
time) -> base64, split into cells of a hidden 'Data' sheet (project_io.py
writes / reads the sheet).
Exact (any dtype / shape), compact and fast -- no side files next to the
project.

In a node:

    def get_params(self):
        return {..., "table": project_data.put_array(self, "table", arr)}

    def set_params(self, params):
        arr = project_data.get_array(params.get("table"))   # None if absent / unreadable

put_arrays / get_arrays do the same for a dict of arrays (one entry).

The value stored in params is a small reference {"$data": key, ...}; the
text itself goes to the Data sheet. Encoded text is cached per key, so an
array that did not change since the last save is not encoded again.
"""

from __future__ import annotations

import base64
import hashlib
import io
import zlib

import numpy as np

_REF = "$data"
_FORMAT = "npz-b64-2"            # text: "z" + base64(zlib(npz)) or "r" + base64(npz)
_SAMPLE = 1 << 20                # bytes test-compressed to decide
_WORTH = 0.8                     # compress when the sample shrinks below this fraction

# Filled by put_*() while a project is being saved; written by project_io.
_saving: dict[str, str] = {}
# The Data sheet of the project being loaded (key -> text), read by get_*().
_loaded: dict[str, str] = {}
# key -> (digest of the arrays, encoded text): skip encoding unchanged arrays.
_cache: dict[str, tuple[bytes, str]] = {}


def begin_save() -> None:
    _saving.clear()


def end_save() -> dict[str, str]:
    """The entries put since begin_save() (key -> text), for project_io."""
    out = dict(_saving)
    _saving.clear()
    return out


def begin_load(data: dict[str, str] | None) -> None:
    _loaded.clear()
    _loaded.update(data or {})
    _cache.clear()


def end_load() -> None:
    _loaded.clear()


def _key(owner, name: str) -> str:
    return f"{getattr(owner, 'node_id', owner)}/{name}"


def _digest(arrays: dict) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    for k in sorted(arrays):
        a = np.ascontiguousarray(arrays[k])
        h.update(k.encode())
        h.update(str(a.dtype).encode())
        h.update(repr(a.shape).encode())
        h.update(memoryview(a).cast("B") if a.size else b"")
    return h.digest()


def encode_arrays(arrays: dict) -> str:
    buf = io.BytesIO()
    np.savez(buf, **{k: np.asarray(v) for k, v in arrays.items()})
    raw = buf.getvalue()
    sample = raw if len(raw) <= 2 * _SAMPLE else raw[len(raw) // 2:len(raw) // 2 + _SAMPLE]
    if len(zlib.compress(sample, 1)) < _WORTH * len(sample):
        return "z" + base64.b64encode(zlib.compress(raw, 1)).decode("ascii")
    return "r" + base64.b64encode(raw).decode("ascii")


def decode_arrays(text: str) -> dict:
    if text[:1] == "r":
        raw = base64.b64decode(text[1:])
    elif text[:1] == "z":
        raw = zlib.decompress(base64.b64decode(text[1:]))
    else:                                            # no prefix: always zlib
        raw = zlib.decompress(base64.b64decode(text))
    with np.load(io.BytesIO(raw), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def put_arrays(owner, name: str, arrays: dict) -> dict:
    """Store a dict of arrays with the project; returns the reference to
    keep in the node's params."""
    key = _key(owner, name)
    arrays = {str(k): np.asarray(v) for k, v in arrays.items()}
    if any(a.dtype == object for a in arrays.values()):
        raise TypeError(f"{key}: object arrays cannot be stored in the project")
    digest = _digest(arrays)
    cached = _cache.get(key)
    if cached is not None and cached[0] == digest:
        text = cached[1]
    else:
        text = encode_arrays(arrays)
        _cache[key] = (digest, text)
    _saving[key] = text
    return {_REF: key, "format": _FORMAT, "names": sorted(arrays)}


def put_array(owner, name: str, arr) -> dict:
    return put_arrays(owner, name, {"array": arr})


def is_ref(value) -> bool:
    return isinstance(value, dict) and _REF in value


def get_arrays(ref) -> dict | None:
    """The arrays of a reference made by put_arrays(), or None (no reference
    / not in the project / unreadable)."""
    if not is_ref(ref):
        return None
    key = str(ref[_REF])
    # The loaded project first; then what this session encoded (a params dict
    # passed straight from get_params to set_params, e.g. a copied node). The
    # cache is emptied by every project load, so another project's data is
    # never used.
    text = _loaded.get(key) or _saving.get(key) or (_cache.get(key) or (None, None))[1]
    if not text:
        return None
    try:
        out = decode_arrays(text)
    except Exception:
        return None
    _cache[key] = (_digest(out), text)
    return out


def get_array(ref):
    arrays = get_arrays(ref)
    return None if arrays is None else arrays.get("array")
