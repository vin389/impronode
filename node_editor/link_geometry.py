# node_editor/link_geometry.py
"""
Curve geometry for the links between node pins (no Tk; pure functions).

A link is a cubic Bezier curve from the output pin S to the input pin E,
drawn by Tk as a line with smooth="raw" (each run of points
p0 c1 c2 p1 c3 c4 p2 ... is a chain of cubic Bezier segments).

Three user-editable points (like PowerPoint's "Edit Points"):

  h1   start tangent handle, stored as an offset (dx, dy) in canvas px from S
  h2   end tangent handle, stored as an offset (dx, dy) from E
  mid  optional point the curve must pass through, stored in the S->E frame
       as (u, v):  M = S + u * (E - S) + v * perp(E - S),
       perp(x, y) = (-y, x). u, v are fractions of the S-E distance, so the
       point follows sensibly when either node is moved.

None for any of them means "default": h1 / h2 horizontal (out of the output
pin to the right, into the input pin from the left) with an automatic
length, and no middle point (a single Bezier segment). Old project files
have no geometry at all and therefore get the all-default curve.
"""

from __future__ import annotations

import math

MIN_HANDLE = 40.0          # px, minimum automatic tangent-handle length
_EPS = 1e-6


def default_geometry() -> dict:
    return {"h1": None, "h2": None, "mid": None}


def is_default(geom: dict | None) -> bool:
    return not geom or all(geom.get(k) is None for k in ("h1", "h2", "mid"))


def _pair(value) -> tuple[float, float] | None:
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


def normalize(geom) -> dict:
    """Validated geometry dict from anything (None, a loaded JSON dict, ...).
    Missing or malformed entries fall back to the default (None)."""
    out = default_geometry()
    if isinstance(geom, dict):
        for k in out:
            if geom.get(k) is not None:
                out[k] = _pair(geom[k])
    return out


def to_json_dict(geom: dict | None) -> dict | None:
    """Compact form for saving; None when everything is default."""
    g = normalize(geom)
    if is_default(g):
        return None
    return {k: [round(v[0], 3), round(v[1], 3)] for k, v in g.items() if v is not None}


def auto_handle_length(s: tuple[float, float], e: tuple[float, float]) -> float:
    return max(MIN_HANDLE, 0.5 * abs(e[0] - s[0]))


def handle_points(s, e, geom: dict | None) -> tuple[tuple[float, float], tuple[float, float]]:
    """Absolute positions of the start / end tangent handles."""
    g = normalize(geom)
    length = auto_handle_length(s, e)
    h1 = g["h1"] if g["h1"] is not None else (length, 0.0)
    h2 = g["h2"] if g["h2"] is not None else (-length, 0.0)
    return (s[0] + h1[0], s[1] + h1[1]), (e[0] + h2[0], e[1] + h2[1])


def _frame(s, e):
    dx, dy = e[0] - s[0], e[1] - s[1]
    if dx * dx + dy * dy < _EPS:
        dx, dy = 1.0, 0.0                    # degenerate: pins on top of each other
    return (dx, dy), (-dy, dx)


def mid_from_frame(s, e, uv) -> tuple[float, float]:
    (ax, ay), (px, py) = _frame(s, e)
    u, v = uv
    return s[0] + u * ax + v * px, s[1] + u * ay + v * py


def mid_to_frame(s, e, point) -> tuple[float, float]:
    (ax, ay), (px, py) = _frame(s, e)
    rx, ry = point[0] - s[0], point[1] - s[1]
    n2 = ax * ax + ay * ay
    return (rx * ax + ry * ay) / n2, (rx * px + ry * py) / n2


def _cubic(p0, p1, p2, p3, t):
    a = (1 - t) ** 3
    b = 3 * (1 - t) ** 2 * t
    c = 3 * (1 - t) * t * t
    d = t ** 3
    return (a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
            a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1])


def mid_point(s, e, geom: dict | None) -> tuple[float, float]:
    """Where the middle edit point is shown: the stored point, or (default)
    the point halfway along the single-segment curve."""
    g = normalize(geom)
    if g["mid"] is not None:
        return mid_from_frame(s, e, g["mid"])
    c1, c2 = handle_points(s, e, g)
    return _cubic(s, c1, c2, e, 0.5)


def control_points(s, e, geom: dict | None) -> list[float]:
    """Flat coordinate list for canvas.create_line(..., smooth="raw").

    Without a middle point: one cubic segment S, C1, C2, E. With a middle
    point M: two segments S, C1, M-t, M and M, M+t, C2, E, where the tangent
    t at M is parallel to C2 - C1 (a smooth point, as in PowerPoint), so the
    curve has no kink at M."""
    g = normalize(geom)
    c1, c2 = handle_points(s, e, g)
    if g["mid"] is None:
        pts = [s, c1, c2, e]
    else:
        m = mid_from_frame(s, e, g["mid"])
        tx, ty = 0.25 * (c2[0] - c1[0]), 0.25 * (c2[1] - c1[1])
        pts = [s, c1, (m[0] - tx, m[1] - ty), m, (m[0] + tx, m[1] + ty), c2, e]
    return [v for p in pts for v in p]
