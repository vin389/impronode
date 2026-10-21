import json
import ast
import os
import re
import zipfile
import zlib
from pathlib import Path
from typing import Any

try:
    from openpyxl import Workbook, load_workbook
    from openpyxl.writer.excel import ExcelWriter
except Exception:  # pragma: no cover - handled at runtime via clear error
    Workbook = None
    load_workbook = None
    ExcelWriter = None

SCHEMA_VERSION = "1.0"

# Everything is saved in the ONE .xlsx file. Excel limits a cell to 32,767
# characters, so longer texts -- a node's params_json that is too long, and
# node arrays (project_data.py: base64 text) -- go to the hidden 'Data'
# sheet, split into cells of CHUNK characters (rows: key, part, text).
DATA_SHEET = "Data"
CHUNK = 30000
PARAMS_IN_DATA = "$params_in_data"
_RESERVED_SHEETS = ("ProjectMeta", "Nodes", "Links", DATA_SHEET)


class ProjectFormatError(Exception):
    pass


class _XlsxZip(zipfile.ZipFile):
    """The .xlsx package: a part whose content hardly compresses (the base64
    text of node arrays) is STORED, not deflated -- deflating it costs most
    of the save time and gains little. Other parts: fast deflate."""

    @staticmethod
    def _dense(sample: bytes) -> bool:
        # base64 text of binary data deflates only to ~75% (its 6 bits per
        # character); ordinary text (JSON, XML) to 10-25%.
        return len(zlib.compress(sample, 1)) > 0.6 * len(sample)

    def writestr(self, zinfo_or_arcname, data, compress_type=None, compresslevel=None):
        if compress_type is None and len(data) > (1 << 20):
            raw = data.encode("utf-8") if isinstance(data, str) else data
            if self._dense(raw[len(raw) // 2:len(raw) // 2 + (1 << 18)]):
                compress_type = zipfile.ZIP_STORED
        super().writestr(zinfo_or_arcname, data, compress_type=compress_type,
                         compresslevel=compresslevel if compress_type != zipfile.ZIP_STORED else None)

    def write(self, filename, arcname=None, compress_type=None, compresslevel=None):
        # write_only worksheets arrive here as temporary files.
        try:
            size = os.path.getsize(filename)
        except OSError:
            size = 0
        if compress_type is None and size > (1 << 20):
            with open(filename, "rb") as f:
                f.seek(size // 2)
                if self._dense(f.read(1 << 18)):
                    compress_type = zipfile.ZIP_STORED
        super().write(filename, arcname, compress_type=compress_type,
                      compresslevel=compresslevel if compress_type != zipfile.ZIP_STORED else None)


def _set_dimension(ws, n_rows: int, last_col: str) -> None:
    """Give a write_only sheet its size (<dimension>, written before the
    first row). Without it, openpyxl's read_only loader parses the WHOLE
    sheet just to find its size -- for the Data sheet, twice the load time."""
    ws.calculate_dimension = lambda: f"A1:{last_col}{max(1, n_rows)}"


def _require_openpyxl() -> None:
    if Workbook is None or load_workbook is None:
        raise RuntimeError(
            "openpyxl is required for XLSX project files. Install with: pip install openpyxl"
        )


def _coerce_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value

    text = str(value).strip()
    if text == "":
        return None

    lower = text.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False

    try:
        if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
            return int(text)
        return float(text)
    except ValueError:
        pass

    if text[:1] in "[{\"":
        try:
            return json.loads(text)
        except Exception:
            return text
    return text


def _compact_text(value: Any, limit: int = 120) -> str:
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _parse_params_object(raw_params: str) -> tuple[dict[str, Any] | None, str | None, str | None]:
    """
    Parse params_json into a dict.

    Returns:
      (dict, None, None) on a clean parse
      (dict, None, warning) on a best-effort salvage that dropped a truncated field
      (None, error_message, None) on failure
    """
    text = str(raw_params).strip()
    if not text:
        return {}, None, None

    # 1) Canonical JSON object format.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed, None, None
        if isinstance(parsed, str):
            # Some legacy files may contain a JSON-encoded JSON string.
            nested = json.loads(parsed)
            if isinstance(nested, dict):
                return nested, None, None
        return None, f"params_json must be a JSON object/dict; got {type(parsed).__name__}", None
    except Exception as e_json:
        json_error = str(e_json)

    # 2) Common Excel/manual-edit case: wrapping single quotes around the whole JSON text.
    if len(text) >= 2 and text[0] == "'" and text[-1] == "'":
        inner = text[1:-1].strip()
        try:
            parsed = json.loads(inner)
            if isinstance(parsed, dict):
                return parsed, None, None
        except Exception:
            pass

    # 2.5) Recover common truncation case where file_list tail is cut and
    #      breaks JSON (for example: Unterminated string ... near file_list).
    if text.startswith("{") and '"file_list"' in text:
        pivot = text.find(',"file_list"')
        if pivot == -1:
            pivot = text.find('"file_list"')
        if pivot > 0:
            trimmed = text[:pivot].rstrip().rstrip(",") + "}"
            try:
                parsed = json.loads(trimmed)
                if isinstance(parsed, dict):
                    return parsed, None, None
            except Exception:
                pass

    # 2.6) Generic recovery for a trailing field truncated at Excel's
    #      32,767-character cell limit (legacy projects that inlined large
    #      array data directly in params_json before it moved to sidecar
    #      files -- see ArrayViewerNode/ArrayAccumulatorNode). Cut the text
    #      at the last complete top-level field boundary before the reported
    #      error position, dropping only the corrupted trailing field so
    #      every other setting on the node is still restored.
    if text.startswith("{"):
        salvaged = _salvage_truncated_params(text, json_error)
        if salvaged is not None:
            return salvaged, None, (
                "params_json was truncated (likely Excel's 32,767-character "
                "cell limit); the oversized trailing field was dropped, "
                "other settings were restored. Re-saving the project will "
                "store large array data in a separate file to prevent this."
            )

    # 3) Legacy Python-literal dict representation.
    try:
        lit = ast.literal_eval(text)
        if isinstance(lit, dict):
            return lit, None, None
    except Exception:
        pass

    # 4) Legacy malformed Windows-path blobs where backslashes were not escaped.
    #    Example: {"pattern":"D:\images\..."} saved without JSON escaping.
    escaped_text = text.replace("\\", "\\\\")
    if escaped_text != text:
        try:
            parsed = json.loads(escaped_text)
            if isinstance(parsed, dict):
                return parsed, None, None
        except Exception:
            pass
        try:
            lit = ast.literal_eval(escaped_text)
            if isinstance(lit, dict):
                return lit, None, None
        except Exception:
            pass

    return None, f"invalid params_json ({json_error})", None


def _salvage_truncated_params(text: str, json_error: str) -> dict[str, Any] | None:
    """Best-effort recovery for params_json cut short mid-value.

    Uses the character position json reported the parse failure at to find
    the last complete top-level "key": "value" boundary before that point,
    then closes the object there. Only safe because a top-level string
    value's closing quote followed by a comma+space (", ) cannot legally
    appear inside a JSON string/number without being escaped/quoted itself,
    so this boundary is unambiguous for the oversized-numeric-array case
    that triggers this (a params_json ending mid-array of floats).
    """
    match = re.search(r"char (\d+)", json_error)
    if not match:
        return None
    error_pos = int(match.group(1))
    prefix = text[:error_pos]
    boundary = prefix.rfind('", ')
    if boundary == -1:
        return None
    candidate = prefix[: boundary + 1] + "}"
    try:
        parsed = json.loads(candidate)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def save_project(
    file_path: str | Path,
    nodes: list[dict[str, Any]],
    links: list[dict[str, str]],
    app_meta: dict[str, Any] | None = None,
    data: dict[str, str] | None = None,
) -> None:
    """Write the project. `data`: long texts (key -> text) for the Data
    sheet, e.g. project_data.end_save(). Written to a temporary file first,
    so a failed save never leaves a broken project."""
    _require_openpyxl()

    data = dict(data or {})
    # write_only: rows are streamed to the file -- much faster and lighter
    # than building the whole workbook in memory.
    wb = Workbook(write_only=True)

    ws_meta = wb.create_sheet("ProjectMeta")
    meta_items = [(k, v) for k, v in (app_meta or {}).items() if k != "extra_tables"]
    _set_dimension(ws_meta, 2 + len(meta_items), "B")
    ws_meta.append(["key", "value"])
    ws_meta.append(["schema_version", SCHEMA_VERSION])
    if app_meta:
        for key, value in app_meta.items():
            if key == "extra_tables":
                continue
            ws_meta.append([str(key), json.dumps(value, ensure_ascii=False)])

    ws_nodes = wb.create_sheet("Nodes")
    _set_dimension(ws_nodes, 1 + len(nodes), "H")
    fixed_cols = [
        "node_id",
        "node_type",
        "node_name",
        "x",
        "y",
        "width",
        "height",
        "params_json",
    ]

    # Keep Nodes schema compact: params_json is the single canonical params field.
    ws_nodes.append(fixed_cols)

    for n in nodes:
        params_text = json.dumps(n.get("params", {}), ensure_ascii=False)
        if len(params_text) > CHUNK:
            key = f"params/{n.get('node_id', '')}"
            data[key] = params_text
            params_text = json.dumps({PARAMS_IN_DATA: key})
        row = {
            "node_id": n.get("node_id", ""),
            "node_type": n.get("node_type", ""),
            "node_name": n.get("node_name", ""),
            "x": n.get("x", 0),
            "y": n.get("y", 0),
            "width": n.get("width", 0),
            "height": n.get("height", 0),
            "params_json": params_text,
        }

        ws_nodes.append([row.get(c, "") for c in fixed_cols])

    ws_links = wb.create_sheet("Links")
    _set_dimension(ws_links, 1 + len(links), "E")
    # geometry_json: optional curve edit points (see link_geometry.py); empty
    # means the default curve. Files written before it existed have only the
    # first four columns and load as default curves.
    ws_links.append(["src_node", "src_pin", "dst_node", "dst_pin", "geometry_json"])
    for lk in links:
        geometry = lk.get("geometry")
        ws_links.append([
            lk.get("src_node", ""),
            lk.get("src_pin", ""),
            lk.get("dst_node", ""),
            lk.get("dst_pin", ""),
            json.dumps(geometry) if geometry else "",
        ])

    # Optional future-facing tables for large array-like data exports/imports.
    extra_tables = (app_meta or {}).get("extra_tables", {}) if app_meta else {}
    if isinstance(extra_tables, dict):
        for sheet_name, rows in extra_tables.items():
            sname = str(sheet_name)
            if not sname or sname in _RESERVED_SHEETS:
                continue
            if sname in wb.sheetnames:
                continue
            ws = wb.create_sheet(sname)
            if not isinstance(rows, list) or not rows:
                continue
            if not isinstance(rows[0], dict):
                continue

            cols = sorted({str(k) for r in rows if isinstance(r, dict) for k in r.keys()})
            ws.append(cols)
            for r in rows:
                if not isinstance(r, dict):
                    continue
                ws.append([r.get(c, "") for c in cols])

    if data:
        ws_data = wb.create_sheet(DATA_SHEET)
        ws_data.sheet_state = "hidden"
        _set_dimension(ws_data, 1 + sum(max(1, -(-len(str(t)) // CHUNK)) for t in data.values()), "C")
        ws_data.append(["key", "part", "text"])
        for key, text in data.items():
            text = str(text)
            for part, start in enumerate(range(0, max(1, len(text)), CHUNK)):
                ws_data.append([key, part, text[start:start + CHUNK]])

    target = Path(file_path)
    tmp = target.with_name(target.name + ".tmp")
    try:
        with _XlsxZip(str(tmp), "w", zipfile.ZIP_DEFLATED, allowZip64=True, compresslevel=1) as archive:
            ExcelWriter(wb, archive).save()
        tmp.replace(target)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def load_project(file_path: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    """(nodes, links, meta). meta['data'] = the Data sheet (key -> text),
    for project_data.begin_load()."""
    _require_openpyxl()

    # read_only: rows are streamed from the file -- much faster for big files.
    wb = load_workbook(str(file_path), data_only=True, read_only=True)
    try:
        return _load_workbook(wb)
    finally:
        wb.close()


def _read_data_sheet(wb) -> dict[str, str]:
    if DATA_SHEET not in wb.sheetnames:
        return {}
    parts: dict[str, list] = {}
    for row in wb[DATA_SHEET].iter_rows(min_row=2, values_only=True):
        if not row or row[0] is None:
            continue
        key = str(row[0])
        try:
            part = int(row[1])
        except (TypeError, ValueError, IndexError):
            continue
        text = row[2] if len(row) > 2 and row[2] is not None else ""
        parts.setdefault(key, []).append((part, str(text)))
    return {k: "".join(t for _p, t in sorted(v)) for k, v in parts.items()}


def _load_workbook(wb) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, Any]]:
    if "Nodes" not in wb.sheetnames:
        raise ProjectFormatError("Missing required worksheet: Nodes")
    if "Links" not in wb.sheetnames:
        raise ProjectFormatError("Missing required worksheet: Links")
    data = _read_data_sheet(wb)

    meta: dict[str, Any] = {}
    issues: list[str] = []
    if "ProjectMeta" in wb.sheetnames:
        ws_meta = wb["ProjectMeta"]
        for row in ws_meta.iter_rows(min_row=2, values_only=True):
            if not row or row[0] is None:
                continue
            key = str(row[0])
            val = row[1]
            try:
                meta[key] = json.loads(val) if isinstance(val, str) else val
            except Exception:
                meta[key] = val

    ws_nodes = wb["Nodes"]
    header = [str(c) if c is not None else "" for c in next(ws_nodes.iter_rows(min_row=1, max_row=1, values_only=True))]
    col_idx = {name: idx for idx, name in enumerate(header)}

    required = ["node_id", "node_type", "x", "y", "width", "height", "params_json"]
    for req in required:
        if req not in col_idx:
            raise ProjectFormatError(f"Nodes sheet missing required column: {req}")

    dynamic_param_cols = [h for h in header if h.startswith("param_")]

    nodes: list[dict[str, Any]] = []
    seen_node_ids: set[str] = set()
    for row_no, row in enumerate(ws_nodes.iter_rows(min_row=2, values_only=True), start=2):
        if not row:
            continue
        row = tuple(row) + (None,) * (len(header) - len(row))   # read-only rows may stop at the last value

        node_id = row[col_idx["node_id"]]
        node_type = row[col_idx["node_type"]]
        if node_id is None or node_type is None:
            issues.append(f"Nodes row {row_no}: missing node_id or node_type; row skipped.")
            continue

        node_id_str = str(node_id)
        if node_id_str in seen_node_ids:
            issues.append(f"Nodes row {row_no}: duplicate node_id '{node_id_str}'.")
        seen_node_ids.add(node_id_str)

        params: dict[str, Any] = {}
        raw_params = row[col_idx["params_json"]]
        if isinstance(raw_params, str) and PARAMS_IN_DATA in raw_params:
            try:
                key = json.loads(raw_params).get(PARAMS_IN_DATA)
            except (ValueError, AttributeError):
                key = None
            if key in data:
                raw_params = data.pop(key)
            else:
                issues.append(f"Nodes row {row_no} (node_id='{node_id_str}'): its params are missing from the "
                              f"'{DATA_SHEET}' sheet; defaults used.")
                raw_params = ""
        if isinstance(raw_params, str) and raw_params.strip():
            parsed_params, parse_error, salvage_warning = _parse_params_object(raw_params)
            if parsed_params is not None:
                params.update(parsed_params)
                if salvage_warning:
                    issues.append(
                        "Nodes row "
                        f"{row_no} (node_id='{node_id_str}', node_type='{node_type}'): "
                        f"{salvage_warning}"
                    )
            else:
                issues.append(
                    "Nodes row "
                    f"{row_no} (node_id='{node_id_str}', node_type='{node_type}'): "
                    "invalid params_json; ignored. "
                    f"error='{_compact_text(parse_error or 'unknown')}' "
                    f"value='{_compact_text(raw_params)}'."
                )
        elif raw_params not in (None, ""):
            issues.append(
                "Nodes row "
                f"{row_no} (node_id='{node_id_str}', node_type='{node_type}'): "
                "params_json is not text; ignored. "
                f"type={type(raw_params).__name__} value='{_compact_text(raw_params)}'."
            )

        for pname in dynamic_param_cols:
            idx = col_idx[pname]
            if idx >= len(row):
                continue
            v = row[idx]
            if v is None:
                continue
            key = pname[len("param_"):]
            coerced = _coerce_scalar(v)
            if coerced is not None:
                params[key] = coerced

        try:
            x = int(float(row[col_idx["x"]] or 0))
            y = int(float(row[col_idx["y"]] or 0))
            width = int(float(row[col_idx["width"]] or 0))
            height = int(float(row[col_idx["height"]] or 0))
        except Exception:
            issues.append(f"Nodes row {row_no}: invalid numeric geometry; row skipped.")
            continue

        if width <= 0 or height <= 0:
            issues.append(f"Nodes row {row_no}: non-positive width/height; defaults may apply.")

        nodes.append({
            "node_id": node_id_str,
            "node_type": str(node_type),
            "node_name": str(row[col_idx["node_name"]]) if "node_name" in col_idx and row[col_idx["node_name"]] is not None else "",
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "params": params,
        })

    ws_links = wb["Links"]
    links: list[dict[str, str]] = []
    for row_no, row in enumerate(ws_links.iter_rows(min_row=2, values_only=True), start=2):
        if not row:
            continue
        row = tuple(row)
        src_node, src_pin, dst_node, dst_pin = (row + (None, None, None, None))[:4]
        if None in (src_node, src_pin, dst_node, dst_pin):
            issues.append(f"Links row {row_no}: missing required column value; row skipped.")
            continue
        src_node_str = str(src_node)
        dst_node_str = str(dst_node)
        if src_node_str not in seen_node_ids or dst_node_str not in seen_node_ids:
            issues.append(
                f"Links row {row_no}: references unknown node(s) '{src_node_str}' or '{dst_node_str}'."
            )
        link = {
            "src_node": src_node_str,
            "src_pin": str(src_pin),
            "dst_node": dst_node_str,
            "dst_pin": str(dst_pin),
        }
        geometry_text = row[4] if len(row) > 4 else None
        if geometry_text not in (None, ""):
            try:
                geometry = json.loads(str(geometry_text))
            except ValueError:
                geometry = None
            if isinstance(geometry, dict):
                link["geometry"] = geometry
            else:
                issues.append(f"Links row {row_no}: unreadable geometry_json; default curve used.")
        links.append(link)

    extra_tables: dict[str, list[dict[str, Any]]] = {}
    for sname in wb.sheetnames:
        if sname in _RESERVED_SHEETS:
            continue
        ws = wb[sname]
        rows_iter = ws.iter_rows(min_row=1, max_row=1, values_only=True)
        first = next(rows_iter, None)
        if not first:
            extra_tables[sname] = []
            continue
        headers = [str(h) if h is not None else "" for h in first]
        table_rows: list[dict[str, Any]] = []
        for row in ws.iter_rows(min_row=2, values_only=True):
            if row is None:
                continue
            row_dict: dict[str, Any] = {}
            for idx, h in enumerate(headers):
                if not h:
                    continue
                v = row[idx] if idx < len(row) else None
                if v is not None:
                    row_dict[h] = v
            if row_dict:
                table_rows.append(row_dict)
        extra_tables[sname] = table_rows

    if issues:
        meta["load_issues"] = issues
    if extra_tables:
        meta["extra_tables"] = extra_tables
    meta["data"] = data

    return nodes, links, meta
