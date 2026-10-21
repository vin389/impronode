"""Test helper: save node params into a real project .xlsx and load them back,
exactly as the editor does (project_data + project_io)."""

from pathlib import Path

from node_editor import project_data
from node_editor.project_context import set_project_file_path
from node_editor.project_io import load_project, save_project


def save_and_load(project_path, node, node_type="test") -> dict:
    """node.get_params() -> project file -> params as loaded. The project
    folder must then hold ONLY the project file. Call restore(params) on the
    result while the loaded data is active: use `with loaded(...)`."""
    project_path = Path(project_path)
    set_project_file_path(str(project_path))
    project_data.begin_save()
    try:
        params = node.get_params()
    finally:
        data = project_data.end_save()
    save_project(project_path, [{"node_id": node.node_id, "node_type": node_type, "node_name": "",
                                 "x": 0, "y": 0, "width": 100, "height": 50, "params": params}],
                 [], {}, data=data)
    extra = [p.name for p in project_path.parent.iterdir() if p != project_path]
    assert not extra, f"files written next to the project: {extra}"
    nodes, _links, meta = load_project(project_path)
    return nodes[0]["params"], meta.get("data", {})


def restore(node, params, data) -> None:
    """set_params with the project's Data sheet active (as during a load)."""
    project_data.begin_load(data)
    try:
        node.set_params(params)
    finally:
        project_data.end_load()
