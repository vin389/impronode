import os
from pathlib import Path

import numpy as np

from node_editor.node_editor_app import NodeEditorApp
from node_editor.nodes.image_save_node import ImageSaveNode
from node_editor.nodes.image_sequence_node import ImageSequenceNode
from node_editor.project_io import load_project, save_project


class _PinNode:
    def __init__(self, node_id, schema):
        self.node_id = node_id
        self.get_pin_schema = lambda: schema


class _CanvasStub:
    def __init__(self):
        self.scrollregion = None

    def configure(self, **kwargs):
        self.scrollregion = kwargs.get("scrollregion")


def test_project_meta_preserves_canvas_dimensions(tmp_path):
    project_path = tmp_path / "canvas-size.xlsx"

    save_project(
        project_path,
        nodes=[],
        links=[],
        app_meta={"canvas_width": 4200, "canvas_height": 2800},
    )

    nodes, links, meta = load_project(project_path)

    assert nodes == []
    assert links == []
    assert meta["canvas_width"] == 4200
    assert meta["canvas_height"] == 2800


def test_app_serializes_and_restores_canvas_dimensions():
    app = object.__new__(NodeEditorApp)
    app.canvas_nodes = {}
    app.link_items = {}
    app.canvas = _CanvasStub()
    app._canvas_width = 4200
    app._canvas_height = 2800

    _nodes, _links, meta = app._serialize_graph()
    issue = app._restore_canvas_size(meta)

    assert issue is None
    assert (app._canvas_width, app._canvas_height) == (4200, 2800)
    assert app.canvas.scrollregion == (0, 0, 4200, 2800)


def test_pin_ids_are_stable_and_unique_across_all_nodes():
    from node_editor.pin_types import PinDef, PinSchema, PinType

    app = object.__new__(NodeEditorApp)
    app.canvas_nodes = {
        "node_0": _PinNode("node_0", PinSchema(inputs=[PinDef("a", PinType.SCALAR, "in")], outputs=[PinDef("out", PinType.SCALAR, "out")])),
        "node_1": _PinNode("node_1", PinSchema(inputs=[PinDef("b", PinType.SCALAR, "in")], outputs=[])),
    }
    app._pin_id_by_key = {}
    app._pin_key_by_id = {}
    app._pin_id_counter = 1

    app._refresh_pin_ids()

    assert app._pin_id_by_key[("node_0", "in", "a")] == 1
    assert app._pin_id_by_key[("node_0", "out", "out")] == 2
    assert app._pin_id_by_key[("node_1", "in", "b")] == 3
    assert app._pin_key_by_id[2] == ("node_0", "out", "out")


def test_select_node_raises_it_to_top_on_canvas():
    app = object.__new__(NodeEditorApp)
    app.canvas = type("Canvas", (), {"tag_raise": lambda self, tag: setattr(self, "raised", tag), "delete": lambda *args, **kwargs: None, "find_withtag": lambda *args, **kwargs: []})()
    app.canvas_nodes = {"node_7": type("Node", (), {"x": 0, "y": 0, "width": 10, "height": 10})()}
    app._selected_node_id = None
    app._selection_items = []

    app._draw_selection_overlay = lambda node_id: None
    app._select_node("node_7")

    assert app.canvas.raised == "node_7"


def test_canvas_link_visibility_toggle_hides_and_shows_link_items():
    app = object.__new__(NodeEditorApp)
    app.link_items = {("src", "out", "dst", "in"): 101, ("src2", "out", "dst2", "in"): 102}
    app._hover_link_key = None
    app._hover_link_candidate = None
    app._show_links = True

    class CanvasStub:
        def __init__(self):
            self.states = {}
            self.configs = {}

        def itemconfigure(self, item, **kwargs):
            self.states[item] = kwargs.get("state", "normal")
            self.configs.setdefault(item, {})
            self.configs[item].update(kwargs)

        def itemcget(self, item, option):
            if option == "state":
                return self.states.get(item, "normal")
            return ""

    app.canvas = CanvasStub()

    app._set_links_visibility(False)
    assert app._show_links is False
    assert app.canvas.states[101] == "hidden"
    assert app.canvas.states[102] == "hidden"

    app._set_links_visibility(True)
    assert app._show_links is True
    assert app.canvas.states[101] == "normal"
    assert app.canvas.states[102] == "normal"


def test_selected_links_follow_canvas_hide_toggle():
    app = object.__new__(NodeEditorApp)
    app._show_links = False
    app._selected_node_id = "src"
    app._selected_node_ids = {"src"}
    app.link_items = {("src", "out", "dst", "in"): 201}
    app.canvas_nodes = {"src": type("Node", (), {"_body_rect": 301})(), "dst": type("Node", (), {"_body_rect": 302})()}

    class CanvasStub:
        def __init__(self):
            self.states = {}
            self.configs = {}

        def itemconfig(self, item, **kwargs):
            self.states[item] = kwargs.get("state", "normal")
            self.configs.setdefault(item, {})
            self.configs[item].update(kwargs)

    app.canvas = CanvasStub()

    app._refresh_selection_visuals()

    assert app.canvas.states[201] == "normal"

    app._show_links = False
    app._set_links_visibility(False)
    assert app.canvas.states[201] == "normal"


def test_image_path_serialization_uses_project_relative_paths():
    project_dir = Path(r"c:\aaa\bbb")
    abs_path = Path(r"c:\aaa\ddd\eee\fff.jpg")

    expected = os.path.relpath(str(abs_path), str(project_dir))

    assert ImageSequenceNode._to_relative(str(abs_path), project_dir) == expected
    assert ImageSaveNode._to_relative(str(abs_path), project_dir) == expected


def test_array_input_manual_shape_minus_one_infers_1d_length():
    from node_editor.nodes.array_nodes import ArrayInputNode

    node = ArrayInputNode("arr", None)
    node._init_state()
    node._shape_mode_var.set("manual")
    node._shape_entry_var.set("-1")
    node._csv_text = "1, 2, 3"

    arr = node._parse_csv()

    assert arr is not None
    assert arr.shape == (3,)
    assert np.array_equal(arr, np.array([1.0, 2.0, 3.0]))


def test_tracking_node_keeps_cached_prev_pts_when_not_reissued():
    from node_editor.nodes.template_match_node import TemplateMatchNode

    node = TemplateMatchNode("tm", None)
    node._init_state()
    node._buffered_inputs["prevPts"] = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
    node._buffered_inputs["prevImg"] = np.zeros((10, 10), dtype=np.uint8)
    node._buffered_inputs["nextImg"] = np.zeros((10, 10), dtype=np.uint8)

    node._sync_buffer({"prevImg": np.zeros((10, 10), dtype=np.uint8), "nextImg": np.zeros((10, 10), dtype=np.uint8), "trig": True})

    assert np.array_equal(node._buffered_inputs["prevPts"], np.array([[10.0, 20.0], [30.0, 40.0]]))


def test_image_save_index_pin_overrides_inspector_numbering(tmp_path):
    node = ImageSaveNode("save_1", None)
    node._init_state()
    node._pattern_var.set(str(tmp_path / "out%04d.png"))
    node._next_index_var.set(99)
    node._step_var.set(2)

    result = node.compute({
        "image": __import__("numpy").zeros((4, 4, 3), dtype=__import__("numpy").uint8),
        "index": 7,
    })

    assert result["index_out"] == 7.0
    assert result["saved_path"].endswith("out0007.png")
    assert node._next_index_var.get() == 99


def test_resizing_node_keeps_title_and_selected_node_above_others():
    app = object.__new__(NodeEditorApp)
    app.canvas_nodes = {
        "node_7": type(
            "Node",
            (),
            {
                "x": 0,
                "y": 0,
                "width": 100,
                "height": 80,
                "MIN_WIDTH": 80,
                "MIN_HEIGHT": 60,
                "_body_rect": 101,
                "_title_item": 102,
                "set_size": lambda self, w, h: None,
                "on_resize": lambda self, *args: None,
                "get_pin_schema": lambda self: type("Schema", (), {"inputs": [], "outputs": []})(),
            },
        )(),
    }
    app.canvas = type(
        "Canvas",
        (),
        {
            "find_withtag": lambda self, tag: [101, 102] if tag == "node_7" else [],
            "coords": lambda self, item, *args: [0, 0, 0, 0] if item == 101 else [0, 0],
            "itemconfigure": lambda *args, **kwargs: None,
            "type": lambda self, item: "rectangle",
            "tag_raise": lambda self, tag: setattr(self, "raised", getattr(self, "raised", []) + [tag]),
            "delete": lambda *args, **kwargs: None,
        },
    )()
    app._selected_node_id = "node_7"
    app._selection_items = []
    app._layout_pins_for_node = lambda node: None
    app._update_links_for_node = lambda node_id: None
    app._draw_selection_overlay = lambda node_id: None
    app._mark_dirty = lambda: None
    app._raise_node_title = lambda node: app.canvas.tag_raise(node._title_item)

    node = app.canvas_nodes["node_7"]
    app._apply_node_geometry("node_7", 0, 0, 90, 70)

    assert app.canvas.raised == ["node_7", 102]


def test_params_json_truncated_at_excel_cell_limit_salvages_other_fields():
    import json as _json
    from node_editor.project_io import _parse_params_object

    huge_data = [[20.000012455750685 + i for i in range(50)] for _ in range(2000)]
    full = _json.dumps({
        "rows_text": ":",
        "cols_text": ":",
        "reshape_text": "-1,20,16,3",
        "data": huge_data,
    })
    truncated = full[:32767]

    parsed, error, warning = _parse_params_object(truncated)

    assert error is None
    assert parsed is not None
    assert warning is not None and "truncated" in warning
    assert parsed["rows_text"] == ":"
    assert parsed["cols_text"] == ":"
    assert parsed["reshape_text"] == "-1,20,16,3"
    assert "data" not in parsed


def test_params_json_clean_input_has_no_salvage_warning():
    import json as _json
    from node_editor.project_io import _parse_params_object

    clean = _json.dumps({"rows_text": ":", "cols_text": ":", "reshape_text": "", "data": [[1.0, 2.0]]})
    parsed, error, warning = _parse_params_object(clean)

    assert error is None
    assert warning is None
    assert parsed == _json.loads(clean)


def test_load_project_reports_salvage_as_a_warning_not_a_hard_error(tmp_path):
    import json as _json
    from node_editor.project_io import save_project, load_project

    huge_data = [[1.5 + i for i in range(60)] for _ in range(2000)]
    full = _json.dumps({
        "rows_text": ":",
        "cols_text": ":",
        "reshape_text": "",
        "data": huge_data,
    })
    truncated = full[:32767]

    project_path = tmp_path / "legacy.xlsx"
    save_project(project_path, nodes=[], links=[], app_meta={})

    # Inject a pre-truncated params_json directly, simulating a project
    # saved before large array data moved to sidecar files.
    from openpyxl import load_workbook
    wb = load_workbook(project_path)
    ws = wb["Nodes"]
    ws.append(["node_0", "array_viewer", "", 10, 10, 210, 105, truncated])
    wb.save(project_path)

    nodes, links, meta = load_project(project_path)

    assert len(nodes) == 1
    assert nodes[0]["params"]["rows_text"] == ":"
    assert "data" not in nodes[0]["params"]
    assert any("truncated" in issue for issue in meta.get("load_issues", []))

