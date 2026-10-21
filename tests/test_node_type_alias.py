import gc
import tkinter as tk
from pathlib import Path

from node_editor.node_editor_app import NodeEditorApp
from node_editor.node_registry import discover_nodes
from node_editor.nodes.camera_calibration_node import CameraCalibrationNode


def test_renamed_calibration_node_loads_old_projects_and_retired_nodes_are_gone():
    gc.collect()
    root = tk.Tk()
    root.withdraw()
    try:
        app = NodeEditorApp(root)
        registry = discover_nodes(Path(__file__).resolve().parents[1] / "node_editor" / "nodes")
        app.set_node_registry(registry)
        assert app._node_type_map["camera_calibration"] is CameraCalibrationNode
        assert app._node_type_map["camera_calibration_unified"] is CameraCalibrationNode
        assert "camera_calib" not in app._node_type_map                  # retired to recycle_bin/
        assert "chessboard_calibration" not in app._node_type_map
        shown = [name for entries in registry.values() for name, _cls in entries]
        assert shown.count("Camera Calibration") == 1                    # one entry in the toolbox
        # a project saved with the old type name
        issues = app._load_graph([{"node_id": "node_4", "node_type": "camera_calibration_unified",
                                   "node_name": "", "x": 10, "y": 10, "width": 240, "height": 110,
                                   "params": {"rows": [], "model": "rational"}}], [])
        node = app.canvas_nodes["node_4"]
        assert isinstance(node, CameraCalibrationNode) and node.NODE_TYPE == "camera_calibration"
        assert node._model_var.get() == "rational"
        assert not [i for i in issues if "node_4" in str(i)]
        for n in list(app.canvas_nodes.values()):
            n.on_destroy()
    finally:
        root.destroy()
