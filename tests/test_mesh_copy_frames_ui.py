import gc
import os
import time
import tkinter as tk

import cv2
import numpy as np

from node_editor.node_editor_app import NodeEditorApp
from node_editor.nodes.image_sequence_node import ImageSequenceNode
from node_editor.nodes.mesh_template_track_node import MeshTemplateTrackNode
from node_editor.project_context import set_project_file_path


def _wait(root, cond, seconds=60):
    t = time.time()
    while not cond():
        root.update()
        time.sleep(0.02)
        assert time.time() - t < seconds


def test_copy_panel_shading_copy_and_new_image_sequence_node(tmp_path, monkeypatch):
    src = tmp_path / "src"
    src.mkdir()
    paths = []
    for k in range(30):
        p = src / f"f_{k + 1:03d}.png"
        cv2.imwrite(str(p), np.full((40, 60, 3), 5 * k, np.uint8))
        paths.append(str(p))
    out = tmp_path / "out"
    out.mkdir()
    gc.collect()
    root = tk.Tk()
    root.withdraw()
    try:
        app = NodeEditorApp(root)
        app._node_type_map["image_sequence"] = ImageSequenceNode
        app._node_type_map["mesh_template_track"] = MeshTemplateTrackNode
        mesh = app._spawn_node(MeshTemplateTrackNode, x=100, y=100)
        mesh._set_local({"kind": "images", "paths": paths})
        mesh.open_inspector()
        root.update()

        # Not ready: no frames chosen / no folder -> Copy disabled, info explains.
        assert str(mesh._copy_btn.cget("state")) == "disabled"
        mesh._copy_frames_var.set("2:5 10:12 30:40")
        _wait(root, lambda: mesh._copy_check_id is None)
        assert "select the folder" in mesh._copy_info_var.get()
        # The chosen frames are shaded on the plot (one artist, behind the lines).
        shades = [c for c in mesh._ax.collections if "to copy" in str(c.get_label())]
        assert len(shades) == 1 and "8 frames" in shades[0].get_label()

        mesh._copy_dir_var.set(str(out))
        mesh._copy_mode_var.set("pattern")
        mesh._copy_pattern_var.set("keep_%04d")
        mesh._copy_number_var.set(mesh._COPY_NUMBERING[1])
        _wait(root, lambda: mesh._copy_check_id is None)
        assert str(mesh._copy_btn.cget("state")) == "normal", mesh._copy_info_var.get()
        assert "keep_0001.png" in mesh._copy_info_var.get()

        # Invalid text: red message, button disabled, no shading.
        mesh._copy_frames_var.set("5:2")
        _wait(root, lambda: mesh._copy_check_id is None)
        assert mesh._copy_info_var.get().startswith("Invalid frames")
        assert str(mesh._copy_btn.cget("state")) == "disabled"
        mesh._copy_frames_var.set("2:5 10:12 30:40")
        _wait(root, lambda: mesh._copy_check_id is None)

        mesh._on_copy_start()
        _wait(root, lambda: mesh._copy_job is None)
        written = sorted(os.listdir(out))
        assert written == [f"keep_{k:04d}.png" for k in range(1, 9)]
        assert "Done: 8 files" in mesh._copy_status_var.get()
        assert str(mesh._copy_node_btn.cget("state")) == "normal"

        # A second copy into the same folder asks first; 'No' keeps the files.
        monkeypatch.setattr("tkinter.messagebox.askyesnocancel", lambda *a, **k: False)
        (out / "keep_0001.png").write_bytes(b"x")
        mesh._on_copy_start()
        _wait(root, lambda: mesh._copy_job is None)
        assert (out / "keep_0001.png").read_bytes() == b"x"

        before = set(app.canvas_nodes)
        mesh._on_copy_create_node()
        root.update()
        new_ids = set(app.canvas_nodes) - before
        assert len(new_ids) == 1
        seq = app.canvas_nodes[new_ids.pop()]
        assert isinstance(seq, ImageSequenceNode)
        assert seq.node_name == "8 frames of Mesh Template Track"
        assert len(seq._file_paths) == 8
        # placed next to the mesh node, overlapping nothing
        assert seq.x >= mesh.x + mesh.width
        for other in app.canvas_nodes.values():
            if other is not seq:
                assert (seq.x >= other.x + other.width or seq.x + seq.width <= other.x
                        or seq.y >= other.y + other.height or seq.y + seq.height <= other.y)
        # a second node goes to another free spot
        mesh._on_copy_create_node()
        root.update()
        nodes = [n for n in app.canvas_nodes.values() if isinstance(n, ImageSequenceNode)]
        assert len(nodes) == 2 and (nodes[0].x, nodes[0].y) != (nodes[1].x, nodes[1].y)

        # the copy settings are saved with the project
        set_project_file_path(str(tmp_path / "p.xlsx"))
        params = mesh.get_params()
        assert params["copy_frames"]["frames"] == "2:5 10:12 30:40"
        assert params["copy_frames"]["folder"] == "out" and params["copy_frames"]["numbering"] == "sequence"
        mesh.close_inspector()
    finally:
        set_project_file_path(None)
        for n in list(app.canvas_nodes.values()):
            n.on_destroy()
        root.destroy()
