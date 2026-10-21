import gc
import time
import tkinter as tk

import cv2
import numpy as np

from node_editor import tooltip
from node_editor.chessboard_search import grid_rc
from node_editor.nodes.camera_calibration_node import CameraCalibrationNode, new_row

from test_camera_calibration_search import K_TRUE, POSES, SIZE, render_view


def _node_with_boards(root, tmp_path):
    canvas = tk.Canvas(root, width=300, height=200)
    node = CameraCalibrationNode("node_0", canvas)
    node.build_body()
    node.open_inspector()
    root.update()
    cands = []
    grid = np.array([(c * 1.0, r * 1.0, 0) for r in range(8) for c in range(11)])
    for k, (rv, tv) in enumerate(POSES[:6]):
        p = tmp_path / f"b{k}.png"
        cv2.imwrite(str(p), render_view(rv, tv))
        R, _ = cv2.Rodrigues(np.asarray(rv, float))
        # corners of the 11 x 8 inner corners (board corner (1, 1) .. (11, 8) in squares)
        img, _ = cv2.projectPoints(grid + [1, 1, 0], np.asarray(rv, float), np.asarray(tv, float), K_TRUE, None)
        cands.append({"path": str(p), "source_index": k, "corners": img.reshape(-1, 2).astype(np.float32),
                      "rc": grid_rc(8, 11), "support": None, "n_settings": None, "sharpness": 100.0,
                      "coverage": 0.3, "center_dist": 0.2, "size": SIZE})
    node._det_vars["square"].set("1")
    node._det_vars["ncol"].set("11")
    node._det_vars["nrow"].set("8")
    node._cb = {"cfg": {"ncol": 11, "nrow": 8, "square": 1.0, "max_width": 0}, "cands": cands}
    node._cb_use = {c["path"] for c in cands}
    node._size_vars[0].set(str(SIZE[0]))
    node._size_vars[1].set(str(SIZE[1]))
    node._use_points_var.set(False)
    node._refresh_all()
    return node


def _settle(root, seconds=0.4):
    t = time.time()
    while time.time() - t < seconds:
        root.update()
        time.sleep(0.02)


def test_solver_line_cancel_runs_columns_and_no_image_jump(tmp_path, monkeypatch):
    monkeypatch.setattr(tooltip, "_SETTINGS", tmp_path / "settings.json")
    gc.collect()
    root = tk.Tk()
    root.withdraw()
    try:
        node = _node_with_boards(root, tmp_path)
        _settle(root)
        assert node._route_var.get().startswith("Solver: OpenCV only"), node._route_var.get()
        node._loss_var.set("huber")
        _settle(root)
        assert "huber" in node._route_var.get() and "Least-squares" in node._route_var.get()
        node._loss_var.set("linear")

        # Solve with OpenCV only; the run is listed with its solver.
        node._on_solve()
        t0 = time.time()
        while node._solve_state is not None and not node._solve_state["finished"]:
            root.update()
            time.sleep(0.02)
            assert time.time() - t0 < 120
        _settle(root, 0.2)
        assert node._res is not None and abs(node._res["K"][0][0] - 800) < 5
        assert "method: OpenCV only" in node._result_text.get("1.0", "end")
        assert node._runs[-1]["solver"] == "OpenCV"
        node._refresh_runs_tree()                                   # done when the Runs tab is shown
        cols = list(node._runs_tree["columns"])
        assert "solver" in cols and "k3" not in cols               # k3 fixed at 0: no column

        # k3 free -> nonzero k3 -> its column appears.
        node._fix_vars["k3"].set(False)
        node._solve_sync()
        assert node._runs[-1]["D"][4] != 0
        node._refresh_runs_tree()
        assert "k3" in list(node._runs_tree["columns"])

        # Cancel a running least-squares solve: the previous result stays.
        node._force_lsq_var.set(True)
        node._model_var.set("thin_prism")
        for n in ("k3", "k4", "k5", "k6", "s1", "s2", "s3", "s4"):
            node._fix_vars[n].set(False)
        before = node._res
        node._on_solve()
        assert str(node._cancel_solve_btn.cget("state")) == "normal"
        node._on_cancel_solve()
        assert node._solve_state is None
        assert str(node._solve_btn.cget("state")) == "normal"
        assert str(node._cancel_solve_btn.cget("state")) == "disabled"
        _settle(root, 1.0)
        assert node._res is before

        # Clicking the points image never switches the view to a board image.
        node._select_tree_path(node._cb["cands"][2]["path"])
        node._on_tree_select()
        assert node._show == node._cb["cands"][2]["path"]
        node._show_board(None)                                    # 'Points image'
        node.rows.append(new_row(node._new_id(), "P1", X="0", Y="0", Z="0"))
        node._sheet.select([0])
        node._on_image_click(100.0, 120.0, None)
        _settle(root, 0.3)
        assert node._show is None
        assert node.rows[0]["u"] == 100.0 or node.rows[-1]["u"] == 100.0

        # Tooltips can be switched off for every widget (and the choice is kept).
        tooltip.set_tooltips_enabled(False)
        tip = tooltip.add_tooltip(node._solve_btn, "x", delay_ms=1)
        tip._schedule()
        assert tip._after is None
        assert tooltip._load_enabled() is False
        tooltip.set_tooltips_enabled(True)
        assert tooltip._load_enabled() is True
    finally:
        tooltip.set_tooltips_enabled(True)
        root.destroy()
