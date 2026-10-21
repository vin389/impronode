import gc
import time
import tkinter as tk

import cv2
import numpy as np

from node_editor.nodes.camera_calibration_node import CameraCalibrationNode

K_TRUE = np.array([[800.0, 0, 480], [0, 800.0, 360], [0, 0, 1]])
SIZE = (960, 720)


def render_view(rvec, tvec, squares=(12, 9), sq_px=40, margin=40, aspect=1.0):
    """Pinhole view (no distortion) of a squares[0] x squares[1] board; a
    square is 1 unit along X and `aspect` units along Y; board corner
    (0, 0) at the outer corner."""
    sx, sy = squares
    px, py = sq_px, int(round(sq_px * aspect))
    flat = np.full((sy * py + 2 * margin, sx * px + 2 * margin), 255, np.uint8)
    for j in range(sy):
        for i in range(sx):
            if (i + j) % 2 == 0:
                flat[margin + j * py: margin + (j + 1) * py, margin + i * px: margin + (i + 1) * px] = 0
    R, _ = cv2.Rodrigues(np.asarray(rvec, float))
    H = K_TRUE @ np.column_stack([R[:, 0], R[:, 1], np.asarray(tvec, float)])
    S = np.array([[sq_px, 0, margin], [0, sq_px, margin], [0, 0, 1.0]])         # sq_px pixels per unit
    img = cv2.warpPerspective(flat, H @ np.linalg.inv(S), SIZE, borderValue=170)
    return cv2.GaussianBlur(img, (3, 3), 0)


POSES = [((0.2, -0.3, 0.05), (-6, -4, 22)), ((-0.25, 0.2, -0.1), (-5, -5, 20)), ((0.1, 0.35, 0.0), (-7, -3, 24)),
         ((-0.3, -0.25, 0.1), (-4, -4, 19)), ((0.0, 0.0, 0.3), (-6, -5, 21)),
         ((0.15, -0.2, -0.05), (-1, -2, 17)),          # cut by the right / bottom image edges
         ((-0.1, 0.25, 0.05), (-10, -7, 17)),          # cut by the left / top image edges
         ((0.3, 0.1, 0.0), (-6, -2, 23))]


def test_search_apply_and_solve_recovers_camera(tmp_path):
    paths = []
    for k, (rv, tv) in enumerate(POSES):
        p = tmp_path / f"view{k}.png"
        cv2.imwrite(str(p), render_view(rv, tv))
        paths.append(str(p))
    # Tk interpreters left over by earlier tests must be freed HERE, on the
    # main thread: freed by the garbage collector on a search worker thread,
    # Tcl aborts the process (Tcl_AsyncDelete from the wrong thread).
    gc.collect()
    root = tk.Tk()
    root.withdraw()
    try:
        canvas = tk.Canvas(root, width=300, height=200)
        node = CameraCalibrationNode("node_0", canvas)
        node.build_body()
        node._cb_manual_files = paths
        node._det_vars["square"].set("1")
        node._det_vars["threads"].set("4")
        node._on_open_search()
        sv = node._search_vars
        sv["cols"].set("8 9 10 11")
        sv["rows"].set("6 7 8")
        sv["widths"].set("0 600")
        sv["adaptive"].set("both")
        sv["normalize"].set("off")
        sv["fast"].set("on")
        node._on_search_start()
        t0 = time.time()
        while not node._search["state"]["finished"]:
            root.update()
            time.sleep(0.02)
            assert time.time() - t0 < 300
        for _ in range(3):
            root.update()
            time.sleep(0.3)
        assert node._search["state"]["error"] is None
        results = node._search["state"]["results"]
        found = [r for r in results if r and r.get("found")]
        assert len(found) >= 6
        assert any(len(r["rc"]) < 11 * 8 for r in found), "some views should be partial boards"
        # the dialog draws both preview modes without errors
        node._search_tree.selection_set(node._search_tree.get_children()[0])
        node._on_search_select()
        node._search_iv.render()
        node._search_mode.set("all")
        node._on_search_select()
        node._on_search_apply()
        assert node._cb is not None and len(node._cb["cands"]) == len(found)

        # Every corner's object point projects onto it with the TRUE camera
        # (each image's lattice has its own origin, so solve each pose).
        for c in node._cb["cands"]:
            obj = node._cand_obj(c)
            ok, rv, tv = cv2.solvePnP(obj, c["corners"].astype(np.float64), K_TRUE, None)
            proj, _ = cv2.projectPoints(obj, rv, tv, K_TRUE, None)
            assert np.abs(proj.reshape(-1, 2) - c["corners"]).max() < 0.5

        node._use_points_var.set(False)
        node._cb_use = {c["path"] for c in node._cb["cands"]}
        node._solve_sync()
        assert node._res is not None, node._status_var.get() if node._status_var else ""
        K = np.asarray(node._res["K"])
        assert abs(K[0, 0] - 800) < 8 and abs(K[1, 1] - 800) < 8
        assert abs(K[0, 2] - 480) < 8 and abs(K[1, 2] - 360) < 8
        out = node._outputs()
        assert out["cb_obj_pts"].shape[0] == len(node._cb_use)

        # save / load keeps every image's corners and lattice indices
        params = node.get_params()
        node2 = CameraCalibrationNode("node_1", canvas)
        node2.build_body()
        node2.set_params(params)
        a = {c["path"]: c for c in node._cb["cands"]}
        for c in node2._cb["cands"]:
            assert np.array_equal(c["rc"], a[c["path"]]["rc"])
            assert np.allclose(c["corners"], a[c["path"]]["corners"], atol=1e-3)
            assert np.array_equal(c["support"], a[c["path"]]["support"])
    finally:
        root.destroy()


def _run_search(root, node, cols, rows):
    node._on_open_search()
    sv = node._search_vars
    sv["cols"].set(cols)
    sv["rows"].set(rows)
    sv["widths"].set("0")
    sv["adaptive"].set("both")
    sv["normalize"].set("off")
    sv["fast"].set("on")
    node._on_search_start()
    t0 = time.time()
    while not node._search["state"]["finished"]:
        root.update()
        time.sleep(0.02)
        assert time.time() - t0 < 300
    root.update()


def _max_reprojection(node, cand):
    obj = node._cand_obj(cand)
    _ok, rv, tv = cv2.solvePnP(obj, cand["corners"].astype(np.float64), K_TRUE, None)
    proj, _ = cv2.projectPoints(obj, rv, tv, K_TRUE, None)
    return float(np.abs(proj.reshape(-1, 2) - cand["corners"]).max())


def test_square_x_and_y_sizes(tmp_path):
    """Squares 1.0 wide and 1.1 high: every usable image must get X / Y the
    right way round; images that cannot tell are not used."""
    aspect = 1.1
    poses = [((0.2, -0.3, 0.05), (-6, -5, 23)), ((-0.25, 0.2, -0.1), (-5, -6, 21)),
             ((0.1, 0.35, 0.0), (-7, -4, 25)), ((-0.3, -0.25, 0.1), (-4, -5, 20)),
             ((0.0, 0.0, 1.4), (4, -6, 22)),            # board turned ~80 degrees in the image
             ((0.15, -0.2, -0.05), (-1, -2, 17))]       # cut by the image edges
    paths = []
    for k, (rv, tv) in enumerate(poses):
        p = tmp_path / f"view{k}.png"
        cv2.imwrite(str(p), render_view(rv, tv, aspect=aspect))
        paths.append(str(p))
    gc.collect()
    root = tk.Tk()
    root.withdraw()
    try:
        canvas = tk.Canvas(root, width=300, height=200)
        node = CameraCalibrationNode("node_0", canvas)
        node.build_body()
        node._cb_manual_files = paths
        node._det_vars["square"].set("1")
        node._det_vars["square_y"].set(str(aspect))
        node._det_vars["ncol"].set("11")                 # the FULL board: 11 x 8 inner corners
        node._det_vars["nrow"].set("8")
        _run_search(root, node, "6 8 11", "5 8 11")       # 8 x 11 lies the other way round
        node._on_search_apply()
        cands = node._cb["cands"]
        assert len(cands) >= 5
        known = [c for c in cands if node._cand_axes(c) is not None]
        assert any(node._cand_axes(c) == "y" for c in known), "an 8 x 11 detection puts the lattice columns along Y"
        for c in known:
            assert _max_reprojection(node, c) < 0.5
        assert all(c["path"] in {k["path"] for k in known} for c in cands if c["path"] in node._cb_use)

        node._det_vars["square_y"].set("1")             # negative control: the wrong Y size must show
        assert max(_max_reprojection(node, c) for c in known) > 2.0
        node._det_vars["square_y"].set(str(aspect))

        node._use_points_var.set(False)
        node._cb_use = {c["path"] for c in known}
        node._solve_sync()
        K = np.asarray(node._res["K"])
        assert abs(K[0, 0] - 800) < 8 and abs(K[1, 1] - 800) < 8

        params = node.get_params()
        node2 = CameraCalibrationNode("node_1", canvas)
        node2.build_body()
        node2.set_params(params)
        assert node2._square_xy() == (1.0, aspect)
        legacy = dict(params)
        legacy["cb_settings"] = dict(params["cb_settings"])
        legacy["cb_settings"]["det"] = {k: v for k, v in params["cb_settings"]["det"].items() if k != "square_y"}
        node2.set_params(legacy)                        # projects saved before square X / Y: Y = X
        assert node2._square_xy() == (1.0, 1.0) and node2._det_vars["square_y"].get() == ""
    finally:
        root.destroy()


def test_lattice_axes_rules():
    from node_editor.chessboard_search import grid_rc
    from node_editor.nodes.camera_calibration_node import lattice_axes, lattice_obj
    assert lattice_axes(grid_rc(8, 11), (11, 8)) == "x"      # full board, normal way round
    assert lattice_axes(grid_rc(11, 8), (11, 8)) == "y"      # full board, turned 90 degrees
    assert lattice_axes(grid_rc(5, 6), (11, 8)) is None      # small part: fits both ways
    assert lattice_axes(grid_rc(9, 9), (9, 9)) is None       # as many columns as rows
    assert lattice_axes(grid_rc(9, 12), (11, 8)) is None     # larger than the board
    obj = lattice_obj(np.array([[2, 3]]), 1.0, 1.1, "x")
    assert np.allclose(obj, [[3.0, 2.2, 0.0]])
    obj = lattice_obj(np.array([[2, 3]]), 1.0, 1.1, "y")
    assert np.allclose(obj, [[3.3, 2.0, 0.0]])
