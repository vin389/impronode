import math

import cv2
import numpy as np

from node_editor import chessboard_search as cs


def render_board(squares_x, squares_y, H, size, sq=60, margin=60):
    """Chessboard (squares_x x squares_y squares) drawn flat, then warped by
    the homography H into an image of `size` = (w, h). Returns the image and
    H_total mapping board corner coordinates (i, j) in squares -> pixels."""
    w0, h0 = squares_x * sq + 2 * margin, squares_y * sq + 2 * margin
    flat = np.full((h0, w0), 255, np.uint8)
    for j in range(squares_y):
        for i in range(squares_x):
            if (i + j) % 2 == 0:
                flat[margin + j * sq: margin + (j + 1) * sq, margin + i * sq: margin + (i + 1) * sq] = 0
    img = cv2.warpPerspective(flat, H, size, borderValue=200)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    S = np.array([[sq, 0, margin], [0, sq, margin], [0, 0, 1.0]])
    return img, H @ S


def to_board(H_total, uv):
    p = cv2.perspectiveTransform(uv.reshape(-1, 1, 2).astype(np.float64), np.linalg.inv(H_total)).reshape(-1, 2)
    return p


def check_mapping(res, H_total):
    """Every merged corner must sit on a true board corner, and ONE lattice
    symmetry + offset must map all lattice indices to the true ones."""
    board = to_board(H_total, res["corners"].astype(np.float64))
    true_ij = np.round(board).astype(np.int64)
    assert np.abs(board - true_ij).max() < 0.05
    true_rc = true_ij[:, ::-1]                         # (row = j, col = i)
    rc = res["rc"].astype(np.int64)
    ok = []
    for g in cs.LATTICE_OPS:
        t = true_rc - rc @ g.T
        if (t == t[0]).all():
            ok.append(g)
    assert len(ok) == 1
    assert len({tuple(x) for x in rc.tolist()}) == len(rc)


def test_partial_boards_merge_into_one_consistent_lattice():
    # 12 x 9 squares (11 x 8 inner corners); the right and bottom parts are outside the image.
    H = np.array([[0.9, 0.0, 0.0], [0.0, 0.9, 0.0], [0.0, 0.0, 1.0]])
    img, Ht = render_board(12, 9, H, (600, 520))
    settings = cs.build_settings([8, 9], [7, 8], [0], [False, True], [False], [False])
    dets = []
    for k, s in enumerate(settings):
        g = cs.find_corners(img, s, {})
        if g is not None:
            dets.append((k, g))
            check_mapping({"rc": cs.grid_rc(*g.shape[:2]), "corners": g.reshape(-1, 2)}, Ht)
    assert len({d[1].shape for d in dets}) >= 2, "the test needs detections of different sizes"
    res = cs.merge_detections(img, dets)
    assert res is not None
    check_mapping(res, Ht)
    assert len(res["rc"]) >= max(g.shape[0] * g.shape[1] for _k, g in dets)
    assert res["rc"].min() == 0
    assert (res["support"] >= 1).all()


def test_conflicting_detection_is_rejected():
    H = np.array([[0.9, 0.0, 30.0], [0.0, 0.9, 30.0], [0.0, 0.0, 1.0]])
    img, Ht = render_board(10, 8, H, (700, 620))
    g = cs.find_corners(img, cs.Setting(9, 7, 0, True, False, False))
    assert g is not None
    m = cs.ImageMerger()
    assert m.add(g, 0) == "new"
    assert m.add(g, 1) == "merged"
    bad = g.copy()
    bad[0, 0] = g[0, 1]                                # two lattice indices on the same image corner
    bad[0, 1] = g[0, 0] + (g[0, 0] - g[0, 1])           # ... and one corner somewhere else
    assert m.add(bad, 2) == "rejected"
    line = g[:1, :]                                     # shares only one lattice row: ambiguous
    assert m.add(line, 3) == "rejected"
    assert m.best_count() == 9 * 7


def test_search_worker_runs_and_counts(tmp_path):
    H = np.array([[0.9, 0.05, 10.0], [-0.03, 0.92, 15.0], [0.0, 0.0, 1.0]])
    img, _Ht = render_board(10, 8, H, (720, 640))
    p = tmp_path / "board.png"
    cv2.imwrite(str(p), img)
    blank = tmp_path / "blank.png"
    cv2.imwrite(str(blank), np.full((200, 300), 128, np.uint8))
    settings = cs.build_settings([7, 9], [7], [0, 400], [True], [False], [True])
    st = cs.new_search_state(2, len(settings))
    cs.search_worker([str(p), str(blank)], settings, 2, st)
    assert st["error"] is None and st["finished"]
    assert st["done"] == st["total"] == 2 * len(settings)
    r0, r1 = st["results"]
    assert r0["found"] and len(r0["rc"]) == 9 * 7 and not r0["partial"]
    assert not r1["found"]
    assert math.isfinite(r0["sharpness"])


def test_false_detection_on_occluder_edge_is_rejected():
    H = np.array([[0.8, 0.05, 20], [-0.03, 0.8, 20], [0.00002, 0.00003, 1]])
    img, Ht = render_board(12, 9, H, (900, 700))
    cv2.rectangle(img, (500, 60), (900, 330), 150, -1)
    settings = cs.build_settings(range(3, 12), range(3, 9), [0], [False, True], [False], [False])
    dets = [(k, g) for k, s in enumerate(settings) for g in [cs.find_corners(img, s, {})] if g is not None]
    assert dets
    for _k, g in dets:
        check_mapping({"rc": cs.grid_rc(*g.shape[:2]), "corners": g.reshape(-1, 2)}, Ht)
    res = cs.merge_detections(img, dets)
    check_mapping(res, Ht)
