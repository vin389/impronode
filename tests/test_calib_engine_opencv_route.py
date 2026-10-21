"""The OpenCV-only route (cv2.calibrateCamera / solvePnP) of run_plan must
give the same answer as the least-squares solver whenever it is used."""

import threading

import cv2
import numpy as np
import pytest

from node_editor import calib_engine as ce

SIZE = (1280, 960)
K_TRUE = np.array([[1100.0, 0, 650], [0, 1104.0, 470], [0, 0, 1]])
D_TRUE = np.array([-0.21, 0.09, 0.0008, -0.0006, -0.015, 0.02, -0.004, 0.001])
GRID = np.array([(c * 30.0, r * 30.0, 0) for r in range(7) for c in range(10)])


def board_views(n=14, seed=3, dist=D_TRUE[:5]):
    rng = np.random.default_rng(seed)
    views = []
    while len(views) < n:
        rv = rng.normal(0, 0.35, 3)
        tv = np.array([rng.uniform(-220, 0), rng.uniform(-160, 0), rng.uniform(600, 1000)])
        img, _ = cv2.projectPoints(GRID, rv, tv, K_TRUE, dist)
        img = img.reshape(-1, 2) + rng.normal(0, 0.15, (len(GRID), 2))
        if (img > 5).all() and (img[:, 0] < SIZE[0] - 5).all() and (img[:, 1] < SIZE[1] - 5).all():
            views.append(ce.View(f"b{len(views)}", kind="chessboard", obj=GRID, img=img))
    return views


def points_view(dist=D_TRUE[:5], unknown=False, seed=5):
    rng = np.random.default_rng(seed)
    obj = np.column_stack([rng.uniform(-400, 400, 40), rng.uniform(-300, 300, 40), rng.uniform(0, 300, 40)])
    rv, tv = np.array([0.1, -0.2, 0.05]), np.array([20.0, -10.0, 1500.0])
    img, _ = cv2.projectPoints(obj, rv, tv, K_TRUE, dist)
    img = img.reshape(-1, 2) + rng.normal(0, 0.2, (40, 2))
    obj = obj.copy()
    if unknown:
        obj[0, 2] = np.nan
    return ce.View("points", kind="points", obj=obj, img=img)


def plan_for(views, has_pts, model="simple", fixed=(), user=None, mode="joint", **kw):
    prob = ce.Problem(SIZE, list(views), dist_model=model, fixed=set(fixed), **kw)
    user = user or {}
    prob.K, prob.D = ce.merge_initial(ce.guess_K(SIZE), np.zeros(0), user, prob)
    return {"prob": prob, "pts": True if has_pts else None, "pt_rows": [], "mode": mode,
            "cb_paths": [v.name for v in views if v.kind == "chessboard"], "user_init": user}


def run_both(make):
    a = ce.run_plan(make())
    b = ce.run_plan(dict(make(), force_lsq=True))
    return a, b


CASES = {
    "boards simple, k3 fixed": dict(fixed={"k3"}),
    "boards rational, all free": dict(model="rational"),
    "boards, principal point fixed": dict(fixed={"cx", "cy", "k3"}),
    "boards, aspect fixed": dict(fixed={"k3"}, fix_aspect=True),
    "boards, focal fixed at a value": dict(fixed={"fx", "fy", "k3"}, user={"fx": 1100.0, "fy": 1104.0}),
    "boards, k1 fixed at a value": dict(fixed={"k1", "k3"}, user={"k1": -0.2}),
    "boards, p1 p2 fixed at 0": dict(fixed={"p1", "p2", "k3"}),
    "boards thin prism, s fixed": dict(model="thin_prism", fixed={"s1", "s2", "s3", "s4"}),
    "boards tilted, tau fixed": dict(model="tilted", fixed={"s1", "s2", "s3", "s4", "taux", "tauy"}),
}


@pytest.mark.parametrize("name", list(CASES))
def test_boards_opencv_route_matches_least_squares(name):
    cfg = dict(CASES[name])
    views = board_views()

    def make():
        return plan_for([ce.View(v.name, kind=v.kind, obj=v.obj, img=v.img) for v in views], False, **cfg)

    assert ce.solver_route(make())["boards"] == "opencv"
    a, b = run_both(make)
    assert a["route"]["reasons"] == [] and a["nfev"] == 0
    assert abs(a["rms"] - b["rms"]) < 2e-4, (a["rms"], b["rms"])
    K, Kb = np.asarray(a["K"]), np.asarray(b["K"])
    assert np.allclose(K, Kb, rtol=2e-4, atol=0.2), (K, Kb)
    for n in cfg.get("fixed", ()):
        if n in ce.DIST_NAMES[:ce.DIST_MODELS[cfg.get("model", "simple")]]:
            j = ce.DIST_NAMES.index(n)
            assert a["D"][j] == pytest.approx(cfg.get("user", {}).get(n, 0.0), abs=1e-12)
    if cfg.get("fix_aspect"):
        assert K[1, 1] / K[0, 0] == pytest.approx(1.0, rel=1e-9)            # automatic initial values: ratio 1
        assert Kb[1, 1] / Kb[0, 0] == pytest.approx(1.0, rel=1e-9)


def test_joint_and_two_step_with_points():
    boards = board_views()
    for mode in ("joint", "two_step"):
        def make():
            return plan_for([points_view()] + [ce.View(v.name, kind=v.kind, obj=v.obj, img=v.img) for v in boards],
                            True, fixed={"k3"}, mode=mode)
        route = ce.solver_route(make())
        assert route["reasons"] == []
        assert route["points"] == ("pnp" if mode == "two_step" else "opencv")
        a, b = run_both(make)
        assert abs(a["rms"] - b["rms"]) < 2e-4
        assert np.allclose(np.asarray(a["K"]), np.asarray(b["K"]), rtol=2e-4, atol=0.2)
        assert np.allclose(np.ravel(a["points_pose"][1]), np.ravel(b["points_pose"][1]), rtol=1e-4, atol=0.05)


def test_points_only_opencv_route():
    def make():
        return plan_for([points_view()], True, fixed={"k2", "k3", "p1", "p2"}, fix_aspect=True)
    assert ce.solver_route(make())["points"] == "opencv"
    a, b = run_both(make)
    assert abs(a["rms"] - b["rms"]) < 1e-3
    assert np.allclose(np.asarray(a["K"]), np.asarray(b["K"]), rtol=1e-3, atol=0.5)


def test_reasons_send_the_plan_to_least_squares():
    boards = board_views(n=6)
    r = ce.solver_route(plan_for([points_view(unknown=True)] + boards, True, fixed={"k3"}))
    assert r["points"] == "lsq" and any("'?'" in x for x in r["reasons"])
    r = ce.solver_route(plan_for([points_view(unknown=True)] + boards, True, fixed={"k3"}, mode="two_step"))
    assert r["boards"] == "opencv" and r["points"] == "lsq"
    r = ce.solver_route(plan_for(boards, False, fixed={"k3"}, loss="huber"))
    assert r["boards"] == "lsq" and any("huber" in x for x in r["reasons"])
    r = ce.solver_route(plan_for(boards, False, fixed={"fx", "k3"}))
    assert r["boards"] == "lsq" and any("fx" in x for x in r["reasons"])
    r = ce.solver_route(plan_for([points_view()] + boards, True, fixed={"k3"},
                                 weights={"point": 2.0, "unknown": 1.0, "line": 1.0, "chessboard": 1.0}))
    assert r["boards"] == "lsq" and any("weighted differently" in x for x in r["reasons"])


def test_cancel_stops_the_least_squares_solver():
    cancel = threading.Event()
    cancel.set()
    plan = dict(plan_for([points_view(unknown=True)] + board_views(n=4), True, fixed={"k3"}), cancel=cancel)
    with pytest.raises(ce.Cancelled):
        ce.run_plan(plan)
