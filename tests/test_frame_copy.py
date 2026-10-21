import os

import cv2
import numpy as np
import pytest

from node_editor import frame_copy as fc


def _images(folder, n=12):
    paths = []
    for k in range(n):
        p = folder / f"img_{k + 1:04d}.png"
        cv2.imwrite(str(p), np.full((20, 30, 3), 10 * k, np.uint8))
        paths.append(str(p))
    return paths


def _video(path, n=20):
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for k in range(n):
        w.write(np.full((48, 64, 3), 10 * k, np.uint8))            # frame k has brightness 10 k
    w.release()
    return str(path)


def test_parse_and_runs():
    sel, dropped = fc.parse_frames("1:3 5 8:10 2", 9)
    assert sel == [0, 1, 2, 4, 7, 8] and dropped == 1              # '2' repeated once; 10 beyond 9
    assert fc.runs(sel) == [(0, 2), (4, 4), (7, 8)]
    assert fc.parse_frames("", 5) == ([], 0)
    with pytest.raises(ValueError):
        fc.parse_frames("3:1", 5)


@pytest.mark.parametrize("pattern, ok", [("selected_%06d", True), ("f%d", True), ("a_%06d_b", True),
                                         ("noNumber", False), ("two_%d_%d", False), ("x_%06d.png", False),
                                         ("bad/%06d", False), ("", False), ("p_%s", False)])
def test_pattern_rules(pattern, ok):
    assert (fc.pattern_error(pattern) is None) is ok


def test_names_for_images_and_video(tmp_path):
    imgs = _images(tmp_path)
    dest = str(tmp_path / "out")
    t = fc.plan_targets("images", imgs, [2, 5], dest, {"mode": "original", "format": "same"})
    assert [os.path.basename(x[2]) for x in t] == ["img_0003.png", "img_0006.png"] and t[0][3] is True
    t = fc.plan_targets("images", imgs, [2, 5], dest, {"mode": "original", "format": "jpg"})
    assert [os.path.basename(x[2]) for x in t] == ["img_0003.jpg", "img_0006.jpg"] and t[0][3] is False
    t = fc.plan_targets("images", imgs, [5, 2], dest, {"mode": "pattern", "pattern": "sel_%04d",
                                                      "number": "sequence", "format": "png"})
    assert [(x[0], os.path.basename(x[2])) for x in t] == [(5, "sel_0001.png"), (2, "sel_0002.png")]
    t = fc.plan_targets("images", imgs, [5], dest, {"mode": "pattern", "pattern": "sel_%04d",
                                                   "number": "frame", "format": "png"})
    assert os.path.basename(t[0][2]) == "sel_0006.png"
    t = fc.plan_targets("video", ["D:/v/clip01.mp4"], [0, 122], dest, {"mode": "original", "format": "jpg"})
    assert [os.path.basename(x[2]) for x in t] == ["clip01_FRM000001.jpg", "clip01_FRM000123.jpg"]
    with pytest.raises(ValueError, match="jpg or png"):
        fc.plan_targets("video", ["v.mp4"], [0], dest, {"mode": "original", "format": "same"})
    with pytest.raises(ValueError, match="own source"):
        fc.plan_targets("images", imgs, [0], str(tmp_path), {"mode": "original", "format": "same"})
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(), b.mkdir()
    same = [str(a / "x.png"), str(b / "x.png")]
    with pytest.raises(ValueError, match="same file name"):
        fc.plan_targets("images", same, [0, 1], dest, {"mode": "original", "format": "png"})


def _run(kind, paths, targets, **kw):
    st = fc.new_copy_state(len(targets))
    fc.copy_worker(kind, paths, targets, st, **kw)
    return st


def test_copy_images_exact_and_reencoded(tmp_path):
    imgs = _images(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    t = fc.plan_targets("images", imgs, [0, 3, 4], str(out), {"mode": "original", "format": "same"})
    st = _run("images", imgs, t)
    assert st["errors"] == [] and len(st["written"]) == 3
    assert open(imgs[3], "rb").read() == open(out / "img_0004.png", "rb").read()      # byte for byte
    t = fc.plan_targets("images", imgs, [3], str(out), {"mode": "pattern", "pattern": "s_%03d",
                                                        "number": "frame", "format": "jpg"})
    st = _run("images", imgs, t, jpeg_quality=90)
    img = cv2.imread(str(out / "s_004.jpg"))
    assert img is not None and abs(float(img.mean()) - 30) < 2
    assert not [p for p in os.listdir(out) if p.endswith(".part")]


def test_copy_video_frames_are_the_right_frames(tmp_path):
    video = _video(tmp_path / "clip.avi")
    out = tmp_path / "out"
    out.mkdir()
    frames = [17, 2, 3, 9]                                          # any order
    t = fc.plan_targets("video", [video], frames, str(out), {"mode": "original", "format": "png"})
    st = _run("video", [video], t)
    assert st["errors"] == [] and st["error"] is None
    assert [os.path.basename(p) for p in st["written"]] == [os.path.basename(x[2]) for x in t]
    for k in frames:
        img = cv2.imread(str(out / f"clip_FRM{k + 1:06d}.png"))
        assert abs(float(img.mean()) - 10 * k) < 3, (k, img.mean())


def test_skip_existing_and_cancel(tmp_path):
    imgs = _images(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    (out / "img_0001.png").write_bytes(b"keep me")
    t = fc.plan_targets("images", imgs, [0, 1], str(out), {"mode": "original", "format": "same"})
    st = _run("images", imgs, t, skip_existing=True)
    assert (out / "img_0001.png").read_bytes() == b"keep me" and len(st["written"]) == 2
    st = fc.new_copy_state(len(t))
    st["cancel"].set()
    fc.copy_worker("images", imgs, t, st)
    assert st["finished"] and st["done"] == 0
