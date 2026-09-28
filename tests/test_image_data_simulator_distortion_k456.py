import numpy as np

from node_editor.nodes.image_data_simulator_node import (
    camera_model, default_camera, default_settings, normalize_settings, simulate_data,
)


def test_default_camera_has_8_dist_coeffs_with_k1_minus_0_3():
    for idx in (0, 1):
        cam = default_camera(idx)
        assert len(cam["dist"]) == 8
        assert cam["dist"][0] == -0.3
        assert cam["dist"][1:] == [0.0] * 7


def test_camera_model_dvec_has_8_coefficients():
    cam = default_camera(0)
    m = camera_model(cam)
    assert m["dvec"].shape == (8,)
    assert m["dvec"][0] == -0.3


def test_simulate_data_dvecs_shape_is_8():
    s = default_settings()
    res = simulate_data(s)
    assert res["dvecs"].shape == (len(s["cameras"]), 8)
    np.testing.assert_allclose(res["dvecs"][:, 0], -0.3)


def test_normalize_settings_pads_legacy_5_coefficient_dist():
    saved = default_settings()
    for cam in saved["cameras"]:
        cam["dist"] = [0.1, 0.2, 0.3, 0.4, 0.5]   # legacy 5-length save
    s = normalize_settings(saved)
    for cam in s["cameras"]:
        assert len(cam["dist"]) == 8
        assert cam["dist"][:5] == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert cam["dist"][5:] == [0.0, 0.0, 0.0]
