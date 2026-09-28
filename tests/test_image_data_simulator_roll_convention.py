import numpy as np

from node_editor.nodes.image_data_simulator_node import camera_extrinsics


def test_roll_zero_camera_x_lies_in_world_xz_plane():
    rng = np.random.default_rng(0)
    for _ in range(50):
        view_dir = rng.normal(size=3)
        if np.linalg.norm(view_dir) < 1e-6:
            continue
        _rvec, _t, R = camera_extrinsics([0.0, 0.0, 0.0], view_dir, 0.0)
        x_c = R[0]
        assert abs(x_c[1]) < 1e-9, f"camera x has nonzero world-Y component: {x_c}"


def test_roll_zero_camera_y_is_world_negative_z_when_looking_along_y():
    _rvec, _t, R = camera_extrinsics([0.0, 0.0, 0.0], [0.0, 1.0, 0.0], 0.0)
    y_c = R[1]
    np.testing.assert_allclose(y_c, [0.0, 0.0, -1.0], atol=1e-9)
    x_c = R[0]
    assert abs(x_c[1]) < 1e-9


def test_roll_zero_camera_y_is_world_negative_z_when_looking_along_negative_y():
    # Same degenerate (view-direction-parallel-to-Y) fallback branch as the
    # +Y case; the fallback's "down" result does not depend on the sign of
    # d along the degenerate axis (matches the old Z-based scheme's
    # analogous +-Z-parallel behavior).
    _rvec, _t, R = camera_extrinsics([0.0, 0.0, 0.0], [0.0, -1.0, 0.0], 0.0)
    y_c = R[1]
    np.testing.assert_allclose(y_c, [0.0, 0.0, -1.0], atol=1e-9)
