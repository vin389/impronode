from functools import partial
from types import SimpleNamespace
import tkinter as tk

import numpy as np
import pytest

from node_editor.nodes.surface_mesh_generator_node import (
    SurfaceMeshGeneratorNode, _fit_cylinder, _perp_basis, _unit,
)


@pytest.fixture
def node(monkeypatch):
    # Tcl variables suffice for the inspector's reactive logic; no Tk windows.
    interpreter = tk.Tcl()
    for name in ("StringVar", "IntVar", "DoubleVar", "BooleanVar"):
        monkeypatch.setattr(tk, name, partial(getattr(tk, name), master=interpreter))
    result = SurfaceMeshGeneratorNode.__new__(SurfaceMeshGeneratorNode)
    result._init_state()
    result._request_downstream = None
    result.set_status = lambda *_args: None
    result._a1_count_var.set(5)
    result._a2_count_var.set(3)
    return result


def test_plane_intervals_update_both_directions(node):
    node._a1_start_var.set(2)
    node._a1_end_var.set(10)
    assert float(node._a1_interval_var.get()) == 2
    node._a1_count_var.set(9)
    assert float(node._a1_interval_var.get()) == 1
    node._a1_interval_var.set("-0.5")
    assert node._a1_end_var.get() == -2
    node._a2_start_var.set(-3)
    node._a2_end_var.set(7)
    assert float(node._a2_interval_var.get()) == 5
    node._a2_interval_var.set("2")
    assert node._a2_end_var.get() == 1
    # Partial text must not corrupt the end or raise a Tcl callback error.
    node._a2_interval_var.set("-")
    assert node._a2_end_var.get() == 1


def test_cylinder_degrees_defaults_and_arc_intervals(node):
    node._surface_type_var.set("cylinder")
    node._on_type_changed(redraw=False)
    assert node._a1_start_var.get() == 0
    assert node._a1_end_var.get() == 90
    assert node._a1_interval_var.get() == ""  # Radius not fitted yet.
    node._radius = 2
    node._update_intervals()
    assert float(node._a1_interval_var.get()) == pytest.approx(np.pi / 4)
    node._a1_interval_var.set(str(np.pi / 2))
    assert node._a1_end_var.get() == pytest.approx(180)
    a1, _ = node._linspaces()
    np.testing.assert_allclose(a1, np.linspace(0, np.pi, 5))
    node._a2_interval_var.set("3")
    assert node._a2_end_var.get() == 6


def test_single_sample_and_partial_count(node):
    node._a1_count_var.set(1)
    assert float(node._a1_interval_var.get()) == 0
    node._a1_interval_var.set("2")
    assert node._a1_end_var.get() == node._a1_start_var.get()
    assert node._mesh_counts() == (1, 3)
    assert node._linspaces()[0].shape == (1,)


def _cylinder_points(direction):
    direction = _unit(np.asarray(direction, dtype=float))
    e1, e2 = _perp_basis(direction)
    angles = np.linspace(-0.4, 1.5, 9)
    heights = np.linspace(-3, 4, 5)
    return np.array([np.array([10, -2, 5]) + 2 * np.cos(t) * e1
                     + 2 * np.sin(t) * e2 + h * direction
                     for t in angles for h in heights])


@pytest.mark.parametrize("direction", [[0, 0, 1], [1, 2, 3], [0, 0, -1]])
def test_axis_hint_is_refined_not_fixed(direction):
    direction = _unit(np.asarray(direction, dtype=float))
    points = _cylinder_points(direction)
    hint = direction + np.array([0.06, -0.03, 0])
    _, fitted, radius, errors = _fit_cylinder(points, axis_hint=hint)
    assert np.dot(fitted, direction) > 0.99999
    assert radius == pytest.approx(2, abs=1e-4)
    assert np.max(errors) < 1e-4


@pytest.mark.parametrize("axis", [[1, 0, 0], [0, 0, 1]])
def test_hint_selects_between_two_valid_cylinders(axis):
    # Intersection of x*x+y*y=4 and z*z+y*y=4: BOTH axes fit exactly.
    points = np.array([[sx * np.sqrt(4 - y*y), y, sz * np.sqrt(4 - y*y)]
                       for y in (-1.5, -0.5, 0.8, 1.6)
                       for sx in (-1, 1) for sz in (-1, 1)])
    hint = np.asarray(axis) + np.array([0.01, 0.02, 0.01])
    _, fitted, radius, errors = _fit_cylinder(points, axis_hint=hint)
    assert np.dot(fitted, axis) > 0.99999
    assert radius == pytest.approx(2, abs=1e-4)
    assert np.max(errors) < 1e-4


@pytest.mark.parametrize("hint", [[0, 0, 0], [np.nan, 0, 1], [0, 1]])
def test_invalid_axis_hint_rejected(hint):
    with pytest.raises(ValueError):
        _fit_cylinder(_cylinder_points([0, 0, 1]), axis_hint=hint)


def test_preview_and_dense_mesh_agree_in_degrees(node):
    node._surface_type_var.set("cylinder")
    node._on_type_changed(redraw=False)
    node._xo = np.zeros(3)
    node._vx, node._vy, node._vz = np.eye(3)
    node._radius = 2
    node._mdraw_var.set(5)
    node._ndraw_var.set(3)
    dense, a1, _ = node._generate_dense()
    node._build_preview_grid()
    np.testing.assert_allclose(dense, node._preview_grid)
    np.testing.assert_allclose(dense[-1, 0], [0, 2, 0], atol=1e-12)
    assert a1[-1] == pytest.approx(np.pi / 2)


def test_reverse_theta_preserves_height_direction(node):
    points = _cylinder_points([0, 0, 1])
    node._surface_type_var.set("cylinder")
    node._fit_cylinder_param(points)
    vx, vy, vz = node._vx.copy(), node._vy.copy(), node._vz.copy()
    node._reverse_theta_var.set(True)
    node._fit_cylinder_param(points)
    np.testing.assert_allclose(node._vx, vx)
    np.testing.assert_allclose(node._vy, -vy)
    np.testing.assert_allclose(node._vz, vz)


def test_saved_radian_ranges_migrate_and_degrees_roundtrip(node):
    node.set_params({"surface_type": "cylinder", "a1_start": -np.pi / 2,
                     "a1_end": np.pi, "height_idx": 2})
    assert node._a1_start_var.get() == pytest.approx(-90)
    assert node._a1_end_var.get() == pytest.approx(180)
    node._on_type_changed(redraw=False)  # Opening inspector must not reset saved range.
    assert node._a1_end_var.get() == pytest.approx(180)
    for var, value in zip(node._axis_hint_vars, (1, 2, 3)):
        var.set(str(value))
    saved = node.get_params()
    assert saved["theta_unit"] == "degrees"
    node.set_params(saved)
    assert node._a1_end_var.get() == pytest.approx(180)
    assert [v.get() for v in node._axis_hint_vars] == ["1", "2", "3"]


def test_auto_fit_ranges_converts_angles_to_degrees(node):
    node._surface_type_var.set("cylinder")
    node._xo = np.zeros(3)
    node._vx, node._vy, node._vz = np.eye(3)
    node._radius = 2
    node._points = np.array([[2, 0, 0], [0, 2, 1]])
    node._on_apply = lambda: None
    node._auto_fit_ranges()
    assert node._a1_start_var.get() == pytest.approx(-4.5)
    assert node._a1_end_var.get() == pytest.approx(94.5)


@pytest.mark.parametrize("intensity", [0.0, 0.35, 1.0])
def test_preview_perspective_projection(node, intensity):
    node._preview_canvas = SimpleNamespace(
        winfo_exists=lambda: True, winfo_width=lambda: 400, winfo_height=lambda: 300)
    node._cam.update(yaw=0.0, pitch=0.0, target=[5.0, 6.0, 7.0])
    node._perspective_var.set(intensity)
    scale = node._scale(2.0)
    relative = np.array([[1., -1., 1.], [1., 0., 1.], [1., 1., 1.], [1., -100., 1.]])
    original = relative + node._cam["target"]
    points = original.copy()
    projection = node._project(points, scale)
    if intensity:
        focal = 2 * (4 / intensity - 3)
        factors = focal / np.maximum(focal + relative[:, 1], focal * 0.1)
        assert projection[0, 0] > projection[1, 0] > projection[2, 0]
    else:
        factors = np.ones(4)
    np.testing.assert_allclose(projection[:, 0], 200 + scale * factors)
    np.testing.assert_allclose(projection[:, 1], 150 - scale * factors)
    assert np.isfinite(projection).all()
    np.testing.assert_array_equal(points, original)


def test_perspective_settings_roundtrip_and_clamping(node):
    assert node._perspective_var.get() == 0.35
    node._perspective_var.set(0.72)
    saved = node.get_params()
    node._perspective_var.set(0)
    node.set_params(saved)
    assert node._perspective_var.get() == 0.72
    for value, expected in ((-1, 0), (2, 1)):
        node.set_params({"perspective_intensity": value})
        assert node._perspective_var.get() == expected
    node.set_params({})
    assert node._perspective_var.get() == 0.35


def test_perspective_does_not_change_dense_mesh_or_cache(node):
    node._xo = np.zeros(3)
    node._vx, node._vy, node._vz = np.eye(3)
    node._perspective_var.set(0)
    first = node._generate_dense()
    node._perspective_var.set(1)
    assert node._generate_dense() is first