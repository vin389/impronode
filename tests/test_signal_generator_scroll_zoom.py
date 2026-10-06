from matplotlib.figure import Figure
import pytest

from node_editor.nodes.signal_generator_node import SignalGeneratorNode


class _DummyCanvasWidget:
    def draw_idle(self):
        pass


def _make_node():
    node = SignalGeneratorNode.__new__(SignalGeneratorNode)
    node._fig = Figure(figsize=(4, 3), dpi=100)
    node._ax = node._fig.add_subplot(111)
    node._ax.set_xlim(0, 10)
    node._ax.set_ylim(-2, 2)
    node._canvas_widget = _DummyCanvasWidget()
    return node


def test_scroll_zoom_with_ctrl_scales_x_only():
    node = _make_node()
    event = type("E", (), {"xdata": 5.0, "ydata": 0.0, "button": "up", "key": "control"})()

    node._on_plot_scroll(event)

    xlim = node._ax.get_xlim()
    ylim = node._ax.get_ylim()
    assert xlim != (0.0, 10.0)
    assert ylim == (-2.0, 2.0)


@pytest.mark.parametrize("key,modifiers", [
    ("shift", ()), ("ctrl+shift", ()),
    (None, ("shift",)), (None, ("ctrl", "shift")),
])
@pytest.mark.parametrize("button,factor", [("up", 0.9), ("down", 1.1)])
def test_scroll_zoom_with_shift_scales_y_only(key, modifiers, button, factor):
    node = _make_node()
    event = type("E", (), {"xdata": 5.0, "ydata": 0.5, "button": button,
                           "key": key, "modifiers": modifiers})()

    node._on_plot_scroll(event)

    xlim = node._ax.get_xlim()
    ylim = node._ax.get_ylim()
    assert xlim == (0.0, 10.0)
    assert ylim == pytest.approx((0.5 - 2.5 * factor, 0.5 + 1.5 * factor))


def test_plain_wheel_scales_both_axes():
    node = _make_node()
    event = type("E", (), {"xdata": 5.0, "ydata": 0.0, "button": "up", "key": None})()
    node._on_plot_scroll(event)
    assert node._ax.get_xlim() == pytest.approx((0.5, 9.5))
    assert node._ax.get_ylim() == pytest.approx((-1.8, 1.8))


def _mouse_event(ax, x, y, **extra):
    """Fake mouse event at data point (x, y): pixel position plus xdata/ydata,
    as matplotlib provides them (pan uses the pixel position)."""
    px, py = ax.transData.transform((x, y))
    return type("E", (), {"x": px, "y": py, "xdata": x, "ydata": y, **extra})()


def test_drag_pans_both_axes_and_release_stops_pan():
    node = _make_node()
    node._pan_state = {"active": False}
    node._on_plot_press(_mouse_event(node._ax, 5.0, 0.0, button=1))
    drag = _mouse_event(node._ax, 6.0, 0.5)
    node._on_plot_drag(drag)
    assert node._ax.get_xlim() == pytest.approx((-1.0, 9.0))
    assert node._ax.get_ylim() == pytest.approx((-2.5, 1.5))
    node._on_plot_release(drag)
    node._on_plot_drag(drag)
    assert not node._pan_state["active"]
    assert node._ax.get_xlim() == pytest.approx((-1.0, 9.0))


def test_drag_uses_view_at_press_time_so_pan_does_not_vibrate():
    """Regression test for the pan-vibration bug: a drag measured with the
    CURRENT transform (which the drag itself keeps changing) oscillated.
    Re-sending the same mouse position must not move the view again."""
    node = _make_node()
    node._pan_state = {"active": False}
    node._on_plot_press(_mouse_event(node._ax, 5.0, 0.0, button=1))
    drag = _mouse_event(node._ax, 6.0, 0.5)    # pixel position fixed from here on
    for _ in range(5):
        node._on_plot_drag(drag)
    assert node._ax.get_xlim() == pytest.approx((-1.0, 9.0))
    assert node._ax.get_ylim() == pytest.approx((-2.5, 1.5))
