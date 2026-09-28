from unittest.mock import Mock

import numpy as np
import pytest

import calcTemplateMatchPyr_codex as matcher
from node_editor.nodes.template_match_node import TemplateMatchNode
from node_editor.pin_types import PinType


def _inputs():
    image = np.random.default_rng(42).integers(0, 256, (128, 128), dtype=np.uint8)
    points = np.array([[48.0, 48.0], [80.0, 80.0]])
    return image, points


@pytest.mark.parametrize("num_levels", [1, 3])
def test_elapsed_timing_preserves_matching_results(num_levels):
    image, points = _inputs()
    expected = matcher.calcTemplateMatchPyr(image, image, points, 31, 8, num_levels)
    ctime = np.empty((len(points), num_levels), dtype=np.float64)
    actual = matcher.calcTemplateMatchPyr(
        image, image, points, 31, 8, num_levels, ctime=ctime,
    )

    assert len(actual) == 3  # Existing callers still unpack three arrays.
    for output, original in zip(actual, expected):
        np.testing.assert_array_equal(output, original)
    assert np.all(actual[1] == 1)
    assert np.all(np.isfinite(ctime))
    assert np.all(ctime >= 0)


def test_high_resolution_clock_is_recorded_by_point_and_level(monkeypatch):
    image, points = _inputs()
    # Processing order: point 0 levels 2,1,0; point 1 levels 2,1,0.
    # Large integer timestamps retain tiny differences before conversion.
    base_ns = 10**18
    ticks = [0, 3, 3, 5, 5, 6, 6, 12, 12, 17, 17, 21]
    clock = Mock(side_effect=[base_ns + tick * 100 for tick in ticks])
    monkeypatch.setattr(matcher.time, "perf_counter_ns", clock)
    ctime = np.empty((2, 3), dtype=np.float64)

    matcher.calcTemplateMatchPyr(
        image, image, points, 31, 8, 3, ctime=ctime,
    )

    np.testing.assert_allclose(ctime, [[1e-7, 2e-7, 3e-7], [4e-7, 5e-7, 6e-7]], atol=0)
    assert clock.call_count == 12


def test_failed_attempt_is_timed_and_unvisited_levels_are_nan(monkeypatch):
    image, _ = _inputs()
    points = np.array([[np.nan, 50.0], [0.0, 0.0]])
    clock = Mock(side_effect=[10_000_000_000, 10_000_250_000])
    monkeypatch.setattr(matcher.time, "perf_counter_ns", clock)
    ctime = np.zeros((2, 3), dtype=np.float64)

    _, status, _ = matcher.calcTemplateMatchPyr(
        image, image, points, 31, 8, 3, ctime=ctime,
    )

    np.testing.assert_array_equal(status, [0, 0])
    assert np.isnan(ctime[0]).all()
    assert np.isnan(ctime[1, :2]).all()
    assert ctime[1, 2] == 0.00025
    assert clock.call_count == 2


def test_empty_point_set_has_empty_timing_array():
    image, _ = _inputs()
    ctime = np.empty((0, 3), dtype=np.float64)
    outputs = matcher.calcTemplateMatchPyr(
        image, image, np.empty((0, 2)), 31, 8, 3, ctime=ctime,
    )
    assert ctime.shape == (0, 3)
    assert outputs[0].shape == (0, 2)


@pytest.mark.parametrize("buffer", [np.empty((2, 2)), np.empty((2, 3), dtype=np.int64)])
def test_invalid_timing_buffer_is_rejected(buffer):
    image, points = _inputs()
    with pytest.raises(ValueError, match="ctime must be a writable float64 array"):
        matcher.calcTemplateMatchPyr(image, image, points, 31, 8, 3, ctime=buffer)


def test_node_publishes_timing_output(monkeypatch):
    image, points = _inputs()
    node = TemplateMatchNode.__new__(TemplateMatchNode)
    node._buffered_inputs = {"prevImg": image, "nextImg": image, "prevPts": points}
    settings = {
        "template_size": (31, 31), "search_range": (8, 8), "num_levels": 3,
        "match_method": matcher.cv2.TM_CCOEFF_NORMED, "subpixel": True,
        "min_correlation": 0.5, "fine_search_radius": (4, 4),
        "sidelobe_exclusion_radius": 2,
    }
    monkeypatch.setattr(node, "_settings_from_inspector", lambda: settings)
    output_pin = next(pin for pin in node.get_pin_schema().outputs if pin.name == "ctime")
    assert output_pin.type == PinType.ARRAY
    assert output_pin.shape == (-1, -1)
    assert output_pin.dtype == "float64"

    for num_levels in (3, 1):
        settings["num_levels"] = num_levels
        output = node._compute_match()
        assert set(output) == {"nextPts", "status", "confidence", "ctime"}
        assert output["ctime"].shape == (len(points), num_levels)
        assert output["ctime"].dtype == np.float64
        assert np.all(np.isfinite(output["ctime"]))
        assert np.all(output["ctime"] >= 0)