import pytest

from node_editor.nodes.image_sequence_node import ImageSequenceNode


def test_image_sequence_exposes_second_frame_index_output():
    schema = ImageSequenceNode.get_pin_schema(None)

    assert [(pin.name, pin.label) for pin in schema.outputs][2:4] == [
        ("frame_index_1", "index 1"),
        ("frame_index_2", "index 2"),
    ]


def test_image_sequence_fractional_fps_converts_to_milliseconds():
    assert ImageSequenceNode._fps_interval_ms("0.3333") == 3000
    assert ImageSequenceNode._fps_interval_ms("2.5") == 400


@pytest.mark.parametrize("value", ["", "not-a-number", "0", "-1", "nan", "inf"])
def test_image_sequence_rejects_invalid_fps(value):
    with pytest.raises(ValueError, match="FPS must be a positive number"):
        ImageSequenceNode._fps_interval_ms(value)
