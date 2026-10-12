import cv2
import numpy as np
import pytest

from node_editor.nodes.image_sequence_node import ImageSequenceNode
from node_editor.video_frame_reader import VideoFrameReader


class _Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


def _node_with_list(paths, end="5", frame_2="i+1"):
    """Build just enough node state to exercise non-GUI validation."""
    node = ImageSequenceNode.__new__(ImageSequenceNode)
    node._source_var = _Value("images")
    node._method_var = _Value("pattern")
    node._batch_start_var = _Value("1")
    node._batch_end_var = _Value(end)
    node._batch_step_var = _Value("1")
    node._batch_current_var = _Value("1")
    node._batch_frame_1_var = _Value("i")
    node._batch_frame_2_var = _Value(frame_2)
    node._file_paths = list(paths)
    node._loop_entries = {}
    node._batch_first_btn = None
    node._batch_prev_btn = None
    node._batch_current_btn = None
    node._batch_next_btn = None
    node._batch_last_btn = None
    node._batch_play_btn = None
    return node


def test_numbered_names_make_the_list_from_a_number_range():
    assert ImageSequenceNode._numbered_paths("img_%04d.png", 3, 6, 1) == [
        "img_0003.png", "img_0004.png", "img_0005.png", "img_0006.png"]
    assert ImageSequenceNode._numbered_paths("f%d.jpg", 10, 4, -3) == ["f10.jpg", "f7.jpg", "f4.jpg"]


@pytest.mark.parametrize("pattern", ["img.png", "img_%d_%d.png", "100%_img_%d.png", "img_%s.png"])
def test_numbered_names_need_exactly_one_number_field(pattern):
    with pytest.raises(ValueError):
        ImageSequenceNode._numbered_paths(pattern, 1, 3, 1)


def test_batch_positions_are_limited_to_the_list_length():
    paths = [f"image_{i:04d}.png" for i in range(1, 7)]       # 6 files (may not exist yet)
    assert _node_with_list(paths, end="5", frame_2="i+1")._validate_loop_fields() is True
    # i+1 at i = 6 would need a 7th file
    assert _node_with_list(paths, end="6", frame_2="i+1")._validate_loop_fields() is False
    # an empty list cannot be run
    assert _node_with_list([], end="5")._validate_loop_fields() is False


def test_video_source_uses_the_exact_video_frame_count(tmp_path):
    path = str(tmp_path / "clip.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    for k in range(25):
        writer.write(np.full((48, 64, 3), k * 10, np.uint8))
    writer.release()
    node = _node_with_list([], end="25", frame_2="i")
    node._source_var = _Value("video")
    node._video_reader = VideoFrameReader(path)
    node._video_reader.build_index()
    try:
        assert node._frame_total() == 25
        assert node._validate_loop_fields() is True
        node._batch_end_var.set("26")
        assert node._validate_loop_fields() is False
        assert node._absolute_file_paths() == []                # a video has no file list
    finally:
        node._video_reader.close()


def test_video_reader_returns_the_requested_frame_in_any_order(tmp_path):
    """Every frame (sequential, jumps forward/backward, after a broken seek)
    must equal the frame decoded in order."""
    path = str(tmp_path / "clip.avi")
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    for k in range(400):
        frame = np.zeros((48, 64, 3), np.uint8)
        cv2.putText(frame, str(k), (2, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()
    reference, cap = [], cv2.VideoCapture(path)
    while True:
        ok, f = cap.read()
        if not ok:
            break
        reference.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()

    reader = VideoFrameReader(path)
    order = [0, 1, 2, 50, 10, 399, 3, 260, 259, 0]
    try:
        for k in order[:4]:                                   # before the index exists
            assert np.array_equal(reader.read(k), reference[k])
        reader.build_index()
        assert reader.frame_count == len(reference)
        for k in order:
            assert np.array_equal(reader.read(k), reference[k])
        assert reader.read(len(reference)) is None

        class OffByFive:                                      # simulate an unreliable seek
            def __init__(self, cap):
                self.cap = cap

            def __getattr__(self, name):
                return getattr(self.cap, name)

            def set(self, prop, value):
                return self.cap.set(prop, value + 5 if prop == cv2.CAP_PROP_POS_FRAMES else value)

        reader._cache.clear()
        reader._cap = OffByFive(reader._cap)
        reopen = reader._reopen
        reader._reopen = lambda: (reopen(), setattr(reader, "_cap", OffByFive(reader._cap)))
        for k in (300, 30, 350, 120):
            assert np.array_equal(reader.read(k), reference[k])
        assert reader.seek_failures > 0
    finally:
        reader.close()


def test_file_list_popup_is_placed_beside_the_pointer_or_flipped_on_screen():
    assert ImageSequenceNode._popup_position_near_pointer(
        100, 200, 700, 500, 1920, 1080,
    ) == (116, 216)
    assert ImageSequenceNode._popup_position_near_pointer(
        1900, 1060, 700, 500, 1920, 1080,
    ) == (1184, 544)


def test_play_marks_batch_complete_only_after_the_final_pair_is_emitted():
    node = ImageSequenceNode.__new__(ImageSequenceNode)
    node._batch_running = True
    node._batch_indices = [2, 3, 4]
    node._batch_position = 3
    node._trigger_linked = False
    stopped = []
    node._stop_batch = lambda status: stopped.append(status)
    node._schedule_batch_tick = lambda: stopped.append("scheduled")

    node._on_batch_step_complete()

    assert stopped == ["batch complete"]
