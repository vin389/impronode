from node_editor.nodes.image_sequence_node import ImageSequenceNode


class _Value:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


def _pattern_node_without_paths():
    """Build just enough node state to exercise non-GUI validation."""
    node = ImageSequenceNode.__new__(ImageSequenceNode)
    node._method_var = _Value("pattern")
    node._pattern_var = _Value("image_%04d.png")
    node._batch_start_var = _Value("1")
    node._batch_end_var = _Value("5")
    node._batch_step_var = _Value("1")
    node._batch_current_var = _Value("1")
    node._batch_frame_1_var = _Value("i")
    node._batch_frame_2_var = _Value("i+1")
    node._file_paths = []
    node._loop_entries = {}
    node._batch_first_btn = None
    node._batch_prev_btn = None
    node._batch_current_btn = None
    node._batch_next_btn = None
    node._batch_last_btn = None
    node._batch_play_btn = None
    return node


def test_pattern_batch_setup_is_valid_before_paths_exist():
    node = _pattern_node_without_paths()

    assert node._validate_loop_fields() is True


def test_pattern_edit_list_includes_frame_two_offset_and_preserves_batch():
    node = _pattern_node_without_paths()
    captured = {}
    node._set_file_paths = lambda paths, *, reset_batch: captured.update(
        paths=paths, reset_batch=reset_batch,
    )
    node._edit_file_list = lambda *, preserve_batch: captured.update(
        preserve_batch=preserve_batch,
    )

    node._edit_pattern_file_list()

    assert captured == {
        "paths": [f"image_{index:04d}.png" for index in range(1, 7)],
        "reset_batch": False,
        "preserve_batch": True,
    }


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
