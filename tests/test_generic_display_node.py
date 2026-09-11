import numpy as np

from node_editor.nodes.generic_display_node import GenericDisplayNode


def test_data_label_includes_length_when_available():
    assert GenericDisplayNode._data_label_text(None, [1, 2, 3]) == (
        "Data (type: list, len: 3)"
    )


def test_data_label_includes_numpy_array_shape():
    value = np.zeros((2, 3, 4))

    assert GenericDisplayNode._data_label_text(None, value) == (
        "Data (type: ndarray, len: 2, shape: (2, 3, 4))"
    )
