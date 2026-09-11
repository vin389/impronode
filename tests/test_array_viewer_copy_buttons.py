import unittest
import tkinter as tk
from pathlib import Path
import json
import tempfile

import numpy as np

from node_editor.nodes.array_nodes import ArrayViewerNode
from node_editor.pin_types import PinType, pins_compatible
from node_editor.project_context import set_project_file_path


class TestArrayViewerCopyButtons(unittest.TestCase):
    def test_array_viewer_copy_csv_uses_full_array_and_selected_slice(self):
        root = tk.Tk()
        root.withdraw()
        canvas = tk.Canvas(root)
        node = ArrayViewerNode("viewer", canvas)
        node.build_body()

        arr = np.array([[1, 2, 3], [4, 5, 6]])
        node._current_array = arr
        node._rows_var.set("0:2")
        node._cols_var.set("1:3")

        self.assertEqual(node._array_to_csv_text(arr), "1,2,3\n4,5,6")
        self.assertEqual(node._array_to_csv_text(node._get_selected_array(arr)), "2,3\n5,6")

        root.destroy()

    def test_any_source_is_compatible_with_array_viewer_and_coerces_ndarray(self):
        self.assertTrue(pins_compatible(PinType.ANY, PinType.ARRAY))

        root = tk.Tk()
        root.withdraw()
        canvas = tk.Canvas(root)
        node = ArrayViewerNode("viewer", canvas)
        node.build_body()

        arr = np.array([[1, 2], [3, 4]])
        result = node.compute({"array": arr})

        self.assertIsInstance(node._current_array, np.ndarray)
        self.assertTrue(np.array_equal(node._current_array, arr))
        self.assertTrue(np.array_equal(result["array"], arr))

        root.destroy()

    def test_scalar_and_trigger_sources_are_accepted_as_1d_arrays(self):
        self.assertTrue(pins_compatible(PinType.SCALAR, PinType.ARRAY))
        self.assertTrue(pins_compatible(PinType.TRIGGER, PinType.ARRAY))

        root = tk.Tk()
        root.withdraw()
        canvas = tk.Canvas(root)
        node = ArrayViewerNode("viewer", canvas)
        node.build_body()

        scalar_result = node.compute({"array": 7.5})
        self.assertTrue(np.array_equal(scalar_result["array"], np.array([7.5])))

        trigger_result = node.compute({"array": True})
        self.assertTrue(np.array_equal(trigger_result["array"], np.array([True])))

        root.destroy()

    def test_array_viewer_serializes_and_restores_array_data(self):
        root = tk.Tk()
        root.withdraw()
        canvas = tk.Canvas(root)
        node = ArrayViewerNode("viewer", canvas)
        node.build_body()

        arr = np.array([[1, 2], [3, 4]], dtype=np.float64)
        node._current_array = arr
        node._update_summary(arr)

        saved = node.get_params()
        self.assertIn("data", saved)
        self.assertEqual(saved["data"], [[1.0, 2.0], [3.0, 4.0]])

        restored = ArrayViewerNode("viewer_restored", canvas)
        restored.build_body()
        restored.set_params(saved)

        self.assertTrue(np.array_equal(restored._current_array, arr))
        self.assertEqual(restored._shape_var.get(), "shape: (2×2)\nfloat64")

        root.destroy()

    def test_array_viewer_keeps_saved_data_when_recomputed_without_input(self):
        root = tk.Tk()
        root.withdraw()
        canvas = tk.Canvas(root)
        node = ArrayViewerNode("viewer", canvas)
        node.build_body()

        arr = np.array([[10.0, 20.0], [30.0, 40.0]])
        node.set_params({"rows_text": ":", "cols_text": ":", "reshape_text": "", "data": arr.tolist()})

        result = node.compute({})

        self.assertTrue(np.array_equal(node._current_array, arr))
        self.assertTrue(np.array_equal(result["array"], arr))
        self.assertEqual(node._shape_var.get(), "shape: (2×2)\nfloat64")

        root.destroy()

    def test_array_viewer_copy_csv_uses_dtype_appropriate_precision(self):
        root = tk.Tk()
        root.withdraw()
        canvas = tk.Canvas(root)
        node = ArrayViewerNode("viewer", canvas)
        node.build_body()

        float32_arr = np.array([[1.23456789, 2.34567891]], dtype=np.float32)
        float64_arr = np.array([[1.2345678901234567, 2.345678901234567]], dtype=np.float64)

        float32_csv = node._array_to_csv_text(float32_arr)
        float64_csv = node._array_to_csv_text(float64_arr)

        self.assertIn("1.23456788", float32_csv)
        self.assertIn("2.34567881", float32_csv)
        self.assertIn("1.2345678901234567", float64_csv)
        self.assertIn("2.3456789012345669", float64_csv)

        root.destroy()

    def test_array_viewer_project_uses_relative_npy_sidecar(self):
        root = tk.Tk()
        root.withdraw()
        canvas = tk.Canvas(root)
        node = ArrayViewerNode("node_17", canvas)
        node.build_body()
        array = np.arange(120_000, dtype=np.int16).reshape(40, 50, 60)
        node._current_array = array
        node._update_summary(array)

        try:
            with tempfile.TemporaryDirectory() as directory:
                project_path = Path(directory) / "measurement.xlsx"
                set_project_file_path(str(project_path))
                params = node.get_params()

                self.assertNotIn("data", params)
                self.assertEqual(
                    params["array_viewer_file"],
                    "measurement.node_17.array_viewer.npy",
                )
                self.assertLess(len(json.dumps(params)), 32_000)
                self.assertTrue((project_path.parent / params["array_viewer_file"]).exists())

                restored = ArrayViewerNode("node_17_restored", canvas)
                restored.build_body()
                restored.set_params(params)

                self.assertEqual(restored._current_array.dtype, np.dtype("int16"))
                self.assertEqual(restored._current_array.shape, (40, 50, 60))
                self.assertTrue(np.array_equal(restored._current_array, array))

                first = restored.compute({"array": np.zeros((1,), dtype=np.int16)})
                self.assertEqual(restored._current_array.shape, (40, 50, 60))
                second = restored.compute({"array": np.zeros((1,), dtype=np.int16)})
                self.assertEqual(first["array"].shape, (2000, 60))
                self.assertEqual(second["array"].shape, (1,))
        finally:
            set_project_file_path(None)
            root.destroy()

    def test_empty_array_viewer_does_not_reference_missing_sidecar(self):
        root = tk.Tk()
        root.withdraw()
        canvas = tk.Canvas(root)
        node = ArrayViewerNode("empty", canvas)
        node.build_body()

        try:
            with tempfile.TemporaryDirectory() as directory:
                set_project_file_path(str(Path(directory) / "empty.xlsx"))
                params = node.get_params()
            self.assertNotIn("data", params)
            self.assertNotIn("array_viewer_file", params)
        finally:
            set_project_file_path(None)
            root.destroy()
