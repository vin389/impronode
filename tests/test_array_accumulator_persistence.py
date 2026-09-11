import csv
import io
import json
from pathlib import Path
import tempfile
import tkinter as tk
import unittest

import numpy as np

from node_editor.nodes.accumulator_nodes import ArrayAccumulatorNode
from node_editor.project_context import set_project_file_path


class TestArrayAccumulatorPersistence(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.canvas = tk.Canvas(self.root)

    def tearDown(self):
        set_project_file_path(None)
        self.root.destroy()

    def _make_node(self, node_id):
        node = ArrayAccumulatorNode(node_id, self.canvas)
        node.build_body()
        return node

    def test_parse_exported_csv_preserves_explicit_row_numbers(self):
        source = io.StringIO(
            "row,x,y\n"
            "1,1.25,2.5\n"
            "3,nan,4.75\n"
        )

        rows, max_row, n_cols, names = ArrayAccumulatorNode._parse_csv_rows(
            csv.reader(source))

        self.assertEqual(set(rows), {1, 3})
        self.assertEqual(max_row, 3)
        self.assertEqual(n_cols, 2)
        self.assertEqual(names, ["x", "y"])
        self.assertTrue(np.allclose(rows[1], [1.25, 2.5]))
        self.assertTrue(np.isnan(rows[3][0]))

    def test_project_params_restore_data_and_gap_rows(self):
        node = self._make_node("source")
        node._col_names_var.set("x, y")
        node._restore_table(
            {
                1: np.array([1.25, 2.5]),
                3: np.array([3.75, np.nan]),
            },
            max_row=3,
            n_cols=2,
        )
        node._fire_counter = 7

        params = node.get_params()
        json.dumps(params)

        restored = self._make_node("restored")
        restored.set_params(params)
        table = restored._build_table()

        self.assertEqual(set(restored._rows), {1, 3})
        self.assertEqual(restored._fire_counter, 7)
        self.assertEqual(restored._col_names_var.get(), "x, y")
        self.assertEqual(table.shape, (3, 2))
        self.assertTrue(np.isnan(table[1]).all())
        self.assertTrue(np.allclose(table[0], [1.25, 2.5]))
        self.assertTrue(np.isnan(table[2, 1]))

    def test_invalid_csv_is_rejected_before_replacing_data(self):
        with self.assertRaisesRegex(ValueError, "inconsistent|expected"):
            ArrayAccumulatorNode._parse_csv_rows(csv.reader(io.StringIO(
                "row,x,y\n"
                "1,1,2\n"
                "2,3\n"
            )))

    def test_first_compute_after_project_restore_does_not_append(self):
        source = self._make_node("source")
        source._restore_table({1: np.array([1.0, 2.0])}, max_row=1, n_cols=2)

        restored = self._make_node("restored")
        restored.set_params(source.get_params())

        first = restored.compute({"data": np.array([3.0, 4.0])})
        second = restored.compute({"data": np.array([3.0, 4.0])})

        self.assertEqual(first["table"].shape, (1, 2))
        self.assertEqual(second["table"].shape, (2, 2))
        self.assertTrue(np.allclose(second["table"][1], [3.0, 4.0]))

    def test_project_uses_relative_sidecar_csv_instead_of_inline_array(self):
        node = self._make_node("node_42")
        data = np.arange(10_000, dtype=np.float64).reshape(100, 100) / 7.0
        node._restore_table(
            {row + 1: data[row] for row in range(data.shape[0])},
            max_row=data.shape[0],
            n_cols=data.shape[1],
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            project_path = Path(temporary_directory) / "experiment.xlsx"
            set_project_file_path(str(project_path))
            params = node.get_params()

            self.assertNotIn("rows", params)
            self.assertEqual(
                params["accumulator_csv"],
                "experiment.node_42.array_accumulator.csv",
            )
            self.assertLess(len(json.dumps(params)), 32_000)

            sidecar_path = project_path.parent / params["accumulator_csv"]
            self.assertTrue(sidecar_path.exists())
            self.assertGreater(sidecar_path.stat().st_size, 32_000)

            restored = self._make_node("node_42_restored")
            restored.set_params(params)
            self.assertTrue(np.array_equal(restored._build_table(), data))

    def test_empty_project_accumulator_does_not_reference_missing_sidecar(self):
        node = self._make_node("empty")
        with tempfile.TemporaryDirectory() as temporary_directory:
            set_project_file_path(str(Path(temporary_directory) / "empty.xlsx"))
            params = node.get_params()

        self.assertNotIn("rows", params)
        self.assertNotIn("accumulator_csv", params)


if __name__ == "__main__":
    unittest.main()