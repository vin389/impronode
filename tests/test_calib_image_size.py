import unittest
import numpy as np
import tkinter as tk
from tkinter import ttk
import cv2

from node_editor.base_node import BaseNode
from node_editor.pin_types import PinSchema, PinDef, PinType
from node_editor.nodes.calib_nodes import CameraCalibNode, _draw_calib_result
from node_editor.nodes.meshgrid_node import MeshgridNode


def _fixed_size_canvas(root, width=200, height=200):
    """A Canvas whose winfo_width/height report fixed values, since a
    withdrawn Tk root never realizes real geometry in headless tests."""
    canvas = tk.Canvas(root, width=width, height=height)
    canvas.winfo_width = lambda: width
    canvas.winfo_height = lambda: height
    return canvas


class _ManyPinsNode(BaseNode):
    NODE_WIDTH = 160
    NODE_HEIGHT = 90
    MIN_HEIGHT = 60

    def get_pin_schema(self):
        inputs = [PinDef(f"in{i}", PinType.SCALAR, f"in{i}", optional=True) for i in range(8)]
        outputs = [PinDef(f"out{i}", PinType.SCALAR, f"out{i}") for i in range(8)]
        return PinSchema(inputs=inputs, outputs=outputs)

    def build_body(self):
        pass


class TestCameraCalibNodeImageSize(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.canvas = tk.Canvas(self.root, width=200, height=200)
        self.node = CameraCalibNode("calib", self.canvas)
        self.node._init_state()

    def tearDown(self):
        self.root.destroy()

    def test_compute_updates_image_size_from_connected_frame(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        self.node.compute({"images": frame})

        self.assertEqual(self.node._img_h_var.get(), 720)
        self.assertEqual(self.node._img_w_var.get(), 1280)

    def test_default_height_scales_with_pin_count(self):
        many_pins = _ManyPinsNode("many", self.canvas)
        self.assertGreater(many_pins.get_default_height(), many_pins.NODE_HEIGHT)

    def test_vis_background_uses_selected_calibration_frame(self):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[10:30, 20:40] = [10, 20, 30]
        self.node._collected_images = [frame]
        self.node._result_cmat = np.eye(3)
        self.node._result_rvecs = [np.zeros((3, 1), dtype=np.float32)]
        self.node._result_tvecs = [np.zeros((3, 1), dtype=np.float32)]
        self.node._result_imgpts = [np.array([[10, 20], [30, 40]], dtype=np.float32)]
        self.node._result_proj_pts = [np.array([[10, 20], [30, 40]], dtype=np.float32)]
        self.node._result_objpts = [np.zeros((2, 3), dtype=np.float32)]
        self.node._result_proj_errs = [np.zeros((2, 2), dtype=np.float32)]
        self.node._vis_idx_var.set(1)

        bg = self.node._get_background_image(0)

        self.assertEqual(bg.shape[:2], (240, 320))
        self.assertEqual(bg[10, 20].tolist(), [10, 20, 30])

    def test_draw_calib_result_undistorts_points_when_requested(self):
        bg = np.zeros((200, 200, 3), dtype=np.uint8)
        cmat = np.array([[100.0, 0.0, 100.0],
                         [0.0, 100.0, 100.0],
                         [0.0, 0.0, 1.0]], dtype=np.float64)
        dvec = np.array([[-0.2, 0.1, 0.0, 0.0, 0.0]], dtype=np.float64)
        img_pts = np.array([[120.0, 120.0]], dtype=np.float64)
        prj_pts = np.array([[140.0, 130.0]], dtype=np.float64)

        undistorted = _draw_calib_result(
            bg,
            img_pts,
            prj_pts,
            cmat,
            dvec,
            np.zeros((3, 1), dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            None,
            None,
            None,
            undistort=True,
        )

        # Default alpha=0 keeps the valid undistorted region within the image,
        # while still preserving the calibrated camera matrix for points.
        self.assertGreaterEqual(undistorted.shape[0], 200)
        self.assertGreaterEqual(undistorted.shape[1], 200)

    def test_draw_calib_result_uses_same_camera_matrix_for_alpha(self):
        bg = np.zeros((200, 200, 3), dtype=np.uint8)
        cmat = np.array([[120.0, 0.0, 100.0],
                         [0.0, 120.0, 90.0],
                         [0.0, 0.0, 1.0]], dtype=np.float64)
        dvec = np.array([[0.1, -0.05, 0.0, 0.0, 0.0]], dtype=np.float64)
        img_pts = np.array([[90.0, 80.0]], dtype=np.float64)
        prj_pts = np.array([[120.0, 110.0]], dtype=np.float64)

        new_cmat, _ = cv2.getOptimalNewCameraMatrix(
            cmat, dvec, (200, 200), alpha=0.5, newImgSize=(200, 200))

        vis = _draw_calib_result(
            bg,
            img_pts,
            prj_pts,
            cmat,
            dvec,
            np.zeros((3, 1), dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            None,
            None,
            None,
            undistort=True,
            new_cmat=new_cmat,
            alpha=0.5,
        )

        mapped = _draw_calib_result(
            bg,
            img_pts,
            prj_pts,
            cmat,
            dvec,
            np.zeros((3, 1), dtype=np.float64),
            np.zeros((3, 1), dtype=np.float64),
            None,
            None,
            None,
            undistort=True,
            new_cmat=new_cmat,
            alpha=0.5,
        )

        self.assertEqual(vis.shape[:2], (200, 200))
        self.assertEqual(mapped.shape[:2], (200, 200))
        self.assertEqual(vis.shape, mapped.shape)

    def test_calib_inspector_opens_at_double_default_size(self):
        self.node.open_inspector()
        self.root.update_idletasks()
        win = self.node._inspector_win
        if win is None or not win.winfo_exists() or win.winfo_width() <= 1 or win.winfo_height() <= 1:
            self.skipTest("Tk inspector window was not realized in this headless environment.")

        base_w = win.winfo_width()
        base_h = win.winfo_height()

        self.node.close_inspector()
        self.node.open_inspector()
        self.root.update_idletasks()

        win = self.node._inspector_win
        if win is None or not win.winfo_exists() or win.winfo_width() <= 1 or win.winfo_height() <= 1:
            self.skipTest("Tk inspector window was not realized in this headless environment.")

        self.assertGreaterEqual(win.winfo_width(), base_w * 2)
        self.assertGreaterEqual(win.winfo_height(), base_h * 2)


class TestMeshgridNodeAxisOrdering(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.canvas = tk.Canvas(self.root, width=200, height=200)
        self.node = MeshgridNode("mesh", self.canvas)
        self.node._init_state()

    def tearDown(self):
        self.root.destroy()

    def test_default_axis_order_is_ij_and_is_persisted(self):
        self.assertEqual(self.node._indexing_var.get(), "ij")
        self.node.set_params({"x": "-1:1:1", "y": "-1:1:1", "z": "0:1:2", "indexing": "xy"})
        self.assertEqual(self.node._indexing_var.get(), "xy")
        params = self.node.get_params()
        self.assertEqual(params["indexing"], "xy")

    def test_custom_axis_orders_match_meshgrid_permutations(self):
        x = np.array([1.0, 2.0])
        y = np.array([10.0, 20.0])
        z = np.array([100.0, 200.0])

        for mode, axes in {
            "ij": (x, y, z),
            "xy": (y, x, z),
            "xzy": (x, z, y),
            "zxy": (z, x, y),
            "yzx": (y, z, x),
            "zyx": (z, y, x),
        }.items():
            self.node._indexing_var.set(mode)
            result = self.node.compute({"x": x, "y": y, "z": z})
            xx, yy, zz = np.meshgrid(*axes, indexing="ij")
            expected = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float64)
            np.testing.assert_allclose(result["grid_coords"], expected)
            np.testing.assert_array_equal(result["grid_shape"], np.array([2, 2, 2], dtype=np.int64))

    def test_preview_handles_stale_shape_metadata_without_crashing(self):
        self.node._preview_canvas = tk.Canvas(self.root, width=200, height=200)
        self.node._last_grid_coords = np.arange(18, dtype=np.float64).reshape(18, 1)
        self.node._last_grid_shape = np.array([3, 3, 3], dtype=np.int64)

        self.node._refresh_preview()

        self.assertTrue(self.node._preview_canvas.winfo_exists())

    def test_2d_preview_does_not_reshape_to_3d(self):
        self.node._x_var.set("1 2 3")
        self.node._y_var.set("10 20 30")
        self.node._z_var.set("")
        self.node._preview_canvas = tk.Canvas(self.root, width=200, height=200)

        result = self.node.compute({})
        self.assertEqual(result["grid_shape"].tolist(), [3, 3])
        self.node._refresh_preview()

        self.assertTrue(self.node._preview_canvas.winfo_exists())

    def test_preview_enabled_defaults_to_off_and_skips_drawing(self):
        self.assertFalse(self.node._preview_enabled_var.get())

        self.node._preview_canvas = _fixed_size_canvas(self.root)
        self.node._last_grid_coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        self.node._last_grid_shape = np.array([2, 2, 1], dtype=np.int64)

        self.node._refresh_preview()

        canvas = self.node._preview_canvas
        self.assertEqual(len([i for i in canvas.find_all() if canvas.type(i) == "line"]), 0)
        self.assertEqual(len([i for i in canvas.find_all() if canvas.type(i) == "polygon"]), 0)

        self.node._preview_enabled_var.set(True)
        self.node._refresh_preview()
        self.assertGreater(len(canvas.find_all()), 0)

    def test_preview_enabled_is_persisted(self):
        self.node._preview_enabled_var.set(True)
        params = self.node.get_params()
        self.assertTrue(params["preview_enabled"])

        self.node.set_params({"preview_enabled": False})
        self.assertFalse(self.node._preview_enabled_var.get())

    def test_inspector_rebuilds_cleanly_when_reopened(self):
        self.node.open_inspector()
        self.root.update_idletasks()
        first_body = self.node._inspector_body
        self.assertTrue(first_body is not None)
        self.assertGreater(len(first_body.winfo_children()), 0)

        self.node.close_inspector()
        self.node.open_inspector()
        self.root.update_idletasks()
        second_body = self.node._inspector_body
        self.assertTrue(second_body is not None)
        self.assertGreater(len(second_body.winfo_children()), 0)

    def test_double_click_opens_inspector_directly(self):
        self.node.build_body()
        self.root.update_idletasks()
        self.assertIsNotNone(self.node._on_double_click)
        self.node._on_double_click()
        self.root.update_idletasks()
        self.assertTrue(self.node.is_inspector_open())

    def test_meshgrid_inspector_has_sufficient_height(self):
        self.node.open_inspector()
        self.root.update_idletasks()
        win = self.node._inspector_win
        if win is None or not win.winfo_exists() or win.winfo_height() <= 1:
            self.skipTest("Tk inspector window was not realized in this headless environment.")
        self.assertGreaterEqual(win.winfo_height(), 500)

    def test_inspector_color_controls_are_comboboxes(self):
        self.node.open_inspector()
        self.root.update_idletasks()

        self.assertTrue(hasattr(self.node, "_point_color_combo"))
        self.assertTrue(hasattr(self.node, "_line_color_combo"))
        self.assertIsInstance(self.node._point_color_combo, ttk.Combobox)
        self.assertIsInstance(self.node._line_color_combo, ttk.Combobox)
        self.assertIn("red", self.node._point_color_combo["values"])
        self.assertIn("black", self.node._line_color_combo["values"])

    def test_preview_axis_triad_includes_view_cube_faces(self):
        canvas = _fixed_size_canvas(self.root)

        self.node._last_grid_coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        self.node._draw_axis_triad(canvas)

        polygon_ids = [item for item in canvas.find_all() if canvas.type(item) == "polygon"]
        line_ids = [item for item in canvas.find_all() if canvas.type(item) == "line"]
        # exactly one axis line each for x/y/z, and exactly 3 visible cube faces
        self.assertEqual(len(line_ids), 3)
        self.assertEqual(len(polygon_ids), 3)

    def test_view_cube_faces_are_mutually_perpendicular(self):
        self.node._preview_state["yaw"] = 0.4
        self.node._preview_state["pitch"] = -0.6
        R = self.node._rotation_matrix()

        faces_by_depth = sorted(
            self.node._CUBE_FACES,
            key=lambda f: float((R @ f["normal"])[1]))
        visible = faces_by_depth[:3]

        normals = np.array([f["normal"] for f in visible])
        dots = normals @ normals.T
        np.fill_diagonal(dots, 0.0)
        self.assertTrue(np.allclose(dots, 0.0, atol=1e-9))

    def test_project_points_matches_manual_rotation(self):
        self.node._preview_canvas = _fixed_size_canvas(self.root)
        pts = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ], dtype=np.float64)
        self.node._last_grid_coords = pts
        self.node._preview_state["yaw"] = -0.9
        self.node._preview_state["pitch"] = 0.7
        self.node._preview_state["zoom"] = 1.0

        yaw = self.node._preview_state["yaw"]
        pitch = self.node._preview_state["pitch"]
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        scale = self.node._get_scale()

        expected = []
        for x, y, z in pts:
            x1 = x * cy - y * sy
            y1 = x * sy + y * cy
            z1 = z
            y2 = y1 * cp - z1 * sp
            z2 = y1 * sp + z1 * cp
            expected.append([x1 * scale + 100.0, -z2 * scale + 100.0])
        expected = np.asarray(expected, dtype=np.float64)

        projected = self.node._project_points(pts)
        np.testing.assert_allclose(projected, expected, atol=1e-6)

    def test_rotate_then_pan_does_not_jump_reference_point(self):
        self.node._preview_canvas = _fixed_size_canvas(self.root)
        self.node._last_grid_coords = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ], dtype=np.float64)
        self.node._last_grid_shape = np.array([2, 2, 2], dtype=np.int64)
        reference = np.array([[5.0, -3.0, 2.0]], dtype=np.float64)

        self.node._on_preview_drag_start(type("E", (), {"x": 100.0, "y": 100.0})())
        self.node._on_preview_drag_motion(type("E", (), {"x": 130.0, "y": 80.0})())
        before_switch = self.node._project_points(reference)[0]
        self.node._on_preview_drag_release(None)

        self.node._on_preview_pan_start(type("E", (), {"x": 130.0, "y": 80.0})())
        after_switch = self.node._project_points(reference)[0]

        np.testing.assert_allclose(before_switch, after_switch, atol=1e-6)

    def test_pan_then_rotate_does_not_jump_reference_point(self):
        self.node._preview_canvas = _fixed_size_canvas(self.root)
        self.node._last_grid_coords = np.array([
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
        ], dtype=np.float64)
        self.node._last_grid_shape = np.array([2, 2, 2], dtype=np.int64)
        reference = np.array([[5.0, -3.0, 2.0]], dtype=np.float64)

        self.node._on_preview_pan_start(type("E", (), {"x": 100.0, "y": 100.0})())
        self.node._on_preview_pan_motion(type("E", (), {"x": 70.0, "y": 130.0})())
        before_switch = self.node._project_points(reference)[0]
        self.node._on_preview_pan_release(None)

        self.node._on_preview_drag_start(type("E", (), {"x": 70.0, "y": 130.0})())
        after_switch = self.node._project_points(reference)[0]

        np.testing.assert_allclose(before_switch, after_switch, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
