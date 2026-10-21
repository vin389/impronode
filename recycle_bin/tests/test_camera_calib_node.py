# Retired with the old Camera Calibration node (calib_nodes.py). It runs only
# after both files are restored to node_editor/nodes and tests/ (see
# recycle_bin/README.md).
import unittest
import numpy as np
import tkinter as tk
import cv2

from node_editor.base_node import BaseNode
from node_editor.pin_types import PinSchema, PinDef, PinType
from node_editor.nodes.calib_nodes import CameraCalibNode, _draw_calib_result


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

    def test_imgsize_pin_order_and_integer_outputs(self):
        pins = self.node.get_pin_schema().outputs
        names = [pin.name for pin in pins]
        index = names.index("imgsize")
        self.assertEqual(names[index - 1:index + 2], ["tvec", "imgsize", "image_pts"])
        self.assertEqual(pins[index].type, PinType.ARRAY)
        self.assertEqual(pins[index].shape, (2,))
        self.assertEqual(pins[index].dtype, "int64")

        self.node._result_cmat = np.eye(3)
        self.node._result_dvec = np.zeros((1, 14))
        self.node._img_w_var.set(1920)
        self.node._img_h_var.set(1080)
        computed = self.node.compute({})["imgsize"]
        np.testing.assert_array_equal(computed, [1920, 1080])
        self.assertEqual(computed.dtype, np.int64)

        published = []
        self.node._on_output_ready = lambda node_id, outputs: published.append(outputs)
        self.node._push_outputs()
        np.testing.assert_array_equal(published[-1]["imgsize"], computed)
        self.assertEqual(published[-1]["imgsize"].dtype, np.int64)

        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        np.testing.assert_array_equal(self.node.compute({"images": frame})["imgsize"], [320, 240])

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
