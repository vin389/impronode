import unittest
import numpy as np
import tkinter as tk

from node_editor.base_node import BaseNode
from node_editor.pin_types import PinSchema, PinDef, PinType
from node_editor.nodes.calib_nodes import CameraCalibNode, _draw_calib_result


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

        self.assertEqual(undistorted.shape[:2], (200, 200))


if __name__ == "__main__":
    unittest.main()
