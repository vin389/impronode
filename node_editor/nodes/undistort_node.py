import tkinter as tk

import cv2
import numpy as np

from node_editor.base_node import BaseNode
from node_editor.pin_types import PinSchema, PinDef, PinType
from node_editor.execution import ExecutionMode


class UndistortNode(BaseNode):
	"""
	OpenCV undistort node.

	Mapping to cv2.undistort:
	  undistort(src, cameraMatrix, distCoeffs[, dst[, newCameraMatrix]])

	Defaults when optional input pin is disconnected:
	  - dst: None (OpenCV allocates output)
	  - newCameraMatrix: cameraMatrix
	"""

	EXECUTION_MODE = ExecutionMode.SYNC
	NODE_TYPE = "undistort"
	DISPLAY_NAME = "Undistort img."
	CATEGORY = "process"
	NODE_WIDTH = 210
	NODE_HEIGHT = 105

	HELP_TEXT = (
		"Undistort Node\n\n"
		"Purpose:\n"
		"- Apply cv2.undistort to remove lens distortion from an image.\n\n"
		"Inputs:\n"
		"- src (required): input image array (H,W) or (H,W,C).\n"
		"- cameraMatrix / K (required): 3x3 intrinsic matrix.\n"
		"- distCoeffs / d (required): distortion coefficients.\n"
		"- dst (optional, default None): destination image buffer.\n"
		"  If not connected, OpenCV allocates output automatically.\n"
		"- newCameraMatrix / newK (optional, default K): camera matrix used during undistortion.\n"
		"  If not connected, this node uses cameraMatrix.\n\n"
		"Output:\n"
		"- image: undistorted output image.\n"
	)

	def get_pin_schema(self) -> PinSchema:
		return PinSchema(
			inputs=[
				PinDef("src", PinType.IMAGE, "src"),
				PinDef("cameraMatrix", PinType.ARRAY, "K"),
				PinDef("distCoeffs", PinType.ARRAY, "d"),
				PinDef("dst", PinType.IMAGE, "dst", optional=True),
				PinDef("newCameraMatrix", PinType.ARRAY, "newK", optional=True),
			],
			outputs=[
				PinDef("image", PinType.IMAGE, "image"),
			],
		)

	def build_body(self) -> None:
		x, y, w, h = self.x, self.y, self.width, self.height

		self._body_rect = self.canvas.create_rectangle(
			x, y, x + w, y + h,
			fill="#bddbd6", outline="#6ba4d9", width=2,
			tags=(self.node_id, "node_body"),
		)
		self._title_item = self.canvas.create_text(
			x + w / 2, y + 13,
			text=self.DISPLAY_NAME,
			font=("Arial", 9, "bold"), fill="#2e5d93",
			tags=(self.node_id,),
		)

		hint = tk.Label(
			self.canvas,
			text="src,K,d -> undistort",
			font=("Arial", 8), bg="#ecf4ff", fg="#3e678f",
		)
		self.canvas.create_window(
			x + w / 2, y + 38, window=hint,
			tags=(self.node_id,),
		)

		self._status_var = tk.StringVar(value="waiting for src,K,d")
		status_lbl = tk.Label(
			self.canvas,
			textvariable=self._status_var,
			font=("Arial", 7), bg="#ecf4ff", fg="#3e678f",
		)
		self.canvas.create_window(
			x + w / 2, y + h - 12, window=status_lbl,
			tags=(self.node_id,),
		)

		self._canvas_items += [self._body_rect, self._title_item]

	@staticmethod
	def _coerce_camera_matrix(value) -> np.ndarray:
		arr = np.asarray(value, dtype=np.float64)
		if arr.size < 9:
			raise ValueError("cameraMatrix needs at least 9 values")
		return arr.reshape(-1)[:9].reshape(3, 3)

	@staticmethod
	def _coerce_dist_coeffs(value) -> np.ndarray:
		arr = np.asarray(value, dtype=np.float64).reshape(-1)
		if arr.size == 0:
			raise ValueError("distCoeffs is empty")
		return arr

	def get_help_text(self) -> str:
		return self.HELP_TEXT

	def compute(self, inputs: dict) -> dict:
		src = inputs.get("src")
		raw_k = inputs.get("cameraMatrix")
		raw_d = inputs.get("distCoeffs")

		if src is None or raw_k is None or raw_d is None:
			self._status_var.set("missing src,K,d")
			self.set_status("missing", "#cc6666")
			return {}

		try:
			image = np.asarray(src)
			if image.ndim not in (2, 3):
				raise ValueError("src must be 2D or 3D image array")

			camera_matrix = self._coerce_camera_matrix(raw_k)
			dist_coeffs = self._coerce_dist_coeffs(raw_d)

			raw_dst = inputs.get("dst")
			dst = None if raw_dst is None else np.asarray(raw_dst)

			raw_new_k = inputs.get("newCameraMatrix")
			if raw_new_k is None:
				# OpenCV default newCameraMatrix is cameraMatrix.
				new_camera_matrix = camera_matrix
			else:
				new_camera_matrix = self._coerce_camera_matrix(raw_new_k)

			undistorted = cv2.undistort(
				image,
				camera_matrix,
				dist_coeffs,
				dst=dst,
				newCameraMatrix=new_camera_matrix,
			)

			self._status_var.set("ok")
			self.set_status("ok", "#55aa55")
			return {
				"image": undistorted,
			}
		except Exception as e:
			self._status_var.set(f"error: {e}")
			self.set_status("error", "#cc0000")
			return {}


class GetOptimalNewCameraMatrixNode(BaseNode):
	"""
	OpenCV getOptimalNewCameraMatrix node.

	Mapping to cv2.getOptimalNewCameraMatrix:
	  getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imageSize,
	                            alpha[, newImgSize[, centerPrincipalPoint]])

	Defaults when optional input pin is disconnected:
	  - alpha: 0.0
	  - newImgSize: imageSize
	  - centerPrincipalPoint: False
	  - imageSize: inferred from src shape if src is connected
	"""

	EXECUTION_MODE = ExecutionMode.SYNC
	NODE_TYPE = "get_optimal_new_camera_matrix"
	DISPLAY_NAME = "GetOptimalNewCameraMatrix"
	CATEGORY = "process"
	NODE_WIDTH = 280
	NODE_HEIGHT = 130

	HELP_TEXT = (
		"GetOptimalNewCameraMatrix Node\n\n"
		"Purpose:\n"
		"- Compute an adjusted camera matrix for undistortion and a valid ROI.\n\n"
		"Inputs:\n"
		"- cameraMatrix (required): 3x3 intrinsic matrix K.\n"
		"- distCoeffs (required): distortion coefficients array.\n"
		"- imageSize (optional): [width, height]. If missing, inferred from src.\n"
		"- src (optional): image used only to infer imageSize when imageSize pin is not connected.\n"
		"- alpha (optional, default 0.0): free scaling parameter in [0, 1].\n"
		"  alpha=0 crops to valid pixels, alpha=1 keeps all source pixels.\n"
		"- newImgSize (optional, default imageSize): output image size [width, height].\n"
		"- centerPrincipalPoint (optional, default False): whether to center principal point.\n\n"
		"Outputs:\n"
		"- newCameraMatrix: optimized 3x3 intrinsic matrix.\n"
		"- validPixROI: [x, y, w, h] rectangle of valid pixels.\n"
	)

	def get_pin_schema(self) -> PinSchema:
		return PinSchema(
			inputs=[
				PinDef("cameraMatrix", PinType.ARRAY, "K"),
				PinDef("distCoeffs", PinType.ARRAY, "d"),
				PinDef("imageSize", PinType.ARRAY, "imgSize", optional=True),
				PinDef("src", PinType.IMAGE, "src", optional=True),
				PinDef("alpha", PinType.SCALAR, "alpha", optional=True),
				PinDef("newImgSize", PinType.ARRAY, "newSize", optional=True),
				PinDef("centerPrincipalPoint", PinType.SCALAR, "centerPP", optional=True),
			],
			outputs=[
				PinDef("newCameraMatrix", PinType.ARRAY, "newK", shape=(3, 3), dtype="float64"),
				PinDef("validPixROI", PinType.ARRAY, "roi", shape=(4,), dtype="int32"),
			],
		)

	def build_body(self) -> None:
		x, y, w, h = self.x, self.y, self.width, self.height

		self._body_rect = self.canvas.create_rectangle(
			x, y, x + w, y + h,
			fill="#edf5ff", outline="#8ab4e8", width=2,
			tags=(self.node_id, "node_body"),
		)
		self._title_item = self.canvas.create_text(
			x + w / 2, y + 13,
			text=self.DISPLAY_NAME,
			font=("Arial", 9, "bold"), fill="#315f93",
			tags=(self.node_id,),
		)

		hint = tk.Label(
			self.canvas,
			text="K,d,(size/src),alpha->newK,roi",
			font=("Arial", 8), bg="#edf5ff", fg="#3f678f",
		)
		self.canvas.create_window(
			x + w / 2, y + 42, window=hint,
			tags=(self.node_id,),
		)

		self._status_var = tk.StringVar(value="waiting for K,d")
		status_lbl = tk.Label(
			self.canvas,
			textvariable=self._status_var,
			font=("Arial", 7), bg="#edf5ff", fg="#3f678f",
		)
		self.canvas.create_window(
			x + w / 2, y + h - 12, window=status_lbl,
			tags=(self.node_id,),
		)

		self._canvas_items += [self._body_rect, self._title_item]

	def get_help_text(self) -> str:
		return self.HELP_TEXT

	@staticmethod
	def _coerce_camera_matrix(value) -> np.ndarray:
		arr = np.asarray(value, dtype=np.float64)
		if arr.size < 9:
			raise ValueError("cameraMatrix needs at least 9 values")
		return arr.reshape(-1)[:9].reshape(3, 3)

	@staticmethod
	def _coerce_dist_coeffs(value) -> np.ndarray:
		arr = np.asarray(value, dtype=np.float64).reshape(-1)
		if arr.size == 0:
			raise ValueError("distCoeffs is empty")
		return arr

	@staticmethod
	def _coerce_size2(value, name: str) -> tuple[int, int]:
		arr = np.asarray(value).reshape(-1)
		if arr.size < 2:
			raise ValueError(f"{name} needs at least 2 values: [width, height]")
		w = int(round(float(arr[0])))
		h = int(round(float(arr[1])))
		if w <= 0 or h <= 0:
			raise ValueError(f"{name} must be positive")
		return (w, h)

	@staticmethod
	def _coerce_alpha(value) -> float:
		alpha = float(value)
		if alpha < 0.0:
			alpha = 0.0
		if alpha > 1.0:
			alpha = 1.0
		return alpha

	@staticmethod
	def _coerce_bool_like(value) -> bool:
		if isinstance(value, str):
			text = value.strip().lower()
			if text in ("1", "true", "yes", "y", "on"):
				return True
			if text in ("0", "false", "no", "n", "off", ""):
				return False
		try:
			return bool(int(float(value)))
		except Exception:
			return bool(value)

	def compute(self, inputs: dict) -> dict:
		raw_k = inputs.get("cameraMatrix")
		raw_d = inputs.get("distCoeffs")

		if raw_k is None or raw_d is None:
			self._status_var.set("missing K or d")
			self.set_status("missing", "#cc6666")
			return {}

		try:
			camera_matrix = self._coerce_camera_matrix(raw_k)
			dist_coeffs = self._coerce_dist_coeffs(raw_d)

			raw_img_size = inputs.get("imageSize")
			raw_src = inputs.get("src")
			if raw_img_size is None:
				if raw_src is None:
					raise ValueError("imageSize is required unless src is connected")
				src = np.asarray(raw_src)
				if src.ndim < 2:
					raise ValueError("src must be a 2D/3D image to infer imageSize")
				image_size = (int(src.shape[1]), int(src.shape[0]))
			else:
				image_size = self._coerce_size2(raw_img_size, "imageSize")

			raw_alpha = inputs.get("alpha")
			alpha = 0.0 if raw_alpha is None else self._coerce_alpha(raw_alpha)

			raw_new_size = inputs.get("newImgSize")
			new_img_size = image_size if raw_new_size is None else self._coerce_size2(raw_new_size, "newImgSize")

			raw_center_pp = inputs.get("centerPrincipalPoint")
			center_principal_point = False if raw_center_pp is None else self._coerce_bool_like(raw_center_pp)

			new_k, roi = cv2.getOptimalNewCameraMatrix(
				camera_matrix,
				dist_coeffs,
				image_size,
				alpha,
				newImgSize=new_img_size,
				centerPrincipalPoint=center_principal_point,
			)

			roi_arr = np.asarray(roi, dtype=np.int32).reshape(4)

			self._status_var.set(
				f"ok: size={image_size[0]}x{image_size[1]} alpha={alpha:.2f}"
			)
			self.set_status("ok", "#55aa55")
			return {
				"newCameraMatrix": np.asarray(new_k, dtype=np.float64),
				"validPixROI": roi_arr,
			}
		except Exception as e:
			self._status_var.set(f"error: {e}")
			self.set_status("error", "#cc0000")
			return {}
