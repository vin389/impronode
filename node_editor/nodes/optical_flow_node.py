import tkinter as tk
from tkinter import messagebox
import time

import cv2
import numpy as np

from node_editor.base_node import BaseNode
from node_editor.execution import ExecutionMode
from node_editor.pin_types import PinDef, PinSchema, PinType


class OpticalFlowNode(BaseNode):
	"""
	OpenCV Lucas-Kanade optical flow node.

	Wraps cv2.calcOpticalFlowPyrLK and adds a one-shot enable pin so
	upstream pin changes can be buffered without recomputing immediately.
	"""

	EXECUTION_MODE = ExecutionMode.BACKGROUND
	NODE_TYPE = "optical_flow"
	DISPLAY_NAME = "Optical Flow"
	CATEGORY = "process"
	SEARCH_KEYWORDS = ("optical", "flow", "of", "lucas pyr", "lucas", "pyr", "lk")
	NODE_WIDTH = 250
	NODE_HEIGHT = 120

	_DEFAULT_CRITERIA = (
		cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
		30,
		0.01,
	)
	_FLOW_INPUT_PINS = (
		"prevImg",
		"nextImg",
		"prevPts",
		"nextPts",
		"winSize",
		"maxLevel",
		"criteria",
		"flags",
		"minEigThreshold",
	)

	HELP_TEXT = (
		"Optical Flow Node\n\n"
		"Purpose:\n"
		"- Run cv2.calcOpticalFlowPyrLK on demand.\n\n"
		"Input Pins:\n"
		"- prevImg [IMAGE]: previous frame (required).\n"
		"- nextImg [IMAGE]: current/next frame (required).\n"
		"- prevPts [ARRAY float32 Nx2]: points to track (required).\n"
		"- nextPts [ARRAY float32 Nx2, optional]: initial guess points.\n"
		"- winSize [ARRAY int32 size-2, optional]: LK search window, default [21, 21].\n"
		"- maxLevel [SCALAR, optional]: pyramid max level, default 3.\n"
		"- criteria [ARRAY size-3, optional]: [type, maxCount, epsilon],\n"
		"  default [EPS|COUNT, 30, 0.01].\n"
		"- flags [SCALAR, optional]: cv2 calcOpticalFlowPyrLK flags, default 0.\n"
		"- minEigThreshold [SCALAR, optional]: default 1e-4.\n"
		"- trig [TRIGGER, optional]: while false, inputs are buffered\n"
		"  without computing; on each truthy pulse, compute exactly once.\n\n"
		"Output Pins:\n"
		"- nextPts [ARRAY float32 Nx2]: tracked output points.\n"
		"- status [ARRAY uint8 N]: 1 means tracked successfully, 0 means failed.\n"
		"- err [ARRAY float32 N]: per-point tracking error from OpenCV.\n"
	)

	def _init_state(self) -> None:
		if hasattr(self, "_status_var"):
			return
		self._status_var = tk.StringVar(value="frozen")
		self._buffered_inputs: dict[str, object] = {}
		self._enable_latched_high = False
		self._trigger_pending = False
		self._suppress_next_low_status = False
		self._last_warn_key: str | None = None

	def get_pin_schema(self) -> PinSchema:
		return PinSchema(
			inputs=[
				PinDef("prevImg", PinType.IMAGE, "prev"),
				PinDef("nextImg", PinType.IMAGE, "next"),
				PinDef("prevPts", PinType.ARRAY, "prevPts", shape=(-1, 2), dtype="float32"),
				PinDef("nextPts", PinType.ARRAY, "nextPts", optional=True, shape=(-1, 2), dtype="float32"),
				PinDef("winSize", PinType.ARRAY, "win", optional=True, shape=(2,), dtype="int32"),
				PinDef("maxLevel", PinType.SCALAR, "lvl", optional=True),
				PinDef("criteria", PinType.ARRAY, "crit", optional=True, shape=(3,)),
				PinDef("flags", PinType.SCALAR, "flags", optional=True),
				PinDef("minEigThreshold", PinType.SCALAR, "eig", optional=True),
				PinDef("trig", PinType.TRIGGER, "trig", optional=True),
			],
			outputs=[
				PinDef("nextPts", PinType.ARRAY, "nextPts", shape=(-1, 2), dtype="float32"),
				PinDef("status", PinType.ARRAY, "status", shape=(-1,), dtype="uint8"),
				PinDef("err", PinType.ARRAY, "err", shape=(-1,), dtype="float32"),
			],
		)

	def build_body(self) -> None:
		self._init_state()
		x, y, w, h = self.x, self.y, self.width, self.height

		self._body_rect = self.canvas.create_rectangle(
			x, y, x + w, y + h,
			fill="#e8f3ea", outline="#4b8d62", width=2,
			tags=(self.node_id, "node_body"),
		)
		self._title_item = self.canvas.create_text(
			x + w / 2, y + 13,
			text=self.DISPLAY_NAME,
			font=("Arial", 9, "bold"), fill="#2e6d43",
			tags=(self.node_id,),
		)

		hint = tk.Label(
			self.canvas,
			text="prev,next,pts + trig pulse",
			font=("Arial", 8), bg="#eef8f0", fg="#356a49",
		)
		self.canvas.create_window(
			x + w / 2, y + 40, window=hint,
			tags=(self.node_id,),
		)

		status_lbl = tk.Label(
			self.canvas,
			textvariable=self._status_var,
			font=("Arial", 7), bg="#eef8f0", fg="#356a49",
		)
		self.canvas.create_window(
			x + w / 2, y + h - 12, window=status_lbl,
			tags=(self.node_id,),
		)

		self._canvas_items += [self._body_rect, self._title_item]

	def get_help_text(self) -> str:
		return self.HELP_TEXT

	def on_upstream_changed(self) -> None:
		self._buffered_inputs.clear()
		self._enable_latched_high = False
		self._trigger_pending = False
		self._suppress_next_low_status = False
		self._last_warn_key = None
		self._status_var.set("upstream changed")
		self.set_status("waiting", "#666666")

	def _sync_buffer(self, inputs: dict) -> None:
		for pin_name in self._FLOW_INPUT_PINS:
			if pin_name in inputs:
				self._buffered_inputs[pin_name] = inputs[pin_name]
			else:
				self._buffered_inputs.pop(pin_name, None)

	def _missing_required_inputs(self) -> list[str]:
		missing: list[str] = []
		for pin_name in ("prevImg", "nextImg", "prevPts"):
			if self._buffered_inputs.get(pin_name) is None:
				missing.append(pin_name)
		return missing

	@staticmethod
	def _time_tag() -> str:
		return time.strftime("%H:%M:%S")

	def _warn_once(self, key: str, title: str, message: str) -> None:
		if self._last_warn_key == key:
			return
		self._last_warn_key = key
		print(f"[OpticalFlowNode][Warning] {title}: {message}", flush=True)
		try:
			messagebox.showwarning(title, message)
		except Exception:
			pass

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

	@staticmethod
	def _coerce_image(value, name: str) -> np.ndarray:
		image = np.asarray(value)
		if image.ndim not in (2, 3):
			raise ValueError(f"{name} must be a 2D or 3D image array")
		return image

	@staticmethod
	def _coerce_points(value, name: str) -> np.ndarray:
		arr = np.asarray(value, dtype=np.float32)
		if arr.size < 2:
			raise ValueError(f"{name} needs at least one (x, y) point")
		if arr.ndim == 1:
			if arr.size % 2 != 0:
				raise ValueError(f"{name} must have an even number of values")
			arr = arr.reshape(-1, 2)
		elif arr.ndim == 2:
			if arr.shape[-1] != 2:
				raise ValueError(f"{name} must have shape (N, 2) or (N, 1, 2)")
		elif arr.ndim == 3:
			if arr.shape[1:] != (1, 2):
				raise ValueError(f"{name} must have shape (N, 2) or (N, 1, 2)")
			return arr
		else:
			raise ValueError(f"{name} must have shape (N, 2) or (N, 1, 2)")
		return arr.reshape(-1, 1, 2)

	@staticmethod
	def _coerce_size2(value, name: str) -> tuple[int, int]:
		arr = np.asarray(value).reshape(-1)
		if arr.size < 2:
			raise ValueError(f"{name} needs 2 values: [width, height]")
		width = int(round(float(arr[0])))
		height = int(round(float(arr[1])))
		if width <= 0 or height <= 0:
			raise ValueError(f"{name} values must be positive")
		return (width, height)

	@staticmethod
	def _coerce_int(value, name: str) -> int:
		return int(round(float(value)))

	@staticmethod
	def _coerce_criteria(value) -> tuple[int, int, float]:
		arr = np.asarray(value, dtype=np.float64).reshape(-1)
		if arr.size < 3:
			raise ValueError("criteria needs 3 values: [type, maxCount, epsilon]")
		return (int(arr[0]), int(arr[1]), float(arr[2]))

	def _compute_flow(self) -> dict[str, np.ndarray]:
		raw_prev = self._buffered_inputs.get("prevImg")
		raw_next = self._buffered_inputs.get("nextImg")
		raw_prev_pts = self._buffered_inputs.get("prevPts")

		if raw_prev is None or raw_next is None or raw_prev_pts is None:
			raise ValueError("prevImg, nextImg, and prevPts are required")

		prev_img = self._coerce_image(raw_prev, "prevImg")
		next_img = self._coerce_image(raw_next, "nextImg")
		prev_pts = self._coerce_points(raw_prev_pts, "prevPts")

		raw_next_pts = self._buffered_inputs.get("nextPts")
		next_pts = None if raw_next_pts is None else self._coerce_points(raw_next_pts, "nextPts")

		raw_win = self._buffered_inputs.get("winSize")
		win_size = (21, 21) if raw_win is None else self._coerce_size2(raw_win, "winSize")

		raw_level = self._buffered_inputs.get("maxLevel")
		max_level = 3 if raw_level is None else self._coerce_int(raw_level, "maxLevel")

		raw_criteria = self._buffered_inputs.get("criteria")
		criteria = self._DEFAULT_CRITERIA if raw_criteria is None else self._coerce_criteria(raw_criteria)

		raw_flags = self._buffered_inputs.get("flags")
		flags = 0 if raw_flags is None else self._coerce_int(raw_flags, "flags")

		raw_min_eig = self._buffered_inputs.get("minEigThreshold")
		min_eig_threshold = 1e-4 if raw_min_eig is None else float(raw_min_eig)

		next_pts_out, status, err = cv2.calcOpticalFlowPyrLK(
			prev_img,
			next_img,
			prev_pts,
			nextPts=next_pts,
			winSize=win_size,
			maxLevel=max_level,
			criteria=criteria,
			flags=flags,
			minEigThreshold=min_eig_threshold,
		)

		if next_pts_out is None:
			next_pts_arr = np.zeros((0, 2), dtype=np.float32)
		else:
			next_pts_arr = np.asarray(next_pts_out, dtype=np.float32).reshape(-1, 2)

		if status is None:
			status_arr = np.zeros((0,), dtype=np.uint8)
		else:
			status_arr = np.asarray(status, dtype=np.uint8).reshape(-1)

		if err is None:
			err_arr = np.zeros((0,), dtype=np.float32)
		else:
			err_arr = np.asarray(err, dtype=np.float32).reshape(-1)

		return {
			"nextPts": next_pts_arr,
			"status": status_arr,
			"err": err_arr,
		}

	def compute(self, inputs: dict) -> dict:
		self._init_state()
		self._sync_buffer(inputs)

		raw_enable = inputs.get("trig")
		if raw_enable is None:
			# Backward compatibility for older saved projects.
			raw_enable = inputs.get("enableOnce")
		enable_now = False if raw_enable is None else self._coerce_bool_like(raw_enable)
		missing_required = self._missing_required_inputs()

		# Rising edge arms one pending compute request.
		if enable_now and not self._enable_latched_high:
			self._trigger_pending = True
			self._enable_latched_high = True
			self._suppress_next_low_status = True
		elif not enable_now:
			self._enable_latched_high = False

		if not self._trigger_pending:
			self._last_warn_key = None
			if raw_enable is not None and not enable_now and self._suppress_next_low_status:
				# Trigger node emits a short high->low pulse. Do not immediately
				# overwrite status to "frozen low" on that expected reset edge.
				self._suppress_next_low_status = False
				return {
					"_skip_downstream": True,
					"_preserve_cache": True,
				}
			if raw_enable is None:
				self._status_var.set("frozen: trig not connected/pulsed")
			else:
				self._status_var.set("frozen: trig low (waiting for pulse)")
			self.set_status("frozen", "#666666")
			return {
				"_skip_downstream": True,
				"_preserve_cache": True,
			}

		if missing_required:
			missing_csv = ", ".join(missing_required)
			self._warn_once(
				f"missing:{missing_csv}",
				"Optical Flow Trigger Blocked",
				"Optical Flow received trig, but compute did not start because required inputs are missing: "
				+ missing_csv,
			)
			self._status_var.set(
				"trigger pending: waiting for " + missing_csv
			)
			self.set_status("waiting inputs", "#cc6666")
			return {
				"_skip_downstream": True,
				"_preserve_cache": True,
			}

		self._trigger_pending = False
		self._suppress_next_low_status = False
		self._last_warn_key = None

		try:
			result = self._compute_flow()
			tracked = int(result["status"].sum()) if result["status"].size else 0
			# Update status to show how many points were successfully tracked and a time(clock) tag.
			self._status_var.set(f"ok: tracked {tracked}/{result['status'].size} (time: {self._time_tag()})")
#			self._status_var.set(f"ok: tracked {tracked}/{result['status'].size}")
			self.set_status("ok", "#55aa55")
			return result
		except Exception as e:
			self._status_var.set(f"error: {e} (time: {self._time_tag()})")
			self.set_status("error", "#cc0000")
			return {
				"_skip_downstream": True,
				"_preserve_cache": True,
			}
