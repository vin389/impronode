import sys
import time
from pathlib import Path as _Path

# calcTemplateMatchPyr lives as a top-level script module, not inside the
# node_editor package — make sure the repo root is importable regardless of
# how this node module gets loaded (main.py, pytest, debugpy, ...).
_REPO_ROOT = _Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
	sys.path.insert(0, str(_REPO_ROOT))

try:
	from calcTemplateMatchPyr import calcTemplateMatchPyr  # type: ignore[import-not-found]
except ImportError:
	from calcTemplateMatchPyr_codex import calcTemplateMatchPyr

import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np

from node_editor.base_node import BaseNode
from node_editor.execution import ExecutionMode
from node_editor.pin_types import PinDef, PinSchema, PinType


class TemplateMatchNode(BaseNode):
	"""
	Coarse-to-fine per-point template matching node (cv2.matchTemplate pyramid).

	Wraps calcTemplateMatchPyr and follows the same trig-gated, buffered-input
	pattern as OpticalFlowNode so it can be dropped in as a drop-in alternative
	tracker for large displacements / small templates on huge images.
	"""

	EXECUTION_MODE = ExecutionMode.BACKGROUND
	NODE_TYPE = "template_match"
	DISPLAY_NAME = "Template Match"
	CATEGORY = "process"
	SEARCH_KEYWORDS = ("template", "match", "matchtemplate", "dic", "pyr", "tm")
	NODE_WIDTH = 250
	NODE_HEIGHT = 120

	_DEFAULT_TEMPLATE_SIZE = (31, 31)
	_DEFAULT_SEARCH_RANGE = (48, 48)
	_DEFAULT_NUM_LEVELS = 3
	_DEFAULT_MATCH_METHOD_NAME = "TM_CCOEFF_NORMED"
	_DEFAULT_SUBPIXEL = True
	_DEFAULT_MIN_CORRELATION = 0.5
	_DEFAULT_FINE_SEARCH_RADIUS = (4, 4)
	_DEFAULT_SIDELOBE_RADIUS = 2

	_MATCH_METHODS = {
		"TM_SQDIFF": cv2.TM_SQDIFF,
		"TM_SQDIFF_NORMED": cv2.TM_SQDIFF_NORMED,
		"TM_CCORR": cv2.TM_CCORR,
		"TM_CCORR_NORMED": cv2.TM_CCORR_NORMED,
		"TM_CCOEFF": cv2.TM_CCOEFF,
		"TM_CCOEFF_NORMED": cv2.TM_CCOEFF_NORMED,
	}

	_MATCH_INPUT_PINS = ("prevImg", "nextImg", "prevPts")
	_OPTIONAL_INPUT_PINS = ("nextPtsInit",)

	HELP_TEXT = (
		"Template Match Node\n\n"
		"Purpose:\n"
		"- Run a coarse-to-fine cv2.matchTemplate pyramid tracker on demand.\n"
		"  Suited to large displacements and/or small templates on huge images,\n"
		"  where an optical-flow linearisation is uncomfortable.\n\n"
		"Input Pins:\n"
		"- prevImg [IMAGE]: previous frame (required).\n"
		"- nextImg [IMAGE]: current/next frame (required).\n"
		"- prevPts [ARRAY float64 Nx2]: source points to track (required).\n"
		"- nextPtsInit [ARRAY float64 Nx2, optional]: initial-guess locations in\n"
		"  nextImg for each point, same role as cv2.calcOpticalFlowPyrLK's\n"
		"  nextPts with OPTFLOW_USE_INITIAL_FLOW -- re-centres the coarsest\n"
		"  pyramid level's search instead of assuming zero motion. Must match\n"
		"  prevPts' shape; a NaN row falls back to zero motion for that point.\n"
		"- trig [TRIGGER, optional]: while false, inputs are buffered\n"
		"  without computing; on each truthy pulse OR each new value while\n"
		"  already truthy (so an ever-incrementing counter works too), compute\n"
		"  exactly once. If required inputs are still missing when the pulse\n"
		"  arrives, the pulse stays queued and fires automatically as soon as\n"
		"  they become available -- no need to re-trigger.\n\n"
		"Inspector Settings:\n"
		"- Template size (w,h): full-resolution source subset size.\n"
		"- Search range (w,h): max full-resolution displacement, used only\n"
		"  at the coarsest pyramid level.\n"
		"- Num levels: Gaussian pyramid depth (1 = plain matchTemplate).\n"
		"- Match method: any cv2.matchTemplate method.\n"
		"- Subpixel: refine the integer peak with a 3x3 quadratic fit.\n"
		"- Min correlation: success threshold in [0, 1] for status.\n"
		"- Fine search radius (w,h): residual search radius at every level\n"
		"  after the coarsest one.\n"
		"- Sidelobe exclusion radius: masked radius around the main peak\n"
		"  when measuring peak-to-sidelobe sharpness for confidence.\n\n"
		"Output Pins:\n"
		"- nextPts [ARRAY float64 Nx2]: tracked output points (NaN if failed).\n"
		"- status [ARRAY uint8 N]: 1 means the peak met min_correlation, else 0.\n"
		"- confidence [ARRAY float64 N]: match quality/sharpness score in [0, 1].\n"
	)

	def _init_state(self) -> None:
		if hasattr(self, "_status_var"):
			return
		self._status_var = tk.StringVar(value="frozen")
		self._tmpl_w_var = tk.StringVar(value=str(self._DEFAULT_TEMPLATE_SIZE[0]))
		self._tmpl_h_var = tk.StringVar(value=str(self._DEFAULT_TEMPLATE_SIZE[1]))
		self._search_w_var = tk.StringVar(value=str(self._DEFAULT_SEARCH_RANGE[0]))
		self._search_h_var = tk.StringVar(value=str(self._DEFAULT_SEARCH_RANGE[1]))
		self._num_levels_var = tk.StringVar(value=str(self._DEFAULT_NUM_LEVELS))
		self._match_method_var = tk.StringVar(value=self._DEFAULT_MATCH_METHOD_NAME)
		self._subpixel_var = tk.BooleanVar(value=self._DEFAULT_SUBPIXEL)
		self._min_corr_var = tk.StringVar(value=str(self._DEFAULT_MIN_CORRELATION))
		self._fine_w_var = tk.StringVar(value=str(self._DEFAULT_FINE_SEARCH_RADIUS[0]))
		self._fine_h_var = tk.StringVar(value=str(self._DEFAULT_FINE_SEARCH_RADIUS[1]))
		self._sidelobe_var = tk.StringVar(value=str(self._DEFAULT_SIDELOBE_RADIUS))
		self._buffered_inputs: dict[str, object] = {}
		# --- trigger edge-detection state ------------------------------
		# See OpticalFlowNode._init_state() for the full rationale -- the
		# same edge-detection design is used here so both nodes behave
		# identically with either a boolean pulse-style trigger source or
		# an ever-incrementing counter-style trigger source.
		self._last_trig_value = None
		self._trig_was_truthy = False
		self._trigger_pending = False
		self._has_computed_once = False
		self._last_warn_key: str | None = None

	def get_pin_schema(self) -> PinSchema:
		return PinSchema(
			inputs=[
				PinDef("prevImg", PinType.IMAGE, "prev"),
				PinDef("nextImg", PinType.IMAGE, "next"),
				PinDef("prevPts", PinType.ARRAY, "prevPts", shape=(-1, 2), dtype="float64"),
				PinDef("nextPtsInit", PinType.ARRAY, "nextPts (init)", shape=(-1, 2), dtype="float64", optional=True),
				PinDef("trig", PinType.TRIGGER, "trig", optional=True),
			],
			outputs=[
				PinDef("nextPts", PinType.ARRAY, "nextPts", shape=(-1, 2), dtype="float64"),
				PinDef("status", PinType.ARRAY, "status", shape=(-1,), dtype="uint8"),
				PinDef("confidence", PinType.ARRAY, "conf", shape=(-1,), dtype="float64"),
			],
		)

	def build_body(self) -> None:
		self._init_state()
		x, y, w, h = self.x, self.y, self.width, self.height

		self._body_rect = self.canvas.create_rectangle(
			x, y, x + w, y + h,
			fill="#e6f7f7", outline="#2f8f8f", width=2,
			tags=(self.node_id, "node_body"),
		)
		self._title_item = self.canvas.create_text(
			x + w / 2, y + 13,
			text=self.DISPLAY_NAME,
			font=("Arial", 9, "bold"), fill="#1f5f5f",
			tags=(self.node_id,),
		)

		hint = tk.Label(
			self.canvas,
			text="prev,next,pts + trig pulse",
			font=("Arial", 8), bg="#e6f7f7", fg="#1f5f5f",
		)
		self.canvas.create_window(
			x + w / 2, y + 40, window=hint,
			tags=(self.node_id,),
		)

		status_lbl = tk.Label(
			self.canvas,
			textvariable=self._status_var,
			font=("Arial", 7), bg="#e6f7f7", fg="#1f5f5f",
		)
		self.canvas.create_window(
			x + w / 2, y + h - 12, window=status_lbl,
			tags=(self.node_id,),
		)

		self._canvas_items += [self._body_rect, self._title_item]

	def get_help_text(self) -> str:
		return self.HELP_TEXT

	def build_inspector(self, parent: tk.Frame) -> None:
		self._init_state()

		tk.Label(
			parent,
			text="Template match defaults used every time it runs:",
			font=("Arial", 9, "bold"),
		).pack(anchor="w", pady=(0, 6))

		grid = tk.Frame(parent)
		grid.pack(fill="x")

		tk.Label(grid, text="Template size (w,h)", font=("Arial", 8)).grid(row=0, column=0, sticky="w", padx=(0, 6), pady=2)
		tk.Entry(grid, textvariable=self._tmpl_w_var, width=6, justify="center").grid(row=0, column=1, sticky="w", pady=2)
		tk.Entry(grid, textvariable=self._tmpl_h_var, width=6, justify="center").grid(row=0, column=2, sticky="w", padx=(4, 0), pady=2)

		tk.Label(grid, text="Search range (w,h)", font=("Arial", 8)).grid(row=1, column=0, sticky="w", padx=(0, 6), pady=2)
		tk.Entry(grid, textvariable=self._search_w_var, width=6, justify="center").grid(row=1, column=1, sticky="w", pady=2)
		tk.Entry(grid, textvariable=self._search_h_var, width=6, justify="center").grid(row=1, column=2, sticky="w", padx=(4, 0), pady=2)

		tk.Label(grid, text="Num levels", font=("Arial", 8)).grid(row=2, column=0, sticky="w", padx=(0, 6), pady=2)
		tk.Entry(grid, textvariable=self._num_levels_var, width=10, justify="center").grid(row=2, column=1, columnspan=2, sticky="w", pady=2)

		tk.Label(grid, text="Match method", font=("Arial", 8)).grid(row=3, column=0, sticky="w", padx=(0, 6), pady=2)
		ttk.Combobox(
			grid, textvariable=self._match_method_var,
			values=list(self._MATCH_METHODS.keys()),
			width=16, state="readonly",
		).grid(row=3, column=1, columnspan=3, sticky="w", pady=2)

		tk.Checkbutton(
			grid, text="Subpixel refinement", variable=self._subpixel_var,
		).grid(row=4, column=0, columnspan=3, sticky="w", pady=2)

		tk.Label(grid, text="Min correlation (0-1)", font=("Arial", 8)).grid(row=5, column=0, sticky="w", padx=(0, 6), pady=2)
		tk.Entry(grid, textvariable=self._min_corr_var, width=10, justify="center").grid(row=5, column=1, columnspan=2, sticky="w", pady=2)

		tk.Label(grid, text="Fine search radius (w,h)", font=("Arial", 8)).grid(row=6, column=0, sticky="w", padx=(0, 6), pady=2)
		tk.Entry(grid, textvariable=self._fine_w_var, width=6, justify="center").grid(row=6, column=1, sticky="w", pady=2)
		tk.Entry(grid, textvariable=self._fine_h_var, width=6, justify="center").grid(row=6, column=2, sticky="w", padx=(4, 0), pady=2)

		tk.Label(grid, text="Sidelobe exclusion radius", font=("Arial", 8)).grid(row=7, column=0, sticky="w", padx=(0, 6), pady=2)
		tk.Entry(grid, textvariable=self._sidelobe_var, width=10, justify="center").grid(row=7, column=1, columnspan=2, sticky="w", pady=2)

		tk.Button(grid, text="Reset Defaults", command=self._reset_defaults).grid(
			row=8, column=0, columnspan=4, sticky="w", pady=(8, 2)
		)

		tk.Label(
			parent,
			text="If a field is empty/invalid, its default is restored automatically.",
			font=("Arial", 8),
			foreground="#555555",
		).pack(anchor="w", pady=(6, 0))

	def _reset_defaults(self) -> None:
		self._tmpl_w_var.set(str(self._DEFAULT_TEMPLATE_SIZE[0]))
		self._tmpl_h_var.set(str(self._DEFAULT_TEMPLATE_SIZE[1]))
		self._search_w_var.set(str(self._DEFAULT_SEARCH_RANGE[0]))
		self._search_h_var.set(str(self._DEFAULT_SEARCH_RANGE[1]))
		self._num_levels_var.set(str(self._DEFAULT_NUM_LEVELS))
		self._match_method_var.set(self._DEFAULT_MATCH_METHOD_NAME)
		self._subpixel_var.set(self._DEFAULT_SUBPIXEL)
		self._min_corr_var.set(str(self._DEFAULT_MIN_CORRELATION))
		self._fine_w_var.set(str(self._DEFAULT_FINE_SEARCH_RADIUS[0]))
		self._fine_h_var.set(str(self._DEFAULT_FINE_SEARCH_RADIUS[1]))
		self._sidelobe_var.set(str(self._DEFAULT_SIDELOBE_RADIUS))

	def _parse_default_int(self, variable: tk.StringVar,
							default: int,
							min_value: int | None = None) -> int:
		text = (variable.get() or "").strip()
		try:
			value = int(round(float(text)))
		except Exception:
			value = default
		if min_value is not None and value < min_value:
			value = default
		variable.set(str(value))
		return value

	def _parse_default_float(self, variable: tk.StringVar,
							  default: float,
							  min_value: float | None = None,
							  max_value: float | None = None) -> float:
		text = (variable.get() or "").strip()
		try:
			value = float(text)
		except Exception:
			value = default
		if min_value is not None and value < min_value:
			value = default
		if max_value is not None and value > max_value:
			value = default
		variable.set(str(value))
		return value

	def _settings_from_inspector(self) -> dict:
		template_size = (
			self._parse_default_int(self._tmpl_w_var, self._DEFAULT_TEMPLATE_SIZE[0], min_value=3),
			self._parse_default_int(self._tmpl_h_var, self._DEFAULT_TEMPLATE_SIZE[1], min_value=3),
		)
		search_range = (
			self._parse_default_int(self._search_w_var, self._DEFAULT_SEARCH_RANGE[0], min_value=0),
			self._parse_default_int(self._search_h_var, self._DEFAULT_SEARCH_RANGE[1], min_value=0),
		)
		num_levels = self._parse_default_int(self._num_levels_var, self._DEFAULT_NUM_LEVELS, min_value=1)
		method_name = self._match_method_var.get().strip()
		match_method = self._MATCH_METHODS.get(method_name, self._MATCH_METHODS[self._DEFAULT_MATCH_METHOD_NAME])
		if method_name not in self._MATCH_METHODS:
			self._match_method_var.set(self._DEFAULT_MATCH_METHOD_NAME)
		subpixel = bool(self._subpixel_var.get())
		min_correlation = self._parse_default_float(
			self._min_corr_var, self._DEFAULT_MIN_CORRELATION, min_value=0.0, max_value=1.0)
		fine_search_radius = (
			self._parse_default_int(self._fine_w_var, self._DEFAULT_FINE_SEARCH_RADIUS[0], min_value=0),
			self._parse_default_int(self._fine_h_var, self._DEFAULT_FINE_SEARCH_RADIUS[1], min_value=0),
		)
		sidelobe_exclusion_radius = self._parse_default_int(
			self._sidelobe_var, self._DEFAULT_SIDELOBE_RADIUS, min_value=0)

		return {
			"template_size": template_size,
			"search_range": search_range,
			"num_levels": num_levels,
			"match_method": match_method,
			"subpixel": subpixel,
			"min_correlation": min_correlation,
			"fine_search_radius": fine_search_radius,
			"sidelobe_exclusion_radius": sidelobe_exclusion_radius,
		}

	def on_upstream_changed(self) -> None:
		self._buffered_inputs.clear()
		self._last_trig_value = None
		self._trig_was_truthy = False
		self._trigger_pending = False
		self._has_computed_once = False
		self._last_warn_key = None
		self._status_var.set("upstream changed")
		self.set_status("waiting", "#666666")

	def _sync_buffer(self, inputs: dict) -> None:
		persistent_pins = {"prevPts", "nextPtsInit"}
		for pin_name in self._MATCH_INPUT_PINS + self._OPTIONAL_INPUT_PINS:
			has_value = pin_name in inputs and inputs[pin_name] is not None
			if has_value:
				value = inputs[pin_name]
				if isinstance(value, np.ndarray):
					# Buffered values are held across multiple compute()
					# calls -- from whenever they arrive until the next
					# trigger pulse actually consumes them. Without a copy
					# here, self._buffered_inputs would merely alias
					# whatever array object the upstream node handed us;
					# if that same array object is later mutated in place
					# by anything (a node reusing a preallocated buffer, a
					# cv2 call with dst=...), the buffered snapshot would
					# silently change too, well after the fact.
					value = value.copy()
				self._buffered_inputs[pin_name] = value
			elif pin_name in persistent_pins and pin_name in self._buffered_inputs:
				# Keep the last valid tracking points across trigger pulses when
				# upstream has not resent them (either the pin is unlinked, or a
				# linked upstream source simply hasn't republished a value this
				# pass). This allows template tracking to continue using the
				# cached prevPts when the point set has not changed, without
				# requiring a redundant upstream recomputation.
				continue
			else:
				self._buffered_inputs.pop(pin_name, None)

	def _missing_required_inputs(self) -> list[str]:
		missing: list[str] = []
		for pin_name in self._MATCH_INPUT_PINS:
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
		print(f"[TemplateMatchNode][Warning] {title}: {message}", flush=True)
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
		arr = np.asarray(value, dtype=np.float64)
		if arr.ndim == 1:
			if arr.size < 2 or arr.size % 2 != 0:
				raise ValueError(f"{name} must have an even number of values")
			arr = arr.reshape(-1, 2)
		elif arr.ndim == 2:
			if arr.shape[-1] != 2:
				raise ValueError(f"{name} must have shape (N, 2)")
		elif arr.ndim == 3 and arr.shape[1:] == (1, 2):
			arr = arr.reshape(-1, 2)
		else:
			raise ValueError(f"{name} must have shape (N, 2) or (N, 1, 2)")
		if arr.shape[0] < 1:
			raise ValueError(f"{name} needs at least one (x, y) point")
		return arr

	def _compute_match(self) -> dict[str, np.ndarray]:
		settings = self._settings_from_inspector()

		raw_prev = self._buffered_inputs.get("prevImg")
		raw_next = self._buffered_inputs.get("nextImg")
		raw_prev_pts = self._buffered_inputs.get("prevPts")

		if raw_prev is None or raw_next is None or raw_prev_pts is None:
			raise ValueError("prevImg, nextImg, and prevPts are required")

		prev_img = self._coerce_image(raw_prev, "prevImg")
		next_img = self._coerce_image(raw_next, "nextImg")
		prev_pts = self._coerce_points(raw_prev_pts, "prevPts")

		raw_next_pts_init = self._buffered_inputs.get("nextPtsInit")
		next_pts_init = None
		if raw_next_pts_init is not None:
			next_pts_init = self._coerce_points(raw_next_pts_init, "nextPtsInit")

		next_pts, status, confidence = calcTemplateMatchPyr(
			prev_img,
			next_img,
			prev_pts,
			template_size=settings["template_size"],
			search_range=settings["search_range"],
			num_levels=settings["num_levels"],
			match_method=settings["match_method"],
			subpixel=settings["subpixel"],
			min_correlation=settings["min_correlation"],
			fine_search_radius=settings["fine_search_radius"],
			sidelobe_exclusion_radius=settings["sidelobe_exclusion_radius"],
			next_pts_init=next_pts_init,
		)

		return {
			"nextPts": np.asarray(next_pts, dtype=np.float64).reshape(-1, 2),
			"status": np.asarray(status, dtype=np.uint8).reshape(-1),
			"confidence": np.asarray(confidence, dtype=np.float64).reshape(-1),
		}

	def compute(self, inputs: dict) -> dict:
		self._init_state()
		self._sync_buffer(inputs)

		raw_trig = inputs.get("trig")
		is_truthy_now = raw_trig is not None and self._coerce_bool_like(raw_trig)
		was_truthy_before = self._trig_was_truthy
		value_changed = raw_trig is not None and raw_trig != self._last_trig_value

		# See OpticalFlowNode.compute() for the full rationale behind this
		# condition: it fires on a classic rising edge (falsy -> truthy)
		# for boolean pulse-style triggers, AND on any value change while
		# already truthy, for ever-incrementing counter-style triggers
		# that never return to a falsy value.
		if is_truthy_now and (not was_truthy_before or value_changed):
			self._trigger_pending = True

		self._trig_was_truthy = is_truthy_now
		self._last_trig_value = raw_trig

		if not self._trigger_pending:
			self._last_warn_key = None
			if raw_trig is None:
				self._status_var.set("frozen: trig not connected/pulsed")
				self.set_status("frozen", "#666666")
			elif not self._has_computed_once:
				self._status_var.set("frozen: waiting for first trigger pulse")
				self.set_status("frozen", "#666666")
			# else: idle in-between call (falling edge, or an unrelated
			# recompute with an unchanged trig value) -- leave the last
			# "ok"/"error" status alone.
			return {
				"_skip_downstream": True,
				"_preserve_cache": True,
			}

		missing_required = self._missing_required_inputs()
		if missing_required:
			missing_csv = ", ".join(missing_required)
			self._warn_once(
				f"missing:{missing_csv}",
				"Template Match Trigger Blocked",
				"Template Match received trig, but compute did not start because required inputs are missing: "
				+ missing_csv,
			)
			self._status_var.set(
				"trigger pending: waiting for " + missing_csv
			)
			self.set_status("waiting inputs", "#cc6666")
			# _trigger_pending intentionally stays True here so the queued
			# pulse fires automatically as soon as the missing inputs
			# arrive on a later call.
			return {
				"_skip_downstream": True,
				"_preserve_cache": True,
			}

		self._trigger_pending = False
		self._last_warn_key = None

		try:
			result = self._compute_match()
			tracked = int(result["status"].sum()) if result["status"].size else 0
			self._status_var.set(f"ok: matched {tracked}/{result['status'].size} (time: {self._time_tag()})")
			self.set_status("ok", "#55aa55")
			self._has_computed_once = True
			return result
		except Exception as e:
			self._status_var.set(f"error: {e} (time: {self._time_tag()})")
			self.set_status("error", "#cc0000")
			self._has_computed_once = True
			return {
				"_skip_downstream": True,
				"_preserve_cache": True,
			}

	def get_params(self) -> dict:
		self._init_state()
		return {
			"template_w": self._tmpl_w_var.get(),
			"template_h": self._tmpl_h_var.get(),
			"search_w": self._search_w_var.get(),
			"search_h": self._search_h_var.get(),
			"num_levels": self._num_levels_var.get(),
			"match_method": self._match_method_var.get(),
			"subpixel": bool(self._subpixel_var.get()),
			"min_correlation": self._min_corr_var.get(),
			"fine_w": self._fine_w_var.get(),
			"fine_h": self._fine_h_var.get(),
			"sidelobe": self._sidelobe_var.get(),
		}

	def set_params(self, params: dict) -> None:
		self._init_state()
		self._tmpl_w_var.set(str(params.get("template_w", self._DEFAULT_TEMPLATE_SIZE[0])))
		self._tmpl_h_var.set(str(params.get("template_h", self._DEFAULT_TEMPLATE_SIZE[1])))
		self._search_w_var.set(str(params.get("search_w", self._DEFAULT_SEARCH_RANGE[0])))
		self._search_h_var.set(str(params.get("search_h", self._DEFAULT_SEARCH_RANGE[1])))
		self._num_levels_var.set(str(params.get("num_levels", self._DEFAULT_NUM_LEVELS)))
		self._match_method_var.set(str(params.get("match_method", self._DEFAULT_MATCH_METHOD_NAME)))
		self._subpixel_var.set(bool(params.get("subpixel", self._DEFAULT_SUBPIXEL)))
		self._min_corr_var.set(str(params.get("min_correlation", self._DEFAULT_MIN_CORRELATION)))
		self._fine_w_var.set(str(params.get("fine_w", self._DEFAULT_FINE_SEARCH_RADIUS[0])))
		self._fine_h_var.set(str(params.get("fine_h", self._DEFAULT_FINE_SEARCH_RADIUS[1])))
		self._sidelobe_var.set(str(params.get("sidelobe", self._DEFAULT_SIDELOBE_RADIUS)))
		# Normalize restored text so invalid/missing saved values fall back to defaults.
		self._settings_from_inspector()