import tkinter as tk

from node_editor.base_node import BaseNode
from node_editor.execution import ExecutionMode
from node_editor.pin_types import PinDef, PinSchema, PinType


class BatchRunnerNode(BaseNode):
	"""
	Drives repeated downstream execution by emitting index + trigger pulses.

	The loop runs on tk.after() to keep the UI responsive.
	Optional input pins (start/end/step) override internal defaults for a run.
	"""

	EXECUTION_MODE = ExecutionMode.SYNC
	NODE_TYPE = "batch_runner"
	DISPLAY_NAME = "Batch Runner"
	CATEGORY = "source"
	NODE_WIDTH = 220
	NODE_HEIGHT = 140

	def get_pin_schema(self) -> PinSchema:
		return PinSchema(
			inputs=[
				PinDef("start", PinType.SCALAR, "start", optional=True),
				PinDef("end", PinType.SCALAR, "end", optional=True),
				PinDef("step", PinType.SCALAR, "step", optional=True),
			],
			outputs=[
				PinDef("current_index", PinType.SCALAR, "idx"),
				PinDef("prev_index", PinType.SCALAR, "prev"),
				PinDef("ref_index", PinType.SCALAR, "ref"),
				PinDef("trigger", PinType.TRIGGER, "trig"),
				PinDef("done", PinType.TRIGGER, "done"),
			],
		)

	def _init_state(self) -> None:
		if hasattr(self, "_status_var"):
			return

		self._status_var = tk.StringVar(value="idle")
		self._run_btn_var = tk.StringVar(value="Run")

		self._default_start_var = tk.StringVar(value="0")
		self._default_iterations_var = tk.StringVar(value="10")
		self._default_step_var = tk.StringVar(value="1")
		self._tick_ms_var = tk.StringVar(value="1")

		self._running = False
		self._run_requested = False
		self._after_id: str | None = None

		self._iter_values: list[int] = []
		self._iter_pos = 0
		self._ref_index = 0

		self._buffered_inputs: dict[str, object] = {}
		self._last_outputs: dict[str, float] = {
			"current_index": 0.0,
			"prev_index": -1.0,
			"ref_index": 0.0,
			"trigger": 0.0,
			"done": 0.0,
		}

	def build_body(self) -> None:
		self._init_state()
		x, y, w, h = self.x, self.y, self.width, self.height

		self._body_rect = self.canvas.create_rectangle(
			x, y, x + w, y + h,
			fill="#efe8ff", outline="#6e58a8", width=2,
			tags=(self.node_id, "node_body"),
		)
		self._title_item = self.canvas.create_text(
			x + w / 2, y + 13,
			text=self.DISPLAY_NAME,
			font=("Arial", 9, "bold"), fill="#4f3c83",
			tags=(self.node_id,),
		)

		run_btn = tk.Button(
			self.canvas,
			textvariable=self._run_btn_var,
			font=("Arial", 8),
			command=self._toggle_run,
			padx=8,
			pady=1,
		)
		self.canvas.create_window(
			x + w / 2, y + 44,
			window=run_btn,
			tags=(self.node_id,),
		)

		status_lbl = tk.Label(
			self.canvas,
			textvariable=self._status_var,
			font=("Arial", 7), bg="#efe8ff", fg="#5a4b86",
		)
		self.canvas.create_window(
			x + w / 2, y + h - 14,
			window=status_lbl,
			tags=(self.node_id,),
		)

		self._canvas_items += [self._body_rect, self._title_item]

	def build_inspector(self, parent: tk.Frame) -> None:
		self._init_state()

		tk.Label(parent, text="Default start", font=("Arial", 9)).grid(row=0, column=0, sticky="w")
		tk.Entry(parent, textvariable=self._default_start_var, width=10, font=("Arial", 9)).grid(
			row=0, column=1, sticky="w", padx=(8, 0)
		)

		tk.Label(parent, text="Default iterations (N)", font=("Arial", 9)).grid(row=1, column=0, sticky="w", pady=(6, 0))
		tk.Entry(parent, textvariable=self._default_iterations_var, width=10, font=("Arial", 9)).grid(
			row=1, column=1, sticky="w", padx=(8, 0), pady=(6, 0)
		)

		tk.Label(parent, text="Default step", font=("Arial", 9)).grid(row=2, column=0, sticky="w", pady=(6, 0))
		tk.Entry(parent, textvariable=self._default_step_var, width=10, font=("Arial", 9)).grid(
			row=2, column=1, sticky="w", padx=(8, 0), pady=(6, 0)
		)

		tk.Label(parent, text="Tick delay ms", font=("Arial", 9)).grid(row=3, column=0, sticky="w", pady=(6, 0))
		tk.Entry(parent, textvariable=self._tick_ms_var, width=10, font=("Arial", 9)).grid(
			row=3, column=1, sticky="w", padx=(8, 0), pady=(6, 0)
		)

		tk.Button(parent, textvariable=self._run_btn_var, command=self._toggle_run, width=10).grid(
			row=4, column=1, sticky="e", pady=(10, 0)
		)
		tk.Label(parent, textvariable=self._status_var, font=("Arial", 9), fg="#4f3c83").grid(
			row=4, column=0, sticky="w", pady=(10, 0)
		)

	def _toggle_run(self) -> None:
		if self._running:
			self._stop_loop("stopped")
			return
		self._run_requested = True
		if self._request_downstream:
			self._request_downstream(self.node_id)

	def _stop_loop(self, status: str = "idle") -> None:
		self._running = False
		self._run_btn_var.set("Run")
		if self._after_id is not None:
			try:
				self.canvas.after_cancel(self._after_id)
			except Exception:
				pass
			self._after_id = None
		self._status_var.set(status)

	def _parse_int(self, value, name: str) -> int:
		try:
			return int(round(float(value)))
		except Exception:
			raise ValueError(f"{name} must be numeric")

	def _resolve_loop_config(self) -> tuple[int, int, int]:
		start_default = self._parse_int(self._default_start_var.get(), "default start")
		step_default = self._parse_int(self._default_step_var.get(), "default step")
		n_default = self._parse_int(self._default_iterations_var.get(), "default iterations")
		if n_default <= 0:
			raise ValueError("default iterations must be > 0")
		if step_default == 0:
			raise ValueError("step must not be 0")

		start = start_default
		step = step_default
		end = start_default + (n_default - 1) * step_default

		raw_start = self._buffered_inputs.get("start")
		raw_end = self._buffered_inputs.get("end")
		raw_step = self._buffered_inputs.get("step")

		if raw_start is not None:
			start = self._parse_int(raw_start, "start")
		if raw_step is not None:
			step = self._parse_int(raw_step, "step")
			if step == 0:
				raise ValueError("step must not be 0")
		if raw_end is not None:
			end = self._parse_int(raw_end, "end")
		elif raw_start is not None or raw_step is not None:
			# Preserve internal default run length N when end pin is disconnected.
			end = start + (n_default - 1) * step

		return start, end, step

	def _build_indices(self, start: int, end: int, step: int) -> list[int]:
		if step > 0:
			if start > end:
				return []
			return list(range(start, end + 1, step))
		if start < end:
			return []
		return list(range(start, end - 1, step))

	def _tick_loop(self) -> None:
		if not self._running:
			return

		if self._iter_pos >= len(self._iter_values):
			self._last_outputs = {
				"current_index": None,
				"prev_index": None,
				"ref_index": float(self._ref_index),
				"trigger": 0.0,
				"done": 1.0,
			}
			self.push_output(dict(self._last_outputs))
			self._stop_loop("done")
			return

		cur = self._iter_values[self._iter_pos]
		prev = cur - 1

		self._last_outputs = {
			"current_index": float(cur),
			"prev_index": float(prev),
			"ref_index": float(self._ref_index),
			"trigger": 1.0,
			"done": 0.0,
		}
		self.push_output(dict(self._last_outputs))

		self._iter_pos += 1
		self._status_var.set(f"running {self._iter_pos}/{len(self._iter_values)}")

		try:
			tick_ms = self._parse_int(self._tick_ms_var.get(), "tick delay")
		except Exception:
			tick_ms = 1
		tick_ms = max(1, tick_ms)
		self._after_id = self.canvas.after(tick_ms, self._tick_loop)

	def compute(self, inputs: dict) -> dict:
		self._buffered_inputs = dict(inputs or {})

		if not self._run_requested:
			return dict(self._last_outputs)

		self._run_requested = False
		if self._running:
			return {
				"_skip_downstream": True,
				"_preserve_cache": True,
			}

		try:
			start, end, step = self._resolve_loop_config()
			values = self._build_indices(start, end, step)
			if not values:
				self._status_var.set("empty range")
				self.set_status("empty", "#cc6666")
				return {
					"_skip_downstream": True,
					"_preserve_cache": True,
				}

			self._iter_values = values
			self._iter_pos = 0
			self._ref_index = values[0]
			self._running = True
			self._run_btn_var.set("Stop")
			self.set_status("running", "#4477aa")
			self._tick_loop()
			return {
				"_skip_downstream": True,
				"_preserve_cache": True,
			}
		except Exception as e:
			self._status_var.set(f"error: {e}")
			self.set_status("error", "#cc0000")
			return {
				"_skip_downstream": True,
				"_preserve_cache": True,
			}

	def get_params(self) -> dict:
		return {
			"default_start": self._default_start_var.get(),
			"default_iterations": self._default_iterations_var.get(),
			"default_step": self._default_step_var.get(),
			"tick_ms": self._tick_ms_var.get(),
		}

	def set_params(self, params: dict) -> None:
		self._init_state()
		self._default_start_var.set(str(params.get("default_start", "0")))
		self._default_iterations_var.set(str(params.get("default_iterations", "10")))
		self._default_step_var.set(str(params.get("default_step", "1")))
		self._tick_ms_var.set(str(params.get("tick_ms", "1")))

	def on_destroy(self) -> None:
		self._stop_loop("destroyed")
		super().on_destroy()
 