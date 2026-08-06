import tkinter as tk

from node_editor.base_node import BaseNode
from node_editor.execution import ExecutionMode
from node_editor.pin_types import PinDef, PinSchema, PinType


class TriggerNode(BaseNode):
	"""
	Dual trigger source node.

	Provides:
	- Manual one-shot trigger button.
	- Timer-based trigger stream with configurable interval and count.
	- Independent counters for manual and timer triggers.
	"""

	EXECUTION_MODE = ExecutionMode.SYNC
	NODE_TYPE = "trigger"
	DISPLAY_NAME = "Trigger"
	CATEGORY = "source"
	NODE_WIDTH = 240
	NODE_HEIGHT = 145

	def get_pin_schema(self) -> PinSchema:
		return PinSchema(
			inputs=[
				PinDef("trig", PinType.TRIGGER, "trig", optional=True),
			],
			outputs=[
				PinDef("trigger", PinType.TRIGGER, "trig"),
				PinDef("button_count", PinType.SCALAR, "btn_n"),
				PinDef("timer_count", PinType.SCALAR, "tmr_n"),
				PinDef("total_count", PinType.SCALAR, "all_n"),
			],
		)

	def _init_state(self) -> None:
		if hasattr(self, "_status_var"):
			return

		self._status_var = tk.StringVar(value="idle")
		self._counts_var = tk.StringVar(value="btn=0  tmr=0  all=0")
		self._timer_btn_var = tk.StringVar(value="Start Timer")

		self._interval_var = tk.StringVar(value="1.0")
		self._timer_target_var = tk.StringVar(value="0")
		self._interval_error_var = tk.StringVar(value="")

		self._button_count = 0
		self._timer_count = 0
		self._timer_run_sent = 0

		self._timer_running = False
		self._timer_after_id: str | None = None
		self._interval_trace_id: str | None = None
		self._input_latched_high = False

		self._last_outputs = {
			"trigger": 0.0,
			"button_count": 0.0,
			"timer_count": 0.0,
			"total_count": 0.0,
		}

		self._interval_trace_id = self._interval_var.trace_add("write", self._on_interval_text_changed)

	def build_body(self) -> None:
		self._init_state()
		x, y, w, h = self.x, self.y, self.width, self.height

		self._body_rect = self.canvas.create_rectangle(
			x,
			y,
			x + w,
			y + h,
			fill="#f5eee3",
			outline="#9e6d27",
			width=2,
			tags=(self.node_id, "node_body"),
		)
		self._title_item = self.canvas.create_text(
			x + w / 2,
			y + 13,
			text=self.DISPLAY_NAME,
			font=("Arial", 9, "bold"),
			fill="#7a4f14",
			tags=(self.node_id,),
		)

		manual_btn = tk.Button(
			self.canvas,
			text="Trigger Once",
			font=("Arial", 8),
			command=self._on_manual_trigger,
			padx=6,
			pady=1,
		)
		self.canvas.create_window(
			x + w * 0.32,
			y + 45,
			window=manual_btn,
			tags=(self.node_id,),
		)

		timer_btn = tk.Button(
			self.canvas,
			textvariable=self._timer_btn_var,
			font=("Arial", 8),
			command=self._toggle_timer,
			padx=6,
			pady=1,
		)
		self.canvas.create_window(
			x + w * 0.72,
			y + 45,
			window=timer_btn,
			tags=(self.node_id,),
		)

		counts_lbl = tk.Label(
			self.canvas,
			textvariable=self._counts_var,
			font=("Arial", 8),
			bg="#f5eee3",
			fg="#7a4f14",
		)
		self.canvas.create_window(
			x + w / 2,
			y + h - 30,
			window=counts_lbl,
			tags=(self.node_id,),
		)

		status_lbl = tk.Label(
			self.canvas,
			textvariable=self._status_var,
			font=("Arial", 7),
			bg="#f5eee3",
			fg="#7a4f14",
		)
		self.canvas.create_window(
			x + w / 2,
			y + h - 12,
			window=status_lbl,
			tags=(self.node_id,),
		)

		self._canvas_items += [self._body_rect, self._title_item]

	def build_inspector(self, parent: tk.Frame) -> None:
		self._init_state()

		tk.Label(parent, text="Timer interval (seconds)", font=("Arial", 9)).grid(
			row=0, column=0, sticky="w"
		)
		tk.Entry(parent, textvariable=self._interval_var, width=12, font=("Arial", 9)).grid(
			row=0, column=1, sticky="w", padx=(8, 0)
		)
		tk.Label(parent, textvariable=self._interval_error_var, font=("Arial", 8), fg="#cc0000").grid(
			row=1, column=0, columnspan=2, sticky="w", pady=(2, 0)
		)

		tk.Label(parent, text="Timer trigger count (0 = infinite)", font=("Arial", 9)).grid(
			row=2, column=0, sticky="w", pady=(6, 0)
		)
		tk.Entry(parent, textvariable=self._timer_target_var, width=12, font=("Arial", 9)).grid(
			row=2, column=1, sticky="w", padx=(8, 0), pady=(6, 0)
		)

		row = tk.Frame(parent)
		row.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(10, 0))
		tk.Button(row, text="Trigger Once", command=self._on_manual_trigger, width=12).pack(
			side="left"
		)
		tk.Button(row, textvariable=self._timer_btn_var, command=self._toggle_timer, width=12).pack(
			side="left", padx=(8, 0)
		)
		tk.Button(row, text="Reset Counters", command=self._reset_counters, width=14).pack(
			side="right"
		)

		tk.Label(parent, textvariable=self._counts_var, font=("Arial", 9), fg="#7a4f14").grid(
			row=4, column=0, columnspan=2, sticky="w", pady=(10, 0)
		)
		tk.Label(parent, textvariable=self._status_var, font=("Arial", 9), fg="#7a4f14").grid(
			row=5, column=0, columnspan=2, sticky="w", pady=(4, 0)
		)

		self._on_interval_text_changed()

	def _on_interval_text_changed(self, *_args) -> None:
		text = self._interval_var.get().strip()
		if not text:
			self._interval_error_var.set("Interval is required")
			return
		try:
			seconds = float(text)
		except Exception:
			self._interval_error_var.set("Interval must be a number")
			return
		if seconds <= 0.0:
			self._interval_error_var.set("Interval must be > 0 seconds")
			return
		self._interval_error_var.set("")

	def _parse_interval_seconds(self) -> float:
		try:
			seconds = float(self._interval_var.get().strip())
		except Exception:
			raise ValueError("interval must be numeric")
		if seconds <= 0.0:
			raise ValueError("interval must be > 0")
		return seconds

	def _parse_timer_target(self) -> int:
		try:
			target = int(round(float(self._timer_target_var.get().strip())))
		except Exception:
			raise ValueError("timer trigger count must be an integer")
		if target < 0:
			raise ValueError("timer trigger count must be >= 0")
		return target

	def _sync_output_cache(self) -> None:
		total = self._button_count + self._timer_count
		self._last_outputs = {
			"trigger": 0.0,
			"button_count": float(self._button_count),
			"timer_count": float(self._timer_count),
			"total_count": float(total),
		}

	def _refresh_counts_text(self) -> None:
		total = self._button_count + self._timer_count
		self._counts_var.set(f"btn={self._button_count}  tmr={self._timer_count}  all={total}")

	def _emit_trigger(self, source: str) -> None:
		if source == "button":
			self._button_count += 1
		else:
			self._timer_count += 1
			self._timer_run_sent += 1

		self._refresh_counts_text()

		total = self._button_count + self._timer_count
		pulse = {
			"trigger": 1.0,
			"button_count": float(self._button_count),
			"timer_count": float(self._timer_count),
			"total_count": float(total),
		}

		self.push_output(pulse)

		# Reset trigger back to low so future graph recomputes do not replay a stale high.
		self.push_output({
			"trigger": 0.0,
			"button_count": float(self._button_count),
			"timer_count": float(self._timer_count),
			"total_count": float(total),
		})

		self._sync_output_cache()

	@staticmethod
	def _truthy_trigger(value) -> bool:
		if isinstance(value, str):
			text = value.strip().lower()
			if text in ("", "0", "false", "no", "off"):
				return False
			return True
		try:
			return bool(float(value))
		except (TypeError, ValueError):
			return bool(value)

	def _on_manual_trigger(self) -> None:
		self._emit_trigger("button")
		self._status_var.set("manual trigger sent")
		self.set_status("manual", "#4f7f2a")

	def _cancel_timer_job(self) -> None:
		if self._timer_after_id is not None:
			try:
				self.canvas.after_cancel(self._timer_after_id)
			except Exception:
				pass
			self._timer_after_id = None

	def _stop_timer(self, reason: str = "timer stopped") -> None:
		self._timer_running = False
		self._cancel_timer_job()
		self._timer_btn_var.set("Start Timer")
		self._status_var.set(reason)
		self.set_status("idle", "#666666")

	def _schedule_next_tick(self) -> None:
		if not self._timer_running:
			return

		try:
			interval_s = self._parse_interval_seconds()
			target = self._parse_timer_target()
		except Exception as e:
			self._stop_timer(f"error: {e}")
			self.set_status("error", "#cc0000")
			return

		if target > 0 and self._timer_run_sent >= target:
			self._stop_timer("timer done")
			return

		delay_ms = max(1, int(round(interval_s * 1000.0)))
		self._timer_after_id = self.canvas.after(delay_ms, self._on_timer_tick)

	def _on_timer_tick(self) -> None:
		self._timer_after_id = None
		if not self._timer_running:
			return

		self._emit_trigger("timer")

		try:
			target = self._parse_timer_target()
		except Exception as e:
			self._stop_timer(f"error: {e}")
			self.set_status("error", "#cc0000")
			return

		if target > 0 and self._timer_run_sent >= target:
			self._stop_timer(f"timer done ({self._timer_run_sent}/{target})")
			return

		if target == 0:
			self._status_var.set(f"timer running ({self._timer_run_sent}/inf)")
		else:
			self._status_var.set(f"timer running ({self._timer_run_sent}/{target})")
		self.set_status("timer", "#4477aa")
		self._schedule_next_tick()

	def _toggle_timer(self) -> None:
		if self._timer_running:
			self._stop_timer("timer stopped")
			return

		try:
			self._parse_interval_seconds()
			target = self._parse_timer_target()
		except Exception as e:
			self._status_var.set(f"error: {e}")
			self.set_status("error", "#cc0000")
			return

		self._timer_running = True
		self._timer_run_sent = 0
		self._timer_btn_var.set("Stop Timer")
		self.set_status("timer", "#4477aa")
		if target == 0:
			self._status_var.set("timer armed (0=inf)")
		else:
			self._status_var.set(f"timer armed ({target})")
		self._schedule_next_tick()

	def _reset_counters(self) -> None:
		self._button_count = 0
		self._timer_count = 0
		self._timer_run_sent = 0
		self._sync_output_cache()
		self._refresh_counts_text()
		self._status_var.set("counters reset")
		self.set_status("reset", "#666666")
		self.push_output(dict(self._last_outputs))

	def compute(self, inputs: dict) -> dict:
		self._init_state()
		raw_trig = inputs.get("trig")
		trig_high = self._truthy_trigger(raw_trig)
		if trig_high and not self._input_latched_high:
			# Input-triggered one-shot behaves the same as pressing Trigger Once.
			self._emit_trigger("button")
			self._status_var.set("input trigger sent")
			self.set_status("input", "#4f7f2a")
			self._input_latched_high = True
		elif not trig_high:
			self._input_latched_high = False
		return dict(self._last_outputs)

	def get_params(self) -> dict:
		return {
			"interval_seconds": self._interval_var.get(),
			"timer_target": self._timer_target_var.get(),
			"button_count": self._button_count,
			"timer_count": self._timer_count,
		}

	def set_params(self, params: dict) -> None:
		self._init_state()

		self._interval_var.set(str(params.get("interval_seconds", "1.0")))
		self._timer_target_var.set(str(params.get("timer_target", "0")))

		try:
			self._button_count = int(params.get("button_count", 0))
		except Exception:
			self._button_count = 0
		try:
			self._timer_count = int(params.get("timer_count", 0))
		except Exception:
			self._timer_count = 0

		self._timer_run_sent = 0
		self._stop_timer("idle")
		self._sync_output_cache()
		self._refresh_counts_text()

	def on_destroy(self) -> None:
		self._stop_timer("destroyed")
		super().on_destroy()
