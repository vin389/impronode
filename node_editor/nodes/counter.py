import tkinter as tk

from node_editor.base_node import BaseNode
from node_editor.execution import ExecutionMode
from node_editor.pin_types import PinDef, PinSchema, PinType


class CounterNode(BaseNode):
	"""Count trigger pulses and expose the running integer count."""

	EXECUTION_MODE = ExecutionMode.SYNC
	NODE_TYPE = "counter"
	DISPLAY_NAME = "Counter"
	CATEGORY = "process"
	NODE_WIDTH = 180
	NODE_HEIGHT = 110

	def get_pin_schema(self) -> PinSchema:
		return PinSchema(
			inputs=[
				PinDef("trigger", PinType.TRIGGER, "trig", optional=True),
			],
			outputs=[
				PinDef("count", PinType.SCALAR, "count"),
			],
		)

	def _init_state(self) -> None:
		if hasattr(self, "_count"):
			return
		self._count = 0
		self._count_var = tk.StringVar(value="0")
		self._status_var = tk.StringVar(value="idle")

	def build_body(self) -> None:
		self._init_state()
		x, y, w, h = self.x, self.y, self.width, self.height

		self._body_rect = self.canvas.create_rectangle(
			x, y, x + w, y + h,
			fill="#eef4ff", outline="#4b71b0", width=2,
			tags=(self.node_id, "node_body"),
		)
		self._title_item = self.canvas.create_text(
			x + w / 2, y + 13,
			text=self.DISPLAY_NAME,
			font=("Arial", 9, "bold"), fill="#345382",
			tags=(self.node_id,),
		)

		count_lbl = tk.Label(
			self.canvas,
			textvariable=self._count_var,
			font=("Arial", 18, "bold"),
			bg="#eef4ff",
			fg="#2f4f79",
		)
		self.canvas.create_window(
			x + w / 2, y + h / 2 + 2,
			window=count_lbl,
			tags=(self.node_id,),
		)

		status_lbl = tk.Label(
			self.canvas,
			textvariable=self._status_var,
			font=("Arial", 7),
			bg="#eef4ff",
			fg="#345382",
		)
		self.canvas.create_window(
			x + w / 2, y + h - 12,
			window=status_lbl,
			tags=(self.node_id,),
		)

		self._canvas_items += [self._body_rect, self._title_item]

	def build_inspector(self, parent: tk.Frame) -> None:
		self._init_state()

		tk.Label(parent, text="Current count", font=("Arial", 9)).grid(
			row=0, column=0, sticky="w"
		)
		tk.Label(parent, textvariable=self._count_var, font=("Arial", 10, "bold"), fg="#2f4f79").grid(
			row=0, column=1, sticky="w", padx=(8, 0)
		)

		tk.Button(parent, text="Reset", width=12, command=self._reset_count).grid(
			row=1, column=1, sticky="e", pady=(10, 0)
		)
		tk.Label(parent, textvariable=self._status_var, font=("Arial", 9), fg="#345382").grid(
			row=1, column=0, sticky="w", pady=(10, 0)
		)

	@staticmethod
	def _truthy(value) -> bool:
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

	def _refresh_count_text(self) -> None:
		self._count_var.set(str(int(self._count)))

	def _reset_count(self) -> None:
		self._count = 0
		self._refresh_count_text()
		self._status_var.set("reset")
		self.set_status("reset", "#666666")
		# Push so downstream observers can update immediately after manual reset.
		self.push_output({"count": float(self._count)})

	def compute(self, inputs: dict) -> dict:
		self._init_state()

		if self._truthy(inputs.get("trigger")):
			self._count += 1
			self._refresh_count_text()
			self._status_var.set("counted")
			self.set_status("ok", "#339966")

		return {"count": float(self._count)}

	def get_params(self) -> dict:
		return {
			"count": self._count,
		}

	def set_params(self, params: dict) -> None:
		self._init_state()
		try:
			self._count = int(params.get("count", 0))
		except Exception:
			self._count = 0
		self._refresh_count_text()
		self._status_var.set("loaded")
