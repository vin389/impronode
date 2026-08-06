# data_flow_engine.py

import tkinter as tk
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from node_editor.execution import ExecutionMode

if TYPE_CHECKING:
    from .base_node import BaseNode


class DataFlowEngine:
    """
    Responsibilities:
      - Maintain the node/link graph structure
      - Perform topological sorting
      - Schedule compute() according to ExecutionMode
      - Safely route STREAMING node push_output() calls back to the main thread
      - Detect cyclic dependencies
    """

    STREAMING_POLL_MS = 33   # ~30fps, polling interval for the queue via tk.after()

    def __init__(self, tk_root: tk.Tk):
        self.root = tk_root

        # Graph structure
        # nodes:  node_id -> BaseNode
        # links:  list of {"src_node", "src_pin", "dst_node", "dst_pin"}
        self.nodes: dict[str, "BaseNode"] = {}
        self.links: list[dict]            = []

        # Thread pool used by BACKGROUND nodes
        self._executor = ThreadPoolExecutor(max_workers=4)

        # Output queue for STREAMING nodes
        # push_output() is enqueued from worker threads and dequeued on the main thread via tk.after()
        self._stream_queue: queue.Queue = queue.Queue()
        self._poll_scheduled = False

        # Global output cache
        self._node_outputs: dict[str, dict] = {}

    # ══ Graph structure management ════════════════════════════════

    def add_node(self, node: "BaseNode") -> None:
        self.nodes[node.node_id] = node
        # Inject callbacks so STREAMING nodes can call push_output()
        node._on_output_ready = self._on_streaming_output
        node._request_downstream = self._trigger_from

    def remove_node(self, node_id: str) -> None:
        node = self.nodes.pop(node_id, None)
        if node:
            node.on_destroy()
        self._node_outputs.pop(node_id, None)             # Clear cache
        self.links = [
            lk for lk in self.links
            if lk["src_node"] != node_id and lk["dst_node"] != node_id
        ]

    def add_link(self, src_node: str, src_pin: str,
                dst_node: str, dst_pin: str) -> bool:
        link = {"src_node": src_node, "src_pin": src_pin,
                "dst_node": dst_node, "dst_pin": dst_pin}
        self.links.append(link)
        order = self._topological_sort()
        if order is None:
            self.links.remove(link)
            return False
        destination = self.nodes.get(dst_node)
        callback = getattr(destination, "on_input_link_changed", None)
        if callable(callback):
            callback(dst_pin, True)
        self._trigger_from(src_node, order)   # Use src_node as the trigger root
        return True

    def remove_link(self, src_node: str, src_pin: str,
                    dst_node: str, dst_pin: str) -> None:
        self.links = [
            lk for lk in self.links
            if not (lk["src_node"] == src_node and lk["src_pin"] == src_pin
                    and lk["dst_node"] == dst_node and lk["dst_pin"] == dst_pin)
        ]
        # Notify the destination node that its upstream inputs changed
        if dst_node in self.nodes:
            destination = self.nodes[dst_node]
            destination.on_upstream_changed()
            callback = getattr(destination, "on_input_link_changed", None)
            if callable(callback):
                still_connected = any(
                    lk["dst_node"] == dst_node and lk["dst_pin"] == dst_pin
                    for lk in self.links
                )
                callback(dst_pin, still_connected)

    # ══ Topological sorting ═══════════════════════════════════════

    def _topological_sort(self) -> list[str] | None:
        """
        Kahn's algorithm.
        Return a sorted list of node IDs; return None if a cycle exists.
        """
        in_degree: dict[str, int] = {nid: 0 for nid in self.nodes}
        adjacency: dict[str, list[str]] = {nid: [] for nid in self.nodes}

        for lk in self.links:
            if not self._is_causal_link(lk):
                continue
            src, dst = lk["src_node"], lk["dst_node"]
            if src in adjacency:
                adjacency[src].append(dst)
            if dst in in_degree:
                in_degree[dst] += 1

        ready = [nid for nid, deg in in_degree.items() if deg == 0]
        order = []

        while ready:
            nid = ready.pop()
            order.append(nid)
            for neighbor in adjacency[nid]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    ready.append(neighbor)

        if len(order) != len(self.nodes):
            return None     # Cycle detected
        return order

    # ══ Data flow triggering ═══════════════════════════════════════

    def trigger_all(self) -> None:
        """Recompute the entire graph starting from the most upstream nodes."""
        order = self._topological_sort()
        if order is None:
            return
        self._execute_in_order(order)

    def _trigger_from(self, start_node_id: str,
                      order: list[str] | None = None) -> None:
        """
        Recompute only start_node and all of its downstream nodes.
        If order is already known, pass it in to avoid sorting again.
        """
        if order is None:
            order = self._topological_sort()
        if order is None:
            return

        # Find all downstream nodes from start_node, including itself
        affected = self._downstream_set(start_node_id)
        # Filter by topological order
        suborder = [nid for nid in order if nid in affected]
        self._execute_in_order(suborder)

    def _downstream_set(self, start_id: str) -> set[str]:
        """
        Use BFS to find all downstream nodes from start_id, including itself.

        Delayed links are included here so nodes that consume delayed inputs
        still get recomputed when their upstream source publishes new data.
        Topological sort remains causal-only, so cycle rejection behaviour is
        unchanged.
        """
        visited = set()
        queue_  = [start_id]
        while queue_:
            cur = queue_.pop()
            if cur in visited:
                continue
            visited.add(cur)
            for lk in self.links:
                if lk["src_node"] == cur and lk["dst_node"] not in visited:
                    queue_.append(lk["dst_node"])
        return visited

    def _is_causal_link(self, link: dict) -> bool:
        """
        Delayed feedback links carry cached values from a previous pass and must
        not participate in cycle detection or same-pass scheduling.
        """
        destination = self.nodes.get(link["dst_node"])
        delayed_pins = getattr(destination, "DELAYED_INPUT_PINS", ())
        return link["dst_pin"] not in delayed_pins

    # ══ Execution scheduling (core) ═══════════════════════════════

    def _execute_in_order(self, order: list[str]) -> None:
        """
        Execute nodes one by one in topological order.
        The next node only runs after the current one finishes, ensuring downstream nodes receive the correct upstream output.

        SYNC       -> call compute() directly on the main thread
        BACKGROUND -> submit to the thread pool, then continue downstream on the main thread when finished
        STREAMING  -> skip (driven by _poll_stream_queue)
        """
        self._execute_step(order, 0, {})

    def _execute_step(self, order, idx, outputs):
        if idx >= len(order):
            return

        node_id = order[idx]
        node    = self.nodes.get(node_id)
        if node is None:
            self._execute_step(order, idx + 1, outputs)
            return

        if node.EXECUTION_MODE == ExecutionMode.STREAMING:
            self._execute_step(order, idx + 1, outputs)
            return

        inputs = self._gather_inputs(node_id, outputs)

        if node.EXECUTION_MODE == ExecutionMode.SYNC:
            result = self._safe_compute(node, inputs)
            self._handle_compute_result(order, idx, outputs, node_id, result)

        elif node.EXECUTION_MODE == ExecutionMode.BACKGROUND:
            def _bg_task():
                return self._safe_compute(node, inputs)

            def _on_done(future):
                result = future.result()
                self.root.after(
                    0,
                    lambda: self._handle_compute_result(
                        order, idx, outputs, node_id, result
                    ),
                )

            future = self._executor.submit(_bg_task)
            future.add_done_callback(_on_done)

    def _handle_compute_result(self, order, idx, outputs, node_id, result) -> None:
        public_outputs, control = self._split_control_metadata(result)

        if not control["preserve_cache"]:
            outputs[node_id] = public_outputs
            self._node_outputs[node_id] = public_outputs
        elif node_id in self._node_outputs:
            outputs[node_id] = self._node_outputs[node_id]

        if control["skip_downstream"]:
            skip_nodes = self._downstream_set(node_id)
            next_idx = idx + 1
            while next_idx < len(order) and order[next_idx] in skip_nodes:
                next_idx += 1
            self._execute_step(order, next_idx, outputs)
            return

        self._execute_step(order, idx + 1, outputs)

    @staticmethod
    def _split_control_metadata(result: dict | None) -> tuple[dict, dict[str, bool]]:
        payload = dict(result or {})
        skip_downstream = bool(payload.pop("_skip_downstream", False))
        preserve_cache = bool(payload.pop("_preserve_cache", False))
        return payload, {
            "skip_downstream": skip_downstream,
            "preserve_cache": preserve_cache,
        }

    def _gather_inputs(self, node_id: str,
                       outputs: dict[str, dict]) -> dict:
        """
        Build this node's inputs dict from completed upstream outputs according to the link structure.
        """
        inputs = {}
        for lk in self.links:
            if lk["dst_node"] != node_id:
                continue
#            src_output = outputs.get(lk["src_node"], {})
#            value = src_output.get(lk["src_pin"])
            src_output = outputs.get(lk["src_node"]) \
                     or self._node_outputs.get(lk["src_node"], {})
#            inputs[lk["dst_pin"]] = value
            inputs[lk["dst_pin"]] = src_output.get(lk["src_pin"])
        return inputs

    @staticmethod
    def _safe_compute(node: "BaseNode", inputs: dict) -> dict:
        """Wrap compute() in try/except so a single node error does not break the full data flow."""
        try:
            return node.compute(inputs) or {}
        except Exception as e:
            node.set_status(f"error: {e}", color="#cc0000")
            return {}

    # ══ STREAMING data flow (push mode) ═══════════════════════════

    def _on_streaming_output(self, node_id: str, outputs: dict) -> None:
        """
        Triggered when a STREAMING node calls push_output().
        This may be invoked from a worker thread, so it only enqueues data;
        the actual data-flow driving happens in _poll_stream_queue() on the main thread.
        """
        self._stream_queue.put((node_id, outputs))
        if not self._poll_scheduled:
            self._poll_scheduled = True
            self.root.after(0, self._poll_stream_queue)

    def _poll_stream_queue(self) -> None:
        processed = 0
        while not self._stream_queue.empty() and processed < 3:
            node_id, outputs = self._stream_queue.get_nowait()

            # strip internal metadata keys (prefixed with _)
            pin_outputs = {k: v for k, v in outputs.items()
                        if not k.startswith("_")}

            order = self._topological_sort()
            if order:
                downstream = [
                    nid for nid in order
                    if nid in self._downstream_set(node_id)
                    and nid != node_id
                ]
                self._node_outputs[node_id] = pin_outputs
                self._execute_step(
                    downstream, 0, {node_id: pin_outputs})
            processed += 1

        has_streaming = any(
            n.EXECUTION_MODE == ExecutionMode.STREAMING
            and n._is_running
            for n in self.nodes.values()
        )
        if has_streaming or not self._stream_queue.empty():
            self.root.after(
                self.STREAMING_POLL_MS, self._poll_stream_queue)
        else:
            self._poll_scheduled = False
    # ══ Lifecycle ══════════════════════════════════════════════════

    def shutdown(self) -> None:
        """Called when the app closes to clean up all resources."""
        for node in self.nodes.values():
            node.on_destroy()
        self._executor.shutdown(wait=False)
