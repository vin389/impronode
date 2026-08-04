import tkinter as tk

import cv2
import numpy as np

from node_editor.base_node import BaseNode
from node_editor.pin_types import PinSchema, PinDef, PinType
from node_editor.execution import ExecutionMode


class R44ToVecsNode(BaseNode):
    """
    Convert a 4x4 rigid transform matrix into (rvec, tvec).

    Input:
      r44: 4x4 transform matrix

    Outputs:
      rvec: Rodrigues vector from r44[0:3, 0:3], shape (3, 1)
      tvec: translation from r44[0:3, 3], shape (3, 1)
    """

    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE = "r44_to_vecs"
    DISPLAY_NAME = "R44-to-Vecs"
    CATEGORY = "process"
    NODE_WIDTH = 190
    NODE_HEIGHT = 95

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[
                PinDef("r44", PinType.ARRAY, "r44", shape=(4, 4), dtype="float64"),
            ],
            outputs=[
                PinDef("rvec", PinType.ARRAY, "rvec", shape=(3, 1), dtype="float64"),
                PinDef("tvec", PinType.ARRAY, "tvec", shape=(3, 1), dtype="float64"),
            ],
        )

    def build_body(self) -> None:
        x, y, w, h = self.x, self.y, self.width, self.height

        self._body_rect = self.canvas.create_rectangle(
            x, y, x + w, y + h,
            fill="#e9f6ea", outline="#5ea96d", width=2,
            tags=(self.node_id, "node_body"),
        )
        self._title_item = self.canvas.create_text(
            x + w / 2, y + 13,
            text=self.DISPLAY_NAME,
            font=("Arial", 9, "bold"), fill="#2b6b39",
            tags=(self.node_id,),
        )

        hint = tk.Label(
            self.canvas,
            text="r44 (4x4) -> rvec, tvec",
            font=("Arial", 8), bg="#e9f6ea", fg="#3d6f48",
        )
        self.canvas.create_window(
            x + w / 2, y + 38, window=hint,
            tags=(self.node_id,),
        )

        self._status_var = tk.StringVar(value="waiting for r44")
        status_lbl = tk.Label(
            self.canvas,
            textvariable=self._status_var,
            font=("Arial", 7), bg="#e9f6ea", fg="#3d6f48",
        )
        self.canvas.create_window(
            x + w / 2, y + h - 12, window=status_lbl,
            tags=(self.node_id,),
        )

        self._canvas_items += [self._body_rect, self._title_item]

    @staticmethod
    def _coerce_r44(value) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64)
        if arr.size < 16:
            raise ValueError("r44 needs at least 16 values")
        return arr.reshape(-1)[:16].reshape(4, 4)

    def compute(self, inputs: dict) -> dict:
        raw = inputs.get("r44")
        if raw is None:
            self._status_var.set("missing r44")
            self.set_status("missing", "#cc6666")
            return {}

        try:
            r44 = self._coerce_r44(raw)
            R = r44[0:3, 0:3]
            tvec = r44[0:3, 3].reshape(3, 1)
            rvec, _ = cv2.Rodrigues(R)

            rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
            tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)

            self._status_var.set("ok")
            self.set_status("ok", "#55aa55")
            return {
                "rvec": rvec,
                "tvec": tvec,
            }
        except Exception as e:
            self._status_var.set(f"error: {e}")
            self.set_status("error", "#cc0000")
            return {}


class VecsToR44Node(BaseNode):
    """
    Convert (rvec, tvec) into a 4x4 rigid transform matrix.

    Inputs:
      rvec: Rodrigues rotation vector
      tvec: translation vector

    Output:
      r44: 4x4 transform matrix
    """

    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE = "vecs_to_r44"
    DISPLAY_NAME = "VecsToR44"
    CATEGORY = "process"
    NODE_WIDTH = 190
    NODE_HEIGHT = 95

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[
                PinDef("rvec", PinType.ARRAY, "rvec", shape=(3, 1), dtype="float64"),
                PinDef("tvec", PinType.ARRAY, "tvec", shape=(3, 1), dtype="float64"),
            ],
            outputs=[
                PinDef("r44", PinType.ARRAY, "r44", shape=(4, 4), dtype="float64"),
            ],
        )

    def build_body(self) -> None:
        x, y, w, h = self.x, self.y, self.width, self.height

        self._body_rect = self.canvas.create_rectangle(
            x, y, x + w, y + h,
            fill="#eaf2ff", outline="#5b86d6", width=2,
            tags=(self.node_id, "node_body"),
        )
        self._title_item = self.canvas.create_text(
            x + w / 2, y + 13,
            text=self.DISPLAY_NAME,
            font=("Arial", 9, "bold"), fill="#2d4f8f",
            tags=(self.node_id,),
        )

        hint = tk.Label(
            self.canvas,
            text="rvec, tvec -> r44 (4x4)",
            font=("Arial", 8), bg="#eaf2ff", fg="#395c8f",
        )
        self.canvas.create_window(
            x + w / 2, y + 38, window=hint,
            tags=(self.node_id,),
        )

        self._status_var = tk.StringVar(value="waiting for rvec,tvec")
        status_lbl = tk.Label(
            self.canvas,
            textvariable=self._status_var,
            font=("Arial", 7), bg="#eaf2ff", fg="#395c8f",
        )
        self.canvas.create_window(
            x + w / 2, y + h - 12, window=status_lbl,
            tags=(self.node_id,),
        )

        self._canvas_items += [self._body_rect, self._title_item]

    @staticmethod
    def _coerce_rvec(value) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size < 3:
            raise ValueError("rvec needs at least 3 values")
        return arr[:3].reshape(3, 1)

    @staticmethod
    def _coerce_tvec(value) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.size < 3:
            raise ValueError("tvec needs at least 3 values")
        return arr[:3].reshape(3, 1)

    def compute(self, inputs: dict) -> dict:
        raw_rvec = inputs.get("rvec")
        raw_tvec = inputs.get("tvec")
        if raw_rvec is None or raw_tvec is None:
            self._status_var.set("missing rvec or tvec")
            self.set_status("missing", "#cc6666")
            return {}

        try:
            rvec = self._coerce_rvec(raw_rvec)
            tvec = self._coerce_tvec(raw_tvec)
            R, _ = cv2.Rodrigues(rvec)

            r44 = np.eye(4, dtype=np.float64)
            r44[0:3, 0:3] = np.asarray(R, dtype=np.float64)
            r44[0:3, 3] = tvec.reshape(3)

            self._status_var.set("ok")
            self.set_status("ok", "#55aa55")
            return {
                "r44": r44,
            }
        except Exception as e:
            self._status_var.set(f"error: {e}")
            self.set_status("error", "#cc0000")
            return {}


class CoordSys3PointsNode(BaseNode):
    """
    Define a coordinate system from 3 points A, B, C.

    Input:
      objectPoints: (3, 3) array where
        A = objectPoints[0, :]
        B = objectPoints[1, :]
        C = objectPoints[2, :]

    Outputs:
      r44: inverse of transform matrix built from axes and origin B
           {point in global coord} = r44 @ {point in local coord}
      r44_inv: forward transform matrix 
           {point in local coord} = r44_inv @ {point in global coord}
      angleB_deg: angle A-B-C in degrees (should be very close to 90
                  if A-B-C are orthogonal)
      (in opencv, rvec is r44_inv[0:3, 0:3] in Rodrigues form, tvec is r44_inv[0:3, 3])
    """

    EXECUTION_MODE = ExecutionMode.SYNC
    NODE_TYPE = "coord_sys_3points"
    DISPLAY_NAME = "CoordSys-3Points"
    CATEGORY = "process"
    NODE_WIDTH = 210
    NODE_HEIGHT = 105

    def get_pin_schema(self) -> PinSchema:
        return PinSchema(
            inputs=[
                PinDef("objectPoints", PinType.ARRAY, "objPts", shape=(3, 3), dtype="float64"),
            ],
            outputs=[
                PinDef("r44", PinType.ARRAY, "r44", shape=(4, 4), dtype="float64"),
                PinDef("r44_inv", PinType.ARRAY, "r44_inv", shape=(4, 4), dtype="float64"),
                PinDef("angleB_deg", PinType.SCALAR, "angleB"),
            ],
        )

    def build_body(self) -> None:
        x, y, w, h = self.x, self.y, self.width, self.height

        self._body_rect = self.canvas.create_rectangle(
            x, y, x + w, y + h,
            fill="#f7effa", outline="#b07ab3", width=2,
            tags=(self.node_id, "node_body"),
        )
        self._title_item = self.canvas.create_text(
            x + w / 2, y + 13,
            text=self.DISPLAY_NAME,
            font=("Arial", 9, "bold"), fill="#6d3f71",
            tags=(self.node_id,),
        )

        hint = tk.Label(
            self.canvas,
            text="A,B,C -> r44(inv), angleB",
            font=("Arial", 8), bg="#f7effa", fg="#6d4a71",
        )
        self.canvas.create_window(
            x + w / 2, y + 40, window=hint,
            tags=(self.node_id,),
        )

        self._status_var = tk.StringVar(value="waiting for objectPoints")
        status_lbl = tk.Label(
            self.canvas,
            textvariable=self._status_var,
            font=("Arial", 7), bg="#f7effa", fg="#6d4a71",
        )
        self.canvas.create_window(
            x + w / 2, y + h - 12, window=status_lbl,
            tags=(self.node_id,),
        )

        self._canvas_items += [self._body_rect, self._title_item]

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n <= 1e-12:
            raise ValueError("zero-length vector encountered")
        return v / n

    @staticmethod
    def _coerce_object_points(value) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64)
        if arr.size < 9:
            raise ValueError("objectPoints needs at least 9 values")
        pts = arr.reshape(-1)[:9].reshape(3, 3)
        return pts

    def compute(self, inputs: dict) -> dict:
        raw = inputs.get("objectPoints")
        if raw is None:
            self._status_var.set("missing objectPoints")
            self.set_status("missing", "#cc6666")
            return {}

        try:
            object_points = self._coerce_object_points(raw)
            A = object_points[0, :]
            B = object_points[1, :]
            C = object_points[2, :]

            v_bc = C - B
            v_ba = A - B

            vx = self._normalize(v_bc)
            vz = self._normalize(v_ba)

            dot = float(np.clip(np.dot(vx, vz), -1.0, 1.0))
            angle_b_deg = float(np.degrees(np.arccos(dot)))

            vy = np.cross(vz, vx)
            vy = self._normalize(vy)
            vz = np.cross(vx, vy)

            vx = self._normalize(vx)
            vy = self._normalize(vy)
            vz = self._normalize(vz)

            r44 = np.eye(4, dtype=np.float64)
            r44[0:3, 0] = vx
            r44[0:3, 1] = vy
            r44[0:3, 2] = vz
            r44[0:3, 3] = B
            r44[3, :] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

            r44_inv = np.linalg.inv(r44)

            self._status_var.set(f"ok: angle={angle_b_deg:.3f} deg")
            self.set_status("ok", "#55aa55")
            return {
                "r44": r44,
                "r44_inv": r44_inv,
                "angleB_deg": angle_b_deg,
            }
        except Exception as e:
            self._status_var.set(f"error: {e}")
            self.set_status("error", "#cc0000")
            return {}
