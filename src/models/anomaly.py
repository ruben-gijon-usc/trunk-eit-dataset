import math
from typing import Literal

from .base import Serializable
from .pos import Pos
from .shape import Shape, ShapeFactory


class Anomaly(Serializable):
    def __init__(self, shape: Shape, conductivity: float, propagation_fn: Literal["Constant", "Linear", "Cos"] = "Constant"):
        self.shape = shape
        self.conductivity = conductivity
        self.propagation_fn = propagation_fn

    @classmethod
    def from_dict(cls, data: dict):
        shape_data = data.get("shape")
        shape = ShapeFactory.from_dict(shape_data)

        conductivity = data.get("conductivity")
        propagation_fn = data.get("propagation_fn", "Constant")
        return Anomaly(shape, conductivity, propagation_fn)

    def to_dict(self) -> dict:
        return {"shape": self.shape.to_dict(), "conductivity": self.conductivity, "propagation_fn": self.propagation_fn}

    def contains(self, pos: Pos) -> bool:
        return self.shape.contains(pos)

    def get_conductivity(self, pos: Pos) -> float:
        # is not contained. I use radius bc is more efficent using r than calling contains method and after that recalculating r
        r = self.shape.get_relative_radius(pos)
        if r > 1:
            return 0.0
        if self.propagation_fn == "Constant":
            return self.conductivity
        if self.propagation_fn == "Linear":
            return (1 - r) * self.conductivity
        if self.propagation_fn == "Cos":
            return math.cos(r * math.pi / 2.) * self.conductivity
        raise ValueError(f"Unkonw mode: {self.propagation_fn}")
