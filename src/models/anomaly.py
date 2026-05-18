from typing import Literal
from .pos import Pos
from .shape import Shape, CircularShape


class Anomaly:
    def __init__(self, shape: Shape, conductivity: float):
        self.shape = shape
        self.conductivity = conductivity

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON."""
        return {"shape": self.shape.to_dict(), "conductivity": self.conductivity}

    def contains(self, pos: Pos) -> bool:
        return self.shape.contains(pos)
    
    def get_conductivity(self, pos: Pos, mode: Literal["Constant", "Linear"] = "Constant") -> float:
        if not isinstance(self.shape, CircularShape):
            return self.conductivity if self.contains(pos) else 0.

        r = self.shape.get_relative_radius(pos)
        if r > 1: # is not contained. I use radius bc is more efficent using r than calling contains method and after that recalculating r
            return 0.
        if mode == "Constant":
            return self.conductivity
        if mode == "Linear":
            return (1 - r) * self.conductivity
        raise ValueError(f"Unkonw mode: {mode}")
