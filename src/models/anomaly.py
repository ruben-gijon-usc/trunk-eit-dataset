from .pos import Pos
from .shape import Shape


class Anomaly:
    def __init__(self, shape: Shape, conductivity: float):
        self.shape = shape
        self.conductivity = conductivity

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON."""
        return {"shape": self.shape.to_dict(), "conductivity": self.conductivity}

    def contains(self, pos: Pos) -> bool:
        return self.shape.contains(pos)
