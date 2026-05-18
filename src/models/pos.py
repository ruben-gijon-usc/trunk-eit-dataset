import math
from dataclasses import dataclass


@dataclass
class Pos:
    """Polar position coordinates."""
    r: float = 0.0  # m
    phi: float = 0.0  # radians

    @classmethod
    def from_cartesian(cls, x: float, y: float) -> "Pos":
        """Alternative constructor."""
        r = math.hypot(x, y)
        phi = math.atan2(y, x)
        return cls(r=r, phi=phi)

    @property
    def x(self) -> float:
        return self.r * math.cos(self.phi)

    @property
    def y(self) -> float:
        return self.r * math.sin(self.phi)

    def to_cartesian(self) -> tuple[float, float]:
        return self.x, self.y
