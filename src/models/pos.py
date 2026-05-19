import math
from dataclasses import dataclass

from .base import Serializable


@dataclass
class Pos(Serializable):
    """Polar position coordinates."""

    r: float = 0.0  # m
    phi: float = 0.0  # radians

    def to_dict(self):
        return {"r": self.r, "phi": self.phi}

    @classmethod
    def from_dict(cls, data):
        r, phi = data.get("r"), data.get("phi")
        if r is None or phi is None:
            raise ValueError("")
        return Pos(r=r, phi=phi)

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
