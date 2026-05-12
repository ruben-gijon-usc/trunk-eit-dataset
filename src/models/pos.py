from dataclasses import dataclass
import math


@dataclass
class Pos:
    """Polar position coordinates."""
    r: float = 0.0  # m
    phi: float = 0.0  # radians

    def to_cartesian(self) -> tuple[float, float]:
        return self.r * math.cos(self.phi), self.r * math.sin(self.phi)
