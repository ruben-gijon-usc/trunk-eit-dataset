from dataclasses import dataclass, field
from typing import Literal

from .anomaly import Anomaly
from .pos import Pos


@dataclass
class Trunk:
    """Tree trunk with base properties and anomalies."""

    radius: float
    base_conductivity: float
    anomalies: list[Anomaly] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON."""
        return {
            "radius": self.radius,
            "base_conductivity": self.base_conductivity,
            "anomalies": [a.to_dict() for a in self.anomalies],
        }

    def get_conductivity(self, pos: Pos, mode: Literal["Sum", "Max"] = "Max") -> float:
        if pos.r > self.radius:
            return 0
        all_conductivities = [
            anomaly.conductivity for anomaly in self.anomalies if anomaly.contains(pos)
        ]
        if not all_conductivities:
            return self.base_conductivity
        if mode == "Sum":
            return sum(all_conductivities)
        if mode == "Max":
            return max(all_conductivities)
        raise ValueError(f"Unknown mode: {mode}")
