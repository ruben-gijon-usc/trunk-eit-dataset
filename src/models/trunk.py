from dataclasses import dataclass, field
from .anomaly import Anomaly


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
