"""
Tree trunk domain model.
"""

from dataclasses import dataclass, field
from typing import List

from .shape import Shape
from .anomaly import Anomaly


@dataclass
class Trunk:
    """Tree trunk with base properties and anomalies."""
    radius: float
    base_conductivity: float
    anomalies: List[Anomaly] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON."""
        return {
            "radius": self.radius,
            "base_conductivity": self.base_conductivity,
            "anomalies": [
                {
                    "shape": a.shape.to_dict(),
                    "conductivity": a.conductivity
                }
                for a in self.anomalies
            ]
        }