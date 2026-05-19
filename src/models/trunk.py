from typing import Literal

from .anomaly import Anomaly
from .base import Serializable
from .pos import Pos
from .shape import Circle


class Trunk(Serializable):
    def __init__(self, base: Anomaly, anomalies: list[Anomaly]):
        self.base = base
        self.anomalies = anomalies

    @classmethod
    def from_dict(cls, data: dict) -> "Trunk":
        base_data = data.get("base")
        base = Anomaly.from_dict(base_data)

        anomalies_data = data.get("anomalies")
        anomalies = [Anomaly.from_dict(a_data) for a_data in anomalies_data]
        return Trunk(
            base=base,
            anomalies=anomalies
        )

    def to_dict(self) -> dict:
        return {
            "base": self.base.to_dict(),
            "anomalies": [a.to_dict() for a in self.anomalies]
        }

    def get_conductivity(self, pos: Pos, mode: Literal["Sum", "Max"] = "Max", add_base_cond: bool = False) -> float:
        if not self.base.contains(pos):
            return 0.
        all_conductivities = [
            a.get_conductivity(pos) for a in self.anomalies if a.contains(pos)
        ]

        if not all_conductivities:
            return self.base.get_conductivity(pos)

        base_conductivity = self.base.get_conductivity(pos) if add_base_cond else 0.
        if mode == "Sum":
            return sum(all_conductivities) + base_conductivity
        if mode == "Max":
            return max(all_conductivities) + base_conductivity
        raise NotImplementedError(f"Unknown mode: {mode}")
    
    def get_bounds(self, n_points: int = 360) -> tuple[float, float, float, float]:
        return self.base.shape.get_bounds(n_points)


class SimpleTrunk:
    """Factory class to generate simple circular trunks."""

    @classmethod
    def create(cls, radius: float, base_conductivity: float, anomalies: list[Anomaly] = None) -> Trunk:
        shape = Circle(Pos(0, 0), radius)
        base = Anomaly(shape, base_conductivity, "Constant")
        return Trunk(base=base, anomalies=anomalies if anomalies else [])
