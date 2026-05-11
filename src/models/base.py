from dataclasses import dataclass, field
import math as m

@dataclass
class Pos:
    r: float = 0. # m
    phi: float = 0. # radians

    def to_cartesian(self) -> tuple[float, float]:
        return self.r * m.cos(self.phi), self.r * m.sin(self.phi)

@dataclass
class Anomaly:
    radius: float # m
    center: Pos
    conductivity: float  # S/m (Simens per meter)

@dataclass
class Trunk:
    radius: float
    base_conductivity: float
    anomalies: list[Anomaly] = field(default_factory=list)
