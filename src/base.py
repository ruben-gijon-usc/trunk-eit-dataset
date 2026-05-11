from dataclasses import dataclass, field
from typing import List


@dataclass
class Pos:
    r: float = 0. # m
    phi: float = 0. # radians

@dataclass
class Anomaly:
    radius: float # m
    center: Pos
    conductivity: float  # S/m (Simens per meter)

@dataclass
class Trunk:
    radius: float
    base_conductivity: float
    anomalies: List[Anomaly] = field(default_factory=list)
