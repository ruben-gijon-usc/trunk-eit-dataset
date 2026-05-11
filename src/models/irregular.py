from abc import ABC
from dataclasses import dataclass
from typing import List


@dataclass
class Shape(ABC):
    pass


@dataclass
class IrregularShape:
    radius: float
    vertices: List[float]

    def __post_init__(self):
        if self.radius <= 0:
            raise ValueError("")
        for v in self.vertices:
            if not 0 < v <= 1:
                raise ValueError("")

