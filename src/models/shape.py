import math
from abc import abstractmethod
from typing import Any

from .base import Serializable
from .pos import Pos


class Shape(Serializable):
    """Abstract base class for geometric shapes."""

    SHAPE_TYPE: str = ""

    def __init__(self, center: Pos):
        if not isinstance(center, Pos):
            raise TypeError(f"Center type should be pos: {center}")
        self.center = center

    @property
    def shape_type(self) -> str:
        """Return shape type identifier."""
        return self.SHAPE_TYPE

    @abstractmethod
    def get_radius(self, theta: float) -> float:
        """Get radius on a certain direction (theta in radians)."""
        pass

    def get_relative_pos(self, pos: Pos) -> Pos:
        """Calculate the position relative to the shape's center in polar coordinates."""
        cx, cy = self.center.to_cartesian()
        dx = pos.x - cx
        dy = pos.y - cy
        return Pos.from_cartesian(dx, dy)

    def get_relative_radius(self, pos: Pos) -> float:
        """Returns relative radius.
        If 0 <= r < 1: is contained on shape.
        if r > 1: is outside shape"""
        pos_rel = self.get_relative_pos(pos)
        theta = pos_rel.phi
        return pos_rel.r / self.get_radius(theta)

    def contains(self, pos: Pos) -> bool:
        """Check if a position is inside the shape using polar coordinates."""
        pos_rel = self.get_relative_pos(pos)
        return pos_rel.r <= self.get_radius(pos_rel.phi)

    def get_points(self, n_points: int) -> list[tuple[float, float]]:
        """Generate boundary points based on the radius at different angles."""
        cx, cy = self.center.to_cartesian()

        points = []
        for i in range(n_points):
            theta = 2 * math.pi * i / n_points
            r = self.get_radius(theta)

            d_x = r * math.cos(theta)
            d_y = r * math.sin(theta)

            points.append((cx + d_x, cy + d_y))
        return points

    def get_bounds(self, n_points: int = 360) -> tuple[float, float, float, float]:
        points = self.get_points(n_points)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        return min(xs), max(xs), min(ys), max(ys)

    @abstractmethod
    def get_area(self) -> float:
        pass


class Circle(Shape):
    """Circle shape."""

    SHAPE_TYPE = "circle"

    def __init__(self, center: Pos, radius: float):
        super().__init__(center)
        if radius <= 0:
            raise ValueError(f"Radius should be greater than zero: {radius}")
        self.radius = radius

    @classmethod
    def from_dict(cls, data):
        center_data, radius = data.get("center"), data.get("radius")
        if center_data is None or radius is None:
            raise ValueError("Missing required keys for Circle: 'cx', 'cy', or 'radius'")
        center = Pos.from_dict(center_data)
        return Circle(center, radius)

    def get_radius(self, theta: float) -> float:
        return self.radius

    def to_dict(self) -> dict:
        return {"shape_type": self.shape_type, "center": self.center.to_dict(), "radius": self.radius}

    def get_area(self) -> float:
        return math.pi * self.radius * self.radius

    def get_bounds(self, n_points: int = None) -> tuple[float, float, float, float]:
        cx, cy = self.center.to_cartesian()
        return (
            cx - self.radius,
            cx + self.radius,
            cy - self.radius,
            cy + self.radius
        )

class Ellipse(Shape):
    """Ellipse shape with rotation."""

    SHAPE_TYPE = "ellipse"

    def __init__(self, center: Pos, rx: float, ry: float, rotation: float = 0.0):
        super().__init__(center)
        if rx <= 0 or ry <= 0:
            raise ValueError(f"Both rx and ry should be greater than 0. {rx}, {ry}")
        self.rx = rx
        self.ry = ry
        self.rotation = rotation

    @classmethod
    def from_dict(cls, data):
        center_data, rx, ry, rotation = data.get("center"), data.get("rx"), data.get("ry"), data.get("rotation")
        if center_data is None or rx is None or ry is None or rotation is None:
            raise ValueError("Missing required keys for Ellipse: 'cx', 'cy', 'rx', 'ry', 'rotation'")
        center = Pos.from_dict(center_data)
        return Ellipse(center, rx, ry, rotation)

    def get_radius(self, theta: float) -> float:
        alpha = theta - self.rotation
        cos_a = math.cos(alpha)
        sin_a = math.sin(alpha)

        denom = math.sqrt((self.ry * cos_a) ** 2 + (self.rx * sin_a) ** 2)
        if denom == 0:
            return 0.0
        return (self.rx * self.ry) / denom

    def to_dict(self) -> dict:
        return {
            "shape_type": self.shape_type,
            "center": self.center.to_dict(),
            "rx": self.rx,
            "ry": self.ry,
            "rotation": self.rotation,
        }

    def get_area(self) -> float:
        return math.pi * self.rx * self.ry

    def get_bounds(self, n_points: int = None) -> tuple[float, float, float, float]:
        cx, cy = self.center.to_cartesian()

        dx = math.sqrt((self.rx * math.cos(self.rotation))**2 + (self.ry * math.sin(self.rotation))**2)
        dy = math.sqrt((self.rx * math.sin(self.rotation))**2 + (self.ry * math.cos(self.rotation))**2)

        return cx - dx, cx + dx, cy - dy, cy + dy


class Harmonic(Shape):
    """
    Harmonic shape - circle with boundary variations based on Fourier series.
    Useful for organic shapes like natural wood anomalies.
    Each harmonic tuple represents (a_n, b_n) coefficients for Cosine and Sine.
    """

    SHAPE_TYPE = "harmonic"

    def __init__(self, center: Pos, base_radius: float, harmonics: list[tuple[float, float]]):
        super().__init__(center)
        if base_radius <= 0:
            raise ValueError(f"Radius should be greater than zero: {base_radius}")
        self.base_radius = base_radius
        self.harmonics = harmonics

    @classmethod
    def from_dict(cls, data):
        center_data, base_radius, harmonics = data.get("center"), data.get("base_radius"), data.get("harmonics")
        if center_data is None or base_radius is None or harmonics is None:
            raise ValueError("Missing required keys for Harmonic: 'cx', 'cy', 'base_radius', 'harmonics'")
        center = Pos.from_dict(center_data)
        return Harmonic(center, base_radius, harmonics)

    def get_radius(self, theta: float) -> float:
        """Calculate radius at given angle with harmonic perturbations."""
        perturbation = sum(
            a * math.cos(n * theta) + b * math.sin(n * theta)
            for n, (a, b) in enumerate(self.harmonics, start=1)
        )
        return self.base_radius + perturbation

    def to_dict(self) -> dict:
        return {
            "shape_type": self.shape_type,
            "center": self.center.to_dict(),
            "base_radius": self.base_radius,
            "harmonics": self.harmonics,
        }

    def is_valid(self, n_points: int = 360) -> bool:
        for i in range(n_points):
            theta = 2 * math.pi * i / n_points
            if self.get_radius(theta) <= 0.0:
                return False
        return True

    def get_area(self) -> float:
        base_area = math.pi * (self.base_radius ** 2)

        harmonic_area = (math.pi / 2.0) * sum(
            a**2 + b**2 for a, b in self.harmonics
        )
        return base_area + harmonic_area


class ShapeFactory:
    ACCEPTED_SHAPES: tuple[Shape] = (Circle, Ellipse, Harmonic)
    SHAPES: dict[str, type[Shape]] = {
        s.SHAPE_TYPE: s 
        for s in ACCEPTED_SHAPES
    }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Shape":
        shape_type_str = data.get("shape_type")
        if shape_type_str is None:
            raise ValueError("")

        if shape_type_str in cls.SHAPES:
            shape_type = cls.SHAPES.get(shape_type_str)
            return shape_type.from_dict(data)

        raise NotImplementedError("")
