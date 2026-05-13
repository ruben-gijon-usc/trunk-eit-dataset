import math
from abc import ABC, abstractmethod

from .pos import Pos


class Shape(ABC):
    """Abstract base class for geometric shapes."""

    SHAPE_TYPE: str = ""

    @abstractmethod
    def contains(self, pos: Pos) -> bool:
        """Check if a position is inside the shape."""
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON."""
        pass

    @abstractmethod
    def get_points(self, n_points: int) -> list[tuple[float, float]]:
        """Generate boundary points for the shape."""
        pass

    @property
    def shape_type(self) -> str:
        """Return shape type identifier."""
        return self.SHAPE_TYPE


class CircularShape(Shape):
    """Abstract base class for polar/circular-based shapes."""

    def __init__(self, center: Pos):
        self.cx, self.cy = center.to_cartesian()

    @abstractmethod
    def get_radius(self, theta: float) -> float:
        """Get radius on a certain direction (theta in radians)."""
        pass

    def contains(self, pos: Pos) -> bool:
        """Check if a position is inside the shape using polar coordinates."""
        x, y = pos.to_cartesian()
        dx = x - self.cx
        dy = y - self.cy
        r = math.sqrt(dx**2 + dy**2)
        theta = math.atan2(dy, dx)
        return r <= self.get_radius(theta)

    def get_points(self, n_points: int) -> list[tuple[float, float]]:
        """Generate boundary points based on the radius at different angles."""
        points = []
        for i in range(n_points):
            # Calculate angle evenly distributed across 2*PI
            theta = 2 * math.pi * i / n_points
            r = self.get_radius(theta)

            # Convert polar to cartesian offsets
            d_x = r * math.cos(theta)
            d_y = r * math.sin(theta)

            points.append((self.cx + d_x, self.cy + d_y))
        return points


class Circle(CircularShape):
    """Circle shape."""

    SHAPE_TYPE = "circle"

    def __init__(self, center: Pos, radius: float):
        super().__init__(center)
        self.radius = radius

    def get_radius(self, theta: float) -> float:
        return self.radius

    def to_dict(self) -> dict:
        return {"shape_type": self.shape_type, "cx": self.cx, "cy": self.cy, "radius": self.radius}


class Ellipse(CircularShape):
    """Ellipse shape with rotation."""

    SHAPE_TYPE = "ellipse"

    def __init__(self, center: Pos, rx: float, ry: float, rotation: float = 0.0):
        super().__init__(center)
        self.rx = rx
        self.ry = ry
        self.rotation = rotation

    def get_radius(self, theta: float) -> float:
        # Calculate radius considering the ellipse's rotation
        alpha = theta - self.rotation
        cos_a = math.cos(alpha)
        sin_a = math.sin(alpha)

        # Prevent division by zero mathematically
        denom = math.sqrt((self.ry * cos_a) ** 2 + (self.rx * sin_a) ** 2)
        if denom == 0:
            return 0.0
        return (self.rx * self.ry) / denom

    def to_dict(self) -> dict:
        return {
            "shape_type": self.shape_type,
            "cx": self.cx,
            "cy": self.cy,
            "rx": self.rx,
            "ry": self.ry,
            "rotation": self.rotation,
        }


class Harmonic(CircularShape):
    """
    Harmonic shape - circle with boundary variations based on Fourier series.
    Useful for organic shapes like natural wood anomalies.
    Each harmonic tuple represents (a_n, b_n) coefficients for Cosine and Sine.
    """

    SHAPE_TYPE = "harmonic"

    def __init__(self, center: Pos, base_radius: float, harmonics: list[tuple[float, float]]):
        super().__init__(center)
        self.base_radius = base_radius
        self.harmonics = harmonics

    def get_radius(self, theta: float) -> float:
        """Calculate radius at given angle with harmonic perturbations."""
        r = self.base_radius
        for n, (a, b) in enumerate(self.harmonics, start=1):
            r += a * math.cos(n * theta) + b * math.sin(n * theta)
        return r

    def to_dict(self) -> dict:
        return {
            "shape_type": self.shape_type,
            "cx": self.cx,
            "cy": self.cy,
            "base_radius": self.base_radius,
            "harmonics": self.harmonics,
        }

    def is_valid(self) -> bool:
        pass


class Rectangle(Shape):
    """Axis-aligned rectangle."""

    SHAPE_TYPE = "rectangle"

    def __init__(self, x_min: float, y_min: float, x_max: float, y_max: float):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        # Derive center variables for the base class get_points() method
        self.cx = (x_min + x_max) / 2.0
        self.cy = (y_min + y_max) / 2.0

    def contains(self, pos: Pos) -> bool:
        x, y = pos.to_cartesian()
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    def get_points(self, n_points: int) -> list[tuple[float, float]]:
        """Generate boundary points for the rectangle."""
        points = []
        for i in range(n_points):
            theta = 2 * math.pi * i / n_points
            r = self.get_radius(theta)

            d_x = r * math.cos(theta)
            d_y = r * math.sin(theta)

            points.append((self.cx + d_x, self.cy + d_y))
        return points

    def to_dict(self) -> dict:
        return {
            "shape_type": self.shape_type,
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
        }
