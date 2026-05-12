from .shape import Shape


class Anomaly:
    def __init__(self, shape: Shape, conductivity: float):
        self.shape = shape
        self.conductivity = conductivity
