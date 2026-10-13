import math

from scripts.data.config import COND_ANOMALY_MAX, COND_ANOMALY_MIN, COND_BASE_MAX, COND_BASE_MIN
from scripts.data.generate_dataset import get_dataset
from src.data.models import Anomaly, Pos
from src.data.models.shape import Harmonic


def test_harmonic_bounds():
    center = Pos.from_cartesian(0.0, 0.0)
    # create a perfect circle using harmonic base class to avoid math complex roots in bounds check
    harmonics = [(0.0, 0.0)]
    shape = Harmonic(center=center, base_radius=1.0, harmonics=harmonics)

    xmin, xmax, ymin, ymax = shape.get_bounds(n_points=360)

    assert math.isclose(xmax, 1.0, abs_tol=0.01)
    assert math.isclose(xmin, -1.0, abs_tol=0.01)
    assert math.isclose(ymax, 1.0, abs_tol=0.01)
    assert math.isclose(ymin, -1.0, abs_tol=0.01)


def test_containment_logic():
    center = Pos.from_cartesian(0.0, 0.0)
    shape = Harmonic(center=center, base_radius=1.0, harmonics=[(0.0, 0.0)])  # Perfect circle

    anomaly = Anomaly(shape=shape, conductivity=10.0, propagation_fn="Linear")

    # Inside point
    assert anomaly.contains(Pos.from_cartesian(0.5, 0.5))
    # Outside point
    assert not anomaly.contains(Pos.from_cartesian(1.5, 1.5))


def test_stochastic_ranges():
    # Use the actual dataset generator from the project to verify its rules
    trunks = get_dataset(50)

    for trunk in trunks:
        # Check base conductivity limits
        assert COND_BASE_MIN <= trunk.base.conductivity <= COND_BASE_MAX

        # Check anomalies limits
        for anomaly in trunk.anomalies:
            assert COND_ANOMALY_MIN <= anomaly.conductivity <= COND_ANOMALY_MAX
