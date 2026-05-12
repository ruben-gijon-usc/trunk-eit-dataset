"""
Tests for the pipeline and grid modules.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models import Trunk, Anomaly, Circle, Pos
from src.simulation import generate_grid, grid2png
from src.pipeline import generate_random_trunk, PipelineConfig, generate_dataset, DatasetSample


class TestGrid:
    """Test grid generation."""

    def test_generate_grid_homogeneous(self):
        """Test generating grid for homogeneous trunk."""
        trunk = Trunk(radius=1.0, base_conductivity=0.1, anomalies=[])
        grid = generate_grid(trunk, resolution=64)

        assert grid.shape == (64, 64)
        assert np.all(grid[grid > 0] == 0.1)
        assert np.sum(grid == 0) > 0  # outside trunk is 0

    def test_generate_grid_with_anomaly(self):
        """Test generating grid with anomaly."""
        trunk = Trunk(
            radius=1.0,
            base_conductivity=0.1,
            anomalies=[
                Anomaly(shape=Circle(center=Pos(r=0.3, phi=0.0), radius=0.2), conductivity=0.5)
            ],
        )
        grid = generate_grid(trunk, resolution=64)

        unique_vals = np.unique(grid[grid > 0])
        assert len(unique_vals) == 2

    def test_grid2png(self):
        """Test saving grid as PNG."""
        trunk = Trunk(radius=1.0, base_conductivity=0.1, anomalies=[])
        grid = generate_grid(trunk, resolution=32)

        with open("/tmp/test_grid.png", "wb") as f:
            pass
        grid2png(grid, "/tmp/test_grid.png")

        assert os.path.exists("/tmp/test_grid.png")
        os.remove("/tmp/test_grid.png")


class TestPipeline:
    """Test pipeline functions."""

    def test_generate_random_trunk(self):
        """Test generating random trunk."""
        config = PipelineConfig(
            trunk_radius=1.0,
            base_conductivity=0.1,
            num_anomalies_range=(1, 3),
            anomaly_radius_range=(0.1, 0.3),
            anomaly_conductivity_range=(0.01, 0.5),
        )

        trunk = generate_random_trunk(config)

        assert trunk.radius == 1.0
        assert trunk.base_conductivity == 0.1
        assert 1 <= len(trunk.anomalies) <= 3

    def test_pipeline_config_defaults(self):
        """Test default pipeline config."""
        config = PipelineConfig()

        assert config.num_samples == 1000
        assert config.grid_resolution == 128
        assert config.trunk_radius == 1.0
        assert config.base_conductivity == 0.1
        assert config.use_eidors is True
        assert config.n_electrodes == 16


class TestBaseModels:
    """Test base domain models."""

    def test_pos_creation(self):
        """Test Pos creation."""
        pos = Pos(r=0.5, phi=np.pi / 4)
        assert pos.r == 0.5
        assert pos.phi == np.pi / 4

    def test_anomaly_creation(self):
        """Test Anomaly creation."""
        anomaly = Anomaly(shape=Circle(center=Pos(r=0.3, phi=0.0), radius=0.2), conductivity=0.5)
        assert anomaly.shape.radius == 0.2
        assert anomaly.conductivity == 0.5

    def test_trunk_creation(self):
        """Test Trunk creation."""
        trunk = Trunk(
            radius=1.0,
            base_conductivity=0.1,
            anomalies=[
                Anomaly(shape=Circle(center=Pos(r=0.3, phi=0.0), radius=0.2), conductivity=0.5)
            ],
        )
        assert trunk.radius == 1.0
        assert len(trunk.anomalies) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
